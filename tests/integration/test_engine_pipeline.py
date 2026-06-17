"""Integration test for the standalone TTS engine.

Tests the full asyncio ↔ engine thread pipeline in stub mode (no GPU):
  1. Spliter: text → segment actions
  2. Dispatcher: session management + text routing
  3. EngineLoop: drain inbox + prefill + decode pipeline
  4. Cross-thread result delivery

Run:
    pytest tests/test_engine_integration.py -v -s
    # or just the fast CPU-only tests:
    pytest tests/test_engine_integration.py -v -s -k "not gpu"
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT, TOKENIZER_DIR, WEIGHTS_DIR, VARIANT

sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

HAS_TOKENIZER = TOKENIZER_DIR.exists()
SKIP_NO_TOKENIZER = pytest.mark.skipif(
    not HAS_TOKENIZER,
    reason=f"Tokenizer not found (variant={VARIANT}, path={TOKENIZER_DIR})",
)


# ---------------------------------------------------------------------------
# 1. Spliter unit tests (pure CPU, no async, no GPU)
# ---------------------------------------------------------------------------

class TestSpliter:
    """Test the Spliter text segmentation logic."""

    def test_classify_punct_level(self):
        from engine.frontend.spliter.spliter import Spliter

        assert Spliter.classify_punct_level("你好。") == 1
        assert Spliter.classify_punct_level("你好，") == 2
        assert Spliter.classify_punct_level("你好") == 0
        assert Spliter.classify_punct_level("hello!") == 1

    def test_classify_punct_level_with_trailing_closers(self):
        from engine.frontend.spliter.spliter import Spliter

        assert Spliter.classify_punct_level("你好！”") == 1
        assert Spliter.classify_punct_level("你好。）") == 1
        assert Spliter.classify_punct_level('hello!")') == 1
        assert Spliter.classify_punct_level("你好，”") == 2

    def test_classify_punct_level_does_not_cross_opening_quote(self):
        from engine.frontend.spliter.spliter import Spliter

        assert Spliter.classify_punct_level("。“") == 0
        assert Spliter.classify_punct_level("：“") == 0

    def test_classify_punct_level_l3_suffixes(self):
        from engine.frontend.spliter.spliter import Spliter

        assert Spliter.classify_punct_level("你好……") == 3
        assert Spliter.classify_punct_level("你好……”") == 3
        assert Spliter.classify_punct_level("\n") == 3

    def test_compute_thresholds(self):
        from engine.frontend.spliter.driver import compute_thresholds

        th = compute_thresholds(remaining_kv=500, ema_ratio=5.0)
        assert (
            th.min_tokens_l1
            < th.min_tokens_l2
            < th.min_tokens_l3
            < th.force_split_at
        )
        assert th.min_tokens_l1 >= 1
        assert th.force_split_at <= 500

    def test_pre_split_short_text(self):
        """Short text → single segment, no split."""
        from engine.frontend.spliter.spliter import Spliter

        spliter = Spliter(engine_max_decode_len=512, ema_ratio=5.0)
        tokens = [(i, f"tok{i}") for i in range(10)]
        segments = spliter.pre_split(tokens)
        assert len(segments) == 1
        assert len(segments[0]) == 10

    def test_pre_split_at_l1_punct(self):
        """Text with L1 punct should split at the punct boundary."""
        from engine.frontend.spliter.spliter import Spliter

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
        th = spliter._make_thresholds()

        tokens = []
        for i in range(th.min_tokens_l1 + 5):
            tokens.append((i, f"tok{i}"))
        tokens.append((999, "句号。"))
        tokens.append((1000, "后续"))
        tokens.append((1001, "文本"))

        segments = spliter.pre_split(tokens)
        assert len(segments) >= 2, f"Expected >=2 segments, got {len(segments)}"

    def test_pre_split_does_not_snap_to_l2_in_offline_mode(self):
        """Offline pre-split should avoid comma-level snap cuts."""
        from engine.frontend.spliter.spliter import Spliter

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
        th = spliter._make_thresholds()

        tokens = [(i, f"tok{i}") for i in range(th.force_split_at - 2)]
        tokens.append((900, "逗号，"))
        tokens.append((901, "后续甲"))
        tokens.append((902, "后续乙"))
        tokens.append((903, "后续丙"))

        segments = spliter.pre_split(tokens)
        assert len(segments) >= 2
        assert segments[0][-1].punct_level != 2, "offline pre-split should not end on L2 punctuation"

    def test_streaming_feed_tokens(self):
        """Streaming mode: feed tokens one by one, expect segment actions."""
        from engine.frontend.spliter.spliter import Spliter
        from engine.frontend.spliter.driver import ActionType

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
        th = spliter._make_thresholds()

        tokens = [(i, f"tok{i}") for i in range(th.min_tokens_l1 + 2)]
        tokens.append((999, "。"))

        actions = spliter.feed_tokens(tokens)

        action_types = [a.action.type for a in actions]
        assert ActionType.PREFILL in action_types, "Should have PREFILL action"
        assert ActionType.DECODE in action_types, "Should have DECODE actions"

    def test_offline_set_full_text(self):
        """Offline mode: set full text, get all segment actions."""
        from engine.frontend.spliter.spliter import Spliter
        from engine.frontend.spliter.driver import ActionType

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)
        th = spliter._make_thresholds()

        tokens = []
        for i in range(th.min_tokens_l1 + 2):
            tokens.append((i, f"tok{i}"))
        tokens.append((100, "。"))
        for i in range(5):
            tokens.append((200 + i, f"after{i}"))

        actions = spliter.set_full_text(tokens)
        action_types = [a.action.type for a in actions]
        assert ActionType.PREFILL in action_types

    def test_push_group_tokens_assigns_monotonic_group_ids(self):
        """Long-segment mode should preserve a stable outer group order."""
        from engine.frontend.spliter.spliter import Spliter
        from engine.frontend.spliter.driver import ActionType

        spliter = Spliter(engine_max_decode_len=100, ema_ratio=2.0)

        a1 = spliter.push_group_tokens([(1, "你好"), (2, "。")])
        a2 = spliter.push_group_tokens([(3, "世界"), (4, "。")])

        groups1 = {a.group_idx for a in a1 if a.action.type in (ActionType.PREFILL, ActionType.DECODE)}
        groups2 = {a.group_idx for a in a2 if a.action.type in (ActionType.PREFILL, ActionType.DECODE)}

        assert groups1 == {0}
        assert groups2 == {1}


# ---------------------------------------------------------------------------
# 2. AudioReorder tests
# ---------------------------------------------------------------------------

class TestAudioReorder:
    def test_in_order(self):
        from engine.frontend.spliter.reorder import AudioReorder

        r = AudioReorder()
        out = r.push(0, 0, b"a0")
        assert out == [b"a0"]
        out = r.push(0, 0, b"a1")
        assert out == [b"a1"]
        out = r.mark_done(0, 0, group_final=True)
        assert out == []
        assert r.next_emit_segment == (1, 0)

    def test_out_of_order(self):
        from engine.frontend.spliter.reorder import AudioReorder

        r = AudioReorder()
        out = r.push(1, 0, b"b0")
        assert out == []
        out = r.push(0, 0, b"a0")
        assert out == [b"a0"]
        out = r.mark_done(0, 0, group_final=True)
        assert b"b0" in out
        assert r.next_emit_segment == (1, 0)


# ---------------------------------------------------------------------------
# 3. Tokenizer test
# ---------------------------------------------------------------------------

class TestTokenizer:
    @SKIP_NO_TOKENIZER
    def test_encode_decode(self):
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tok = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        ids = tok.encode_ids("你好世界")
        assert len(ids) > 0

        ids2, texts = tok.encode_with_text("你好，世界。")
        assert len(ids2) == len(texts)
        assert any("。" in t for t in texts), f"Expected punct in texts: {texts}"

    @SKIP_NO_TOKENIZER
    def test_spliter_with_real_tokenizer(self):
        """End-to-end: real tokenizer → Spliter → actions."""
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer
        from engine.frontend.spliter.spliter import Spliter
        from engine.frontend.spliter.driver import ActionType

        tok = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        text = "你好世界。今天天气真好，我们一起出去玩吧！"
        ids, texts = tok.encode_with_text(text, add_special_tokens=False)
        tokens = list(zip(ids, texts))

        spliter = Spliter(engine_max_decode_len=200, ema_ratio=5.0)
        actions = spliter.feed_tokens(tokens)

        prefills = [a for a in actions if a.action.type == ActionType.PREFILL]
        decodes = [a for a in actions if a.action.type == ActionType.DECODE]
        assert len(prefills) >= 1, "Should have at least one PREFILL"
        assert len(decodes) >= 1, "Should have DECODE actions"
        print(f"\nTokens: {len(tokens)}, Prefills: {len(prefills)}, "
              f"Decodes: {len(decodes)}, Segments: {spliter.current_segment_idx}")


# ---------------------------------------------------------------------------
# 4. Prefill boundary tests
# ---------------------------------------------------------------------------

class TestPrefillBuilderBoundary:
    @SKIP_NO_TOKENIZER
    @pytest.mark.parametrize(
        ("task_type_name", "kwargs"),
        [
            ("custom_voice", {"speaker": "vivian", "instruct": "用温柔的语气说"}),
            ("voice_design", {"instruct": "请保持平静、克制的旁白语气"}),
        ],
    )
    def test_build_plan_from_ids_matches_text_path(self, task_type_name, kwargs):
        torch = pytest.importorskip("torch")
        from engine.backend.prefill import PrefillBuilder, TaskType
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        class _FakeWeights:
            def __init__(self):
                self.device = torch.device("cpu")
                self.hidden_size = 4
                self.variant = "fake"
                self.codec_bos_id = 10
                self.codec_pad_id = 11
                self.codec_nothink_id = 12
                self.codec_think_bos_id = 13
                self.codec_think_eos_id = 14
                self.codec_think_id = 15
                self.codec_language_id = {}
                self.spk_id_map = {"vivian": 16}
                self.spk_is_dialect = {}
                self.default_speaker = "vivian"
                self.fallback_speaker = "vivian"
                self.codec_embeddings_3d = None
                self.tts_pad_embed = torch.full((1, 1, self.hidden_size), 1, dtype=torch.bfloat16)
                self.tts_bos_embed = torch.full((1, 1, self.hidden_size), 2, dtype=torch.bfloat16)
                self.tts_eos_embed = torch.full((1, 1, self.hidden_size), 3, dtype=torch.bfloat16)

            def _embed(self, token_ids: torch.Tensor, scale: float) -> torch.Tensor:
                base = token_ids.to(dtype=torch.float32).unsqueeze(-1)
                cols = [base * scale + float(i) for i in range(self.hidden_size)]
                return torch.cat(cols, dim=-1).to(dtype=torch.bfloat16)

            def text_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
                return self._embed(token_ids, 0.01)

            def codec_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
                return self._embed(token_ids, 0.02)

        tokenizer = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        builder = PrefillBuilder(_FakeWeights(), tokenizer)
        task_type = getattr(TaskType, task_type_name.upper())
        text = "一个月后，国王的城堡里挤满了来自世界各地的王子，"
        token_ids = tokenizer.encode_ids(text, add_special_tokens=False)

        plan_from_text = builder.build_plan(
            task_type=task_type,
            text=text,
            include_eos=True,
            **kwargs,
        )
        prompt_kwargs = {}
        if kwargs.get("instruct"):
            prompt_kwargs["instruct"] = None
            prompt_kwargs["instruct_token_ids"] = tokenizer.encode_ids(
                kwargs["instruct"], add_special_tokens=False,
            )
        kwargs_from_ids = dict(kwargs)
        kwargs_from_ids.update(prompt_kwargs)

        plan_from_ids = builder.build_plan_from_ids(
            task_type=task_type,
            token_ids=token_ids,
            include_eos=True,
            **kwargs_from_ids,
        )

        assert plan_from_text.prefix_cache_key == plan_from_ids.prefix_cache_key
        assert torch.equal(plan_from_text.prefill_embeds, plan_from_ids.prefill_embeds)
        assert len(plan_from_text.trailing) == len(plan_from_ids.trailing)
        assert plan_from_text.warnings == plan_from_ids.warnings

        for lhs, rhs in zip(plan_from_text.trailing, plan_from_ids.trailing):
            assert torch.equal(lhs, rhs)

        if plan_from_text.cacheable_prefix_embeds is not None:
            assert torch.equal(
                plan_from_text.cacheable_prefix_embeds,
                plan_from_ids.cacheable_prefix_embeds,
            )
        if plan_from_text.request_prefill_embeds is not None:
            assert torch.equal(
                plan_from_text.request_prefill_embeds,
                plan_from_ids.request_prefill_embeds,
            )


# ---------------------------------------------------------------------------
# 5. Full pipeline integration test (asyncio + engine thread, stub mode)
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    """Test the full engine pipeline in stub (no-GPU) mode."""

    @SKIP_NO_TOKENIZER
    @pytest.mark.asyncio
    async def test_single_session_stub(self):
        """One session, streaming text, verify audio callback chain."""
        from engine.core.types import (
            EngineRequest, EngineResult, RequestType, ResultType,
        )
        from engine.frontend.interface import FrontendInterface
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tokenizer = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        async_inbox = asyncio.Queue(maxsize=256)

        interface = FrontendInterface(
            engine_inbox=async_inbox,
            tokenizer=tokenizer,
            max_sessions=8,
            engine_max_decode_len=200,
        )

        received_audio: list[bytes] = []
        done_event = asyncio.Event()

        async def on_audio(sid: str, data: bytes):
            received_audio.append(data)

        async def on_done(sid: str, metrics: dict):
            done_event.set()

        session = await interface.create_session(
            "test-001",
            speaker_key="vivian",
            task_type="custom_voice",
            on_audio=on_audio,
            on_done=on_done,
        )
        assert interface.active_count == 1

        await interface.push_text_input("test-001", "你好世界。")
        await interface.mark_input_complete("test-001")

        requests_sent = []
        while not async_inbox.empty():
            req = await async_inbox.get()
            requests_sent.append(req)

        req_types = [r.type for r in requests_sent]
        assert RequestType.NEW_SESSION in req_types
        has_segment = (RequestType.START_TOKENS in req_types
                       or RequestType.APPEND_TOKENS in req_types)
        assert has_segment, f"Expected segment requests, got: {req_types}"

        print(f"\nRequests sent to engine: {len(requests_sent)}")
        for r in requests_sent:
            print(f"  {r.type.name} seg={r.segment_idx} tokens={r.token_ids}")

        await session.result_queue.put(EngineResult(
            type=ResultType.SESSION_DONE,
            session_id="test-001",
        ))

        await asyncio.wait_for(done_event.wait(), timeout=2.0)
        assert interface.active_count == 0
        print("Session completed successfully")

    @pytest.mark.asyncio
    async def test_full_text_normalizes_newlines_before_tokenization(self):
        from engine.frontend.interface import FrontendInterface
        from engine.core.types import EngineResult, InputMode, ResultType, SessionConfig

        class _FakeTokenizer:
            def __init__(self):
                self.last_text = None

            def encode_with_text(self, text, add_special_tokens=False):
                self.last_text = text
                return [1], [text]

        tokenizer = _FakeTokenizer()
        async_inbox = asyncio.Queue(maxsize=16)
        interface = FrontendInterface(
            engine_inbox=async_inbox,
            tokenizer=tokenizer,
            max_sessions=4,
            engine_max_decode_len=200,
        )

        session = await interface.create_session(
            "norm-001",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.FULL_TEXT,
            ),
        )

        await interface.push_text_input("norm-001", "第一行，\n第二行。\t第三行。")
        await interface.mark_input_complete("norm-001")

        assert tokenizer.last_text == "第一行，第二行。 第三行。"
        assert session.engine_tokens_done_sent is True
        await session.result_queue.put(EngineResult(
            type=ResultType.SESSION_DONE,
            session_id="norm-001",
        ))
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_create_session_prepares_prompt_token_specs(self):
        from engine.core.types import EngineResult, ResultType, SessionConfig
        from engine.frontend.interface import FrontendInterface

        class _FakeTokenizer:
            def encode_ids(self, text, add_special_tokens=False):
                return [len(text), len(text) + 1]

            def encode_with_text(self, text, add_special_tokens=False):
                return [1], [text]

        config = SessionConfig(
            task_type="voice_clone",
            ref_text="参考文段，\n第二行。",
            instruct="请温柔地说。\t",
        )
        interface = FrontendInterface(
            engine_inbox=asyncio.Queue(maxsize=16),
            tokenizer=_FakeTokenizer(),
            max_sessions=4,
            engine_max_decode_len=200,
        )

        session = await interface.create_session("prompt-001", config=config)

        assert session.config.instruct == "请温柔地说。"
        assert session.config.ref_text == "参考文段，第二行。"
        assert session.config.instruct_spec is not None
        assert session.config.ref_text_spec is not None
        assert session.config.instruct_spec.text == "请温柔地说。"
        assert session.config.ref_text_spec.text == "参考文段，第二行。"
        assert session.config.instruct_spec.token_ids == [6, 7]
        assert session.config.ref_text_spec.token_ids == [9, 10]
        await session.result_queue.put(EngineResult(
            type=ResultType.SESSION_DONE,
            session_id="prompt-001",
        ))
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_streaming_text_events_support_token_player(self):
        from engine.core.types import EngineResult, InputMode, ResultType, SessionConfig
        from engine.frontend.interface import FrontendInterface

        class _FakeTokenizer:
            def encode_ids(self, text, add_special_tokens=False):
                return [ord(ch) for ch in text]

            def encode_with_text(self, text, add_special_tokens=False):
                ids = [ord(ch) for ch in text]
                return ids, list(text)

        async_inbox = asyncio.Queue(maxsize=32)
        interface = FrontendInterface(
            engine_inbox=async_inbox,
            tokenizer=_FakeTokenizer(),
            max_sessions=4,
            engine_max_decode_len=200,
        )

        events = []

        async def on_event(sid: str, event: dict):
            events.append(event)

        session = await interface.create_session(
            "token-player-001",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.CLAUSE,
            ),
            on_event=on_event,
        )

        await interface.push_text_input("token-player-001", "你好。")
        await interface.mark_input_complete("token-player-001")
        await asyncio.sleep(0)

        event_types = [event["type"] for event in events]
        token_text = "".join(event["text"] for event in events if event["type"] == "text_token")

        assert token_text == "你好。"
        assert "text_boundary_commit" in event_types
        boundary = next(event for event in events if event["type"] == "text_boundary_commit")
        assert boundary["text"] == "你好。"
        assert boundary["meta"]["text_complete"] == "true"

        await session.result_queue.put(EngineResult(
            type=ResultType.SESSION_DONE,
            session_id="token-player-001",
        ))
        await asyncio.sleep(0)

    @SKIP_NO_TOKENIZER
    @pytest.mark.asyncio
    async def test_multi_session_stub(self):
        """Multiple concurrent sessions."""
        from engine.core.types import (
            EngineResult, RequestType, ResultType,
        )
        from engine.frontend.interface import FrontendInterface
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tokenizer = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        async_inbox = asyncio.Queue(maxsize=1024)

        interface = FrontendInterface(
            engine_inbox=async_inbox,
            tokenizer=tokenizer,
            max_sessions=64,
            engine_max_decode_len=200,
        )

        n_sessions = 8
        done_events = {}
        sessions = {}

        for i in range(n_sessions):
            sid = f"multi-{i:03d}"
            done_events[sid] = asyncio.Event()

            async def on_done(sid_inner: str, metrics: dict, _e=done_events[sid]):
                _e.set()

            s = await interface.create_session(
                sid, task_type="custom_voice", on_done=on_done,
            )
            sessions[sid] = s

        assert interface.active_count == n_sessions

        texts = [
            "你好。", "世界！", "今天天气好。", "明天见。",
            "测试。", "一二三四五。", "很高兴见到你！", "再见！",
        ]
        for i, sid in enumerate(sessions):
            await interface.push_text_input(sid, texts[i])
            await interface.mark_input_complete(sid)

        req_count = 0
        while not async_inbox.empty():
            await async_inbox.get()
            req_count += 1

        print(f"\n{n_sessions} sessions generated {req_count} engine requests")
        assert req_count >= n_sessions

        for sid, session in sessions.items():
            await session.result_queue.put(EngineResult(
                type=ResultType.SESSION_DONE,
                session_id=sid,
            ))

        for sid, evt in done_events.items():
            await asyncio.wait_for(evt.wait(), timeout=2.0)

        assert interface.active_count == 0
        print(f"All {n_sessions} sessions completed")

    @SKIP_NO_TOKENIZER
    @pytest.mark.asyncio
    async def test_cancel_session_releases_frontend_slot_immediately(self):
        from engine.frontend.interface import FrontendInterface
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tokenizer = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        async_inbox = asyncio.Queue(maxsize=64)
        interface = FrontendInterface(
            engine_inbox=async_inbox,
            tokenizer=tokenizer,
            max_sessions=2,
            engine_max_decode_len=200,
        )

        await interface.create_session(
            "cancel-me",
            task_type="custom_voice",
        )
        assert interface.active_count == 1

        await interface.cancel_session("cancel-me")
        await asyncio.sleep(0)

        assert interface.active_count == 0

    @SKIP_NO_TOKENIZER
    @pytest.mark.asyncio
    async def test_long_segment_streaming_queues_followup_groups_until_segment_done(self):
        from engine.core.types import (
            EngineResult, GroupPolicy, InputMode, ResultType, SessionConfig,
        )
        from engine.frontend.interface import FrontendInterface
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer

        tokenizer = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        async_inbox = asyncio.Queue(maxsize=256)
        interface = FrontendInterface(
            engine_inbox=async_inbox,
            tokenizer=tokenizer,
            max_sessions=8,
            engine_max_decode_len=200,
        )

        session = await interface.create_session(
            "long-seg-001",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.LONG_SEGMENT,
                group_policy=GroupPolicy.AUTO,
            ),
        )

        for chunk in [
            "你好，这是流式文本输入测试。",
            "我们正在验证",
            "文本追加功能",
            "是否工作正常。",
        ]:
            await interface.push_text_input("long-seg-001", chunk)
        await interface.mark_input_complete("long-seg-001")

        initial_types = []
        while not async_inbox.empty():
            req = await async_inbox.get()
            initial_types.append((req.type.name, req.segment_idx))

        assert ("START_TOKENS", 0) in initial_types
        assert ("START_TOKENS", 1) in initial_types
        assert ("START_TOKENS", 2) not in initial_types
        assert ("START_TOKENS", 3) not in initial_types

        await session.result_queue.put(EngineResult(
            type=ResultType.SEGMENT_END,
            session_id="long-seg-001",
            segment_idx=0,
            metrics={"audio_steps": 10, "text_tokens": 9},
        ))
        await asyncio.sleep(0)

        after_seg0 = []
        while not async_inbox.empty():
            req = await async_inbox.get()
            after_seg0.append((req.type.name, req.segment_idx))
        assert ("START_TOKENS", 2) in after_seg0

        await session.result_queue.put(EngineResult(
            type=ResultType.SEGMENT_END,
            session_id="long-seg-001",
            segment_idx=1,
            metrics={"audio_steps": 10, "text_tokens": 3},
        ))
        await asyncio.sleep(0)

        after_seg1 = []
        while not async_inbox.empty():
            req = await async_inbox.get()
            after_seg1.append((req.type.name, req.segment_idx))
        assert ("START_TOKENS", 3) in after_seg1

        await session.result_queue.put(EngineResult(
            type=ResultType.SESSION_DONE,
            session_id="long-seg-001",
        ))
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# 6. Performance benchmark (optional, needs real tokenizer)
# ---------------------------------------------------------------------------

class TestPerformance:
    @SKIP_NO_TOKENIZER
    def test_spliter_throughput(self):
        """Benchmark: how fast can the Spliter process tokens?"""
        import time
        from engine.frontend.spliter.tokenizer import LightQwen3TTSTokenizer
        from engine.frontend.spliter.spliter import Spliter

        tok = LightQwen3TTSTokenizer(str(TOKENIZER_DIR))
        long_text = "这是一段测试文本。" * 200
        ids, texts = tok.encode_with_text(long_text, add_special_tokens=False)
        tokens = list(zip(ids, texts))

        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            s = Spliter(engine_max_decode_len=512, ema_ratio=5.0)
            s.set_full_text(tokens)
        elapsed = time.perf_counter() - start

        tokens_per_sec = (len(tokens) * n_iters) / elapsed
        print(f"\nSpliter throughput: {tokens_per_sec:.0f} tokens/sec "
              f"({len(tokens)} tokens × {n_iters} iters in {elapsed:.3f}s)")
        assert tokens_per_sec > 10_000, "Spliter should handle >10K tokens/sec"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
