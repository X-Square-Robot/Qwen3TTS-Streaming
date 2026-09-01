from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from engine.core.types import (
    EngineResult,
    GroupPolicy,
    InputMode,
    RequestType,
    ResultType,
    SessionConfig,
)
from engine.frontend.interface import (
    FrontendInterface,
    _metric_bool,
    _normalize_tts_text,
)
from engine.frontend.diagnostic_text import (
    DEFAULT_ENGINE_MODEL_VERSION,
    DEFAULT_ENGINE_VERSION,
    DEFAULT_MODEL_VERSION,
    VERSION_QUERY_TEXT,
    format_engine_model_version,
    resolve_diagnostic_text,
)
from engine.core.text_journal import CanonicalTextJournal


@pytest.mark.parametrize("value", [True, 1, "1", "true", "YES", "on"])
def test_metric_bool_true_values(value):
    assert _metric_bool(value) is True


@pytest.mark.parametrize("value", [False, 0, None, "0", "false", "NO", "off", ""])
def test_metric_bool_false_values(value):
    assert _metric_bool(value) is False


def test_tts_text_normalization_strips_emoji_noise():
    assert _normalize_tts_text("你好😊，世界🌍！") == "你好，世界！"
    assert _normalize_tts_text("good😊morning") == "good morning"
    assert _normalize_tts_text("第1️⃣步完成✅。") == "第步完成。"
    assert _normalize_tts_text("😊🚀") == ""
    assert _normalize_tts_text("湿度19%，气温为23℃。") == "湿度19%，气温为23℃。"
    assert _normalize_tts_text("湿度19%.") == "湿度19%."


class _CharTokenizer:
    def encode_ids(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def encode_with_text(self, text, add_special_tokens=False):
        ids = self.encode_ids(text, add_special_tokens=add_special_tokens)
        return ids, list(text)

    def debug_snapshot(self, text, add_special_tokens=False):
        ids = self.encode_ids(text, add_special_tokens=add_special_tokens)
        return {
            "text": text,
            "ids": ids,
            "pieces": [
                {
                    "index": idx,
                    "id": token_id,
                    "token": ch,
                    "span": ch,
                    "offset": [idx, idx + 1],
                }
                for idx, (token_id, ch) in enumerate(zip(ids, text))
            ],
        }


def test_count_text_tokens_uses_synthesis_normalization_and_tokenizer():
    interface = FrontendInterface(
        engine_inbox=asyncio.Queue(maxsize=16),
        tokenizer=_CharTokenizer(),
        max_sessions=2,
        engine_max_decode_len=64,
    )

    assert interface.count_text_tokens(" 你好😊\n世界 ") == len("你好世界")
    assert interface.count_text_tokens("😊🚀") == 0


def test_direct_frontend_constructor_inherits_legacy_ratio_for_safety() -> None:
    interface = FrontendInterface(
        engine_inbox=asyncio.Queue(maxsize=16),
        tokenizer=_CharTokenizer(),
        max_sessions=2,
        engine_max_decode_len=120,
        ema_ratio=2.0,
    )

    assert interface._safety_ratio_initial == pytest.approx(2.0)


def test_frontend_rejects_safety_baseline_above_ratio_max_at_startup() -> None:
    with pytest.raises(ValueError, match="ema_max_ratio"):
        FrontendInterface(
            engine_inbox=asyncio.Queue(maxsize=16),
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            safety_ratio_initial=5.5,
            ema_max_ratio=5.0,
        )


def test_diagnostic_text_alias_is_exact_and_uses_independent_versions():
    version_text = format_engine_model_version(
        DEFAULT_ENGINE_VERSION,
        DEFAULT_MODEL_VERSION,
    )
    assert version_text == (
        "引擎版本号：v0.2.0a14，"
        "模型版本号：rime@20260820_580_5090_v1-zehan@20260818"
    )
    assert version_text == DEFAULT_ENGINE_MODEL_VERSION
    assert resolve_diagnostic_text(VERSION_QUERY_TEXT, version_text) == version_text
    assert resolve_diagnostic_text(
        f"请合成{VERSION_QUERY_TEXT}", version_text
    ) == f"请合成{VERSION_QUERY_TEXT}"


@pytest.mark.parametrize("input_mode", [InputMode.FULL_TEXT, InputMode.AUTO])
def test_version_query_synthesizes_engine_model_version(input_mode):
    async def run():
        # The labeled engine + model identity is intentionally longer than the
        # old bare version pair; keep this isolated unit inbox large enough to
        # hold the complete dispatch without a concurrent engine consumer.
        inbox = asyncio.Queue(maxsize=256)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=256,
            # This test covers diagnostic replacement, not production safety
            # segmentation; keep the historical two-slot fixture.
            safety_ratio_initial=4.5,
        )
        session = await interface.create_session(
            f"version-query-{input_mode.value}",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=input_mode,
                group_policy=GroupPolicy.NONE,
            ),
        )

        # Exercise packetized transport input: replacement happens only after
        # the full-text session is complete.
        await interface.push_text_input(session.session_id, VERSION_QUERY_TEXT[:5])
        await interface.push_text_input(session.session_id, VERSION_QUERY_TEXT[5:])
        await interface.mark_input_complete(session.session_id)

        requests = await _drain_requests(inbox)
        token_text = "".join(
            chr(token_id)
            for request in requests
            for token_id in (request.token_ids or [])
        )
        assert token_text == DEFAULT_ENGINE_MODEL_VERSION
        assert session.text_journal.raw_text == DEFAULT_ENGINE_MODEL_VERSION
        assert session.text_journal.normalized_text == DEFAULT_ENGINE_MODEL_VERSION

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SESSION_DONE,
                session_id=session.session_id,
            )
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_token_spans_keep_session_raw_coordinates_after_normalization():
    interface = FrontendInterface(
        engine_inbox=asyncio.Queue(maxsize=16),
        tokenizer=_CharTokenizer(),
        max_sessions=2,
        engine_max_decode_len=64,
    )
    journal = CanonicalTextJournal(_normalize_tts_text)
    normalized, offset = journal.append("你好😊 世界")
    tokens = interface._tokenize_segment_text(
        normalized,
        normalized_offset=offset,
        journal=journal,
    )

    world = next(token for token in tokens if token.text == "世")
    assert world.normalized_start == 3
    assert world.normalized_end == 4
    assert world.raw_start == 4
    assert world.raw_end == 5


def test_text_progress_keeps_later_segment_global_coordinates():
    session = SimpleNamespace(
        spliter=SimpleNamespace(ema_ratio_for_segment=lambda _segment_idx: 4.5),
        segment_progress_frames={},
        text_progress_estimators={},
        segment_token_spans={
            1: [
                {
                    "normalized_start": 10,
                    "normalized_end": 11,
                    "raw_start": 12,
                    "raw_end": 13,
                },
            ]
        },
        text_journal=None,
        input_complete=False,
    )
    event = FrontendInterface._make_text_progress_event(
        None,
        session,
        1,
        {"source_frame_end": "5", "text_tokens": "1"},
    )
    assert event is not None
    assert event["meta"]["normalized_codepoint_start"] == "10"
    assert event["meta"]["normalized_codepoint_end"] == "11"
    assert event["meta"]["raw_codepoint_start"] == "12"
    assert event["meta"]["raw_codepoint_end"] == "13"


def test_text_progress_keeps_trailing_audio_at_segment_end():
    session = SimpleNamespace(
        spliter=SimpleNamespace(ema_ratio_for_segment=lambda _segment_idx: 4.5),
        segment_progress_frames={},
        text_progress_estimators={},
        segment_token_spans={
            0: [
                {
                    "normalized_start": start,
                    "normalized_end": end,
                    "raw_start": start,
                    "raw_end": end,
                }
                for start, end in ((0, 3), (3, 5), (5, 7), (7, 9))
            ]
        },
        text_journal=None,
        input_complete=True,
    )

    before_end = FrontendInterface._make_text_progress_event(
        None,
        session,
        0,
        {"source_frame_end": "13", "text_tokens": "4"},
    )
    reaches_end = FrontendInterface._make_text_progress_event(
        None,
        session,
        0,
        {"source_frame_end": "18", "text_tokens": "4"},
    )
    trailing_audio = FrontendInterface._make_text_progress_event(
        None,
        session,
        0,
        {"source_frame_end": "19", "text_tokens": "4"},
    )

    assert before_end is not None
    assert reaches_end is not None
    assert trailing_audio is not None
    assert reaches_end["meta"]["raw_codepoint_end"] == "9"
    assert trailing_audio["meta"]["raw_codepoint_start"] == "9"
    assert trailing_audio["meta"]["raw_codepoint_end"] == "9"
    assert trailing_audio["meta"]["normalized_codepoint_start"] == "9"
    assert trailing_audio["meta"]["normalized_codepoint_end"] == "9"


async def _drain_requests(inbox: asyncio.Queue) -> list:
    requests = []
    while not inbox.empty():
        requests.append(await inbox.get())
    return requests


def test_token_mode_filters_leading_whitespace_only_chunks():
    async def run():
        inbox = asyncio.Queue(maxsize=16)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )

        session = await interface.create_session(
            "space-token",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input("space-token", " ")

        requests = await _drain_requests(inbox)
        assert [request.type for request in requests] == [RequestType.NEW_SESSION]
        assert session.text_journal.raw_text == " "
        assert session.text_journal.normalized_text == ""
        assert not getattr(session, "_first_text_sent", False)

        await interface.mark_input_complete("space-token")
        completed = await _drain_requests(inbox)
        assert [request.type for request in completed] == [
            RequestType.SESSION_TOKENS_DONE
        ]

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="space-token")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("input_mode", "group_policy"),
    [
        (InputMode.TOKEN, GroupPolicy.NONE),
        (InputMode.AUTO, GroupPolicy.NONE),
        (InputMode.CLAUSE, GroupPolicy.NONE),
        (InputMode.LONG_SEGMENT, GroupPolicy.NONE),
        (InputMode.LONG_SEGMENT, GroupPolicy.AUTO),
        (InputMode.FULL_TEXT, GroupPolicy.NONE),
    ],
)
def test_input_modes_filter_only_the_session_leading_whitespace(
    input_mode: InputMode,
    group_policy: GroupPolicy,
):
    async def run():
        inbox = asyncio.Queue(maxsize=64)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )
        session = await interface.create_session(
            f"prefix-{input_mode.value}-{group_policy.value}",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=input_mode,
                group_policy=group_policy,
            ),
        )

        await interface.push_text_input(session.session_id, " \t\n\u00a0\u2003\u3000")
        before_text = await _drain_requests(inbox)
        assert [request.type for request in before_text] == [RequestType.NEW_SESSION]
        assert session.text_journal.normalized_text == ""

        await interface.push_text_input(session.session_id, "你好。")
        await interface.mark_input_complete(session.session_id)

        requests = await _drain_requests(inbox)
        token_text = "".join(
            chr(token_id)
            for request in requests
            for token_id in (request.token_ids or [])
        )
        assert token_text == "你好。"
        assert session.text_journal.normalized_text == "你好。"
        assert session.text_journal.raw_span(0, 1) == (6, 7)

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SESSION_DONE,
                session_id=session.session_id,
            )
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_token_mode_preserves_whitespace_after_text_begins():
    async def run():
        inbox = asyncio.Queue(maxsize=64)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )
        session = await interface.create_session(
            "interior-token",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input(session.session_id, " hello")
        await interface.push_text_input(session.session_id, " ")
        await interface.push_text_input(session.session_id, "world。")
        await interface.mark_input_complete(session.session_id)

        requests = await _drain_requests(inbox)
        token_text = "".join(
            chr(token_id)
            for request in requests
            for token_id in (request.token_ids or [])
        )
        assert token_text == "hello world。"
        assert session.text_journal.normalized_text == "hello world。"
        assert session.text_journal.raw_span(5, 6) == (6, 7)

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SESSION_DONE,
                session_id=session.session_id,
            )
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_full_text_finish_keeps_a_literal_digit_after_leading_whitespace():
    async def run():
        inbox = asyncio.Queue(maxsize=16)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )
        session = await interface.create_session(
            "full-text-leading-digit",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.FULL_TEXT,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input(session.session_id, " 1")
        assert session.text_journal.normalized_text == ""
        await interface.mark_input_complete(session.session_id)

        requests = await _drain_requests(inbox)
        token_text = "".join(
            chr(token_id)
            for request in requests
            for token_id in (request.token_ids or [])
        )
        assert token_text == "1"
        assert session.text_journal.normalized_text == "1"
        assert session.text_journal.raw_span(0, 1) == (1, 2)

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SESSION_DONE,
                session_id=session.session_id,
            )
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_cross_packet_keycap_emoji_does_not_leak_base_digit():
    """A keycap '1️⃣' split across two transport packets must not feed the bare
    base digit '1' to the backend (Stage-0 cross-packet carry)."""

    async def run():
        inbox = asyncio.Queue(maxsize=64)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )
        await interface.create_session(
            "kc",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input(
            "kc", "hello1"
        )  # keycap base, sequence incomplete
        await interface.push_text_input("kc", "️⃣world")  # VS-16 + combining keycap
        await interface.mark_input_complete("kc")

        requests = await _drain_requests(inbox)
        token_ids = [tid for r in requests for tid in (r.token_ids or [])]
        text = "".join(chr(t) for t in token_ids)
        assert ord("1") not in token_ids, text  # the '1' must not reach the backend
        assert "hello" in text and "world" in text, text

    asyncio.run(run())


def test_token_mode_serial_segments_defers_session_done_until_buffer_drains():
    async def run():
        inbox = asyncio.Queue(maxsize=128)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            # C=floor((48-12-8)/2)=14 and T1=10, so the two 10/11-token
            # sentences form exactly two serial segments under the exact cap.
            engine_max_decode_len=48,
            ema_ratio=2.0,
            safety_ratio_initial=2.0,
            max_concurrent_segments=1,
        )

        session = await interface.create_session(
            "serial-token",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input(
            "serial-token",
            "第一句很长很长很长。第二句也很长很长很长。",
        )
        await interface.mark_input_complete("serial-token")

        initial = await _drain_requests(inbox)
        assert any(
            request.type == RequestType.SEGMENT_TOKENS_DONE and request.segment_idx == 0
            for request in initial
        )
        assert not any(
            request.type == RequestType.SESSION_TOKENS_DONE for request in initial
        )

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="serial-token",
                segment_idx=0,
                metrics={"audio_steps": 10, "text_tokens": 10},
            )
        )
        await asyncio.sleep(0)

        after_first_segment = await _drain_requests(inbox)
        request_types = [request.type for request in after_first_segment]
        second_start_idx = request_types.index(RequestType.START_TOKENS)
        session_done_idx = request_types.index(RequestType.SESSION_TOKENS_DONE)
        assert second_start_idx < session_done_idx
        assert after_first_segment[second_start_idx].segment_idx == 1

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="serial-token",
                segment_idx=1,
                metrics={"audio_steps": 10, "text_tokens": 10},
            )
        )
        await asyncio.sleep(0)

        assert await _drain_requests(inbox) == []

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="serial-token")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_full_text_overflow_replans_hidden_suffix_before_opening_next_segment():
    async def run():
        inbox = asyncio.Queue(maxsize=512)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=120,
            prefill_len=12,
            ema_ratio=2.0,
            safety_ratio_initial=2.0,
            ema_overflow_alpha=1.0,
            safety_failure_multiplier=1.0,
            max_concurrent_segments=1,
        )
        session = await interface.create_session(
            "full-replan",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.FULL_TEXT,
                group_policy=GroupPolicy.AUTO,
            ),
        )

        await interface.push_text_input("full-replan", "甲" * 150)
        await interface.mark_input_complete("full-replan")
        initial = await _drain_requests(inbox)
        assert sum(
            len(request.token_ids or [])
            for request in initial
            if request.segment_idx == 0
        ) == 50

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="full-replan",
                segment_idx=0,
                metrics={
                    "audio_steps": 105,
                    "text_tokens": 50,
                    "overflow": True,
                    "eos_reason": "kv_overflow",
                },
            )
        )
        await asyncio.sleep(0)

        after_overflow = await _drain_requests(inbox)
        assert sum(
            len(request.token_ids or [])
            for request in after_overflow
            if request.segment_idx == 1
        ) == 47
        assert session.spliter.safety_ratio == pytest.approx(2.1)

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="full-replan")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_token_mode_final_punct_flush_sends_session_done_without_empty_segment():
    async def run():
        inbox = asyncio.Queue(maxsize=128)
        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=100,
            ema_ratio=10.0,
            max_concurrent_segments=2,
        )

        session = await interface.create_session(
            "boundary-token",
            config=SessionConfig(
                task_type="custom_voice",
                speaker="Serena",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
        )

        await interface.push_text_input("boundary-token", "甲乙丙丁戊己。")
        await interface.mark_input_complete("boundary-token")

        initial = await _drain_requests(inbox)
        assert any(
            request.type == RequestType.SEGMENT_TOKENS_DONE and request.segment_idx == 0
            for request in initial
        )
        assert any(
            request.type == RequestType.SESSION_TOKENS_DONE for request in initial
        )
        assert not any(
            request.type == RequestType.START_TOKENS and request.segment_idx == 1
            for request in initial
        )

        await session.result_queue.put(
            EngineResult(
                type=ResultType.SEGMENT_END,
                session_id="boundary-token",
                segment_idx=0,
                metrics={"audio_steps": 10, "text_tokens": 7},
            )
        )
        await asyncio.sleep(0)

        assert await _drain_requests(inbox) == []

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="boundary-token")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_prefill_done_event_exposes_reference_metadata():
    async def run():
        inbox = asyncio.Queue(maxsize=16)
        events = []

        async def on_event(sid: str, event: dict):
            events.append((sid, event))

        interface = FrontendInterface(
            engine_inbox=inbox,
            tokenizer=_CharTokenizer(),
            max_sessions=2,
            engine_max_decode_len=64,
        )

        session = await interface.create_session(
            "icl-meta",
            config=SessionConfig(
                task_type="voice_clone",
                input_mode=InputMode.TOKEN,
                group_policy=GroupPolicy.NONE,
            ),
            on_event=on_event,
        )

        await session.result_queue.put(
            EngineResult(
                type=ResultType.PREFILL_DONE,
                session_id="icl-meta",
                segment_idx=0,
                metrics={
                    "ref_source": "registry",
                    "ref_id": "Vivian",
                    "ref_audio_sha256": "abcdef123456",
                    "icl_cache_hit": "true",
                    "ref_preprocess_runtime": "trt",
                },
            )
        )
        await asyncio.sleep(0)

        assert events == [
            (
                "icl-meta",
                {
                    "type": "prefill_done",
                    "segment_idx": 0,
                    "text": "",
                    "meta": {
                        "ref_source": "registry",
                        "ref_id": "Vivian",
                        "ref_audio_sha256": "abcdef123456",
                        "icl_cache_hit": "true",
                        "ref_preprocess_runtime": "trt",
                    },
                },
            )
        ]

        await session.result_queue.put(
            EngineResult(type=ResultType.SESSION_DONE, session_id="icl-meta")
        )
        await asyncio.sleep(0)

    asyncio.run(run())


def test_tokenizer_observability_logs_raw_and_normalized_text(caplog):
    interface = FrontendInterface(
        engine_inbox=asyncio.Queue(maxsize=16),
        tokenizer=_CharTokenizer(),
        max_sessions=2,
        engine_max_decode_len=64,
    )

    with caplog.at_level(logging.DEBUG, logger="engine.frontend.interface"):
        interface._tokenize_segment_text("湿度19%，气温为23℃。")

    observability_logs = [
        record.message
        for record in caplog.records
        if "Tokenizer observability:" in record.message
    ]
    assert observability_logs, "expected tokenizer observability logs"
    payload = json.loads(
        observability_logs[-1].split("Tokenizer observability: ", 1)[1]
    )
    assert payload["field"] == "segment_text"
    assert payload["normalized_text"] == "湿度19%，气温为23℃。"
    assert payload["tokenizer"]["ids"]
    assert payload["tokenizer"]["pieces"]


def _make_summary_stub(session_id: str):
    """Minimal stand-in carrying the attributes _cleanup_session's summary reads."""
    return SimpleNamespace(
        session_id=session_id,
        segments_done=0,
        segments_submitted=0,
        total_audio_bytes=0,
        session_create_to_first_raw_audio_ms=None,
        first_raw_audio_at=None,
        first_text_enqueued_at=None,
        first_text_dequeued_at=None,
        prefill_completed_at=None,
        prefill_started_at=None,
        config=SimpleNamespace(timing=SimpleNamespace(request_id="", turn_id="")),
    )


def test_cleanup_session_identity_guard_protects_recreated_session():
    """A stale cancelled task must not clobber a session re-created under the
    same id (the session_id reuse race)."""
    interface = FrontendInterface.__new__(FrontendInterface)
    interface._sessions = {}
    interface._consumer_tasks = {}

    sid = "reused"
    old = _make_summary_stub(sid)
    new = _make_summary_stub(sid)
    # The NEW session currently owns the id.
    interface._sessions[sid] = new
    interface._consumer_tasks[sid] = "new-task"

    # The OLD task's deferred cleanup must be a no-op (its session is gone).
    interface._cleanup_session(sid, expected=old)
    assert interface._sessions[sid] is new
    assert interface._consumer_tasks[sid] == "new-task"

    # The matching cleanup still removes it.
    interface._cleanup_session(sid, expected=new)
    assert sid not in interface._sessions
    assert sid not in interface._consumer_tasks
