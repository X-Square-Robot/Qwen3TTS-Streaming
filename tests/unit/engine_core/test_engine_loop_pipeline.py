"""Tests for engine loop pipeline correctness and session timeout."""

import asyncio
import queue
import time
from types import SimpleNamespace

import torch
import pytest

from engine.backend.kv_cache_pool import KVCachePool, ModelConfig, SlotKVState
from engine.backend.prefill import PrefillPlan
from engine.backend.executor import StepOutput
from engine.backend.engine_loop import (
    EngineLoop,
    EngineSegment,
    EngineSessionGroup,
)
from engine.core.types import (
    EngineRequest,
    EngineResult,
    RequestType,
    ResultType,
)


@pytest.fixture
def model_config():
    return ModelConfig(
        num_layers=2,
        kv_heads=2,
        head_dim=4,
        max_seq_len=16,
        n_c2w_layers=2,
        c2w_kv_heads=2,
        c2w_head_dim=4,
        c2w_sliding_window=8,
    )


class TestEngineSessionGroup:
    def test_created_at_is_set(self):
        req = EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        group = EngineSessionGroup("s1", req)
        assert group.created_at > 0
        assert time.monotonic() - group.created_at < 1.0

    def test_active_slot_count(self):
        req = EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        group = EngineSessionGroup("s1", req)
        seg = EngineSegment("s1", 0)
        seg.slot = SlotKVState(slot_id=0)
        seg.state = "active"
        group.segments[0] = seg
        assert group.active_slot_count == 1


class TestEngineLoopHealth:
    def test_health_stats_initial(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        stats = engine_loop.health_stats()
        assert stats["running"] is False
        assert stats["active_sessions"] == 0
        assert stats["total_steps"] == 0
        assert stats["total_prefills"] == 0
        assert stats["total_evictions"] == 0
        assert stats["total_timeouts"] == 0
        assert stats["free_slots"] == 4
        loop.close()


class TestSessionTimeout:
    def test_timeout_detection(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
            session_timeout_sec=0.05,
        )

        result_queue = asyncio.Queue()
        req = EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id="s1",
            result_queue=result_queue,
        )
        engine_loop._handle_request(req)
        assert "s1" in engine_loop._groups

        engine_loop._groups["s1"].created_at = time.monotonic() - 1.0

        engine_loop._try_timeout_sessions()
        assert "s1" not in engine_loop._groups
        assert engine_loop._total_timeouts == 1
        loop.close()

    def test_no_timeout_within_limit(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
            session_timeout_sec=300.0,
        )

        req = EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id="s1",
        )
        engine_loop._handle_request(req)
        engine_loop._try_timeout_sessions()
        assert "s1" in engine_loop._groups
        assert engine_loop._total_timeouts == 0
        loop.close()


class _ImmediateLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class TestSessionCancel:
    def test_cancel_emits_session_done_and_removes_group(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        result_queue = queue.Queue()
        new_req = EngineRequest(
            type=RequestType.NEW_SESSION,
            session_id="cancel-me",
            result_queue=result_queue,
        )
        engine_loop._handle_request(new_req)
        assert "cancel-me" in engine_loop._groups

        cancel_req = EngineRequest(
            type=RequestType.CANCEL_SESSION,
            session_id="cancel-me",
        )
        engine_loop._handle_request(cancel_req)

        result = result_queue.get_nowait()
        assert isinstance(result, EngineResult)
        assert result.type == ResultType.SESSION_DONE
        assert result.session_id == "cancel-me"
        assert result.metrics == {"cancelled": True}
        assert "cancel-me" not in engine_loop._groups

    def test_new_session_replacement_releases_existing_slot(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        engine_loop._handle_request(
            EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        )
        slot = StubExecutor.kv_pool.allocate("s1:0")
        seg = EngineSegment("s1", 0)
        seg.slot = slot
        seg.state = "active"
        engine_loop._groups["s1"].segments[0] = seg
        engine_loop._seg_by_slot[slot.slot_id] = seg

        engine_loop._handle_request(
            EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        )

        assert slot.is_free is True
        assert slot.slot_id not in engine_loop._seg_by_slot
        assert engine_loop._groups["s1"].segments == {}
        assert StubExecutor.kv_pool.free_count == 4

    def test_duplicate_start_tokens_releases_replaced_segment_slot(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        engine_loop._handle_request(
            EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        )
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[1],
            )
        )
        old_seg = engine_loop._groups["s1"].segments[0]
        slot = StubExecutor.kv_pool.allocate("s1:0")
        old_seg.slot = slot
        old_seg.state = "active"
        engine_loop._seg_by_slot[slot.slot_id] = old_seg

        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[2],
            )
        )

        new_seg = engine_loop._groups["s1"].segments[0]
        assert slot.is_free is True
        assert old_seg.slot is None
        assert new_seg is not old_seg
        assert new_seg.pending_token_ids == [2]
        assert new_seg.slot is None
        assert slot.slot_id not in engine_loop._seg_by_slot
        assert StubExecutor.kv_pool.free_count == 4

    def test_failed_prefill_cleanup_releases_slot_and_removes_session(
        self, model_config
    ):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                result_queue=result_queue,
            )
        )
        seg = EngineSegment("s1", 0)
        seg.state = "pending_prefill"
        seg.slot = StubExecutor.kv_pool.allocate("s1:0")
        engine_loop._groups["s1"].segments[0] = seg
        engine_loop._seg_by_slot[seg.slot.slot_id] = seg

        engine_loop._cleanup_failed_prefills()

        result = result_queue.get_nowait()
        assert result.type == ResultType.ERROR
        assert result.session_id == "s1"
        assert "s1" not in engine_loop._groups
        assert StubExecutor.kv_pool.free_count == 4


class TestProcessStepOutput:
    def test_process_updates_slot_state(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        pool = KVCachePool(
            max_slots=4,
            config=model_config,
            device=torch.device("cpu"),
            preallocate=False,
        )

        class StubExecutor:
            kv_pool = pool
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        slot = pool.allocate("s1")
        slot.past_len = 5
        slot.frame_idx = 3

        req = EngineRequest(type=RequestType.NEW_SESSION, session_id="s1")
        engine_loop._handle_request(req)

        seg = EngineSegment("s1", 0)
        seg.slot = slot
        seg.state = "active"
        engine_loop._groups["s1"].segments[0] = seg
        engine_loop._seg_by_slot[slot.slot_id] = seg

        output = StepOutput(
            slots=[slot],
            eos_flags=[False],
            audio_chunks=[b"\x00\x01"],
            batch_talker_kv=None,
            batch_c2w_kv=None,
            split_c2w_conv=[[]],
            split_c2w_transconv=[[]],
            codec_sum=torch.randn(1, 1, model_config.hidden_size),
            updated_tc=torch.zeros(1, model_config.codec_vocab_size, dtype=torch.int64),
        )

        engine_loop._process_step_output(output)
        assert slot.past_len == 6
        assert slot.frame_idx == 4
        assert slot.next_embed is None
        assert slot.last_codec_sum is not None
        loop.close()


class TestConfigNewFields:
    def test_server_config_has_warmup_and_health(self):
        from engine.config import ServerConfig

        sc = ServerConfig()
        assert sc.warmup_rounds == 3
        assert sc.health_port == 8080

    def test_scheduler_config_has_timeout_and_pad_silence_thresholds(self):
        from engine.config import SchedulerConfig

        sc = SchedulerConfig()
        assert sc.session_timeout_sec == 300.0
        assert sc.pad_silence_peak_threshold == 5e-4
        assert sc.pad_silence_mean_abs_threshold == 2e-4


class TestPadSilenceDetection:
    def test_pad_silence_detection_uses_peak_and_mean_abs(self, model_config):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4,
                config=model_config,
                device=torch.device("cpu"),
                preallocate=False,
            )
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
        )

        near_silence = torch.full((1920,), 1.5e-4, dtype=torch.float32)
        near_silence[0] = 4.5e-4
        assert engine_loop._is_pad_silence(near_silence.numpy().tobytes()) is True

        audible = torch.full((1920,), 1.5e-4, dtype=torch.float32)
        audible[0] = 8e-4
        assert engine_loop._is_pad_silence(audible.numpy().tobytes()) is False
        loop.close()


class _StubPrefillBuilder:
    def __init__(
        self,
        *,
        plan: PrefillPlan | None = None,
        suffix: tuple[torch.Tensor, list[torch.Tensor]] | None = None,
        cache_key: str = "cache-key",
        hidden_size: int = 2048,
    ):
        self._plan = plan
        self._suffix = suffix
        self._cache_key = cache_key
        self._hidden_size = hidden_size
        self.w = SimpleNamespace(
            tts_pad_embed=torch.zeros(1, 1, hidden_size, dtype=torch.bfloat16),
        )

    def compute_cache_key(self, *args, **kwargs):
        return self._cache_key

    def build_plan_from_ids(self, **kwargs):
        if self._plan is None:
            raise AssertionError("build_plan_from_ids should not be called")
        return self._plan

    def build_suffix_from_ids(self, token_ids, include_eos=True):
        if self._suffix is None:
            raise AssertionError("build_suffix_from_ids should not be called")
        return self._suffix

    def build_suffix_batch(self, token_lists, include_eos_flags):
        self.batch_calls = getattr(self, "batch_calls", 0) + 1
        results = []
        for i in range(len(token_lists)):
            req = torch.full((1, 1, self._hidden_size), float(i + 1))
            trailing = [torch.full((1, 1, self._hidden_size), float(100 + i))]
            results.append((req, trailing))
        return results


class _StubExecutorForPrefill:
    def __init__(self, model_config):
        self._config = model_config
        self._device = torch.device("cpu")
        self.kv_pool = KVCachePool(
            max_slots=4,
            config=model_config,
            device=torch.device("cpu"),
            preallocate=False,
        )
        self.prefill_inputs: list[torch.Tensor] = []
        self.prefill_prefix_only_inputs: list[torch.Tensor] = []
        self.prefill_from_prefix_inputs: list[torch.Tensor] = []

    def make_zero_conv_states(self):
        return [torch.zeros(1, 1, 1)]

    def make_zero_transconv_states(self):
        return [torch.zeros(1, 1, 1)]

    def make_zero_states_batch(self, count):
        return [
            (
                self.make_zero_conv_states(),
                self.make_zero_transconv_states(),
                self.make_zero_conv_states(),
                self.make_zero_transconv_states(),
            )
            for _ in range(count)
        ]

    def prefill(self, slot, embeds):
        self.prefill_inputs.append(embeds.clone())
        seq = int(embeds.shape[1])
        slot.talker_kv = torch.zeros(
            1,
            self._config.num_layers * 2,
            self._config.kv_heads,
            seq,
            self._config.head_dim,
        )
        slot.past_len = seq
        slot.frame_idx = 1
        slot.next_embed = torch.full(
            (1, 1, self._config.hidden_size),
            10.0,
        )
        slot.c2w_conv_states = self.make_zero_conv_states()
        slot.c2w_transconv_states = self.make_zero_transconv_states()
        slot.init_pingpong_buffers()
        slot.token_counts = torch.zeros(
            1,
            self._config.codec_vocab_size,
            dtype=torch.int64,
        )
        return b"", False

    def prefill_prefix_only(self, slot, embeds):
        self.prefill_prefix_only_inputs.append(embeds.clone())
        seq = int(embeds.shape[1])
        slot.talker_kv = torch.zeros(
            1,
            self._config.num_layers * 2,
            self._config.kv_heads,
            seq,
            self._config.head_dim,
        )
        slot.past_len = seq

    def prefill_from_prefix(self, slot, embeds):
        self.prefill_from_prefix_inputs.append(embeds.clone())
        seq = int(embeds.shape[1])
        slot.talker_kv = torch.zeros(
            1,
            self._config.num_layers * 2,
            self._config.kv_heads,
            slot.past_len + seq,
            self._config.head_dim,
        )
        slot.past_len += seq
        slot.frame_idx = 1
        slot.next_embed = torch.full(
            (1, 1, self._config.hidden_size),
            7.0,
        )
        slot.token_counts = torch.ones(
            1,
            self._config.codec_vocab_size,
            dtype=torch.int64,
        )
        return b"audio", False


class TestPrefillBoundary:
    def test_full_prefill_can_prime_decode0_from_prefix_only_state(self, model_config):
        loop = _ImmediateLoop()
        executor = _StubExecutorForPrefill(model_config)
        hidden = model_config.hidden_size

        prefill = torch.randn(1, 3, hidden)
        trailing = [torch.full((1, 1, hidden), 2.0)]
        plan = PrefillPlan(
            prefill_embeds=prefill,
            trailing=trailing,
            prefix_cache_key="cache-key",
            cacheable_prefix_embeds=prefill[:, :2, :].clone(),
            request_prefill_embeds=prefill[:, 2:, :].clone(),
        )
        builder = _StubPrefillBuilder(plan=plan, hidden_size=hidden)

        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=loop,
            executor=executor,
            prefill_builder=builder,
            max_batch_size=4,
        )

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                task_type="custom_voice",
                result_queue=result_queue,
            )
        )
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[1, 2, 3],
            )
        )

        assert engine_loop._try_prefill_one() is True

        seg = engine_loop._groups["s1"].segments[0]
        slot = seg.slot
        assert slot is not None
        assert executor.prefill_inputs == []
        assert len(executor.prefill_prefix_only_inputs) == 1
        torch.testing.assert_close(
            executor.prefill_prefix_only_inputs[0],
            prefill[:, :2, :],
        )
        assert executor.prefill_from_prefix_inputs == []
        assert slot.prefill_source == "full_prefill_prefix_only"
        assert slot.past_len == 2
        assert slot.frame_idx == 0
        assert slot.text_idx == 0
        torch.testing.assert_close(
            slot.next_embed,
            prefill[:, 2:, :].to(torch.float32),
        )

    def test_prefix_cache_hit_restores_prefix_and_lets_decode0_consume_text(
        self, model_config
    ):
        loop = _ImmediateLoop()
        executor = _StubExecutorForPrefill(model_config)
        hidden = model_config.hidden_size

        req_embeds = torch.randn(1, 1, hidden)
        trailing = [torch.full((1, 1, hidden), 2.0)]
        builder = _StubPrefillBuilder(
            suffix=(req_embeds, trailing),
            cache_key="cache-key",
            hidden_size=hidden,
        )

        engine_loop = EngineLoop(
            engine_inbox=queue.Queue(),
            async_loop=loop,
            executor=executor,
            prefill_builder=builder,
            max_batch_size=4,
        )
        cached_kv = torch.zeros(
            1,
            model_config.num_layers * 2,
            model_config.kv_heads,
            2,
            model_config.head_dim,
        )
        engine_loop._prefix_cache.put("cache-key", cached_kv, 2)

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                task_type="custom_voice",
                result_queue=result_queue,
            )
        )
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s1",
                segment_idx=0,
                token_ids=[1, 2],
            )
        )

        assert engine_loop._try_prefill_one() is True

        seg = engine_loop._groups["s1"].segments[0]
        slot = seg.slot
        assert slot is not None
        assert executor.prefill_inputs == []
        assert executor.prefill_prefix_only_inputs == []
        assert executor.prefill_from_prefix_inputs == []
        assert slot.prefill_source == "prefix_cache_prefix_only"
        assert slot.past_len == 2
        assert slot.frame_idx == 0
        assert slot.text_idx == 0
        torch.testing.assert_close(
            slot.next_embed,
            req_embeds.to(torch.float32),
        )


class TestTokenLoopGuard:
    """Token loop guard: same codebook-0 token N consecutive steps → loop_abort."""

    def _make_env(self, model_config, guard_frames):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        pool = KVCachePool(
            max_slots=4,
            config=model_config,
            device=torch.device("cpu"),
            preallocate=False,
        )

        class StubExecutor:
            kv_pool = pool
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
            token_loop_abort_frames=guard_frames,
        )

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                result_queue=result_queue,
            )
        )
        slot = pool.allocate("s1:0")
        seg = EngineSegment("s1", 0)
        seg.slot = slot
        seg.state = "active"
        engine_loop._groups["s1"].segments[0] = seg
        engine_loop._seg_by_slot[slot.slot_id] = seg
        return engine_loop, loop, pool, seg, slot, result_queue

    @staticmethod
    def _step(engine_loop, slot, token, audio):
        output = StepOutput(
            slots=[slot],
            eos_flags=[False],
            tokens=[token],
            audio_chunks=[audio],
            split_c2w_conv=[[]],
            split_c2w_transconv=[[]],
        )
        engine_loop._process_step_output(output)

    @staticmethod
    def _drain(loop, result_queue):
        loop.run_until_complete(asyncio.sleep(0.01))
        results = []
        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                return results

    def test_loop_abort_after_threshold_with_faded_last_chunk(self, model_config):
        import numpy as np

        engine_loop, loop, pool, seg, slot, result_queue = self._make_env(
            model_config, guard_frames=4
        )
        audio = (np.ones(1920, dtype=np.float32) * 0.5).tobytes()

        for _ in range(3):
            self._step(engine_loop, slot, 1354, audio)
        assert seg.state == "active"
        assert seg.loop_run == 3

        self._step(engine_loop, slot, 1354, audio)
        assert seg.state == "done"
        assert pool.free_count == 4

        results = self._drain(loop, result_queue)
        chunks = [r for r in results if r.type == ResultType.AUDIO_CHUNK]
        ends = [r for r in results if r.type == ResultType.SEGMENT_END]
        assert len(chunks) == 4
        assert len(ends) == 1
        assert ends[0].metrics["eos_reason"] == "loop_abort"

        # First three chunks unmodified; the final one linearly faded to zero.
        for r in chunks[:3]:
            assert r.audio_bytes == audio
        faded = np.frombuffer(chunks[3].audio_bytes, dtype=np.float32)
        assert faded[0] == pytest.approx(0.5)
        assert faded[-1] == 0.0
        assert faded[960] == pytest.approx(0.25, abs=1e-3)
        loop.close()

    def test_run_counter_resets_on_token_change(self, model_config):
        engine_loop, loop, pool, seg, slot, result_queue = self._make_env(
            model_config, guard_frames=4
        )
        audio = b"\x00" * 8
        for token in (7, 7, 7, 8, 8, 8, 7):
            self._step(engine_loop, slot, token, audio)
        assert seg.state == "active"
        assert seg.loop_token == 7
        assert seg.loop_run == 1
        loop.close()

    def test_guard_disabled_with_zero(self, model_config):
        engine_loop, loop, pool, seg, slot, result_queue = self._make_env(
            model_config, guard_frames=0
        )
        audio = b"\x00" * 8
        for _ in range(10):
            self._step(engine_loop, slot, 42, audio)
        assert seg.state == "active"
        loop.close()

    def test_missing_tokens_field_skips_guard(self, model_config):
        engine_loop, loop, pool, seg, slot, result_queue = self._make_env(
            model_config, guard_frames=4
        )
        for _ in range(6):
            output = StepOutput(
                slots=[slot],
                eos_flags=[False],
                audio_chunks=[b"\x00" * 8],
                split_c2w_conv=[[]],
                split_c2w_transconv=[[]],
            )
            engine_loop._process_step_output(output)
        assert seg.state == "active"
        assert seg.loop_run == 0
        loop.close()

    def test_scheduler_config_has_token_loop_abort_frames(self):
        from engine.config import SchedulerConfig

        sc = SchedulerConfig()
        assert sc.token_loop_abort_frames == 4

    def test_fade_out_chunk_is_linear(self):
        import numpy as np

        samples = np.ones(4, dtype=np.float32)
        faded = np.frombuffer(
            EngineLoop._fade_out_chunk(samples.tobytes()), dtype=np.float32
        )
        expected = np.linspace(1.0, 0.0, 4, dtype=np.float32)
        np.testing.assert_allclose(faded, expected)
        assert EngineLoop._fade_out_chunk(b"") == b""


class TestSegmentRetry:
    """Reseed-and-rerun for hallucinated lookahead segments (V2)."""

    def _make_env(self, model_config, *, max_retries=1):
        inbox = queue.Queue()
        loop = asyncio.new_event_loop()

        pool = KVCachePool(
            max_slots=4,
            config=model_config,
            device=torch.device("cpu"),
            preallocate=False,
        )

        class StubExecutor:
            kv_pool = pool
            _device = torch.device("cpu")
            _config = model_config

        engine_loop = EngineLoop(
            engine_inbox=inbox,
            async_loop=loop,
            executor=StubExecutor(),
            max_batch_size=4,
            token_loop_abort_frames=4,
            token_loop_max_retries=max_retries,
        )

        result_queue = queue.Queue()
        engine_loop._handle_request(
            EngineRequest(
                type=RequestType.NEW_SESSION,
                session_id="s1",
                result_queue=result_queue,
            )
        )
        return engine_loop, loop, pool, result_queue

    @staticmethod
    def _add_segment(engine_loop, pool, segment_idx, state="active"):
        seg = EngineSegment("s1", segment_idx)
        seg.slot = pool.allocate(f"s1:{segment_idx}")
        seg.slot.segment_idx = segment_idx
        seg.state = state
        engine_loop._groups["s1"].segments[segment_idx] = seg
        engine_loop._seg_by_slot[seg.slot.slot_id] = seg
        return seg

    @staticmethod
    def _loop_step(engine_loop, slot, token=1354):
        output = StepOutput(
            slots=[slot],
            eos_flags=[False],
            tokens=[token],
            audio_chunks=[b"\x00\x00\x80\x3f" * 480],
            split_c2w_conv=[[]],
            split_c2w_transconv=[[]],
        )
        engine_loop._process_step_output(output)

    @staticmethod
    def _drain(loop, result_queue):
        loop.run_until_complete(asyncio.sleep(0.01))
        results = []
        while True:
            try:
                results.append(result_queue.get_nowait())
            except queue.Empty:
                return results

    def test_lookahead_segment_retries_with_reseed(self, model_config):
        engine_loop, loop, pool, result_queue = self._make_env(model_config)
        self._add_segment(engine_loop, pool, 0)  # earlier live segment
        seg1 = self._add_segment(engine_loop, pool, 1)

        for _ in range(4):
            self._loop_step(engine_loop, seg1.slot)

        assert seg1.state == "pending_prefill"
        assert seg1.retry_idx == 1
        assert seg1.slot is None
        assert seg1.loop_run == 0 and seg1.loop_token == -1
        assert pool.free_count == 3  # seg1's slot back, seg0 still holds one

        results = self._drain(loop, result_queue)
        types = [r.type for r in results]
        assert ResultType.SEGMENT_RETRY in types
        assert ResultType.SEGMENT_END not in types
        retry = next(r for r in results if r.type == ResultType.SEGMENT_RETRY)
        assert retry.segment_idx == 1
        assert retry.metrics["retry_idx"] == 1
        assert retry.metrics["retry_reason"] == "loop"
        # Steps 1-3 streamed; the triggering 4th frame is not sent on retry.
        assert types.count(ResultType.AUDIO_CHUNK) == 3
        loop.close()

    def test_retry_cap_falls_back_to_loop_abort(self, model_config):
        engine_loop, loop, pool, result_queue = self._make_env(model_config)
        self._add_segment(engine_loop, pool, 0)
        seg1 = self._add_segment(engine_loop, pool, 1)
        seg1.retry_idx = 1  # cap (max_retries=1) already spent

        for _ in range(4):
            self._loop_step(engine_loop, seg1.slot)

        assert seg1.state == "done"
        results = self._drain(loop, result_queue)
        ends = [r for r in results if r.type == ResultType.SEGMENT_END]
        assert len(ends) == 1
        assert ends[0].metrics["eos_reason"] == "loop_abort"
        assert not any(r.type == ResultType.SEGMENT_RETRY for r in results)
        loop.close()

    def test_playhead_segment_never_retries(self, model_config):
        engine_loop, loop, pool, result_queue = self._make_env(model_config)
        seg0 = self._add_segment(engine_loop, pool, 0)  # no earlier live seg

        for _ in range(4):
            self._loop_step(engine_loop, seg0.slot)

        assert seg0.state == "done"
        assert seg0.retry_idx == 0
        results = self._drain(loop, result_queue)
        assert any(
            r.type == ResultType.SEGMENT_END
            and r.metrics["eos_reason"] == "loop_abort"
            for r in results
        )
        loop.close()

    def test_retry_salt_changes_seed_conditionally(self):
        from types import SimpleNamespace

        from engine.backend.executor import Executor, _stable_sampling_seed
        from engine.backend.kv_cache_pool import SlotKVState

        ns = SimpleNamespace(_random_seed=0, _device=torch.device("cpu"))

        def seed_for(retry_idx):
            slot = SlotKVState(slot_id=0)
            slot.session_id = "sess:0"
            slot.segment_idx = 0
            slot.retry_idx = retry_idx
            Executor._slot_sampling_generator(ns, slot)
            return slot.sampling_seed

        # retry-0 must keep the historical derivation bit-identical.
        assert seed_for(0) == _stable_sampling_seed(0, "sess:0", 0)
        assert seed_for(1) != seed_for(0)
        assert seed_for(2) != seed_for(1)
        assert seed_for(1) == _stable_sampling_seed(0, "sess:0", 0, "retry:1")

    def test_scheduler_config_has_max_retries(self):
        from engine.config import SchedulerConfig

        assert SchedulerConfig().token_loop_max_retries == 1
