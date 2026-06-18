"""Tests for engine loop pipeline correctness and session timeout."""

import asyncio
import queue
import time
import threading
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
    RequestPriority,
)


@pytest.fixture
def model_config():
    return ModelConfig(
        num_layers=2, kv_heads=2, head_dim=4, max_seq_len=16,
        n_c2w_layers=2, c2w_kv_heads=2, c2w_head_dim=4, c2w_sliding_window=8,
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
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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

    def test_failed_prefill_cleanup_releases_slot_and_removes_session(self, model_config):
        inbox = queue.Queue()
        loop = _ImmediateLoop()

        class StubExecutor:
            kv_pool = KVCachePool(
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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
            max_slots=4, config=model_config,
            device=torch.device("cpu"), preallocate=False,
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
                max_slots=4, config=model_config,
                device=torch.device("cpu"), preallocate=False,
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

    def test_prefix_cache_hit_restores_prefix_and_lets_decode0_consume_text(self, model_config):
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
