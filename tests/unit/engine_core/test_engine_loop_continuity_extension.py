"""CPU-only lifecycle tests for the optional EngineLoop continuity policy."""

import queue
from types import SimpleNamespace

import pytest
import torch

from engine.backend.engine_loop import (
    EngineLoop,
    EngineSegment,
    EngineSessionGroup,
)
from engine.backend.executor import StepOutput
from engine.core.extensions import CursorContinuationState, EngineExtensions
from engine.core.types import EngineRequest, RequestType, SessionConfig


class _FakeSlot:
    def __init__(self, slot_id: int, session_id: str = ""):
        self.slot_id = slot_id
        self.allocation_epoch = 1
        self.session_id = session_id
        self.segment_idx = -1
        self.retry_idx = 0
        self.is_free = False
        self.frame_idx = 0
        self.past_len = 0
        self.prefill_source = ""
        self.trailing = []
        self.text_idx = 0
        self.extension_state = {}
        self.c2w_kv = None
        self.c2w_pooled = False
        self.c2w_len = 0
        self.c2w_conv_states = None
        self.c2w_transconv_states = None
        self.c2w_arena_backed = False
        self.next_embed = None
        self.last_codec_sum = None
        self.pad_start_frame = -1
        self.pad_consecutive_silence = 0

    @property
    def pingpong_ready(self) -> bool:
        return False

    def touch(self) -> None:
        pass

    def init_pingpong_buffers(self, **_kwargs) -> None:
        pass


class _FakePool:
    _preallocate = False

    def __init__(self, max_slots: int = 1):
        self._next_slot_id = 0
        self._slots = {}
        self._free_count = max_slots
        self.events = []

    @property
    def free_count(self) -> int:
        return self._free_count

    def allocate(self, session_key: str):
        if self._free_count == 0:
            return None
        slot = _FakeSlot(self._next_slot_id, session_key)
        self._next_slot_id += 1
        self._slots[slot.slot_id] = slot
        self._free_count -= 1
        return slot

    def register_live_slot(self, slot: _FakeSlot) -> None:
        self._slots[slot.slot_id] = slot
        self._free_count -= 1

    def release(self, slot_id: int, *, expected_allocation_epoch: int) -> None:
        slot = self._slots[slot_id]
        assert slot.allocation_epoch == expected_allocation_epoch
        self.events.append("release")
        if not slot.is_free:
            slot.session_id = None
            slot.segment_idx = -1
            slot.extension_state = {}
            slot.is_free = True
            self._free_count += 1


class _FakeExecutor:
    def __init__(self, pool: _FakePool):
        self.kv_pool = pool
        self._device = torch.device("cpu")
        self._config = SimpleNamespace(
            dtype=torch.float32,
            hidden_size=4,
            codec_vocab_size=4,
            c2w_sliding_window=8,
        )
        self.prefill_calls = 0
        self._do_sample = False
        self._temperature = 1.0
        self._repetition_penalty = 1.0

    def prefill(self, _slot, _embeds):
        self.prefill_calls += 1
        return None, False

    @staticmethod
    def make_zero_conv_states():
        return [torch.zeros(1, 1, 1)]

    @staticmethod
    def make_zero_transconv_states():
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


class _ImmediateAsyncLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class _AdmissionPolicy:
    def __init__(self, ready: bool):
        self.ready = ready
        self.ready_calls = []
        self.invalidations = []

    def ready_to_admit(self, segment_idx):
        self.ready_calls.append(segment_idx)
        return self.ready

    def invalidate(self, reason):
        self.invalidations.append(reason)


class _RaisingAdmissionPolicy(_AdmissionPolicy):
    def ready_to_admit(self, _segment_idx):
        raise RuntimeError("optional admission callback failed")


class _FinalizePolicy:
    def __init__(self, slot: _FakeSlot, events: list[str]):
        self.slot = slot
        self.events = events
        self.states = []

    def finalize_segment(self, _segment_idx, _token_ids, **kwargs):
        assert self.slot.is_free is False
        self.events.append("finalize")
        self.states.append(kwargs["state"])
        return True


class _SuccessorContextPolicy(_AdmissionPolicy):
    def __init__(self, ready: bool):
        super().__init__(ready)
        self.context_calls = []

    def context_for(self, _segment_idx, **_kwargs):
        self.context_calls.append(_segment_idx)
        return SimpleNamespace(
            c2w_kv=torch.ones(1, 1, 1, 1, 1),
            c2w_conv=[torch.ones(1, 1, 1)],
            c2w_transconv=[torch.ones(1, 1, 1)],
            c2w_frame_idx=3,
        )


class _CursorContextPolicy(_SuccessorContextPolicy):
    def __init__(self, cursor_state):
        super().__init__(ready=True)
        self.cursor_state = cursor_state

    def context_for(self, segment_idx, **kwargs):
        context = super().context_for(segment_idx, **kwargs)
        context.cursor_state = self.cursor_state
        return context


class _CursorCapturePolicy(_SuccessorContextPolicy):
    def __init__(self, *, carried=True):
        super().__init__(ready=True)
        self.captured_state = None
        self.carried = carried

    def finalize_segment(self, _segment_idx, _token_ids, **kwargs):
        self.captured_state = kwargs["state"]
        return self.carried


class _GateFinalizePolicy(_AdmissionPolicy):
    def __init__(self, slot: _FakeSlot, events: list[str]):
        super().__init__(ready=False)
        self.slot = slot
        self.events = events
        self.finalized = []

    def finalize_segment(self, segment_idx, _token_ids, **kwargs):
        assert self.slot.is_free is False
        self.events.append("finalize")
        self.finalized.append((segment_idx, kwargs["state"]))
        return True


class _StatefulContinuityPolicy(_AdmissionPolicy):
    def __init__(self):
        super().__init__(ready=True)
        self.bridge_state = {"bridge": "old"}
        self.capture_state = {"capture": "old"}

    def invalidate(self, reason):
        super().invalidate(reason)
        self.bridge_state = None
        self.capture_state = None

    def reset_segment_capture(self, _segment_idx):
        self.bridge_state = None
        self.capture_state = None


class _RetryContinuityPolicy(_StatefulContinuityPolicy):
    def __init__(self):
        super().__init__()
        self.reset_calls = []
        self.reset_snapshots = []

    def reset_segment_capture(self, segment_idx):
        self.reset_calls.append(segment_idx)
        self.reset_snapshots.append((self.bridge_state, self.capture_state))
        super().reset_segment_capture(segment_idx)


class _BatchPrefillBuilder:
    def __init__(self):
        self.batch_calls = 0
        self.w = SimpleNamespace(tts_pad_embed=torch.zeros(1, 1, 4))

    def compute_cache_key(self, *_args, **_kwargs):
        return "cache-key"

    def build_suffix_batch(self, token_lists, _include_eos_flags):
        self.batch_calls += 1
        return [
            (
                torch.zeros(1, 1, 4),
                [torch.zeros(1, 1, 4)],
            )
            for _ in token_lists
        ]


def _make_loop(*, pool=None, extensions=None):
    pool = pool or _FakePool()
    executor = _FakeExecutor(pool)
    loop = EngineLoop(
        engine_inbox=queue.Queue(),
        async_loop=_ImmediateAsyncLoop(),
        executor=executor,
        extensions=extensions,
    )
    return loop, executor, pool


def _make_group(session_id: str, extensions: EngineExtensions) -> EngineSessionGroup:
    request = EngineRequest(
        type=RequestType.NEW_SESSION,
        session_id=session_id,
        session_config=SessionConfig(),
    )
    return EngineSessionGroup(session_id, request, extensions=extensions)


def _add_done_predecessor(group: EngineSessionGroup, successor_idx: int) -> EngineSegment:
    predecessor = EngineSegment(group.session_id, successor_idx - 1)
    predecessor.state = "done"
    group.segments[predecessor.segment_idx] = predecessor
    return predecessor


def _cursor_state():
    return CursorContinuationState(
        cursor_mu=torch.ones(1),
        cursor_frames_since_advance=torch.ones(1),
        cursor_delta_history=torch.ones(1, 8),
        cursor_conv_history=torch.ones(1, 6, 4),
        cursor_last_trunk_input=torch.ones(1, 4),
        cursor_seen_frames=torch.ones(1, dtype=torch.int64),
    )


def test_ready_to_admit_blocks_successor_before_slot_allocation():
    policy = _AdmissionPolicy(ready=False)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, pool = _make_loop(extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group

    predecessor = EngineSegment("session", 0)
    predecessor.state = "active"
    predecessor.slot = _FakeSlot(10, "session:0")
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments = {0: predecessor, 1: successor}

    assert loop._try_prefill_one() is False
    assert successor.slot is None
    assert pool.free_count == 1
    assert policy.ready_calls == [1]


def test_factory_failure_is_fail_closed_and_default_admission_still_works():
    def broken_factory(_session_id, _config):
        raise RuntimeError("optional continuity package unavailable")

    extensions = EngineExtensions(continuity_factory=broken_factory)
    loop, executor, pool = _make_loop(extensions=extensions)
    request = EngineRequest(
        type=RequestType.NEW_SESSION,
        session_id="session",
        session_config=SessionConfig(),
    )
    loop._handle_request(request)
    group = loop._groups["session"]
    segment = EngineSegment("session", 0)
    segment.pending_token_ids = [7]
    group.segments[0] = segment

    assert group.extension_continuity is None
    assert loop._try_prefill_one() is True
    assert segment.state == "active"
    assert segment.slot is not None
    assert executor.prefill_calls == 1
    assert pool.free_count == 0


def test_eos_does_not_append_c2w_frame_and_finalizes_before_release():
    pool = _FakePool()
    events = []
    pool.events = events
    slot = _FakeSlot(0, "session:0")
    slot.c2w_kv = torch.ones(1, 1, 1, 2, 1)
    slot.c2w_conv_states = [torch.ones(1, 1, 1)]
    slot.c2w_transconv_states = [torch.ones(1, 1, 1)]
    pool.register_live_slot(slot)

    policy = _FinalizePolicy(slot, events)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(pool=pool, extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    segment = EngineSegment("session", 0)
    segment.state = "active"
    segment.slot = slot
    group.segments[0] = segment
    loop._seg_by_slot[slot.slot_id] = segment

    output = StepOutput(
        slots=[slot],
        eos_flags=[True],
        audio_chunks=[None],
        tokens=[None],
        batch_c2w_kv=torch.zeros(1, 1, 1, 1, 1),
    )
    loop._process_step_output_inner(output)

    assert len(policy.states) == 1
    assert policy.states[0].c2w_kv.shape[3] == 2
    assert events == ["finalize", "release"]
    assert slot.is_free is True
    assert segment.slot is None


def test_engine_owns_cursor_state_when_external_c2w_policy_drops_it():
    pool = _FakePool()
    slot = _FakeSlot(0, "session:0")
    slot.c2w_kv = torch.ones(1, 1, 1, 2, 1)
    slot.c2w_conv_states = [torch.ones(1, 1, 1)]
    slot.c2w_transconv_states = [torch.ones(1, 1, 1)]
    slot.cursor_mu = torch.ones(1)
    slot.cursor_frames_since_advance = torch.ones(1)
    slot.cursor_delta_history = torch.ones(1, 8)
    slot.cursor_conv_history = torch.ones(1, 6, 4)
    slot.cursor_last_trunk_input = torch.ones(1, 4)
    slot.cursor_seen_frames = torch.ones(1, dtype=torch.int64)
    pool.register_live_slot(slot)

    policy = _CursorCapturePolicy()
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(pool=pool, extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    predecessor = EngineSegment("session", 0)
    predecessor.state = "active"
    predecessor.slot = slot
    group.segments[0] = predecessor
    loop._seg_by_slot[slot.slot_id] = predecessor

    loop._handle_segment_eos(group, predecessor)

    assert policy.captured_state is not None
    assert policy.captured_state.cursor_state is not None
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments[1] = successor
    context = loop._extension_context_for(group, successor)

    assert context is not None
    assert context.cursor_state is policy.captured_state.cursor_state
    assert context.c2w_frame_idx == 3


def test_context_is_not_available_before_predecessor_is_done():
    policy = _SuccessorContextPolicy(ready=True)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(extensions=extensions)
    group = _make_group("session", extensions)
    predecessor = EngineSegment("session", 0)
    predecessor.state = "active"
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments = {0: predecessor, 1: successor}

    assert loop._extension_context_for(group, successor) is None
    assert policy.context_calls == []


@pytest.mark.parametrize("carried", [False, True])
def test_terminal_successor_releases_predecessor_cursor_payload(carried):
    policy = _CursorCapturePolicy(carried=carried)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, pool = _make_loop(extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    predecessor = EngineSegment("session", 0)
    predecessor.state = "done"
    group.segments[0] = predecessor
    group.extension_cursor_states[0] = _cursor_state()
    # A stale payload from the same segment must not survive a final capture
    # that contains no cursor state, even when the C2W-only policy accepts it.
    group.extension_cursor_states[1] = _cursor_state()
    successor = EngineSegment("session", 1)
    successor.state = "active"
    successor.slot = pool.allocate("session:1")
    successor.slot.c2w_kv = torch.ones(1, 1, 1, 2, 1)
    successor.slot.c2w_conv_states = [torch.ones(1, 1, 1)]
    successor.slot.c2w_transconv_states = [torch.ones(1, 1, 1)]
    group.segments[1] = successor
    loop._seg_by_slot[successor.slot.slot_id] = successor

    loop._handle_segment_eos(group, successor)

    assert policy.captured_state is not None
    assert policy.captured_state.cursor_state is None
    assert group.extension_cursor_states == {}


def test_disabling_policy_releases_cursor_payloads_without_touching_other_sessions():
    policy = _SuccessorContextPolicy(ready=True)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(extensions=extensions)
    group = _make_group("session", extensions)
    other = _make_group("other", extensions)
    group.extension_cursor_states[0] = _cursor_state()
    retained = _cursor_state()
    other.extension_cursor_states[0] = retained

    loop._disable_extension(group, "restore_failed")

    assert group.extension_cursor_states == {}
    assert other.extension_cursor_states[0] is retained
    assert not other.extension_continuity_disabled


def test_admission_callback_failure_preserves_default_path():
    policy = _RaisingAdmissionPolicy(ready=True)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, pool = _make_loop(extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    segment = EngineSegment("session", 0)
    segment.pending_token_ids = [9]
    group.segments[0] = segment

    assert loop._try_prefill_one() is True
    assert segment.state == "active"
    assert executor.prefill_calls == 1
    assert pool.free_count == 0
    assert policy.invalidations == ["callback_failed:ready_to_admit"]


def test_successor_with_acoustic_continuity_downgrades_unmigrated_cursor_to_ema():
    policy = _SuccessorContextPolicy(ready=True)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, pool = _make_loop(extensions=extensions)
    executor.native_cursor_enabled = True
    executor.cursor_plan_calls = []
    executor.set_cursor_text_plan = lambda *args, **kwargs: executor.cursor_plan_calls.append(
        (args, kwargs)
    )
    executor.apply_c2w_warm_state = lambda *_args: True
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    _add_done_predecessor(group, 1)
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    successor.cursor_label_plan = SimpleNamespace(
        revision=1, label_ids=(7,), label_count=1, active=True
    )
    group.segments[1] = successor

    assert loop._try_prefill_one() is True
    assert successor.cursor_progress_disabled is True
    assert executor.cursor_plan_calls == []
    assert successor.state == "active"
    assert pool.free_count == 0


def test_successor_with_cursor_context_restores_recurrent_state_and_keeps_native_progress():
    cursor_state = CursorContinuationState(
        cursor_mu=torch.ones(1),
        cursor_frames_since_advance=torch.ones(1),
        cursor_delta_history=torch.ones(1, 8),
        cursor_conv_history=torch.ones(1, 6, 4),
        cursor_last_trunk_input=torch.ones(1, 4),
        cursor_seen_frames=torch.ones(1, dtype=torch.int64),
    )
    policy = _CursorContextPolicy(cursor_state)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, pool = _make_loop(extensions=extensions)
    executor.native_cursor_enabled = True
    restored = []
    executor.apply_c2w_warm_state = lambda *_args: True
    executor.restore_cursor_state = lambda slot, state: restored.append((slot, state))
    reanchors = []
    executor.set_cursor_reanchor = lambda slot, mu: reanchors.append((slot, mu))
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    _add_done_predecessor(group, 1)
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    successor.cursor_label_plan = SimpleNamespace(
        revision=1, label_ids=(7,), label_count=1, active=True
    )
    group.segments[1] = successor

    assert loop._try_prefill_one() is True
    assert successor.cursor_progress_disabled is False
    assert restored == [(successor.slot, cursor_state)]
    assert reanchors == [(successor.slot, 0.0)]
    assert successor.state == "active"
    assert pool.free_count == 0


def test_successor_cursor_state_without_streaming_tn_plan_stays_on_ema():
    cursor_state = CursorContinuationState(
        cursor_mu=torch.ones(1),
        cursor_frames_since_advance=torch.ones(1),
        cursor_delta_history=torch.ones(1, 8),
        cursor_conv_history=torch.ones(1, 6, 4),
        cursor_last_trunk_input=torch.ones(1, 4),
        cursor_seen_frames=torch.ones(1, dtype=torch.int64),
    )
    policy = _CursorContextPolicy(cursor_state)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, pool = _make_loop(extensions=extensions)
    executor.native_cursor_enabled = True
    executor.cursor_plan_calls = []
    executor.set_cursor_text_plan = lambda *args, **kwargs: executor.cursor_plan_calls.append(
        (args, kwargs)
    )
    executor.apply_c2w_warm_state = lambda *_args: True
    executor.restore_cursor_state = lambda *_args: None
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    _add_done_predecessor(group, 1)
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments[1] = successor

    assert loop._try_prefill_one() is True
    assert successor.cursor_progress_disabled is True
    assert executor.cursor_plan_calls == []
    assert successor.state == "active"
    assert pool.free_count == 0


def test_cursor_preflight_failure_does_not_install_partial_c2w_state():
    cursor_state = _cursor_state()
    policy = _CursorContextPolicy(cursor_state)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, _pool = _make_loop(extensions=extensions)
    group = _make_group("session", extensions)
    segment = EngineSegment("session", 1)
    segment.slot = _FakeSlot(0, "session:1")
    context = SimpleNamespace(
        c2w_kv=torch.ones(1, 1, 1, 1, 1),
        c2w_conv=[torch.ones(1, 1, 1)],
        c2w_transconv=[torch.ones(1, 1, 1)],
        c2w_frame_idx=3,
        cursor_state=cursor_state,
    )
    calls = []

    def reject_cursor(_slot, _state):
        raise ValueError("cursor ABI mismatch")

    executor.validate_cursor_state = reject_cursor
    executor.apply_c2w_warm_state = lambda *_args: calls.append("c2w") or True

    assert loop._extension_restore_context(group, segment, context) is False
    assert calls == []
    assert segment.cursor_progress_disabled is True
    assert group.extension_continuity_disabled is True
    assert policy.invalidations == ["cursor_state_validation_failed"]


def test_cursor_restore_unavailable_does_not_install_c2w_state():
    cursor_state = _cursor_state()
    policy = _CursorContextPolicy(cursor_state)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, _pool = _make_loop(extensions=extensions)
    group = _make_group("session", extensions)
    segment = EngineSegment("session", 1)
    segment.slot = _FakeSlot(0, "session:1")
    context = SimpleNamespace(
        c2w_kv=torch.ones(1, 1, 1, 1, 1),
        c2w_conv=[torch.ones(1, 1, 1)],
        c2w_transconv=[torch.ones(1, 1, 1)],
        c2w_frame_idx=3,
        cursor_state=cursor_state,
    )
    calls = []
    executor.apply_c2w_warm_state = lambda *_args: calls.append("c2w") or True

    assert loop._extension_restore_context(group, segment, context) is False
    assert calls == []
    assert segment.cursor_progress_disabled is True
    assert group.extension_continuity_disabled_reason == "cursor_state_restore_unavailable"
    assert policy.invalidations == ["cursor_state_restore_unavailable"]


def test_successor_restores_after_prefill_before_c2w_adoption():
    policy = _SuccessorContextPolicy(ready=True)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, pool = _make_loop(extensions=extensions)
    events = []

    original_prefill = executor.prefill

    def prefill(slot, embeds):
        events.append("prefill")
        return original_prefill(slot, embeds)

    def restore(*_args):
        events.append("restore")
        return True

    def adopt(slot):
        events.append("adopt")

    executor.prefill = prefill
    executor.apply_c2w_warm_state = restore
    executor.adopt_c2w_states = adopt

    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    _add_done_predecessor(group, 1)
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments[1] = successor

    assert loop._try_prefill_one() is True
    assert events == ["prefill", "restore", "adopt"]
    assert successor.state == "active"
    assert successor.cursor_progress_disabled is True
    assert successor.slot is not None
    assert pool.free_count == 0


def test_successor_gate_reopens_after_predecessor_eos_releases_slot():
    pool = _FakePool(max_slots=2)
    events = []
    pool.events = events
    predecessor_slot = pool.allocate("session:0")
    policy = _GateFinalizePolicy(predecessor_slot, events)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(pool=pool, extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group

    predecessor = EngineSegment("session", 0)
    predecessor.state = "active"
    predecessor.slot = predecessor_slot
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments = {0: predecessor, 1: successor}
    loop._seg_by_slot[predecessor_slot.slot_id] = predecessor

    assert loop._try_prefill_one() is False
    assert successor.slot is None
    assert predecessor.slot is predecessor_slot
    assert pool.free_count == 1

    loop._handle_segment_eos(group, predecessor)

    assert predecessor.state == "done"
    assert predecessor.slot is None
    assert predecessor_slot.is_free is True
    assert events == ["finalize", "release"]
    assert policy.finalized == [(0, None)]

    policy.ready = True
    assert loop._try_prefill_one() is True
    assert successor.state == "active"
    assert successor.slot is not None
    assert pool.free_count == 1
    assert policy.ready_calls == [1, 1]


def test_serial_context_restore_failure_falls_back_at_hard_boundary():
    pool = _FakePool(max_slots=2)
    policy = _SuccessorContextPolicy(ready=True)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, _pool = _make_loop(pool=pool, extensions=extensions)
    restore_calls = []

    def fail_restore(*args):
        restore_calls.append(args)
        return False

    executor.apply_c2w_warm_state = fail_restore
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group
    _add_done_predecessor(group, 1)
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments[1] = successor

    assert loop._try_prefill_one() is True
    assert successor.state == "active"
    assert successor.cursor_progress_disabled is True
    assert successor.slot.extension_state == {}
    assert len(restore_calls) == 1
    assert policy.context_calls == [1]
    assert policy.invalidations == ["c2w_restore_failed"]
    assert group.extension_continuity_disabled is True
    assert group.extension_continuity_disabled_reason == "c2w_restore_failed"

    # Once the serial handoff fails, a later segment must start cold at the
    # hard boundary instead of reusing the failed continuation context.
    next_successor = EngineSegment("session", 2)
    next_successor.pending_token_ids = [202]
    group.segments[2] = next_successor
    assert loop._try_prefill_one() is True
    assert next_successor.state == "active"
    assert next_successor.cursor_progress_disabled is False
    assert next_successor.slot.extension_state == {}
    assert policy.context_calls == [1]


@pytest.mark.parametrize("removal", ["cancel", "remove"])
def test_cancel_or_session_removal_clears_continuity_state(removal):
    pool = _FakePool()
    policy = _StatefulContinuityPolicy()
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(pool=pool, extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group

    slot = pool.allocate("session:0")
    slot.extension_state = {
        "continuity_bridge": {"old": True},
        "continuity_last_query": torch.ones(1, 1, 1),
        "continuity_bridge_apply_count": 3,
    }
    segment = EngineSegment("session", 0)
    segment.state = "active"
    segment.slot = slot
    segment.cursor_plan_revision = 4
    segment.cursor_progress_disabled = True
    group.segments[0] = segment
    group.extension_cursor_states[0] = _cursor_state()
    loop._seg_by_slot[slot.slot_id] = segment

    if removal == "cancel":
        loop._handle_request(
            EngineRequest(
                type=RequestType.CANCEL_SESSION,
                session_id="session",
            )
        )
    else:
        loop._remove_session("session")

    assert "session" not in loop._groups
    assert policy.invalidations == ["session_removed"]
    assert policy.bridge_state is None
    assert policy.capture_state is None
    assert slot.is_free is True
    assert slot.extension_state == {}
    assert segment.slot is None
    assert segment.cursor_plan_revision == -1
    assert segment.cursor_progress_disabled is False
    assert loop._seg_by_slot == {}
    assert group.extension_cursor_states == {}


def test_retry_reset_drops_old_bridge_and_capture_state():
    pool = _FakePool(max_slots=2)
    policy = _RetryContinuityPolicy()
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(pool=pool, extensions=extensions)
    group = _make_group("session", extensions)
    loop._groups[group.session_id] = group

    predecessor = EngineSegment("session", 0)
    predecessor.state = "active"
    group.segments[0] = predecessor
    predecessor_cursor = _cursor_state()
    group.extension_cursor_states[0] = predecessor_cursor

    segment = EngineSegment("session", 1)
    segment.state = "active"
    segment.slot = pool.allocate("session:1")
    segment.pending_token_ids = [101]
    segment.slot.extension_state = {
        "continuity_bridge": {"old": True},
        "continuity_last_query": torch.ones(1, 1, 1),
        "continuity_bridge_apply_count": 8,
    }
    segment.cursor_plan_revision = 7
    segment.cursor_progress_disabled = True
    group.segments[1] = segment
    loop._seg_by_slot[segment.slot.slot_id] = segment
    old_slot = segment.slot

    assert loop._try_segment_retry(group, segment, reason="loop") is True

    assert policy.reset_calls == [1]
    assert policy.reset_snapshots == [
        ({"bridge": "old"}, {"capture": "old"})
    ]
    assert policy.bridge_state is None
    assert policy.capture_state is None
    assert old_slot.is_free is True
    assert old_slot.extension_state == {}
    assert segment.slot is None
    assert segment.state == "pending_prefill"
    assert segment.retry_idx == 1
    assert segment.cursor_plan_revision == -1
    assert segment.cursor_progress_disabled is False
    assert group.extension_cursor_states[0] is predecessor_cursor

    assert loop._try_prefill_one() is True
    assert segment.state == "active"
    assert segment.slot is not old_slot
    assert segment.slot.extension_state == {}


def test_successor_retry_drops_only_its_cursor_state_and_preserves_session_owners():
    pool = _FakePool(max_slots=2)
    policy = _RetryContinuityPolicy()
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(pool=pool, extensions=extensions)
    group = _make_group("session", extensions)
    other = _make_group(
        "other",
        EngineExtensions(continuity_factory=lambda *_: _AdmissionPolicy(ready=True)),
    )
    loop._groups[group.session_id] = group
    loop._groups[other.session_id] = other

    predecessor_cursor = _cursor_state()
    stale_successor_cursor = _cursor_state()
    other_session_cursor = _cursor_state()
    group.extension_cursor_states[0] = predecessor_cursor
    group.extension_cursor_states[1] = stale_successor_cursor
    other.extension_cursor_states[0] = other_session_cursor

    predecessor = EngineSegment("session", 0)
    predecessor.state = "active"
    group.segments[0] = predecessor
    successor = EngineSegment("session", 1)
    successor.state = "active"
    successor.slot = pool.allocate("session:1")
    successor.pending_token_ids = [101]
    group.segments[1] = successor
    loop._seg_by_slot[successor.slot.slot_id] = successor

    assert loop._try_segment_retry(group, successor, reason="loop") is True

    assert group.extension_cursor_states.keys() == {0}
    assert group.extension_cursor_states[0] is predecessor_cursor
    assert other.extension_cursor_states[0] is other_session_cursor


def test_cache_hit_batch_respects_successor_ready_gate():
    pool = _FakePool()
    policy = _AdmissionPolicy(ready=False)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, executor, _pool = _make_loop(pool=pool, extensions=extensions)
    loop._prefill_builder = _BatchPrefillBuilder()
    group = _make_group("session", extensions)
    group.request.session_config.task_type = "custom_voice"
    loop._groups[group.session_id] = group
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments[1] = successor
    loop._prefix_cache.put(
        "cache-key",
        torch.zeros(1, 1, 1, 1, 1),
        prefix_len=1,
    )

    loop._try_prefill_pending()

    assert successor.state == "pending_prefill"
    assert successor.slot is None
    assert loop._prefill_builder.batch_calls == 0
    assert executor.prefill_calls == 0
    assert pool.free_count == 1

    policy.ready = True
    loop._try_prefill_pending()

    assert successor.state == "active"
    assert successor.slot is not None
    assert successor.cache_hit is True
    assert loop._prefill_builder.batch_calls == 1
    assert executor.prefill_calls == 0


def test_cache_hit_batch_excludes_successor_with_continuation_context():
    """A continuation successor must use serial restore, never prefix cache."""

    pool = _FakePool(max_slots=2)
    policy = _SuccessorContextPolicy(ready=True)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop, _executor, _pool = _make_loop(pool=pool, extensions=extensions)
    loop._prefill_builder = _BatchPrefillBuilder()
    group = _make_group("session", extensions)
    group.request.session_config.task_type = "custom_voice"
    loop._groups[group.session_id] = group
    _add_done_predecessor(group, 1)
    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    group.segments[1] = successor
    loop._prefix_cache.put(
        "cache-key",
        torch.zeros(1, 1, 1, 1, 1),
        prefix_len=1,
    )

    assert loop._admit_cache_hit_batch() == 0
    assert successor.state == "pending_prefill"
    assert successor.slot is None
    assert loop._prefill_builder.batch_calls == 0
    assert policy.context_calls == [1]


def test_wait_text_segment_is_not_reported_as_engine_work():
    slot = SimpleNamespace(
        last_codec_sum=torch.zeros(1),
        next_embed=None,
        trailing=[],
        text_idx=0,
    )
    segment = SimpleNamespace(
        state="active",
        input_complete=False,
        slot=slot,
    )

    assert EngineLoop._is_waiting_for_text(segment)
    assert not EngineLoop._is_waiting_for_text(
        SimpleNamespace(state="active", input_complete=True, slot=slot)
    )
