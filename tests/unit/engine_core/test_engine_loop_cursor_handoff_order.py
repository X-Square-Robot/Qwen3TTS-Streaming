"""Audit the successor cursor-plan/state handoff at the first decode step."""

import queue
from types import SimpleNamespace

import pytest
import torch

from engine.backend.engine_loop import EngineLoop, EngineSegment, EngineSessionGroup
from engine.core.extensions import CursorContinuationState, EngineExtensions
from engine.core.native_cursor import CursorLabelPlan, CursorOwnerSpan
from engine.core.types import EngineRequest, RequestType


class _FakeSlot:
    def __init__(self, slot_id: int, session_id: str):
        self.slot_id = slot_id
        self.allocation_epoch = 1
        self.session_id = session_id
        self.segment_idx = -1
        self.retry_idx = 0
        self.is_free = False
        self.prefill_source = ""
        self.past_len = 0
        self.frame_idx = 0
        self.next_embed = None
        self.trailing = []
        self.text_idx = 0
        self.extension_state = {}
        self.native_plan_installed = False
        self.c2w_restored = False
        self.cursor_restored = False


class _FakePool:
    max_seq_len = 64

    def __init__(self):
        self._free_count = 1

    @property
    def free_count(self) -> int:
        return self._free_count

    def allocate(self, session_id: str):
        if self._free_count == 0:
            return None
        self._free_count = 0
        return _FakeSlot(0, session_id)


class _FakeExecutor:
    def __init__(self, events: list[str], *, failure: str | None = None):
        self.events = events
        self.failure = failure
        self.kv_pool = _FakePool()
        self._device = torch.device("cpu")
        self._config = SimpleNamespace(dtype=torch.float32, hidden_size=4)
        self.native_cursor_enabled = True
        self.native_decode_slots = []
        self.invalid_native_handoffs = []

    def set_cursor_text_plan(self, slot, _label_ids, **kwargs):
        self.events.append("plan setter")
        slot.native_plan_installed = bool(kwargs.get("active", True))

    def prefill(self, slot, _embeds):
        self.events.append("prefill")
        slot.prefill_source = "full_prefill"
        slot.next_embed = torch.zeros(1, 1, 4)
        return None, False

    def apply_c2w_warm_state(self, slot, *_state):
        self.events.append("C2W restore")
        if self.failure == "c2w":
            return False
        slot.c2w_restored = True
        return True

    def restore_cursor_state(self, slot, _state):
        self.events.append("cursor restore")
        if self.failure == "cursor":
            raise RuntimeError("cursor restore failed")
        slot.cursor_restored = True

    def adopt_c2w_states(self, _slot):
        self.events.append("adopt")

    def launch_decode_step(self, slots):
        self.events.append("first decode")
        for slot in slots:
            if not slot.native_plan_installed:
                continue
            if slot.c2w_restored and slot.cursor_restored:
                self.native_decode_slots.append(slot)
            else:
                self.invalid_native_handoffs.append(slot)


class _ImmediateAsyncLoop:
    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


class _ContinuityPolicy:
    def __init__(self, cursor_state: CursorContinuationState):
        self.cursor_state = cursor_state
        self.invalidations = []

    def ready_to_admit(self, _segment_idx: int) -> bool:
        return True

    def context_for(self, _segment_idx: int, **_kwargs):
        return SimpleNamespace(
            c2w_kv=torch.zeros(1),
            c2w_conv=(torch.zeros(1),),
            c2w_transconv=(torch.zeros(1),),
            c2w_frame_idx=3,
            cursor_state=self.cursor_state,
        )

    def invalidate(self, reason: str):
        self.invalidations.append(reason)


def _cursor_state() -> CursorContinuationState:
    return CursorContinuationState(
        cursor_mu=torch.ones(1),
        cursor_frames_since_advance=torch.ones(1),
        cursor_delta_history=torch.ones(1, 8),
        cursor_conv_history=torch.ones(1, 6, 4),
        cursor_last_trunk_input=torch.ones(1, 4),
        cursor_seen_frames=torch.ones(1, dtype=torch.int64),
    )


def _plan() -> CursorLabelPlan:
    return CursorLabelPlan(
        label_ids=(7,),
        owner_spans=(CursorOwnerSpan(1, 0, 1, 0, 1, 0, 1),),
        revision=1,
    )


def _case(*, failure: str | None = None):
    events: list[str] = []
    executor = _FakeExecutor(events, failure=failure)
    cursor_state = _cursor_state()
    policy = _ContinuityPolicy(cursor_state)
    extensions = EngineExtensions(continuity_factory=lambda *_: policy)
    loop = EngineLoop(
        engine_inbox=queue.Queue(),
        async_loop=_ImmediateAsyncLoop(),
        executor=executor,
        extensions=extensions,
    )
    request = EngineRequest(type=RequestType.NEW_SESSION, session_id="session")
    group = EngineSessionGroup("session", request, extensions=extensions)
    loop._groups[group.session_id] = group

    predecessor = EngineSegment("session", 0)
    predecessor.state = "done"
    group.segments[0] = predecessor

    successor = EngineSegment("session", 1)
    successor.pending_token_ids = [101]
    successor.cursor_label_plan = _plan()
    group.segments[1] = successor
    return loop, executor, policy, successor, events


def _prefill_then_first_decode(loop, executor, successor):
    assert loop._try_prefill_one() is True
    active_slots = loop._get_active_slots_mlfq()
    assert active_slots == [successor.slot]
    executor.launch_decode_step(active_slots)


def test_successor_handoff_restores_cursor_before_first_decode():
    loop, executor, _policy, successor, events = _case()

    _prefill_then_first_decode(loop, executor, successor)

    assert events == [
        "plan setter",
        "prefill",
        "C2W restore",
        "cursor restore",
        "adopt",
        "first decode",
    ]
    assert executor.native_decode_slots == [successor.slot]
    assert executor.invalid_native_handoffs == []
    assert successor.cursor_progress_disabled is False


@pytest.mark.parametrize("failure", ["c2w", "cursor"])
def test_failed_successor_restore_falls_back_before_first_decode(failure):
    loop, executor, policy, successor, events = _case(failure=failure)

    _prefill_then_first_decode(loop, executor, successor)

    expected = ["plan setter", "prefill", "C2W restore"]
    if failure == "cursor":
        expected.append("cursor restore")
    expected.append("plan setter")
    expected.extend(["adopt", "first decode"])
    assert events == expected
    assert successor.cursor_progress_disabled is True
    assert executor.native_decode_slots == []
    assert executor.invalid_native_handoffs == []
    expected_reason = (
        "cursor_state_restore_failed" if failure == "cursor" else "c2w_restore_failed"
    )
    assert policy.invalidations == [expected_reason]
