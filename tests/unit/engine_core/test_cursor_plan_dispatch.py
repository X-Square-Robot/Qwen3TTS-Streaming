import asyncio
import queue

import torch

from engine.backend.engine_loop import EngineLoop, EngineSegment, EngineSessionGroup
from engine.backend.kv_cache_pool import KVCachePool, ModelConfig, SlotKVState
from engine.core.native_cursor import CursorLabelPlan, CursorOwnerSpan
from engine.core.types import EngineRequest, RequestType


def _plan(revision: int, label_id: int = 7) -> CursorLabelPlan:
    return CursorLabelPlan(
        label_ids=(label_id,),
        owner_spans=(CursorOwnerSpan(1, 0, 1, 0, 1, 0, 1),),
        revision=revision,
    )


class _Executor:
    def __init__(self, *, enabled: bool = False):
        config = ModelConfig(
            num_layers=1,
            kv_heads=1,
            head_dim=4,
            max_seq_len=16,
            n_c2w_layers=1,
            c2w_kv_heads=1,
            c2w_head_dim=4,
            c2w_sliding_window=8,
        )
        self._config = config
        self._device = torch.device("cpu")
        self.kv_pool = KVCachePool(
            max_slots=2,
            config=config,
            device=self._device,
            preallocate=False,
        )
        self.native_cursor_enabled = enabled
        self.calls = []
        self.reanchors = []

    def set_cursor_text_plan(self, slot, label_ids, **kwargs):
        self.calls.append((slot, tuple(label_ids), kwargs))

    def set_cursor_reanchor(self, slot, mu):
        self.reanchors.append((slot, float(mu)))


def _loop(executor):
    loop = asyncio.new_event_loop()
    return EngineLoop(queue.Queue(), loop, executor), loop


def test_start_request_carries_plan_until_slot_admission():
    executor = _Executor()
    engine, event_loop = _loop(executor)
    try:
        engine._handle_request(EngineRequest(type=RequestType.NEW_SESSION, session_id="s"))
        plan = _plan(3)
        engine._handle_request(
            EngineRequest(
                type=RequestType.START_TOKENS,
                session_id="s",
                segment_idx=0,
                cursor_label_plan=plan,
                token_ids=[1],
            )
        )
        assert engine._groups["s"].segments[0].cursor_label_plan is plan
    finally:
        event_loop.close()


def test_update_plan_applies_to_active_slot_and_ignores_stale_revision():
    executor = _Executor(enabled=True)
    engine, event_loop = _loop(executor)
    try:
        group = EngineSessionGroup("s", EngineRequest(type=RequestType.NEW_SESSION, session_id="s"))
        seg = EngineSegment("s", 0)
        seg.state = "active"
        seg.slot = SlotKVState(slot_id=0)
        group.segments[0] = seg
        engine._groups["s"] = group

        engine._handle_request(
            EngineRequest(
                type=RequestType.UPDATE_CURSOR_PLAN,
                session_id="s",
                segment_idx=0,
                cursor_label_plan=_plan(2, 8),
            )
        )
        engine._handle_request(
            EngineRequest(
                type=RequestType.UPDATE_CURSOR_PLAN,
                session_id="s",
                segment_idx=0,
                cursor_label_plan=_plan(1, 9),
            )
        )

        assert len(executor.calls) == 1
        assert executor.calls[0][1] == (8,)
        assert seg.cursor_plan_revision == 2
        assert seg.cursor_label_plan.label_ids == (8,)
    finally:
        event_loop.close()


def test_plan_revision_reanchors_live_cursor_by_owner_id():
    executor = _Executor(enabled=True)
    engine, event_loop = _loop(executor)
    try:
        group = EngineSessionGroup("s", EngineRequest(type=RequestType.NEW_SESSION, session_id="s"))
        seg = EngineSegment("s", 0)
        seg.state = "active"
        seg.slot = SlotKVState(slot_id=0)
        seg.slot.cursor_mu = torch.tensor([2.0])
        seg.cursor_label_plan = CursorLabelPlan(
            label_ids=(7, 8),
            owner_spans=(CursorOwnerSpan(1, 0, 2, 0, 2, 0, 2),),
            revision=1,
        )
        seg.cursor_plan_revision = 1
        group.segments[0] = seg
        engine._groups["s"] = group

        current = CursorLabelPlan(
            label_ids=(7, 8, 9),
            owner_spans=(CursorOwnerSpan(1, 0, 3, 0, 3, 0, 2),),
            revision=2,
        )
        engine._handle_request(
            EngineRequest(
                type=RequestType.UPDATE_CURSOR_PLAN,
                session_id="s",
                segment_idx=0,
                cursor_label_plan=current,
            )
        )

        assert executor.reanchors == [(seg.slot, 3.0)]
        assert seg.cursor_plan_revision == 2
    finally:
        event_loop.close()


def test_disabled_cursor_consumes_revision_without_touching_executor():
    executor = _Executor(enabled=False)
    engine, event_loop = _loop(executor)
    try:
        seg = EngineSegment("s", 0)
        seg.slot = SlotKVState(slot_id=0)
        seg.cursor_label_plan = _plan(4)
        engine._apply_cursor_plan(seg)
        assert seg.cursor_plan_revision == 4
        assert executor.calls == []
    finally:
        event_loop.close()


def test_released_slot_invalidates_applied_revision_for_retry():
    executor = _Executor(enabled=True)
    engine, event_loop = _loop(executor)
    try:
        seg = EngineSegment("s", 0)
        seg.cursor_label_plan = _plan(4)
        first = executor.kv_pool.allocate("s:0")
        assert first is not None
        seg.slot = first
        engine._seg_by_slot[first.slot_id] = seg
        engine._apply_cursor_plan(seg)
        assert seg.cursor_plan_revision == 4

        engine._release_segment_slot(seg)
        assert seg.cursor_plan_revision == -1
        assert seg.slot is None
        second = executor.kv_pool.allocate("s:0")
        assert second is not None
        seg.slot = second
        engine._apply_cursor_plan(seg)

        assert len(executor.calls) == 2
        assert executor.calls[-1][1] == (7,)
        assert seg.cursor_plan_revision == 4
    finally:
        event_loop.close()
