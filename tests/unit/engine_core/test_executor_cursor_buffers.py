from __future__ import annotations

import pytest
import torch

from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import ModelConfig, SlotKVState
from engine.core.extensions import CursorContinuationState


def _executor(*, vocab_size: int = 0) -> Executor:
    executor = Executor.__new__(Executor)
    executor._cursor_enabled = True
    executor._cursor_max_labels = 8
    executor._cursor_d = 4
    executor._cursor_history = 6
    executor._cursor_vocab_size = vocab_size
    executor._device = torch.device("cpu")
    executor._config = ModelConfig(dtype=torch.float32)
    return executor


def test_cursor_plan_reuses_buffers_clears_tail_and_preserves_neural_state():
    executor = _executor(vocab_size=16)
    slot = SlotKVState(slot_id=0, is_free=False)
    executor.set_cursor_text_plan(slot, torch.tensor([4, 5, 6]), text_start_frame=2)
    label_ptr = slot.cursor_label_ids.data_ptr()
    slot.cursor_mu.fill_(3.5)

    executor.set_cursor_text_plan(slot, torch.tensor([7, 8]), text_start_frame=3)

    assert slot.cursor_label_ids.data_ptr() == label_ptr
    assert slot.cursor_label_ids.tolist() == [[7, 8, 0, 0, 0, 0, 0, 0]]
    assert float(slot.cursor_mu.item()) == 3.5
    assert int(slot.cursor_text_start_frame.item()) == 3


def test_cursor_plan_accepts_final_inclusive_vocabulary_id():
    executor = _executor(vocab_size=503)
    slot = SlotKVState(slot_id=0, is_free=False)

    executor.set_cursor_text_plan(slot, torch.tensor([503]))

    assert slot.cursor_label_ids.tolist() == [[503, 0, 0, 0, 0, 0, 0, 0]]


@pytest.mark.parametrize("labels", [torch.tensor([-1]), torch.tensor([17])])
def test_cursor_plan_rejects_invalid_label_ids_without_mutating_slot(labels):
    executor = _executor(vocab_size=16)
    slot = SlotKVState(slot_id=0, is_free=False)
    executor.set_cursor_text_plan(slot, torch.tensor([4, 5]))
    before = slot.cursor_label_ids.clone()
    before_count = slot.cursor_label_count.clone()

    with pytest.raises(ValueError, match="label"):
        executor.set_cursor_text_plan(slot, labels)

    torch.testing.assert_close(slot.cursor_label_ids, before)
    torch.testing.assert_close(slot.cursor_label_count, before_count)


def test_cursor_plan_buffers_are_isolated_per_slot():
    executor = _executor(vocab_size=16)
    first = SlotKVState(slot_id=0, is_free=False)
    second = SlotKVState(slot_id=1, is_free=False)

    executor.set_cursor_text_plan(first, torch.tensor([1, 2]))
    executor.set_cursor_text_plan(second, torch.tensor([3]))
    first.cursor_label_ids[:, 0] = 9

    assert second.cursor_label_ids.tolist() == [[3, 0, 0, 0, 0, 0, 0, 0]]


def test_cursor_reanchor_sets_one_shot_override_without_resetting_neural_state():
    executor = _executor()
    slot = SlotKVState(slot_id=0, is_free=False)
    executor.set_cursor_text_plan(slot, torch.tensor([4, 5]))
    slot.cursor_mu.fill_(1.5)
    executor.set_cursor_reanchor(slot, 2.0)

    assert float(slot.cursor_mu.item()) == 1.5
    assert int(slot.cursor_override_valid.item()) == 1
    assert float(slot.cursor_override_mu.item()) == 2.0

    executor.update_cursor_state(slot, {})
    assert int(slot.cursor_override_valid.item()) == 0


def test_restore_cursor_state_restores_neural_fields_without_overwriting_labels():
    executor = _executor()
    executor._cursor_state_handoff_enabled = True
    slot = SlotKVState(slot_id=0, is_free=False)
    executor.set_cursor_text_plan(slot, torch.tensor([4, 5]), text_start_frame=3)
    state = CursorContinuationState(
        cursor_mu=torch.full((1,), 2.0),
        cursor_frames_since_advance=torch.full((1,), 3.0),
        cursor_delta_history=torch.full((1, 8), 4.0),
        cursor_conv_history=torch.full((1, 6, 4), 5.0),
        cursor_last_trunk_input=torch.full((1, 4), 6.0),
        cursor_seen_frames=torch.full((1,), 7, dtype=torch.int64),
    )

    executor.restore_cursor_state(slot, state)

    assert torch.all(slot.cursor_mu == 2)
    assert torch.all(slot.cursor_conv_history == 5)
    assert int(slot.cursor_seen_frames.item()) == 7
    assert slot.cursor_label_ids.tolist() == [[4, 5, 0, 0, 0, 0, 0, 0]]
    assert int(slot.cursor_text_start_frame.item()) == 3


def test_validate_cursor_state_is_non_mutating_and_checks_the_fused_abi():
    executor = _executor()
    executor._cursor_state_handoff_enabled = True
    slot = SlotKVState(slot_id=0, is_free=False)
    executor.set_cursor_text_plan(slot, torch.tensor([4, 5]), text_start_frame=3)
    before = slot.cursor_label_ids.clone()
    state = CursorContinuationState(
        cursor_mu=torch.full((1,), 2.0),
        cursor_frames_since_advance=torch.full((1,), 3.0),
        cursor_delta_history=torch.full((1, 8), 4.0),
        cursor_conv_history=torch.full((1, 6, 4), 5.0),
        cursor_last_trunk_input=torch.full((1, 4), 6.0),
        cursor_seen_frames=torch.full((1,), 7, dtype=torch.int64),
    )

    executor.validate_cursor_state(slot, state)

    torch.testing.assert_close(slot.cursor_label_ids, before)
    assert float(slot.cursor_mu.item()) == 0.0

    invalid = CursorContinuationState(
        cursor_mu=torch.zeros(1),
        cursor_frames_since_advance=torch.zeros(1),
        cursor_delta_history=torch.zeros(1, 7),
        cursor_conv_history=torch.zeros(1, 6, 4),
        cursor_last_trunk_input=torch.zeros(1, 4),
        cursor_seen_frames=torch.zeros(1, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="ABI mismatch"):
        executor.validate_cursor_state(slot, invalid)


def test_restore_cursor_state_requires_declared_handoff_abi():
    executor = _executor()
    slot = SlotKVState(slot_id=0, is_free=False)
    state = CursorContinuationState(
        cursor_mu=torch.zeros(1),
        cursor_frames_since_advance=torch.zeros(1),
        cursor_delta_history=torch.zeros(1, 8),
        cursor_conv_history=torch.zeros(1, 6, 4),
        cursor_last_trunk_input=torch.zeros(1, 4),
        cursor_seen_frames=torch.zeros(1, dtype=torch.int64),
    )

    with pytest.raises(RuntimeError, match="state handoff is unavailable"):
        executor.restore_cursor_state(slot, state)


def test_c2w_arena_snapshot_round_trip_restores_parity_and_rows():
    executor = _executor()
    executor._c2w_conv_shapes = [(1, 2), (1, 1)]
    executor._c2w_transconv_shapes = []
    executor._c2w_arena_a = [torch.zeros(2, *shape[1:], dtype=executor._config.dtype) for shape in executor._c2w_conv_shapes]
    executor._c2w_arena_b = [torch.zeros(2, *shape[1:], dtype=executor._config.dtype) for shape in executor._c2w_conv_shapes]
    class Pool:
        _max_slots = 2
        def get(self, slot_id):
            return slot
    executor._kv_pool = Pool()
    slot = SlotKVState(slot_id=1, is_free=False, allocation_epoch=4,
                       c2w_arena_backed=True, c2w_write_in_a=True)
    slot.c2w_conv_states, slot.c2w_transconv_states = executor._arena_row_views(
        executor._c2w_arena_b, slot.slot_id
    )
    slot._c2w_conv_write, slot._c2w_transconv_write = executor._arena_row_views(
        executor._c2w_arena_a, slot.slot_id
    )
    for tensor in slot.c2w_conv_states + slot._c2w_conv_write:
        tensor.fill_(7)
    snapshot = executor.snapshot_c2w_arena(slot, expected_allocation_epoch=4)
    for tensor in executor._c2w_arena_a + executor._c2w_arena_b:
        tensor.zero_()
    executor.restore_c2w_arena(
        slot, snapshot, expected_allocation_epoch=4,
        expected_source_slot_id=1, expected_source_allocation_epoch=4,
    )
    assert slot.c2w_write_in_a is True
    assert all(torch.all(t == 7) for t in slot.c2w_conv_states + slot._c2w_conv_write)

    with pytest.raises(ValueError, match="owner mismatch"):
        executor.restore_c2w_arena(
            slot, snapshot, expected_allocation_epoch=5,
            expected_source_slot_id=1, expected_source_allocation_epoch=4,
        )


def test_c2w_arena_snapshot_rejects_misbound_slot_views():
    executor = _executor()
    executor._c2w_conv_shapes = [(1, 2)]
    executor._c2w_transconv_shapes = []
    executor._c2w_arena_a = [torch.zeros(2, 2, dtype=executor._config.dtype)]
    executor._c2w_arena_b = [torch.zeros(2, 2, dtype=executor._config.dtype)]
    slot = SlotKVState(slot_id=1, is_free=False, allocation_epoch=1,
                       c2w_arena_backed=True, c2w_write_in_a=False)
    slot.c2w_conv_states = [torch.zeros(1, 2, dtype=executor._config.dtype)]
    slot._c2w_conv_write = [executor._c2w_arena_b[0][1:2]]
    class Pool:
        _max_slots = 2
        def get(self, slot_id):
            return slot
    executor._kv_pool = Pool()
    with pytest.raises(ValueError, match="not bound"):
        executor.snapshot_c2w_arena(slot, expected_allocation_epoch=1)
