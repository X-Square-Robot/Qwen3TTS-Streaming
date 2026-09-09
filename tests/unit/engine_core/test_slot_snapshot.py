from dataclasses import fields

import pytest
import torch

from engine.backend.kv_cache_pool import SlotKVState
from engine.backend.slot_snapshot import SlotAuxiliarySnapshot, StandaloneSlotSnapshot
from engine.backend.speech_state import SpeechStateContractError


def _slot():
    slot = SlotKVState(
        slot_id=0, is_free=False, session_id="s:2", segment_idx=2,
        talker_kv=torch.ones(1, 2, 1, 3, 2), past_len=3,
        c2w_kv=torch.ones(1, 2, 1, 2, 2), frame_idx=7,
        c2w_conv_states=[torch.ones(1, 3)],
        c2w_transconv_states=[torch.ones(1, 2)],
        trailing=[torch.ones(1, 1, 4)], token_queue=[5, 6], text_idx=1,
        sampling_generator=torch.Generator().manual_seed(7),
    )
    slot.init_pingpong_buffers()
    slot.flip_c2w_buffers()
    slot.next_embed = torch.ones(1, 1, 4)
    slot.last_codec_sum = torch.ones(1, 1, 4)
    slot.token_counts = torch.ones(1, 8, dtype=torch.int64)
    for item in fields(slot):
        if item.name.startswith("cursor_"):
            setattr(slot, item.name, torch.ones(1, 2))
    return slot


def test_snapshot_isolates_every_tensor_list_and_rng():
    source = _slot()
    snapshot = StandaloneSlotSnapshot(source, max_tensor_bytes=100_000)
    expected_rng = torch.rand(8, generator=source.sampling_generator)
    for item in fields(source):
        value = getattr(source, item.name)
        if isinstance(value, torch.Tensor):
            value.zero_()
        elif isinstance(value, list):
            for element in value:
                if isinstance(element, torch.Tensor):
                    element.zero_()
            value.clear()
    restored = snapshot.restore(slot_id=4)
    assert restored.session_id == "s:2"
    assert restored.segment_idx == 2
    assert restored.slot_id == 4
    assert restored.frame_idx == 7
    assert restored.c2w_write_in_a
    assert restored.token_queue == [5, 6]
    assert restored.text_idx == 1
    torch.testing.assert_close(torch.rand(8, generator=restored.sampling_generator), expected_rng)
    for item in fields(restored):
        value = getattr(restored, item.name)
        if isinstance(value, torch.Tensor):
            assert torch.all(value == 1), item.name
        elif isinstance(value, list):
            for element in value:
                if isinstance(element, torch.Tensor):
                    assert torch.all(element == 1), item.name
    restored.cursor_mu.zero_()
    restored.trailing.clear()
    again = snapshot.restore(slot_id=5)
    assert torch.all(again.cursor_mu == 1)
    assert len(again.trailing) == 1
    torch.testing.assert_close(torch.rand(8, generator=again.sampling_generator), expected_rng)


@pytest.mark.parametrize("field", ["c2w_pooled", "c2w_arena_backed", "is_free"])
def test_snapshot_rejects_unsupported_storage(field):
    slot = _slot()
    setattr(slot, field, True)
    with pytest.raises(SpeechStateContractError):
        StandaloneSlotSnapshot(slot, max_tensor_bytes=100_000)


def test_snapshot_budget_and_missing_talker_history_fail_before_mutation():
    slot = _slot()
    with pytest.raises(SpeechStateContractError, match="budget"):
        StandaloneSlotSnapshot(slot, max_tensor_bytes=1)
    assert torch.all(slot.talker_kv == 1)
    slot.talker_kv = None
    with pytest.raises(SpeechStateContractError, match="history"):
        StandaloneSlotSnapshot(slot, max_tensor_bytes=100_000)


def test_snapshot_budget_counts_tensor_and_generator_storage():
    slot = _slot()
    snapshot = StandaloneSlotSnapshot(slot, max_tensor_bytes=100_000)
    assert snapshot.tensor_bytes > slot.sampling_generator.get_state().numel()
    StandaloneSlotSnapshot(slot, max_tensor_bytes=snapshot.tensor_bytes)
    with pytest.raises(SpeechStateContractError, match="budget"):
        StandaloneSlotSnapshot(slot, max_tensor_bytes=snapshot.tensor_bytes - 1)


def test_missing_c2w_storage_is_not_an_empty_history():
    slot = _slot()
    slot.c2w_len = 2
    slot.c2w_kv = None
    with pytest.raises(SpeechStateContractError, match="C2W KV"):
        StandaloneSlotSnapshot(slot, max_tensor_bytes=100_000)


def test_auxiliary_snapshot_restores_non_storage_state_only():
    source = _slot()
    source.c2w_pooled = True
    source.c2w_arena_backed = True
    source.allocation_epoch = 7
    snapshot = SlotAuxiliarySnapshot(source, max_tensor_bytes=100_000)
    target = SlotKVState(slot_id=4, is_free=False, allocation_epoch=12)
    snapshot.restore_into(target)
    assert target.session_id == "s:2"
    assert target.segment_idx == 2
    assert target.token_queue == [5, 6]
    assert target.c2w_pooled is False
    assert target.c2w_arena_backed is False
    assert target.allocation_epoch == 12
    target.token_queue.clear()
    assert source.token_queue == [5, 6]


def test_auxiliary_snapshot_rejects_free_target():
    source = _slot()
    snapshot = SlotAuxiliarySnapshot(source, max_tensor_bytes=100_000)
    with pytest.raises(SpeechStateContractError, match="free slot"):
        snapshot.restore_into(SlotKVState(slot_id=1))


def test_snapshot_preserves_allocation_epoch():
    slot = _slot()
    slot.allocation_epoch = 17
    restored = StandaloneSlotSnapshot(slot, max_tensor_bytes=100_000).restore(slot_id=4)
    assert restored.allocation_epoch == 17
