from types import SimpleNamespace

import torch

from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import KVCachePool, ModelConfig


def _executor():
    executor = Executor.__new__(Executor)
    executor._c2w_conv_input_names = ["conv_0", "conv_1"]
    executor._c2w_transconv_input_names = ["transconv_0"]
    executor._config = SimpleNamespace(
        c2w_sliding_window=8,
        dtype=torch.float32,
    )
    executor._device = torch.device("cpu")
    return executor


def test_warm_state_prepares_all_tensors_before_slot_mutation():
    executor = _executor()
    old_kv = torch.full((1, 1, 1, 2, 1), 7.0)
    old_conv = [torch.full((1, 1, 1), 8.0), torch.full((1, 1, 1), 9.0)]
    old_transconv = [torch.full((1, 1, 1), 10.0)]
    slot = SimpleNamespace(
        c2w_kv=old_kv,
        c2w_conv_states=old_conv,
        c2w_transconv_states=old_transconv,
        frame_idx=13,
    )

    class BrokenTensor:
        def to(self, **_kwargs):
            raise RuntimeError("bad successor state")

    result = executor.apply_c2w_warm_state(
        slot,
        torch.ones(1, 1, 1, 2, 1),
        [torch.ones(1, 1, 1), BrokenTensor()],
        [torch.ones(1, 1, 1)],
        2,
    )

    assert result is False
    assert slot.c2w_kv is old_kv
    assert slot.c2w_conv_states is old_conv
    assert slot.c2w_transconv_states is old_transconv
    assert slot.frame_idx == 13


def test_warm_state_commits_after_all_tensors_are_prepared():
    executor = _executor()
    slot = SimpleNamespace(
        c2w_kv=None,
        c2w_conv_states=None,
        c2w_transconv_states=None,
        frame_idx=0,
    )

    result = executor.apply_c2w_warm_state(
        slot,
        torch.ones(1, 1, 1, 10, 1),
        [torch.ones(1, 1, 1), torch.ones(1, 1, 1)],
        [torch.ones(1, 1, 1)],
        4,
    )

    assert result is True
    assert slot.c2w_kv.shape[3] == 7
    assert len(slot.c2w_conv_states) == 2
    assert len(slot.c2w_transconv_states) == 1
    assert slot.frame_idx == 4


def test_warm_state_replaces_pooled_c2w_row_instead_of_hidden_slot_field():
    executor = _executor()
    calls = []

    class Pool:
        def restore_pooled_c2w_kv(self, slot_id, kv, *, expected_allocation_epoch):
            calls.append((slot_id, kv.clone(), expected_allocation_epoch))
            return 3

    executor._kv_pool = Pool()
    slot = SimpleNamespace(
        slot_id=5,
        allocation_epoch=9,
        c2w_pooled=True,
        c2w_kv=None,
        c2w_conv_states=None,
        c2w_transconv_states=None,
        frame_idx=0,
    )

    result = executor.apply_c2w_warm_state(
        slot,
        torch.ones(1, 1, 1, 3, 1),
        [torch.ones(1, 1, 1), torch.ones(1, 1, 1)],
        [torch.ones(1, 1, 1)],
        4,
    )

    assert result is True
    assert len(calls) == 1
    assert calls[0][0] == 5
    assert calls[0][2] == 9
    assert calls[0][1].shape[3] == 3
    assert slot.c2w_kv is None
    assert slot.c2w_pooled is True
    assert slot.frame_idx == 4


def test_pooled_warm_state_restore_failure_keeps_slot_state_unchanged():
    executor = _executor()

    class Pool:
        def restore_pooled_c2w_kv(self, *_args, **_kwargs):
            raise ValueError("bad pool row")

    executor._kv_pool = Pool()
    old_conv = [torch.full((1, 1, 1), 8.0), torch.full((1, 1, 1), 9.0)]
    old_transconv = [torch.full((1, 1, 1), 10.0)]
    slot = SimpleNamespace(
        slot_id=5,
        allocation_epoch=9,
        c2w_pooled=True,
        c2w_kv=None,
        c2w_conv_states=old_conv,
        c2w_transconv_states=old_transconv,
        frame_idx=13,
    )

    result = executor.apply_c2w_warm_state(
        slot,
        torch.ones(1, 1, 1, 3, 1),
        [torch.ones(1, 1, 1), torch.ones(1, 1, 1)],
        [torch.ones(1, 1, 1)],
        4,
    )

    assert result is False
    assert slot.c2w_kv is None
    assert slot.c2w_conv_states is old_conv
    assert slot.c2w_transconv_states is old_transconv
    assert slot.frame_idx == 13


def test_pooled_warm_state_updates_the_real_decode_owner_row():
    config = ModelConfig(
        num_layers=1,
        kv_heads=1,
        head_dim=2,
        max_seq_len=8,
        n_c2w_layers=2,
        c2w_kv_heads=1,
        c2w_head_dim=2,
        c2w_sliding_window=8,
        dtype=torch.float32,
    )
    pool = KVCachePool(
        max_slots=1,
        config=config,
        device=torch.device("cpu"),
        preallocate=True,
    )
    slot = pool.allocate("successor")
    assert slot is not None
    slot.c2w_pooled = True
    executor = _executor()
    executor._kv_pool = pool

    old = torch.full((1, 4, 1, 3, 2), 1.0)
    new = torch.full((1, 4, 1, 3, 2), 9.0)
    pool.write_c2w_right_aligned(slot.slot_id, old)
    slot.c2w_len = 3
    assert executor.apply_c2w_warm_state(
        slot,
        new,
        [torch.ones(1, 1, 1), torch.ones(1, 1, 1)],
        [torch.ones(1, 1, 1)],
        6,
    )

    restored = pool.snapshot_pooled_c2w_kv(
        slot.slot_id,
        expected_allocation_epoch=slot.allocation_epoch,
    )
    assert restored is not None
    torch.testing.assert_close(restored, new)
    assert slot.c2w_kv is None
    assert slot.c2w_len == 3
