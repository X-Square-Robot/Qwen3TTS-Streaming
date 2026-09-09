"""Real CUDA pooled/arena speech-state bundle round-trip.

This validates storage ownership and detached payload restoration only.  It
does not claim that a model's logits, codec or PCM are equivalent after a
transfer; that requires the real verified checkpoint and TRT evidence.
"""

from __future__ import annotations

import os

import pytest
import torch

from engine.backend.engine_loop import EngineSegment
from engine.backend.executor import Executor
from engine.backend.kv_cache_pool import KVCachePool, ModelConfig
from engine.backend.speech_state import capture_segment_runtime_metadata


def test_real_cuda_pooled_and_arena_bundle_round_trip():
    if os.environ.get("RUN_CUDA_STATE_TRANSFER_TESTS") != "1":
        pytest.skip("set RUN_CUDA_STATE_TRANSFER_TESTS=1 for CUDA state transfer")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device("cuda", 0)
    config = ModelConfig(
        num_layers=1,
        kv_heads=1,
        head_dim=2,
        max_seq_len=8,
        hidden_size=4,
        n_c2w_layers=1,
        c2w_kv_heads=1,
        c2w_head_dim=2,
        c2w_sliding_window=4,
        n_c2w_conv_states=1,
        n_c2w_transconv_states=1,
        dtype=torch.float32,
    )
    pool = KVCachePool(2, config, device, preallocate=True)
    executor = Executor.__new__(Executor)
    executor._device = device
    executor._config = config
    executor._max_batch = 2
    executor._kv_pool = pool
    executor._compute_stream = torch.cuda.Stream(device=device)
    executor._c2w_conv_shapes = [(1, 2, 2)]
    executor._c2w_transconv_shapes = [(1, 2, 1)]
    executor._init_c2w_state_arenas()

    source = pool.allocate("session")
    assert source is not None
    source.segment_idx = 0
    source.past_len = 3
    source.c2w_len = 2
    source.c2w_pooled = True
    pool._talker_kv_pool[source.slot_id, :, :, :3, :].fill_(2.0)
    pool.write_c2w_right_aligned(
        source.slot_id,
        torch.full((1, 2, 1, 2, 2), 3.0, device=device),
    )
    rows = executor.take_zeroed_state_rows([source.slot_id])
    assert rows is not None
    conv, transconv, conv_write, transconv_write = rows[0]
    source.c2w_conv_states = conv
    source.c2w_transconv_states = transconv
    source._c2w_conv_write = conv_write
    source._c2w_transconv_write = transconv_write
    source.c2w_arena_backed = True
    source.c2w_write_in_a = False
    source.c2w_conv_states[0].fill_(4.0)
    source.c2w_transconv_states[0].fill_(5.0)
    source._c2w_conv_write[0].fill_(6.0)
    source._c2w_transconv_write[0].fill_(7.0)
    source.cursor_mu = torch.tensor([1.5], device=device)

    segment = EngineSegment("session", 0)
    segment.state = "active"
    segment.slot = source
    metadata = capture_segment_runtime_metadata(segment)

    executor.synchronize_state_transfer()
    bundle = executor.capture_speech_state_bundle(
        source,
        metadata,
        expected_slot_session_id="session",
        max_tensor_bytes=1_000_000,
    )
    target = pool.allocate("session")
    assert target is not None
    target.segment_idx = 0

    try:
        executor.restore_speech_state_bundle(
            target,
            bundle,
            expected_allocation_epoch=target.allocation_epoch,
        )
        executor.synchronize_state_transfer()
        assert target.past_len == 3
        assert target.c2w_len == 2
        assert target.c2w_pooled is True
        assert target.c2w_arena_backed is True
        assert torch.all(pool._talker_kv_pool[target.slot_id, :, :, :3, :] == 2.0)
        assert torch.all(pool._c2w_kv_pool[target.slot_id, :, :, -2:, :] == 3.0)
        assert torch.all(target.c2w_conv_states[0] == 4.0)
        assert torch.all(target.c2w_transconv_states[0] == 5.0)
        assert torch.all(target._c2w_conv_write[0] == 6.0)
        assert torch.all(target._c2w_transconv_write[0] == 7.0)
        assert float(target.cursor_mu.item()) == 1.5
    finally:
        pool.release(target.slot_id, expected_allocation_epoch=target.allocation_epoch)
        pool.release(source.slot_id, expected_allocation_epoch=source.allocation_epoch)
        torch.cuda.synchronize(device)
