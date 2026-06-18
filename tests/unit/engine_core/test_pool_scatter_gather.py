"""Tests for KVCachePool scatter/gather (pool-based KV management)."""

import torch
import pytest

from engine.backend.kv_cache_pool import KVCachePool, ModelConfig


DTYPE = torch.float32


@pytest.fixture
def pool():
    cfg = ModelConfig(
        num_layers=2, kv_heads=2, head_dim=4, max_seq_len=16,
        n_c2w_layers=2, c2w_kv_heads=2, c2w_head_dim=4, c2w_sliding_window=8,
        dtype=DTYPE,
    )
    return KVCachePool(max_slots=4, config=cfg, device=torch.device("cpu"), preallocate=True)


class TestScatterPrefillKV:
    def test_writes_to_pool(self, pool):
        slot = pool.allocate("s1")
        kv = torch.ones(1, 4, 2, 5, 4)
        pool.scatter_prefill_kv(slot.slot_id, kv, 5)
        gathered = pool.gather_talker_kv([slot.slot_id], 5)
        assert gathered.shape == (1, 4, 2, 5, 4)
        assert torch.allclose(gathered, kv)

    def test_only_writes_seq_len_portion(self, pool):
        slot = pool.allocate("s1")
        kv = torch.ones(1, 4, 2, 3, 4)
        pool.scatter_prefill_kv(slot.slot_id, kv, 3)
        full = pool.gather_talker_kv([slot.slot_id], 5)
        assert torch.allclose(full[:, :, :, :3, :], kv)
        assert torch.all(full[:, :, :, 3:, :] == 0)


class TestScatterPrefillC2WKV:
    def test_writes_c2w_to_pool(self, pool):
        slot = pool.allocate("s1")
        kv = torch.ones(1, 4, 2, 3, 4)
        pool.scatter_prefill_c2w_kv(slot.slot_id, kv)
        gathered = pool.gather_c2w_kv([slot.slot_id], 3)
        assert torch.allclose(gathered, kv)


class TestGatherTalkerKV:
    def test_gather_multiple_slots(self, pool):
        s0 = pool.allocate("s0")
        s1 = pool.allocate("s1")
        kv0 = torch.full((1, 4, 2, 4, 4), 1.0)
        kv1 = torch.full((1, 4, 2, 4, 4), 2.0)
        pool.scatter_prefill_kv(s0.slot_id, kv0, 4)
        pool.scatter_prefill_kv(s1.slot_id, kv1, 4)

        batched = pool.gather_talker_kv([s0.slot_id, s1.slot_id], 4)
        assert batched.shape == (2, 4, 2, 4, 4)
        assert torch.allclose(batched[0], kv0[0])
        assert torch.allclose(batched[1], kv1[0])


class TestScatterTalkerKV:
    def test_uniform_past_lens(self, pool):
        s0 = pool.allocate("s0")
        s1 = pool.allocate("s1")
        kv0 = torch.full((1, 4, 2, 3, 4), 1.0)
        kv1 = torch.full((1, 4, 2, 3, 4), 2.0)
        pool.scatter_prefill_kv(s0.slot_id, kv0, 3)
        pool.scatter_prefill_kv(s1.slot_id, kv1, 3)

        present_kv = torch.cat([
            torch.full((1, 4, 2, 4, 4), 10.0),
            torch.full((1, 4, 2, 4, 4), 20.0),
        ], dim=0)

        pool.scatter_talker_kv(
            [s0.slot_id, s1.slot_id], present_kv,
            original_past_lens=[3, 3], padded_past_len=3, seq=1,
        )

        g0 = pool.gather_talker_kv([s0.slot_id], 4)
        g1 = pool.gather_talker_kv([s1.slot_id], 4)
        assert torch.allclose(g0[0, :, :, :4, :], present_kv[0, :, :, :4, :])
        assert torch.allclose(g1[0, :, :, :4, :], present_kv[1, :, :, :4, :])

    def test_heterogeneous_past_lens(self, pool):
        s0 = pool.allocate("s0")
        s1 = pool.allocate("s1")
        kv0 = torch.full((1, 4, 2, 2, 4), 1.0)
        kv1 = torch.full((1, 4, 2, 4, 4), 2.0)
        pool.scatter_prefill_kv(s0.slot_id, kv0, 2)
        pool.scatter_prefill_kv(s1.slot_id, kv1, 4)

        present_kv = torch.cat([
            torch.full((1, 4, 2, 5, 4), 10.0),
            torch.full((1, 4, 2, 5, 4), 20.0),
        ], dim=0)

        pool.scatter_talker_kv(
            [s0.slot_id, s1.slot_id], present_kv,
            original_past_lens=[2, 4], padded_past_len=4, seq=1,
        )

        g0 = pool.gather_talker_kv([s0.slot_id], 3)
        assert g0.shape == (1, 4, 2, 3, 4)
        assert torch.allclose(g0[0, :, :, :2, :], present_kv[0, :, :, :2, :])
        assert torch.allclose(g0[0, :, :, 2:3, :], present_kv[0, :, :, 4:5, :])


class TestScatterC2WKV:
    def test_scatter_c2w(self, pool):
        s0 = pool.allocate("s0")
        s1 = pool.allocate("s1")
        present = torch.cat([
            torch.full((1, 4, 2, 5, 4), 10.0),
            torch.full((1, 4, 2, 5, 4), 20.0),
        ], dim=0)
        pool.scatter_c2w_kv([s0.slot_id, s1.slot_id], present)

        g0 = pool.gather_c2w_kv([s0.slot_id], 5)
        g1 = pool.gather_c2w_kv([s1.slot_id], 5)
        assert torch.allclose(g0[0], present[0])
        assert torch.allclose(g1[0], present[1])


class TestScatterTalkerKVDelta:
    def test_appends_delta(self, pool):
        s0 = pool.allocate("s0")
        base = torch.full((1, 4, 2, 3, 4), 1.0)
        delta = torch.full((1, 4, 2, 2, 4), 9.0)
        pool.scatter_prefill_kv(s0.slot_id, base, 3)

        pool.scatter_talker_kv_delta([s0.slot_id], delta, [3])

        gathered = pool.gather_talker_kv([s0.slot_id], 5)
        assert torch.allclose(gathered[0, :, :, :3, :], base[0])
        assert torch.allclose(gathered[0, :, :, 3:5, :], delta[0])


class TestScatterC2WKVDelta:
    def test_appends_and_crops_window(self, pool):
        s0 = pool.allocate("s0")
        base = torch.arange(1, 1 + 4 * 2 * 7 * 4, dtype=DTYPE).reshape(1, 4, 2, 7, 4)
        delta = torch.full((1, 4, 2, 2, 4), 99.0)
        pool.scatter_prefill_c2w_kv(s0.slot_id, base)

        pool.scatter_c2w_kv_delta([s0.slot_id], delta, [7])

        gathered = pool.gather_c2w_kv([s0.slot_id], 7)
        expected = torch.cat([base[:, :, :, -5:, :], delta], dim=3)
        assert torch.allclose(gathered, expected)


class TestStepOutputFields:
    def test_step_output_has_batch_kv(self):
        from engine.backend.executor import StepOutput
        out = StepOutput(
            slots=[], eos_flags=[], audio_chunks=[],
            batch_talker_kv=torch.zeros(2, 4, 2, 5, 4),
            batch_c2w_kv=torch.zeros(2, 4, 2, 3, 4),
            original_past_lens=[3, 4],
            padded_past_len=4,
        )
        assert out.batch_talker_kv is not None
        assert out.batch_c2w_kv is not None
        assert out.original_past_lens == [3, 4]
