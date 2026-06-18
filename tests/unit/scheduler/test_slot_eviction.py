"""Tests for KVCachePool slot eviction."""

import time

import torch
import pytest

from engine.backend.kv_cache_pool import KVCachePool, ModelConfig, SlotKVState


@pytest.fixture
def pool():
    cfg = ModelConfig(
        num_layers=2, kv_heads=2, head_dim=4, max_seq_len=16,
        n_c2w_layers=2, c2w_kv_heads=2, c2w_head_dim=4, c2w_sliding_window=8,
    )
    return KVCachePool(max_slots=4, config=cfg, device=torch.device("cpu"), preallocate=False)


class TestSlotActivity:
    def test_touch_updates_time(self):
        slot = SlotKVState(slot_id=0)
        t0 = slot.last_active_time
        time.sleep(0.01)
        slot.touch()
        assert slot.last_active_time > t0

    def test_idle_seconds(self):
        slot = SlotKVState(slot_id=0)
        time.sleep(0.05)
        assert slot.idle_seconds >= 0.04


class TestEviction:
    def test_no_candidate_when_all_free(self, pool):
        assert pool.find_eviction_candidate(max_idle_sec=0.0) is None

    def test_no_candidate_within_window(self, pool):
        pool.allocate("s1")
        assert pool.find_eviction_candidate(max_idle_sec=999.0) is None

    def test_finds_idle_candidate(self, pool):
        slot = pool.allocate("s1")
        slot.last_active_time = time.monotonic() - 20.0
        candidate = pool.find_eviction_candidate(max_idle_sec=10.0)
        assert candidate is not None
        assert candidate.slot_id == slot.slot_id

    def test_force_evict_returns_session(self, pool):
        slot = pool.allocate("s1")
        evicted = pool.force_evict(slot.slot_id)
        assert evicted == "s1"
        assert pool.free_count == 4

    def test_force_evict_free_slot_returns_none(self, pool):
        assert pool.force_evict(0) is None

    def test_release_is_idempotent(self, pool):
        slot = pool.allocate("s1")
        pool.release(slot.slot_id)
        pool.release(slot.slot_id)
        allocated = [pool.allocate(f"s{i}") for i in range(4)]
        allocated_ids = [s.slot_id for s in allocated if s is not None]
        assert allocated_ids == [slot.slot_id, 2, 1, 0]
        assert pool.allocate("overflow") is None

    def test_reused_slot_clears_pad_state(self, pool):
        slot = pool.allocate("s1")
        slot.pad_start_frame = 12
        slot.pad_consecutive_silence = 7
        pool.release(slot.slot_id)

        reused = pool.allocate("s2")
        assert reused.slot_id == slot.slot_id
        assert reused.pad_start_frame == -1
        assert reused.pad_consecutive_silence == 0

    def test_reused_slot_clears_sampling_state(self, pool):
        slot = pool.allocate("s1")
        slot.sampling_seed = 123
        slot.sampling_generator = torch.Generator()
        pool.release(slot.slot_id)

        reused = pool.allocate("s2")
        assert reused.slot_id == slot.slot_id
        assert reused.sampling_seed is None
        assert reused.sampling_generator is None

    def test_evicts_most_idle(self, pool):
        s1 = pool.allocate("s1")
        s2 = pool.allocate("s2")
        s1.last_active_time = time.monotonic() - 30.0
        s2.last_active_time = time.monotonic() - 10.0
        candidate = pool.find_eviction_candidate(max_idle_sec=5.0)
        assert candidate.session_id == "s1"


class TestUtilization:
    def test_utilization_zero(self, pool):
        assert pool.utilization == 0.0

    def test_utilization_partial(self, pool):
        pool.allocate("s1")
        pool.allocate("s2")
        assert pool.utilization == pytest.approx(0.5)

    def test_utilization_full(self, pool):
        for i in range(4):
            pool.allocate(f"s{i}")
        assert pool.utilization == pytest.approx(1.0)
