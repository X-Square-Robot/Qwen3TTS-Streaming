"""Tests for engine.backend.prefix_cache.PrefixKVCache."""

import time

import torch
import pytest

from engine.backend.prefix_cache import PrefixKVCache, PrefixCacheEntry


def _make_kv(prefix_len: int = 10, layers: int = 56, heads: int = 8, dim: int = 64):
    return torch.randn(1, layers, heads, prefix_len, dim)


class TestPrefixKVCache:
    def test_miss_returns_none(self):
        cache = PrefixKVCache(max_entries=4)
        assert cache.get("nonexistent") is None
        assert cache.get(None) is None

    def test_disabled_put_is_noop(self):
        """max_entries=0 (prefix_cache.enabled=false) must not raise on put."""
        cache = PrefixKVCache(max_entries=0)
        cache.put("k", _make_kv(5), 5)
        assert cache.size == 0
        assert cache.get("k") is None

    def test_put_and_hit(self):
        cache = PrefixKVCache(max_entries=4)
        kv = _make_kv(20)
        cache.put("key_a", kv, 20)
        entry = cache.get("key_a")
        assert entry is not None
        assert entry.prefix_len == 20
        assert torch.equal(entry.talker_kv, kv)
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 0

    def test_lru_eviction(self):
        cache = PrefixKVCache(max_entries=2)
        cache.put("a", _make_kv(5), 5)
        cache.put("b", _make_kv(6), 6)
        cache.put("c", _make_kv(7), 7)
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_access_refreshes_lru(self):
        cache = PrefixKVCache(max_entries=2)
        cache.put("a", _make_kv(5), 5)
        cache.put("b", _make_kv(6), 6)
        cache.get("a")
        cache.put("c", _make_kv(7), 7)
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_max_prefix_len_rejects_long(self):
        cache = PrefixKVCache(max_entries=4, max_prefix_len=10)
        cache.put("long", _make_kv(20), 20)
        assert cache.get("long") is None
        assert cache.size == 0

    def test_invalidate(self):
        cache = PrefixKVCache(max_entries=4)
        cache.put("x", _make_kv(3), 3)
        assert cache.invalidate("x") is True
        assert cache.invalidate("x") is False
        assert cache.get("x") is None

    def test_clear(self):
        cache = PrefixKVCache(max_entries=4)
        cache.put("a", _make_kv(3), 3)
        cache.put("b", _make_kv(4), 4)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_stats(self):
        cache = PrefixKVCache(max_entries=4)
        cache.put("a", _make_kv(3), 3)
        cache.get("a")
        cache.get("miss")
        stats = cache.stats
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert 0.4 < stats["hit_rate"] < 0.6

    def test_duplicate_put_no_error(self):
        cache = PrefixKVCache(max_entries=4)
        cache.put("a", _make_kv(3), 3)
        cache.put("a", _make_kv(5), 5)
        assert cache.size == 1

    def test_clone_independence(self):
        cache = PrefixKVCache(max_entries=4)
        kv = _make_kv(3)
        cache.put("a", kv, 3)
        kv.zero_()
        entry = cache.get("a")
        assert entry is not None
        assert entry.talker_kv.abs().sum() > 0
