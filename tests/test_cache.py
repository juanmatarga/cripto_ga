"""Tests for per-window evaluation cache."""

import pytest


class TestEvalCache:
    def test_cache_hit(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        genome = (1, 2, 3, 4, 5)
        window_id = "w_0_8640"
        metrics = {'sortino': 1.5, 'return_pct': 10.0, 'max_dd': -0.15, 'n_trades': 25}
        cache.put(genome, window_id, metrics)
        result = cache.get(genome, window_id)
        assert result == metrics

    def test_cache_miss(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        assert cache.get((1, 2, 3), "w_0") is None

    def test_evict_old_windows(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        cache.put((1,), "old_window", {'sortino': 1.0})
        cache.put((1,), "active_window", {'sortino': 2.0})
        cache.evict_except(active_windows={"active_window"})
        assert cache.get((1,), "old_window") is None
        assert cache.get((1,), "active_window") is not None

    def test_stats(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        cache.put((1,), "w1", {'sortino': 1.0})
        cache.record_hit()
        cache.record_miss()
        cache.record_miss()
        stats = cache.stats()
        assert stats['entries'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert abs(stats['hit_rate'] - 1/3) < 0.01

    def test_reset_counters(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        cache.record_hit()
        cache.record_hit()
        cache.reset_counters()
        stats = cache.stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
