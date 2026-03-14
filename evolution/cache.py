"""
Per-window evaluation cache for evolution.

Keys: (genome_tuple, window_id) → per-window metrics dict.
Eviction: purge entries for windows no longer in active set.
"""

from typing import Dict, Optional, Set, Tuple


class EvalCache:
    """Cache backtest results keyed by (genome, window_id)."""

    def __init__(self):
        self._cache: Dict[Tuple[tuple, str], dict] = {}
        self._hits = 0
        self._misses = 0

    def get(self, genome: tuple, window_id: str) -> Optional[dict]:
        """Look up cached metrics. Returns None on miss."""
        return self._cache.get((genome, window_id))

    def put(self, genome: tuple, window_id: str, metrics: dict):
        """Store metrics for a (genome, window) pair."""
        self._cache[(genome, window_id)] = metrics

    def evict_except(self, active_windows: Set[str]):
        """Remove entries for windows not in active set."""
        to_remove = [k for k in self._cache if k[1] not in active_windows]
        for k in to_remove:
            del self._cache[k]

    def record_hit(self):
        self._hits += 1

    def record_miss(self):
        self._misses += 1

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            'entries': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0.0,
        }

    def reset_counters(self):
        self._hits = 0
        self._misses = 0
