"""
Window sampling for evolution and validation.

Two modes:
1. sample_evolution_windows: Random windows per generation (window rotation)
2. create_cpcv_folds: Deterministic folds for post-evolution validation (Sprint 3)
"""

import random
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def sample_evolution_windows(data: pd.DataFrame,
                             n_windows: int = 10,
                             window_bars: int = 2880,
                             ) -> List[pd.DataFrame]:
    """
    Sample random non-overlapping windows from data for one generation.

    Called each generation with different random state to implement
    window rotation (anti-overfitting during evolution).

    Args:
        data: Full OHLCV DataFrame (evolution period only, excl. OTS)
        n_windows: Number of windows to sample
        window_bars: Bars per window (default 2880 = ~1 month at 15m)

    Returns:
        List of DataFrame slices
    """
    total_bars = len(data)
    if total_bars < window_bars:
        logger.warning(f"Data ({total_bars} bars) shorter than window ({window_bars} bars)")
        return [data]

    max_start = total_bars - window_bars
    if max_start <= 0:
        return [data]

    # Sample random start positions
    starts = []
    attempts = 0
    while len(starts) < n_windows and attempts < n_windows * 10:
        start = random.randint(0, max_start)
        # Check no overlap with existing windows
        overlaps = False
        for s in starts:
            if abs(start - s) < window_bars:
                overlaps = True
                break
        if not overlaps:
            starts.append(start)
        attempts += 1

    # If we couldn't get enough non-overlapping, allow overlapping
    if len(starts) < n_windows:
        while len(starts) < n_windows:
            starts.append(random.randint(0, max_start))

    windows = [data.iloc[s:s + window_bars].copy() for s in sorted(starts)]

    logger.debug(f"Sampled {len(windows)} windows of {window_bars} bars each")
    return windows


def sample_windows_with_rotation(data: pd.DataFrame,
                                  n_windows: int = 5,
                                  window_bars: int = 8640,
                                  previous_windows: List[Tuple[pd.DataFrame, str]] = None,
                                  keep_ratio: float = 0.6,
                                  ) -> List[Tuple[pd.DataFrame, str]]:
    """
    Sample windows with partial rotation for NSGA-II evaluation.

    Returns list of (DataFrame, window_id) tuples.
    window_id is "w_{start}_{bars}" — used as cache key.

    Args:
        data: Full training OHLCV
        n_windows: Total windows per generation
        window_bars: Bars per window (8640 = ~3 months at 15m)
        previous_windows: [(df, window_id), ...] from previous generation
        keep_ratio: Fraction of windows to keep from previous gen
    """
    total_bars = len(data)
    if total_bars < window_bars:
        logger.warning(f"Data ({total_bars}) shorter than window ({window_bars})")
        wid = f"w_0_{window_bars}"
        return [(data, wid)]

    max_start = total_bars - window_bars

    # Build pool of all valid start positions (non-overlapping grid)
    all_starts = list(range(0, max_start + 1, window_bars))
    if not all_starts:
        all_starts = [0]

    # Determine which windows to keep from previous generation
    kept = []
    if previous_windows:
        n_keep = int(n_windows * keep_ratio)
        kept = random.sample(previous_windows, min(n_keep, len(previous_windows)))

    # Sample fresh windows for remaining slots
    kept_ids = {wid for _, wid in kept}
    n_fresh = n_windows - len(kept)

    available = [s for s in all_starts if f"w_{s}_{window_bars}" not in kept_ids]
    if len(available) < n_fresh:
        fresh_starts = [random.randint(0, max_start) for _ in range(n_fresh)]
    else:
        fresh_starts = random.sample(available, n_fresh)

    fresh = [(data.iloc[s:s + window_bars].copy(), f"w_{s}_{window_bars}")
             for s in fresh_starts]

    result = kept + fresh
    logger.debug(f"Windows: {len(kept)} kept + {len(fresh)} fresh = {len(result)} total")
    return result
