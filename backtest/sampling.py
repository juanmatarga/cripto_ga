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
