"""
Combinatorial Purged Cross-Validation (CPCV)
Bailey & Lopez de Prado, 2014.

Generates C(N, k) train/test combinations with purge gaps and embargo
periods to prevent information leakage. Produces a distribution of
OOS performance for PBO calculation.
"""

import numpy as np
import pandas as pd
import logging
from itertools import combinations
from typing import List, Tuple, Dict

from strategy.phenotype import Strategy
from strategy.vectorized_eval import generate_signals
from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
from backtest.metrics import (
    calculate_returns, calculate_sortino_ratio, calculate_calmar_ratio,
    cagr, max_drawdown, calculate_all_metrics
)

logger = logging.getLogger(__name__)


def create_cpcv_groups(data: pd.DataFrame, n_groups: int) -> List[pd.DataFrame]:
    """
    Split data into N contiguous groups of roughly equal size.

    Returns list of DataFrames (one per group).
    """
    n = len(data)
    group_size = n // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else n
        groups.append(data.iloc[start:end])
    return groups


def generate_cpcv_splits(n_groups: int, k: int = None
                         ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Generate all C(N, k) combinations of train/test group assignments.

    Args:
        n_groups: Total number of groups
        k: Number of groups in test set. Default: n_groups // 2

    Returns:
        List of (train_group_indices, test_group_indices)
    """
    if k is None:
        k = n_groups // 2

    all_indices = list(range(n_groups))
    splits = []
    for test_indices in combinations(all_indices, k):
        train_indices = tuple(i for i in all_indices if i not in test_indices)
        splits.append((train_indices, test_indices))
    return splits


def apply_purge_embargo(groups: List[pd.DataFrame],
                        train_indices: Tuple[int, ...],
                        test_indices: Tuple[int, ...],
                        purge_bars: int = 96,
                        embargo_bars: int = 48
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble train and test DataFrames with purge and embargo.

    Purge: Remove `purge_bars` from the END of each train group that is
           immediately BEFORE a test group.
    Embargo: Remove `embargo_bars` from the START of each test group that is
             immediately AFTER a train group.

    This prevents lookahead bias at train/test boundaries.
    """
    train_parts = []
    test_parts = []

    train_set = set(train_indices)
    test_set = set(test_indices)

    for idx in sorted(train_indices):
        group = groups[idx]
        # Check if next group is in test set → purge end of this train group
        if idx + 1 in test_set:
            if len(group) > purge_bars:
                group = group.iloc[:-purge_bars]
            else:
                continue  # Group too small after purge
        train_parts.append(group)

    for idx in sorted(test_indices):
        group = groups[idx]
        # Check if previous group is in train set → embargo start of this test group
        if idx - 1 in train_set:
            if len(group) > embargo_bars:
                group = group.iloc[embargo_bars:]
            else:
                continue
        test_parts.append(group)

    train_df = pd.concat(train_parts) if train_parts else pd.DataFrame()
    test_df = pd.concat(test_parts) if test_parts else pd.DataFrame()

    return train_df, test_df


def cpcv_evaluate(strategy: Strategy, data: pd.DataFrame,
                  config: dict,
                  n_groups: int = 10,
                  purge_bars: int = 96,
                  embargo_bars: int = 48,
                  max_splits: int = 252,
                  ) -> Dict:
    """
    Evaluate a strategy using CPCV.

    Args:
        strategy: Decoded Strategy phenotype
        data: Full OHLCV DataFrame (evolution period, excl. OTS)
        config: Config with costs section
        n_groups: Number of data groups (default 10)
        purge_bars: Bars to purge at train/test boundary (default 96 = 24h at 15m)
        embargo_bars: Bars to embargo after train (default 48 = 12h at 15m)
        max_splits: Maximum number of splits to evaluate (default 252 = C(10,5))

    Returns:
        Dict with:
        - oos_sortinos: list of OOS Sortino ratios per split
        - oos_calmars: list of OOS Calmar ratios per split
        - oos_returns: list of OOS total returns per split
        - n_splits: number of splits evaluated
        - mean_sortino: mean OOS Sortino
        - mean_calmar: mean OOS Calmar
    """
    groups = create_cpcv_groups(data, n_groups)
    splits = generate_cpcv_splits(n_groups)

    if len(splits) > max_splits:
        # Subsample splits randomly
        rng = np.random.RandomState(42)
        indices = rng.choice(len(splits), max_splits, replace=False)
        splits = [splits[i] for i in indices]

    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)

    oos_sortinos = []
    oos_calmars = []
    oos_returns = []
    oos_trades = []

    for train_idx, test_idx in splits:
        train_df, test_df = apply_purge_embargo(
            groups, train_idx, test_idx, purge_bars, embargo_bars
        )

        if len(test_df) < 100:
            continue

        try:
            equity, trades = _run_single_window(
                strategy, test_df, costs_config, atr_period
            )

            if len(trades) < 3:
                oos_sortinos.append(0.0)
                oos_calmars.append(0.0)
                oos_returns.append(0.0)
                oos_trades.append(len(trades))
                continue

            returns = equity.pct_change().dropna()
            sortino = calculate_sortino_ratio(returns, BARS_PER_YEAR_15M)
            cagr_val = cagr(equity, BARS_PER_YEAR_15M)
            max_dd_val = max_drawdown(equity)
            calmar = calculate_calmar_ratio(cagr_val, max_dd_val)

            # Cap extreme values
            sortino = min(max(sortino, -10.0), 10.0)
            calmar = min(max(calmar, -10.0), 10.0)

            oos_sortinos.append(sortino)
            oos_calmars.append(calmar)
            oos_returns.append(float((equity.iloc[-1] / equity.iloc[0]) - 1))
            oos_trades.append(len(trades))

        except Exception as e:
            logger.debug(f"CPCV split failed: {e}")
            continue

    result = {
        'oos_sortinos': oos_sortinos,
        'oos_calmars': oos_calmars,
        'oos_returns': oos_returns,
        'oos_trades': oos_trades,
        'n_splits': len(oos_sortinos),
        'mean_sortino': float(np.mean(oos_sortinos)) if oos_sortinos else 0.0,
        'mean_calmar': float(np.mean(oos_calmars)) if oos_calmars else 0.0,
        'std_sortino': float(np.std(oos_sortinos)) if oos_sortinos else 0.0,
        'pct_positive_sortino': sum(1 for s in oos_sortinos if s > 0) / len(oos_sortinos) if oos_sortinos else 0.0,
    }

    logger.info(f"CPCV: {result['n_splits']} splits, "
                f"mean Sortino={result['mean_sortino']:.3f}, "
                f"mean Calmar={result['mean_calmar']:.3f}, "
                f"pct positive={result['pct_positive_sortino']:.0%}")

    return result
