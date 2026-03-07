"""
Signal Permutation Test.

Shuffles a strategy's SIGNALS (not trades) and recalculates metrics.
If the shuffled signals produce similar performance, the signal has
no predictive power.

More rigorous than trade shuffling (analysis/monte_carlo.py) because
it tests the signal itself, not just trade ordering.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict

from strategy.phenotype import Strategy
from strategy.vectorized_eval import generate_signals
from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
from backtest.metrics import calculate_returns, calculate_sortino_ratio

logger = logging.getLogger(__name__)


def signal_permutation_test(strategy: Strategy,
                            data: pd.DataFrame,
                            config: dict,
                            n_permutations: int = 1000,
                            seed: int = 42) -> Dict:
    """
    Test whether a strategy's signal has predictive power.

    1. Calculate real metrics from the strategy's actual signals
    2. Repeat n_permutations times:
       a. Shuffle the signal vector (preserving same number of True signals)
       b. Run backtest with shuffled signals
       c. Calculate metrics
    3. p-value = proportion of permutations with metric >= real metric

    Args:
        strategy: Decoded Strategy phenotype
        data: OHLCV DataFrame
        config: Config with costs section
        n_permutations: Number of shuffle iterations
        seed: Random seed

    Returns:
        Dict with p_value, real_sortino, permuted_sortinos, interpretation
    """
    rng = np.random.RandomState(seed)

    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)

    # Generate real signals and run real backtest
    real_signals = generate_signals(strategy, data)
    n_signals = int(real_signals.sum())

    if n_signals < 3:
        return {
            'p_value': 1.0,
            'real_sortino': 0.0,
            'n_signals': n_signals,
            'n_permutations': 0,
            'interpretation': 'Too few signals to test',
        }

    # Real backtest
    try:
        real_equity, real_trades = _run_single_window(
            strategy, data, costs_config, atr_period
        )
        if len(real_trades) < 3:
            return {
                'p_value': 1.0,
                'real_sortino': 0.0,
                'n_signals': n_signals,
                'n_permutations': 0,
                'interpretation': 'Too few trades',
            }
        real_returns = real_equity.pct_change().dropna()
        real_sortino = calculate_sortino_ratio(real_returns, BARS_PER_YEAR_15M)
    except Exception:
        return {
            'p_value': 1.0,
            'real_sortino': 0.0,
            'n_signals': n_signals,
            'n_permutations': 0,
            'interpretation': 'Real backtest failed',
        }

    # Permutation loop
    sig_array = real_signals.values.copy()
    permuted_sortinos = []

    for i in range(n_permutations):
        # Shuffle the signal vector (preserves total count of True)
        shuffled = sig_array.copy()
        rng.shuffle(shuffled)

        # Create a temporary strategy-like object for the backtest
        # We override the signals by patching the backtest
        shuffled_equity = _run_with_shuffled_signals(
            shuffled, strategy, data, costs_config, atr_period
        )
        if shuffled_equity is not None:
            perm_returns = shuffled_equity.pct_change().dropna()
            perm_sortino = calculate_sortino_ratio(perm_returns, BARS_PER_YEAR_15M)
            perm_sortino = min(max(perm_sortino, -10.0), 10.0)
            permuted_sortinos.append(perm_sortino)

    if not permuted_sortinos:
        return {
            'p_value': 1.0,
            'real_sortino': real_sortino,
            'n_signals': n_signals,
            'n_permutations': 0,
            'interpretation': 'All permutations failed',
        }

    # p-value: how often does random beat real?
    n_better = sum(1 for s in permuted_sortinos if s >= real_sortino)
    p_value = n_better / len(permuted_sortinos)

    if p_value < 0.01:
        interp = f'Very strong signal (p={p_value:.4f} < 0.01)'
    elif p_value < 0.05:
        interp = f'Significant signal (p={p_value:.4f} < 0.05)'
    elif p_value < 0.10:
        interp = f'Marginal signal (p={p_value:.4f} < 0.10)'
    else:
        interp = f'No significant signal (p={p_value:.4f} >= 0.10)'

    result = {
        'p_value': p_value,
        'real_sortino': float(real_sortino),
        'mean_permuted_sortino': float(np.mean(permuted_sortinos)),
        'n_signals': n_signals,
        'n_permutations': len(permuted_sortinos),
        'interpretation': interp,
    }

    logger.info(f"Signal permutation: p={p_value:.4f} | "
                f"real Sortino={real_sortino:.3f} | "
                f"mean permuted={np.mean(permuted_sortinos):.3f}")

    return result


def _run_with_shuffled_signals(sig_array: np.ndarray,
                                strategy: Strategy,
                                df: pd.DataFrame,
                                costs_config: dict,
                                atr_period: int) -> pd.Series:
    """
    Run backtest with pre-computed (shuffled) signals.

    Reimplements the core backtest loop from fitness._run_single_window
    but uses the provided signal array instead of generating from strategy.
    """
    from backtest.exits import calculate_atr

    atr = calculate_atr(df, period=atr_period)
    direction = strategy.direction
    tp_mult = strategy.tp_atr_mult
    sl_mult = strategy.sl_atr_mult

    fees_bps = costs_config.get(f'fees_bps_{direction.lower()}', 1.0)
    slip_bps = costs_config.get(f'slippage_bps_{direction.lower()}', 1.0)
    total_cost = (fees_bps + slip_bps) / 10000.0

    equity = 100.0
    equity_curve = np.full(len(df), equity)

    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    atr_vals = atr.values

    for i in range(len(df)):
        if in_position:
            if direction == 'LONG':
                stop_hit = lows[i] <= stop_loss
                target_hit = highs[i] >= take_profit
            else:
                stop_hit = highs[i] >= stop_loss
                target_hit = lows[i] <= take_profit

            exit_price = None
            if stop_hit and target_hit:
                exit_price = stop_loss
            elif stop_hit:
                exit_price = stop_loss
            elif target_hit:
                exit_price = take_profit

            if exit_price is not None:
                if direction == 'LONG':
                    adj_exit = exit_price * (1 - total_cost)
                    pnl_pct = (adj_exit - entry_price) / entry_price
                else:
                    adj_exit = exit_price * (1 + total_cost)
                    pnl_pct = (entry_price - adj_exit) / entry_price
                equity *= (1 + pnl_pct)
                in_position = False

        else:
            if sig_array[i] and not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
                raw_price = closes[i]
                if direction == 'LONG':
                    entry_price = raw_price * (1 + total_cost)
                    stop_loss = entry_price - sl_mult * atr_vals[i]
                    take_profit = entry_price + tp_mult * atr_vals[i]
                else:
                    entry_price = raw_price * (1 - total_cost)
                    stop_loss = entry_price + sl_mult * atr_vals[i]
                    take_profit = entry_price - tp_mult * atr_vals[i]
                in_position = True

        equity_curve[i] = equity

    if in_position:
        close_price = closes[-1]
        if direction == 'LONG':
            adj_exit = close_price * (1 - total_cost)
            pnl_pct = (adj_exit - entry_price) / entry_price
        else:
            adj_exit = close_price * (1 + total_cost)
            pnl_pct = (entry_price - adj_exit) / entry_price
        equity *= (1 + pnl_pct)
        equity_curve[-1] = equity

    return pd.Series(equity_curve, index=df.index)
