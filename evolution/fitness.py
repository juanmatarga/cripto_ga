"""
Fitness evaluation for evolved strategies.

v2: Multi-component fitness targeting CAGR + risk-adjustment.
    Trailing stop support. Trend-following friendly constraints.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from typing import Optional as OptionalType
from strategy.phenotype import Strategy
from strategy.vectorized_eval import generate_signals
from backtest.exits import calculate_atr, calculate_exit_levels, check_exit_conditions
from backtest.metrics import (
    calculate_returns, calculate_sortino_ratio, calculate_calmar_ratio,
    calculate_all_metrics, cagr, max_drawdown
)
from data.multi_timeframe import prepare_multi_tf_data

logger = logging.getLogger(__name__)

FAIL_FITNESS = (-999.0, -999.0)

# Default constraints — trend-following friendly
DEFAULT_MIN_TRADES = 30
DEFAULT_MAX_DRAWDOWN = 0.30
DEFAULT_MIN_WIN_RATE = 0.20       # Low: trend-following has 25-40% win rate
DEFAULT_PARSIMONY_COEFF = 0.01

# Bars per year for 15m timeframe
BARS_PER_YEAR_15M = 35040


def evaluate_strategy(strategy: Strategy, windows: List[pd.DataFrame],
                      config: dict, **kwargs) -> Strategy:
    """
    Evaluate a strategy across multiple data windows.

    Sets strategy.fitness, strategy.metrics, strategy.n_trades in-place.
    """
    if not strategy.conditions:
        strategy.fitness = FAIL_FITNESS
        return strategy

    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)
    fitness_cfg = config.get('fitness', {})

    # Check signal rate before expensive backtest
    max_signal_rate = fitness_cfg.get('max_signal_rate', 0.30)
    windows_tf_data = kwargs.get('windows_tf_data', None)
    try:
        first_tf = windows_tf_data[0] if windows_tf_data else None
        signals = generate_signals(strategy, windows[0] if windows else pd.DataFrame(),
                                   tf_data=first_tf)
        signal_rate = float(signals.sum()) / len(signals) if len(signals) > 0 else 0
        if signal_rate > max_signal_rate:
            strategy.fitness = FAIL_FITNESS
            return strategy
    except Exception:
        pass

    all_trades = []
    all_equity_curves = []
    min_trades = fitness_cfg.get('min_trades', DEFAULT_MIN_TRADES)
    max_dd = fitness_cfg.get('max_drawdown', DEFAULT_MAX_DRAWDOWN)
    min_wr = fitness_cfg.get('min_win_rate', DEFAULT_MIN_WIN_RATE)
    parsimony = fitness_cfg.get('parsimony_coefficient', DEFAULT_PARSIMONY_COEFF)

    # Pre-computed multi-TF data per window (passed in via kwargs to avoid recompute)
    windows_tf_data = kwargs.get('windows_tf_data', None)

    for i_w, window_df in enumerate(windows):
        try:
            tf_data = windows_tf_data[i_w] if windows_tf_data else None
            equity, trades = _run_single_window(
                strategy, window_df, costs_config, atr_period,
                tf_data=tf_data,
            )
            all_equity_curves.append(equity)
            all_trades.extend(trades)
        except Exception as e:
            logger.debug(f"Window eval failed: {e}")
            continue

    if not all_equity_curves:
        strategy.fitness = FAIL_FITNESS
        return strategy

    n_trades = len(all_trades)
    strategy.n_trades = n_trades

    # Hard constraint: minimum trades
    if n_trades < min_trades:
        strategy.fitness = FAIL_FITNESS
        return strategy

    # Compute aggregate metrics
    combined_equity = _combine_equity_curves(all_equity_curves)
    returns = calculate_returns(combined_equity).dropna()

    if len(returns) < 10:
        strategy.fitness = FAIL_FITNESS
        return strategy

    # Win rate
    winning = sum(1 for t in all_trades if t['pnl_pct'] > 0)
    win_rate = winning / n_trades if n_trades > 0 else 0.0

    # Hard constraint: win rate (lenient for trend-following)
    if win_rate < min_wr:
        strategy.fitness = FAIL_FITNESS
        return strategy

    # Metrics
    sortino = calculate_sortino_ratio(returns, BARS_PER_YEAR_15M)
    cagr_val = cagr(combined_equity, BARS_PER_YEAR_15M)
    max_dd_val = max_drawdown(combined_equity)
    calmar = calculate_calmar_ratio(cagr_val, max_dd_val)

    # Hard constraint: max drawdown
    if abs(max_dd_val) > max_dd:
        strategy.fitness = FAIL_FITNESS
        return strategy

    # Cap extreme values
    sortino = max(min(sortino, 10.0), -10.0)
    calmar = min(calmar, 10.0)

    # Trade-level metrics
    winning_pnl = sum(t['pnl_pct'] for t in all_trades if t['pnl_pct'] > 0)
    losing_pnl = abs(sum(t['pnl_pct'] for t in all_trades if t['pnl_pct'] < 0))
    profit_factor = winning_pnl / max(losing_pnl, 1e-10)

    # Average winner / average loser ratio (edge quality)
    avg_win = winning_pnl / max(winning, 1)
    avg_loss = losing_pnl / max(n_trades - winning, 1)

    # Expectancy per trade
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Hard constraint: positive expectancy
    if expectancy <= 0:
        strategy.fitness = FAIL_FITNESS
        return strategy

    # Cross-regime check
    regime_labels = kwargs.get('regime_labels', None)
    regime_penalty = 0.0
    regime_pfs = {}
    if regime_labels is not None and len(all_trades) > 0:
        regime_pfs = _compute_regime_profit_factors(all_trades, windows, regime_labels)
        for regime, rpf in regime_pfs.items():
            if rpf < 0.5:
                regime_penalty += 0.5

    # ================================================================
    # FITNESS: Sortino + CAGR bonus + PF bonus + edge quality
    # ================================================================
    pf_bonus = min(profit_factor - 1.0, 3.0) if profit_factor > 1.0 else 0.0

    # Direct CAGR incentive (CAGR of 20% → bonus of 2.0)
    cagr_bonus = max(cagr_val, 0) * 10.0

    # Calmar bonus (risk-adjusted returns)
    calmar_bonus = max(min(calmar, 5.0), 0) * 0.3

    # Win/loss ratio bonus (rewards big winners vs small losers)
    wl_ratio = avg_win / max(avg_loss, 1e-6)
    wl_bonus = min(wl_ratio - 1.0, 3.0) if wl_ratio > 1.0 else 0.0

    fitness_0 = sortino + cagr_bonus + calmar_bonus + pf_bonus + wl_bonus - regime_penalty

    # Parsimony pressure
    fitness_0 -= parsimony * strategy.n_nodes
    calmar_adj = calmar - parsimony * strategy.n_nodes

    strategy.fitness = (fitness_0, calmar_adj)
    strategy.metrics = {
        'sortino': sortino,
        'calmar': calmar,
        'profit_factor': profit_factor,
        'cagr': cagr_val,
        'max_dd': max_dd_val,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'n_windows': len(all_equity_curves),
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'wl_ratio': wl_ratio,
        'regime_pfs': regime_pfs,
        'regime_penalty': regime_penalty,
    }

    return strategy


def _run_single_window(strategy: Strategy, df: pd.DataFrame,
                       costs_config: dict, atr_period: int,
                       max_hold_bars: int = 960,
                       tf_data: OptionalType[Dict[str, 'pd.DataFrame']] = None,
                       ) -> Tuple[pd.Series, List[dict]]:
    """
    Run backtest on a single window with trailing stop support.

    Args:
        max_hold_bars: Force-close after N bars (default 960 = 10 days of 15m).
        tf_data: Optional pre-computed multi-TF data dict. If None, computed on the fly.

    Returns (equity_curve, trades_list).
    """
    # Prepare multi-timeframe data if not provided
    if tf_data is None:
        tf_data = prepare_multi_tf_data(df)
    signals = generate_signals(strategy, df, tf_data=tf_data)
    atr = calculate_atr(df, period=atr_period)

    direction = strategy.direction
    tp_mult = strategy.tp_atr_mult
    sl_mult = strategy.sl_atr_mult
    trail_mult = strategy.trail_atr_mult
    has_tp = tp_mult > 0

    # Transaction cost
    fees_bps = costs_config.get(f'fees_bps_{direction.lower()}', 1.0)
    slip_bps = costs_config.get(f'slippage_bps_{direction.lower()}', 1.0)
    total_cost = (fees_bps + slip_bps) / 10000.0

    equity = 100.0
    equity_curve = np.full(len(df), equity)
    trades = []

    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    initial_stop = 0.0
    best_price = 0.0
    atr_at_entry = 0.0
    entry_bar = 0

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    atr_vals = atr.values
    sig_vals = signals.values

    for i in range(len(df)):
        if in_position:
            high_i = highs[i]
            low_i = lows[i]

            # Update trailing stop
            if trail_mult > 0 and atr_at_entry > 0:
                if direction == 'LONG':
                    best_price = max(best_price, high_i)
                    trail_level = best_price - trail_mult * atr_at_entry
                    if trail_level > stop_loss:
                        stop_loss = trail_level
                else:
                    best_price = min(best_price, low_i)
                    trail_level = best_price + trail_mult * atr_at_entry
                    if trail_level < stop_loss:
                        stop_loss = trail_level

            # Check exits
            if direction == 'LONG':
                stop_hit = low_i <= stop_loss
                target_hit = has_tp and high_i >= take_profit
            else:
                stop_hit = high_i >= stop_loss
                target_hit = has_tp and low_i <= take_profit

            # Time-based exit
            time_exit = (i - entry_bar) >= max_hold_bars

            exit_price = None
            exit_type = None
            if stop_hit and target_hit:
                exit_price = stop_loss  # conservative
                exit_type = 'trail' if (trail_mult > 0 and stop_loss != initial_stop) else 'stop'
            elif stop_hit:
                exit_price = stop_loss
                exit_type = 'trail' if (trail_mult > 0 and stop_loss != initial_stop) else 'stop'
            elif target_hit:
                exit_price = take_profit
                exit_type = 'target'
            elif time_exit:
                exit_price = closes[i]
                exit_type = 'time'

            if exit_price is not None:
                if direction == 'LONG':
                    adj_exit = exit_price * (1 - total_cost)
                    pnl_pct = (adj_exit - entry_price) / entry_price
                else:
                    adj_exit = exit_price * (1 + total_cost)
                    pnl_pct = (entry_price - adj_exit) / entry_price

                equity *= (1 + pnl_pct)
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': direction,
                    'pnl_pct': pnl_pct,
                    'exit_type': exit_type,
                    'bars_held': i - entry_bar,
                })
                in_position = False

        else:
            # Check entry signal
            if sig_vals[i] and not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
                raw_price = closes[i]
                atr_at_entry = atr_vals[i]

                if direction == 'LONG':
                    entry_price = raw_price * (1 + total_cost)
                    stop_loss = entry_price - sl_mult * atr_at_entry
                    take_profit = entry_price + tp_mult * atr_at_entry if has_tp else 0.0
                    best_price = entry_price
                else:
                    entry_price = raw_price * (1 - total_cost)
                    stop_loss = entry_price + sl_mult * atr_at_entry
                    take_profit = entry_price - tp_mult * atr_at_entry if has_tp else 0.0
                    best_price = entry_price

                initial_stop = stop_loss
                entry_bar = i
                in_position = True

        equity_curve[i] = equity

    # Force close open position at end
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
        trades.append({
            'entry_bar': entry_bar,
            'exit_bar': len(df) - 1,
            'direction': direction,
            'pnl_pct': pnl_pct,
            'exit_type': 'eod',
            'bars_held': len(df) - 1 - entry_bar,
        })

    return pd.Series(equity_curve, index=df.index), trades


def _combine_equity_curves(curves: List[pd.Series]) -> pd.Series:
    """Combine multiple equity curves by chaining returns."""
    if len(curves) == 1:
        return curves[0]

    combined = []
    running_equity = 100.0

    for curve in curves:
        if len(curve) < 2:
            continue
        scale = running_equity / curve.iloc[0]
        scaled = curve * scale
        combined.append(scaled)
        running_equity = scaled.iloc[-1]

    if not combined:
        return pd.Series([100.0])

    return pd.concat(combined)


def _compute_regime_profit_factors(trades: List[dict], windows: List[pd.DataFrame],
                                   regime_labels: pd.Series) -> Dict[str, float]:
    """Classify trades by regime at entry and compute profit factor per regime."""
    regime_pnl = {'bull': {'win': 0.0, 'loss': 0.0},
                  'bear': {'win': 0.0, 'loss': 0.0},
                  'sideways': {'win': 0.0, 'loss': 0.0}}

    for trade in trades:
        entry_bar = trade['entry_bar']
        regime = 'sideways'
        for window_df in windows:
            if entry_bar < len(window_df):
                try:
                    idx = window_df.index[entry_bar]
                    if idx in regime_labels.index:
                        regime = regime_labels.loc[idx]
                except (IndexError, KeyError):
                    pass
                break

        pnl = trade['pnl_pct']
        if pnl > 0:
            regime_pnl[regime]['win'] += pnl
        else:
            regime_pnl[regime]['loss'] += abs(pnl)

    result = {}
    for regime in ['bull', 'bear', 'sideways']:
        win = regime_pnl[regime]['win']
        loss = regime_pnl[regime]['loss']
        if loss > 0:
            result[regime] = win / loss
        elif win > 0:
            result[regime] = 10.0
        else:
            result[regime] = 1.0

    return result
