"""
Backtesting Engine - Pattern Evaluation with Position Management
Supports LONG and SHORT with ATR exits and transaction costs
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

from ga_patterns.chromosome import Pattern
from backtest.exits import (
    calculate_atr, calculate_exit_levels,
    check_exit_conditions, calculate_time_exit_bar
)

logger = logging.getLogger(__name__)

def apply_transaction_costs(price: float, direction: str,
                           action: str, config: dict) -> float:
    """
    Apply fees and slippage to a trade price.

    Args:
        price: Raw price before costs
        direction: 'LONG' or 'SHORT'
        action: 'entry' or 'exit'
        config: Config dict with costs section

    Returns:
        Adjusted price after costs

    Example:
        LONG entry at 50000, fees=4bps, slippage=2bps
        - Total cost: 6bps = 0.0006
        - Entry: 50000 * (1 + 0.0006) = 50030

        LONG exit at 51000, fees=4bps, slippage=2bps
        - Exit: 51000 * (1 - 0.0006) = 50969.40

    Notes:
        - Entry: Pay fees + slippage (worse price)
        - Exit: Pay fees + slippage (worse price)
        - LONG entry: price goes UP, exit: price goes DOWN
        - SHORT entry: price goes DOWN, exit: price goes UP
    """
    fees_bps = config['costs'][f'fees_bps_{direction.lower()}']
    slippage_bps = config['costs'][f'slippage_bps_{direction.lower()}']

    total_cost_bps = fees_bps + slippage_bps
    total_cost = total_cost_bps / 10000.0  # Convert basis points to decimal

    if direction == 'LONG':
        if action == 'entry':
            # Buy at higher price (worse fill)
            adjusted_price = price * (1 + total_cost)
        else:  # exit
            # Sell at lower price (worse fill)
            adjusted_price = price * (1 - total_cost)
    elif direction == 'SHORT':
        if action == 'entry':
            # Short at lower price (worse fill)
            adjusted_price = price * (1 - total_cost)
        else:  # exit
            # Cover at higher price (worse fill)
            adjusted_price = price * (1 + total_cost)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return adjusted_price

def generate_signals(pattern: Pattern, data: pd.DataFrame) -> pd.Series:
    """
    Generate trading signals by evaluating pattern on each bar.

    Args:
        pattern: Pattern to evaluate
        data: OHLCV DataFrame

    Returns:
        pd.Series of bool: True = signal triggered, False = no signal

    Notes:
        - Signals are generated bar-by-bar
        - Pattern must have sufficient history (window + max bar_offset)
        - Early bars return False if insufficient data
    """
    signals = pd.Series(False, index=data.index)

    # Calculate minimum bars needed
    min_bars_needed = pattern.window + 20  # Conservative buffer

    for i in range(min_bars_needed, len(data)):
        # Get window of data up to current bar (no lookahead)
        window_data = data.iloc[max(0, i - pattern.window - 20):i + 1].copy()

        try:
            # Evaluate pattern
            signal = pattern.expression.evaluate(window_data)
            signals.iloc[i] = bool(signal)
        except (IndexError, KeyError, ValueError) as e:
            # Insufficient data or evaluation error
            logger.debug(f"Signal generation error at bar {i}: {e}")
            signals.iloc[i] = False

    logger.debug(f"Signals generated: {signals.sum()} signals out of {len(signals)} bars")

    return signals

def run_backtest(pattern: Pattern, data: pd.DataFrame,
                config: dict) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run full backtest for a pattern with position management.

    Args:
        pattern: Pattern to backtest
        data: OHLCV DataFrame (must have OHLCV + index)
        config: Config dict

    Returns:
        (equity_curve, trades_df)
        - equity_curve: pd.Series of portfolio value over time
        - trades_df: pd.DataFrame with trade details

    Algorithm:
        1. Generate signals from pattern
        2. Loop through bars:
           - If no position and signal: ENTER
           - If in position: Check exits (ATR stop/target or time)
           - Track equity
        3. Apply transaction costs
        4. Return equity curve + trades

    Notes:
        - One position at a time (no pyramiding)
        - Position size: 100% of capital
        - Equity starts at 100
    """
    logger.info(f"Running backtest for pattern: {pattern.direction}")

    # Generate signals
    signals = generate_signals(pattern, data)

    # Calculate ATR if using ATR exits
    if config['exits']['use_atr_exits']:
        atr = calculate_atr(data, period=config['exits']['atr_period'])
    else:
        atr = None

    # Initialize tracking
    equity = 100.0
    equity_curve = pd.Series(equity, index=data.index)

    trades = []
    in_position = False
    entry_bar_idx = None
    entry_price = None
    position_size = None
    stop_loss = None
    take_profit = None
    time_exit_bar = None

    # Main backtest loop
    for i in range(len(data)):
        bar = data.iloc[i]
        current_equity = equity

        # Position management
        if in_position:
            # Check exits
            exit_triggered = False
            exit_type = None
            exit_price = None

            # 1. Check ATR exits (stop/target)
            if config['exits']['use_atr_exits']:
                exit_triggered, exit_type, exit_price = check_exit_conditions(
                    bar, stop_loss, take_profit, pattern.direction
                )

            # 2. Check time exit
            if not exit_triggered and config['exits']['use_time_exit']:
                if i >= time_exit_bar:
                    exit_triggered = True
                    exit_type = 'time'
                    exit_price = bar['Close']

            # Execute exit if triggered
            if exit_triggered:
                # Apply costs
                exit_price_adjusted = apply_transaction_costs(
                    exit_price, pattern.direction, 'exit', config
                )

                # Calculate P&L
                if pattern.direction == 'LONG':
                    pnl_pct = (exit_price_adjusted - entry_price) / entry_price
                elif pattern.direction == 'SHORT':
                    pnl_pct = (entry_price - exit_price_adjusted) / entry_price

                equity = equity * (1 + pnl_pct)

                # Record trade
                trades.append({
                    'entry_bar': entry_bar_idx,
                    'entry_date': data.index[entry_bar_idx],
                    'entry_price': entry_price,
                    'exit_bar': i,
                    'exit_date': data.index[i],
                    'exit_price': exit_price_adjusted,
                    'exit_type': exit_type,
                    'direction': pattern.direction,
                    'pnl_pct': pnl_pct,
                    'equity_after': equity
                })

                logger.debug(f"EXIT ({exit_type}): Bar {i}, Price {exit_price_adjusted:.2f}, "
                           f"PnL {pnl_pct:.2%}, Equity {equity:.2f}")

                # Reset position
                in_position = False
                entry_bar_idx = None
                entry_price = None
                position_size = None
                stop_loss = None
                take_profit = None
                time_exit_bar = None

        else:  # Not in position
            # Check for entry signal
            if signals.iloc[i]:
                # Enter position
                entry_bar_idx = i
                raw_entry_price = bar['Close']

                # Apply costs
                entry_price = apply_transaction_costs(
                    raw_entry_price, pattern.direction, 'entry', config
                )

                position_size = equity  # 100% of capital
                in_position = True

                # Calculate exits
                if config['exits']['use_atr_exits']:
                    if pd.notna(atr.iloc[i]):
                        stop_loss, take_profit = calculate_exit_levels(
                            entry_price, atr.iloc[i], pattern.direction, config
                        )
                    else:
                        # No valid ATR - skip this trade
                        logger.warning(f"No valid ATR at bar {i}. Skipping entry.")
                        in_position = False
                        continue

                if config['exits']['use_time_exit']:
                    time_exit_bar = calculate_time_exit_bar(
                        i, config['exits']['bars_hold']
                    )

                logger.debug(f"ENTRY: Bar {i}, Price {entry_price:.2f}, "
                           f"Stop {stop_loss:.2f}, Target {take_profit:.2f}")

        # Update equity curve
        equity_curve.iloc[i] = equity

    # Close any remaining position at end of data
    if in_position:
        exit_price = data.iloc[-1]['Close']
        exit_price_adjusted = apply_transaction_costs(
            exit_price, pattern.direction, 'exit', config
        )

        if pattern.direction == 'LONG':
            pnl_pct = (exit_price_adjusted - entry_price) / entry_price
        elif pattern.direction == 'SHORT':
            pnl_pct = (entry_price - exit_price_adjusted) / entry_price

        equity = equity * (1 + pnl_pct)

        trades.append({
            'entry_bar': entry_bar_idx,
            'entry_date': data.index[entry_bar_idx],
            'entry_price': entry_price,
            'exit_bar': len(data) - 1,
            'exit_date': data.index[-1],
            'exit_price': exit_price_adjusted,
            'exit_type': 'eod',
            'direction': pattern.direction,
            'pnl_pct': pnl_pct,
            'equity_after': equity
        })

        equity_curve.iloc[-1] = equity

        logger.debug(f"EOD EXIT: Final equity {equity:.2f}")

    # Convert trades to DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame(columns=[
            'entry_bar', 'entry_date', 'entry_price',
            'exit_bar', 'exit_date', 'exit_price', 'exit_type',
            'direction', 'pnl_pct', 'equity_after'
        ])

    logger.info(f"[OK] Backtest completed: {len(trades)} trades, "
               f"Final equity: {equity:.2f}")

    return equity_curve, trades_df
