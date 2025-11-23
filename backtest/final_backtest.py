"""
Final Portfolio Backtest with Realistic Futures Position Sizing.

Runs complete backtest on full data for top patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from backtest.futures_position_sizing import FuturesPositionManager
from ga_patterns.evaluator import evaluate_expression

logger = logging.getLogger(__name__)


def run_final_backtest(pattern,
                      data: pd.DataFrame,
                      config: dict,
                      initial_capital: float = 1000.0) -> Dict:
    """
    Run realistic futures backtest for a single pattern.

    Args:
        pattern: PatternChromosome with to_expression() method
        data: Full historical data with OHLCV + indicators
        config: Configuration dict with exits settings
        initial_capital: Starting capital

    Returns:
        Dict with results: {
            'equity_curve': pd.DataFrame,
            'trades': List[Dict],
            'metrics': Dict,
            'pattern': PatternChromosome
        }
    """
    logger.info(f"Running final backtest for: {pattern.to_readable()}")

    # Initialize position manager
    pm = FuturesPositionManager(
        initial_capital=initial_capital,
        risk_per_trade_pct=0.02,
        leverage=10.0
    )

    # Get ATR for stops (14 period default)
    atr_period = config.get('exits', {}).get('atr_period', 14)

    # Calculate ATR if not already present
    if 'ATR' not in data.columns:
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift(1)).abs()
        low_close = (data['Low'] - data['Close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(atr_period).mean()
        logger.debug(f"Calculated ATR with period {atr_period}")

    # Get pattern expression
    expression = pattern.to_expression()
    logger.debug(f"Pattern expression: {expression}")

    # Track trades
    total_signals = 0
    skipped_signals = 0

    for i in range(len(data)):
        if i < atr_period:  # Need ATR
            continue

        current_bar = data.iloc[i]
        timestamp = data.index[i]

        # Check exits for open positions first
        for position in pm.open_positions.copy():
            # Check TP
            if pattern.direction == 'LONG':
                if current_bar['High'] >= position['take_profit_price']:
                    pm.close_position(position, timestamp, position['take_profit_price'], 'TP')
                    continue
                # Check SL
                if current_bar['Low'] <= position['stop_loss_price']:
                    pm.close_position(position, timestamp, position['stop_loss_price'], 'SL')
                    continue
            else:  # SHORT
                if current_bar['Low'] <= position['take_profit_price']:
                    pm.close_position(position, timestamp, position['take_profit_price'], 'TP')
                    continue
                if current_bar['High'] >= position['stop_loss_price']:
                    pm.close_position(position, timestamp, position['stop_loss_price'], 'SL')
                    continue

        # Check entry signal
        try:
            signal = evaluate_expression(expression, data, i)
        except Exception as e:
            logger.debug(f"Error evaluating expression at bar {i}: {e}")
            signal = False

        if signal:
            total_signals += 1

            # Check if we can enter (need margin available)
            if len(pm.open_positions) > 0:
                # For simplicity, only allow 1 position at a time
                skipped_signals += 1
                continue

            entry_price = current_bar['Close']
            atr = current_bar['ATR']

            # Skip if ATR is invalid
            if pd.isna(atr) or atr <= 0:
                skipped_signals += 1
                continue

            # Calculate TP/SL from ATR
            if pattern.direction == 'LONG':
                sl_price = entry_price - (atr * pattern.sl_atr_mult)
                tp_price = entry_price + (atr * pattern.tp_atr_mult)
            else:  # SHORT
                sl_price = entry_price + (atr * pattern.sl_atr_mult)
                tp_price = entry_price - (atr * pattern.tp_atr_mult)

            # Open position
            try:
                pm.open_position(
                    timestamp=timestamp,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                    direction=pattern.direction
                )
            except Exception as e:
                logger.warning(f"Failed to open position at {timestamp}: {e}")
                skipped_signals += 1

    # Close any remaining open positions at final bar
    if pm.open_positions:
        final_bar = data.iloc[-1]
        final_timestamp = data.index[-1]
        final_price = final_bar['Close']

        for position in pm.open_positions.copy():
            logger.info(f"Closing remaining position at end: {position['direction']}")
            pm.close_position(position, final_timestamp, final_price, 'TIME')

    # Get results
    equity_curve = pm.get_equity_curve()
    trades = pm.closed_trades
    metrics = pm.get_metrics()

    # Add signal statistics
    metrics['total_signals'] = total_signals
    metrics['skipped_signals'] = skipped_signals
    metrics['signal_to_trade_ratio'] = metrics['total_trades'] / total_signals if total_signals > 0 else 0

    logger.info(f"Backtest complete: {metrics['total_trades']} trades from {total_signals} signals, "
               f"${metrics['final_equity']:.2f} final equity "
               f"({metrics['total_return_pct']*100:+.1f}%)")

    if metrics['total_trades'] > 0:
        logger.info(f"  Win rate: {metrics['win_rate']*100:.1f}%, "
                   f"Profit factor: {metrics['profit_factor']:.2f}, "
                   f"Max DD: {metrics['max_drawdown_pct']*100:.1f}%")

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'metrics': metrics,
        'pattern': pattern
    }
