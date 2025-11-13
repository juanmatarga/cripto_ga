"""
Exit Management - ATR-Based Stops and Targets
Designed for crypto futures with intra-bar detection
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR = SMA(TR, period)
    TR = max(H-L, |H-C_prev|, |L-C_prev|)

    Args:
        data: DataFrame with OHLCV (must have High, Low, Close)
        period: ATR lookback period (default 14)

    Returns:
        pd.Series: ATR values (same index as data)

    Notes:
        - First `period` values will be NaN
        - Uses exponential moving average for smoothing
    """
    if len(data) < period:
        logger.warning(f"Data length ({len(data)}) < ATR period ({period}). ATR will be mostly NaN.")

    # True Range components
    high_low = data['High'] - data['Low']
    high_close_prev = (data['High'] - data['Close'].shift(1)).abs()
    low_close_prev = (data['Low'] - data['Close'].shift(1)).abs()

    # True Range = max of the three
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # ATR = EMA of TR
    atr = true_range.ewm(span=period, adjust=False).mean()

    logger.debug(f"ATR calculated: period={period}, mean={atr.mean():.4f}, median={atr.median():.4f}")

    return atr

def calculate_exit_levels(entry_price: float, atr_value: float,
                         direction: str, config: dict,
                         tp_mult: float = None, sl_mult: float = None) -> Tuple[float, float]:
    """
    Calculate stop loss and take profit levels based on ATR.

    SPRINT 12: Now accepts pattern-specific TP/SL multipliers.

    Args:
        entry_price: Entry price for the position
        atr_value: Current ATR value
        direction: 'LONG' or 'SHORT'
        config: Config dict with exits section
        tp_mult: Pattern-specific take profit multiplier (SPRINT 12)
        sl_mult: Pattern-specific stop loss multiplier (SPRINT 12)

    Returns:
        (stop_loss, take_profit)

    Example:
        LONG entry at 50000, ATR=500, sl_mult=1.5, tp_mult=3.0
        - Stop:  50000 - 1.5*500 = 49250
        - Target: 50000 + 3.0*500 = 51500

    Notes:
        - Returns None if ATR is NaN or invalid
        - Stop is always closer to entry than target (risk/reward)
        - Uses pattern multipliers if provided, else falls back to config
    """
    if pd.isna(atr_value) or atr_value <= 0:
        logger.warning(f"Invalid ATR value: {atr_value}. Cannot calculate exits.")
        return None, None

    # SPRINT 12: Use pattern-specific multipliers if provided
    if sl_mult is not None and tp_mult is not None:
        atr_stop_mult = sl_mult
        atr_take_mult = tp_mult
        logger.debug(f"Using pattern TP/SL: sl={sl_mult:.2f}, tp={tp_mult:.2f}")
    else:
        # Fallback to config
        atr_stop_mult = config['exits']['atr_stop']
        atr_take_mult = config['exits']['atr_take']
        logger.debug(f"Using config TP/SL: sl={atr_stop_mult:.2f}, tp={atr_take_mult:.2f}")

    if direction == 'LONG':
        stop_loss = entry_price - (atr_stop_mult * atr_value)
        take_profit = entry_price + (atr_take_mult * atr_value)
    elif direction == 'SHORT':
        stop_loss = entry_price + (atr_stop_mult * atr_value)
        take_profit = entry_price - (atr_take_mult * atr_value)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'.")

    logger.debug(f"{direction} exits calculated: Entry={entry_price:.2f}, "
                f"Stop={stop_loss:.2f}, Target={take_profit:.2f}, ATR={atr_value:.2f}")

    return stop_loss, take_profit

def check_exit_conditions(bar: pd.Series, stop_loss: float, take_profit: float,
                         direction: str) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Check if a bar triggered stop or target (intra-bar detection).

    Uses High/Low to detect intra-bar exits. Assumes stop is hit before target
    if both are triggered in the same bar (conservative approach).

    Args:
        bar: Single bar with High, Low, Close
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        direction: 'LONG' or 'SHORT'

    Returns:
        (exit_triggered, exit_type, exit_price)
        - exit_triggered: True if stop or target hit
        - exit_type: 'stop' or 'target' or None
        - exit_price: Actual exit price or None

    Example:
        LONG position: stop=49250, target=51500
        Bar: High=51600, Low=49800
        - Both levels touched, but stop hit first (conservative)
        - Returns (True, 'stop', 49250)

    Notes:
        - Conservative: Stop takes priority if both triggered
        - Uses actual High/Low for realistic fills
        - Does NOT assume mid-bar execution
    """
    high = bar['High']
    low = bar['Low']
    close = bar['Close']

    if direction == 'LONG':
        # LONG: Stop below, Target above
        stop_hit = low <= stop_loss
        target_hit = high >= take_profit

        if stop_hit and target_hit:
            # Both hit - conservative: assume stop first
            logger.debug(f"LONG: Both stop and target hit in same bar. Using stop (conservative).")
            return True, 'stop', stop_loss
        elif stop_hit:
            return True, 'stop', stop_loss
        elif target_hit:
            return True, 'target', take_profit
        else:
            return False, None, None

    elif direction == 'SHORT':
        # SHORT: Stop above, Target below
        stop_hit = high >= stop_loss
        target_hit = low <= take_profit

        if stop_hit and target_hit:
            # Both hit - conservative: assume stop first
            logger.debug(f"SHORT: Both stop and target hit in same bar. Using stop (conservative).")
            return True, 'stop', stop_loss
        elif stop_hit:
            return True, 'stop', stop_loss
        elif target_hit:
            return True, 'target', take_profit
        else:
            return False, None, None
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'LONG' or 'SHORT'.")

def calculate_time_exit_bar(entry_bar_index: int, bars_hold: int) -> int:
    """
    Calculate the bar index for time-based exit.

    Args:
        entry_bar_index: Index of entry bar
        bars_hold: Number of bars to hold position

    Returns:
        Exit bar index

    Example:
        Entry at bar 100, hold 96 bars → exit at bar 196
    """
    return entry_bar_index + bars_hold
