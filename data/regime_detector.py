"""
Market Regime Detection.

Classifies each bar as bull, bear, or sideways based on
SMA slope and realized volatility.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_SMA_WINDOW = 100
DEFAULT_VOL_WINDOW = 100
DEFAULT_SLOPE_THRESHOLD = 0.0002  # Minimum abs slope to qualify as trending
DEFAULT_HIGH_VOL_MULT = 1.5       # Volatility multiplier for "high vol" regime


def detect_regime(df: pd.DataFrame,
                  sma_window: int = DEFAULT_SMA_WINDOW,
                  vol_window: int = DEFAULT_VOL_WINDOW,
                  slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
                  high_vol_mult: float = DEFAULT_HIGH_VOL_MULT) -> pd.Series:
    """
    Classify each bar into bull/bear/sideways.

    Method:
    - Compute SMA slope (normalized by price) over sma_window
    - Compute realized volatility over vol_window
    - Bull:     slope > threshold AND vol < high_vol_threshold
    - Bear:     slope < -threshold AND vol < high_vol_threshold
    - Sideways: abs(slope) < threshold OR vol >= high_vol_threshold

    Args:
        df: OHLCV DataFrame with 'Close' column
        sma_window: Window for SMA and slope calculation
        vol_window: Window for volatility calculation
        slope_threshold: Min abs slope for trending classification
        high_vol_mult: Multiplier on median vol to define "high volatility"

    Returns:
        Series with values 'bull', 'bear', 'sideways' (same index as df)
    """
    close = df['Close']

    # SMA and its slope (normalized by price level)
    sma = close.rolling(sma_window).mean()
    slope = sma.diff(5) / sma  # 5-bar slope, normalized

    # Realized volatility (rolling std of log returns)
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(vol_window).std()

    # High volatility threshold: multiplier on median
    median_vol = vol.median()
    high_vol_threshold = median_vol * high_vol_mult

    # Classification
    regime = pd.Series('sideways', index=df.index)

    bull_mask = (slope > slope_threshold) & (vol < high_vol_threshold)
    bear_mask = (slope < -slope_threshold) & (vol < high_vol_threshold)

    regime[bull_mask] = 'bull'
    regime[bear_mask] = 'bear'

    # Fill NaN region at start (before SMA is valid)
    regime.iloc[:sma_window] = 'sideways'

    logger.info(f"Regime detection: bull={bull_mask.sum()}, bear={bear_mask.sum()}, "
                f"sideways={(regime == 'sideways').sum()}")

    return regime


def regime_summary(regime: pd.Series) -> dict:
    """Return counts and proportions of each regime."""
    counts = regime.value_counts()
    total = len(regime)
    return {
        label: {'count': int(counts.get(label, 0)),
                'pct': counts.get(label, 0) / total}
        for label in ['bull', 'bear', 'sideways']
    }
