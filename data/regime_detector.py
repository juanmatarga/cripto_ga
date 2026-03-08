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


def detect_regime_with_confidence(df: pd.DataFrame,
                                  sma_window: int = DEFAULT_SMA_WINDOW,
                                  vol_window: int = DEFAULT_VOL_WINDOW,
                                  slope_threshold: float = DEFAULT_SLOPE_THRESHOLD,
                                  high_vol_mult: float = DEFAULT_HIGH_VOL_MULT
                                  ) -> dict:
    """
    Classify current regime with confidence score.

    For live trading: uses the last bar to classify, plus a multi-timeframe
    confirmation approach (short + medium + long SMA slopes must agree).

    Returns:
        {
            'regime': 'bull'/'bear'/'sideways',
            'confidence': float 0-1 (how strongly we believe this regime),
            'slope': float (normalized SMA slope),
            'vol_ratio': float (current vol / median vol),
            'confirmations': int (how many timeframe slopes agree),
        }
    """
    close = df['Close']

    # Multi-scale slope analysis (short, medium, long)
    scales = [50, 100, 200] if len(df) >= 200 else [50, 100]
    slopes = []
    for w in scales:
        if len(df) < w + 5:
            continue
        sma = close.rolling(w).mean()
        s = float((sma.diff(5) / sma).iloc[-1])
        if not np.isnan(s):
            slopes.append(s)

    if not slopes:
        return {'regime': 'sideways', 'confidence': 0.0, 'slope': 0.0,
                'vol_ratio': 1.0, 'confirmations': 0}

    # Primary slope (medium-term)
    primary_slope = slopes[1] if len(slopes) > 1 else slopes[0]

    # Volatility
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(vol_window).std()
    current_vol = float(vol.iloc[-1]) if not np.isnan(vol.iloc[-1]) else 0
    median_vol = float(vol.median()) if not np.isnan(vol.median()) else current_vol
    vol_ratio = current_vol / max(median_vol, 1e-10)
    high_vol = vol_ratio > high_vol_mult

    # Count confirmations: how many scales agree on direction
    n_bull = sum(1 for s in slopes if s > slope_threshold)
    n_bear = sum(1 for s in slopes if s < -slope_threshold)
    n_scales = len(slopes)

    # Determine regime
    if high_vol:
        regime = 'sideways'
        confidence = 0.3  # Low confidence when volatile
    elif n_bull == n_scales:
        regime = 'bull'
        # Confidence scales with slope strength and unanimity
        confidence = min(1.0, abs(primary_slope) / (slope_threshold * 5))
    elif n_bear == n_scales:
        regime = 'bear'
        confidence = min(1.0, abs(primary_slope) / (slope_threshold * 5))
    elif n_bull > n_bear:
        regime = 'bull'
        confidence = 0.3 + 0.3 * (n_bull / n_scales)  # Partial agreement
    elif n_bear > n_bull:
        regime = 'bear'
        confidence = 0.3 + 0.3 * (n_bear / n_scales)
    else:
        regime = 'sideways'
        confidence = 0.5

    return {
        'regime': regime,
        'confidence': round(confidence, 2),
        'slope': round(primary_slope, 6),
        'vol_ratio': round(vol_ratio, 2),
        'confirmations': max(n_bull, n_bear),
    }


def regime_summary(regime: pd.Series) -> dict:
    """Return counts and proportions of each regime."""
    counts = regime.value_counts()
    total = len(regime)
    return {
        label: {'count': int(counts.get(label, 0)),
                'pct': counts.get(label, 0) / total}
        for label in ['bull', 'bear', 'sideways']
    }
