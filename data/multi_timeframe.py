"""
Multi-timeframe data preparation.

Resamples 15m OHLCV to 1h and 4h, computes indicators on each timeframe,
and creates a unified DataFrame aligned to 15m resolution.

Higher timeframe values are forward-filled to 15m bars -- each 15m bar sees
the LAST CLOSED higher-TF value (no lookahead).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Timeframes supported
TIMEFRAMES = ['15m', '1h', '4h']

# Resample rules for core OHLCV columns
OHLCV_RESAMPLE = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
}

# Resample rules for alternative data columns (if present)
ALT_DATA_RESAMPLE = {
    'funding_rate': 'last',
    'open_interest': 'last',
    'taker_ratio': 'mean',
    'ls_ratio': 'mean',
}

# Pandas resample rule strings
_TF_RESAMPLE_RULE = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1D',
}


def resample_ohlcv(df_15m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample 15m OHLCV data to a higher timeframe.

    Args:
        df_15m: 15m OHLCV DataFrame with DatetimeIndex.
        target_tf: Target timeframe string ('1h', '4h', or '1d').

    Returns:
        Resampled DataFrame with same column set as input (OHLCV + any
        alternative data columns that were present).

    Raises:
        ValueError: If target_tf is not supported.
    """
    rule = _TF_RESAMPLE_RULE.get(target_tf)
    if rule is None:
        raise ValueError(
            f"Unsupported timeframe: {target_tf}. "
            f"Supported: {list(_TF_RESAMPLE_RULE.keys())}"
        )

    # Build aggregation dict: OHLCV columns + any alternative data columns
    agg_dict = {}
    for col, func in OHLCV_RESAMPLE.items():
        if col in df_15m.columns:
            agg_dict[col] = func

    for col, func in ALT_DATA_RESAMPLE.items():
        if col in df_15m.columns:
            agg_dict[col] = func

    resampled = df_15m.resample(rule).agg(agg_dict)

    # Drop rows where ALL values are NaN (incomplete bars at boundaries)
    resampled = resampled.dropna(how='all')

    return resampled


def prepare_multi_tf_data(df_15m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare OHLCV data for all timeframes.

    Args:
        df_15m: 15m OHLCV DataFrame with DatetimeIndex.

    Returns:
        Dict mapping timeframe string to OHLCV DataFrame:
        {'15m': df_15m, '1h': df_1h, '4h': df_4h}
    """
    result = {'15m': df_15m}

    for tf in ['1h', '4h']:
        try:
            resampled = resample_ohlcv(df_15m, tf)
            result[tf] = resampled
            logger.debug(f"Resampled to {tf}: {len(resampled)} bars "
                         f"({resampled.index.min()} to {resampled.index.max()})")
        except Exception as e:
            logger.warning(f"Failed to resample to {tf}: {e}")

    return result


def align_higher_tf_to_15m(series_htf: pd.Series,
                            index_15m: pd.DatetimeIndex,
                            tf_label: str) -> pd.Series:
    """
    Align a higher-timeframe indicator series to 15m resolution.

    Uses forward-fill to propagate the last closed HTF value to each 15m bar.
    This avoids lookahead bias -- each 15m bar only sees the previously
    completed HTF candle value.

    Args:
        series_htf: Indicator values at HTF resolution (e.g., RSI on 1h).
        index_15m: The 15m DatetimeIndex to align to.
        tf_label: Timeframe label for the output Series name.

    Returns:
        Series indexed by index_15m, forward-filled from HTF values.
        Leading NaN bars (before the first completed HTF candle) are left as NaN.
    """
    # Shift by 1 so we only see the LAST COMPLETED candle (no lookahead).
    # Without this shift, a 1h candle's value would be visible at its opening
    # bar, but it hasn't closed yet.
    shifted = series_htf.shift(1)

    # Reindex to 15m resolution and forward-fill
    aligned = shifted.reindex(index_15m, method='ffill')

    return aligned


def compute_and_align_indicators(
    df_15m: pd.DataFrame,
    tf_data: Dict[str, pd.DataFrame],
    indicators_fn,
    indicator_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute indicators on all timeframes and create a unified DataFrame.

    For each timeframe, computes indicators using the provided function,
    then aligns higher-TF indicators to 15m resolution. The resulting
    DataFrame has columns like:
      - RSI_14           (15m, no suffix)
      - RSI_14_1h        (1h, aligned to 15m)
      - RSI_14_4h        (4h, aligned to 15m)

    Args:
        df_15m: The 15m OHLCV DataFrame (used for its index).
        tf_data: Dict from prepare_multi_tf_data().
        indicators_fn: Callable(df: DataFrame) -> Dict[str, Series] that
            computes named indicators on a single-TF OHLCV DataFrame.
        indicator_names: Optional list of indicator names to compute.
            If None, all indicators returned by indicators_fn are used.

    Returns:
        Unified DataFrame indexed at 15m resolution with all indicators
        from all timeframes.

    Example:
        >>> def my_indicators(df):
        ...     from strategy.vectorized_eval import compute_rsi, compute_atr_pct
        ...     return {
        ...         'RSI_14': compute_rsi(df['Close'], 14),
        ...         'ATR_PCT_14': compute_atr_pct(df, 14),
        ...     }
        >>> tf_data = prepare_multi_tf_data(df_15m)
        >>> unified = compute_and_align_indicators(df_15m, tf_data, my_indicators)
    """
    index_15m = df_15m.index
    result = pd.DataFrame(index=index_15m)

    for tf in TIMEFRAMES:
        df_tf = tf_data.get(tf)
        if df_tf is None:
            logger.warning(f"No data for timeframe {tf}, skipping")
            continue

        # Compute indicators on this timeframe
        try:
            indicators = indicators_fn(df_tf)
        except Exception as e:
            logger.warning(f"Failed to compute indicators for {tf}: {e}")
            continue

        for name, series in indicators.items():
            if indicator_names is not None and name not in indicator_names:
                continue

            if tf == '15m':
                # Native timeframe -- no alignment needed
                col_name = name
                result[col_name] = series
            else:
                # Higher timeframe -- align to 15m with no-lookahead shift
                col_name = f"{name}_{tf}"
                result[col_name] = align_higher_tf_to_15m(
                    series, index_15m, tf
                )

    n_cols = len(result.columns)
    n_15m = sum(1 for c in result.columns if '_1h' not in c and '_4h' not in c)
    n_1h = sum(1 for c in result.columns if '_1h' in c)
    n_4h = sum(1 for c in result.columns if '_4h' in c)
    logger.info(
        f"Unified multi-TF DataFrame: {n_cols} columns "
        f"({n_15m} x 15m, {n_1h} x 1h, {n_4h} x 4h), "
        f"{len(result)} rows"
    )

    return result


def get_htf_column_name(indicator_name: str, timeframe: str) -> str:
    """
    Get the column name for a higher-timeframe indicator.

    Args:
        indicator_name: Base indicator name (e.g., 'RSI_14').
        timeframe: Timeframe string ('15m', '1h', '4h').

    Returns:
        Column name with suffix for HTF, or bare name for 15m.
        e.g., 'RSI_14' for 15m, 'RSI_14_1h' for 1h.
    """
    if timeframe == '15m':
        return indicator_name
    return f"{indicator_name}_{timeframe}"
