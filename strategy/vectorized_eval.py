"""
Vectorized Strategy Evaluation.

v3: Scale-invariant indicators. All indicators are normalized —
no raw price/volume values in comparisons.
"""

import numpy as np
import pandas as pd
import re
import logging
from typing import Dict
from strategy.phenotype import Strategy, Condition

logger = logging.getLogger(__name__)


# ============================================================================
# BASE INDICATOR COMPUTATION
# ============================================================================

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_stoch(df: pd.DataFrame, period: int, component: str) -> pd.Series:
    low_min = df['Low'].rolling(window=period, min_periods=period).min()
    high_max = df['High'].rolling(window=period, min_periods=period).max()
    denom = high_max - low_min
    k = 100 * (df['Close'] - low_min) / denom.replace(0, np.nan)
    if component == 'k':
        return k
    return k.rolling(window=3, min_periods=3).mean()


# ============================================================================
# NORMALIZED INDICATORS (scale-invariant)
# ============================================================================

def compute_pct_b(df: pd.DataFrame, period: int, num_std: float) -> pd.Series:
    """Percent B: position within Bollinger Bands.
    0 = at lower band, 1 = at upper band, 0.5 = at mid.
    Scale-invariant because it's a ratio."""
    close = df['Close']
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = upper - lower
    return (close - lower) / width.replace(0, np.nan)


def compute_macd_norm(df: pd.DataFrame, fast: int, slow: int,
                      signal: int) -> pd.Series:
    """MACD histogram normalized by ATR. Dimensionless ratio.
    Captures momentum relative to volatility — scale-invariant."""
    close = df['Close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    atr = compute_atr(df, period=14)
    return hist / atr.replace(0, np.nan)


def compute_price_pos(df: pd.DataFrame, period: int) -> pd.Series:
    """Price position: (Close - SMA) / ATR. Dimensionless.
    Positive = above SMA, negative = below. Scale-invariant."""
    close = df['Close']
    sma = close.rolling(window=period, min_periods=period).mean()
    atr = compute_atr(df, period=min(period, 14))
    return (close - sma) / atr.replace(0, np.nan)


def compute_roc(df: pd.DataFrame, source: str, period: int) -> pd.Series:
    """Rate of change in percent. Scale-invariant."""
    series = df[source.capitalize()] if source != 'volume' else df['Volume']
    return series.pct_change(periods=period) * 100


def compute_vol_ratio(df: pd.DataFrame, period: int) -> pd.Series:
    """Volume relative to its SMA. Dimensionless ratio.
    >1 = above average volume, <1 = below. Scale-invariant."""
    vol = df['Volume']
    sma = vol.rolling(window=period, min_periods=period).mean()
    return vol / sma.replace(0, np.nan)


def compute_bbwidth(df: pd.DataFrame, period: int, num_std: float) -> pd.Series:
    """Bollinger Band width as % of mid. Scale-invariant.
    High = volatile, Low = compressed. Useful for breakout detection."""
    close = df['Close']
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (upper - lower) / mid.replace(0, np.nan) * 100


def compute_atr_pct(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR as percentage of close price. Scale-invariant.
    Measures volatility regime — high = volatile, low = quiet."""
    atr = compute_atr(df, period)
    return atr / df['Close'].replace(0, np.nan) * 100


def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Average Directional Index (0-100). Measures trend STRENGTH.
    >25 = trending, <20 = ranging. Does not indicate direction."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.copy()
    plus_dm[(up_move <= down_move) | (up_move <= 0)] = 0.0

    minus_dm = down_move.copy()
    minus_dm[(down_move <= up_move) | (down_move <= 0)] = 0.0

    atr = compute_atr(df, period)

    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx.clip(0, 100)


def compute_mfi(df: pd.DataFrame, period: int) -> pd.Series:
    """Money Flow Index (0-100). Volume-weighted RSI.
    Combines price momentum with volume — more reliable than RSI alone."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']

    delta = typical_price.diff()
    pos_flow = money_flow.where(delta > 0, 0.0)
    neg_flow = money_flow.where(delta < 0, 0.0)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    return mfi.clip(0, 100)


# ============================================================================
# INDICATOR CACHE
# ============================================================================

class IndicatorCache:
    """Cache computed indicators to avoid recomputation."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._cache: Dict[str, pd.Series] = {}

    def get(self, indicator_str: str) -> pd.Series:
        if indicator_str in self._cache:
            return self._cache[indicator_str]
        series = self._compute(indicator_str)
        self._cache[indicator_str] = series
        return series

    def _compute(self, indicator_str: str) -> pd.Series:
        s = indicator_str.strip()

        # Raw number → constant series
        try:
            val = float(s)
            return pd.Series(val, index=self.df.index, dtype=np.float64)
        except ValueError:
            pass

        # Parse function call: NAME(arg1, arg2, ...)
        match = re.match(r'^(\w+)\((.+)\)$', s)
        if not match:
            logger.warning(f"Cannot parse indicator: {s}")
            return pd.Series(np.nan, index=self.df.index)

        func_name = match.group(1)
        args_str = match.group(2)
        args = [a.strip() for a in _split_args(args_str)]

        return self._dispatch(func_name, args)

    def _dispatch(self, func: str, args: list) -> pd.Series:
        df = self.df

        # Oscillators (0-100)
        if func == 'RSI':
            source_name = args[0]
            source = df[source_name.capitalize()] if source_name != 'volume' else df['Volume']
            period = int(args[1])
            return compute_rsi(source, period)

        elif func == 'STOCH_K':
            period = int(args[0])
            return compute_stoch(df, period, 'k')

        elif func == 'STOCH_D':
            period = int(args[0])
            return compute_stoch(df, period, 'd')

        elif func == 'ADX':
            period = int(args[0])
            return compute_adx(df, period)

        elif func == 'MFI':
            period = int(args[0])
            return compute_mfi(df, period)

        # Normalized indicators (dimensionless)
        elif func == 'PCT_B':
            period = int(args[0])
            num_std = float(args[1])
            return compute_pct_b(df, period, num_std)

        elif func == 'MACD_NORM':
            fast = int(args[0])
            slow = int(args[1])
            signal = int(args[2])
            return compute_macd_norm(df, fast, slow, signal)

        elif func == 'PRICE_POS':
            period = int(args[0])
            return compute_price_pos(df, period)

        elif func == 'ROC':
            period = int(args[0])
            return compute_roc(df, 'close', period)

        elif func == 'VOL_RATIO':
            period = int(args[0])
            return compute_vol_ratio(df, period)

        elif func == 'BBWIDTH':
            period = int(args[0])
            num_std = float(args[1])
            return compute_bbwidth(df, period, num_std)

        elif func == 'ATR_PCT':
            period = int(args[0])
            return compute_atr_pct(df, period)

        # Legacy support for old indicators (backwards compatibility)
        elif func == 'SMA':
            source_name = args[0]
            source = df[source_name.capitalize()] if source_name != 'volume' else df['Volume']
            period = int(args[1])
            return compute_sma(source, period)

        elif func == 'EMA':
            source_name = args[0]
            source = df[source_name.capitalize()] if source_name != 'volume' else df['Volume']
            period = int(args[1])
            return compute_ema(source, period)

        elif func == 'MACD_HIST':
            fast = int(args[0])
            slow = int(args[1])
            sig = int(args[2])
            close = df['Close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=sig, adjust=False).mean()
            return macd_line - signal_line

        elif func == 'MACD_LINE':
            fast = int(args[0])
            slow = int(args[1])
            close = df['Close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            return ema_fast - ema_slow

        elif func == 'MACD_SIGNAL':
            fast = int(args[0])
            slow = int(args[1])
            sig = int(args[2])
            close = df['Close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            return macd_line.ewm(span=sig, adjust=False).mean()

        elif func == 'ATR':
            period = int(args[0])
            return compute_atr(df, period)

        elif func == 'BBAND_UPPER':
            source_name = args[0]
            source = df[source_name.capitalize()] if source_name != 'volume' else df['Volume']
            period = int(args[1])
            num_std = float(args[2])
            mid = source.rolling(window=period, min_periods=period).mean()
            std = source.rolling(window=period, min_periods=period).std()
            return mid + num_std * std

        elif func == 'BBAND_LOWER':
            source_name = args[0]
            source = df[source_name.capitalize()] if source_name != 'volume' else df['Volume']
            period = int(args[1])
            num_std = float(args[2])
            mid = source.rolling(window=period, min_periods=period).mean()
            std = source.rolling(window=period, min_periods=period).std()
            return mid - num_std * std

        elif func == 'VOLUME_SMA':
            period = int(args[0])
            return compute_sma(df['Volume'], period)

        else:
            logger.warning(f"Unknown indicator function: {func}")
            return pd.Series(np.nan, index=df.index)


def _split_args(args_str: str) -> list:
    """Split function arguments respecting nested parentheses."""
    args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def evaluate_condition(cond: Condition, cache: IndicatorCache) -> pd.Series:
    left = cache.get(cond.left)
    right = cache.get(cond.right)

    if cond.comparator == '>':
        return left > right
    elif cond.comparator == '<':
        return left < right
    elif cond.comparator == 'CROSSES_ABOVE':
        return (left > right) & (left.shift(1) <= right.shift(1))
    elif cond.comparator == 'CROSSES_BELOW':
        return (left < right) & (left.shift(1) >= right.shift(1))
    else:
        logger.warning(f"Unknown comparator: {cond.comparator}")
        return pd.Series(False, index=cache.df.index)


def evaluate_logic(condition_signals: list, logic: str) -> pd.Series:
    if len(condition_signals) == 1:
        return condition_signals[0]

    expr = logic
    ns = {}
    for i, sig in enumerate(condition_signals):
        ns[f"c{i}"] = sig.fillna(False).values

    expr = expr.replace(' AND ', ' & ')
    expr = expr.replace(' OR ', ' | ')

    try:
        result = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
        return pd.Series(result, index=condition_signals[0].index)
    except Exception as e:
        logger.warning(f"Logic evaluation failed: {logic} -> {e}")
        combined = condition_signals[0].fillna(False)
        for sig in condition_signals[1:]:
            combined = combined & sig.fillna(False)
        return combined


def generate_signals(strategy: Strategy, df: pd.DataFrame) -> pd.Series:
    if not strategy.conditions:
        return pd.Series(False, index=df.index)

    cache = IndicatorCache(df)

    condition_signals = []
    for cond in strategy.conditions:
        try:
            sig = evaluate_condition(cond, cache)
            condition_signals.append(sig)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {cond} -> {e}")
            condition_signals.append(pd.Series(False, index=df.index))

    signal = evaluate_logic(condition_signals, strategy.logic)
    signal = signal.fillna(False).astype(bool)

    return signal
