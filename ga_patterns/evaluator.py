"""
Expression Evaluator for Building Blocks

This module evaluates module expressions on real OHLCV data.
It's the bridge between string expressions and boolean results.

Critical challenge: Convert expressions like "C[0] > C[1] AND V[0] > V[5]"
into actual boolean evaluation on DataFrame.

Approach: Token replacement + safe eval
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def evaluate_expression(expression: str, data: pd.DataFrame, bar_index: int) -> bool:
    """
    Evaluate expression at specific bar in DataFrame.

    Args:
        expression: String like "(C[0] > C[1]) AND (V[0] > V[5])"
        data: DataFrame with OHLCV + indicators
        bar_index: Index of bar to evaluate (must have enough lookback)

    Returns:
        bool: True if expression evaluates to True

    Process:
        1. Check bar_index has enough lookback (max offset in expression)
        2. Replace all tokens (C[n], V[n], etc.) with actual values
        3. Replace 'AND' → 'and', 'OR' → 'or' (Python syntax)
        4. Safely evaluate expression

    Example:
        >>> # Assuming data is preprocessed DataFrame
        >>> expr = "C[0] > C[1] AND V[0] > V[5]"
        >>> result = evaluate_expression(expr, data, 100)
        >>> isinstance(result, bool)
        True
    """
    # Extract max offset needed
    max_offset = extract_max_offset(expression)

    if bar_index < max_offset:
        logger.debug(f"Insufficient lookback at bar {bar_index} (need {max_offset})")
        return False

    # Replace tokens
    parsed_expr = expression

    # Find all tokens like C[n], V[n], RSI[14][n], etc.
    tokens = find_all_tokens(expression)

    # CRITICAL FIX: Sort tokens by length (longest first) to avoid substring replacement issues
    # Example: Must replace "SMA_V[20][0]" before "V[0]" to avoid "SMA_8100.627[20][0]"
    tokens = sorted(tokens, key=len, reverse=True)

    for token in tokens:
        try:
            value = parse_token(token, data, bar_index)

            # FIX: Handle NaN values properly (don't convert to string "nan")
            if pd.isna(value) or np.isnan(value):
                # If any indicator is NaN (insufficient data), pattern should not trigger
                logger.debug(f"Token '{token}' is NaN at bar {bar_index} - returning False")
                return False

            # Replace token with its value
            parsed_expr = parsed_expr.replace(token, str(value))
        except Exception as e:
            logger.error(f"Error parsing token '{token}': {e}")
            return False

    # Replace logical operators
    parsed_expr = parsed_expr.replace(' AND ', ' and ')
    parsed_expr = parsed_expr.replace(' OR ', ' or ')

    # Safely evaluate
    try:
        result = eval(parsed_expr)
        return bool(result)
    except Exception as e:
        logger.error(f"Error evaluating '{parsed_expr}': {e}")
        return False


def find_all_tokens(expression: str) -> List[str]:
    """
    Find all tokens in expression that need replacement.

    Args:
        expression: Expression string

    Returns:
        List of unique tokens in order of appearance

    Patterns to match:
        - C[n], O[n], H[n], L[n], V[n]
        - Body%[n], Range%[n], ClosePos[n]
        - RSI[period][n]
        - SMA[period][n], SMA_V[period][n]
        - MACD[n], Signal[n], MACDHist[n]
        - BB_Upper[n], BB_Lower[n], BB_Width[n]
        - BB_Width_SMA[period][n]
        - ATR[period][n], ATR_SMA[period][n]
        - Stoch_K[n], Stoch_D[n]

    Example:
        >>> expr = "C[0] > C[1] AND V[5] > RSI[14][0]"
        >>> tokens = find_all_tokens(expr)
        >>> tokens
        ['C[0]', 'C[1]', 'V[5]', 'RSI[14][0]']
    """
    patterns = [
        r'[COHLV]\[\d+\]',                      # C[0], V[5]
        r'Body%\[\d+\]',                         # Body%[0]
        r'Range%\[\d+\]',                        # Range%[0]
        r'ClosePos\[\d+\]',                      # ClosePos[0]
        r'RSI\[\d+\]\[\d+\]',                    # RSI[14][0]
        r'SMA(?:_V)?\[\d+\]\[\d+\]',            # SMA[20][0], SMA_V[20][0]
        r'MACD(?:Hist)?\[\d+\]',                 # MACD[0], MACDHist[0]
        r'Signal\[\d+\]',                        # Signal[0]
        r'BB_(?:Upper|Lower|Width)\[\d+\]',     # BB_Upper[0]
        r'BB_Width_SMA\[\d+\]\[\d+\]',          # BB_Width_SMA[20][0]
        r'ATR(?:_SMA)?\[\d+\]\[\d+\]',          # ATR[14][0], ATR_SMA[20][0]
        r'Stoch_[KD]\[\d+\]',                    # Stoch_K[0], Stoch_D[0]
    ]

    tokens = []
    for pattern in patterns:
        matches = re.findall(pattern, expression)
        tokens.extend(matches)

    # Return unique while preserving order
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    return unique_tokens


def parse_token(token: str, data: pd.DataFrame, bar_index: int) -> float:
    """
    Convert token to actual numeric value from DataFrame.

    Args:
        token: Token like 'C[0]', 'RSI[14][0]', etc.
        data: DataFrame with required columns
        bar_index: Current bar index

    Returns:
        float: Numeric value

    Example:
        >>> # Assuming data is preprocessed DataFrame
        >>> value = parse_token('C[0]', data, 100)
        >>> isinstance(value, (int, float))
        True
    """
    # Basic OHLCV
    if token.startswith('C['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['Close'])

    elif token.startswith('O['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['Open'])

    elif token.startswith('H['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['High'])

    elif token.startswith('L['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['Low'])

    elif token.startswith('V['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['Volume'])

    # Derived OHLCV
    elif token.startswith('Body%['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['body_pct'])

    elif token.startswith('Range%['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['range_pct'])

    elif token.startswith('ClosePos['):
        offset = extract_offset(token)
        return float(data.iloc[bar_index - offset]['close_position_in_range'])

    # Indicators (require preprocessing)
    elif token.startswith('RSI['):
        # Extract: RSI[14][0] → period=14, offset=0
        period, offset = extract_indicator_params(token)
        column_name = f'RSI_{period}'
        if column_name not in data.columns:
            raise ValueError(f"Missing indicator column: {column_name}")
        return float(data.iloc[bar_index - offset][column_name])

    elif token.startswith('SMA_V['):
        # SMA_V[20][0] → period=20, offset=0, apply to Volume
        period, offset = extract_indicator_params(token)
        column_name = f'SMA_V_{period}'
        if column_name not in data.columns:
            raise ValueError(f"Missing indicator column: {column_name}")
        return float(data.iloc[bar_index - offset][column_name])

    elif token.startswith('SMA['):
        # SMA[20][0] → period=20, offset=0, apply to Close
        period, offset = extract_indicator_params(token)
        column_name = f'SMA_{period}'
        if column_name not in data.columns:
            raise ValueError(f"Missing indicator column: {column_name}")
        return float(data.iloc[bar_index - offset][column_name])

    elif token.startswith('MACD['):
        offset = extract_offset(token)
        if 'MACD' not in data.columns:
            raise ValueError("Missing MACD column")
        return float(data.iloc[bar_index - offset]['MACD'])

    elif token.startswith('Signal['):
        offset = extract_offset(token)
        if 'MACD_signal' not in data.columns:
            raise ValueError("Missing MACD_signal column")
        return float(data.iloc[bar_index - offset]['MACD_signal'])

    elif token.startswith('MACDHist['):
        offset = extract_offset(token)
        if 'MACD_hist' not in data.columns:
            raise ValueError("Missing MACD_hist column")
        return float(data.iloc[bar_index - offset]['MACD_hist'])

    # Bollinger Bands
    elif token.startswith('BB_Upper['):
        offset = extract_offset(token)
        if 'BB_Upper' not in data.columns:
            raise ValueError("Missing BB_Upper column")
        return float(data.iloc[bar_index - offset]['BB_Upper'])

    elif token.startswith('BB_Lower['):
        offset = extract_offset(token)
        if 'BB_Lower' not in data.columns:
            raise ValueError("Missing BB_Lower column")
        return float(data.iloc[bar_index - offset]['BB_Lower'])

    elif token.startswith('BB_Width['):
        offset = extract_offset(token)
        if 'BB_Width' not in data.columns:
            raise ValueError("Missing BB_Width column")
        return float(data.iloc[bar_index - offset]['BB_Width'])

    elif token.startswith('BB_Width_SMA['):
        period, offset = extract_indicator_params(token)
        column_name = f'BB_Width_SMA_{period}'
        if column_name not in data.columns:
            raise ValueError(f"Missing indicator column: {column_name}")
        return float(data.iloc[bar_index - offset][column_name])

    # ATR
    elif token.startswith('ATR_SMA['):
        period, offset = extract_indicator_params(token)
        column_name = f'ATR_SMA_{period}'
        if column_name not in data.columns:
            raise ValueError(f"Missing indicator column: {column_name}")
        return float(data.iloc[bar_index - offset][column_name])

    elif token.startswith('ATR['):
        period, offset = extract_indicator_params(token)
        column_name = f'ATR_{period}'
        if column_name not in data.columns:
            raise ValueError(f"Missing indicator column: {column_name}")
        return float(data.iloc[bar_index - offset][column_name])

    # Stochastic
    elif token.startswith('Stoch_K['):
        offset = extract_offset(token)
        if 'Stoch_K' not in data.columns:
            raise ValueError("Missing Stoch_K column")
        return float(data.iloc[bar_index - offset]['Stoch_K'])

    elif token.startswith('Stoch_D['):
        offset = extract_offset(token)
        if 'Stoch_D' not in data.columns:
            raise ValueError("Missing Stoch_D column")
        return float(data.iloc[bar_index - offset]['Stoch_D'])

    else:
        raise ValueError(f"Unknown token: {token}")


def extract_offset(token: str) -> int:
    """
    Extract offset from simple token like 'C[5]'.

    Args:
        token: Token like 'C[5]', 'V[0]'

    Returns:
        int: Offset value

    Example:
        >>> extract_offset('C[5]')
        5
        >>> extract_offset('V[0]')
        0
    """
    # Match pattern: [number]
    match = re.search(r'\[(\d+)\]', token)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract offset from token: {token}")


def extract_indicator_params(token: str) -> Tuple[int, int]:
    """
    Extract (period, offset) from indicator token.

    Args:
        token: Token like 'RSI[14][0]', 'SMA[20][3]'

    Returns:
        Tuple[int, int]: (period, offset)

    Example:
        >>> extract_indicator_params('RSI[14][0]')
        (14, 0)
        >>> extract_indicator_params('SMA[20][3]')
        (20, 3)
    """
    # Match pattern: [period][offset]
    matches = re.findall(r'\[(\d+)\]', token)
    if len(matches) >= 2:
        period = int(matches[0])
        offset = int(matches[1])
        return period, offset
    else:
        raise ValueError(f"Cannot extract indicator params from token: {token}")


def extract_max_offset(expression: str) -> int:
    """
    Find maximum offset in expression (determines minimum lookback).

    Args:
        expression: Expression string

    Returns:
        int: Maximum offset found

    Example:
        >>> extract_max_offset("C[0] > C[1] AND V[5] > V[10]")
        10
        >>> extract_max_offset("RSI[14][0] < 30 AND C[2] > C[3]")
        3
    """
    # Find all numbers in brackets
    all_offsets = re.findall(r'\[(\d+)\]', expression)

    if not all_offsets:
        return 0

    # Convert to integers and return max
    return max(int(offset) for offset in all_offsets)


def preprocess_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all required indicator columns to DataFrame.

    This should be called ONCE on the full dataset before any evaluation.

    Indicators to add:
        - body_pct: abs(Close - Open) / Close
        - range_pct: (High - Low) / Close
        - close_position_in_range: (Close - Low) / (High - Low)
        - RSI_14: 14-period RSI
        - SMA_20, SMA_50: Simple moving averages of Close
        - SMA_V_20: Simple moving average of Volume
        - MACD, MACD_signal, MACD_hist: MACD components (12, 26, 9)
        - BB_Upper, BB_Lower, BB_Width: Bollinger Bands (20, 2)
        - BB_Width_SMA_20: SMA of BB width
        - ATR_14: 14-period Average True Range
        - ATR_SMA_20: SMA of ATR
        - Stoch_K, Stoch_D: Stochastic oscillator (14, 3)

    Args:
        data: DataFrame with OHLCV columns

    Returns:
        DataFrame with all indicator columns added

    Example:
        >>> data = pd.DataFrame({
        ...     'Open': [...],
        ...     'High': [...],
        ...     'Low': [...],
        ...     'Close': [...],
        ...     'Volume': [...]
        ... })
        >>> data = preprocess_indicators(data)
        >>> 'RSI_14' in data.columns
        True
    """
    logger.info("Preprocessing indicators...")

    # Make a copy to avoid modifying original
    data = data.copy()

    # Basic features
    logger.debug("  Computing basic features...")
    data['body_pct'] = abs(data['Close'] - data['Open']) / data['Close']
    data['range_pct'] = (data['High'] - data['Low']) / data['Close']

    # Close position in range
    range_val = data['High'] - data['Low']
    data['close_position_in_range'] = np.where(
        range_val != 0,
        (data['Close'] - data['Low']) / range_val,
        0.5  # Default to middle if range is 0
    )

    # Try to use talib if available, otherwise use pandas
    try:
        import talib

        logger.debug("  Using TA-Lib for indicators...")

        # RSI
        data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)

        # Moving averages
        data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
        data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        data['SMA_V_20'] = talib.SMA(data['Volume'], timeperiod=20)

        # MACD
        macd, signal, hist = talib.MACD(data['Close'],
                                        fastperiod=12,
                                        slowperiod=26,
                                        signalperiod=9)
        data['MACD'] = macd
        data['MACD_signal'] = signal
        data['MACD_hist'] = hist

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(data['Close'],
                                           timeperiod=20,
                                           nbdevup=2,
                                           nbdevdn=2)
        data['BB_Upper'] = upper
        data['BB_Lower'] = lower
        data['BB_Width'] = upper - lower
        data['BB_Width_SMA_20'] = talib.SMA(data['BB_Width'], timeperiod=20)

        # ATR
        data['ATR_14'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['ATR_SMA_20'] = talib.SMA(data['ATR_14'], timeperiod=20)

        # Stochastic
        slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'],
                                   fastk_period=14,
                                   slowk_period=3,
                                   slowk_matype=0,
                                   slowd_period=3,
                                   slowd_matype=0)
        data['Stoch_K'] = slowk
        data['Stoch_D'] = slowd

    except ImportError:
        logger.warning("TA-Lib not available, using pandas implementations...")

        # RSI using pandas
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        data['RSI_14'] = compute_rsi(data['Close'], 14)

        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_V_20'] = data['Volume'].rolling(window=20).mean()

        # MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']

        # Bollinger Bands
        sma20 = data['Close'].rolling(window=20).mean()
        std20 = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = sma20 + (std20 * 2)
        data['BB_Lower'] = sma20 - (std20 * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Width_SMA_20'] = data['BB_Width'].rolling(window=20).mean()

        # ATR
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR_14'] = true_range.rolling(window=14).mean()
        data['ATR_SMA_20'] = data['ATR_14'].rolling(window=20).mean()

        # Stochastic
        low_min = data['Low'].rolling(window=14).min()
        high_max = data['High'].rolling(window=14).max()
        data['Stoch_K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        data['Stoch_K'] = data['Stoch_K'].rolling(window=3).mean()  # Slow K
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

    logger.info(f"  Added {len(data.columns) - 5} indicator columns")
    logger.debug(f"  Total columns: {len(data.columns)}")

    return data


if __name__ == '__main__':
    # Test with sample data
    print("="*70)
    print("TESTING EVALUATOR")
    print("="*70)

    # Test token extraction
    print("\nTest 1: Token extraction")
    expr1 = "C[0] > C[1] AND V[5] > V[10]"
    tokens1 = find_all_tokens(expr1)
    print(f"  Expression: {expr1}")
    print(f"  Tokens: {tokens1}")
    print(f"  Max offset: {extract_max_offset(expr1)}")

    expr2 = "RSI[14][0] < 30 AND MACD[0] > Signal[0]"
    tokens2 = find_all_tokens(expr2)
    print(f"\n  Expression: {expr2}")
    print(f"  Tokens: {tokens2}")
    print(f"  Max offset: {extract_max_offset(expr2)}")

    # Test extract functions
    print("\nTest 2: Extract functions")
    print(f"  extract_offset('C[5]'): {extract_offset('C[5]')}")
    print(f"  extract_indicator_params('RSI[14][0]'): {extract_indicator_params('RSI[14][0]')}")
    print(f"  extract_indicator_params('SMA[20][3]'): {extract_indicator_params('SMA[20][3]')}")

    print(f"\n{'='*70}\n")
