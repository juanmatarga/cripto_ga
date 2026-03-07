"""
Data loader for BTC/USDT OHLCV data.

Downloads from Binance via ccxt with pagination, caches to parquet.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / 'data_cache'


def load_data(config: dict, use_cache: bool = True) -> pd.DataFrame:
    """
    Load OHLCV data, downloading from Binance if not cached.

    Args:
        config: Config dict with 'data' section
        use_cache: If True, use cached parquet file if available

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns
    """
    data_cfg = config.get('data', {})
    symbol = data_cfg.get('symbol', 'BTC/USDT')
    timeframe = data_cfg.get('timeframe', '15m')
    start = data_cfg.get('start', '2023-01-01')
    end = data_cfg.get('end', '2025-11-21')

    # Cache path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace('/', '_')
    cache_file = CACHE_DIR / f'{safe_symbol}_{timeframe}_{start}_{end}.csv'

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} bars ({df.index.min()} to {df.index.max()})")
        return df

    # Download from Binance
    logger.info(f"Downloading {symbol} {timeframe} from Binance ({start} to {end})...")
    df = _download_from_binance(symbol, timeframe, start, end, data_cfg)

    # Cache
    df.to_csv(cache_file)
    logger.info(f"Cached {len(df)} bars to {cache_file}")

    return df


def _download_from_binance(symbol: str, timeframe: str,
                           start: str, end: str,
                           data_cfg: dict) -> pd.DataFrame:
    """Download OHLCV data from Binance with pagination."""
    market_type = data_cfg.get('market_type', 'future')

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': market_type},
    })

    # Parse dates
    start_dt = pd.Timestamp(start, tz='UTC')
    end_dt = pd.Timestamp(end, tz='UTC')
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    tf_ms = _timeframe_to_ms(timeframe)
    max_candles = 1000
    all_candles = []
    current_ts = start_ts
    batch = 0

    while current_ts < end_ts:
        batch += 1
        for attempt in range(3):
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol, timeframe=timeframe,
                    since=current_ts, limit=max_candles,
                )
                if not ohlcv:
                    current_ts = end_ts  # Stop
                    break

                all_candles.extend(ohlcv)
                current_ts = ohlcv[-1][0] + tf_ms

                if batch % 10 == 0:
                    logger.info(f"  Batch {batch}: {len(all_candles)} candles so far")

                time.sleep(0.1)
                break

            except ccxt.NetworkError as e:
                logger.warning(f"  Batch {batch} attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

        if all_candles and all_candles[-1][0] >= end_ts:
            break

    if not all_candles:
        raise ValueError("No data downloaded from Binance")

    logger.info(f"Downloaded {len(all_candles)} candles in {batch} batches")

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    df = df.sort_index()

    # Validate
    _validate(df)

    return df


def _validate(df: pd.DataFrame):
    """Basic OHLC validation."""
    invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
    invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()

    if invalid_high > 0:
        logger.warning(f"{invalid_high} candles with High < max(O,C) — fixing")
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)

    if invalid_low > 0:
        logger.warning(f"{invalid_low} candles with Low > min(O,C) — fixing")
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"{missing} null values — forward filling")
        df.fillna(method='ffill', inplace=True)

    logger.info(f"Data validation OK: {len(df)} bars, "
                f"{df.index.min()} to {df.index.max()}")


def _timeframe_to_ms(tf: str) -> int:
    return {
        '1m': 60_000, '5m': 300_000, '15m': 900_000,
        '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000,
    }[tf]
