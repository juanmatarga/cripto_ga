"""
Alternative data fetcher for Binance Futures.

Downloads funding rate, open interest, long/short ratios, and taker volume
data. Caches to parquet files. Merges with OHLCV DataFrame.

Data sources:
- Funding rate: ccxt fetch_funding_rate_history() — 8h intervals
- Open interest: Binance fapi /futures/data/openInterestHist — variable periods
- Long/short ratio: Binance fapi /futures/data/globalLongShortAccountRatio
- Taker buy/sell volume: Binance fapi /futures/data/takerlongshortRatio
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ccxt

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / 'data_cache'

# Rate limit delay between paginated requests (seconds)
_RATE_LIMIT_DELAY = 0.2

# Max retries per request
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_symbol(symbol: str) -> str:
    """
    Convert any symbol format to bare Binance format.

    'BTC/USDT'      -> 'BTCUSDT'
    'BTC/USDT:USDT' -> 'BTCUSDT'
    'BTCUSDT'       -> 'BTCUSDT'
    """
    return symbol.replace('/', '').replace(':USDT', '').replace(':usdt', '')


def _ccxt_symbol(symbol: str) -> str:
    """
    Convert to ccxt futures symbol format: 'BTC/USDT:USDT'.
    """
    clean = _clean_symbol(symbol)
    # Expect format like BTCUSDT — split before USDT
    if clean.endswith('USDT'):
        base = clean[:-4]
        return f'{base}/USDT:USDT'
    return symbol


def _ts_ms(date_str: str) -> int:
    """Convert date string to Unix timestamp in milliseconds."""
    dt = pd.Timestamp(date_str, tz='UTC')
    return int(dt.timestamp() * 1000)


def _cache_path(symbol: str, data_type: str, start: str, end: str) -> Path:
    """Generate cache file path for a given data type."""
    clean = _clean_symbol(symbol)
    safe_start = start.replace('-', '')
    safe_end = end.replace('-', '')
    return CACHE_DIR / f'{clean}_alt_{data_type}_{safe_start}_{safe_end}.csv'


def _load_cache(cache_file: Path) -> Optional[pd.DataFrame]:
    """Load cached CSV file if it exists."""
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded cache: {cache_file.name} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Cache file corrupt, re-downloading: {e}")
    return None


def _save_cache(df: pd.DataFrame, cache_file: Path):
    """Save DataFrame to CSV cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file)
    logger.info(f"Cached {len(df)} rows to {cache_file.name}")


def _create_exchange() -> ccxt.binance:
    """Create a ccxt Binance exchange instance configured for futures."""
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })


# ---------------------------------------------------------------------------
# Funding Rate
# ---------------------------------------------------------------------------

def fetch_funding_rate(symbol: str, start: str, end: str,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch funding rate history from Binance.

    Funding is settled every 8h on Binance (00:00, 08:00, 16:00 UTC).

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT' or 'BTCUSDT'
        start: Start date, e.g. '2023-01-01'
        end: End date, e.g. '2025-05-31'
        use_cache: If True, use cached parquet file if available

    Returns:
        DataFrame with 'funding_rate' column indexed by UTC timestamp.
        Empty DataFrame if fetch fails.
    """
    cache_file = _cache_path(symbol, 'funding', start, end)

    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

    logger.info(f"Downloading funding rate for {symbol} ({start} to {end})...")

    try:
        exchange = _create_exchange()
        ccxt_sym = _ccxt_symbol(symbol)

        start_ts = _ts_ms(start)
        end_ts = _ts_ms(end)

        all_records = []
        current_ts = start_ts
        batch = 0

        while current_ts < end_ts:
            batch += 1

            for attempt in range(_MAX_RETRIES):
                try:
                    # ccxt fetch_funding_rate_history returns list of dicts
                    data = exchange.fetch_funding_rate_history(
                        symbol=ccxt_sym,
                        since=current_ts,
                        limit=1000,
                    )

                    if not data:
                        logger.debug(f"Funding rate batch {batch}: no more data")
                        current_ts = end_ts  # Exit loop
                        break

                    all_records.extend(data)

                    # Advance past last record
                    last_ts = data[-1]['timestamp']
                    current_ts = last_ts + 1  # +1ms to avoid overlap

                    if batch % 20 == 0:
                        logger.info(f"  Funding rate batch {batch}: "
                                    f"{len(all_records)} records so far")

                    time.sleep(_RATE_LIMIT_DELAY)
                    break

                except ccxt.NetworkError as e:
                    logger.warning(f"Funding rate batch {batch}, attempt "
                                   f"{attempt + 1}: {e}")
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise

            # Check if we've passed end
            if all_records and all_records[-1]['timestamp'] >= end_ts:
                break

        if not all_records:
            logger.warning("No funding rate data returned")
            return pd.DataFrame(columns=['funding_rate'])

        logger.info(f"Downloaded {len(all_records)} funding rate records "
                    f"in {batch} batches")

        # Build DataFrame
        df = pd.DataFrame(all_records)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Keep only the funding rate column
        if 'fundingRate' in df.columns:
            df = df[['fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
        elif 'rate' in df.columns:
            df = df[['rate']].rename(columns={'rate': 'funding_rate'})
        else:
            # Fallback: look for anything that looks like the rate
            rate_cols = [c for c in df.columns if 'rate' in c.lower()
                         or 'funding' in c.lower()]
            if rate_cols:
                df = df[[rate_cols[0]]].rename(columns={rate_cols[0]: 'funding_rate'})
            else:
                logger.warning(f"Funding rate columns not found in: {df.columns.tolist()}")
                return pd.DataFrame(columns=['funding_rate'])

        df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')

        # Filter to date range
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        _save_cache(df, cache_file)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch funding rate: {e}")
        return pd.DataFrame(columns=['funding_rate'])


# ---------------------------------------------------------------------------
# Open Interest
# ---------------------------------------------------------------------------

def fetch_open_interest(symbol: str, start: str, end: str,
                        period: str = '15m',
                        use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch open interest history from Binance.

    Uses the /futures/data/openInterestHist endpoint.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT' or 'BTCUSDT'
        start: Start date, e.g. '2023-01-01'
        end: End date, e.g. '2025-05-31'
        period: Kline interval — '5m','15m','30m','1h','2h','4h','6h','12h','1d'
        use_cache: If True, use cached parquet file if available

    Returns:
        DataFrame with 'open_interest' (contracts) and 'oi_value' (USDT) columns.
        Empty DataFrame if fetch fails.
    """
    cache_file = _cache_path(symbol, f'oi_{period}', start, end)

    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

    logger.info(f"Downloading open interest for {symbol} ({start} to {end}, "
                f"period={period})...")

    try:
        exchange = _create_exchange()
        clean_sym = _clean_symbol(symbol)

        start_ts = _ts_ms(start)
        end_ts = _ts_ms(end)

        all_records = []
        current_ts = start_ts
        batch = 0
        limit = 500  # Binance max for this endpoint

        while current_ts < end_ts:
            batch += 1

            for attempt in range(_MAX_RETRIES):
                try:
                    # Use the Binance futures data endpoint
                    data = exchange.fapiDataGetOpenInterestHist({
                        'symbol': clean_sym,
                        'period': period,
                        'limit': limit,
                        'startTime': current_ts,
                        'endTime': end_ts,
                    })

                    if not data:
                        logger.debug(f"OI batch {batch}: no more data")
                        current_ts = end_ts
                        break

                    all_records.extend(data)

                    # Advance past last record
                    last_ts = int(data[-1]['timestamp'])
                    current_ts = last_ts + 1

                    if batch % 20 == 0:
                        logger.info(f"  OI batch {batch}: "
                                    f"{len(all_records)} records so far")

                    time.sleep(_RATE_LIMIT_DELAY)
                    break

                except ccxt.NetworkError as e:
                    logger.warning(f"OI batch {batch}, attempt "
                                   f"{attempt + 1}: {e}")
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise

            # If last batch had fewer than limit records, we're done
            if data and len(data) < limit:
                break

        if not all_records:
            logger.warning("No open interest data returned")
            return pd.DataFrame(columns=['open_interest', 'oi_value'])

        logger.info(f"Downloaded {len(all_records)} OI records in {batch} batches")

        # Build DataFrame
        df = pd.DataFrame(all_records)
        df['timestamp'] = pd.to_datetime(
            df['timestamp'].astype(int), unit='ms', utc=True
        )
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Rename columns to standard names
        rename_map = {}
        if 'sumOpenInterest' in df.columns:
            rename_map['sumOpenInterest'] = 'open_interest'
        if 'sumOpenInterestValue' in df.columns:
            rename_map['sumOpenInterestValue'] = 'oi_value'

        df = df.rename(columns=rename_map)

        # Keep only the columns we need
        keep_cols = [c for c in ['open_interest', 'oi_value'] if c in df.columns]
        if not keep_cols:
            logger.warning(f"OI columns not found in: {df.columns.tolist()}")
            return pd.DataFrame(columns=['open_interest', 'oi_value'])

        df = df[keep_cols]
        for col in keep_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter to date range
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        _save_cache(df, cache_file)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch open interest: {e}")
        return pd.DataFrame(columns=['open_interest', 'oi_value'])


# ---------------------------------------------------------------------------
# Long/Short Ratio
# ---------------------------------------------------------------------------

def fetch_long_short_ratio(symbol: str, start: str, end: str,
                           period: str = '15m',
                           use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch global long/short account ratio from Binance.

    Uses the /futures/data/globalLongShortAccountRatio endpoint.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT' or 'BTCUSDT'
        start: Start date, e.g. '2023-01-01'
        end: End date, e.g. '2025-05-31'
        period: Kline interval — '5m','15m','30m','1h','2h','4h','6h','12h','1d'
        use_cache: If True, use cached parquet file if available

    Returns:
        DataFrame with 'ls_ratio', 'long_account', 'short_account' columns.
        Empty DataFrame if fetch fails.
    """
    cache_file = _cache_path(symbol, f'lsratio_{period}', start, end)

    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

    logger.info(f"Downloading long/short ratio for {symbol} ({start} to {end}, "
                f"period={period})...")

    try:
        exchange = _create_exchange()
        clean_sym = _clean_symbol(symbol)

        start_ts = _ts_ms(start)
        end_ts = _ts_ms(end)

        all_records = []
        current_ts = start_ts
        batch = 0
        limit = 500

        while current_ts < end_ts:
            batch += 1

            for attempt in range(_MAX_RETRIES):
                try:
                    data = exchange.fapiDataGetGlobalLongShortAccountRatio({
                        'symbol': clean_sym,
                        'period': period,
                        'limit': limit,
                        'startTime': current_ts,
                        'endTime': end_ts,
                    })

                    if not data:
                        logger.debug(f"L/S ratio batch {batch}: no more data")
                        current_ts = end_ts
                        break

                    all_records.extend(data)

                    last_ts = int(data[-1]['timestamp'])
                    current_ts = last_ts + 1

                    if batch % 20 == 0:
                        logger.info(f"  L/S ratio batch {batch}: "
                                    f"{len(all_records)} records so far")

                    time.sleep(_RATE_LIMIT_DELAY)
                    break

                except ccxt.NetworkError as e:
                    logger.warning(f"L/S ratio batch {batch}, attempt "
                                   f"{attempt + 1}: {e}")
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise

            if data and len(data) < limit:
                break

        if not all_records:
            logger.warning("No long/short ratio data returned")
            return pd.DataFrame(columns=['ls_ratio', 'long_account', 'short_account'])

        logger.info(f"Downloaded {len(all_records)} L/S ratio records "
                    f"in {batch} batches")

        # Build DataFrame
        df = pd.DataFrame(all_records)
        df['timestamp'] = pd.to_datetime(
            df['timestamp'].astype(int), unit='ms', utc=True
        )
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Rename columns
        rename_map = {}
        if 'longShortRatio' in df.columns:
            rename_map['longShortRatio'] = 'ls_ratio'
        if 'longAccount' in df.columns:
            rename_map['longAccount'] = 'long_account'
        if 'shortAccount' in df.columns:
            rename_map['shortAccount'] = 'short_account'

        df = df.rename(columns=rename_map)

        keep_cols = [c for c in ['ls_ratio', 'long_account', 'short_account']
                     if c in df.columns]
        if not keep_cols:
            logger.warning(f"L/S ratio columns not found in: {df.columns.tolist()}")
            return pd.DataFrame(columns=['ls_ratio', 'long_account', 'short_account'])

        df = df[keep_cols]
        for col in keep_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter to date range
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        _save_cache(df, cache_file)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch long/short ratio: {e}")
        return pd.DataFrame(columns=['ls_ratio', 'long_account', 'short_account'])


# ---------------------------------------------------------------------------
# Taker Buy/Sell Volume Ratio
# ---------------------------------------------------------------------------

def fetch_taker_volume(symbol: str, start: str, end: str,
                       period: str = '15m',
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch taker buy/sell volume ratio from Binance.

    Uses the /futures/data/takerlongshortRatio endpoint.

    Args:
        symbol: Trading pair, e.g. 'BTC/USDT' or 'BTCUSDT'
        start: Start date, e.g. '2023-01-01'
        end: End date, e.g. '2025-05-31'
        period: Kline interval — '5m','15m','30m','1h','2h','4h','6h','12h','1d'
        use_cache: If True, use cached parquet file if available

    Returns:
        DataFrame with 'taker_buy_sell_ratio', 'buy_vol', 'sell_vol' columns.
        Empty DataFrame if fetch fails.
    """
    cache_file = _cache_path(symbol, f'taker_{period}', start, end)

    if use_cache:
        cached = _load_cache(cache_file)
        if cached is not None:
            return cached

    logger.info(f"Downloading taker volume for {symbol} ({start} to {end}, "
                f"period={period})...")

    try:
        exchange = _create_exchange()
        clean_sym = _clean_symbol(symbol)

        start_ts = _ts_ms(start)
        end_ts = _ts_ms(end)

        all_records = []
        current_ts = start_ts
        batch = 0
        limit = 500

        while current_ts < end_ts:
            batch += 1

            for attempt in range(_MAX_RETRIES):
                try:
                    data = exchange.fapiDataGetTakerlongshortRatio({
                        'symbol': clean_sym,
                        'period': period,
                        'limit': limit,
                        'startTime': current_ts,
                        'endTime': end_ts,
                    })

                    if not data:
                        logger.debug(f"Taker vol batch {batch}: no more data")
                        current_ts = end_ts
                        break

                    all_records.extend(data)

                    last_ts = int(data[-1]['timestamp'])
                    current_ts = last_ts + 1

                    if batch % 20 == 0:
                        logger.info(f"  Taker vol batch {batch}: "
                                    f"{len(all_records)} records so far")

                    time.sleep(_RATE_LIMIT_DELAY)
                    break

                except ccxt.NetworkError as e:
                    logger.warning(f"Taker vol batch {batch}, attempt "
                                   f"{attempt + 1}: {e}")
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise

            if data and len(data) < limit:
                break

        if not all_records:
            logger.warning("No taker volume data returned")
            return pd.DataFrame(
                columns=['taker_buy_sell_ratio', 'buy_vol', 'sell_vol']
            )

        logger.info(f"Downloaded {len(all_records)} taker volume records "
                    f"in {batch} batches")

        # Build DataFrame
        df = pd.DataFrame(all_records)
        df['timestamp'] = pd.to_datetime(
            df['timestamp'].astype(int), unit='ms', utc=True
        )
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # Rename columns
        rename_map = {}
        if 'buySellRatio' in df.columns:
            rename_map['buySellRatio'] = 'taker_buy_sell_ratio'
        if 'buyVol' in df.columns:
            rename_map['buyVol'] = 'buy_vol'
        if 'sellVol' in df.columns:
            rename_map['sellVol'] = 'sell_vol'

        df = df.rename(columns=rename_map)

        keep_cols = [c for c in ['taker_buy_sell_ratio', 'buy_vol', 'sell_vol']
                     if c in df.columns]
        if not keep_cols:
            logger.warning(f"Taker vol columns not found in: {df.columns.tolist()}")
            return pd.DataFrame(
                columns=['taker_buy_sell_ratio', 'buy_vol', 'sell_vol']
            )

        df = df[keep_cols]
        for col in keep_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Filter to date range
        start_dt = pd.Timestamp(start, tz='UTC')
        end_dt = pd.Timestamp(end, tz='UTC')
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]

        _save_cache(df, cache_file)
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch taker volume: {e}")
        return pd.DataFrame(
            columns=['taker_buy_sell_ratio', 'buy_vol', 'sell_vol']
        )


# ---------------------------------------------------------------------------
# Merge all alternative data into OHLCV DataFrame
# ---------------------------------------------------------------------------

def merge_alternative_data(df: pd.DataFrame, symbol: str,
                           start: str, end: str,
                           use_cache: bool = True) -> pd.DataFrame:
    """
    Merge alternative data into the main OHLCV DataFrame.

    Currently only funding rate has full historical data on Binance.
    OI, L/S ratio, and taker volume are limited to ~30 days and thus
    not useful for evolution (2022-2025).

    Added columns:
    - funding_rate: Raw funding rate (forward-filled from 8h to 15m)

    Args:
        df: Main OHLCV DataFrame (DatetimeIndex, UTC or naive)
        symbol: Trading pair, e.g. 'BTC/USDT'
        start: Start date for alternative data
        end: End date for alternative data
        use_cache: If True, use cached files

    Returns:
        Copy of df with funding_rate column added.
        Original df is not modified.
    """
    result = df.copy()

    # --- Funding rate (8h intervals, needs forward-fill to 15m) ---
    funding_df = fetch_funding_rate(symbol, start, end, use_cache=use_cache)
    if not funding_df.empty and 'funding_rate' in funding_df.columns:
        # Strip timezone from funding_df to match OHLCV (which may be naive)
        if funding_df.index.tz is not None and result.index.tz is None:
            funding_df.index = funding_df.index.tz_localize(None)
        elif funding_df.index.tz is None and result.index.tz is not None:
            funding_df.index = funding_df.index.tz_localize(result.index.tz)

        # Use merge_asof for robust alignment
        funding_aligned = pd.merge_asof(
            result[[]],  # Just the index
            funding_df.reset_index(),
            left_index=True,
            right_on=funding_df.index.name or 'index',
            direction='backward',
        )
        result['funding_rate'] = funding_aligned['funding_rate'].values
        n_valid = result['funding_rate'].notna().sum()
        logger.info(f"Merged funding rate: {n_valid}/{len(result)} non-null values")
    else:
        result['funding_rate'] = np.nan
        logger.warning("Funding rate: no data available, column filled with NaN")

    return result
