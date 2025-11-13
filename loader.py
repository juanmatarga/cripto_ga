"""
Binance Data Loader - OHLCV with pagination
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_binance_data(config: dict) -> pd.DataFrame:
    """
    Carga datos OHLCV desde Binance con paginación automática.

    Args:
        config: dict completo del config.yaml

    Returns:
        pd.DataFrame con columnas: Open, High, Low, Close, Volume
        Index: DatetimeIndex (timezone UTC)
        Attrs: metadata (symbol, timeframe, etc.)

    Raises:
        ValueError: Si datos incompletos o inconsistentes
        ccxt.NetworkError: Si falla conexión con Binance

    Features:
    - Paginación automática (>1000 barras)
    - Rate limiting (100ms entre requests)
    - Retry logic (3 intentos)
    - Validación OHLC consistency
    - Timezone UTC normalizado
    """
    data_config = config['data']
    pagination_config = data_config['pagination']

    # Inicializar exchange
    exchange_id = data_config['exchange']
    logger.info(f"Initializing {exchange_id} exchange...")

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': data_config['market_type'],  # 'future' o 'spot'
        }
    })

    # Parsear fechas
    start_dt = datetime.strptime(data_config['start'], '%Y-%m-%d %H:%M:%S')
    end_dt = datetime.strptime(data_config['end'], '%Y-%m-%d %H:%M:%S')

    # Convertir a timestamps (milisegundos)
    start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

    symbol = data_config['symbol']
    timeframe = data_config['timeframe']

    logger.info(f"Loading {symbol} {timeframe} from {data_config['start']} to {data_config['end']}")
    logger.info(f"Market type: {data_config['market_type']}")

    # PAGINACIÓN: Cargar en batches de max_candles_per_request
    all_candles = []
    current_ts = start_ts
    max_candles = pagination_config['max_candles_per_request']
    rate_limit_delay = pagination_config['rate_limit_delay_ms'] / 1000.0  # ms -> segundos
    max_retries = pagination_config['max_retries']

    batch_num = 0

    while current_ts < end_ts:
        batch_num += 1

        # Intentar cargar batch con retries
        for attempt in range(max_retries):
            try:
                logger.debug(f"Batch {batch_num}: Fetching from {datetime.fromtimestamp(current_ts/1000, tz=timezone.utc)}")

                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=max_candles
                )

                if not ohlcv:
                    logger.warning(f"Batch {batch_num}: No data returned")
                    break

                all_candles.extend(ohlcv)
                logger.info(f"Batch {batch_num}: Fetched {len(ohlcv)} candles (total: {len(all_candles)})")

                # Actualizar timestamp para siguiente batch
                last_candle_ts = ohlcv[-1][0]

                # Calcular siguiente timestamp (última vela + 1 timeframe)
                timeframe_ms = _timeframe_to_milliseconds(timeframe)
                current_ts = last_candle_ts + timeframe_ms

                # Rate limiting
                time.sleep(rate_limit_delay)

                break  # Éxito, salir del retry loop

            except ccxt.NetworkError as e:
                logger.warning(f"Batch {batch_num}, Attempt {attempt+1}/{max_retries}: Network error - {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

            except Exception as e:
                logger.error(f"Batch {batch_num}: Unexpected error - {e}")
                raise

        # Si llegamos a una vela que supera end_ts, terminar
        if all_candles and all_candles[-1][0] >= end_ts:
            logger.info(f"Reached end timestamp, stopping pagination")
            break

    if not all_candles:
        raise ValueError("No data loaded from Binance")

    logger.info(f"[OK] Loaded {len(all_candles)} total candles in {batch_num} batches")

    # Convertir a DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Convertir timestamp a datetime (UTC)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)

    # Filtrar por rango exacto (pueden haber venido velas extra)
    df = df[(df.index >= start_dt.replace(tzinfo=timezone.utc)) &
            (df.index <= end_dt.replace(tzinfo=timezone.utc))]

    logger.info(f"After filtering: {len(df)} candles")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    # VALIDACIONES
    _validate_ohlc_data(df, config)

    # Agregar metadata
    df.attrs = {
        'symbol': symbol,
        'timeframe': timeframe,
        'market_type': data_config['market_type'],
        'exchange': exchange_id,
        'start': data_config['start'],
        'end': data_config['end'],
        'source': 'binance',
        'downloaded_at': datetime.now(timezone.utc).isoformat()
    }

    return df

def _timeframe_to_milliseconds(timeframe: str) -> int:
    """
    Convierte timeframe string a milisegundos.

    Args:
        timeframe: "1m", "5m", "15m", "1h", "4h", "1d"

    Returns:
        int: Milisegundos
    """
    timeframe_map = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }

    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return timeframe_map[timeframe]

def _validate_ohlc_data(df: pd.DataFrame, config: dict):
    """
    Valida consistencia y completitud de datos OHLCV.

    Validaciones:
    1. OHLC consistency: H >= max(O,C), L <= min(O,C)
    2. Completitud: <5% missing data
    3. No duplicates en index
    4. Outliers: detectar movimientos extremos (>10σ para cripto)

    Raises:
        ValueError: Si datos inválidos
    """
    logger.info("Validating OHLCV data...")

    # 1. OHLC consistency
    invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
    invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()

    if invalid_high > 0:
        logger.error(f"Found {invalid_high} candles with High < max(Open,Close)")
        raise ValueError("OHLC consistency violated: High < max(Open,Close)")

    if invalid_low > 0:
        logger.error(f"Found {invalid_low} candles with Low > min(Open,Close)")
        raise ValueError("OHLC consistency violated: Low > min(Open,Close)")

    logger.info("  [OK] OHLC consistency OK")

    # 2. Completitud
    missing_pct = df.isnull().sum().sum() / df.size
    if missing_pct > 0.05:
        raise ValueError(f"Too much missing data: {missing_pct:.2%} (threshold: 5%)")

    if missing_pct > 0:
        logger.warning(f"  Missing data: {missing_pct:.4%}")
    else:
        logger.info("  [OK] No missing data")

    # 3. Duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"  Found {duplicates} duplicate timestamps, removing...")
        df = df[~df.index.duplicated(keep='first')]
    else:
        logger.info("  [OK] No duplicate timestamps")

    # 4. Outliers (cripto puede ser muy volátil)
    returns = df['Close'].pct_change()
    mean_return = returns.mean()
    std_return = returns.std()

    # Usar 10 sigma para cripto (más volátil que acciones)
    outliers = (returns.abs() > mean_return + 10 * std_return).sum()

    if outliers > 0:
        logger.warning(f"  Detected {outliers} outliers (>10 sigma movements)")
        logger.warning(f"    Max return: {returns.max():.2%}, Min return: {returns.min():.2%}")
    else:
        logger.info("  [OK] No extreme outliers detected")

    # 5. Enforce real data (si está configurado)
    if config['data'].get('enforce_real_data', True):
        # Chequear que no sea data sintética (ccxt no marca esto, pero verificamos)
        if 'synthetic' in df.attrs and df.attrs['synthetic']:
            raise ValueError("Synthetic data detected, but enforce_real_data=True")

    logger.info("[OK] Data validation passed")

def check_binance_connection(config: dict) -> bool:
    """
    Test de conexión a Binance sin descargar datos.

    Returns:
        bool: True si conexión exitosa
    """
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': config['data']['market_type']}
        })

        # Intentar fetch de 1 vela reciente
        symbol = config['data']['symbol']
        timeframe = config['data']['timeframe']

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1)

        if ohlcv:
            logger.info(f"[OK] Binance connection OK")
            logger.info(f"  Latest {symbol} {timeframe} close: ${ohlcv[0][4]:,.2f}")
            return True
        else:
            logger.error("[FAIL] No data returned from Binance")
            return False

    except Exception as e:
        logger.error(f"[FAIL] Binance connection failed: {e}")
        return False
