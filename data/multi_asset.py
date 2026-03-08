"""
Multi-asset data management.

Downloads, caches, and validates OHLCV data for multiple crypto assets.
Each asset uses the same loader.py infrastructure.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data.loader import load_data, CACHE_DIR

logger = logging.getLogger(__name__)

# ============================================================================
# ASSET DEFINITIONS
# ============================================================================

# Assets selected for multi-asset evolution
# Criteria: futures available, >$500M daily volume, >2 years history
ASSETS = {
    'BTC/USDT': {
        'start': '2022-01-01',   # Include bear market
        'min_bars': 50000,       # ~1.5 years of 15m
        'notes': 'Reference asset, most liquid',
    },
    'ETH/USDT': {
        'start': '2022-01-01',
        'min_bars': 50000,
        'notes': 'Second most liquid, high correlation with BTC',
    },
    'SOL/USDT': {
        'start': '2022-01-01',
        'min_bars': 50000,
        'notes': 'High volatility, potential for more alpha',
    },
    'BNB/USDT': {
        'start': '2022-01-01',
        'min_bars': 50000,
        'notes': 'Binance ecosystem, moderate volatility',
    },
}


def make_asset_config(symbol: str, base_config: dict) -> dict:
    """
    Create a per-asset config by overriding symbol-specific fields.

    Args:
        symbol: Trading pair (e.g. 'ETH/USDT')
        base_config: Base config dict (from config_v2.yaml)

    Returns:
        Config dict with symbol-specific overrides
    """
    asset_info = ASSETS.get(symbol, {})
    config = {k: dict(v) if isinstance(v, dict) else v
              for k, v in base_config.items()}

    config['data'] = dict(base_config.get('data', {}))
    config['data']['symbol'] = symbol
    if 'start' in asset_info:
        config['data']['start'] = asset_info['start']

    return config


def load_all_assets(base_config: dict,
                    symbols: Optional[List[str]] = None,
                    end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for all specified assets.

    Args:
        base_config: Base config dict
        symbols: List of symbols to load (default: all ASSETS)
        end_date: Override end date (for OTS boundary enforcement)

    Returns:
        Dict mapping symbol to DataFrame
    """
    symbols = symbols or list(ASSETS.keys())
    result = {}

    for symbol in symbols:
        logger.info(f"Loading {symbol}...")
        config = make_asset_config(symbol, base_config)
        if end_date:
            config['data']['end'] = end_date

        try:
            df = load_data(config)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Validate minimum bars
            min_bars = ASSETS.get(symbol, {}).get('min_bars', 10000)
            if len(df) < min_bars:
                logger.warning(f"{symbol}: only {len(df)} bars (need {min_bars}), skipping")
                continue

            result[symbol] = df
            logger.info(f"{symbol}: {len(df)} bars "
                        f"({df.index.min()} to {df.index.max()})")

        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")

    logger.info(f"Loaded {len(result)}/{len(symbols)} assets")
    return result


def validate_asset(df: pd.DataFrame, symbol: str) -> dict:
    """
    Validate asset data quality for evolution.

    Returns dict with validation results.
    """
    n_bars = len(df)
    date_range = (df.index.max() - df.index.min()).days

    # Check for gaps (missing 15m bars)
    expected_bars = date_range * 24 * 4  # 96 bars per day
    coverage = n_bars / expected_bars if expected_bars > 0 else 0

    # Check for zero volume
    zero_vol_pct = (df['Volume'] == 0).mean() * 100

    # Check price range (detect stale data)
    price_range = df['Close'].max() / df['Close'].min()

    result = {
        'symbol': symbol,
        'n_bars': n_bars,
        'days': date_range,
        'coverage': coverage,
        'zero_vol_pct': zero_vol_pct,
        'price_range': price_range,
        'valid': coverage > 0.95 and zero_vol_pct < 5.0,
    }

    if not result['valid']:
        logger.warning(f"{symbol} validation FAILED: "
                       f"coverage={coverage:.1%}, zero_vol={zero_vol_pct:.1f}%")
    else:
        logger.info(f"{symbol} validation OK: "
                    f"{n_bars} bars, {date_range}d, "
                    f"coverage={coverage:.1%}")

    return result
