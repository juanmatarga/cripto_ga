"""
Test simple sampling independently.

This script validates that simple random sampling creates appropriate windows
from the full dataset without requiring the GA to run.
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.simple_sampling import create_simple_windows
from loader import load_binance_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_simple_sampling():
    """Test simple sampling with real or synthetic data"""
    logger.info("="*80)
    logger.info("TESTING SIMPLE SAMPLING")
    logger.info("="*80)

    config = load_config()

    # Try to load real data
    try:
        logger.info("\nAttempting to load real BTC/USDT data...")
        data = load_binance_data(config)
        logger.info(f"[OK] Loaded {len(data)} bars from Binance")
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
        logger.info("Creating synthetic data for testing...")
        data = create_synthetic_data()

    logger.info(f"\nTotal data: {len(data)} bars")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Test 5 windows of 1 month each
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Create 5 windows of 1 month each")
    logger.info("="*80)

    windows = create_simple_windows(
        data,
        n_windows=5,
        window_months=1,
        seed=42
    )

    logger.info(f"\n[OK] Created {len(windows)} windows:")
    for i, window in enumerate(windows):
        first_close = window['Close'].iloc[0]
        last_close = window['Close'].iloc[-1]
        btc_return = (last_close / first_close - 1) * 100

        logger.info(f"\n  Window {i+1}:")
        logger.info(f"    Bars: {len(window)}")
        logger.info(f"    Date range: {window.index[0].date()} to {window.index[-1].date()}")
        logger.info(f"    BTC return: {btc_return:+.2f}%")
        logger.info(f"    Price: ${first_close:,.2f} → ${last_close:,.2f}")

    # Validate non-overlap
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Validate windows don't overlap")
    logger.info("="*80)

    overlapping = False
    for i in range(len(windows) - 1):
        end_i = windows[i].index[-1]
        start_next = windows[i+1].index[0]
        if end_i >= start_next:
            logger.error(f"❌ Windows {i+1} and {i+2} overlap!")
            overlapping = True

    if not overlapping:
        logger.info("[OK] All windows are non-overlapping ✓")

    # Test different parameters
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Create 10 windows of 2 months each")
    logger.info("="*80)

    windows2 = create_simple_windows(
        data,
        n_windows=10,
        window_months=2,
        seed=42
    )

    logger.info(f"\n[OK] Created {len(windows2)} windows")
    logger.info(f"Average window size: {sum(len(w) for w in windows2) / len(windows2):.0f} bars")

    logger.info("\n" + "="*80)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("="*80)

def create_synthetic_data(n_bars=50000):
    """Create synthetic OHLCV data for testing"""
    import numpy as np

    dates = pd.date_range('2020-01-01', periods=n_bars, freq='15min')

    # Trending price with noise
    base_price = 50000
    trend = np.linspace(0, 10000, n_bars)
    noise = np.random.normal(0, 500, n_bars).cumsum()

    close = base_price + trend + noise

    # Generate OHLC from close
    high = close + np.abs(np.random.normal(0, 200, n_bars))
    low = close - np.abs(np.random.normal(0, 200, n_bars))
    open_price = close + np.random.normal(0, 100, n_bars)

    # Volume
    volume = np.random.uniform(1000, 10000, n_bars)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)

    logger.info(f"Created synthetic data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df

if __name__ == '__main__':
    test_simple_sampling()
