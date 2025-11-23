"""
Comprehensive backtest audit script.

This script tests every component independently to identify bugs.
Run this BEFORE making any fixes to understand what's broken.

Usage:
    python debug/audit_backtest.py

Expected behavior:
    - All tests should pass
    - If any test fails, it identifies the exact bug location
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ga_patterns.chromosome_v2 import PatternChromosome
from ga_patterns.evaluator import preprocess_indicators, evaluate_expression
from backtest.runner import run_backtest
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_data_sample(config):
    """Load a sample of data for testing"""
    # Check for processed data
    data_file = Path('data/processed/btc_usdt_15m.parquet')

    if not data_file.exists():
        logger.warning(f"Data file not found: {data_file}")
        logger.info("Creating synthetic data for testing...")
        return create_synthetic_data()

    logger.info(f"Loading data from {data_file}")
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(data)} bars")

    # Use subset for speed
    if len(data) > 20000:
        data = data.iloc[-20000:].copy()
        logger.info(f"Using last {len(data)} bars for testing")

    return data


def create_synthetic_data(n_bars=5000):
    """Create synthetic OHLCV data if real data not available"""
    logger.info(f"Creating {n_bars} bars of synthetic data...")

    dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')

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

    logger.info(f"Synthetic data created: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Price range: ${df['Close'].min():,.2f} to ${df['Close'].max():,.2f}")

    return df


# ============================================================================
# TEST 1: DATA SANITY
# ============================================================================

def test_1_data_sanity():
    """Test that data is loaded correctly and has expected properties."""
    logger.info("="*80)
    logger.info("TEST 1: DATA SANITY CHECK")
    logger.info("="*80)

    config = load_config()
    data = load_data_sample(config)

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    logger.info(f"Columns: {data.columns.tolist()}")

    # Check for NaNs
    nan_count = data.isnull().sum().sum()
    logger.info(f"Total NaNs: {nan_count}")

    if nan_count > 0:
        logger.warning(f"[WARNING]  Found {nan_count} NaN values")
        nan_cols = data.columns[data.isnull().any()].tolist()
        logger.warning(f"Columns with NaNs: {nan_cols}")

    # Check price movement
    first_close = data['Close'].iloc[0]
    last_close = data['Close'].iloc[-1]
    total_return = (last_close / first_close - 1) * 100

    logger.info(f"\nPrice movement:")
    logger.info(f"  First close: ${first_close:,.2f}")
    logger.info(f"  Last close: ${last_close:,.2f}")
    logger.info(f"  Total return: {total_return:.2f}%")

    # Sample prices
    logger.info(f"\nFirst 5 bars:")
    logger.info(data[['Open', 'High', 'Low', 'Close', 'Volume']].head().to_string())

    # Sanity checks
    assert data.shape[0] > 1000, f"Expected >1000 bars, got {data.shape[0]}"
    assert first_close > 100, "Price should be >$100"

    logger.info("\n[OK] TEST 1 PASSED: Data is clean and reasonable\n")
    return data


# ============================================================================
# TEST 2: INDICATOR CALCULATION
# ============================================================================

def test_2_indicator_calculation():
    """Test that technical indicators are calculated correctly."""
    logger.info("="*80)
    logger.info("TEST 2: INDICATOR CALCULATION")
    logger.info("="*80)

    config = load_config()
    data = load_data_sample(config)

    initial_cols = set(data.columns)

    # Preprocess
    logger.info("Preprocessing indicators...")
    data = preprocess_indicators(data)

    new_cols = set(data.columns) - initial_cols
    logger.info(f"Indicators added: {sorted(list(new_cols))}")
    logger.info(f"Total new columns: {len(new_cols)}")

    # Expected indicators
    expected_indicators = [
        'body_pct', 'range_pct', 'close_position_in_range',
        'RSI_14', 'SMA_20', 'SMA_50', 'SMA_V_20',
        'MACD', 'MACD_signal', 'MACD_hist',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Width_SMA_20',
        'ATR_14', 'ATR_SMA_20',
        'Stoch_K', 'Stoch_D'
    ]

    missing = [ind for ind in expected_indicators if ind not in data.columns]
    if missing:
        logger.error(f"[ERROR] Missing indicators: {missing}")
        raise AssertionError(f"Missing indicators: {missing}")

    # Check RSI range
    rsi_sample = data['RSI_14'].iloc[100:110]
    logger.info(f"\nRSI_14 sample (should be 0-100):")
    logger.info(rsi_sample.to_string())

    rsi_min = data['RSI_14'].dropna().min()
    rsi_max = data['RSI_14'].dropna().max()
    logger.info(f"RSI range: {rsi_min:.2f} to {rsi_max:.2f}")

    assert rsi_min >= 0, f"RSI min should be ≥0, got {rsi_min}"
    assert rsi_max <= 100, f"RSI max should be ≤100, got {rsi_max}"

    # Check body_pct
    body_sample = data['body_pct'].iloc[100:110]
    logger.info(f"\nbody_pct sample (should be small decimals):")
    logger.info(body_sample.to_string())

    body_max = data['body_pct'].max()
    logger.info(f"body_pct max: {body_max:.4f}")

    assert body_max < 0.5, f"body_pct should be <0.5 (50%), got {body_max}"

    logger.info("\n[OK] TEST 2 PASSED: Indicators calculated correctly\n")
    return data


# ============================================================================
# TEST 3: EXPRESSION EVALUATION
# ============================================================================

def test_3_expression_evaluation():
    """Test that expressions evaluate correctly on real data."""
    logger.info("="*80)
    logger.info("TEST 3: EXPRESSION EVALUATION")
    logger.info("="*80)

    config = load_config()
    data = load_data_sample(config)
    data = preprocess_indicators(data)

    # Test simple expression
    expr = "C[0] > C[1]"  # Current close > previous close (bullish)

    logger.info(f"Testing expression: '{expr}'")
    logger.info("This should return True when price goes UP")

    # Find bullish and bearish bars
    bullish_bars = []
    bearish_bars = []

    for i in range(100, 200):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            bullish_bars.append(i)
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            bearish_bars.append(i)

    logger.info(f"\nFound {len(bullish_bars)} bullish bars")
    logger.info(f"Found {len(bearish_bars)} bearish bars")

    # Test on bullish bar
    test_bar = bullish_bars[0]
    close_current = data['Close'].iloc[test_bar]
    close_prev = data['Close'].iloc[test_bar-1]

    result = evaluate_expression(expr, data, test_bar)

    logger.info(f"\nTest on BULLISH bar {test_bar}:")
    logger.info(f"  Close[{test_bar}] = ${close_current:.2f}")
    logger.info(f"  Close[{test_bar-1}] = ${close_prev:.2f}")
    logger.info(f"  Price went: {'UP' if close_current > close_prev else 'DOWN'}")
    logger.info(f"  Expression '{expr}' evaluated to: {result}")
    logger.info(f"  Expected: True")

    if result != True:
        logger.error("[ERROR] EXPRESSION BUG: Expression returned False on bullish bar!")
        raise AssertionError("Expression evaluation is INVERTED")

    # Test on bearish bar
    test_bar_bear = bearish_bars[0]
    close_current_bear = data['Close'].iloc[test_bar_bear]
    close_prev_bear = data['Close'].iloc[test_bar_bear-1]

    result_bear = evaluate_expression(expr, data, test_bar_bear)

    logger.info(f"\nTest on BEARISH bar {test_bar_bear}:")
    logger.info(f"  Close[{test_bar_bear}] = ${close_current_bear:.2f}")
    logger.info(f"  Close[{test_bar_bear-1}] = ${close_prev_bear:.2f}")
    logger.info(f"  Price went: {'UP' if close_current_bear > close_prev_bear else 'DOWN'}")
    logger.info(f"  Expression '{expr}' evaluated to: {result_bear}")
    logger.info(f"  Expected: False")

    if result_bear != False:
        logger.error("[ERROR] EXPRESSION BUG: Expression returned True on bearish bar!")
        raise AssertionError("Expression evaluation is INVERTED")

    logger.info("\n[OK] TEST 3 PASSED: Expressions evaluate correctly\n")


# ============================================================================
# TEST 4: SIMPLE BACKTEST
# ============================================================================

def test_4_simple_backtest():
    """Test basic backtest functionality with a simple LONG pattern."""
    logger.info("="*80)
    logger.info("TEST 4: SIMPLE BACKTEST")
    logger.info("="*80)

    config = load_config()
    data = load_data_sample(config)
    data = preprocess_indicators(data)

    # Create simple LONG pattern
    pattern = PatternChromosome(
        direction='LONG',
        modules=['momentum_up_2bar'],
        logic='AND',
        window=3
    )

    logger.info(f"Pattern: {pattern.to_readable()}")

    # Take sample (1000 bars)
    sample_data = data.iloc[1000:2000].copy()

    btc_first = sample_data['Close'].iloc[0]
    btc_last = sample_data['Close'].iloc[-1]
    btc_return = (btc_last / btc_first - 1) * 100

    logger.info(f"\nSample period: {sample_data.index[0]} to {sample_data.index[-1]}")
    logger.info(f"Price: ${btc_first:,.2f} -> ${btc_last:,.2f}")
    logger.info(f"Return: {btc_return:+.2f}%")

    # Run backtest
    logger.info("\nRunning backtest...")
    equity_curve, trades = run_backtest(pattern, sample_data, config)

    final_equity = equity_curve.iloc[-1]
    strategy_return = (final_equity - 100)

    logger.info(f"\nBacktest results:")
    logger.info(f"  Number of trades: {len(trades)}")
    logger.info(f"  Initial equity: 100.00")
    logger.info(f"  Final equity: {final_equity:.2f}")
    logger.info(f"  Strategy return: {strategy_return:+.2f}%")

    if len(trades) > 0:
        logger.info(f"\nFirst 3 trades:")
        # Trades is a DataFrame
        for i in range(min(3, len(trades))):
            trade = trades.iloc[i]
            logger.info(f"  Trade {i+1}:")
            logger.info(f"    Entry: ${trade['entry_price']:.2f}")
            logger.info(f"    Exit: ${trade['exit_price']:.2f}")
            logger.info(f"    PnL: {trade['pnl_pct']*100:+.4f}%")

        avg_pnl = trades['pnl_pct'].mean()
        logger.info(f"\nAverage PnL per trade: {avg_pnl*100:+.4f}%")
    else:
        logger.warning("[WARNING]  No trades generated - pattern too restrictive")

    # CRITICAL CHECK
    if btc_return > 5 and final_equity < 80:
        logger.error("\n" + "="*80)
        logger.error("🚨 CRITICAL BUG DETECTED")
        logger.error("="*80)
        logger.error(f"Price went UP {btc_return:.1f}% but LONG strategy lost {100-final_equity:.1f}%")
        logger.error("\nThis indicates DIRECTION IS INVERTED")
        logger.error("="*80)
        raise AssertionError("CRITICAL: Strategy loses money when price is up")

    logger.info("\n[OK] TEST 4 PASSED: Basic backtest works\n")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all diagnostic tests in sequence."""
    logger.info("\n" + "="*80)
    logger.info("STARTING COMPREHENSIVE AUDIT")
    logger.info("="*80)
    logger.info("This will test every component to find bugs\n")

    try:
        # Run tests in order
        data = test_1_data_sanity()
        data = test_2_indicator_calculation()
        test_3_expression_evaluation()
        test_4_simple_backtest()

        # All passed
        logger.info("="*80)
        logger.info("🎉 ALL TESTS PASSED")
        logger.info("="*80)
        logger.info("\nNo critical bugs detected in isolated components.")
        logger.info("\nIf patterns still show fitness=-999, the bug is likely in:")
        logger.info("  1. fitness.py - window combination logic")
        logger.info("  2. Pattern generation - patterns too restrictive")
        logger.info("  3. Fitness constraints - thresholds too strict")

    except AssertionError as e:
        logger.error("\n" + "="*80)
        logger.error(f"[ERROR] TEST FAILED")
        logger.error("="*80)
        logger.error(f"\nError: {e}")
        logger.error("\nA critical bug was detected.")
        logger.error("Fix this bug before running the GA.")
        logger.error("="*80)
        raise

    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error(f"[ERROR] UNEXPECTED ERROR")
        logger.error("="*80)
        logger.error(f"\nError: {e}")
        import traceback
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        logger.error("="*80)
        raise


if __name__ == '__main__':
    run_all_tests()
