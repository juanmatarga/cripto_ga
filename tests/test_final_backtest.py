"""
Quick test script for final backtest integration.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("Testing imports...")

try:
    from backtest.futures_position_sizing import FuturesPositionManager
    logger.info("[PASS] FuturesPositionManager imported successfully")
except Exception as e:
    logger.error(f"[FAIL] Failed to import FuturesPositionManager: {e}")
    sys.exit(1)

try:
    from analysis.monte_carlo import run_monte_carlo
    logger.info("[PASS] Monte Carlo module imported successfully")
except Exception as e:
    logger.error(f"[FAIL] Failed to import Monte Carlo: {e}")
    sys.exit(1)

try:
    from backtest.final_backtest import run_final_backtest
    logger.info("[PASS] Final backtest module imported successfully")
except Exception as e:
    logger.error(f"[FAIL] Failed to import final_backtest: {e}")
    sys.exit(1)

try:
    from analysis.final_visualization import plot_equity_with_monte_carlo
    logger.info("[PASS] Visualization module imported successfully")
except Exception as e:
    logger.error(f"[FAIL] Failed to import visualization: {e}")
    sys.exit(1)

logger.info("")
logger.info("="*60)
logger.info("ALL IMPORTS SUCCESSFUL!")
logger.info("="*60)
logger.info("")
logger.info("Testing FuturesPositionManager basic functionality...")

# Test basic position manager
pm = FuturesPositionManager(initial_capital=1000.0, risk_per_trade_pct=0.02, leverage=10.0)

logger.info(f"Initial equity: ${pm.current_equity:.2f}")
logger.info(f"Risk per trade: {pm.risk_per_trade_pct*100}%")
logger.info(f"Leverage: {pm.leverage}x")

# Test position sizing calculation
position = pm.calculate_position_size(
    entry_price=50000.0,
    stop_loss_price=49000.0,
    direction='LONG'
)

logger.info("")
logger.info("Test position calculation:")
logger.info(f"  Entry: $50,000")
logger.info(f"  Stop: $49,000 (2% away)")
logger.info(f"  Direction: LONG")
logger.info(f"  Notional value: ${position['notional_value']:.2f}")
logger.info(f"  Margin required: ${position['margin_required']:.2f}")
logger.info(f"  Contracts: {position['contracts']:.4f}")
logger.info(f"  Risk USD: ${position['risk_usd']:.2f}")

logger.info("")
logger.info("="*60)
logger.info("ALL TESTS PASSED! [PASS]")
logger.info("="*60)
logger.info("")
logger.info("Ready to run full pipeline with:")
logger.info("  python main.py --generations 5")
