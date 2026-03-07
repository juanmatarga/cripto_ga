"""
CriptoGA Live Trading Runner.

Main event loop: fetches data, evaluates signals, executes trades.
Designed to be restart-safe (all state persisted to disk).

Usage:
    python -m live.runner              # Start trading (uses .env config)
    python -m live.runner --status     # Show current status
    python -m live.runner --close-all  # Emergency close all positions
    python -m live.runner --reset      # Reset halt state (after circuit breaker)
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from live.config import load_config, LiveConfig
from live.connector import BinanceConnector
from live.signals import LiveSignalEngine
from live.state import StateManager
from live.executor import Executor

# ============================================================================
# LOGGING
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(trading_mode: str):
    """Configure logging to file + console."""
    log_file = LOG_DIR / f'live_{trading_mode}_{datetime.now().strftime("%Y%m%d")}.log'

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler (all levels)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    # Suppress noisy libraries
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# ============================================================================
# MAIN LOOP
# ============================================================================

class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: LiveConfig):
        self.config = config
        self.connector = BinanceConnector(config)
        self.signal_engine = LiveSignalEngine(config.strategies)
        self.state = StateManager()
        self.executor = Executor(config, self.connector, self.signal_engine, self.state)
        self.running = True
        self.logger = logging.getLogger(__name__)

    def startup(self):
        """Initialize: set leverage, check balance, load state."""
        mode = "TESTNET" if self.config.is_testnet else "LIVE"
        self.logger.info(f"{'='*60}")
        self.logger.info(f"CriptoGA Live Trading Bot — {mode}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Symbol: {self.config.symbol}")
        self.logger.info(f"Timeframe: {self.config.timeframe}")
        self.logger.info(f"Strategies: {len(self.config.strategies)}")

        for sc in self.config.strategies:
            self.logger.info(f"  {sc.key}: {sc.direction} | "
                              f"TP={sc.tp_atr_mult} SL={sc.sl_atr_mult} "
                              f"Trail={sc.trail_atr_mult} | w={sc.weight:.0%}")

        # Set leverage
        self.connector.set_leverage(self.config.risk.leverage)
        self.logger.info(f"Leverage: {self.config.risk.leverage}x")

        # Get balance
        balance = self.connector.get_balance()
        self.logger.info(f"Balance: ${balance['total']:,.2f} "
                          f"(free: ${balance['free']:,.2f})")

        # Initialize state
        self.state.initialize(balance['total'], self.config.trading_mode)

        # Risk parameters
        r = self.config.risk
        self.logger.info(f"Risk: max_dd={r.max_portfolio_dd_pct}%, "
                          f"daily_max_loss={r.max_daily_loss_pct}%, "
                          f"max_position={r.max_position_pct}%")

        # Check for existing positions
        positions = self.connector.get_positions()
        if positions:
            self.logger.warning(f"Found {len(positions)} existing positions on exchange!")
            for p in positions:
                self.logger.warning(
                    f"  {p['side'].upper()} {p['contracts']} BTC "
                    f"@ ${p['entry_price']:,.2f} "
                    f"PnL=${p['unrealized_pnl']:+,.2f}"
                )

        self.logger.info(f"{'='*60}")
        self.logger.info("Bot started. Waiting for signals...")

    def run_cycle(self):
        """
        Single trading cycle:
        1. Fetch latest OHLCV data
        2. Check if new candle (avoid re-processing)
        3. Check circuit breakers
        4. Evaluate signals
        5. Execute entries for new signals
        6. Check/update exits
        7. Update trailing stops
        """
        # 1. Fetch data
        try:
            df = self.connector.fetch_ohlcv(self.config.lookback_bars)
        except Exception as e:
            self.logger.error(f"Data fetch failed: {e}")
            return

        # 2. Check for new candle
        last_candle = str(df.index[-1])
        if last_candle == self.state.state.last_candle_time:
            # Same candle, just check exits
            self.executor.check_and_close_filled_exits()
            return

        self.logger.info(f"New candle: {last_candle} | "
                          f"Close=${df['Close'].iloc[-1]:,.2f}")
        self.state.state.last_candle_time = last_candle
        self.state.save()

        # 3. Update capital and check circuit breakers
        try:
            balance = self.connector.get_balance()
            self.state.update_capital(balance['total'])
        except Exception as e:
            self.logger.error(f"Balance check failed: {e}")

        if not self.state.check_circuit_breakers(self.config):
            self.executor.emergency_close_all()
            return

        # 4. Check for filled exits
        self.executor.check_and_close_filled_exits()

        # 5. Evaluate signals on new candle
        signals = self.signal_engine.evaluate(df)

        # 6. Execute entries for new signals
        for strategy_key, sig_data in signals.items():
            if sig_data.get('new_signal', False):
                if not self.state.has_position(strategy_key):
                    self.logger.info(f"NEW SIGNAL: {strategy_key} → executing entry")
                    self.executor.execute_entry(strategy_key, sig_data)

        # 7. Update trailing stops
        self.executor.update_trailing_stops(df)

        # 8. Log status summary
        self._log_status()

    def _log_status(self):
        """Log current portfolio status."""
        n_open = len(self.state.state.open_positions)
        total_pnl = self.state.state.total_pnl_usdt
        daily_pnl = self.state.state.daily_pnl
        n_trades = self.state.state.total_trades
        win_rate = (self.state.state.winning_trades / n_trades * 100
                    if n_trades > 0 else 0)

        self.logger.info(
            f"STATUS: {n_open} open | "
            f"{n_trades} trades (WR {win_rate:.0f}%) | "
            f"daily=${daily_pnl:+,.2f} | "
            f"total=${total_pnl:+,.2f} | "
            f"capital=${self.state.state.current_capital:,.2f}"
        )

    def _seconds_until_next_candle(self, offset: int = 5) -> float:
        """
        Calculate seconds until next candle close + offset.

        For 15m candles, closes happen at :00, :15, :30, :45.
        We add `offset` seconds to ensure the candle is fully closed
        and available on the exchange.
        """
        now = datetime.now(timezone.utc)
        tf_seconds = self.connector._timeframe_to_seconds(self.config.timeframe)
        # Current candle started at the last multiple of tf_seconds
        epoch = now.timestamp()
        current_candle_start = (epoch // tf_seconds) * tf_seconds
        next_candle_close = current_candle_start + tf_seconds + offset
        wait = next_candle_close - epoch
        if wait <= 0:
            wait = tf_seconds + wait  # Already past, wait for next one
        return wait

    def run(self):
        """Main loop — synchronized with candle closes."""
        self.startup()

        # Run one cycle immediately on start
        while self.running:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.error(f"Cycle error: {e}", exc_info=True)

            # Sleep until 2 seconds after next candle close
            wait = self._seconds_until_next_candle(offset=2)
            next_time = datetime.now(timezone.utc).timestamp() + wait
            next_dt = datetime.fromtimestamp(next_time, tz=timezone.utc)
            self.logger.info(f"Next check at {next_dt.strftime('%H:%M:%S')} UTC "
                             f"(sleeping {wait:.0f}s)")
            time.sleep(wait)

    def shutdown(self):
        """Graceful shutdown."""
        self.running = False
        self.logger.info("Shutting down...")

        # Don't close positions on shutdown — they have SL/TP on exchange
        n_open = len(self.state.state.open_positions)
        if n_open > 0:
            self.logger.info(
                f"{n_open} positions remain open with exchange SL/TP orders. "
                f"Use --close-all to close them manually."
            )

        self.state.save()
        self.logger.info("Bot stopped.")


# ============================================================================
# CLI
# ============================================================================

def show_status():
    """Show current bot status."""
    state = StateManager()
    s = state.state

    print(f"\nCriptoGA Live Trading Status ({s.trading_mode})")
    print(f"{'='*50}")
    print(f"Started:       {s.started_at or 'never'}")
    print(f"Last update:   {s.last_update or 'never'}")
    print(f"Last candle:   {s.last_candle_time or 'none'}")
    print(f"Halted:        {'YES — ' + s.halt_reason if s.is_halted else 'No'}")
    print(f"\nCapital:")
    print(f"  Initial:     ${s.initial_capital:,.2f}")
    print(f"  Current:     ${s.current_capital:,.2f}")
    print(f"  Peak:        ${s.peak_capital:,.2f}")
    dd = ((s.current_capital - s.peak_capital) / s.peak_capital * 100
          if s.peak_capital > 0 else 0)
    print(f"  Drawdown:    {dd:+.2f}%")
    print(f"\nTrading:")
    print(f"  Total trades: {s.total_trades}")
    wr = s.winning_trades / s.total_trades * 100 if s.total_trades > 0 else 0
    print(f"  Win rate:     {wr:.0f}%")
    print(f"  Total PnL:    ${s.total_pnl_usdt:+,.2f}")
    print(f"  Daily PnL:    ${s.daily_pnl:+,.2f}")
    print(f"\nOpen Positions: {len(s.open_positions)}")
    for key, pos in s.open_positions.items():
        print(f"  {key}: {pos['direction']} @ ${pos['entry_price']:,.2f} "
              f"SL=${pos['stop_loss']:,.2f} "
              f"TP=${pos['take_profit']:,.2f}" if pos.get('take_profit') else
              f"  {key}: {pos['direction']} @ ${pos['entry_price']:,.2f} "
              f"SL=${pos['stop_loss']:,.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='CriptoGA Live Trading Bot')
    parser.add_argument('--status', action='store_true',
                        help='Show current status')
    parser.add_argument('--close-all', action='store_true',
                        help='Emergency close all positions')
    parser.add_argument('--reset', action='store_true',
                        help='Reset halt state (after circuit breaker)')
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.reset:
        state = StateManager()
        state.state.is_halted = False
        state.state.halt_reason = ""
        state.save()
        print("Halt state cleared.")
        return

    # Normal run
    config = load_config()
    logger = setup_logging(config.trading_mode)

    if args.close_all:
        connector = BinanceConnector(config)
        signal_engine = LiveSignalEngine(config.strategies)
        state = StateManager()
        executor = Executor(config, connector, signal_engine, state)
        executor.emergency_close_all()
        return

    bot = TradingBot(config)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        bot.run()
    except KeyboardInterrupt:
        bot.shutdown()


if __name__ == '__main__':
    main()
