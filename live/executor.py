"""
Order execution and position management.

Handles the logic of: when to enter, how to size, placing SL/TP on exchange,
and monitoring/closing positions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from live.config import LiveConfig, StrategyConfig
from live.connector import BinanceConnector
from live.signals import LiveSignalEngine
from live.state import StateManager, OpenPosition

logger = logging.getLogger(__name__)


class Executor:
    """Manages order execution with risk controls."""

    def __init__(self, config: LiveConfig, connector: BinanceConnector,
                 signal_engine: LiveSignalEngine, state: StateManager):
        self.config = config
        self.connector = connector
        self.signals = signal_engine
        self.state = state

    def calculate_position_size(self, strategy_config: StrategyConfig,
                                balance: float) -> float:
        """
        Calculate position size in USDT (notional value, before leverage).

        Equal weight allocation: each strategy gets 1/N of capital.
        With leverage, the notional = allocated_capital × leverage.

        Example:
            Capital = $1000, 3 strategies, 5x leverage
            Per strategy = $1000 × 33.3% = $333
            Notional per trade = $333 × 5 = $1667
            Margin used = $333

        Returns USDT notional value for the order.
        """
        max_pct = self.config.risk.max_position_pct / 100.0
        allocated = balance * strategy_config.weight
        allocated = min(allocated, balance * max_pct)

        # Apply leverage to get notional
        notional = allocated * self.config.risk.leverage

        logger.debug(f"{strategy_config.key}: balance=${balance:.2f}, "
                      f"allocated=${allocated:.2f}, "
                      f"notional=${notional:.2f} ({self.config.risk.leverage}x)")
        return notional

    def execute_entry(self, strategy_key: str, signal_data: dict) -> bool:
        """
        Execute entry for a strategy signal.

        Returns True if order was placed successfully.
        """
        # Find strategy config
        sc = next(s for s in self.config.strategies if s.key == strategy_key)
        symbol = sc.symbol
        direction = signal_data['direction']
        atr = signal_data['atr']
        entry_price = signal_data['last_close']

        # Don't enter if already in position
        if self.state.has_position(strategy_key):
            logger.debug(f"{strategy_key}: already in position, skipping")
            return False

        # Check max positions
        n_open = len(self.state.state.open_positions)
        if n_open >= self.config.risk.max_open_positions:
            logger.warning(f"Max positions reached ({n_open}), skipping {strategy_key}")
            return False

        # Calculate position size
        balance = self.connector.get_balance()
        free_balance = balance['free']

        if free_balance < self.config.risk.min_order_usdt:
            logger.warning(f"Insufficient free balance: ${free_balance:.2f}")
            return False

        notional = self.calculate_position_size(sc, free_balance)
        if notional < self.config.risk.min_order_usdt:
            logger.warning(f"Position too small: ${notional:.2f}")
            return False

        # Calculate exit levels
        tp, sl = self.signals.compute_exit_levels(
            entry_price, atr, direction,
            sc.tp_atr_mult, sc.sl_atr_mult
        )

        logger.info(
            f"ENTRY {strategy_key} ({symbol}): {direction} "
            f"@ ~${entry_price:,.2f} | "
            f"notional=${notional:,.2f} | "
            f"SL=${sl:,.2f} | "
            f"TP=${tp:,.2f}" if tp else
            f"ENTRY {strategy_key} ({symbol}): {direction} "
            f"@ ~${entry_price:,.2f} | notional=${notional:,.2f} | SL=${sl:,.2f}"
        )

        try:
            # Place market entry order
            side = 'buy' if direction == 'LONG' else 'sell'
            order = self.connector.place_market_order(symbol, side, notional)

            actual_price = order['price']
            quantity = order['quantity']

            # Recalculate exits with actual fill price
            tp, sl = self.signals.compute_exit_levels(
                actual_price, atr, direction,
                sc.tp_atr_mult, sc.sl_atr_mult
            )

            # Place SL order on exchange — CRITICAL for risk management
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            sl_order_id = None
            tp_order_id = None
            try:
                sl_order = self.connector.place_stop_loss(symbol, sl_side, quantity, sl)
                sl_order_id = sl_order.get('id')
            except Exception as sl_err:
                logger.error(f"SL placement failed for {strategy_key}: {sl_err}")
                # SAFETY: close position immediately — never leave unprotected
                try:
                    logger.warning(f"Emergency close {strategy_key}: no SL protection")
                    close_side = 'sell' if direction == 'LONG' else 'buy'
                    self.connector.place_market_order(symbol, close_side, quantity * actual_price)
                except Exception as close_err:
                    logger.critical(f"CRITICAL: Cannot close unprotected position {strategy_key}: {close_err}")
                return False

            # Place TP order if applicable (non-critical — position is SL-protected)
            if tp is not None:
                try:
                    tp_order = self.connector.place_take_profit(
                        symbol, sl_side, quantity, tp
                    )
                    tp_order_id = tp_order.get('id')
                except Exception as tp_err:
                    logger.warning(f"TP placement failed for {strategy_key}: {tp_err} (position is SL-protected)")

            # Record position in state
            now = datetime.now(timezone.utc).isoformat()
            pos = OpenPosition(
                strategy_key=strategy_key,
                symbol=symbol,
                direction=direction,
                entry_price=actual_price,
                quantity=quantity,
                entry_time=now,
                entry_bar_time=str(signal_data.get('last_bar_time', '')),
                atr_at_entry=atr,
                stop_loss=sl,
                take_profit=tp,
                trail_atr_mult=sc.trail_atr_mult,
                initial_stop=sl,
                best_price=actual_price,
                sl_order_id=sl_order_id,
                tp_order_id=tp_order_id,
                notional_usdt=notional,
            )
            self.state.open_position(pos)

            logger.info(
                f"ENTRY COMPLETE {strategy_key} ({symbol}): {direction} "
                f"@ ${actual_price:,.2f} qty={quantity:.6f} "
                f"SL=${sl:,.2f} TP=${tp:,.2f}" if tp else
                f"ENTRY COMPLETE {strategy_key} ({symbol}): {direction} "
                f"@ ${actual_price:,.2f} qty={quantity:.6f} SL=${sl:,.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Entry failed for {strategy_key}: {e}", exc_info=True)
            return False

    def check_and_close_filled_exits(self):
        """
        Check if any SL/TP orders have been filled by the exchange.

        Strategy: check actual exchange positions. If the bot thinks we have
        a position but the exchange doesn't, the SL/TP must have filled.
        Matches by (symbol, side) to handle multi-symbol portfolios correctly.
        """
        if not self.state.state.open_positions:
            return

        try:
            exchange_positions = self.connector.get_positions(self.config.symbols)
        except Exception as e:
            logger.error(f"Failed to fetch exchange positions: {e}")
            return

        # Build set of (symbol, side) pairs that are open on exchange
        exchange_open = set()
        for p in exchange_positions:
            exchange_open.add((p['symbol'], p['side']))  # ('BTC/USDT:USDT', 'long')

        # Check each bot-tracked position
        for key, pos_data in list(self.state.state.open_positions.items()):
            pos = OpenPosition(**pos_data)
            expected_side = 'long' if pos.direction == 'LONG' else 'short'
            symbol = pos.symbol

            if (symbol, expected_side) not in exchange_open:
                # Position was closed by exchange (SL or TP filled)
                try:
                    price = self.connector.get_ticker_price(symbol)

                    # Determine exit type by comparing price to SL/TP levels
                    if pos.direction == 'LONG':
                        exit_type = 'target' if (pos.take_profit and
                            price >= pos.take_profit * 0.99) else 'stop'
                    else:
                        exit_type = 'target' if (pos.take_profit and
                            price <= pos.take_profit * 1.01) else 'stop'

                    logger.info(f"Position {key} ({symbol}) closed by exchange ({exit_type})")
                    self.state.close_position(key, price, exit_type)

                    # Cancel any remaining conditional orders
                    if pos.sl_order_id:
                        self.connector.cancel_order_by_id(
                            pos.sl_order_id, symbol, is_trigger=True
                        )
                    if pos.tp_order_id:
                        self.connector.cancel_order_by_id(
                            pos.tp_order_id, symbol, is_trigger=True
                        )

                except Exception as e:
                    logger.error(f"Error reconciling closed position {key}: {e}")

    def update_trailing_stops(self, prices_by_symbol: dict):
        """
        Update trailing stops for open positions.

        Args:
            prices_by_symbol: {symbol: last_close_price} from latest candles

        For strategies with trail_atr_mult > 0, we move the stop loss
        to follow the price upward (for longs) or downward (for shorts).
        """
        for key, pos_data in list(self.state.state.open_positions.items()):
            pos = OpenPosition(**pos_data)

            if pos.trail_atr_mult <= 0:
                continue

            current_price = prices_by_symbol.get(pos.symbol)
            if current_price is None:
                continue

            # Update best price
            if pos.direction == 'LONG':
                new_best = max(pos.best_price, current_price)
                new_trail = new_best - pos.trail_atr_mult * pos.atr_at_entry
                new_sl = max(pos.stop_loss, new_trail)
            else:
                new_best = min(pos.best_price, current_price)
                new_trail = new_best + pos.trail_atr_mult * pos.atr_at_entry
                new_sl = min(pos.stop_loss, new_trail)

            if new_sl != pos.stop_loss or new_best != pos.best_price:
                # Update state
                pos_data['best_price'] = new_best
                pos_data['stop_loss'] = new_sl
                self.state.save()

                symbol = pos.symbol

                # Update SL order on exchange (cancel old, place new)
                if pos.sl_order_id:
                    try:
                        self.connector.cancel_order_by_id(
                            pos.sl_order_id, symbol, is_trigger=True
                        )
                        if pos.tp_order_id:
                            self.connector.cancel_order_by_id(
                                pos.tp_order_id, symbol, is_trigger=True
                            )

                        sl_side = 'sell' if pos.direction == 'LONG' else 'buy'
                        new_sl_order = self.connector.place_stop_loss(
                            symbol, sl_side, pos.quantity, new_sl
                        )
                        pos_data['sl_order_id'] = new_sl_order['id']

                        # Re-place TP if it existed
                        if pos.take_profit:
                            new_tp_order = self.connector.place_take_profit(
                                symbol, sl_side, pos.quantity, pos.take_profit
                            )
                            pos_data['tp_order_id'] = new_tp_order['id']

                        self.state.save()
                        logger.info(
                            f"TRAIL {key} ({symbol}): SL moved "
                            f"${pos.stop_loss:,.2f} → ${new_sl:,.2f} "
                            f"(best=${new_best:,.2f})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to update trail SL for {key}: {e}")

    def emergency_close_all(self):
        """Close all positions immediately (circuit breaker)."""
        logger.critical("EMERGENCY CLOSE ALL POSITIONS")

        # Cancel all orders for all symbols
        for symbol in self.config.symbols:
            try:
                self.connector.cancel_all_orders(symbol)
            except Exception as e:
                logger.error(f"Failed to cancel orders for {symbol}: {e}")

        # Close all positions
        for key, pos_data in list(self.state.state.open_positions.items()):
            pos = OpenPosition(**pos_data)
            try:
                side = 'long' if pos.direction == 'LONG' else 'short'
                self.connector.close_position(pos.symbol, side, pos.quantity)
                price = self.connector.get_ticker_price(pos.symbol)
                self.state.close_position(key, price, 'circuit_breaker')
            except Exception as e:
                logger.error(f"Failed to close {key}: {e}")
