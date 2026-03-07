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
            f"ENTRY {strategy_key}: {direction} "
            f"@ ~${entry_price:,.2f} | "
            f"notional=${notional:,.2f} | "
            f"SL=${sl:,.2f} | "
            f"TP=${tp:,.2f}" if tp else
            f"ENTRY {strategy_key}: {direction} "
            f"@ ~${entry_price:,.2f} | notional=${notional:,.2f} | SL=${sl:,.2f}"
        )

        try:
            # Place market entry order
            side = 'buy' if direction == 'LONG' else 'sell'
            order = self.connector.place_market_order(side, notional)

            actual_price = order['price']
            quantity = order['quantity']

            # Recalculate exits with actual fill price
            tp, sl = self.signals.compute_exit_levels(
                actual_price, atr, direction,
                sc.tp_atr_mult, sc.sl_atr_mult
            )

            # Place SL order on exchange
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            sl_order = self.connector.place_stop_loss(sl_side, quantity, sl)

            # Place TP order if applicable
            tp_order = None
            if tp is not None:
                tp_order = self.connector.place_take_profit(sl_side, quantity, tp)

            # Record position in state
            now = datetime.now(timezone.utc).isoformat()
            pos = OpenPosition(
                strategy_key=strategy_key,
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
                sl_order_id=sl_order.get('id'),
                tp_order_id=tp_order.get('id') if tp_order else None,
                notional_usdt=notional,
            )
            self.state.open_position(pos)

            logger.info(
                f"ENTRY COMPLETE {strategy_key}: {direction} "
                f"@ ${actual_price:,.2f} qty={quantity:.6f} "
                f"SL=${sl:,.2f} TP=${tp:,.2f}" if tp else
                f"ENTRY COMPLETE {strategy_key}: {direction} "
                f"@ ${actual_price:,.2f} qty={quantity:.6f} SL=${sl:,.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Entry failed for {strategy_key}: {e}", exc_info=True)
            return False

    def check_and_close_filled_exits(self):
        """
        Check if any SL/TP orders have been filled by the exchange.
        If so, update state accordingly.
        """
        positions_to_close = []

        for key, pos_data in list(self.state.state.open_positions.items()):
            pos = OpenPosition(**pos_data)

            # Check if SL/TP orders are still open
            try:
                open_orders = self.connector.get_open_orders()
                open_ids = {o['id'] for o in open_orders}

                sl_filled = pos.sl_order_id and pos.sl_order_id not in open_ids
                tp_filled = pos.tp_order_id and pos.tp_order_id not in open_ids

                if sl_filled and tp_filled:
                    # Both gone — check which one filled
                    # The exchange cancels the other when one fills (OCO-like)
                    # We need to check the position to determine
                    positions_to_close.append((key, 'exchange_exit'))
                elif sl_filled:
                    positions_to_close.append((key, 'stop'))
                elif tp_filled:
                    positions_to_close.append((key, 'target'))

            except Exception as e:
                logger.error(f"Error checking orders for {key}: {e}")

        for key, exit_type in positions_to_close:
            try:
                # Get actual exit price from current state
                price = self.connector.get_ticker_price()
                self.state.close_position(key, price, exit_type)

                # Cancel any remaining orders for this position
                self.connector.cancel_all_orders()
            except Exception as e:
                logger.error(f"Error closing {key}: {e}")

    def update_trailing_stops(self, df):
        """
        Update trailing stops for open positions.

        For strategies with trail_atr_mult > 0, we move the stop loss
        to follow the price upward (for longs) or downward (for shorts).
        """
        current_price = float(df['Close'].iloc[-1])

        for key, pos_data in list(self.state.state.open_positions.items()):
            pos = OpenPosition(**pos_data)

            if pos.trail_atr_mult <= 0:
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

                # Update SL order on exchange
                if pos.sl_order_id:
                    try:
                        self.connector.cancel_all_orders()
                        sl_side = 'sell' if pos.direction == 'LONG' else 'buy'
                        new_sl_order = self.connector.place_stop_loss(
                            sl_side, pos.quantity, new_sl
                        )
                        pos_data['sl_order_id'] = new_sl_order['id']

                        # Re-place TP if it existed
                        if pos.take_profit:
                            new_tp_order = self.connector.place_take_profit(
                                sl_side, pos.quantity, pos.take_profit
                            )
                            pos_data['tp_order_id'] = new_tp_order['id']

                        self.state.save()
                        logger.info(
                            f"TRAIL {key}: SL moved "
                            f"${pos.stop_loss:,.2f} → ${new_sl:,.2f} "
                            f"(best=${new_best:,.2f})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to update trail SL for {key}: {e}")

    def emergency_close_all(self):
        """Close all positions immediately (circuit breaker)."""
        logger.critical("EMERGENCY CLOSE ALL POSITIONS")

        # Cancel all orders first
        try:
            self.connector.cancel_all_orders()
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

        # Close all positions
        for key, pos_data in list(self.state.state.open_positions.items()):
            pos = OpenPosition(**pos_data)
            try:
                side = 'long' if pos.direction == 'LONG' else 'short'
                self.connector.close_position(side, pos.quantity)
                price = self.connector.get_ticker_price()
                self.state.close_position(key, price, 'circuit_breaker')
            except Exception as e:
                logger.error(f"Failed to close {key}: {e}")
