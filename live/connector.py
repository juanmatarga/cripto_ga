"""
Binance Futures connector via ccxt.

Handles: connection, data fetching, order placement, position management.
All methods include error handling and logging.
"""

import ccxt
import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, Dict, List
from datetime import datetime, timezone

from live.config import LiveConfig

logger = logging.getLogger(__name__)

# Binance Futures testnet URL
TESTNET_URL = 'https://testnet.binancefuture.com'


class BinanceConnector:
    """Thread-safe Binance Futures connection."""

    def __init__(self, config: LiveConfig):
        self.config = config
        self.exchange = self._create_exchange()
        self._markets_loaded = False

    @staticmethod
    def _precision_to_decimals(precision) -> int:
        """Convert ccxt precision to decimal places for round().
        ccxt 4.x may return precision as float step (0.001) or int decimals (3)."""
        if isinstance(precision, int):
            return precision
        if isinstance(precision, float) and precision > 0:
            import math
            return max(0, -int(math.floor(math.log10(precision))))
        return 3  # fallback

    def _create_exchange(self) -> ccxt.binance:
        """Initialize ccxt exchange instance."""
        params = {
            'apiKey': self.config.active_api_key,
            'secret': self.config.active_api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
        }

        if self.config.is_testnet:
            params['options']['sandboxMode'] = True

        exchange = ccxt.binance(params)

        if self.config.is_testnet:
            exchange.set_sandbox_mode(True)

        mode = "TESTNET" if self.config.is_testnet else "LIVE"
        logger.info(f"Binance Futures connector initialized ({mode})")
        return exchange

    def _ensure_markets(self):
        """Load markets if not already loaded."""
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    # ================================================================
    # DATA
    # ================================================================

    def fetch_ohlcv(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """
        Fetch recent OHLCV candles for a symbol.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (UTC)

        Only returns CLOSED candles (drops the last forming candle).
        """
        self._ensure_markets()

        raw = self.exchange.fetch_ohlcv(
            symbol,
            timeframe=self.config.timeframe,
            limit=limit + 1,  # +1 because we drop the forming candle
        )

        if not raw:
            raise RuntimeError("No OHLCV data received from Binance")

        df = pd.DataFrame(raw, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)

        # Drop the last candle (it's still forming, not yet closed)
        # Only drop if the last candle's timestamp + timeframe > now
        now = datetime.now(timezone.utc)
        tf_seconds = self._timeframe_to_seconds(self.config.timeframe)
        last_candle_close = df.index[-1] + pd.Timedelta(seconds=tf_seconds)

        if last_candle_close > now:
            df = df.iloc[:-1]
            logger.debug(f"Dropped forming candle. Last closed: {df.index[-1]}")

        logger.info(f"Fetched {len(df)} closed candles. "
                     f"Range: {df.index[0]} to {df.index[-1]}. "
                     f"Last close: ${df['Close'].iloc[-1]:,.2f}")
        return df

    def _timeframe_to_seconds(self, tf: str) -> int:
        """Convert timeframe string to seconds."""
        multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
        unit = tf[-1]
        value = int(tf[:-1])
        return value * multipliers.get(unit, 60)

    # ================================================================
    # ACCOUNT
    # ================================================================

    def get_balance(self) -> Dict:
        """Get futures account balance."""
        self._ensure_markets()
        balance = self.exchange.fetch_balance({'type': 'future'})
        usdt = balance.get('USDT', {})
        return {
            'total': float(usdt.get('total', 0)),
            'free': float(usdt.get('free', 0)),
            'used': float(usdt.get('used', 0)),
        }

    def get_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Get all open positions across given symbols (or all if None)."""
        self._ensure_markets()
        positions = self.exchange.fetch_positions(symbols)
        open_positions = []
        for p in positions:
            contracts = float(p.get('contracts', 0))
            if contracts > 0:
                open_positions.append({
                    'symbol': p['symbol'],
                    'side': p['side'],  # 'long' or 'short'
                    'contracts': contracts,
                    'notional': float(p.get('notional') or 0),
                    'entry_price': float(p.get('entryPrice') or 0),
                    'mark_price': float(p.get('markPrice') or 0),
                    'unrealized_pnl': float(p.get('unrealizedPnl') or 0),
                    'leverage': int(p.get('leverage') or 1),
                    'liquidation_price': float(p.get('liquidationPrice') or 0),
                })
        return open_positions

    def get_open_orders(self, symbol: str) -> List[Dict]:
        """Get all open orders for a symbol."""
        self._ensure_markets()
        orders = self.exchange.fetch_open_orders(symbol)
        return [{
            'id': o['id'],
            'type': o['type'],
            'side': o['side'],
            'price': o.get('price'),
            'stop_price': o.get('stopPrice'),
            'amount': o.get('amount'),
            'status': o['status'],
        } for o in orders]

    # ================================================================
    # LEVERAGE
    # ================================================================

    def set_leverage(self, leverage: int, symbol: str):
        """Set leverage for a trading symbol."""
        self._ensure_markets()
        try:
            # Binance requires setting margin mode first
            try:
                self.exchange.set_margin_mode('isolated', symbol)
                logger.info(f"Set margin mode to ISOLATED for {symbol}")
            except ccxt.ExchangeError as e:
                if 'No need to change margin type' not in str(e):
                    logger.warning(f"Could not set margin mode for {symbol}: {e}")

            self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
        except ccxt.ExchangeError as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            raise

    # ================================================================
    # ORDERS
    # ================================================================

    def place_market_order(self, symbol: str, side: str, amount_usdt: float) -> Dict:
        """
        Place a market order.

        Args:
            symbol: Trading symbol (e.g. 'BTC/USDT:USDT')
            side: 'buy' or 'sell'
            amount_usdt: Notional value in USDT

        Returns:
            Order info dict
        """
        self._ensure_markets()

        # Get current price to calculate quantity
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker['last']

        # Calculate quantity in base currency
        quantity = amount_usdt / price

        # Apply market precision
        market = self.exchange.market(symbol)
        decimals = self._precision_to_decimals(
            market.get('precision', {}).get('amount', 3)
        )
        quantity = round(quantity, decimals)

        if quantity * price < self.config.risk.min_order_usdt:
            raise ValueError(
                f"Order too small: {quantity} × ${price:.2f} = "
                f"${quantity * price:.2f} < min ${self.config.risk.min_order_usdt}"
            )

        base = symbol.split('/')[0]
        logger.info(f"Placing {side.upper()} market order: {quantity} {base} "
                     f"(~${quantity * price:,.2f} notional, price ${price:,.2f})")

        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=quantity,
        )

        logger.info(f"Order filled: id={order['id']}, "
                     f"avg_price=${order.get('average', price):,.2f}, "
                     f"filled={order.get('filled', quantity)}")
        return {
            'id': order['id'],
            'side': side,
            'quantity': float(order.get('filled', quantity)),
            'price': float(order.get('average', price)),
            'cost': float(order.get('cost', quantity * price)),
            'timestamp': order.get('timestamp'),
        }

    def place_stop_loss(self, symbol: str, side: str, quantity: float,
                        stop_price: float) -> Dict:
        """
        Place a stop-loss order via ccxt Algo endpoint.

        Args:
            symbol: Trading symbol
            side: 'buy' (to close short) or 'sell' (to close long)
            quantity: Amount in base currency
            stop_price: Trigger price
        """
        self._ensure_markets()

        market = self.exchange.market(symbol)
        price_decimals = self._precision_to_decimals(
            market.get('precision', {}).get('price', 2)
        )
        stop_price = round(stop_price, price_decimals)

        base = symbol.split('/')[0]
        logger.info(f"Placing SL: {side.upper()} stop-market {quantity} {base} "
                     f"@ ${stop_price:,.2f}")

        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=quantity,
            params={
                'stopLossPrice': stop_price,
                'reduceOnly': True,
            }
        )

        logger.info(f"SL placed: id={order['id']}")
        return {'id': order['id'], 'stop_price': stop_price, 'type': 'stop_loss'}

    def place_take_profit(self, symbol: str, side: str, quantity: float,
                          stop_price: float) -> Dict:
        """
        Place a take-profit order via ccxt Algo endpoint.

        Args:
            symbol: Trading symbol
            side: 'buy' (to close short) or 'sell' (to close long)
            quantity: Amount in base currency
            stop_price: Trigger price
        """
        self._ensure_markets()

        market = self.exchange.market(symbol)
        price_decimals = self._precision_to_decimals(
            market.get('precision', {}).get('price', 2)
        )
        stop_price = round(stop_price, price_decimals)

        base = symbol.split('/')[0]
        logger.info(f"Placing TP: {side.upper()} take-profit-market {quantity} {base} "
                     f"@ ${stop_price:,.2f}")

        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=quantity,
            params={
                'takeProfitPrice': stop_price,
                'reduceOnly': True,
            }
        )

        logger.info(f"TP placed: id={order['id']}")
        return {'id': order['id'], 'stop_price': stop_price, 'type': 'take_profit'}

    def cancel_order_by_id(self, order_id: str, symbol: str,
                           is_trigger: bool = True):
        """
        Cancel a specific order.

        Args:
            order_id: The order ID to cancel
            symbol: Trading symbol
            is_trigger: True for SL/TP (algo) orders, False for regular orders
        """
        self._ensure_markets()
        try:
            params = {'trigger': True} if is_trigger else {}
            self.exchange.cancel_order(order_id, symbol, params=params)
            logger.info(f"Order cancelled: {order_id} (trigger={is_trigger})")
        except ccxt.OrderNotFound:
            logger.debug(f"Order {order_id} already filled or cancelled")
        except ccxt.ExchangeError as e:
            logger.warning(f"Cancel order {order_id} failed: {e}")

    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol (regular + algo/trigger)."""
        self._ensure_markets()
        try:
            self.exchange.cancel_all_orders(symbol)
            logger.info(f"All regular orders cancelled for {symbol}")
        except ccxt.ExchangeError as e:
            logger.warning(f"Cancel regular orders failed for {symbol}: {e}")

        # Also cancel algo/conditional orders
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for o in open_orders:
                try:
                    self.cancel_order_by_id(o['id'], symbol, is_trigger=True)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Cancel algo orders failed for {symbol}: {e}")

    def close_position(self, symbol: str, side: str, quantity: float) -> Dict:
        """
        Close a position with a market order.

        Args:
            symbol: Trading symbol
            side: Current position side ('long' or 'short')
            quantity: Amount to close
        """
        close_side = 'sell' if side == 'long' else 'buy'
        base = symbol.split('/')[0]

        logger.info(f"Closing {side} position: {close_side} {quantity} {base} ({symbol})")

        order = self.exchange.create_order(
            symbol=symbol,
            type='market',
            side=close_side,
            amount=quantity,
            params={'reduceOnly': True},
        )

        logger.info(f"Position closed: id={order['id']}, "
                     f"price=${order.get('average', 0):,.2f}")
        return {
            'id': order['id'],
            'side': close_side,
            'quantity': float(order.get('filled', quantity)),
            'price': float(order.get('average', 0)),
        }

    def get_ticker_price(self, symbol: str) -> float:
        """Get current mark price for a symbol."""
        self._ensure_markets()
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker['last'])
