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

    def fetch_ohlcv(self, limit: int = 200) -> pd.DataFrame:
        """
        Fetch recent OHLCV candles.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (UTC)

        Only returns CLOSED candles (drops the last forming candle).
        """
        self._ensure_markets()

        raw = self.exchange.fetch_ohlcv(
            self.config.symbol,
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

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        self._ensure_markets()
        positions = self.exchange.fetch_positions([self.config.symbol])
        open_positions = []
        for p in positions:
            contracts = float(p.get('contracts', 0))
            if contracts > 0:
                open_positions.append({
                    'symbol': p['symbol'],
                    'side': p['side'],  # 'long' or 'short'
                    'contracts': contracts,
                    'notional': float(p.get('notional', 0)),
                    'entry_price': float(p.get('entryPrice', 0)),
                    'mark_price': float(p.get('markPrice', 0)),
                    'unrealized_pnl': float(p.get('unrealizedPnl', 0)),
                    'leverage': int(p.get('leverage', 1)),
                    'liquidation_price': float(p.get('liquidationPrice', 0) or 0),
                })
        return open_positions

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders for the symbol."""
        self._ensure_markets()
        orders = self.exchange.fetch_open_orders(self.config.symbol)
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

    def set_leverage(self, leverage: int):
        """Set leverage for the trading symbol."""
        self._ensure_markets()
        try:
            # Binance requires setting margin mode first
            try:
                self.exchange.set_margin_mode('isolated', self.config.symbol)
                logger.info(f"Set margin mode to ISOLATED for {self.config.symbol}")
            except ccxt.ExchangeError as e:
                if 'No need to change margin type' not in str(e):
                    logger.warning(f"Could not set margin mode: {e}")

            self.exchange.set_leverage(leverage, self.config.symbol)
            logger.info(f"Leverage set to {leverage}x for {self.config.symbol}")
        except ccxt.ExchangeError as e:
            logger.error(f"Failed to set leverage: {e}")
            raise

    # ================================================================
    # ORDERS
    # ================================================================

    def place_market_order(self, side: str, amount_usdt: float) -> Dict:
        """
        Place a market order.

        Args:
            side: 'buy' or 'sell'
            amount_usdt: Notional value in USDT

        Returns:
            Order info dict
        """
        self._ensure_markets()

        # Get current price to calculate quantity
        ticker = self.exchange.fetch_ticker(self.config.symbol)
        price = ticker['last']

        # Calculate quantity in BTC (or base currency)
        # For futures, amount is in contracts (= base currency units)
        quantity = amount_usdt / price

        # Binance BTC futures: min 0.001 BTC, step 0.001
        market = self.exchange.market(self.config.symbol)
        precision = market.get('precision', {}).get('amount', 3)
        quantity = round(quantity, precision)

        if quantity * price < self.config.risk.min_order_usdt:
            raise ValueError(
                f"Order too small: {quantity} × ${price:.2f} = "
                f"${quantity * price:.2f} < min ${self.config.risk.min_order_usdt}"
            )

        logger.info(f"Placing {side.upper()} market order: {quantity} BTC "
                     f"(~${quantity * price:,.2f} notional, price ${price:,.2f})")

        order = self.exchange.create_order(
            symbol=self.config.symbol,
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

    def place_stop_loss(self, side: str, quantity: float,
                        stop_price: float) -> Dict:
        """
        Place a stop-loss order (stop-market).

        Args:
            side: 'buy' (to close short) or 'sell' (to close long)
            quantity: Amount in base currency
            stop_price: Trigger price
        """
        self._ensure_markets()

        market = self.exchange.market(self.config.symbol)
        price_precision = market.get('precision', {}).get('price', 2)
        stop_price = round(stop_price, price_precision)

        logger.info(f"Placing SL: {side.upper()} stop-market {quantity} BTC "
                     f"@ ${stop_price:,.2f}")

        order = self.exchange.create_order(
            symbol=self.config.symbol,
            type='stop_market',
            side=side,
            amount=quantity,
            params={
                'stopPrice': stop_price,
                'closePosition': False,
                'reduceOnly': True,
            }
        )

        logger.info(f"SL placed: id={order['id']}")
        return {'id': order['id'], 'stop_price': stop_price, 'type': 'stop_loss'}

    def place_take_profit(self, side: str, quantity: float,
                          stop_price: float) -> Dict:
        """
        Place a take-profit order (take-profit-market).

        Args:
            side: 'buy' (to close short) or 'sell' (to close long)
            quantity: Amount in base currency
            stop_price: Trigger price
        """
        self._ensure_markets()

        market = self.exchange.market(self.config.symbol)
        price_precision = market.get('precision', {}).get('price', 2)
        stop_price = round(stop_price, price_precision)

        logger.info(f"Placing TP: {side.upper()} take-profit-market {quantity} BTC "
                     f"@ ${stop_price:,.2f}")

        order = self.exchange.create_order(
            symbol=self.config.symbol,
            type='take_profit_market',
            side=side,
            amount=quantity,
            params={
                'stopPrice': stop_price,
                'closePosition': False,
                'reduceOnly': True,
            }
        )

        logger.info(f"TP placed: id={order['id']}")
        return {'id': order['id'], 'stop_price': stop_price, 'type': 'take_profit'}

    def cancel_all_orders(self):
        """Cancel all open orders for the symbol."""
        self._ensure_markets()
        try:
            self.exchange.cancel_all_orders(self.config.symbol)
            logger.info(f"All orders cancelled for {self.config.symbol}")
        except ccxt.ExchangeError as e:
            logger.warning(f"Cancel all orders failed: {e}")

    def close_position(self, side: str, quantity: float) -> Dict:
        """
        Close a position with a market order.

        Args:
            side: Current position side ('long' or 'short')
            quantity: Amount to close
        """
        # To close a long, we sell. To close a short, we buy.
        close_side = 'sell' if side == 'long' else 'buy'

        logger.info(f"Closing {side} position: {close_side} {quantity} BTC")

        order = self.exchange.create_order(
            symbol=self.config.symbol,
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

    def get_ticker_price(self) -> float:
        """Get current mark price."""
        self._ensure_markets()
        ticker = self.exchange.fetch_ticker(self.config.symbol)
        return float(ticker['last'])
