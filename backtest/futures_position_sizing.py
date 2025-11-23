"""
Futures Position Sizing with Leverage and Risk Management.

Simulates realistic futures trading with:
- 10x leverage
- 2% risk per trade
- Dynamic position sizing based on equity
- Proper margin calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FuturesPositionManager:
    """Manage position sizing for leveraged futures trading."""

    def __init__(self,
                 initial_capital: float = 1000.0,
                 risk_per_trade_pct: float = 0.02,
                 leverage: float = 10.0):
        """
        Initialize position manager.

        Args:
            initial_capital: Starting capital in USD
            risk_per_trade_pct: Risk per trade as % of equity (0.02 = 2%)
            leverage: Leverage multiplier (10 = 10x)
        """
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.leverage = leverage
        self.current_equity = initial_capital

        # Track all positions
        self.open_positions = []
        self.closed_trades = []

        logger.info(f"FuturesPositionManager initialized: ${initial_capital} @ {leverage}x leverage, {risk_per_trade_pct*100}% risk")

    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss_price: float,
                                direction: str) -> Dict:
        """
        Calculate position size based on risk and ATR-based stop loss.

        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price (already calculated from ATR)
            direction: 'LONG' or 'SHORT'

        Returns:
            Dict with position details: {
                'notional_value': float,  # Total exposure with leverage
                'margin_required': float,  # Actual margin from equity
                'contracts': float,  # Number of contracts (for crypto, can be fractional)
                'risk_usd': float  # Dollar risk on this trade
            }
        """
        # Calculate risk in USD
        risk_usd = self.current_equity * self.risk_per_trade_pct

        # Calculate stop distance
        if direction == 'LONG':
            stop_distance = entry_price - stop_loss_price
        else:  # SHORT
            stop_distance = stop_loss_price - entry_price

        if stop_distance <= 0:
            logger.warning(f"Invalid stop distance: {stop_distance}, using default 2%")
            stop_distance = entry_price * 0.02

        # Position size: risk_usd / (stop_distance / entry_price)
        # This gives us the notional value that would lose risk_usd if SL hits
        risk_pct = stop_distance / entry_price
        notional_value = risk_usd / risk_pct

        # Apply leverage cap (max 10x of current equity)
        max_notional = self.current_equity * self.leverage
        notional_value = min(notional_value, max_notional)

        # Margin required (10% of notional for 10x leverage)
        margin_required = notional_value / self.leverage

        # Check if we have enough margin
        available_margin = self.current_equity - sum(pos['margin_required'] for pos in self.open_positions)

        if margin_required > available_margin:
            # Scale down position to available margin
            margin_required = available_margin * 0.95  # Use 95% max
            notional_value = margin_required * self.leverage
            logger.debug(f"Position scaled down to available margin: ${margin_required:.2f}")

        # Contracts (fractional for crypto)
        contracts = notional_value / entry_price

        position_details = {
            'notional_value': notional_value,
            'margin_required': margin_required,
            'contracts': contracts,
            'risk_usd': risk_usd,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'direction': direction
        }

        logger.debug(f"Position calculated: {direction} {contracts:.4f} contracts @ ${entry_price:.2f}, "
                    f"margin ${margin_required:.2f}, risk ${risk_usd:.2f}")

        return position_details

    def open_position(self,
                     timestamp: pd.Timestamp,
                     entry_price: float,
                     stop_loss_price: float,
                     take_profit_price: float,
                     direction: str) -> Dict:
        """
        Open a new leveraged position.

        Returns:
            Position dict with all details
        """
        position = self.calculate_position_size(entry_price, stop_loss_price, direction)

        # Add trade metadata
        position['timestamp'] = timestamp
        position['take_profit_price'] = take_profit_price
        position['status'] = 'OPEN'
        position['pnl'] = 0.0

        # Add to open positions
        self.open_positions.append(position)

        logger.debug(f"Opened {direction} position: {position['contracts']:.4f} contracts, "
                    f"margin ${position['margin_required']:.2f}")

        return position

    def close_position(self,
                      position: Dict,
                      exit_timestamp: pd.Timestamp,
                      exit_price: float,
                      exit_reason: str) -> Dict:
        """
        Close an open position and update equity.

        Args:
            position: Position dict from open_position
            exit_timestamp: When position closed
            exit_price: Exit price
            exit_reason: 'TP', 'SL', or 'TIME'

        Returns:
            Closed trade dict with PnL
        """
        # Calculate PnL
        if position['direction'] == 'LONG':
            price_change = exit_price - position['entry_price']
        else:  # SHORT
            price_change = position['entry_price'] - exit_price

        pnl_usd = price_change * position['contracts']
        pnl_pct = pnl_usd / position['margin_required']  # Return on margin

        # Update equity
        self.current_equity += pnl_usd

        # Create trade record
        trade = {
            **position,
            'exit_timestamp': exit_timestamp,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'holding_bars': (exit_timestamp - position['timestamp']).total_seconds() / 900,  # 15min bars
            'equity_after': self.current_equity,
            'status': 'CLOSED'
        }

        # Remove from open, add to closed
        self.open_positions.remove(position)
        self.closed_trades.append(trade)

        logger.debug(f"Closed {position['direction']} position: PnL ${pnl_usd:.2f} ({pnl_pct*100:.1f}%), "
                    f"equity now ${self.current_equity:.2f}")

        return trade

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve from closed trades.

        Returns:
            DataFrame with columns: timestamp, equity, trade_pnl, cumulative_pnl
        """
        if not self.closed_trades:
            return pd.DataFrame({
                'timestamp': [],
                'equity': [],
                'trade_pnl': [],
                'cumulative_pnl': []
            })

        df = pd.DataFrame(self.closed_trades)

        equity_curve = pd.DataFrame({
            'timestamp': df['exit_timestamp'],
            'equity': df['equity_after'],
            'trade_pnl': df['pnl_usd'],
            'cumulative_pnl': df['pnl_usd'].cumsum()
        })

        return equity_curve

    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'final_equity': self.current_equity,
                'total_return_pct': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0
            }

        df = pd.DataFrame(self.closed_trades)

        wins = df[df['pnl_usd'] > 0]
        losses = df[df['pnl_usd'] <= 0]

        equity_curve = df['equity_after'].values
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max

        metrics = {
            'total_trades': len(df),
            'final_equity': self.current_equity,
            'total_return_pct': (self.current_equity - self.initial_capital) / self.initial_capital,
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'avg_win': wins['pnl_usd'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl_usd'].mean() if len(losses) > 0 else 0,
            'profit_factor': abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 and losses['pnl_usd'].sum() != 0 else 0,
            'max_drawdown_pct': abs(drawdown.min()) if len(drawdown) > 0 else 0
        }

        return metrics
