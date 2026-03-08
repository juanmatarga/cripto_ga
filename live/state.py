"""
Persistent state for the live trading bot.

Saves to JSON file. Designed to be restart-safe — all state needed to
resume trading is persisted after every state change.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent / 'live_state.json'


@dataclass
class OpenPosition:
    """An open position managed by the bot."""
    strategy_key: str
    symbol: str                 # e.g. 'BTC/USDT:USDT'
    direction: str              # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float             # In base currency
    entry_time: str             # ISO format
    entry_bar_time: str         # Candle time that triggered the signal
    atr_at_entry: float
    stop_loss: float
    take_profit: Optional[float]
    trail_atr_mult: float
    initial_stop: float         # Original SL (before trail)
    best_price: float           # For trailing stop tracking
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    notional_usdt: float = 0.0


@dataclass
class ClosedTrade:
    """A completed trade for logging."""
    strategy_key: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_type: str              # 'stop', 'target', 'trail', 'manual', 'circuit_breaker'
    bars_held: int = 0


@dataclass
class BotState:
    """Complete bot state — persisted to disk."""
    # Metadata
    started_at: str = ""
    last_update: str = ""
    trading_mode: str = "testnet"

    # Capital tracking
    initial_capital: float = 0.0
    peak_capital: float = 0.0
    current_capital: float = 0.0

    # Circuit breakers
    is_halted: bool = False
    halt_reason: str = ""
    daily_pnl: float = 0.0
    daily_reset_date: str = ""

    # Positions (strategy_key -> OpenPosition)
    open_positions: Dict[str, dict] = field(default_factory=dict)

    # Trade history
    closed_trades: List[dict] = field(default_factory=list)

    # Last processed candle (to avoid re-processing)
    last_candle_time: str = ""

    # Counters
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl_usdt: float = 0.0


class StateManager:
    """Manages persistent bot state."""

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> BotState:
        """Load state from disk, or create new."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                state = BotState(**{
                    k: v for k, v in data.items()
                    if k in BotState.__dataclass_fields__
                })
                logger.info(f"State loaded: {len(state.open_positions)} open positions, "
                            f"{state.total_trades} total trades")
                return state
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Corrupted state file, creating new: {e}")
        return BotState()

    def save(self):
        """Persist state to disk."""
        self.state.last_update = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        logger.debug("State saved")

    def initialize(self, capital: float, trading_mode: str):
        """Initialize state for a new session."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.state.started_at:
            self.state.started_at = now
            self.state.initial_capital = capital
            self.state.peak_capital = capital
        self.state.current_capital = capital
        self.state.trading_mode = trading_mode

        # Reset daily PnL if new day
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self.state.daily_reset_date != today:
            self.state.daily_pnl = 0.0
            self.state.daily_reset_date = today
            if self.state.is_halted and self.state.halt_reason.startswith('daily_loss'):
                self.state.is_halted = False
                self.state.halt_reason = ""
                logger.info("Daily loss halt cleared (new day)")

        self.save()

    def has_position(self, strategy_key: str) -> bool:
        """Check if a strategy has an open position."""
        return strategy_key in self.state.open_positions

    def open_position(self, pos: OpenPosition):
        """Record a new open position."""
        self.state.open_positions[pos.strategy_key] = asdict(pos)
        self.save()
        logger.info(f"Position opened: {pos.strategy_key} {pos.direction} "
                     f"@ ${pos.entry_price:,.2f} qty={pos.quantity:.6f}")

    def close_position(self, strategy_key: str, exit_price: float,
                       exit_type: str) -> Optional[ClosedTrade]:
        """Close a position and record the trade."""
        if strategy_key not in self.state.open_positions:
            logger.warning(f"No position to close for {strategy_key}")
            return None

        pos_data = self.state.open_positions.pop(strategy_key)
        pos = OpenPosition(**pos_data)

        # Calculate PnL
        if pos.direction == 'LONG':
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usdt = pnl_pct * pos.notional_usdt

        trade = ClosedTrade(
            strategy_key=strategy_key,
            symbol=getattr(pos, 'symbol', ''),
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl_usdt=pnl_usdt,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc).isoformat(),
            exit_type=exit_type,
        )

        self.state.closed_trades.append(asdict(trade))
        self.state.total_trades += 1
        if pnl_usdt > 0:
            self.state.winning_trades += 1
        self.state.total_pnl_usdt += pnl_usdt
        self.state.daily_pnl += pnl_usdt

        self.save()

        logger.info(f"Trade closed: {strategy_key} {exit_type} "
                     f"PnL=${pnl_usdt:+.2f} ({pnl_pct:+.2%})")
        return trade

    def update_capital(self, new_capital: float):
        """Update current capital and track peak."""
        self.state.current_capital = new_capital
        if new_capital > self.state.peak_capital:
            self.state.peak_capital = new_capital
        self.save()

    def check_circuit_breakers(self, config) -> bool:
        """
        Check if trading should be halted.

        Returns True if trading is allowed, False if halted.
        """
        if self.state.is_halted:
            logger.warning(f"Trading HALTED: {self.state.halt_reason}")
            return False

        # Portfolio drawdown check
        if self.state.peak_capital > 0:
            dd_pct = ((self.state.current_capital - self.state.peak_capital)
                      / self.state.peak_capital * 100)
            if dd_pct < -config.risk.max_portfolio_dd_pct:
                self.state.is_halted = True
                self.state.halt_reason = (
                    f"portfolio_dd: {dd_pct:.1f}% exceeds "
                    f"-{config.risk.max_portfolio_dd_pct}% limit"
                )
                self.save()
                logger.critical(f"CIRCUIT BREAKER: {self.state.halt_reason}")
                return False

        # Daily loss check
        if self.state.initial_capital > 0:
            daily_loss_pct = (self.state.daily_pnl / self.state.initial_capital * 100)
            if daily_loss_pct < -config.risk.max_daily_loss_pct:
                self.state.is_halted = True
                self.state.halt_reason = (
                    f"daily_loss: {daily_loss_pct:.1f}% exceeds "
                    f"-{config.risk.max_daily_loss_pct}% limit"
                )
                self.save()
                logger.critical(f"CIRCUIT BREAKER: {self.state.halt_reason}")
                return False

        return True
