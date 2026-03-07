"""Tests for live trading modules (without actual exchange connection)."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from live.config import StrategyConfig, RiskConfig, LiveConfig, load_strategies_from_results
from live.signals import LiveSignalEngine
from live.state import StateManager, OpenPosition, BotState


# ================================================================
# CONFIG
# ================================================================

class TestConfig:
    def test_load_strategies_from_results(self):
        """Should load the 3 HIGH_RETURN portfolio strategies."""
        strategies = load_strategies_from_results()
        assert len(strategies) == 3

        keys = {s.key for s in strategies}
        assert 'seed123_s2' in keys
        assert 'seed123_s0' in keys
        assert 'seed777_s4' in keys

    def test_strategy_configs_valid(self):
        """Each strategy should have valid exit parameters."""
        strategies = load_strategies_from_results()
        for s in strategies:
            assert s.direction in ('LONG', 'SHORT')
            assert s.sl_atr_mult > 0
            assert s.tp_atr_mult > 0 or s.trail_atr_mult > 0
            assert len(s.genome) > 0
            assert 0 < s.weight <= 1.0

    def test_risk_config_defaults(self):
        """Risk config should have safe defaults."""
        r = RiskConfig()
        assert r.leverage >= 1
        assert r.max_portfolio_dd_pct > 0
        assert r.max_daily_loss_pct > 0
        assert r.max_position_pct <= 50
        assert r.max_open_positions >= 1


# ================================================================
# SIGNALS
# ================================================================

@pytest.fixture
def sample_ohlcv():
    """200 bars of synthetic 15m data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2025-01-01', periods=n, freq='15min')
    close = 90000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        'Open': close + np.random.randn(n) * 20,
        'High': close + abs(np.random.randn(n) * 50),
        'Low': close - abs(np.random.randn(n) * 50),
        'Close': close,
        'Volume': np.random.exponential(1000, n),
    }, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    return df


class TestSignals:
    def test_signal_engine_initialization(self):
        """Should decode all 3 strategies successfully."""
        strategies = load_strategies_from_results()
        engine = LiveSignalEngine(strategies)
        assert len(engine.decoded_strategies) == 3

    def test_evaluate_returns_all_strategies(self, sample_ohlcv):
        """Evaluate should return results for all strategies."""
        strategies = load_strategies_from_results()
        engine = LiveSignalEngine(strategies)
        results = engine.evaluate(sample_ohlcv)

        assert len(results) == 3
        for key, data in results.items():
            assert 'signal' in data
            assert 'new_signal' in data
            assert 'atr' in data
            assert isinstance(data['signal'], bool)
            assert data['atr'] > 0

    def test_compute_exit_levels_long(self):
        """Exit levels for LONG should have SL below and TP above entry."""
        strategies = load_strategies_from_results()
        engine = LiveSignalEngine(strategies)

        tp, sl = engine.compute_exit_levels(100000, 500, 'LONG', 3.0, 1.0)
        assert sl < 100000
        assert tp > 100000
        assert sl == 100000 - 1.0 * 500
        assert tp == 100000 + 3.0 * 500

    def test_compute_exit_levels_short(self):
        """Exit levels for SHORT should have SL above and TP below entry."""
        strategies = load_strategies_from_results()
        engine = LiveSignalEngine(strategies)

        tp, sl = engine.compute_exit_levels(100000, 500, 'SHORT', 3.0, 1.0)
        assert sl > 100000
        assert tp < 100000
        assert sl == 100000 + 1.0 * 500
        assert tp == 100000 - 3.0 * 500


# ================================================================
# STATE
# ================================================================

class TestState:
    def test_state_persistence(self):
        """State should survive save/reload cycle."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        sm = StateManager(path)
        sm.initialize(1000.0, 'testnet')
        sm.state.total_trades = 5
        sm.save()

        # Reload
        sm2 = StateManager(path)
        assert sm2.state.initial_capital == 1000.0
        assert sm2.state.total_trades == 5
        assert sm2.state.trading_mode == 'testnet'

        path.unlink()

    def test_open_and_close_position(self):
        """Should track positions and calculate PnL correctly."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        sm = StateManager(path)
        sm.initialize(1000.0, 'testnet')

        pos = OpenPosition(
            strategy_key='test_s1',
            direction='LONG',
            entry_price=100000,
            quantity=0.01,
            entry_time='2026-01-01T00:00:00',
            entry_bar_time='2026-01-01T00:00:00',
            atr_at_entry=500,
            stop_loss=99500,
            take_profit=101500,
            trail_atr_mult=0,
            initial_stop=99500,
            best_price=100000,
            notional_usdt=5000,
        )

        sm.open_position(pos)
        assert sm.has_position('test_s1')

        trade = sm.close_position('test_s1', 101000, 'target')
        assert not sm.has_position('test_s1')
        assert trade.pnl_pct == pytest.approx(0.01, abs=0.001)
        assert trade.pnl_usdt == pytest.approx(50.0, abs=1.0)
        assert sm.state.total_trades == 1
        assert sm.state.winning_trades == 1

        path.unlink()

    def test_short_pnl_calculation(self):
        """SHORT PnL should be positive when price drops."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        sm = StateManager(path)
        sm.initialize(1000.0, 'testnet')

        pos = OpenPosition(
            strategy_key='test_short',
            direction='SHORT',
            entry_price=100000,
            quantity=0.01,
            entry_time='2026-01-01T00:00:00',
            entry_bar_time='2026-01-01T00:00:00',
            atr_at_entry=500,
            stop_loss=100500,
            take_profit=99000,
            trail_atr_mult=0,
            initial_stop=100500,
            best_price=100000,
            notional_usdt=5000,
        )
        sm.open_position(pos)

        # Price drops to 99000 — profitable for SHORT
        trade = sm.close_position('test_short', 99000, 'target')
        assert trade.pnl_pct == pytest.approx(0.01, abs=0.001)
        assert trade.pnl_usdt > 0

        path.unlink()

    def test_circuit_breaker_portfolio_dd(self):
        """Should halt trading when portfolio drawdown exceeds limit."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        sm = StateManager(path)
        sm.initialize(1000.0, 'testnet')
        sm.state.peak_capital = 1000.0
        sm.state.current_capital = 880.0  # -12% DD

        config = LiveConfig()
        config.risk.max_portfolio_dd_pct = 10.0

        allowed = sm.check_circuit_breakers(config)
        assert not allowed
        assert sm.state.is_halted
        assert 'portfolio_dd' in sm.state.halt_reason

        path.unlink()

    def test_circuit_breaker_daily_loss(self):
        """Should halt trading when daily loss exceeds limit."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = Path(f.name)

        sm = StateManager(path)
        sm.initialize(1000.0, 'testnet')
        sm.state.daily_pnl = -40.0  # -4% daily loss

        config = LiveConfig()
        config.risk.max_daily_loss_pct = 3.0

        allowed = sm.check_circuit_breakers(config)
        assert not allowed
        assert sm.state.is_halted
        assert 'daily_loss' in sm.state.halt_reason

        path.unlink()
