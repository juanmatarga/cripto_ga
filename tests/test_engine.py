"""Tests for NSGA-II Evolution Engine."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    """3 months of synthetic OHLCV data for testing."""
    n = 20000
    np.random.seed(42)
    idx = pd.date_range('2024-01-01', periods=n, freq='15min')
    close = np.random.randn(n).cumsum() + 50000
    return pd.DataFrame({
        'Open': close + np.random.randn(n) * 10,
        'High': close + abs(np.random.randn(n) * 50),
        'Low': close - abs(np.random.randn(n) * 50),
        'Close': close,
        'Volume': np.random.rand(n) * 1e6,
    }, index=idx)


@pytest.fixture
def config():
    return {
        'evolution': {
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'genome_length': 50,
            'n_windows_per_gen': 3,
            'window_bars': 4320,
            'max_generations': 5,
            'archive_parent_pct': 0.10,
        },
        'costs': {'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
                  'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0},
        'exits': {'atr_period': 14},
        'fitness': {'parsimony_coefficient': 0.02},
    }


class TestInitialize:
    def test_creates_valid_population(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        assert len(engine.population) == 10
        for s in engine.population:
            assert s.direction in ('LONG', 'SHORT')
            assert len(s.conditions) > 0

    def test_handles_small_target(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=3)
        assert len(engine.population) == 3


class TestStep:
    def test_returns_generation_stats(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        stats = engine.step()
        assert stats.generation == 0
        assert hasattr(stats, 'best_sortino')
        assert hasattr(stats, 'best_return')
        assert hasattr(stats, 'front1_size')
        assert stats.total_count == 10

    def test_population_size_maintained(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        engine.step()
        assert len(engine.population) == 10

    def test_generation_increments(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        engine.step()
        assert engine.generation == 1
        engine.step()
        assert engine.generation == 2


class TestRun:
    def test_returns_evolution_result(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        result = engine.run(n_generations=3, patience=10)
        assert hasattr(result, 'pareto_front')
        assert hasattr(result, 'archive')
        assert result.total_evaluations > 0
        assert result.final_generation > 0
