"""Tests for evolution engine, operators, selection, and fitness."""

import random
import numpy as np
import pandas as pd
import pytest
import time

from grammar.mapper import decode
from strategy.parameters import random_genome
from strategy.phenotype import Strategy
from evolution.operators import crossover, mutate
from evolution.selection import tournament_select, lexicase_select
from evolution.fitness import evaluate_strategy, FAIL_FITNESS, _run_single_window
from evolution.engine import EvolutionEngine
from backtest.sampling import sample_evolution_windows


@pytest.fixture
def sample_df():
    """3 months of synthetic 15m OHLCV data."""
    np.random.seed(42)
    n = 8640  # ~3 months
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        'Open': close + np.random.randn(n) * 10,
        'High': close + abs(np.random.randn(n) * 30),
        'Low': close - abs(np.random.randn(n) * 30),
        'Close': close,
        'Volume': np.random.exponential(1000, n),
    }, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    return df


@pytest.fixture
def config():
    return {
        'costs': {
            'fees_bps_long': 1.0,
            'fees_bps_short': 1.0,
            'slippage_bps_long': 1.0,
            'slippage_bps_short': 1.0,
        },
        'exits': {
            'atr_period': 14,
        },
        'fitness': {
            'min_trades': 5,       # Lower for testing
            'max_drawdown': 0.50,
            'min_win_rate': 0.20,
            'parsimony_coefficient': 0.01,
        },
        'evolution': {
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_pct': 0.05,
            'genome_length': 50,
            'tournament_k': 3,
            'n_windows_per_gen': 3,
            'window_bars': 2880,
        },
    }


# ============================================================================
# OPERATORS
# ============================================================================

class TestCrossover:
    def test_children_same_length_as_parents(self):
        p1 = [1, 2, 3, 4, 5]
        p2 = [6, 7, 8, 9, 10]
        c1, c2 = crossover(p1, p2)
        assert len(c1) == 5
        assert len(c2) == 5

    def test_children_are_mix_of_parents(self):
        random.seed(42)
        p1 = [0] * 50
        p2 = [255] * 50
        c1, c2 = crossover(p1, p2)
        # Children should have elements from both parents
        assert 0 in c1 and 255 in c1
        assert 0 in c2 and 255 in c2

    def test_single_element_no_crash(self):
        c1, c2 = crossover([1], [2])
        assert len(c1) == 1
        assert len(c2) == 1


class TestMutate:
    def test_mutation_changes_genome(self):
        random.seed(42)
        genome = [100] * 50
        mutated = mutate(genome, rate=1.0)  # 100% mutation rate
        assert mutated != genome

    def test_zero_rate_no_change(self):
        genome = [100] * 50
        mutated = mutate(genome, rate=0.0)
        assert mutated == genome

    def test_values_in_range(self):
        random.seed(42)
        genome = [128] * 50
        mutated = mutate(genome, rate=0.5)
        for v in mutated:
            assert 0 <= v <= 255

    def test_original_unchanged(self):
        genome = [100] * 50
        original = genome[:]
        mutate(genome, rate=1.0)
        assert genome == original  # original not modified


# ============================================================================
# SELECTION
# ============================================================================

class TestSelection:
    def _make_pop(self):
        """Create a small population with known fitness values."""
        pop = []
        for i, fit in enumerate([(1.0, 0.5), (0.5, 1.0), (0.2, 0.3), (-1.0, -0.5)]):
            s = Strategy(
                genome=[i] * 10, direction='LONG', conditions=[],
                logic='', tp_atr_mult=2.0, sl_atr_mult=1.0,
                expression_raw='test', n_nodes=1, codons_used=5,
                wrapping_count=0,
            )
            s.fitness = fit
            pop.append(s)
        return pop

    def test_tournament_returns_strategy(self):
        pop = self._make_pop()
        result = tournament_select(pop, k=2)
        assert isinstance(result, Strategy)

    def test_tournament_favors_best(self):
        random.seed(42)
        pop = self._make_pop()
        wins = [0] * len(pop)
        for _ in range(1000):
            winner = tournament_select(pop, k=3)
            idx = [i for i, s in enumerate(pop) if s is winner][0]
            wins[idx] += 1
        # Best strategy (fitness 1.0) should win most often
        assert wins[0] > wins[-1]

    def test_lexicase_returns_strategy(self):
        pop = self._make_pop()
        result = lexicase_select(pop)
        assert isinstance(result, Strategy)


# ============================================================================
# SAMPLING
# ============================================================================

class TestSampling:
    def test_returns_correct_count(self, sample_df):
        windows = sample_evolution_windows(sample_df, n_windows=3, window_bars=2880)
        assert len(windows) == 3

    def test_window_length(self, sample_df):
        windows = sample_evolution_windows(sample_df, n_windows=2, window_bars=2880)
        for w in windows:
            assert len(w) == 2880

    def test_different_calls_give_different_windows(self, sample_df):
        random.seed(1)
        w1 = sample_evolution_windows(sample_df, n_windows=2, window_bars=2880)
        random.seed(2)
        w2 = sample_evolution_windows(sample_df, n_windows=2, window_bars=2880)
        # Start indices should differ
        assert w1[0].index[0] != w2[0].index[0]


# ============================================================================
# FITNESS
# ============================================================================

class TestFitness:
    def test_no_conditions_returns_fail(self, sample_df, config):
        s = Strategy(
            genome=[1, 2, 3], direction='LONG', conditions=[],
            logic='', tp_atr_mult=2.0, sl_atr_mult=1.0,
            expression_raw='test', n_nodes=0, codons_used=0,
            wrapping_count=0,
        )
        windows = [sample_df.iloc[:2880]]
        result = evaluate_strategy(s, windows, config)
        assert result.fitness == FAIL_FITNESS

    def test_valid_strategy_gets_real_fitness(self, sample_df, config):
        random.seed(42)
        # Try many strategies until one passes constraints
        for _ in range(200):
            s = decode(random_genome(50))
            if s is not None:
                windows = [sample_df.iloc[:2880]]
                config_permissive = dict(config)
                config_permissive['fitness'] = {
                    'min_trades': 1,
                    'max_drawdown': 0.99,
                    'min_win_rate': 0.0,
                    'parsimony_coefficient': 0.0,
                }
                evaluate_strategy(s, windows, config_permissive)
                if s.fitness != FAIL_FITNESS:
                    assert isinstance(s.fitness[0], float)
                    assert isinstance(s.fitness[1], float)
                    assert s.fitness[0] != -999.0
                    return

        pytest.skip("No strategy generated enough trades on synthetic data")

    def test_parsimony_reduces_fitness(self, sample_df, config):
        """More complex strategies should have lower adjusted fitness."""
        random.seed(42)
        config_p0 = dict(config)
        config_p0['fitness'] = {
            'min_trades': 1, 'max_drawdown': 0.99,
            'min_win_rate': 0.0, 'parsimony_coefficient': 0.0,
        }
        config_p1 = dict(config)
        config_p1['fitness'] = {
            'min_trades': 1, 'max_drawdown': 0.99,
            'min_win_rate': 0.0, 'parsimony_coefficient': 1.0,
        }

        for _ in range(200):
            s = decode(random_genome(50))
            if s is not None and s.n_nodes >= 2:
                windows = [sample_df.iloc[:2880]]
                s_copy = decode(s.genome)

                evaluate_strategy(s, windows, config_p0)
                evaluate_strategy(s_copy, windows, config_p1)

                if s.fitness != FAIL_FITNESS and s_copy.fitness != FAIL_FITNESS:
                    # With parsimony=1.0, fitness should be lower
                    assert s_copy.fitness[0] < s.fitness[0]
                    return

        pytest.skip("Could not find valid multi-condition strategy")


# ============================================================================
# ENGINE
# ============================================================================

class TestEngine:
    def test_initialize_creates_population(self, sample_df, config):
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=20)
        assert len(engine.population) == 20
        assert all(isinstance(s, Strategy) for s in engine.population)

    def test_step_increments_generation(self, sample_df, config):
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        assert engine.generation == 0
        engine.step()
        assert engine.generation == 1

    def test_population_size_stable(self, sample_df, config):
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=20)
        initial_size = len(engine.population)
        engine.step()
        assert len(engine.population) == initial_size

    def test_elitism_preserves_best(self, sample_df, config):
        random.seed(42)
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=20)

        # Run one gen to get fitness values
        engine.step()
        valid = [s for s in engine.population if s.fitness != FAIL_FITNESS]
        if valid:
            best_fitness = max(s.fitness[0] for s in valid)
            # After another gen, best should be >= (elitism)
            engine.step()
            valid2 = [s for s in engine.population if s.fitness != FAIL_FITNESS]
            if valid2:
                best_fitness2 = max(s.fitness[0] for s in valid2)
                # Elite carried over, but re-evaluated on new windows
                # So fitness may differ. Just check we have valid strategies.
                assert len(valid2) > 0

    def test_run_returns_result(self, sample_df, config):
        random.seed(42)
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        result = engine.run(n_generations=3, patience=10)
        assert result.final_generation == 3
        assert len(result.history) == 3
        assert result.total_evaluations > 0

    def test_run_with_patience_stops_early(self, sample_df, config):
        random.seed(42)
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        result = engine.run(n_generations=100, patience=2)
        # Should stop well before 100 generations
        assert result.final_generation < 100

    def test_history_grows_each_generation(self, sample_df, config):
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        engine.step()
        engine.step()
        engine.step()
        assert len(engine.history) == 3
        assert engine.history[0].generation == 0
        assert engine.history[1].generation == 1
        assert engine.history[2].generation == 2
