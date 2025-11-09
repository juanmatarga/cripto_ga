"""
Integration Tests - End-to-end pipeline validation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def small_dataset():
    """Dataset pequeño para testing rápido."""
    dates = pd.date_range('2024-01-01', '2024-06-01', freq='1H')
    data = pd.DataFrame({
        'Open': 50000 + np.random.randn(len(dates)) * 1000,
        'High': 51000 + np.random.randn(len(dates)) * 1000,
        'Low': 49000 + np.random.randn(len(dates)) * 1000,
        'Close': 50000 + np.random.randn(len(dates)) * 1000,
        'Volume': 1e9 + np.random.randn(len(dates)) * 1e8
    }, index=dates)

    # Ensure OHLC constraints
    data['High'] = data[['Open', 'Close']].max(axis=1) + np.abs(np.random.randn(len(dates)) * 100)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.abs(np.random.randn(len(dates)) * 100)

    return data


@pytest.fixture
def test_config():
    """Config mínimo para testing."""
    return {
        'data': {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'exchange': 'binance',
            'time_map': {
                '1h': {'bars_per_year': 8760}
            }
        },
        'ga': {
            'population': 10,
            'generations_max': 5,
            'patience_no_improve': 3,
            'elitism': 2,
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'window_min': 2,
            'window_max': 5,
            'seed': 42,
            'allow_indicators': False,
            'max_expression_depth': 2,
            'max_children': 2,
            'fitness': {
                'cagr_min_threshold': 0.0,
                'weight_upi': 0.4,
                'weight_sharpe': 0.3,
                'weight_cagr': 0.3,
                'sharpe_cap': 3.0,
                'cagr_cap': 1.0
            },
            'fitness_weights': {
                'combined': 0.5,
                'long': 0.25,
                'short': 0.25
            },
            'evolution_tracking': {
                'save_best_per_gen': True,
                'save_diversity': True
            },
            'fast_mode': {
                'enabled': False,
                'n_windows': 5
            }
        },
        'exits': {
            'use_atr_exits': False,
            'use_time_exit': True,
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'max_hold_bars': 100,
            'bars_hold': 50,
            'atr_period': 14,
            'atr_stop': 1.5,
            'atr_take': 3.0
        },
        'costs': {
            'fees_pct': 0.001,
            'slippage_pct': 0.0005,
            'fees_bps_long': 10.0,
            'fees_bps_short': 10.0,
            'slippage_bps_long': 5.0,
            'slippage_bps_short': 5.0
        },
        'selection': {
            'max_patterns': 5,
            'min_sharpe': 0.0,
            'min_trades': 1,
            'max_correlation': 0.7,
            'filters': {
                'min_trades_per_window': 1
            }
        },
        'robustness': {
            'hansen_spa': {'n_bootstrap': 50},
            'white_rc': {'n_bootstrap': 50},
            'bootstrap_ci': {'n_bootstrap': 50, 'confidence_level': 0.95}
        },
        'output': {
            'output_dir': 'test_output',
            'reports_dir': 'test_output',
            'evolution_dir': 'test_evolution',
            'logs_dir': 'test_logs',
            'verbose_logging': False
        },
        'walkforward': {
            'enabled': True,
            'train_months': 2,
            'test_months': 1,
            'step_months': 1
        }
    }


def test_pipeline_grammar_to_pattern(test_config):
    """Pipeline: Grammar → Pattern generation."""
    from ga_patterns.grammar import PREDICATE_REGISTRY
    from ga_patterns.generator import generate_random_pattern

    # Verify registry exists
    assert len(PREDICATE_REGISTRY) > 0

    # Generate pattern
    pattern = generate_random_pattern(generation=1, config=test_config['ga'])

    # Validate
    from ga_patterns.chromosome import validate_pattern
    assert validate_pattern(pattern)
    assert pattern.direction in ['LONG', 'SHORT']
    assert 2 <= pattern.window <= 5


def test_pipeline_pattern_evaluation(small_dataset, test_config):
    """Pipeline: Pattern → Evaluation."""
    from ga_patterns.generator import generate_random_pattern

    # Generate pattern
    pattern = generate_random_pattern(generation=1, config=test_config['ga'])

    # Test evaluation on data
    window_data = small_dataset.head(pattern.window + 10)
    result = pattern.evaluate_on_data(window_data)

    assert isinstance(result, bool)


def test_pipeline_backtest_execution(small_dataset, test_config):
    """Pipeline: Pattern → Backtest → Trades."""
    from ga_patterns.generator import generate_random_pattern
    from backtest.runner import run_backtest

    # Generate simple pattern
    pattern = generate_random_pattern(generation=1, config=test_config['ga'])

    # Run backtest
    equity_curve, trades = run_backtest(pattern, small_dataset, test_config)

    # Validate outputs
    assert isinstance(equity_curve, pd.Series)
    assert len(equity_curve) > 0
    assert isinstance(trades, list)
    assert equity_curve.iloc[0] == 100.0  # Initial capital


def test_pipeline_walkforward_windows(small_dataset, test_config):
    """Pipeline: Data → Walk-forward windows."""
    from backtest.walkforward import create_walkforward_windows

    windows = create_walkforward_windows(
        small_dataset,
        train_months=2,
        test_months=1,
        step_months=1
    )

    # Validate
    assert len(windows) > 0

    for train_df, test_df in windows:
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert train_df.index.max() < test_df.index.min()  # Anti-lookahead


def test_pipeline_population_evolution(small_dataset, test_config):
    """Pipeline: Population → Evolution → Fitness."""
    import random
    from ga_patterns.generator import initialize_population
    from ga_patterns.fitness import evaluate_population

    # Set seeds
    random.seed(42)
    np.random.seed(42)

    # Initialize
    population = initialize_population(
        population_size=test_config['ga']['population'],
        generation=0,
        config=test_config['ga']
    )

    assert len(population) == test_config['ga']['population']

    # Evaluate
    evaluate_population(population, small_dataset, test_config)

    # Check fitness assigned
    assert all(hasattr(p, 'fitness') for p in population)
    assert all(isinstance(p.fitness, float) for p in population)


def test_pipeline_ga_operators(test_config):
    """Pipeline: Crossover + Mutation."""
    from ga_patterns.generator import (
        generate_random_pattern,
        subtree_crossover,
        mutate_pattern
    )

    # Generate parents
    parent1 = generate_random_pattern(1, test_config['ga'])
    parent2 = generate_random_pattern(1, test_config['ga'])

    # Crossover
    offspring = subtree_crossover(parent1, parent2, 2, test_config['ga'])

    from ga_patterns.chromosome import validate_pattern
    assert validate_pattern(offspring)

    # Mutation
    mutated = mutate_pattern(offspring, 2, test_config['ga'])
    assert validate_pattern(mutated)


def test_pipeline_metrics_calculation(small_dataset, test_config):
    """Pipeline: Equity → Metrics."""
    from backtest.metrics import calculate_all_metrics

    # Create synthetic equity
    equity = pd.Series(
        [100] + list(100 * (1 + np.random.randn(len(small_dataset)-1) * 0.01).cumprod()),
        index=small_dataset.index
    )

    metrics = calculate_all_metrics(equity, test_config['data']['time_map']['1h']['bars_per_year'])

    # Validate all required metrics
    required = ['upi', 'sharpe', 'cagr', 'max_dd', 'volatility', 'win_rate']
    for metric in required:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))


def test_pipeline_portfolio_selection(small_dataset, test_config):
    """Pipeline: Patterns → Decorrelation → Portfolio."""
    from ga_patterns.generator import generate_random_pattern
    from backtest.correlation import select_portfolio

    # Generate multiple patterns
    patterns = [generate_random_pattern(1, test_config['ga']) for _ in range(5)]

    # Assign mock fitness
    for i, p in enumerate(patterns):
        p.fitness = 0.5 - i * 0.05

    # Select portfolio
    portfolio = select_portfolio(patterns, small_dataset, test_config)

    assert isinstance(portfolio, list)
    assert len(portfolio) <= len(patterns)


@pytest.mark.slow
def test_full_pipeline_integration(small_dataset, test_config, tmp_path):
    """Full end-to-end pipeline test (SLOW)."""
    import random
    from ga_patterns.generator import (
        initialize_population,
        tournament_selection,
        subtree_crossover,
        mutate_pattern
    )
    from ga_patterns.fitness import evaluate_population
    from ga_patterns.evolution_tracker import EvolutionTracker

    # Setup
    random.seed(42)
    np.random.seed(42)

    test_config['output']['evolution_dir'] = str(tmp_path / 'evolution')
    test_config['output']['logs_dir'] = str(tmp_path / 'logs')
    test_config['output']['verbose_logging'] = False

    # Initialize
    tracker = EvolutionTracker(test_config)
    population = initialize_population(5, generation=0, config=test_config['ga'])

    # Evaluate
    evaluate_population(population, small_dataset, test_config)

    # Track
    best = max(population, key=lambda p: p.fitness)
    mean_fit = np.mean([p.fitness for p in population])
    tracker.track_generation(0, population, best, mean_fit)

    # Evolve 2 generations
    for gen in range(1, 3):
        new_pop = []

        # Elitism
        sorted_pop = sorted(population, key=lambda p: p.fitness, reverse=True)
        new_pop.extend(sorted_pop[:2])

        # Generate offspring
        while len(new_pop) < 5:
            p1 = tournament_selection(population)
            p2 = tournament_selection(population)

            offspring = subtree_crossover(p1, p2, gen, test_config['ga'])

            if random.random() < 0.2:
                offspring = mutate_pattern(offspring, gen, test_config['ga'])

            new_pop.append(offspring)

        population = new_pop

        # Evaluate
        evaluate_population(population, small_dataset, test_config)

        # Track
        best = max(population, key=lambda p: p.fitness)
        mean_fit = np.mean([p.fitness for p in population])
        tracker.track_generation(gen, population, best, mean_fit)

    # Validate outputs
    evolution_dir = tmp_path / 'evolution'
    assert evolution_dir.exists()

    # Final portfolio
    top_patterns = sorted(population, key=lambda p: p.fitness, reverse=True)[:3]
    assert len(top_patterns) == 3
    assert all(p.fitness >= -999 for p in top_patterns)


@pytest.mark.slow
def test_statistical_validation_pipeline(small_dataset, test_config):
    """Pipeline: Portfolio → Statistical tests."""
    from ga_patterns.generator import generate_random_pattern
    from robustness.hansen_spa import hansen_spa_test
    from robustness.white_rc import whites_reality_check

    # Generate patterns
    patterns = [generate_random_pattern(1, test_config['ga']) for _ in range(3)]

    # Mock returns
    np.random.seed(42)
    strategy_returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)
    benchmark_returns = pd.Series(np.random.randn(100) * 0.01)

    # Hansen SPA
    hansen_results = hansen_spa_test(
        strategy_returns,
        benchmark_returns,
        n_bootstrap=50,
        alpha=0.05,
        seed=42
    )

    assert 'p_value' in hansen_results
    assert 'reject_null' in hansen_results
    assert 0 <= hansen_results['p_value'] <= 1

    # White RC
    strategies_list = [
        pd.Series(np.random.randn(100) * 0.01 + 0.0005 * i)
        for i in range(3)
    ]

    white_results = whites_reality_check(
        strategies_list,
        benchmark_returns,
        n_bootstrap=50,
        alpha=0.05,
        seed=42
    )

    assert 'p_value' in white_results
    assert 0 <= white_results['p_value'] <= 1


def test_reproducibility(small_dataset, test_config):
    """Test que resultados sean reproducibles con mismo seed."""
    import random
    from ga_patterns.generator import generate_random_pattern

    # Run 1
    random.seed(42)
    np.random.seed(42)
    pattern1 = generate_random_pattern(1, test_config['ga'])

    # Run 2 (same seed)
    random.seed(42)
    np.random.seed(42)
    pattern2 = generate_random_pattern(1, test_config['ga'])

    # Should be identical
    assert pattern1.direction == pattern2.direction
    assert pattern1.window == pattern2.window
