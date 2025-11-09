"""
Performance Tests - Benchmarks and profiling
"""

import pytest
import pandas as pd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def benchmark_data():
    """Dataset for benchmarking."""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='1h')
    data = pd.DataFrame({
        'Open': 50000 + np.random.randn(len(dates)) * 1000,
        'High': 51000 + np.random.randn(len(dates)) * 1000,
        'Low': 49000 + np.random.randn(len(dates)) * 1000,
        'Close': 50000 + np.random.randn(len(dates)) * 1000,
        'Volume': 1e9 + np.random.randn(len(dates)) * 1e8
    }, index=dates)

    data['High'] = data[['Open', 'Close']].max(axis=1) + np.abs(np.random.randn(len(dates)) * 100)
    data['Low'] = data[['Open', 'Close']].min(axis=1) - np.abs(np.random.randn(len(dates)) * 100)

    return data


@pytest.fixture
def config_fixture():
    """Config for performance tests."""
    return {
        'ga': {
            'window_min': 2,
            'window_max': 8,
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
        'data': {
            'timeframe': '1h',
            'time_map': {
                '1h': {'bars_per_year': 8760}
            }
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
        'walkforward': {
            'enabled': True,
            'train_months': 2,
            'test_months': 1,
            'step_months': 1
        }
    }


@pytest.mark.benchmark
def test_pattern_generation_speed(config_fixture):
    """Benchmark: Pattern generation speed."""
    from ga_patterns.generator import generate_random_pattern

    start = time.time()

    for _ in range(100):
        pattern = generate_random_pattern(1, config_fixture['ga'])

    elapsed = time.time() - start

    avg_time_ms = (elapsed / 100) * 1000
    logger.info(f"Pattern generation: {avg_time_ms:.2f} ms per pattern")

    # Should generate pattern in < 10ms
    assert avg_time_ms < 10


@pytest.mark.benchmark
def test_pattern_evaluation_speed(benchmark_data, config_fixture):
    """Benchmark: Pattern evaluation speed."""
    from ga_patterns.generator import generate_random_pattern

    np.random.seed(42)
    pattern = generate_random_pattern(1, config_fixture['ga'])
    window_data = benchmark_data.head(pattern.window + 100)

    start = time.time()

    for _ in range(1000):
        result = pattern.evaluate_on_data(window_data)

    elapsed = time.time() - start

    avg_time_ms = (elapsed / 1000) * 1000
    logger.info(f"Pattern evaluation: {avg_time_ms:.2f} ms per evaluation")

    # Should evaluate in < 5ms
    assert avg_time_ms < 5


@pytest.mark.benchmark
def test_backtest_speed(benchmark_data, config_fixture):
    """Benchmark: Backtest execution speed."""
    from ga_patterns.generator import generate_random_pattern
    from backtest.runner import run_backtest

    np.random.seed(42)
    pattern = generate_random_pattern(1, config_fixture['ga'])
    test_data = benchmark_data.head(1000)  # 1000 bars

    start = time.time()

    equity_curve, trades = run_backtest(pattern, test_data, config_fixture)

    elapsed = time.time() - start

    logger.info(f"Backtest (1000 bars): {elapsed:.3f} seconds")

    # Should backtest 1000 bars in < 200ms
    assert elapsed < 0.2


@pytest.mark.benchmark
def test_population_evaluation_speed(benchmark_data, config_fixture):
    """Benchmark: Population evaluation speed."""
    from ga_patterns.generator import initialize_population
    from ga_patterns.fitness import evaluate_population

    np.random.seed(42)
    population = initialize_population(10, 0, config_fixture['ga'])
    test_data = benchmark_data.head(1000)

    start = time.time()

    evaluate_population(population, test_data, config_fixture)

    elapsed = time.time() - start

    logger.info(f"Population evaluation (10 patterns, 1000 bars): {elapsed:.2f} seconds")

    # Should evaluate 10 patterns in < 10s
    assert elapsed < 10.0


@pytest.mark.slow
def test_memory_usage_pattern_generation():
    """Test: Memory usage durante generación de patrones."""
    import tracemalloc
    from ga_patterns.generator import initialize_population

    tracemalloc.start()

    # Generate large population
    config = {
        'window_min': 2,
        'window_max': 8,
        'allow_indicators': False,
        'max_expression_depth': 2,
        'max_children': 2
    }

    population = initialize_population(100, 0, config)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Should use < 50 MB
    peak_mb = peak / 1024 / 1024
    logger.info(f"Peak memory usage: {peak_mb:.2f} MB")
    assert peak_mb < 50


@pytest.mark.slow
def test_walkforward_scalability(benchmark_data):
    """Test: Scalability de walk-forward con data grande."""
    from backtest.walkforward import create_walkforward_windows

    start = time.time()

    windows = create_walkforward_windows(
        benchmark_data,
        train_months=6,
        test_months=2,
        step_months=1
    )

    elapsed = time.time() - start

    logger.info(f"Created {len(windows)} windows in {elapsed:.2f}s")

    # Should create windows in < 5s
    assert elapsed < 5.0
    assert len(windows) > 0


def test_bootstrap_performance():
    """Test: Bootstrap performance."""
    from robustness.bootstrap import block_bootstrap_returns

    np.random.seed(42)
    returns = pd.Series(np.random.randn(1000))

    start = time.time()

    bootstrapped = block_bootstrap_returns(
        returns,
        block_size=20,
        n_iterations=100,
        seed=42
    )

    elapsed = time.time() - start

    logger.info(f"Bootstrap (100 iterations) completed in {elapsed:.2f}s")

    # Should complete in < 3s
    assert elapsed < 3.0


def test_correlation_matrix_performance(benchmark_data, config_fixture):
    """Test: Correlation matrix calculation."""
    from ga_patterns.generator import generate_random_pattern
    from backtest.runner import run_backtest
    import pandas as pd

    # Generate patterns and their equity curves
    np.random.seed(42)
    patterns = [generate_random_pattern(1, config_fixture['ga']) for _ in range(5)]

    test_data = benchmark_data.head(1000)
    equity_curves = []

    for p in patterns:
        equity, _ = run_backtest(p, test_data, config_fixture)
        equity_curves.append(equity)

    start = time.time()

    # Calculate correlation matrix
    returns_df = pd.DataFrame({
        f'p{i}': equity.pct_change().dropna()
        for i, equity in enumerate(equity_curves)
    })

    corr_matrix = returns_df.corr()

    elapsed = time.time() - start

    logger.info(f"Correlation matrix (5 patterns, 1000 bars) in {elapsed:.2f}s")

    # Should complete in < 10s
    assert elapsed < 10.0


@pytest.mark.benchmark
def test_statistical_test_speed():
    """Benchmark: Statistical test speed."""
    from robustness.hansen_spa import hansen_spa_test

    np.random.seed(42)
    strategy_returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)
    benchmark_returns = pd.Series(np.random.randn(100) * 0.01)

    start = time.time()

    result = hansen_spa_test(
        strategy_returns,
        benchmark_returns,
        n_bootstrap=100,
        alpha=0.05,
        seed=42
    )

    elapsed = time.time() - start

    logger.info(f"Hansen SPA (100 bootstrap): {elapsed:.2f}s")

    # Should complete in < 2s
    assert elapsed < 2.0


def test_metrics_calculation_speed(benchmark_data):
    """Test: Metrics calculation speed."""
    from backtest.metrics import calculate_all_metrics

    # Create synthetic equity
    equity = pd.Series(
        [100] + list(100 * (1 + np.random.randn(len(benchmark_data)-1) * 0.01).cumprod()),
        index=benchmark_data.index
    )

    start = time.time()

    for _ in range(100):
        metrics = calculate_all_metrics(equity, 8760)

    elapsed = time.time() - start

    avg_time_ms = (elapsed / 100) * 1000
    logger.info(f"Metrics calculation: {avg_time_ms:.2f} ms")

    # Should complete in < 50ms (relaxed threshold for large datasets)
    assert avg_time_ms < 50
