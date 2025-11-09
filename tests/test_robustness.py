"""Tests de validación estadística."""

import pytest
import pandas as pd
import numpy as np
from robustness.bootstrap import block_bootstrap_returns, calculate_bootstrap_statistics
from robustness.hansen_spa import hansen_spa_test, calculate_relative_performance
from robustness.white_rc import whites_reality_check

def test_block_bootstrap():
    """Block bootstrap preserva estructura."""
    returns = pd.Series(np.random.randn(100))

    bootstrapped = block_bootstrap_returns(returns, block_size=10, n_iterations=10, seed=42)

    assert bootstrapped.shape == (10, 100)
    assert isinstance(bootstrapped, np.ndarray)

def test_bootstrap_statistics(config_fixture):
    """Bootstrap calcula estadísticas."""
    np.random.seed(42)
    equity = pd.Series([100] + list(100 * (1 + np.random.randn(99) * 0.01).cumprod()))

    # Usar menos iteraciones para test rápido
    bootstrap_config = config_fixture['robustness']['bootstrap'].copy()
    bootstrap_config['n_iterations'] = 50  # Reducido para test rápido

    results = calculate_bootstrap_statistics(
        equity,
        periods_per_year=35040,
        config=bootstrap_config
    )

    assert 'upi' in results
    assert 'ci_lower' in results['upi']
    assert 'ci_upper' in results['upi']
    assert results['upi']['ci_lower'] < results['upi']['ci_upper']

def test_hansen_spa():
    """Hansen SPA test funciona."""
    np.random.seed(42)
    # Strategy mejor que benchmark
    strategy_returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)
    benchmark_returns = pd.Series(np.random.randn(100) * 0.01)

    results = hansen_spa_test(
        strategy_returns,
        benchmark_returns,
        n_bootstrap=100,
        alpha=0.05,
        seed=42
    )

    assert 'p_value' in results
    assert 'reject_null' in results
    assert isinstance(results['p_value'], float)
    assert 0 <= results['p_value'] <= 1

def test_whites_reality_check():
    """White's RC funciona con múltiples estrategias."""
    np.random.seed(42)
    # 5 estrategias
    strategies = [
        pd.Series(np.random.randn(100) * 0.01 + 0.0005 * i)
        for i in range(5)
    ]
    benchmark = pd.Series(np.random.randn(100) * 0.01)

    results = whites_reality_check(
        strategies,
        benchmark,
        n_bootstrap=100,
        alpha=0.05,
        seed=42
    )

    assert 'p_value' in results
    assert 'n_strategies_tested' in results
    assert results['n_strategies_tested'] == 5

def test_relative_performance():
    """Relative performance calculation."""
    strategy = pd.Series([0.01, 0.02, -0.01, 0.03], index=[0, 1, 2, 3])
    benchmark = pd.Series([0.005, 0.01, 0.00, 0.02], index=[0, 1, 2, 3])

    rel_perf = calculate_relative_performance(strategy, benchmark)

    expected = pd.Series([0.005, 0.01, -0.01, 0.01], index=[0, 1, 2, 3])

    pd.testing.assert_series_equal(rel_perf, expected)

def test_hansen_spa_no_overlap():
    """Hansen SPA handles no overlapping periods."""
    strategy = pd.Series([0.01, 0.02], index=[0, 1])
    benchmark = pd.Series([0.01, 0.02], index=[2, 3])

    results = hansen_spa_test(strategy, benchmark, n_bootstrap=10, seed=42)

    assert results['p_value'] == 1.0
    assert results['reject_null'] == False

def test_white_rc_empty_strategies():
    """White RC handles empty strategy list."""
    strategies = []
    benchmark = pd.Series(np.random.randn(100) * 0.01)

    results = whites_reality_check(strategies, benchmark, n_bootstrap=10, seed=42)

    assert results['p_value'] == 1.0
    assert results['reject_null'] == False
    assert results['n_strategies_tested'] == 0
