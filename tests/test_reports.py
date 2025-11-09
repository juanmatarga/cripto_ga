"""Tests for reports module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from ga_patterns.chromosome import Pattern, PredicateNode, LogicalNode
from reports.pattern_explainer import (
    explain_predicate, explain_logical_node, explain_pattern, explain_portfolio
)
from reports import visualizations, report_generator, latex_exporter


@pytest.fixture
def sample_pattern():
    """Create a sample pattern for testing."""
    # Simple predicate: close[0] > close[1]
    predicate = PredicateNode(
        predicate_name='close',
        operator='>',
        bar_offset=0,
        compare_with_bar=1,
        threshold=None
    )

    pattern = Pattern(
        direction='LONG',
        window=10,
        expression=predicate
    )
    pattern.fitness = 0.5
    pattern.fitness_long = 0.6
    pattern.fitness_short = 0.4

    return pattern


@pytest.fixture
def sample_portfolio(sample_pattern):
    """Create sample portfolio for testing."""
    # Create second pattern
    predicate2 = PredicateNode(
        predicate_name='rsi',
        operator='<',
        bar_offset=0,
        compare_with_bar=None,
        threshold=30.0
    )

    pattern2 = Pattern(
        direction='SHORT',
        window=5,
        expression=predicate2
    )
    pattern2.fitness = 0.45
    pattern2.fitness_long = 0.3
    pattern2.fitness_short = 0.55

    # Portfolio with metrics
    portfolio = [
        (sample_pattern, pd.Series([100, 105, 110]), {'sharpe': 1.5, 'total_trades': 10, 'win_rate': 0.6}),
        (pattern2, pd.Series([100, 98, 102]), {'sharpe': 1.2, 'total_trades': 8, 'win_rate': 0.5})
    ]

    return portfolio


@pytest.fixture
def sample_data():
    """Create sample price data."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    data = pd.DataFrame({
        'Open': 100 + np.random.randn(100),
        'High': 102 + np.random.randn(100),
        'Low': 98 + np.random.randn(100),
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'Volume': 1000 + np.random.randint(-100, 100, 100)
    }, index=dates)

    return data


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'data': {
            'exchange': 'binance',
            'symbol': 'BTC/USDT',
            'timeframe': '15m',
            'start': '2023-01-01',
            'end': '2023-12-31',
            'time_map': {
                '15m': {'bars_per_year': 35040}
            }
        },
        'ga': {
            'population': 100,
            'generations_max': 50,
            'patience_no_improve': 10,
            'elitism': 5,
            'crossover_rate': 0.7,
            'mutation_rate': 0.2,
            'seed': 42,
            'max_expression_depth': 3,
            'max_children': 3,
            'window_min': 5,
            'window_max': 20,
            'fitness_weights': {
                'combined': 0.5,
                'long': 0.25,
                'short': 0.25
            }
        },
        'exits': {
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'max_hold_bars': 100
        },
        'selection': {
            'max_patterns': 10,
            'min_sharpe': 0.5,
            'min_trades': 5,
            'max_correlation': 0.7
        },
        'robustness': {
            'hansen_spa': {'n_bootstrap': 1000},
            'white_rc': {'n_bootstrap': 1000},
            'bootstrap_ci': {'n_bootstrap': 1000, 'confidence_level': 0.95}
        },
        'output': {
            'reports_dir': 'output_reports',
            'evolution_dir': 'output_evolution',
            'logs_dir': 'logs',
            'verbose_logging': True
        }
    }


# ============================================================================
# Pattern Explainer Tests
# ============================================================================

def test_explain_predicate_bar_comparison(sample_pattern):
    """Test explaining predicate with bar comparison."""
    explanation = explain_predicate(sample_pattern.expression)

    assert 'close price' in explanation.lower()
    assert 'current' in explanation.lower()
    assert 'previous' in explanation.lower()
    assert 'greater than' in explanation.lower()


def test_explain_predicate_threshold_comparison():
    """Test explaining predicate with threshold comparison."""
    predicate = PredicateNode(
        predicate_name='price_change_pct',
        operator='>',
        bar_offset=0,
        compare_with_bar=None,
        threshold=0.02
    )

    explanation = explain_predicate(predicate)

    assert 'price change' in explanation.lower()
    assert '2.00%' in explanation
    assert 'greater than' in explanation.lower()


def test_explain_logical_node_and():
    """Test explaining AND logical node."""
    pred1 = PredicateNode(
        predicate_name='close',
        operator='>',
        bar_offset=0,
        compare_with_bar=1,
        threshold=None
    )
    pred2 = PredicateNode(
        predicate_name='volume',
        operator='>',
        bar_offset=0,
        compare_with_bar=1,
        threshold=None
    )

    logical = LogicalNode(operator='AND', children=[pred1, pred2])
    explanation = explain_logical_node(logical)

    assert 'AND' in explanation or 'ALL' in explanation


def test_explain_pattern_complete(sample_pattern):
    """Test complete pattern explanation."""
    explanation = explain_pattern(sample_pattern)

    assert 'Pattern Description' in explanation
    assert 'LONG' in explanation
    assert 'Window' in explanation
    assert 'Fitness' in explanation
    assert 'Entry Condition' in explanation


def test_explain_portfolio(sample_portfolio):
    """Test portfolio explanation."""
    explanation = explain_portfolio(sample_portfolio)

    assert 'PORTFOLIO EXPLANATION' in explanation
    assert 'Total Patterns: 2' in explanation
    assert 'LONG Patterns: 1' in explanation
    assert 'SHORT Patterns: 1' in explanation


# ============================================================================
# Visualization Tests
# ============================================================================

def test_plot_equity_curves():
    """Test equity curves plot generation."""
    portfolio_equity = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=pd.date_range('2023-01-01', periods=100, freq='D')
    )
    benchmark_equity = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.3),
        index=pd.date_range('2023-01-01', periods=100, freq='D')
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'equity_curves.png'
        visualizations.plot_equity_curves(portfolio_equity, benchmark_equity, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_drawdown_analysis():
    """Test drawdown analysis plot generation."""
    equity = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=pd.date_range('2023-01-01', periods=100, freq='D')
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'drawdown.png'
        visualizations.plot_drawdown_analysis(equity, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_evolution_fitness():
    """Test evolution fitness plot generation."""
    evolution_data = {
        'best_fitness_history': [
            {'generation': i, 'fitness': 0.1 + i * 0.01} for i in range(10)
        ],
        'mean_fitness_history': [
            {'generation': i, 'fitness': 0.05 + i * 0.005} for i in range(10)
        ],
        'best_long_history': [
            {'generation': i, 'fitness': 0.08 + i * 0.008} for i in range(10)
        ],
        'best_short_history': [
            {'generation': i, 'fitness': 0.06 + i * 0.006} for i in range(10)
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'evolution.png'
        visualizations.plot_evolution_fitness(evolution_data, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_statistical_tests():
    """Test statistical tests plot generation."""
    hansen_results = {
        'test_statistic': 0.15,
        'p_value': 0.44,
        'reject_null': False,
        'mean_outperformance': 0.0001,
        'alpha': 0.05
    }

    white_results = {
        'test_statistic': 0.12,
        'p_value': 0.38,
        'reject_null': False,
        'mean_outperformance': 0.0001,
        'alpha': 0.05
    }

    bootstrap_results = {
        'upi': {'mean': 0.5, 'median': 0.48, 'std': 0.1, 'ci_lower': 0.3, 'ci_upper': 0.7},
        'sharpe': {'mean': 1.2, 'median': 1.15, 'std': 0.2, 'ci_lower': 0.8, 'ci_upper': 1.6},
        'cagr': {'mean': 0.15, 'median': 0.14, 'std': 0.05, 'ci_lower': 0.05, 'ci_upper': 0.25}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'statistical_tests.png'
        visualizations.plot_statistical_tests(hansen_results, white_results, bootstrap_results, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


def test_plot_returns_distribution():
    """Test returns distribution plot generation."""
    returns = pd.Series(np.random.randn(1000) * 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'returns_dist.png'
        visualizations.plot_returns_distribution(returns, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


# ============================================================================
# Report Generator Tests
# ============================================================================

def test_format_metric_value():
    """Test metric value formatting."""
    from reports.report_generator import format_metric_value

    assert '50.00%' in format_metric_value('cagr', 0.5)
    assert '1.5000' in format_metric_value('sharpe', 1.5)
    assert '100' in format_metric_value('total_trades', 100)


def test_generate_report(sample_portfolio, sample_data, sample_config):
    """Test full report generation."""
    portfolio_equity = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=pd.date_range('2023-01-01', periods=100, freq='15min')
    )
    benchmark_equity = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.3),
        index=pd.date_range('2023-01-01', periods=100, freq='15min')
    )

    evolution_data = {
        'best_fitness_history': [
            {'generation': i, 'fitness': 0.1 + i * 0.01} for i in range(10)
        ],
        'mean_fitness_history': [
            {'generation': i, 'fitness': 0.05 + i * 0.005} for i in range(10)
        ]
    }

    hansen_results = {'test_statistic': 0.15, 'p_value': 0.44, 'reject_null': False, 'alpha': 0.05, 'mean_outperformance': 0.0001}
    white_results = {'test_statistic': 0.12, 'p_value': 0.38, 'reject_null': False, 'alpha': 0.05, 'mean_outperformance': 0.0001}
    bootstrap_results = {
        'upi': {'mean': 0.5, 'median': 0.48, 'std': 0.1, 'ci_lower': 0.3, 'ci_upper': 0.7}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'report.md'

        report_generator.generate_report(
            portfolio=sample_portfolio,
            portfolio_equity=portfolio_equity,
            benchmark_equity=benchmark_equity,
            evolution_data=evolution_data,
            final_generation=10,
            hansen_results=hansen_results,
            white_results=white_results,
            bootstrap_results=bootstrap_results,
            data=sample_data,
            config=sample_config,
            output_path=output_path
        )

        assert output_path.exists()

        # Check content
        content = output_path.read_text(encoding='utf-8')
        assert 'BTC/USDT Pattern Discovery Experiment Report' in content
        assert 'Executive Summary' in content
        assert 'Methodology' in content
        assert 'Evolution Analysis' in content
        assert 'Portfolio Patterns' in content
        assert 'Statistical Validation' in content


# ============================================================================
# LaTeX Exporter Tests
# ============================================================================

def test_escape_latex():
    """Test LaTeX escaping."""
    from reports.latex_exporter import escape_latex

    assert escape_latex('50%') == r'50\%'
    assert escape_latex('test_var') == r'test\_var'
    assert escape_latex('a & b') == r'a \& b'


def test_export_patterns_table(sample_portfolio):
    """Test patterns table export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'patterns.tex'
        latex_exporter.export_patterns_table(sample_portfolio, output_path)

        assert output_path.exists()

        content = output_path.read_text(encoding='utf-8')
        assert r'\begin{table}' in content
        assert r'\begin{tabular}' in content
        assert 'LONG' in content
        assert 'SHORT' in content


def test_export_metrics_table():
    """Test metrics table export."""
    portfolio_metrics = {
        'cagr': 0.15,
        'sharpe': 1.5,
        'max_dd': -0.1,
        'total_trades': 100
    }

    benchmark_metrics = {
        'cagr': 0.10,
        'sharpe': 1.2,
        'max_dd': -0.15,
        'total_trades': 0
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'metrics.tex'
        latex_exporter.export_metrics_table(portfolio_metrics, benchmark_metrics, output_path)

        assert output_path.exists()

        content = output_path.read_text(encoding='utf-8')
        assert r'\begin{table}' in content
        assert 'CAGR' in content
        assert 'Sharpe Ratio' in content


def test_export_statistical_tests_table():
    """Test statistical tests table export."""
    hansen_results = {'test_statistic': 0.15, 'p_value': 0.44, 'reject_null': False, 'alpha': 0.05}
    white_results = {'test_statistic': 0.12, 'p_value': 0.38, 'reject_null': False, 'alpha': 0.05}
    bootstrap_results = {
        'upi': {'mean': 0.5, 'median': 0.48, 'std': 0.1, 'ci_lower': 0.3, 'ci_upper': 0.7}
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'stats.tex'
        latex_exporter.export_statistical_tests_table(hansen_results, white_results, bootstrap_results, output_path)

        assert output_path.exists()

        content = output_path.read_text(encoding='utf-8')
        assert r'\begin{table}' in content
        assert 'Hansen SPA' in content
        assert r"White's RC" in content
