"""
Report Generator - Comprehensive Markdown experiment report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
import yaml

from .pattern_explainer import explain_portfolio
from backtest.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


def format_metric_value(metric_name: str, value) -> str:
    """
    Format metric value based on type.

    Args:
        metric_name: Name of the metric
        value: Metric value

    Returns:
        str: Formatted string
    """
    if isinstance(value, float):
        if metric_name in ['cagr', 'total_return', 'max_dd']:
            return f"{value:.2%}"
        elif metric_name in ['sharpe', 'sortino', 'calmar', 'upi']:
            return f"{value:.4f}"
        elif metric_name in ['profit_factor', 'win_rate']:
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"
    elif isinstance(value, int):
        return f"{value:,}"
    else:
        return str(value)


def generate_executive_summary(portfolio_metrics: Dict,
                              benchmark_metrics: Dict,
                              config: Dict) -> str:
    """
    Generate executive summary section.

    Args:
        portfolio_metrics: Portfolio performance metrics
        benchmark_metrics: Benchmark performance metrics
        config: Experiment configuration

    Returns:
        str: Executive summary markdown
    """
    summary = "## Executive Summary\n\n"

    # Experiment info
    summary += f"**Exchange**: {config['data']['exchange']}\n\n"
    summary += f"**Symbol**: {config['data']['symbol']}\n\n"
    summary += f"**Timeframe**: {config['data']['timeframe']}\n\n"
    summary += f"**Period**: {config['data']['start']} to {config['data']['end']}\n\n"
    summary += f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Key results
    summary += "### Key Results\n\n"
    summary += "| Metric | Portfolio | Buy & Hold | Difference |\n"
    summary += "|--------|-----------|------------|------------|\n"

    key_metrics = ['cagr', 'sharpe', 'sortino', 'max_dd', 'upi']

    for metric in key_metrics:
        port_val = portfolio_metrics.get(metric, 0)
        bench_val = benchmark_metrics.get(metric, 0)

        if metric in ['cagr', 'total_return', 'max_dd']:
            diff = (port_val - bench_val) * 100  # percentage points
            diff_str = f"{diff:+.2f}pp"
        else:
            diff = port_val - bench_val
            diff_str = f"{diff:+.4f}"

        summary += f"| {metric.upper()} | {format_metric_value(metric, port_val)} | "
        summary += f"{format_metric_value(metric, bench_val)} | {diff_str} |\n"

    summary += "\n"

    # Interpretation
    summary += "### Performance Interpretation\n\n"

    if portfolio_metrics.get('sharpe', 0) > benchmark_metrics.get('sharpe', 0):
        summary += "- **Risk-Adjusted Returns**: Portfolio outperforms Buy & Hold on Sharpe ratio.\n"
    else:
        summary += "- **Risk-Adjusted Returns**: Portfolio underperforms Buy & Hold on Sharpe ratio.\n"

    if portfolio_metrics.get('max_dd', 0) > benchmark_metrics.get('max_dd', 0):
        summary += "- **Drawdown Control**: Portfolio has larger drawdown than Buy & Hold.\n"
    else:
        summary += "- **Drawdown Control**: Portfolio has better drawdown control than Buy & Hold.\n"

    if portfolio_metrics.get('upi', 0) > 0:
        summary += f"- **UPI Score**: {portfolio_metrics['upi']:.4f} (positive indicates edge).\n"
    else:
        summary += f"- **UPI Score**: {portfolio_metrics['upi']:.4f} (negative indicates no edge).\n"

    summary += "\n"

    return summary


def generate_methodology_section(config: Dict) -> str:
    """
    Generate methodology section.

    Args:
        config: Experiment configuration

    Returns:
        str: Methodology markdown
    """
    method = "## Methodology\n\n"

    method += "### Genetic Algorithm Configuration\n\n"
    method += f"- **Population Size**: {config['ga']['population']}\n"
    method += f"- **Max Generations**: {config['ga']['generations_max']}\n"
    method += f"- **Early Stopping Patience**: {config['ga']['patience_no_improve']} generations\n"
    method += f"- **Elitism**: Top {config['ga']['elitism']} individuals preserved\n"
    method += f"- **Crossover Rate**: {config['ga']['crossover_rate']:.1%}\n"
    method += f"- **Mutation Rate**: {config['ga']['mutation_rate']:.1%}\n"
    method += f"- **Seed**: {config['ga']['seed']}\n\n"

    method += "### Pattern Structure\n\n"
    method += f"- **Max Expression Depth**: {config['ga']['max_expression_depth']}\n"
    method += f"- **Max Children per Node**: {config['ga']['max_children']}\n"
    method += f"- **Window Range**: {config['ga']['window_min']}-{config['ga']['window_max']} bars\n\n"

    method += "### Fitness Evaluation\n\n"
    method += f"- **Bidirectional**: Patterns evaluated for both LONG and SHORT directions\n"
    method += f"- **Combined Fitness**: {config['ga']['fitness_weights']['combined']:.1%}\n"
    method += f"- **LONG Fitness Weight**: {config['ga']['fitness_weights']['long']:.1%}\n"
    method += f"- **SHORT Fitness Weight**: {config['ga']['fitness_weights']['short']:.1%}\n\n"

    method += "### Exit Strategy\n\n"
    method += f"- **Stop Loss**: {config['exits']['stop_loss']:.1%}\n"
    method += f"- **Take Profit**: {config['exits']['take_profit']:.1%}\n"
    method += f"- **Max Hold Bars**: {config['exits']['max_hold_bars']}\n\n"

    method += "### Portfolio Selection\n\n"
    method += f"- **Max Patterns**: {config['selection']['max_patterns']}\n"
    method += f"- **Min Sharpe**: {config['selection']['min_sharpe']}\n"
    method += f"- **Min Trades**: {config['selection']['min_trades']}\n"
    method += f"- **Max Correlation**: {config['selection']['max_correlation']}\n\n"

    method += "### Statistical Validation\n\n"
    method += f"- **Hansen SPA Test**: {config['robustness']['hansen_spa']['n_bootstrap']} bootstrap samples\n"
    method += f"- **White's Reality Check**: {config['robustness']['white_rc']['n_bootstrap']} bootstrap samples\n"
    method += f"- **Bootstrap CI**: {config['robustness']['bootstrap_ci']['n_bootstrap']} samples, "
    method += f"{config['robustness']['bootstrap_ci']['confidence_level']:.1%} confidence level\n\n"

    return method


def generate_evolution_section(evolution_data: Dict, final_generation: int) -> str:
    """
    Generate evolution analysis section.

    Args:
        evolution_data: Evolution tracker data
        final_generation: Final generation number

    Returns:
        str: Evolution analysis markdown
    """
    evolution = "## Evolution Analysis\n\n"

    evolution += f"### Training Progress\n\n"
    evolution += f"- **Generations Completed**: {final_generation}\n"

    best_history = evolution_data.get('best_fitness_history', [])
    if best_history:
        initial_fitness = best_history[0]['fitness']
        final_fitness = best_history[-1]['fitness']
        improvement = final_fitness - initial_fitness

        evolution += f"- **Initial Best Fitness**: {initial_fitness:.4f}\n"
        evolution += f"- **Final Best Fitness**: {final_fitness:.4f}\n"
        evolution += f"- **Total Improvement**: {improvement:.4f} ({improvement/abs(initial_fitness):.1%})\n\n"

    # LONG/SHORT best patterns
    best_long = evolution_data.get('best_long_history', [])
    best_short = evolution_data.get('best_short_history', [])

    if best_long:
        evolution += f"### Best LONG Pattern\n\n"
        evolution += f"- **Fitness**: {best_long[-1]['fitness']:.4f}\n"
        evolution += f"- **Generation**: {best_long[-1]['generation']}\n\n"

    if best_short:
        evolution += f"### Best SHORT Pattern\n\n"
        evolution += f"- **Fitness**: {best_short[-1]['fitness']:.4f}\n"
        evolution += f"- **Generation**: {best_short[-1]['generation']}\n\n"

    evolution += "### Convergence\n\n"
    evolution += "Refer to the **Evolution Fitness Plot** for visual analysis of convergence behavior.\n\n"

    return evolution


def generate_portfolio_section(portfolio: List, data: pd.DataFrame, config: Dict) -> str:
    """
    Generate portfolio patterns section with natural language explanations.

    Args:
        portfolio: List of (Pattern, equity, metrics) tuples
        data: Price data
        config: Configuration

    Returns:
        str: Portfolio section markdown
    """
    section = "## Portfolio Patterns\n\n"

    section += f"### Overview\n\n"
    section += f"- **Total Patterns Selected**: {len(portfolio)}\n"

    # Extract patterns
    if len(portfolio) > 0 and isinstance(portfolio[0], tuple):
        patterns = [p[0] for p in portfolio]
    else:
        patterns = portfolio

    long_count = sum(1 for p in patterns if p.direction == 'LONG')
    short_count = sum(1 for p in patterns if p.direction == 'SHORT')

    section += f"- **LONG Patterns**: {long_count}\n"
    section += f"- **SHORT Patterns**: {short_count}\n\n"

    # Individual pattern details
    section += "### Pattern Details\n\n"

    for i, item in enumerate(portfolio, 1):
        if isinstance(item, tuple):
            pattern, equity, metrics = item
        else:
            pattern = item
            metrics = {}

        section += f"#### Pattern #{i}: {pattern.direction}\n\n"
        section += f"- **Fitness**: {pattern.fitness:.4f}\n"
        section += f"- **Window**: {pattern.window} bars\n"

        if metrics:
            section += f"- **Sharpe Ratio**: {format_metric_value('sharpe', metrics.get('sharpe', 0))}\n"
            section += f"- **Total Trades**: {metrics.get('total_trades', 0)}\n"
            section += f"- **Win Rate**: {format_metric_value('win_rate', metrics.get('win_rate', 0))}\n"

        section += f"\n**Entry Condition**:\n\n"
        section += "```\n"
        section += f"{pattern.expression}\n"
        section += "```\n\n"

    section += "### Natural Language Explanations\n\n"
    section += "```\n"
    section += explain_portfolio(portfolio)
    section += "```\n\n"

    return section


def generate_statistical_section(hansen_results: Optional[Dict],
                                white_results: Optional[Dict],
                                bootstrap_results: Optional[Dict]) -> str:
    """
    Generate statistical validation section.

    Args:
        hansen_results: Hansen SPA test results
        white_results: White's Reality Check results
        bootstrap_results: Bootstrap confidence intervals

    Returns:
        str: Statistical validation markdown
    """
    section = "## Statistical Validation\n\n"

    # Hansen SPA
    if hansen_results:
        section += "### Hansen's Superior Predictive Ability (SPA) Test\n\n"
        section += f"- **Test Statistic**: {hansen_results['test_statistic']:.4f}\n"
        section += f"- **P-value**: {hansen_results['p_value']:.4f}\n"
        section += f"- **Significance Level (α)**: {hansen_results['alpha']}\n"
        section += f"- **Result**: {'[PASS] Reject null hypothesis (strategy has edge)' if hansen_results['reject_null'] else '[FAIL] Fail to reject null (no significant edge)'}\n"
        section += f"- **Mean Outperformance**: {hansen_results['mean_outperformance']:.6f}\n\n"

    # White's RC
    if white_results:
        section += "### White's Reality Check\n\n"
        section += f"- **Test Statistic**: {white_results['test_statistic']:.4f}\n"
        section += f"- **P-value**: {white_results['p_value']:.4f}\n"
        section += f"- **Significance Level (α)**: {white_results['alpha']}\n"
        section += f"- **Result**: {'[PASS] Reject null hypothesis (strategy has edge)' if white_results['reject_null'] else '[FAIL] Fail to reject null (no significant edge)'}\n"
        section += f"- **Mean Outperformance**: {white_results['mean_outperformance']:.6f}\n\n"

    # Bootstrap CI
    if bootstrap_results:
        section += "### Bootstrap Confidence Intervals (95%)\n\n"
        section += "| Metric | Mean | Median | Std | CI Lower | CI Upper |\n"
        section += "|--------|------|--------|-----|----------|----------|\n"

        for metric_name in ['upi', 'sharpe', 'cagr']:
            if metric_name in bootstrap_results:
                stats = bootstrap_results[metric_name]
                section += f"| {metric_name.upper()} | "
                section += f"{stats['mean']:.4f} | "
                section += f"{stats['median']:.4f} | "
                section += f"{stats['std']:.4f} | "
                section += f"{stats['ci_lower']:.4f} | "
                section += f"{stats['ci_upper']:.4f} |\n"

        section += "\n"

    section += "### Interpretation\n\n"
    section += "Refer to the **Statistical Tests Plot** for visual comparison of p-values and confidence intervals.\n\n"

    return section


def generate_configuration_section(config: Dict) -> str:
    """
    Generate full configuration dump.

    Args:
        config: Configuration dict

    Returns:
        str: Configuration YAML markdown
    """
    section = "## Full Configuration\n\n"
    section += "```yaml\n"
    section += yaml.dump(config, default_flow_style=False, sort_keys=False)
    section += "```\n\n"

    return section


def generate_report(portfolio: List,
                   portfolio_equity: pd.Series,
                   benchmark_equity: pd.Series,
                   evolution_data: Dict,
                   final_generation: int,
                   hansen_results: Optional[Dict],
                   white_results: Optional[Dict],
                   bootstrap_results: Optional[Dict],
                   data: pd.DataFrame,
                   config: Dict,
                   output_path: Path):
    """
    Generate complete experiment report in Markdown format.

    Args:
        portfolio: Selected patterns (list of tuples or Pattern objects)
        portfolio_equity: Portfolio equity curve
        benchmark_equity: Benchmark equity curve
        evolution_data: Evolution tracker data
        final_generation: Final generation number
        hansen_results: Hansen SPA results
        white_results: White's Reality Check results
        bootstrap_results: Bootstrap CI results
        data: Price data
        config: Experiment configuration
        output_path: Output file path
    """
    logger.info("Generating experiment report...")

    # Calculate metrics
    timeframe = config['data']['timeframe']
    periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']

    portfolio_metrics = calculate_all_metrics(portfolio_equity, periods_per_year)
    benchmark_metrics = calculate_all_metrics(benchmark_equity, periods_per_year)

    # Build report
    report = "# BTC/USDT Pattern Discovery Experiment Report\n\n"
    report += "---\n\n"

    # Sections
    report += generate_executive_summary(portfolio_metrics, benchmark_metrics, config)
    report += generate_methodology_section(config)
    report += generate_evolution_section(evolution_data, final_generation)
    report += generate_portfolio_section(portfolio, data, config)
    report += generate_statistical_section(hansen_results, white_results, bootstrap_results)
    report += generate_configuration_section(config)

    # Footer
    report += "---\n\n"
    report += f"*Report generated by cripto_ga v1.0 on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n"

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"[OK] Saved experiment report to {output_path}")
