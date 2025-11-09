"""
LaTeX Exporter - Export tables for academic papers
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters.

    Args:
        text: Input text

    Returns:
        str: LaTeX-safe text
    """
    # Escape special characters
    replacements = {
        '\\': r'\textbackslash{}',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '&': r'\&',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}'
    }

    for char, escaped in replacements.items():
        text = text.replace(char, escaped)

    return text


def format_latex_value(value, metric_name: str = '') -> str:
    """
    Format value for LaTeX table.

    Args:
        value: Value to format
        metric_name: Optional metric name for context

    Returns:
        str: Formatted LaTeX string
    """
    if isinstance(value, float):
        if metric_name in ['cagr', 'total_return', 'max_dd']:
            # Percentage
            return f"{value * 100:.2f}\\%"
        elif metric_name in ['sharpe', 'sortino', 'calmar', 'upi']:
            return f"{value:.4f}"
        elif metric_name in ['profit_factor', 'win_rate']:
            return f"{value:.2f}"
        elif metric_name in ['p_value', 'test_statistic', 'mean_outperformance']:
            return f"{value:.4f}"
        else:
            return f"{value:.4f}"
    elif isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, bool):
        return r"\checkmark" if value else r"\times"
    else:
        return escape_latex(str(value))


def export_patterns_table(portfolio: List, output_path: Path):
    """
    Export patterns table in LaTeX format.

    Args:
        portfolio: List of (Pattern, equity, metrics) tuples
        output_path: Output .tex file path
    """
    logger.info("Exporting patterns table to LaTeX...")

    # Extract data
    if len(portfolio) > 0 and isinstance(portfolio[0], tuple):
        patterns_data = [(p[0], p[2]) for p in portfolio]  # (Pattern, metrics)
    else:
        patterns_data = [(p, {}) for p in portfolio]

    # Build table
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Discovered Trading Patterns}" + "\n"
    latex += r"\label{tab:patterns}" + "\n"
    latex += r"\begin{tabular}{ccccccc}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"\textbf{ID} & \textbf{Direction} & \textbf{Fitness} & \textbf{Window} & "
    latex += r"\textbf{Trades} & \textbf{Sharpe} & \textbf{Win Rate} \\" + "\n"
    latex += r"\hline" + "\n"

    for i, (pattern, metrics) in enumerate(patterns_data, 1):
        latex += f"{i} & "
        latex += f"{pattern.direction} & "
        latex += f"{pattern.fitness:.4f} & "
        latex += f"{pattern.window} & "
        latex += f"{metrics.get('total_trades', 0)} & "
        latex += f"{metrics.get('sharpe', 0):.4f} & "
        latex += f"{metrics.get('win_rate', 0):.2f} "
        latex += r"\\" + "\n"

    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    logger.info(f"[OK] Saved patterns table to {output_path}")


def export_metrics_table(portfolio_metrics: Dict,
                        benchmark_metrics: Dict,
                        output_path: Path):
    """
    Export performance metrics comparison table.

    Args:
        portfolio_metrics: Portfolio metrics dict
        benchmark_metrics: Benchmark metrics dict
        output_path: Output .tex file path
    """
    logger.info("Exporting metrics table to LaTeX...")

    # Metrics to include
    metrics_list = [
        ('CAGR', 'cagr'),
        ('Sharpe Ratio', 'sharpe'),
        ('Sortino Ratio', 'sortino'),
        ('Calmar Ratio', 'calmar'),
        ('UPI', 'upi'),
        ('Max Drawdown', 'max_dd'),
        ('Total Return', 'total_return'),
        ('Profit Factor', 'profit_factor'),
        ('Win Rate', 'win_rate'),
        ('Total Trades', 'total_trades')
    ]

    # Build table
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Performance Metrics Comparison}" + "\n"
    latex += r"\label{tab:metrics}" + "\n"
    latex += r"\begin{tabular}{lccc}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"\textbf{Metric} & \textbf{Portfolio} & \textbf{Buy \& Hold} & \textbf{Difference} \\" + "\n"
    latex += r"\hline" + "\n"

    for metric_label, metric_key in metrics_list:
        port_val = portfolio_metrics.get(metric_key, 0)
        bench_val = benchmark_metrics.get(metric_key, 0)

        # Calculate difference
        if metric_key in ['cagr', 'total_return', 'max_dd']:
            diff = (port_val - bench_val) * 100  # percentage points
            diff_str = f"{diff:+.2f}pp"
        else:
            diff = port_val - bench_val
            diff_str = f"{diff:+.4f}"

        latex += f"{metric_label} & "
        latex += f"{format_latex_value(port_val, metric_key)} & "
        latex += f"{format_latex_value(bench_val, metric_key)} & "
        latex += f"{escape_latex(diff_str)} "
        latex += r"\\" + "\n"

    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    logger.info(f"[OK] Saved metrics table to {output_path}")


def export_statistical_tests_table(hansen_results: Optional[Dict],
                                   white_results: Optional[Dict],
                                   bootstrap_results: Optional[Dict],
                                   output_path: Path):
    """
    Export statistical test results table.

    Args:
        hansen_results: Hansen SPA results
        white_results: White's RC results
        bootstrap_results: Bootstrap CI results
        output_path: Output .tex file path
    """
    logger.info("Exporting statistical tests table to LaTeX...")

    # Build table
    latex = r"\begin{table}[htbp]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Statistical Validation Results}" + "\n"
    latex += r"\label{tab:statistical_tests}" + "\n"
    latex += r"\begin{tabular}{lcccc}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"\textbf{Test} & \textbf{Statistic} & \textbf{P-value} & "
    latex += r"\textbf{Reject Null} & \textbf{$\alpha$} \\" + "\n"
    latex += r"\hline" + "\n"

    # Hansen SPA
    if hansen_results:
        latex += r"Hansen SPA & "
        latex += f"{format_latex_value(hansen_results['test_statistic'], 'test_statistic')} & "
        latex += f"{format_latex_value(hansen_results['p_value'], 'p_value')} & "
        latex += f"{format_latex_value(hansen_results['reject_null'])} & "
        latex += f"{format_latex_value(hansen_results['alpha'], 'p_value')} "
        latex += r"\\" + "\n"

    # White's RC
    if white_results:
        latex += r"White's RC & "
        latex += f"{format_latex_value(white_results['test_statistic'], 'test_statistic')} & "
        latex += f"{format_latex_value(white_results['p_value'], 'p_value')} & "
        latex += f"{format_latex_value(white_results['reject_null'])} & "
        latex += f"{format_latex_value(white_results['alpha'], 'p_value')} "
        latex += r"\\" + "\n"

    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"

    # Bootstrap CI table (separate)
    if bootstrap_results:
        latex += "\n\n"
        latex += r"\begin{table}[htbp]" + "\n"
        latex += r"\centering" + "\n"
        latex += r"\caption{Bootstrap Confidence Intervals (95\%)}" + "\n"
        latex += r"\label{tab:bootstrap_ci}" + "\n"
        latex += r"\begin{tabular}{lccccc}" + "\n"
        latex += r"\hline" + "\n"
        latex += r"\textbf{Metric} & \textbf{Mean} & \textbf{Median} & "
        latex += r"\textbf{Std} & \textbf{CI Lower} & \textbf{CI Upper} \\" + "\n"
        latex += r"\hline" + "\n"

        for metric_name in ['upi', 'sharpe', 'cagr']:
            if metric_name in bootstrap_results:
                stats = bootstrap_results[metric_name]
                latex += f"{metric_name.upper()} & "
                latex += f"{stats['mean']:.4f} & "
                latex += f"{stats['median']:.4f} & "
                latex += f"{stats['std']:.4f} & "
                latex += f"{stats['ci_lower']:.4f} & "
                latex += f"{stats['ci_upper']:.4f} "
                latex += r"\\" + "\n"

        latex += r"\hline" + "\n"
        latex += r"\end{tabular}" + "\n"
        latex += r"\end{table}" + "\n"

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    logger.info(f"[OK] Saved statistical tests table to {output_path}")


def export_all_latex_tables(portfolio: List,
                           portfolio_metrics: Dict,
                           benchmark_metrics: Dict,
                           hansen_results: Optional[Dict],
                           white_results: Optional[Dict],
                           bootstrap_results: Optional[Dict],
                           output_dir: Path):
    """
    Export all LaTeX tables to output directory.

    Args:
        portfolio: Portfolio patterns
        portfolio_metrics: Portfolio metrics
        benchmark_metrics: Benchmark metrics
        hansen_results: Hansen SPA results
        white_results: White's RC results
        bootstrap_results: Bootstrap CI results
        output_dir: Output directory
    """
    logger.info("Exporting all LaTeX tables...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export each table
    export_patterns_table(portfolio, output_dir / 'patterns_table.tex')
    export_metrics_table(portfolio_metrics, benchmark_metrics,
                        output_dir / 'metrics_table.tex')
    export_statistical_tests_table(hansen_results, white_results,
                                   bootstrap_results,
                                   output_dir / 'statistical_tests_table.tex')

    logger.info("[OK] All LaTeX tables exported successfully")
