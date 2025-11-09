"""
Robustness Testing Module - Statistical validation suite
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

from robustness.bootstrap import calculate_bootstrap_statistics, print_bootstrap_summary
from robustness.hansen_spa import hansen_spa_test
from robustness.white_rc import whites_reality_check

logger = logging.getLogger(__name__)

def run_robustness_tests(portfolio_patterns: List[Tuple],
                        data: pd.DataFrame,
                        config: dict) -> Dict:
    """
    Ejecuta suite completa de tests de robustez estadística.

    Args:
        portfolio_patterns: Lista de (Pattern, equity, metrics) del portfolio final
        data: DataFrame OHLCV completo
        config: config dict

    Returns:
        dict: Resultados de todos los tests
    """
    logger.info(f"\n{'='*80}")
    logger.info("ROBUSTNESS TESTING SUITE")
    logger.info(f"{'='*80}")
    logger.info(f"Portfolio size: {len(portfolio_patterns)} patterns")

    # Obtener períodos por año
    timeframe = config['data']['timeframe']
    periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']

    # Extraer patterns del portfolio (puede ser lista de tuplas o Pattern objects)
    if len(portfolio_patterns) > 0 and isinstance(portfolio_patterns[0], tuple):
        # Es (pattern, equity, metrics)
        patterns = [p[0] for p in portfolio_patterns]
        # Usar las equity curves ya calculadas
        pattern_equity_curves = [p[1] for p in portfolio_patterns]
    else:
        # Es lista de Pattern objects
        patterns = portfolio_patterns
        pattern_equity_curves = None

    logger.info(f"Using full dataset for validation: {len(data)} bars")

    # ========================================================================
    # 1. GENERAR EQUITY CURVES DE PORTFOLIO Y PATTERNS INDIVIDUALES
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("1. GENERATING EQUITY CURVES")
    logger.info(f"{'='*80}")

    # Si no tenemos equity curves, generarlas
    if pattern_equity_curves is None:
        from backtest.runner import run_backtest

        logger.info(f"  Running backtests on full dataset...")
        pattern_equity_curves = []
        pattern_returns_list = []

        for i, pattern in enumerate(patterns):
            logger.info(f"  Generating equity for pattern {i+1}/{len(patterns)}...")

            equity_curve, trades = run_backtest(pattern, data, config)
            pattern_equity_curves.append(equity_curve)

            # Calcular returns
            returns = equity_curve.pct_change().dropna()
            pattern_returns_list.append(returns)
    else:
        logger.info(f"  Using pre-computed equity curves...")
        pattern_returns_list = []
        for equity in pattern_equity_curves:
            returns = equity.pct_change().dropna()
            pattern_returns_list.append(returns)

    # Portfolio equity (equal weight)
    logger.info(f"  Generating portfolio equity (equal weight)...")

    # Alinear todas las equity curves al mismo índice
    common_index = pattern_equity_curves[0].index
    for ec in pattern_equity_curves[1:]:
        common_index = common_index.intersection(ec.index)

    # Recalcular con índice común
    aligned_equities = []
    for ec in pattern_equity_curves:
        aligned_equities.append(ec.reindex(common_index))

    portfolio_equity = sum(aligned_equities) / len(aligned_equities)
    portfolio_returns = portfolio_equity.pct_change().dropna()

    # Benchmark (buy & hold)
    logger.info(f"  Generating benchmark equity (buy & hold)...")
    benchmark_equity = data['Close'].reindex(portfolio_equity.index, method='nearest')
    benchmark_equity = benchmark_equity / benchmark_equity.iloc[0] * portfolio_equity.iloc[0]
    benchmark_returns = benchmark_equity.pct_change().dropna()

    # ========================================================================
    # 2. BOOTSTRAP STATISTICS
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("2. BOOTSTRAP STATISTICS")
    logger.info(f"{'='*80}")

    bootstrap_results = calculate_bootstrap_statistics(
        portfolio_equity,
        periods_per_year,
        config['robustness']['bootstrap']
    )

    print_bootstrap_summary(bootstrap_results)

    # ========================================================================
    # 3. HANSEN SPA TEST (Portfolio vs Benchmark)
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("3. HANSEN SPA TEST (Portfolio vs Benchmark)")
    logger.info(f"{'='*80}")

    if config['robustness']['hansen_spa']['enabled']:
        hansen_results = hansen_spa_test(
            portfolio_returns,
            benchmark_returns,
            n_bootstrap=config['robustness']['bootstrap']['n_iterations'],
            alpha=config['robustness']['hansen_spa']['alpha'],
            seed=config['robustness']['seed']
        )
    else:
        logger.info("Hansen SPA Test DISABLED in config")
        hansen_results = None

    # ========================================================================
    # 4. WHITE'S REALITY CHECK (All patterns vs Benchmark)
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("4. WHITE'S REALITY CHECK (All patterns vs Benchmark)")
    logger.info(f"{'='*80}")

    if config['robustness']['white_rc']['enabled']:
        white_results = whites_reality_check(
            pattern_returns_list,
            benchmark_returns,
            n_bootstrap=config['robustness']['bootstrap']['n_iterations'],
            alpha=config['robustness']['white_rc']['alpha'],
            seed=config['robustness']['seed']
        )
    else:
        logger.info("White's Reality Check DISABLED in config")
        white_results = None

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info(f"\n{'='*80}")
    logger.info("ROBUSTNESS TESTING SUMMARY")
    logger.info(f"{'='*80}")

    summary = {
        'portfolio_equity': portfolio_equity,
        'benchmark_equity': benchmark_equity,
        'bootstrap': bootstrap_results,
        'hansen_spa': hansen_results,
        'white_rc': white_results
    }

    # Interpretación final
    logger.info(f"\nSTATISTICAL VALIDATION:")

    if hansen_results and hansen_results['reject_null']:
        logger.info(f"  [OK] Hansen SPA: Portfolio OUTPERFORMS benchmark (p={hansen_results['p_value']:.4f})")
    elif hansen_results:
        logger.info(f"  [X] Hansen SPA: Cannot conclude outperformance (p={hansen_results['p_value']:.4f})")

    if white_results and white_results['reject_null']:
        logger.info(f"  [OK] White RC: Results ROBUST after correcting for {white_results['n_strategies_tested']} strategies (p={white_results['p_value']:.4f})")
    elif white_results:
        logger.info(f"  [X] White RC: Possible data snooping (p={white_results['p_value']:.4f})")

    if bootstrap_results and 'upi' in bootstrap_results:
        upi_ci = bootstrap_results['upi']
        logger.info(f"  [OK] Bootstrap: UPI 95% CI = [{upi_ci['ci_lower']:.4f}, {upi_ci['ci_upper']:.4f}]")

    return summary
