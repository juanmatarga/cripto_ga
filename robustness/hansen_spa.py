"""
Hansen's SPA Test - Superior Predictive Ability
Tests if strategy has superior performance vs benchmark
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def calculate_relative_performance(strategy_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> pd.Series:
    """
    Calcula performance relativo: strategy - benchmark.

    Args:
        strategy_returns: Returns de estrategia
        benchmark_returns: Returns de benchmark (buy & hold)

    Returns:
        pd.Series: Diferencias de returns
    """
    # Alinear índices
    aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')

    # Diferencia
    relative_perf = aligned_strategy - aligned_benchmark

    return relative_perf

def hansen_spa_test(strategy_returns: pd.Series,
                   benchmark_returns: pd.Series,
                   n_bootstrap: int = 1000,
                   alpha: float = 0.05,
                   seed: int = 42) -> Dict:
    """
    Hansen's Superior Predictive Ability (SPA) Test.

    H0: max(E[strategy - benchmark]) <= 0
    H1: max(E[strategy - benchmark]) > 0

    Args:
        strategy_returns: Returns de estrategia a testear
        benchmark_returns: Returns de benchmark (ej: buy & hold)
        n_bootstrap: Número de iteraciones bootstrap
        alpha: Nivel de significancia (default 0.05)
        seed: Seed para reproducibilidad

    Returns:
        dict: {
            'test_statistic': float,
            'p_value': float,
            'reject_null': bool,
            'mean_outperformance': float
        }
    """
    logger.info("Running Hansen SPA Test...")
    logger.info(f"  Null hypothesis: Strategy does NOT outperform benchmark")
    logger.info(f"  Alternative: Strategy DOES outperform benchmark")
    logger.info(f"  Significance level: {alpha}")

    np.random.seed(seed)

    # Calcular performance relativo
    d = calculate_relative_performance(strategy_returns, benchmark_returns)

    if len(d) == 0:
        logger.error("No overlapping periods between strategy and benchmark")
        return {
            'test_statistic': 0.0,
            'p_value': 1.0,
            'reject_null': False,
            'mean_outperformance': 0.0
        }

    # Test statistic: mean de outperformance
    d_mean = d.mean()
    d_std = d.std()
    n = len(d)

    if d_std == 0:
        logger.warning("Zero variance in relative performance")
        t_stat = 0.0 if d_mean == 0 else (np.inf if d_mean > 0 else -np.inf)
    else:
        t_stat = np.sqrt(n) * d_mean / d_std

    logger.info(f"  Test statistic (t): {t_stat:.4f}")
    logger.info(f"  Mean outperformance: {d_mean:.6f} ({d_mean*100:.4f}% per period)")

    # Bootstrap para p-value
    logger.info(f"  Running {n_bootstrap} bootstrap iterations...")

    bootstrap_stats = []
    d_values = d.values

    for i in range(n_bootstrap):
        # Resample con reemplazo (centered bootstrap)
        # Centrar en 0 bajo H0 (no outperformance)
        boot_sample = np.random.choice(d_values - d_mean, size=n, replace=True)

        boot_mean = boot_sample.mean()
        boot_std = boot_sample.std()

        if boot_std > 0:
            boot_t = np.sqrt(n) * boot_mean / boot_std
        else:
            boot_t = 0.0

        bootstrap_stats.append(boot_t)

        if (i + 1) % 100 == 0:
            logger.debug(f"    Bootstrap iteration {i+1}/{n_bootstrap}")

    bootstrap_stats = np.array(bootstrap_stats)

    # P-value: proporción de bootstrap stats >= observed stat
    p_value = (bootstrap_stats >= t_stat).mean()

    # Decisión
    reject_null = p_value < alpha

    logger.info(f"\n{'='*80}")
    logger.info("HANSEN SPA TEST RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Test statistic: {t_stat:.4f}")
    logger.info(f"P-value: {p_value:.4f}")
    logger.info(f"Significance level: {alpha}")
    logger.info(f"Decision: {'REJECT H0' if reject_null else 'FAIL TO REJECT H0'}")

    if reject_null:
        logger.info(f"[OK] Strategy has SUPERIOR performance vs benchmark (p < {alpha})")
    else:
        logger.info(f"[X] Cannot conclude superior performance (p >= {alpha})")

    return {
        'test_statistic': float(t_stat),
        'p_value': float(p_value),
        'reject_null': bool(reject_null),
        'mean_outperformance': float(d_mean),
        'alpha': alpha
    }
