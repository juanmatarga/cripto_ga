"""
White's Reality Check - Data Snooping Correction
Adjusts for multiple testing when selecting from many strategies
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def whites_reality_check(strategy_returns_list: List[pd.Series],
                        benchmark_returns: pd.Series,
                        n_bootstrap: int = 1000,
                        alpha: float = 0.05,
                        seed: int = 42) -> Dict:
    """
    White's Reality Check Test.

    Corrige por data snooping cuando se elige el mejor de K estrategias.

    H0: max_k(E[strategy_k - benchmark]) <= 0
    H1: max_k(E[strategy_k - benchmark]) > 0

    Args:
        strategy_returns_list: Lista de returns de K estrategias candidatas
        benchmark_returns: Returns de benchmark
        n_bootstrap: Iteraciones bootstrap
        alpha: Nivel de significancia
        seed: Seed

    Returns:
        dict: {
            'max_performance': float,
            'p_value': float,
            'reject_null': bool,
            'best_strategy_idx': int,
            'n_strategies_tested': int
        }
    """
    logger.info("Running White's Reality Check...")
    logger.info(f"  Testing {len(strategy_returns_list)} strategies")
    logger.info(f"  Null hypothesis: NO strategy outperforms benchmark")
    logger.info(f"  Significance level: {alpha}")

    np.random.seed(seed)

    if len(strategy_returns_list) == 0:
        logger.error("No strategies provided")
        return {
            'max_performance': 0.0,
            'p_value': 1.0,
            'reject_null': False,
            'best_strategy_idx': -1,
            'n_strategies_tested': 0
        }

    # Calcular performance relativo para cada estrategia
    relative_perfs = []

    for i, strat_returns in enumerate(strategy_returns_list):
        # Alinear con benchmark
        aligned_strat, aligned_bench = strat_returns.align(benchmark_returns, join='inner')

        if len(aligned_strat) == 0:
            logger.warning(f"Strategy {i}: No overlapping periods, skipping")
            continue

        # Diferencia
        d = aligned_strat - aligned_bench
        relative_perfs.append(d)

    if len(relative_perfs) == 0:
        logger.error("No valid strategies after alignment")
        return {
            'max_performance': 0.0,
            'p_value': 1.0,
            'reject_null': False,
            'best_strategy_idx': -1,
            'n_strategies_tested': 0
        }

    # Convertir a DataFrame para operaciones vectorizadas
    rel_perf_df = pd.DataFrame(relative_perfs).T

    # Test statistic: MAX de las medias
    mean_perfs = rel_perf_df.mean(axis=0)
    max_mean_perf = mean_perfs.max()
    best_idx = mean_perfs.idxmax()

    logger.info(f"  Best strategy: #{best_idx} with mean outperformance: {max_mean_perf:.6f}")

    # Bootstrap bajo H0
    logger.info(f"  Running {n_bootstrap} bootstrap iterations...")

    n_periods = len(rel_perf_df)
    bootstrap_max_stats = []

    # Centrar bajo H0 (restar mean de cada estrategia)
    centered_df = rel_perf_df - rel_perf_df.mean(axis=0)

    for i in range(n_bootstrap):
        # Resample períodos con reemplazo
        boot_indices = np.random.choice(n_periods, size=n_periods, replace=True)
        boot_sample = centered_df.iloc[boot_indices]

        # Max de las medias en esta muestra
        boot_means = boot_sample.mean(axis=0)
        boot_max = boot_means.max()

        bootstrap_max_stats.append(boot_max)

        if (i + 1) % 100 == 0:
            logger.debug(f"    Bootstrap iteration {i+1}/{n_bootstrap}")

    bootstrap_max_stats = np.array(bootstrap_max_stats)

    # P-value: proporción de bootstrap max >= observed max
    p_value = (bootstrap_max_stats >= max_mean_perf).mean()

    # Decisión
    reject_null = p_value < alpha

    logger.info(f"\n{'='*80}")
    logger.info("WHITE'S REALITY CHECK RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Number of strategies tested: {len(relative_perfs)}")
    logger.info(f"Best strategy index: {best_idx}")
    logger.info(f"Max mean outperformance: {max_mean_perf:.6f}")
    logger.info(f"P-value (corrected): {p_value:.4f}")
    logger.info(f"Significance level: {alpha}")
    logger.info(f"Decision: {'REJECT H0' if reject_null else 'FAIL TO REJECT H0'}")

    if reject_null:
        logger.info(f"[OK] At least one strategy outperforms benchmark (p < {alpha})")
        logger.info(f"  Even after correcting for {len(relative_perfs)} comparisons")
    else:
        logger.info(f"[X] Cannot conclude any strategy outperforms (p >= {alpha})")
        logger.info(f"  Results may be due to data snooping")

    return {
        'max_performance': float(max_mean_perf),
        'p_value': float(p_value),
        'reject_null': bool(reject_null),
        'best_strategy_idx': int(best_idx),
        'n_strategies_tested': len(relative_perfs),
        'alpha': alpha
    }
