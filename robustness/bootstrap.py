"""
Monte Carlo Bootstrap - Confidence intervals with block bootstrap
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def block_bootstrap_returns(returns: pd.Series, block_size: int,
                           n_iterations: int, seed: int = 42) -> np.ndarray:
    """
    Block bootstrap de returns preservando autocorrelación.

    Args:
        returns: Serie de returns
        block_size: Tamaño del bloque (para preservar autocorrelación)
        n_iterations: Número de iteraciones bootstrap
        seed: Seed para reproducibilidad

    Returns:
        np.ndarray: (n_iterations, len(returns)) array de returns bootstrapped
    """
    np.random.seed(seed)

    n = len(returns)
    returns_array = returns.values

    # Calcular número de bloques
    n_blocks = int(np.ceil(n / block_size))

    bootstrapped_returns = []

    for _ in range(n_iterations):
        # Samplear bloques con reemplazo
        bootstrapped_sample = []

        for _ in range(n_blocks):
            # Índice de inicio random
            start_idx = np.random.randint(0, max(1, n - block_size + 1))
            end_idx = min(start_idx + block_size, n)

            # Extraer bloque
            block = returns_array[start_idx:end_idx]
            bootstrapped_sample.extend(block)

        # Truncar a longitud original
        bootstrapped_sample = bootstrapped_sample[:n]
        bootstrapped_returns.append(bootstrapped_sample)

    return np.array(bootstrapped_returns)

def calculate_bootstrap_statistics(equity_curve: pd.Series,
                                  periods_per_year: int,
                                  config: dict) -> Dict[str, Dict]:
    """
    Calcula estadísticas con intervalos de confianza via bootstrap.

    Args:
        equity_curve: Serie de equity
        periods_per_year: Períodos por año (para anualización)
        config: config['robustness']['bootstrap'] dict

    Returns:
        dict: Estadísticas con intervalos de confianza
        {
            'upi': {'mean': float, 'ci_lower': float, 'ci_upper': float, 'std': float},
            'sharpe': {...},
            'cagr': {...},
            ...
        }
    """
    from backtest.metrics import calculate_all_metrics

    logger.info("Calculating bootstrap statistics...")

    # Parámetros
    n_iterations = config['n_iterations']
    block_size = config['block_size']
    confidence_level = config['confidence_level']
    seed = config.get('seed', 42)

    # Calcular returns
    returns = equity_curve.pct_change().dropna()

    if len(returns) < block_size:
        logger.warning(f"Equity curve too short ({len(returns)}) for block size {block_size}")
        return {}

    # Bootstrap returns
    logger.info(f"  Running {n_iterations} bootstrap iterations (block size: {block_size})...")
    bootstrapped_returns = block_bootstrap_returns(returns, block_size, n_iterations, seed)

    # Calcular métricas para cada muestra bootstrap
    bootstrap_metrics = {
        'upi': [],
        'sharpe': [],
        'cagr': [],
        'max_dd': [],
        'ulcer_index': [],
        'volatility': []
    }

    for i in range(n_iterations):
        # Reconstruir equity curve
        boot_returns = pd.Series(bootstrapped_returns[i])
        boot_equity = equity_curve.iloc[0] * (1 + boot_returns).cumprod()

        # Calcular métricas
        try:
            metrics = calculate_all_metrics(boot_equity, periods_per_year)

            # Sanity check: reject obviously corrupt values
            if abs(metrics['upi']) > 1e6 or abs(metrics['cagr']) > 1e6:
                logger.debug(f"  Bootstrap iter {i}: rejecting corrupt metrics "
                           f"(UPI={metrics['upi']:.0f}, CAGR={metrics['cagr']:.0f})")
                continue

            bootstrap_metrics['upi'].append(metrics['upi'])
            bootstrap_metrics['sharpe'].append(metrics['sharpe'])
            bootstrap_metrics['cagr'].append(metrics['cagr'])
            bootstrap_metrics['max_dd'].append(metrics['max_dd'])
            bootstrap_metrics['ulcer_index'].append(metrics['ulcer_index'])
            bootstrap_metrics['volatility'].append(metrics['volatility'])
        except Exception as e:
            logger.debug(f"  Bootstrap iter {i} failed: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"    Completed {i+1}/{n_iterations} iterations")

    # Calcular intervalos de confianza
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {}
    for metric_name, values in bootstrap_metrics.items():
        if not values:
            continue

        results[metric_name] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, lower_percentile),
            'ci_upper': np.percentile(values, upper_percentile),
            'values': values  # Para visualización
        }

    logger.info(f"[OK] Bootstrap statistics calculated")
    logger.info(f"  Confidence level: {confidence_level*100}%")

    return results

def print_bootstrap_summary(bootstrap_results: Dict[str, Dict]):
    """
    Imprime resumen de resultados bootstrap.

    Args:
        bootstrap_results: Output de calculate_bootstrap_statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info("BOOTSTRAP RESULTS (95% Confidence Intervals)")
    logger.info(f"{'='*80}")

    for metric_name, stats in bootstrap_results.items():
        logger.info(f"\n{metric_name.upper()}:")
        logger.info(f"  Mean:   {stats['mean']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  Std:    {stats['std']:.4f}")
        logger.info(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
