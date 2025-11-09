"""
Fitness Evaluation - BIDIRECTIONAL (LONG vs SHORT)
REAL BACKTESTING VERSION (Sprint 3)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

from ga_patterns.chromosome import Pattern
from backtest.runner import run_backtest

logger = logging.getLogger(__name__)

def evaluate_fitness_bidirectional(pattern: Pattern, data: pd.DataFrame,
                                   config: dict) -> Tuple[float, str]:
    """
    Evalúa patrón en LONG y SHORT con BACKTESTING REAL, retorna mejor fitness + dirección.

    Args:
        pattern: Pattern a evaluar
        data: DataFrame OHLCV
        config: config dict

    Returns:
        (best_fitness, best_direction)

    REAL BACKTESTING VERSION: Usa backtest.runner con ATR exits y costos reales.
    """
    from backtest.metrics import calculate_all_metrics

    timeframe = config['data']['timeframe']
    periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']

    # Evaluar LONG
    logger.debug(f"Evaluating pattern as LONG...")

    # Crear copia del patrón para LONG
    pattern_long = pattern.__class__(
        direction='LONG',
        window=pattern.window,
        expression=pattern.expression,
        fitness=pattern.fitness,
        fitness_long=pattern.fitness_long,
        fitness_short=pattern.fitness_short,
        generation_created=pattern.generation_created
    )

    try:
        # Run backtest LONG
        equity_long, trades_long = run_backtest(pattern_long, data, config)

        # Filtrar trades inválidos
        if len(trades_long) < 5:  # Mínimo 5 trades para ser válido
            logger.debug(f"LONG: Insufficient trades ({len(trades_long)})")
            fitness_long = -999.0
        else:
            metrics_long = calculate_all_metrics(equity_long, periods_per_year)

            # Calcular fitness LONG
            if metrics_long['cagr'] < config['ga']['fitness']['cagr_min_threshold']:
                fitness_long = -999.0
            else:
                # Safeguards para valores infinitos o muy grandes
                upi_norm = min(metrics_long['upi'], 100.0)  # Cap UPI
                sharpe_norm = min(metrics_long['sharpe'] / config['ga']['fitness']['sharpe_cap'], 1.0)
                cagr_norm = min(metrics_long['cagr'] / config['ga']['fitness']['cagr_cap'], 1.0)

                fitness_long = (
                    config['ga']['fitness']['weight_upi'] * upi_norm +
                    config['ga']['fitness']['weight_sharpe'] * sharpe_norm +
                    config['ga']['fitness']['weight_cagr'] * cagr_norm
                )

                # Final safeguard: cap total fitness
                if np.isinf(fitness_long) or np.isnan(fitness_long) or fitness_long > 1000:
                    fitness_long = -999.0

            logger.debug(f"LONG: {len(trades_long)} trades, UPI={metrics_long['upi']:.2f}, fitness={fitness_long:.4f}")

    except Exception as e:
        logger.error(f"Error calculating LONG fitness: {e}")
        fitness_long = -999.0

    # Evaluar SHORT
    logger.debug(f"Evaluating pattern as SHORT...")

    # Crear copia del patrón para SHORT
    pattern_short = pattern.__class__(
        direction='SHORT',
        window=pattern.window,
        expression=pattern.expression,
        fitness=pattern.fitness,
        fitness_long=pattern.fitness_long,
        fitness_short=pattern.fitness_short,
        generation_created=pattern.generation_created
    )

    try:
        # Run backtest SHORT
        equity_short, trades_short = run_backtest(pattern_short, data, config)

        # Filtrar trades inválidos
        if len(trades_short) < 5:  # Mínimo 5 trades para ser válido
            logger.debug(f"SHORT: Insufficient trades ({len(trades_short)})")
            fitness_short = -999.0
        else:
            metrics_short = calculate_all_metrics(equity_short, periods_per_year)

            # Calcular fitness SHORT
            if metrics_short['cagr'] < config['ga']['fitness']['cagr_min_threshold']:
                fitness_short = -999.0
            else:
                # Safeguards para valores infinitos o muy grandes
                upi_norm = min(metrics_short['upi'], 100.0)  # Cap UPI
                sharpe_norm = min(metrics_short['sharpe'] / config['ga']['fitness']['sharpe_cap'], 1.0)
                cagr_norm = min(metrics_short['cagr'] / config['ga']['fitness']['cagr_cap'], 1.0)

                fitness_short = (
                    config['ga']['fitness']['weight_upi'] * upi_norm +
                    config['ga']['fitness']['weight_sharpe'] * sharpe_norm +
                    config['ga']['fitness']['weight_cagr'] * cagr_norm
                )

                # Final safeguard: cap total fitness
                if np.isinf(fitness_short) or np.isnan(fitness_short) or fitness_short > 1000:
                    fitness_short = -999.0

            logger.debug(f"SHORT: {len(trades_short)} trades, UPI={metrics_short['upi']:.2f}, fitness={fitness_short:.4f}")

    except Exception as e:
        logger.error(f"Error calculating SHORT fitness: {e}")
        fitness_short = -999.0

    # Guardar ambos fitness
    pattern.fitness_long = fitness_long
    pattern.fitness_short = fitness_short

    # Determinar mejor dirección
    if fitness_long >= fitness_short:
        best_fitness = fitness_long
        best_direction = 'LONG'
        logger.debug(f"Pattern chose LONG (L:{fitness_long:.4f} vs S:{fitness_short:.4f})")
    else:
        best_fitness = fitness_short
        best_direction = 'SHORT'
        logger.debug(f"Pattern chose SHORT (S:{fitness_short:.4f} vs L:{fitness_long:.4f})")

    pattern.direction = best_direction
    pattern.fitness = best_fitness

    return best_fitness, best_direction

def evaluate_population(population: List[Pattern], data: pd.DataFrame,
                       config: dict) -> List[float]:
    """
    Evalúa población completa con evaluación bidireccional REAL.

    Returns:
        Lista de fitness scores
    """
    logger.info(f"Evaluating population of {len(population)} patterns (BIDIRECTIONAL, REAL BACKTEST)...")

    fitness_scores = []
    long_count = 0
    short_count = 0

    for i, pattern in enumerate(population):
        fitness, direction = evaluate_fitness_bidirectional(pattern, data, config)
        fitness_scores.append(fitness)

        if direction == 'LONG':
            long_count += 1
        else:
            short_count += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Evaluated {i+1}/{len(population)}")

    logger.info(f"[OK] Population evaluated (BIDIRECTIONAL, REAL BACKTEST)")
    logger.info(f"  Best fitness: {max(fitness_scores):.4f}")
    logger.info(f"  Mean fitness: {np.mean(fitness_scores):.4f}")
    logger.info(f"  Valid patterns: {sum(1 for f in fitness_scores if f > -999)}")
    logger.info(f"  Direction split: {long_count} LONG, {short_count} SHORT")

    return fitness_scores
