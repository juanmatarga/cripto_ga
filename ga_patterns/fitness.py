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
                                   config: dict, fast_mode: bool = True) -> Tuple[float, str]:
    """
    Evalúa patrón en LONG y SHORT con WALK-FORWARD REAL.

    CRÍTICO: Esta versión SÍ usa walk-forward correctamente.

    Args:
        pattern: Pattern a evaluar
        data: DataFrame OHLCV completo
        config: Config dict
        fast_mode: Si True, usa stratified sampling de ventanas

    Returns:
        (best_fitness, best_direction)
    """
    from backtest.metrics import calculate_all_metrics
    from backtest.walkforward import create_walkforward_windows, stratified_sampling_windows

    timeframe = config['data']['timeframe']
    periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']

    # ========================================================================
    # CREAR VENTANAS WALK-FORWARD
    # ========================================================================
    windows = create_walkforward_windows(
        data,
        train_months=config['walkforward']['train_months'],
        test_months=config['walkforward']['test_months'],
        step_months=config['walkforward']['step_months']
    )

    if len(windows) == 0:
        logger.warning("No valid windows created")
        pattern.fitness_long = -999.0
        pattern.fitness_short = -999.0
        pattern.fitness = -999.0
        pattern.direction = 'LONG'
        return -999.0, 'LONG'

    # Stratified sampling si fast_mode
    if fast_mode and config['ga'].get('fast_mode', {}).get('enabled', False):
        windows_to_eval = stratified_sampling_windows(
            windows,
            n_sample=config['ga']['fast_mode']['n_windows'],
            seed=config['ga']['seed']
        )
    else:
        windows_to_eval = windows

    logger.debug(f"Evaluating on {len(windows_to_eval)} windows (fast_mode={fast_mode})")

    # ========================================================================
    # EVALUAR LONG
    # ========================================================================
    pattern.direction = 'LONG'

    # Combinar equity curves de SOLO test sets (OOS)
    all_equity_long = []
    all_trades_long = []

    for i, (train_df, test_df) in enumerate(windows_to_eval):
        # CRÍTICO: Backtest SOLO en test_df (OOS)
        equity_curve, trades = run_backtest(pattern, test_df, config)

        if len(trades) > 0:
            all_equity_long.append(equity_curve)
            all_trades_long.extend(trades)

    # Calcular fitness LONG
    if len(all_equity_long) == 0 or len(all_trades_long) == 0:
        fitness_long = -999.0
    else:
        # Concatenar equity curves
        combined_equity_long = pd.concat(all_equity_long)

        try:
            metrics = calculate_all_metrics(combined_equity_long, periods_per_year)

            # Hard constraints
            min_trades_total = config['selection']['filters']['min_trades_per_window'] * len(windows_to_eval)
            if metrics['cagr'] < config['ga']['fitness']['cagr_min_threshold']:
                fitness_long = -999.0
            elif len(all_trades_long) < min_trades_total:
                fitness_long = -999.0
            else:
                # Fitness combinado con caps
                upi_norm = min(metrics['upi'], 100.0)
                sharpe_norm = min(metrics['sharpe'] / config['ga']['fitness']['sharpe_cap'], 1.0)
                cagr_norm = min(metrics['cagr'] / config['ga']['fitness']['cagr_cap'], 1.0)

                fitness_long = (
                    config['ga']['fitness']['weight_upi'] * upi_norm +
                    config['ga']['fitness']['weight_sharpe'] * sharpe_norm +
                    config['ga']['fitness']['weight_cagr'] * cagr_norm
                )

                # Final safeguards
                if np.isinf(fitness_long) or np.isnan(fitness_long) or fitness_long > 1000:
                    fitness_long = -999.0

        except Exception as e:
            logger.error(f"Error calculating LONG fitness: {e}")
            fitness_long = -999.0

    pattern.fitness_long = fitness_long
    logger.debug(f"LONG: {len(all_trades_long)} trades, fitness={fitness_long:.4f}")

    # ========================================================================
    # EVALUAR SHORT
    # ========================================================================
    pattern.direction = 'SHORT'

    all_equity_short = []
    all_trades_short = []

    for i, (train_df, test_df) in enumerate(windows_to_eval):
        equity_curve, trades = run_backtest(pattern, test_df, config)

        if len(trades) > 0:
            all_equity_short.append(equity_curve)
            all_trades_short.extend(trades)

    # Calcular fitness SHORT
    if len(all_equity_short) == 0 or len(all_trades_short) == 0:
        fitness_short = -999.0
    else:
        combined_equity_short = pd.concat(all_equity_short)

        try:
            metrics = calculate_all_metrics(combined_equity_short, periods_per_year)

            min_trades_total = config['selection']['filters']['min_trades_per_window'] * len(windows_to_eval)
            if metrics['cagr'] < config['ga']['fitness']['cagr_min_threshold']:
                fitness_short = -999.0
            elif len(all_trades_short) < min_trades_total:
                fitness_short = -999.0
            else:
                upi_norm = min(metrics['upi'], 100.0)
                sharpe_norm = min(metrics['sharpe'] / config['ga']['fitness']['sharpe_cap'], 1.0)
                cagr_norm = min(metrics['cagr'] / config['ga']['fitness']['cagr_cap'], 1.0)

                fitness_short = (
                    config['ga']['fitness']['weight_upi'] * upi_norm +
                    config['ga']['fitness']['weight_sharpe'] * sharpe_norm +
                    config['ga']['fitness']['weight_cagr'] * cagr_norm
                )

                if np.isinf(fitness_short) or np.isnan(fitness_short) or fitness_short > 1000:
                    fitness_short = -999.0

        except Exception as e:
            logger.error(f"Error calculating SHORT fitness: {e}")
            fitness_short = -999.0

    pattern.fitness_short = fitness_short
    logger.debug(f"SHORT: {len(all_trades_short)} trades, fitness={fitness_short:.4f}")

    # ========================================================================
    # MEJOR DIRECCIÓN
    # ========================================================================
    if fitness_long >= fitness_short:
        pattern.direction = 'LONG'
        pattern.fitness = fitness_long
        logger.debug(f"Pattern chose LONG (L:{fitness_long:.4f} vs S:{fitness_short:.4f})")
        return fitness_long, 'LONG'
    else:
        pattern.direction = 'SHORT'
        pattern.fitness = fitness_short
        logger.debug(f"Pattern chose SHORT (S:{fitness_short:.4f} vs L:{fitness_long:.4f})")
        return fitness_short, 'SHORT'

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
