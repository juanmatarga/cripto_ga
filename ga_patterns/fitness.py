"""
Fitness Evaluation - BIDIRECTIONAL (LONG vs SHORT)
SPRINT 10: Simple Random Sampling (replaced walk-forward)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

from ga_patterns.chromosome import Pattern
from ga_patterns.chromosome_v2 import PatternChromosome
from ga_patterns.evaluator import evaluate_expression, preprocess_indicators
from backtest.runner import run_backtest

logger = logging.getLogger(__name__)

def evaluate_fitness_bidirectional(pattern, data: pd.DataFrame,
                                   config: dict, fast_mode: bool = True) -> Tuple[float, str]:
    """
    Evalúa patrón en LONG y SHORT con SIMPLE RANDOM SAMPLING.

    SPRINT 10: Replaced walk-forward with simple random windows.
    FIX SPRINT 8: Equity normalization entre ventanas usando returns.

    Args:
        pattern: Pattern o PatternChromosome a evaluar
        data: DataFrame OHLCV completo
        config: Config dict
        fast_mode: Si True, usa stratified sampling de ventanas

    Returns:
        (best_fitness, best_direction)
    """
    from backtest.metrics import calculate_all_metrics
    from backtest.simple_sampling import create_simple_windows

    # Check if pattern is new v2 or legacy
    is_v2 = isinstance(pattern, PatternChromosome)

    if is_v2:
        logger.debug(f"Evaluating PatternChromosome: {pattern.to_readable()}")
    else:
        logger.debug(f"Evaluating legacy Pattern")

    timeframe = config['data']['timeframe']
    periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']

    # ========================================================================
    # CREAR VENTANAS SIMPLE RANDOM SAMPLING (SPRINT 10)
    # ========================================================================
    # Determine number of windows based on fast_mode
    if fast_mode and config['ga'].get('fast_mode', {}).get('enabled', False):
        n_windows = config['ga']['fast_mode']['n_windows']
        window_months = config['ga']['fast_mode']['window_months']
    else:
        # Full mode: more windows
        n_windows = 10
        window_months = 1

    windows_to_eval = create_simple_windows(
        data,
        n_windows=n_windows,
        window_months=window_months,
        seed=config['ga']['seed']
    )

    if len(windows_to_eval) == 0:
        logger.warning("No valid windows created")
        pattern.fitness_long = -999.0
        pattern.fitness_short = -999.0
        pattern.fitness = -999.0
        pattern.direction = 'LONG'
        return -999.0, 'LONG'

    logger.debug(f"Evaluating on {len(windows_to_eval)} windows (fast_mode={fast_mode})")

    # ========================================================================
    # EVALUAR LONG
    # ========================================================================
    pattern.direction = 'LONG'

    # FIX SPRINT 8: Store returns instead of equity curves
    all_returns_long = []  # Store returns, not equity
    all_trades_long = []

    for i, window_data in enumerate(windows_to_eval):
        # SPRINT 10: Simple sampling - just evaluate on window_data
        equity_curve, trades = run_backtest(pattern, window_data, config)

        if len(trades) > 0:
            # CRITICAL: Extract returns from this window
            # equity_curve goes from 100 → X
            # We want the % returns per bar
            window_returns = equity_curve.pct_change().fillna(0)
            all_returns_long.append(window_returns)
            all_trades_long.extend(trades)

    # Calcular fitness LONG
    if len(all_returns_long) == 0 or len(all_trades_long) == 0:
        fitness_long = -999.0
    else:
        # FIX: Concatenate returns from all windows
        combined_returns = pd.concat(all_returns_long, ignore_index=True)

        # Rebuild equity curve starting from 100
        equity_series = 100.0 * (1 + combined_returns).cumprod()
        combined_equity_long = pd.concat([pd.Series([100.0]), equity_series], ignore_index=True)
        combined_equity_long = combined_equity_long.reset_index(drop=True)

        try:
            metrics = calculate_all_metrics(combined_equity_long, periods_per_year)

            # Hard constraints
            min_trades_total = config['selection']['filters']['min_trades_per_window'] * len(windows_to_eval)
            if metrics['cagr'] < config['ga']['fitness']['cagr_min_threshold']:
                fitness_long = -999.0
            elif len(all_trades_long) < min_trades_total:
                fitness_long = -999.0
            else:
                # SPRINT 12: New fitness calculation
                from backtest.metrics import calculate_sortino_ratio, calculate_calmar_ratio

                returns = combined_returns.dropna()
                sortino = calculate_sortino_ratio(returns, periods_per_year)
                calmar = calculate_calmar_ratio(metrics['cagr'], metrics['max_dd'])
                win_rate = len([t for t in all_trades_long if t['pnl_pct'] > 0]) / len(all_trades_long) if len(all_trades_long) > 0 else 0.0

                # Normalize
                sortino_norm = min(sortino / 3.0, 1.0) if sortino > 0 else 0.0
                calmar_norm = min(calmar / 2.0, 1.0) if calmar > 0 else 0.0
                win_rate_norm = win_rate

                fitness_long = (
                    0.4 * sortino_norm +
                    0.4 * calmar_norm +
                    0.2 * win_rate_norm
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

    # FIX SPRINT 8: Store returns instead of equity curves
    all_returns_short = []
    all_trades_short = []

    for i, window_data in enumerate(windows_to_eval):
        # SPRINT 10: Simple sampling - just evaluate on window_data
        equity_curve, trades = run_backtest(pattern, window_data, config)

        if len(trades) > 0:
            # CRITICAL: Extract returns from this window
            window_returns = equity_curve.pct_change().fillna(0)
            all_returns_short.append(window_returns)
            all_trades_short.extend(trades)

    # Calcular fitness SHORT
    if len(all_returns_short) == 0 or len(all_trades_short) == 0:
        fitness_short = -999.0
    else:
        # FIX: Concatenate returns from all windows
        combined_returns = pd.concat(all_returns_short, ignore_index=True)

        # Rebuild equity curve starting from 100
        equity_series = 100.0 * (1 + combined_returns).cumprod()
        combined_equity_short = pd.concat([pd.Series([100.0]), equity_series], ignore_index=True)
        combined_equity_short = combined_equity_short.reset_index(drop=True)

        try:
            metrics = calculate_all_metrics(combined_equity_short, periods_per_year)

            min_trades_total = config['selection']['filters']['min_trades_per_window'] * len(windows_to_eval)
            if metrics['cagr'] < config['ga']['fitness']['cagr_min_threshold']:
                fitness_short = -999.0
            elif len(all_trades_short) < min_trades_total:
                fitness_short = -999.0
            else:
                # SPRINT 12: New fitness calculation
                returns = combined_returns.dropna()
                sortino = calculate_sortino_ratio(returns, periods_per_year)
                calmar = calculate_calmar_ratio(metrics['cagr'], metrics['max_dd'])
                win_rate = len([t for t in all_trades_short if t['pnl_pct'] > 0]) / len(all_trades_short) if len(all_trades_short) > 0 else 0.0

                # Normalize
                sortino_norm = min(sortino / 3.0, 1.0) if sortino > 0 else 0.0
                calmar_norm = min(calmar / 2.0, 1.0) if calmar > 0 else 0.0
                win_rate_norm = win_rate

                fitness_short = (
                    0.4 * sortino_norm +
                    0.4 * calmar_norm +
                    0.2 * win_rate_norm
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

def evaluate_fitness_unidirectional(pattern,
                                   windows: List[pd.DataFrame],
                                   config: dict) -> Tuple[float, str]:
    """
    Evaluate pattern fitness in its NATIVE direction only (SPRINT 11).

    This is more efficient and makes more sense:
        - momentum_up patterns should only be tested as LONG
        - momentum_down patterns should only be tested as SHORT

    Args:
        pattern: PatternChromosome with native direction
        windows: Pre-created evaluation windows (reused across patterns)
        config: Configuration dict

    Returns:
        (fitness, direction): Fitness score and native direction

    Example:
        >>> pattern = PatternChromosome(direction='LONG', modules=['momentum_up_2bar'])
        >>> fitness, direction = evaluate_fitness_unidirectional(pattern, windows, config)
        >>> direction
        'LONG'
    """
    from backtest.runner import run_backtest
    from backtest.metrics import calculate_all_metrics, calculate_sortino_ratio, calculate_calmar_ratio

    logger.debug(f"Evaluating {pattern.direction} pattern: {pattern.to_readable()}")

    # Evaluate in NATIVE direction only
    all_returns = []
    all_trades = []

    for window_data in windows:
        equity_curve, trades = run_backtest(pattern, window_data, config)

        if len(trades) > 0:
            # Extract returns from this window
            window_returns = equity_curve.pct_change().fillna(0)
            all_returns.append(window_returns)
            # Convert DataFrame trades to list of dicts for win rate calculation
            all_trades.extend(trades.to_dict('records'))

    # Calculate fitness
    if len(all_returns) == 0 or len(all_trades) == 0:
        logger.debug(f"Pattern generated no trades, fitness = -999")
        pattern.fitness = -999.0
        pattern.n_trades = 0
        return -999.0, pattern.direction

    # Combine returns from all windows
    combined_returns = pd.concat(all_returns, ignore_index=True)

    # Rebuild equity curve
    combined_equity = pd.Series([100.0])
    equity_series = 100.0 * (1 + combined_returns).cumprod()
    combined_equity = pd.concat([combined_equity, equity_series], ignore_index=True)
    combined_equity = combined_equity.reset_index(drop=True)

    # Calculate metrics
    try:
        # Get timeframe-specific periods per year
        timeframe = config['data']['timeframe']
        periods_per_year = config['data']['time_map'][timeframe]['bars_per_year']

        metrics = calculate_all_metrics(combined_equity, periods_per_year)

        # Hard constraints
        if metrics['cagr'] < config['ga']['fitness']['cagr_min_threshold']:
            logger.debug(f"CAGR {metrics['cagr']:.4f} below threshold, fitness = -999")
            pattern.fitness = -999.0
            pattern.n_trades = len(all_trades)
            return -999.0, pattern.direction

        min_trades_required = (
            config['selection']['filters']['min_trades_per_window'] * len(windows)
        )

        if len(all_trades) < min_trades_required:
            logger.debug(f"Only {len(all_trades)} trades (need {min_trades_required}), fitness = -999")
            pattern.fitness = -999.0
            pattern.n_trades = len(all_trades)
            return -999.0, pattern.direction

        # SPRINT 12: Calculate fitness using Sortino + Calmar + Win Rate
        # Get returns for Sortino
        returns = combined_returns.dropna()

        # Sortino ratio (like Sharpe but only penalizes downside volatility)
        sortino = calculate_sortino_ratio(returns, periods_per_year)

        # Calmar ratio (CAGR / |max_dd|)
        calmar = calculate_calmar_ratio(metrics['cagr'], metrics['max_dd'])

        # Win rate from trades
        win_rate = len([t for t in all_trades if t['pnl_pct'] > 0]) / len(all_trades) if len(all_trades) > 0 else 0.0

        # Normalize metrics
        sortino_norm = min(sortino / 3.0, 1.0) if sortino > 0 else 0.0  # Cap at 3.0
        calmar_norm = min(calmar / 2.0, 1.0) if calmar > 0 else 0.0     # Cap at 2.0
        win_rate_norm = win_rate  # Already 0-1

        # Combined fitness with new weights
        fitness = (
            0.4 * sortino_norm +
            0.4 * calmar_norm +
            0.2 * win_rate_norm
        )

        # SPRINT 13: Trade frequency regularization penalty
        # Prevents overtrading disasters (e.g., 5395 trades on 15min timeframe)
        # Target: 1-10 trades per month (reasonable for 15min patterns)
        n_months = len(windows) * config['ga']['fast_mode']['window_months']
        avg_trades_per_month = len(all_trades) / n_months if n_months > 0 else 0

        trade_freq_penalty = 0.0
        if avg_trades_per_month > 120:  # Overtrading threshold
            # Exponential penalty for extreme overtrading
            excess = (avg_trades_per_month - 120) / 50  # Normalize
            trade_freq_penalty = min(0.3, 0.15 * excess)  # Cap at -0.3
            logger.debug(f"OVERTRADING penalty: -{trade_freq_penalty:.3f} ({avg_trades_per_month:.1f} trades/month)")
        elif avg_trades_per_month < 0.5:  # Too rare (< 1 trade per 2 months)
            trade_freq_penalty = 0.1  # Fixed penalty
            logger.debug(f"UNDERTRADING penalty: -{trade_freq_penalty:.3f} ({avg_trades_per_month:.1f} trades/month)")

        # Apply penalty
        fitness = max(0.0, fitness - trade_freq_penalty)

        # Store components for debugging
        pattern.fitness_components = {
            'sortino': sortino,
            'sortino_norm': sortino_norm,
            'calmar': calmar,
            'calmar_norm': calmar_norm,
            'win_rate': win_rate,
            'win_rate_norm': win_rate_norm,
            'trade_freq_penalty': trade_freq_penalty,
            'avg_trades_per_month': avg_trades_per_month
        }

        # Safeguard against inf/nan
        if np.isinf(fitness) or np.isnan(fitness) or fitness > 1000:
            fitness = -999.0

        logger.debug(f"Fitness = {fitness:.4f} (Sortino={sortino:.2f}/{sortino_norm:.2f}, Calmar={calmar:.2f}/{calmar_norm:.2f}, WinRate={win_rate:.2%}, Trades/mo={avg_trades_per_month:.1f})")

        # Store detailed metrics in pattern
        pattern.fitness = fitness
        pattern.metrics = metrics
        pattern.n_trades = len(all_trades)

        # Store direction-specific fitness
        if pattern.direction == 'LONG':
            pattern.fitness_long = fitness
            pattern.fitness_short = -999.0
        else:
            pattern.fitness_short = fitness
            pattern.fitness_long = -999.0

        return fitness, pattern.direction

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        pattern.fitness = -999.0
        pattern.n_trades = 0
        return -999.0, pattern.direction

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
