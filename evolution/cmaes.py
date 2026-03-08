"""
CMA-ES parameter optimization for evolved strategies.

Phase 2 of hybrid optimization:
  Phase 1 (GE): Discover strategy STRUCTURE (which indicators, comparators, logic)
  Phase 2 (CMA-ES): Fine-tune continuous PARAMETERS (periods, thresholds, exits)

CMA-ES can explore parameter values BEYOND the grammar's discrete lists
(e.g., RSI period 11 instead of only {7, 9, 14, 21}).
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import cma

from strategy.phenotype import Strategy
from evolution.param_extractor import (
    ParamSpec, extract_params, rebuild_strategy, tighten_bounds
)
from evolution.fitness import (
    evaluate_strategy, _run_single_window, BARS_PER_YEAR_15M, FAIL_FITNESS,
    _combine_equity_curves
)
from backtest.metrics import (
    calculate_returns, calculate_sortino_ratio, calculate_calmar_ratio,
    cagr, max_drawdown
)
from strategy.vectorized_eval import generate_signals
from data.multi_timeframe import prepare_multi_tf_data
from validation.cpcv import create_cpcv_groups, generate_cpcv_splits, apply_purge_embargo

logger = logging.getLogger(__name__)


@dataclass
class CMAResult:
    """Result of CMA-ES optimization for a single strategy."""
    original_strategy: Strategy
    optimized_strategy: Strategy
    original_fitness: float
    optimized_fitness: float
    improvement_pct: float
    param_specs: List[ParamSpec]
    original_params: List[float]
    optimized_params: List[float]
    n_evals: int
    converged: bool


def _sample_windows(data: pd.DataFrame, n_windows: int,
                    window_bars: int, rng: np.random.RandomState
                    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
    """Sample evaluation windows with pre-computed multi-TF data."""
    max_start = len(data) - window_bars
    if max_start <= 0:
        windows = [data]
        tf_data = [prepare_multi_tf_data(data)]
        return windows, tf_data

    starts = rng.choice(max_start, size=min(n_windows, max_start), replace=False)
    windows = []
    tf_data_list = []
    for s in sorted(starts):
        w = data.iloc[s:s + window_bars]
        windows.append(w)
        tf_data_list.append(prepare_multi_tf_data(w))
    return windows, tf_data_list


def _evaluate_params_soft(param_vector: List[float], param_specs: List[ParamSpec],
                          base_strategy: Strategy, windows: List[pd.DataFrame],
                          windows_tf_data: List[Dict], config: dict) -> float:
    """
    Evaluate a parameter vector with SOFT penalties (no cliff edges).

    Unlike the evolution fitness which uses hard FAIL_FITNESS for constraint
    violations, this uses gradual penalties. This is critical for CMA-ES
    because small parameter changes near constraint boundaries shouldn't
    cause catastrophic fitness jumps.

    Returns NEGATIVE fitness (CMA-ES minimizes).
    """
    strategy = rebuild_strategy(base_strategy, param_vector, param_specs)

    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)
    fitness_cfg = config.get('fitness', {})
    min_trades = fitness_cfg.get('min_trades', 30)
    max_dd = fitness_cfg.get('max_drawdown', 0.30)
    min_wr = fitness_cfg.get('min_win_rate', 0.20)

    all_trades = []
    all_equity_curves = []

    for i_w, window_df in enumerate(windows):
        try:
            tf_data = windows_tf_data[i_w] if windows_tf_data else None
            equity, trades = _run_single_window(
                strategy, window_df, costs_config, atr_period,
                tf_data=tf_data,
            )
            all_equity_curves.append(equity)
            all_trades.extend(trades)
        except Exception:
            continue

    if not all_equity_curves:
        return 100.0  # Bad but not catastrophic

    n_trades = len(all_trades)

    # Soft penalty: trade count
    # Instead of FAIL_FITNESS at n_trades < min_trades, apply gradual penalty
    trade_penalty = 0.0
    if n_trades < min_trades:
        # Penalty proportional to shortfall (0 to ~5)
        trade_penalty = 5.0 * (1.0 - n_trades / max(min_trades, 1))
    if n_trades == 0:
        return 50.0  # No trades — bad

    # Compute aggregate metrics
    combined_equity = _combine_equity_curves(all_equity_curves)
    returns = calculate_returns(combined_equity).dropna()

    if len(returns) < 10:
        return 50.0

    # Win rate
    winning = sum(1 for t in all_trades if t['pnl_pct'] > 0)
    win_rate = winning / n_trades

    # Soft penalty: win rate
    wr_penalty = 0.0
    if win_rate < min_wr:
        wr_penalty = 3.0 * (1.0 - win_rate / max(min_wr, 0.01))

    # Metrics
    sortino = calculate_sortino_ratio(returns, BARS_PER_YEAR_15M)
    cagr_val = cagr(combined_equity, BARS_PER_YEAR_15M)
    max_dd_val = max_drawdown(combined_equity)
    calmar = calculate_calmar_ratio(cagr_val, max_dd_val)

    # Soft penalty: drawdown
    dd_penalty = 0.0
    if abs(max_dd_val) > max_dd:
        dd_penalty = 5.0 * (abs(max_dd_val) - max_dd) / max_dd

    # Cap extreme values
    sortino = max(min(sortino, 10.0), -10.0)
    calmar = min(calmar, 10.0)

    # Trade-level metrics
    winning_pnl = sum(t['pnl_pct'] for t in all_trades if t['pnl_pct'] > 0)
    losing_pnl = abs(sum(t['pnl_pct'] for t in all_trades if t['pnl_pct'] < 0))
    profit_factor = winning_pnl / max(losing_pnl, 1e-10)

    avg_win = winning_pnl / max(winning, 1)
    avg_loss = losing_pnl / max(n_trades - winning, 1)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Soft penalty: negative expectancy
    exp_penalty = 0.0
    if expectancy <= 0:
        exp_penalty = 3.0 + abs(expectancy) * 10.0

    # Fitness (same formula as evolution but with soft penalties)
    pf_bonus = min(profit_factor - 1.0, 3.0) if profit_factor > 1.0 else 0.0
    cagr_bonus = max(cagr_val, 0) * 10.0
    calmar_bonus = max(min(calmar, 5.0), 0) * 0.3
    wl_ratio = avg_win / max(avg_loss, 1e-6)
    wl_bonus = min(wl_ratio - 1.0, 3.0) if wl_ratio > 1.0 else 0.0

    fitness = sortino + cagr_bonus + calmar_bonus + pf_bonus + wl_bonus
    fitness -= (trade_penalty + wr_penalty + dd_penalty + exp_penalty)

    # Return negative (CMA-ES minimizes)
    return -fitness


def _prepare_cpcv_splits(data: pd.DataFrame, n_groups: int = 6,
                          purge_bars: int = 96, embargo_bars: int = 48
                          ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Pre-compute mini-CPCV train/test splits for reuse across evaluations."""
    groups = create_cpcv_groups(data, n_groups)
    raw_splits = generate_cpcv_splits(n_groups)
    splits = []
    for train_idx, test_idx in raw_splits:
        train_df, test_df = apply_purge_embargo(
            groups, train_idx, test_idx, purge_bars, embargo_bars
        )
        if len(train_df) > 500 and len(test_df) > 200:
            splits.append((train_df, test_df))
    return splits


def _evaluate_cpcv(param_vector: List[float], param_specs: List[ParamSpec],
                    base_strategy: Strategy, cpcv_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
                    config: dict) -> float:
    """
    Evaluate parameters using mini-CPCV: mean OOS sortino across splits.

    This is MORE ROBUST than windowed fitness because:
    1. Uses ALL data with proper train/test splits
    2. Purge + embargo prevent information leakage
    3. Multiple splits average out noise
    4. OOS evaluation prevents within-sample overfitting

    Returns NEGATIVE fitness (CMA-ES minimizes).
    """
    strategy = rebuild_strategy(base_strategy, param_vector, param_specs)

    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)

    oos_sortinos = []

    for train_df, test_df in cpcv_splits:
        try:
            # Run on TEST data only (OOS evaluation)
            equity, trades = _run_single_window(
                strategy, test_df, costs_config, atr_period
            )
            returns = calculate_returns(equity).dropna()
            if len(returns) > 10 and len(trades) > 2:
                sortino = calculate_sortino_ratio(returns, BARS_PER_YEAR_15M)
                sortino = max(min(sortino, 10.0), -10.0)
                oos_sortinos.append(sortino)
            else:
                oos_sortinos.append(-5.0)  # Soft penalty for no trades
        except Exception:
            oos_sortinos.append(-5.0)

    if not oos_sortinos:
        return 50.0

    mean_sortino = np.mean(oos_sortinos)
    # Bonus for consistency (low std = more robust)
    std_sortino = np.std(oos_sortinos) if len(oos_sortinos) > 1 else 0
    consistency_bonus = max(0, 1.0 - std_sortino) * 0.5

    # Bonus for positive percentage
    pct_positive = sum(1 for s in oos_sortinos if s > 0) / len(oos_sortinos)
    pct_bonus = pct_positive * 1.0

    fitness = mean_sortino + consistency_bonus + pct_bonus
    return -fitness


def _count_cpcv_trades(param_vector: List[float], param_specs: List[ParamSpec],
                       base_strategy: Strategy,
                       cpcv_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
                       config: dict) -> float:
    """Count mean trades across CPCV OOS splits for a parameter vector."""
    strategy = rebuild_strategy(base_strategy, param_vector, param_specs)
    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)

    trade_counts = []
    for _, test_df in cpcv_splits:
        try:
            _, trades = _run_single_window(strategy, test_df, costs_config, atr_period)
            trade_counts.append(len(trades))
        except Exception:
            trade_counts.append(0)
    return float(np.mean(trade_counts)) if trade_counts else 0.0


def optimize_strategy(strategy: Strategy, data: pd.DataFrame,
                      config: dict, max_evals: int = 200,
                      sigma0: float = 0.05, seed: int = 42,
                      n_windows: int = 20, window_bars: int = 5760,
                      ) -> CMAResult:
    """
    Optimize a strategy's parameters using CMA-ES with multi-seed consensus.

    Architecture (v5 — multi-seed consensus):
      1. Adaptive bounds centered on GE-discovered values
      2. Ultra-conservative sigma=0.05 (tiny parameter nudges only)
      3. Run CMA-ES with 3 different seeds
      4. Average best parameters across seeds (consensus)
      5. Full CPCV gate: only accept if consensus improves full CPCV

    Key insight: Tiny parameter changes (sigma=0.05) are robust.
    Multi-seed consensus prevents seed-dependent overfitting.

    Args:
        strategy: Decoded Strategy with structure to preserve
        data: Evolution data (NOT OTS)
        config: Full config dict
        max_evals: Max evaluations PER SEED
        sigma0: Initial step size (0.05 = ultra-conservative)
        seed: Base random seed
        n_windows: Windows per evaluation sample
        window_bars: Bars per window
    """
    # Extract parameters with ADAPTIVE bounds
    raw_specs = extract_params(strategy)
    if not raw_specs:
        logger.warning("No tunable parameters found")
        return CMAResult(
            original_strategy=strategy, optimized_strategy=strategy,
            original_fitness=0.0, optimized_fitness=0.0,
            improvement_pct=0.0, param_specs=[], original_params=[],
            optimized_params=[], n_evals=0, converged=False,
        )

    param_specs = tighten_bounds(raw_specs)
    n_params = len(param_specs)
    logger.info(f"CMA-ES: {n_params} parameters (multi-seed consensus, σ={sigma0})")
    for ps in param_specs:
        logger.info(f"  {ps.name}: {ps.value:.2f} [{ps.bounds[0]:.1f}, {ps.bounds[1]:.1f}]")

    x0 = np.array([ps.value for ps in param_specs])
    lower = np.array([ps.bounds[0] for ps in param_specs])
    upper = np.array([ps.bounds[1] for ps in param_specs])
    ranges = np.maximum(upper - lower, 1e-8)

    # ======================================================================
    # MULTI-SEED OPTIMIZATION: run 3 independent CMA-ES with different seeds
    # Average best params → consensus parameters
    # ======================================================================
    n_seeds = 3
    all_best_params = []
    total_evals = 0

    for seed_i in range(n_seeds):
        run_seed = seed + seed_i * 100
        rng = np.random.RandomState(run_seed)
        best_fitness = -1e9
        best_params = x0.tolist()

        n_resamples = 5
        evals_per_resample = max_evals // n_resamples

        for resample_i in range(n_resamples):
            windows, windows_tf_data = _sample_windows(
                data, n_windows, window_bars, rng
            )

            def objective(x_norm):
                x = lower + np.array(x_norm) * ranges
                return _evaluate_params_soft(
                    x.tolist(), param_specs, strategy, windows, windows_tf_data, config
                )

            opts = {
                'seed': run_seed + resample_i,
                'bounds': [0, 1],
                'verbose': -9,
                'tolfun': 1e-7,
                'tolx': 1e-7,
                'maxfevals': evals_per_resample,
                'popsize': max(4 + int(3 * np.log(n_params)), 8),
            }
            x0_round = np.clip((np.array(best_params) - lower) / ranges, 0, 1)
            es = cma.CMAEvolutionStrategy(x0_round.tolist(), sigma0, opts)

            while not es.stop():
                solutions = es.ask()
                fitnesses = [objective(x) for x in solutions]
                es.tell(solutions, fitnesses)
                total_evals += len(solutions)

            result_denorm = lower + np.array(es.result.xbest) * ranges
            result_fitness = -es.result.fbest

            if result_fitness > best_fitness:
                best_fitness = result_fitness
                best_params = result_denorm.tolist()

        all_best_params.append(best_params)
        logger.info(f"  Seed {seed_i + 1}/{n_seeds}: best fitness={best_fitness:.4f}")

    # Consensus: average best params across seeds (averaging = regularization)
    consensus_params = np.mean(all_best_params, axis=0).tolist()
    logger.info(f"Consensus from {n_seeds} seeds ({total_evals} total evals)")

    # ======================================================================
    # "DO NO HARM" GATE: Full CPCV comparison + trade stability
    # ======================================================================
    logger.info("Do-no-harm: full CPCV comparison...")
    val_cfg = config.get('validation', {})
    purge_bars = val_cfg.get('cpcv_purge_bars', 96)
    embargo_bars = val_cfg.get('cpcv_embargo_bars', 48)

    full_splits = _prepare_cpcv_splits(data, 10, purge_bars, embargo_bars)

    orig_cpcv = -_evaluate_cpcv(x0.tolist(), param_specs, strategy, full_splits, config)
    opt_cpcv = -_evaluate_cpcv(consensus_params, param_specs, strategy, full_splits, config)

    logger.info(f"  CPCV: original={orig_cpcv:.4f}, optimized={opt_cpcv:.4f}")

    if opt_cpcv <= orig_cpcv:
        logger.info("  → REJECTED: no CPCV improvement.")
        return CMAResult(
            original_strategy=strategy,
            optimized_strategy=strategy,
            original_fitness=orig_cpcv,
            optimized_fitness=orig_cpcv,
            improvement_pct=0.0,
            param_specs=param_specs,
            original_params=x0.tolist(),
            optimized_params=x0.tolist(),
            n_evals=total_evals,
            converged=False,
        )

    # Trade stability check: reject if trade count changes too much on CPCV splits
    orig_strategy_trades = _count_cpcv_trades(
        x0.tolist(), param_specs, strategy, full_splits, config)
    opt_strategy_trades = _count_cpcv_trades(
        consensus_params, param_specs, strategy, full_splits, config)

    if orig_strategy_trades > 5 and opt_strategy_trades > 0:
        trade_ratio = opt_strategy_trades / orig_strategy_trades
        logger.info(f"  Trade stability: {orig_strategy_trades:.0f} → "
                     f"{opt_strategy_trades:.0f} (ratio={trade_ratio:.2f})")
        if trade_ratio < 0.6:
            logger.info("  → REJECTED: trade count dropped >40% (overfitting risk)")
            return CMAResult(
                original_strategy=strategy,
                optimized_strategy=strategy,
                original_fitness=orig_cpcv,
                optimized_fitness=orig_cpcv,
                improvement_pct=0.0,
                param_specs=param_specs,
                original_params=x0.tolist(),
                optimized_params=x0.tolist(),
                n_evals=total_evals,
                converged=False,
            )

    improvement_pct = (opt_cpcv - orig_cpcv) / max(abs(orig_cpcv), 0.01) * 100
    logger.info(f"  → ACCEPTED: +{improvement_pct:.1f}% CPCV improvement")

    # Build optimized strategy
    optimized = rebuild_strategy(strategy, consensus_params, param_specs)
    optimized.fitness = (opt_cpcv, 0.0)

    for spec, orig_val, new_val in zip(param_specs, x0.tolist(), consensus_params):
        if spec.param_type == 'period':
            orig_disp = str(round(orig_val))
            new_disp = str(round(new_val))
        else:
            orig_disp = f"{orig_val:.2f}"
            new_disp = f"{new_val:.2f}"
        changed = " *" if abs(new_val - orig_val) > 0.5 else ""
        logger.info(f"  {spec.name}: {orig_disp} → {new_disp}{changed}")

    return CMAResult(
        original_strategy=strategy,
        optimized_strategy=optimized,
        original_fitness=orig_cpcv,
        optimized_fitness=opt_cpcv,
        improvement_pct=improvement_pct,
        param_specs=param_specs,
        original_params=x0.tolist(),
        optimized_params=consensus_params,
        n_evals=total_evals,
        converged=True,
    )
