"""
Walk-Forward Validation Engine.

Tests whether the evolution PROCESS consistently produces profitable strategies,
not just whether individual strategies are robust.

Three modes:
  - stability: Backtest existing strategies across all windows (diagnostic)
  - cmaes: Re-optimize parameters per window (medium cost)
  - evolve: Full re-evolution per window (high cost, most rigorous)

Usage:
    from validation.walk_forward import WalkForwardEngine, WFConfig
    engine = WalkForwardEngine(config, wf_config)
    results = engine.run(df, mode='stability', strategies=existing_strategies)
    results = engine.run(df, mode='evolve', seed=42)
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from evolution.fitness import _run_single_window, _combine_equity_curves, BARS_PER_YEAR_15M
from backtest.metrics import calculate_all_metrics
from data.multi_timeframe import prepare_multi_tf_data

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WFWindow:
    """A single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_bars: int = 0
    test_bars: int = 0


@dataclass
class WFStrategyResult:
    """OOS result for a single strategy in one window."""
    strategy_index: int
    expression: str
    direction: str
    cagr: float
    sortino: float
    max_dd: float
    n_trades: int
    win_rate: float
    profit_factor: float
    equity_final: float  # normalized to 100


@dataclass
class WFWindowResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_period: str
    test_period: str
    test_regime: str  # bull/bear/sideways
    test_bh_return: float  # buy & hold return during test
    n_evolved: int  # strategies evolved (0 for stability mode)
    n_validated: int  # strategies that passed validation
    n_tested: int  # strategies backtested on OOS
    # Ensemble OOS (equal-weight average of all tested strategies)
    ensemble_cagr: float = 0.0
    ensemble_sortino: float = 0.0
    ensemble_max_dd: float = 0.0
    ensemble_n_trades: int = 0
    # Best single strategy OOS
    best_cagr: float = 0.0
    best_sortino: float = 0.0
    # Per-strategy details
    strategy_results: List[dict] = field(default_factory=list)
    # Timing
    elapsed_seconds: float = 0.0


@dataclass
class WFConfig:
    """Walk-forward configuration."""
    # Window structure
    mode: str = 'expanding'  # 'expanding' or 'rolling'
    test_months: int = 3
    step_months: int = 3
    min_train_months: int = 18  # minimum train period
    embargo_days: int = 7  # gap between train and test

    # Evolution settings (for 'evolve' mode)
    population: int = 150
    generations: int = 80
    patience: int = 15
    n_top: int = 10  # max strategies to carry to OOS per window

    # CMA-ES settings (for 'cmaes' mode)
    cmaes_sigma: float = 0.05
    cmaes_evals: int = 200

    # Validation (quick validation per window)
    cpcv_groups: int = 6  # fewer groups for faster validation
    pbo_threshold: float = 0.50
    min_trades_oos: int = 5  # minimum trades in OOS window


# ============================================================================
# WINDOW GENERATION
# ============================================================================

def generate_windows(df: pd.DataFrame, wf_config: WFConfig) -> List[WFWindow]:
    """
    Generate walk-forward windows.

    Expanding: train always starts at data start, grows each step.
    Rolling: train has fixed length, slides forward.
    """
    data_start = df.index.min()
    data_end = df.index.max()
    windows = []

    if wf_config.mode == 'expanding':
        # Train always starts at data_start, end advances by step_months
        train_end_base = data_start + pd.DateOffset(months=wf_config.min_train_months)
        window_id = 0

        while True:
            train_end = train_end_base + pd.DateOffset(months=wf_config.step_months * window_id)
            test_start = train_end + pd.DateOffset(days=wf_config.embargo_days)
            test_end = test_start + pd.DateOffset(months=wf_config.test_months)

            if test_end > data_end:
                # Try shorter last window
                test_end = data_end
                if test_start >= test_end:
                    break
                # At least 1 month of test data
                if (test_end - test_start).days < 28:
                    break

            train_df = df[(df.index >= data_start) & (df.index < train_end)]
            test_df = df[(df.index >= test_start) & (df.index <= test_end)]

            if len(train_df) < 1000 or len(test_df) < 200:
                window_id += 1
                continue

            windows.append(WFWindow(
                window_id=window_id,
                train_start=str(data_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_bars=len(train_df),
                test_bars=len(test_df),
            ))
            window_id += 1

    elif wf_config.mode == 'rolling':
        train_length = pd.DateOffset(months=wf_config.min_train_months)
        window_id = 0
        current_start = data_start

        while True:
            train_end = current_start + train_length
            test_start = train_end + pd.DateOffset(days=wf_config.embargo_days)
            test_end = test_start + pd.DateOffset(months=wf_config.test_months)

            if test_end > data_end:
                break

            train_df = df[(df.index >= current_start) & (df.index < train_end)]
            test_df = df[(df.index >= test_start) & (df.index <= test_end)]

            if len(train_df) < 1000 or len(test_df) < 200:
                current_start += pd.DateOffset(months=wf_config.step_months)
                continue

            windows.append(WFWindow(
                window_id=window_id,
                train_start=str(current_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_bars=len(train_df),
                test_bars=len(test_df),
            ))
            window_id += 1
            current_start += pd.DateOffset(months=wf_config.step_months)

    return windows


# ============================================================================
# HELPERS
# ============================================================================

def _classify_regime(df: pd.DataFrame) -> Tuple[str, float]:
    """Classify market regime from OHLCV data. Returns (regime, bh_return)."""
    bh_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    if bh_return > 0.15:
        return 'bull', bh_return
    elif bh_return < -0.15:
        return 'bear', bh_return
    return 'sideways', bh_return


def _backtest_strategy_on_window(strategy, test_df: pd.DataFrame,
                                  config: dict) -> Optional[dict]:
    """Run a single strategy on a test window and return metrics."""
    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)

    try:
        tf_data = prepare_multi_tf_data(test_df)
        equity, trades = _run_single_window(
            strategy, test_df, costs_config, atr_period, tf_data=tf_data
        )
        metrics = calculate_all_metrics(equity, BARS_PER_YEAR_15M)

        n_trades = len(trades)
        winning = sum(1 for t in trades if t['pnl_pct'] > 0)
        win_rate = winning / n_trades if n_trades > 0 else 0

        winning_pnl = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
        losing_pnl = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
        pf = winning_pnl / max(losing_pnl, 1e-10)

        return {
            'expression': strategy.expression_raw,
            'direction': strategy.direction,
            'cagr': float(metrics.get('cagr', 0)),
            'sortino': float(metrics.get('sortino', 0)),
            'max_dd': float(metrics.get('max_dd', 0)),
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': pf,
            'equity_final': float(equity.iloc[-1]),
            'equity_curve': equity,  # for ensemble computation
        }
    except Exception as e:
        logger.debug(f"Backtest failed: {e}")
        return None


def _compute_ensemble(strategy_results: List[dict],
                      regime: Optional[str] = None) -> dict:
    """
    Compute equal-weight ensemble from individual strategy results.

    If regime is provided, filters strategies by direction:
      - bull: LONG only
      - bear: SHORT only
      - sideways: all strategies
    """
    if not strategy_results:
        return {'cagr': 0, 'sortino': 0, 'max_dd': 0, 'n_trades': 0,
                'n_active': 0}

    # Regime-aware filtering
    if regime == 'bull':
        filtered = [r for r in strategy_results if r.get('direction') == 'LONG']
    elif regime == 'bear':
        filtered = [r for r in strategy_results if r.get('direction') == 'SHORT']
    else:
        filtered = strategy_results

    if not filtered:
        filtered = strategy_results  # fallback to all if no match

    # Equal-weight portfolio: average of normalized equity curves
    curves = []
    for r in filtered:
        eq = r.get('equity_curve')
        if eq is not None and len(eq) > 0:
            curves.append(eq)

    if not curves:
        return {'cagr': 0, 'sortino': 0, 'max_dd': 0, 'n_trades': 0,
                'n_active': 0}

    # Align curves by index, compute average return
    all_returns = []
    for curve in curves:
        rets = curve.pct_change().fillna(0)
        all_returns.append(rets)

    # Average returns (equal weight)
    aligned = pd.concat(all_returns, axis=1).fillna(0)
    ensemble_returns = aligned.mean(axis=1)

    # Build ensemble equity
    ensemble_equity = (1 + ensemble_returns).cumprod() * 100
    ensemble_equity = pd.Series(ensemble_equity.values, index=aligned.index)

    metrics = calculate_all_metrics(ensemble_equity, BARS_PER_YEAR_15M)
    total_trades = sum(r['n_trades'] for r in filtered)

    return {
        'cagr': float(metrics.get('cagr', 0)),
        'sortino': float(metrics.get('sortino', 0)),
        'max_dd': float(metrics.get('max_dd', 0)),
        'n_trades': total_trades,
        'n_active': len(filtered),
    }


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

class WalkForwardEngine:
    """
    Walk-forward analysis engine.

    Tests the evolution process by re-evolving (or re-optimizing) strategies
    on successive train windows and evaluating on held-out test windows.
    """

    def __init__(self, config: dict, wf_config: WFConfig):
        self.config = config
        self.wf_config = wf_config

    def run_stability(self, df: pd.DataFrame,
                      strategies: list) -> List[WFWindowResult]:
        """
        Mode 1: Strategy Stability Analysis.

        Takes existing strategies (already evolved), backtests each on every
        test window. Shows how strategies perform across time — are they
        all-weather or regime-dependent?

        This is diagnostic: tells us IF we need walk-forward re-evolution.
        """
        windows = generate_windows(df, self.wf_config)
        logger.info(f"Walk-forward stability: {len(windows)} windows, "
                    f"{len(strategies)} strategies")

        results = []
        for w in windows:
            t0 = time.time()
            test_df = df[(df.index >= pd.Timestamp(w.test_start)) &
                         (df.index <= pd.Timestamp(w.test_end))]

            if len(test_df) < 100:
                continue

            regime, bh_return = _classify_regime(test_df)

            strat_results = []
            for i, strategy in enumerate(strategies):
                r = _backtest_strategy_on_window(strategy, test_df, self.config)
                if r:
                    r['strategy_index'] = i
                    strat_results.append(r)

            # Raw ensemble (all strategies)
            ensemble_raw = _compute_ensemble(strat_results)
            # Regime-filtered ensemble (LONG in bull, SHORT in bear, all in sideways)
            ensemble_filtered = _compute_ensemble(strat_results, regime=regime)

            # Best single
            best_cagr = max((r['cagr'] for r in strat_results), default=0)
            best_sortino = max((r['sortino'] for r in strat_results), default=0)

            # Clean equity curves from results (not serializable)
            clean_results = []
            for r in strat_results:
                rc = {k: v for k, v in r.items() if k != 'equity_curve'}
                clean_results.append(rc)

            # Use regime-filtered ensemble as primary metric
            results.append(WFWindowResult(
                window_id=w.window_id,
                train_period=f"{w.train_start} to {w.train_end}",
                test_period=f"{w.test_start} to {w.test_end}",
                test_regime=regime,
                test_bh_return=bh_return,
                n_evolved=0,
                n_validated=len(strategies),
                n_tested=len(strat_results),
                ensemble_cagr=ensemble_filtered['cagr'],
                ensemble_sortino=ensemble_filtered['sortino'],
                ensemble_max_dd=ensemble_filtered['max_dd'],
                ensemble_n_trades=ensemble_filtered['n_trades'],
                best_cagr=best_cagr,
                best_sortino=best_sortino,
                strategy_results=clean_results,
                elapsed_seconds=time.time() - t0,
            ))

            logger.info(
                f"  W{w.window_id} [{w.test_start} → {w.test_end}] "
                f"{regime:>8s} B&H={bh_return:+.1%} | "
                f"filtered CAGR={ensemble_filtered['cagr']:+.2%} "
                f"({ensemble_filtered['n_active']} strats) | "
                f"raw CAGR={ensemble_raw['cagr']:+.2%} | "
                f"best={best_cagr:+.2%}"
            )

        return results

    def run_evolve(self, df: pd.DataFrame, seed: int = 42) -> List[WFWindowResult]:
        """
        Mode 2: Full Walk-Forward Re-Evolution.

        For each window:
        1. Evolve on train data (IslandModel, reduced config)
        2. Quick validate (CPCV with fewer splits)
        3. Select top K validated strategies
        4. Backtest on test window
        5. Report ensemble OOS performance
        """
        from evolution.island import IslandModel
        from validation.cpcv import cpcv_evaluate
        from validation.pbo import calculate_pbo

        windows = generate_windows(df, self.wf_config)
        logger.info(f"Walk-forward evolve: {len(windows)} windows, seed={seed}")

        wfc = self.wf_config
        results = []

        for w in windows:
            t0 = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"WINDOW {w.window_id}: train {w.train_start}→{w.train_end}, "
                        f"test {w.test_start}→{w.test_end}")
            logger.info(f"{'='*60}")

            train_df = df[(df.index >= pd.Timestamp(w.train_start)) &
                          (df.index < pd.Timestamp(w.train_end))]
            test_df = df[(df.index >= pd.Timestamp(w.test_start)) &
                         (df.index <= pd.Timestamp(w.test_end))]

            if len(train_df) < 1000 or len(test_df) < 100:
                logger.warning(f"  Skipping: insufficient data "
                              f"(train={len(train_df)}, test={len(test_df)})")
                continue

            regime, bh_return = _classify_regime(test_df)

            # --- 1. Evolve on train data ---
            # Use unique seed per window for independence
            window_seed = seed + w.window_id * 1000
            random.seed(window_seed)
            np.random.seed(window_seed)

            model = IslandModel(self.config, train_df)
            model.initialize(total_pop_size=wfc.population)
            evo_result = model.run(
                n_generations=wfc.generations,
                patience=wfc.patience
            )

            evolved = evo_result['best_strategies']
            n_evolved = len(evolved)
            logger.info(f"  Evolved {n_evolved} strategies")

            # --- 2. Quick validate on train data (CPCV with fewer groups) ---
            validated = []
            for strategy in evolved:
                try:
                    cpcv = cpcv_evaluate(
                        strategy, train_df, self.config,
                        n_groups=wfc.cpcv_groups,
                        purge_bars=96, embargo_bars=48,
                        max_splits=50,  # reduced for speed
                    )
                    pbo = calculate_pbo(cpcv)

                    if pbo['pbo'] < wfc.pbo_threshold and cpcv['mean_sortino'] > 0:
                        validated.append((strategy, cpcv['mean_sortino']))
                except Exception as e:
                    logger.debug(f"  Validation failed: {e}")

            # Sort by CPCV Sortino, take top N
            validated.sort(key=lambda x: x[1], reverse=True)
            selected = [s for s, _ in validated[:wfc.n_top]]
            n_validated = len(validated)

            logger.info(f"  Validated: {n_validated}/{n_evolved}, "
                        f"selected top {len(selected)}")

            if not selected:
                logger.warning(f"  No strategies passed validation. "
                              f"Recording zero for this window.")
                results.append(WFWindowResult(
                    window_id=w.window_id,
                    train_period=f"{w.train_start} to {w.train_end}",
                    test_period=f"{w.test_start} to {w.test_end}",
                    test_regime=regime,
                    test_bh_return=bh_return,
                    n_evolved=n_evolved,
                    n_validated=0,
                    n_tested=0,
                    elapsed_seconds=time.time() - t0,
                ))
                continue

            # --- 3. Backtest selected strategies on test window ---
            strat_results = []
            for i, strategy in enumerate(selected):
                r = _backtest_strategy_on_window(strategy, test_df, self.config)
                if r and r['n_trades'] >= wfc.min_trades_oos:
                    r['strategy_index'] = i
                    strat_results.append(r)

            # Ensemble
            ensemble = _compute_ensemble(strat_results)
            best_cagr = max((r['cagr'] for r in strat_results), default=0)
            best_sortino = max((r['sortino'] for r in strat_results), default=0)

            clean_results = [{k: v for k, v in r.items() if k != 'equity_curve'}
                             for r in strat_results]

            elapsed = time.time() - t0
            results.append(WFWindowResult(
                window_id=w.window_id,
                train_period=f"{w.train_start} to {w.train_end}",
                test_period=f"{w.test_start} to {w.test_end}",
                test_regime=regime,
                test_bh_return=bh_return,
                n_evolved=n_evolved,
                n_validated=n_validated,
                n_tested=len(strat_results),
                ensemble_cagr=ensemble['cagr'],
                ensemble_sortino=ensemble['sortino'],
                ensemble_max_dd=ensemble['max_dd'],
                ensemble_n_trades=ensemble['n_trades'],
                best_cagr=best_cagr,
                best_sortino=best_sortino,
                strategy_results=clean_results,
                elapsed_seconds=elapsed,
            ))

            logger.info(
                f"  OOS: ensemble CAGR={ensemble['cagr']:+.2%} "
                f"Sortino={ensemble['sortino']:.2f} "
                f"MaxDD={ensemble['max_dd']:.1%} | "
                f"B&H={bh_return:+.1%} ({regime}) | "
                f"{elapsed:.0f}s"
            )

        return results

    def run_cmaes(self, df: pd.DataFrame,
                  strategies: list, seed: int = 42) -> List[WFWindowResult]:
        """
        Mode 3: CMA-ES Walk-Forward Re-Optimization.

        Takes existing GE structures, re-optimizes parameters on each train
        window, tests on test window.

        Tests whether structures are stable but parameters need updating.
        """
        from evolution.cmaes import optimize_strategy
        from evolution.param_extractor import extract_params

        windows = generate_windows(df, self.wf_config)
        logger.info(f"Walk-forward CMA-ES: {len(windows)} windows, "
                    f"{len(strategies)} base structures")

        wfc = self.wf_config
        results = []

        for w in windows:
            t0 = time.time()
            logger.info(f"\nW{w.window_id}: train {w.train_start}→{w.train_end}, "
                        f"test {w.test_start}→{w.test_end}")

            train_df = df[(df.index >= pd.Timestamp(w.train_start)) &
                          (df.index < pd.Timestamp(w.train_end))]
            test_df = df[(df.index >= pd.Timestamp(w.test_start)) &
                         (df.index <= pd.Timestamp(w.test_end))]

            if len(train_df) < 1000 or len(test_df) < 100:
                continue

            regime, bh_return = _classify_regime(test_df)

            # Re-optimize each strategy on train window
            window_seed = seed + w.window_id * 1000
            optimized = []

            for i, strategy in enumerate(strategies):
                params = extract_params(strategy)
                if not params:
                    optimized.append(strategy)
                    continue

                try:
                    cma_result = optimize_strategy(
                        strategy, train_df, self.config,
                        max_evals=wfc.cmaes_evals,
                        sigma0=wfc.cmaes_sigma,
                        seed=window_seed + i,
                    )
                    if cma_result.converged:
                        optimized.append(cma_result.optimized_strategy)
                        logger.debug(f"  Strategy {i}: CMA-ES improved "
                                     f"{cma_result.improvement_pct:+.1f}%")
                    else:
                        optimized.append(strategy)
                except Exception as e:
                    logger.debug(f"  CMA-ES failed for strategy {i}: {e}")
                    optimized.append(strategy)

            # Backtest optimized strategies on test window
            strat_results = []
            for i, strategy in enumerate(optimized):
                r = _backtest_strategy_on_window(strategy, test_df, self.config)
                if r and r['n_trades'] >= wfc.min_trades_oos:
                    r['strategy_index'] = i
                    strat_results.append(r)

            ensemble = _compute_ensemble(strat_results)
            best_cagr = max((r['cagr'] for r in strat_results), default=0)
            best_sortino = max((r['sortino'] for r in strat_results), default=0)

            clean_results = [{k: v for k, v in r.items() if k != 'equity_curve'}
                             for r in strat_results]

            elapsed = time.time() - t0
            results.append(WFWindowResult(
                window_id=w.window_id,
                train_period=f"{w.train_start} to {w.train_end}",
                test_period=f"{w.test_start} to {w.test_end}",
                test_regime=regime,
                test_bh_return=bh_return,
                n_evolved=0,
                n_validated=len(strategies),
                n_tested=len(strat_results),
                ensemble_cagr=ensemble['cagr'],
                ensemble_sortino=ensemble['sortino'],
                ensemble_max_dd=ensemble['max_dd'],
                ensemble_n_trades=ensemble['n_trades'],
                best_cagr=best_cagr,
                best_sortino=best_sortino,
                strategy_results=clean_results,
                elapsed_seconds=elapsed,
            ))

            logger.info(
                f"  OOS: ensemble CAGR={ensemble['cagr']:+.2%} "
                f"Sortino={ensemble['sortino']:.2f} | "
                f"B&H={bh_return:+.1%} ({regime}) | "
                f"{len(strat_results)} strats | {elapsed:.0f}s"
            )

        return results


# ============================================================================
# AGGREGATE RESULTS
# ============================================================================

def aggregate_wf_results(results: List[WFWindowResult]) -> dict:
    """
    Aggregate walk-forward results across all windows.

    Produces summary statistics: consistency, total return, decay analysis.
    """
    if not results:
        return {'error': 'no results'}

    n_windows = len(results)
    n_positive = sum(1 for r in results if r.ensemble_cagr > 0)
    n_beat_bh = sum(1 for r in results if r.ensemble_cagr > r.test_bh_return)

    # Total CAGR: compound across windows
    # Each window's CAGR is annualized, but test period is ~3 months
    # Convert to 3-month return, compound, then re-annualize
    total_return = 1.0
    for r in results:
        if r.ensemble_cagr != 0:
            # Approximate 3-month return from annualized CAGR
            test_months = 3.0  # approximate
            period_return = (1 + r.ensemble_cagr) ** (test_months / 12) - 1
            total_return *= (1 + period_return)

    total_months = sum(3 for r in results)  # approximate total test months
    total_cagr = total_return ** (12 / max(total_months, 1)) - 1

    # B&H total
    bh_total = 1.0
    for r in results:
        bh_period = (1 + r.test_bh_return)
        bh_total *= bh_period
    bh_cagr = bh_total ** (12 / max(total_months, 1)) - 1

    # By regime
    regime_results = {}
    for r in results:
        regime = r.test_regime
        if regime not in regime_results:
            regime_results[regime] = []
        regime_results[regime].append(r.ensemble_cagr)

    regime_summary = {}
    for regime, cagrs in regime_results.items():
        regime_summary[regime] = {
            'n_windows': len(cagrs),
            'mean_cagr': float(np.mean(cagrs)),
            'pct_positive': sum(1 for c in cagrs if c > 0) / len(cagrs),
        }

    # Decay analysis: does performance decrease for later windows?
    if len(results) >= 4:
        first_half = results[:len(results)//2]
        second_half = results[len(results)//2:]
        first_mean = np.mean([r.ensemble_cagr for r in first_half])
        second_mean = np.mean([r.ensemble_cagr for r in second_half])
        decay_detected = second_mean < first_mean * 0.5
    else:
        first_mean = second_mean = 0
        decay_detected = False

    total_trades = sum(r.ensemble_n_trades for r in results)
    total_time = sum(r.elapsed_seconds for r in results)

    return {
        'n_windows': n_windows,
        'n_positive': n_positive,
        'consistency': n_positive / n_windows,
        'n_beat_bh': n_beat_bh,
        'alpha_rate': n_beat_bh / n_windows,
        'total_cagr': total_cagr,
        'total_compounded_return': total_return - 1,
        'bh_cagr': bh_cagr,
        'bh_compounded_return': bh_total - 1,
        'excess_return': total_cagr - bh_cagr,
        'total_trades': total_trades,
        'regime_summary': regime_summary,
        'decay_first_half_cagr': float(first_mean),
        'decay_second_half_cagr': float(second_mean),
        'decay_detected': decay_detected,
        'total_elapsed_seconds': total_time,
        'per_window': [asdict(r) for r in results],
    }


def print_wf_summary(agg: dict):
    """Pretty-print walk-forward results."""
    print(f"\n{'='*80}")
    print("WALK-FORWARD RESULTS")
    print(f"{'='*80}")

    print(f"\nWindows: {agg['n_windows']}")
    print(f"Positive: {agg['n_positive']}/{agg['n_windows']} "
          f"({agg['consistency']:.0%} consistency)")
    print(f"Beat B&H: {agg['n_beat_bh']}/{agg['n_windows']} "
          f"({agg['alpha_rate']:.0%} alpha rate)")

    print(f"\nCompounded return: {agg['total_compounded_return']:+.1%} "
          f"(CAGR: {agg['total_cagr']:+.1%})")
    print(f"B&H return:        {agg['bh_compounded_return']:+.1%} "
          f"(CAGR: {agg['bh_cagr']:+.1%})")
    print(f"Excess return:     {agg['excess_return']:+.1%}")
    print(f"Total trades:      {agg['total_trades']}")

    print(f"\nBy regime:")
    for regime, info in agg.get('regime_summary', {}).items():
        print(f"  {regime:>10s}: {info['n_windows']} windows, "
              f"mean CAGR={info['mean_cagr']:+.1%}, "
              f"{info['pct_positive']:.0%} positive")

    if agg.get('decay_detected'):
        print(f"\n⚠ DECAY DETECTED: first half CAGR={agg['decay_first_half_cagr']:+.1%}, "
              f"second half={agg['decay_second_half_cagr']:+.1%}")
    else:
        print(f"\nDecay: first half={agg['decay_first_half_cagr']:+.1%}, "
              f"second half={agg['decay_second_half_cagr']:+.1%} (OK)")

    # Per-window detail
    print(f"\n{'W':>3} {'Period':>26} {'Regime':>9} {'B&H':>8} "
          f"{'Ens.CAGR':>9} {'Sortino':>8} {'Trades':>7} {'Alpha':>6}")
    print(f"{'-'*80}")
    for pw in agg.get('per_window', []):
        alpha = 'YES' if pw['ensemble_cagr'] > pw['test_bh_return'] else 'no'
        print(f"W{pw['window_id']:>2} {pw['test_period']:>26} "
              f"{pw['test_regime']:>9} {pw['test_bh_return']:>+7.1%} "
              f"{pw['ensemble_cagr']:>+8.1%} {pw['ensemble_sortino']:>8.2f} "
              f"{pw['ensemble_n_trades']:>7} {alpha:>6}")

    print(f"\nTotal time: {agg['total_elapsed_seconds']:.0f}s")
    print(f"{'='*80}")
