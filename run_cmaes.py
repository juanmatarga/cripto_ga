#!/usr/bin/env python3
"""
CMA-ES parameter optimization for evolved strategies.

Takes an experiment's top strategies, optimizes their parameters with CMA-ES,
then runs the FULL validation pipeline (CPCV + PBO + permutation) on both
original and optimized versions for fair comparison.

Usage:
    python3 run_cmaes.py --results results/experiment_seed123_* [--config config_v2.yaml]
    python3 run_cmaes.py --results results/experiment_BNB_USDT_seed777_* --max-evals 500
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from grammar.mapper import decode
from strategy.phenotype import Strategy
from evolution.param_extractor import extract_params, rebuild_strategy
from evolution.cmaes import optimize_strategy, CMAResult
from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
from validation.cpcv import cpcv_evaluate
from validation.pbo import calculate_pbo
from validation.signal_permutation import signal_permutation_test
from backtest.metrics import calculate_all_metrics
from scipy import stats as scipy_stats

logger = logging.getLogger('cripto_ga')

OTS_START = '2025-06-01'


def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'cmaes_{ts}.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_data_for_asset(config: dict) -> tuple:
    """Load evolution + OTS data for a given config."""
    from data.loader import load_data
    df = load_data(config)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ots_start = config.get('data', {}).get('ots_start', OTS_START)
    df_evo = df[df.index < pd.Timestamp(ots_start)]
    df_ots = df[df.index >= pd.Timestamp(ots_start)]
    return df_evo, df_ots


def validate_strategy(strategy: Strategy, data: pd.DataFrame,
                      config: dict) -> dict:
    """Run full validation (CPCV + PBO + permutation + t-test)."""
    val_cfg = config.get('validation', {})

    cpcv = cpcv_evaluate(
        strategy, data, config,
        n_groups=val_cfg.get('cpcv_groups', 10),
        purge_bars=val_cfg.get('cpcv_purge_bars', 96),
        embargo_bars=val_cfg.get('cpcv_embargo_bars', 48),
        max_splits=val_cfg.get('cpcv_max_splits', 252),
    )
    pbo = calculate_pbo(cpcv)

    oos_sortinos = cpcv.get('oos_sortinos', [])
    if len(oos_sortinos) > 1:
        t_stat, t_pval_2s = scipy_stats.ttest_1samp(oos_sortinos, 0)
        t_pval = t_pval_2s / 2 if t_stat > 0 else 1.0 - t_pval_2s / 2
    else:
        t_stat, t_pval = 0.0, 1.0

    perm = signal_permutation_test(
        strategy, data, config,
        n_permutations=val_cfg.get('permutation_n', 1000),
    )

    passes_pbo = pbo['pbo'] < val_cfg.get('pbo_threshold', 0.50)
    passes_perm = perm['p_value'] < val_cfg.get('permutation_alpha', 0.05)
    passes_ttest = t_pval < val_cfg.get('permutation_alpha', 0.05)

    return {
        'cpcv_mean_sortino': cpcv['mean_sortino'],
        'cpcv_std_sortino': cpcv.get('std_sortino', 0),
        'cpcv_pct_positive': cpcv['pct_positive_sortino'],
        'cpcv_n_splits': cpcv['n_splits'],
        'cpcv_mean_trades': float(np.mean(cpcv.get('oos_trades', [0]))),
        'pbo': pbo['pbo'],
        'pbo_interpretation': pbo['interpretation'],
        'ttest_pval': float(t_pval),
        'perm_p_value': perm['p_value'],
        'passes_all': passes_pbo and passes_perm and passes_ttest,
    }


def run_ots(strategy: Strategy, df_ots: pd.DataFrame, config: dict) -> dict:
    """Run OTS backtest on a strategy."""
    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    try:
        equity, trades = _run_single_window(strategy, df_ots, costs_config, atr_period)
        metrics = calculate_all_metrics(equity, BARS_PER_YEAR_15M)
        return {
            'n_trades': len(trades),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating))
                       else v for k, v in metrics.items()},
        }
    except Exception as e:
        logger.error(f"OTS failed: {e}")
        return {'n_trades': 0, 'metrics': {}}


def main():
    parser = argparse.ArgumentParser(description='CMA-ES Parameter Optimization')
    parser.add_argument('--results', required=True, help='Experiment results directory')
    parser.add_argument('--config', default='config_v2.yaml', help='Config file')
    parser.add_argument('--max-evals', type=int, default=400,
                        help='Max CMA-ES evaluations per strategy')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for CMA-ES')
    parser.add_argument('--top-n', type=int, default=None,
                        help='Only optimize top N strategies (by OTS CAGR)')
    parser.add_argument('--skip-ots', action='store_true',
                        help='Skip OTS evaluation (for testing)')
    args = parser.parse_args()

    setup_logging()

    results_dir = Path(args.results)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Check if this is a multi-asset experiment
    meta_path = results_dir / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        symbol = meta.get('symbol', 'BTC/USDT')
        # Override config for this asset
        if symbol != config.get('data', {}).get('symbol', 'BTC/USDT'):
            from data.multi_asset import make_asset_config
            config = make_asset_config(symbol, config)
            logger.info(f"Using asset config for {symbol}")

    # Load data
    logger.info("Loading data...")
    df_evo, df_ots = load_data_for_asset(config)
    logger.info(f"Evolution: {len(df_evo)} bars, OTS: {len(df_ots)} bars")

    # Load strategies
    with open(results_dir / 'top_strategies.json') as f:
        strategies_data = json.load(f)

    # Load OTS results to select which strategies to optimize
    ots_path = results_dir / 'ots_results.json'
    if ots_path.exists():
        with open(ots_path) as f:
            ots_results = json.load(f)
        # Only optimize strategies with positive OTS CAGR
        positive_indices = set()
        for r in ots_results:
            if r.get('metrics', {}).get('cagr', 0) > 0:
                positive_indices.add(r['strategy_index'])
        if positive_indices:
            logger.info(f"Found {len(positive_indices)} OTS-positive strategies to optimize")
        else:
            # If no OTS-positive, optimize all validated
            val_path = results_dir / 'validation.json'
            if val_path.exists():
                with open(val_path) as f:
                    val_data = json.load(f)
                positive_indices = {v['strategy_index'] for v in val_data
                                    if v.get('passes_all')}
                logger.info(f"No OTS-positive strategies. "
                            f"Optimizing {len(positive_indices)} validated strategies")
    else:
        positive_indices = set(range(len(strategies_data)))

    if args.top_n:
        # Sort by OTS CAGR and take top N
        ots_by_idx = {r['strategy_index']: r.get('metrics', {}).get('cagr', 0)
                      for r in ots_results}
        sorted_indices = sorted(positive_indices,
                                key=lambda i: ots_by_idx.get(i, 0), reverse=True)
        positive_indices = set(sorted_indices[:args.top_n])

    if not positive_indices:
        logger.error("No strategies to optimize")
        return

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Process each strategy
    all_results = []
    t0 = time.time()

    for idx in sorted(positive_indices):
        sd = strategies_data[idx]
        strategy = decode(sd['genome'])
        if strategy is None:
            logger.warning(f"Strategy {idx}: failed to decode")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy {idx}: {strategy.expression_raw[:70]}")
        logger.info(f"{'='*60}")

        # Run CMA-ES
        t1 = time.time()
        cma_result = optimize_strategy(
            strategy, df_evo, config,
            max_evals=args.max_evals,
            seed=args.seed + idx,  # Different seed per strategy
        )
        cma_time = time.time() - t1

        logger.info(f"CMA-ES done in {cma_time:.0f}s, {cma_result.n_evals} evals, "
                     f"improvement: {cma_result.improvement_pct:+.1f}%")
        logger.info(f"Optimized: {cma_result.optimized_strategy.expression_raw[:70]}")

        # Validate BOTH original and optimized
        logger.info("\nValidating original...")
        val_orig = validate_strategy(strategy, df_evo, config)
        logger.info(f"  Original: PBO={val_orig['pbo']:.3f} "
                     f"perm_p={val_orig['perm_p_value']:.4f} "
                     f"CPCV_sortino={val_orig['cpcv_mean_sortino']:.3f} "
                     f"passes={val_orig['passes_all']}")

        logger.info("Validating optimized...")
        val_opt = validate_strategy(cma_result.optimized_strategy, df_evo, config)
        logger.info(f"  Optimized: PBO={val_opt['pbo']:.3f} "
                     f"perm_p={val_opt['perm_p_value']:.4f} "
                     f"CPCV_sortino={val_opt['cpcv_mean_sortino']:.3f} "
                     f"passes={val_opt['passes_all']}")

        # POST-VALIDATION SAFETY: revert if CMA-ES degraded validation
        # Only accept CMA-ES changes if validation doesn't degrade significantly
        use_optimized = cma_result.converged  # False if do-no-harm rejected
        if use_optimized:
            pbo_degraded = val_opt['pbo'] > val_orig['pbo'] + 0.15
            cpcv_degraded = val_opt['cpcv_mean_sortino'] < val_orig['cpcv_mean_sortino'] - 0.1
            orig_passed_opt_failed = val_orig['passes_all'] and not val_opt['passes_all']

            if pbo_degraded or cpcv_degraded or orig_passed_opt_failed:
                logger.warning(f"  POST-VALIDATION REVERT: Optimized degraded validation "
                               f"(PBO: {val_orig['pbo']:.3f}→{val_opt['pbo']:.3f}, "
                               f"CPCV: {val_orig['cpcv_mean_sortino']:.3f}→{val_opt['cpcv_mean_sortino']:.3f})")
                use_optimized = False
                cma_result = cma_result.__class__(
                    original_strategy=strategy,
                    optimized_strategy=strategy,
                    original_fitness=cma_result.original_fitness,
                    optimized_fitness=cma_result.original_fitness,
                    improvement_pct=0.0,
                    param_specs=cma_result.param_specs,
                    original_params=cma_result.original_params,
                    optimized_params=cma_result.original_params,
                    n_evals=cma_result.n_evals,
                    converged=False,
                )
                val_opt = val_orig

        # OTS evaluation
        ots_orig = {}
        ots_opt = {}
        if not args.skip_ots and len(df_ots) > 0:
            logger.info("OTS evaluation...")
            ots_orig = run_ots(strategy, df_ots, config)
            ots_opt = run_ots(cma_result.optimized_strategy, df_ots, config)

            orig_cagr = ots_orig.get('metrics', {}).get('cagr', 0)
            opt_cagr = ots_opt.get('metrics', {}).get('cagr', 0)
            orig_sortino = ots_orig.get('metrics', {}).get('sortino', 0)
            opt_sortino = ots_opt.get('metrics', {}).get('sortino', 0)

            logger.info(f"  OTS Original:  CAGR={orig_cagr:+.4f} Sortino={orig_sortino:.3f} "
                         f"trades={ots_orig.get('n_trades', 0)}")
            logger.info(f"  OTS Optimized: CAGR={opt_cagr:+.4f} Sortino={opt_sortino:.3f} "
                         f"trades={ots_opt.get('n_trades', 0)}")

        result_entry = {
            'strategy_index': idx,
            'original_expression': strategy.expression_raw,
            'optimized_expression': cma_result.optimized_strategy.expression_raw,
            'n_params': len(cma_result.param_specs),
            'n_evals': cma_result.n_evals,
            'cma_improvement_pct': cma_result.improvement_pct,
            'param_changes': [
                {
                    'name': spec.name,
                    'original': orig,
                    'optimized': opt,
                    'type': spec.param_type,
                }
                for spec, orig, opt in zip(
                    cma_result.param_specs,
                    cma_result.original_params,
                    cma_result.optimized_params,
                )
            ],
            'validation_original': val_orig,
            'validation_optimized': val_opt,
            'ots_original': ots_orig,
            'ots_optimized': ots_opt,
            # Optimized strategy data for live use
            'optimized_strategy': {
                'genome': cma_result.optimized_strategy.genome,
                'direction': cma_result.optimized_strategy.direction,
                'conditions': [str(c) for c in cma_result.optimized_strategy.conditions],
                'logic': cma_result.optimized_strategy.logic,
                'tp_atr_mult': cma_result.optimized_strategy.tp_atr_mult,
                'sl_atr_mult': cma_result.optimized_strategy.sl_atr_mult,
                'trail_atr_mult': cma_result.optimized_strategy.trail_atr_mult,
                'expression_raw': cma_result.optimized_strategy.expression_raw,
                'n_nodes': cma_result.optimized_strategy.n_nodes,
            },
        }
        all_results.append(result_entry)

    elapsed = time.time() - t0

    # Save results
    output_path = results_dir / 'cmaes_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*80}")
    print("CMA-ES OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Idx':>4} {'CMA%':>7} {'PBO_orig':>9} {'PBO_opt':>8} "
          f"{'CPCV_orig':>10} {'CPCV_opt':>9} "
          f"{'OTS_orig':>9} {'OTS_opt':>8} {'Status':>8}")
    print(f"{'-'*80}")

    n_improved_val = 0
    n_improved_ots = 0

    for r in all_results:
        idx = r['strategy_index']
        cma_pct = r['cma_improvement_pct']
        pbo_o = r['validation_original']['pbo']
        pbo_n = r['validation_optimized']['pbo']
        cpcv_o = r['validation_original']['cpcv_mean_sortino']
        cpcv_n = r['validation_optimized']['cpcv_mean_sortino']
        ots_o = r.get('ots_original', {}).get('metrics', {}).get('cagr', None)
        ots_n = r.get('ots_optimized', {}).get('metrics', {}).get('cagr', None)

        ots_o_str = f"{ots_o:+.4f}" if ots_o is not None else "N/A"
        ots_n_str = f"{ots_n:+.4f}" if ots_n is not None else "N/A"

        # Determine status
        if ots_n is not None and ots_o is not None:
            if ots_n > ots_o:
                status = "BETTER"
                n_improved_ots += 1
            elif ots_n < ots_o:
                status = "WORSE"
            else:
                status = "SAME"
        elif cpcv_n > cpcv_o:
            status = "val+"
            n_improved_val += 1
        else:
            status = "val-"

        print(f"{idx:>4} {cma_pct:>+6.1f}% {pbo_o:>9.3f} {pbo_n:>8.3f} "
              f"{cpcv_o:>10.3f} {cpcv_n:>9.3f} "
              f"{ots_o_str:>9} {ots_n_str:>8} {status:>8}")

    print(f"\n{'-'*80}")
    print(f"Total: {len(all_results)} strategies optimized in {elapsed:.0f}s")
    if all_results and all_results[0].get('ots_optimized', {}).get('metrics'):
        print(f"OTS improved: {n_improved_ots}/{len(all_results)}")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
