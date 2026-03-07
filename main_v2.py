"""
CriptoGA v2 — CLI Entry Point.

Commands:
    python main_v2.py evolve   [--config FILE] [--seed N]
    python main_v2.py validate [--config FILE] [--results DIR]
    python main_v2.py ots      [--config FILE] [--validated DIR]
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger('cripto_ga')

# OTS enforcement
OTS_START = '2025-06-01'


def set_global_seed(seed: int):
    """Fix all seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, verbose: bool = True):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_path / f'experiment_{timestamp}.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_evolution_data(config: dict) -> pd.DataFrame:
    """
    Load OHLCV data for evolution period only (excludes OTS).
    """
    from data.loader import load_data

    ots_start = config.get('data', {}).get('ots_start', OTS_START)
    df = load_data(config)

    # Strip timezone for consistency if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Enforce OTS boundary
    df = df[df.index < pd.Timestamp(ots_start)]
    assert len(df) > 0, "No data before OTS boundary"
    logger.info(f"Evolution data: {len(df)} bars (up to {ots_start})")
    return df


def load_ots_data(config: dict) -> pd.DataFrame:
    """Load OTS holdout data only."""
    from data.loader import load_data

    ots_start = config.get('data', {}).get('ots_start', OTS_START)
    df = load_data(config)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[df.index >= pd.Timestamp(ots_start)]
    assert len(df) > 0, "No OTS data found"
    logger.info(f"OTS data: {len(df)} bars (from {ots_start})")
    return df


# ============================================================================
# EVOLVE COMMAND
# ============================================================================

def cmd_evolve(args):
    """Run evolution and save best strategies."""
    config = load_config(args.config)
    seed = args.seed or config.get('evolution', {}).get('seed', 42)
    set_global_seed(seed)

    log_dir = config.get('output', {}).get('logs_dir', './logs')
    setup_logging(log_dir)
    logger.info(f"CriptoGA v2 — evolve (seed={seed})")

    data = load_evolution_data(config)
    logger.info(f"Evolution data: {len(data)} bars, "
                f"{data.index.min()} to {data.index.max()}")

    # Use island model
    from evolution.island import IslandModel

    model = IslandModel(config, data)
    total_pop = config.get('evolution', {}).get('population', 200)
    n_gens = config.get('evolution', {}).get('generations_max', 100)
    patience = config.get('evolution', {}).get('patience', 20)

    model.initialize(total_pop_size=total_pop)
    result = model.run(n_generations=n_gens, patience=patience)

    # Save results
    output_dir = Path(config.get('output', {}).get('results_dir', './results'))
    exp_dir = output_dir / f'experiment_seed{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config used
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Save top strategies
    strategies_data = []
    for s in result['best_strategies']:
        strategies_data.append(s.to_dict())
    with open(exp_dir / 'top_strategies.json', 'w') as f:
        json.dump(strategies_data, f, indent=2, default=str)

    # Save evolution log
    history_data = []
    for gen_stats_list in result['history']:
        gen_entry = {}
        for stats in gen_stats_list:
            gen_entry[f'island_{stats.island_id}'] = {
                'selection': stats.selection_type,
                'best_fitness': stats.best_fitness,
                'mean_fitness': stats.mean_fitness,
                'valid_count': stats.valid_count,
            }
        history_data.append(gen_entry)
    with open(exp_dir / 'evolution_log.json', 'w') as f:
        json.dump(history_data, f, indent=2)

    # Save archive summary
    archive_summary = result['archive'].summary()
    with open(exp_dir / 'archive.json', 'w') as f:
        json.dump(archive_summary, f, indent=2)

    # Save metadata for validation (unique phenotypes for DSR)
    metadata = {
        'total_evaluations': result['total_evaluations'],
        'unique_phenotypes': result.get('unique_phenotypes', result['total_evaluations']),
        'final_generation': result['final_generation'],
    }
    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nResults saved to {exp_dir}")
    logger.info(f"  Top strategies: {len(result['best_strategies'])}")
    logger.info(f"  Archive: {result['archive'].n_occupied}/45 cells")
    logger.info(f"  Total evaluations: {result['total_evaluations']}")
    logger.info(f"  Unique phenotypes: {result.get('unique_phenotypes', '?')}")

    return exp_dir


# ============================================================================
# VALIDATE COMMAND
# ============================================================================

def cmd_validate(args):
    """Validate evolved strategies with CPCV + DSR + PBO + signal permutation."""
    config = load_config(args.config)
    seed = args.seed or config.get('evolution', {}).get('seed', 42)
    set_global_seed(seed)

    log_dir = config.get('output', {}).get('logs_dir', './logs')
    setup_logging(log_dir)
    logger.info(f"CriptoGA v2 — validate")

    # Load strategies
    results_dir = Path(args.results)
    with open(results_dir / 'top_strategies.json', 'r') as f:
        strategies_data = json.load(f)

    logger.info(f"Loaded {len(strategies_data)} strategies from {results_dir}")

    # Reconstruct Strategy objects
    from grammar.mapper import decode

    strategies = []
    for sd in strategies_data:
        genome = sd['genome']
        s = decode(genome)
        if s is not None:
            strategies.append(s)

    if not strategies:
        logger.error("No valid strategies to validate")
        return

    data = load_evolution_data(config)
    val_cfg = config.get('validation', {})

    from validation.cpcv import cpcv_evaluate
    from validation.pbo import calculate_pbo
    from validation.deflated_sharpe import deflated_sharpe_ratio
    from validation.signal_permutation import signal_permutation_test
    from scipy import stats as scipy_stats

    # Use unique phenotypes if available (more honest than pop × gens)
    metadata_path = results_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        n_trials = metadata.get('unique_phenotypes',
                                metadata.get('total_evaluations', 1000))
        logger.info(f"Using n_trials={n_trials} (unique phenotypes)")
    else:
        n_trials = config.get('evolution', {}).get('population', 200) * \
                   config.get('evolution', {}).get('generations_max', 100)
        logger.info(f"Using n_trials={n_trials} (pop × gens estimate)")

    validation_results = []

    for i, strategy in enumerate(strategies):
        logger.info(f"\nValidating strategy {i+1}/{len(strategies)}: "
                    f"{strategy.expression_raw[:60]}...")

        # CPCV
        cpcv = cpcv_evaluate(
            strategy, data, config,
            n_groups=val_cfg.get('cpcv_groups', 10),
            purge_bars=val_cfg.get('cpcv_purge_bars', 96),
            embargo_bars=val_cfg.get('cpcv_embargo_bars', 48),
            max_splits=val_cfg.get('cpcv_max_splits', 252),
        )

        # PBO
        pbo = calculate_pbo(cpcv)

        # DSR (with n_trials=1 for OOS metrics — CPCV already debiases)
        dsr = deflated_sharpe_ratio(
            observed_sharpe=cpcv['mean_sortino'],
            n_trials=1,  # OOS metric is already debiased by CPCV
            T=len(data),
        )

        # Also compute DSR with full n_trials for reference
        dsr_full = deflated_sharpe_ratio(
            observed_sharpe=cpcv['mean_sortino'],
            n_trials=n_trials,
            T=len(data),
        )

        # T-test on OOS Sortinos: H0: mean_sortino = 0
        oos_sortinos = cpcv.get('oos_sortinos', [])
        if len(oos_sortinos) > 1:
            t_stat, t_pval_2sided = scipy_stats.ttest_1samp(oos_sortinos, 0)
            # One-sided: H1: mean > 0
            t_pval = t_pval_2sided / 2 if t_stat > 0 else 1.0 - t_pval_2sided / 2
        else:
            t_stat, t_pval = 0.0, 1.0

        # Signal permutation
        perm = signal_permutation_test(
            strategy, data, config,
            n_permutations=val_cfg.get('permutation_n', 1000),
        )

        result = {
            'strategy_index': i,
            'expression': strategy.expression_raw,
            'direction': strategy.direction,
            'cpcv_mean_sortino': cpcv['mean_sortino'],
            'cpcv_std_sortino': cpcv.get('std_sortino', 0),
            'cpcv_pct_positive': cpcv['pct_positive_sortino'],
            'cpcv_n_splits': cpcv['n_splits'],
            'cpcv_mean_trades': float(np.mean(cpcv.get('oos_trades', [0]))),
            'pbo': pbo['pbo'],
            'pbo_interpretation': pbo['interpretation'],
            'dsr_oos': dsr['dsr'],
            'dsr_full_trials': dsr_full['dsr'],
            'cpcv_ttest_stat': float(t_stat),
            'cpcv_ttest_pval': float(t_pval),
            'perm_p_value': perm['p_value'],
            'perm_interpretation': perm['interpretation'],
        }

        # Pass/fail criteria:
        # 1. PBO < threshold (anti-overfitting)
        # 2. Signal permutation p < alpha (genuine signal)
        # 3. CPCV t-test p < alpha (OOS performance significantly > 0)
        passes_pbo = pbo['pbo'] < val_cfg.get('pbo_threshold', 0.50)
        passes_perm = perm['p_value'] < val_cfg.get('permutation_alpha', 0.05)
        passes_ttest = t_pval < val_cfg.get('permutation_alpha', 0.05)
        result['passes_all'] = passes_pbo and passes_perm and passes_ttest

        validation_results.append(result)

        status = 'PASS' if result['passes_all'] else 'FAIL'
        logger.info(f"  [{status}] PBO={pbo['pbo']:.3f} t_p={t_pval:.4f} "
                    f"perm_p={perm['p_value']:.4f} DSR(oos)={dsr['dsr']:.3f}")

    # Save
    with open(results_dir / 'validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    n_passed = sum(1 for r in validation_results if r['passes_all'])
    logger.info(f"\nValidation complete: {n_passed}/{len(validation_results)} passed all tests")
    logger.info(f"Results saved to {results_dir / 'validation.json'}")

    return validation_results


# ============================================================================
# OTS COMMAND
# ============================================================================

def cmd_ots(args):
    """Final OTS evaluation. ONE SHOT — no iteration allowed."""
    config = load_config(args.config)
    seed = args.seed or config.get('evolution', {}).get('seed', 42)
    set_global_seed(seed)

    log_dir = config.get('output', {}).get('logs_dir', './logs')
    setup_logging(log_dir)
    logger.info("CriptoGA v2 — OTS FINAL EVALUATION")
    logger.info("WARNING: This is a one-shot evaluation. Results are final.")

    results_dir = Path(args.results)

    # Load validated strategies
    val_path = results_dir / 'validation.json'
    if not val_path.exists():
        logger.error("No validation.json found. Run 'validate' first.")
        return

    with open(val_path, 'r') as f:
        validation = json.load(f)

    # Only strategies that passed all tests
    passed_indices = [v['strategy_index'] for v in validation if v.get('passes_all')]

    if not passed_indices:
        logger.warning("No strategies passed validation. OTS evaluation skipped.")
        logger.info("This is a valid negative result — no alpha found.")
        with open(results_dir / 'ots_results.json', 'w') as f:
            json.dump({'result': 'no_alpha', 'strategies_tested': 0}, f, indent=2)
        return

    # Reconstruct strategies
    with open(results_dir / 'top_strategies.json', 'r') as f:
        strategies_data = json.load(f)

    from grammar.mapper import decode
    from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
    from backtest.metrics import calculate_all_metrics

    ots_data = load_ots_data(config)
    costs_config = config.get('costs', {})
    atr_period = config.get('exits', {}).get('atr_period', 14)

    ots_results = []

    for idx in passed_indices:
        sd = strategies_data[idx]
        strategy = decode(sd['genome'])
        if strategy is None:
            continue

        logger.info(f"\nOTS evaluation: strategy {idx} ({strategy.direction})")
        logger.info(f"  {strategy.expression_raw[:80]}")

        try:
            equity, trades = _run_single_window(
                strategy, ots_data, costs_config, atr_period
            )
            metrics = calculate_all_metrics(equity, BARS_PER_YEAR_15M)

            result = {
                'strategy_index': idx,
                'expression': strategy.expression_raw,
                'direction': strategy.direction,
                'n_trades': len(trades),
                'metrics': {k: float(v) if isinstance(v, (int, float, np.floating))
                           else v for k, v in metrics.items()},
            }
            ots_results.append(result)

            logger.info(f"  OTS: {len(trades)} trades, "
                        f"Sortino={metrics.get('sortino', 0):.3f}, "
                        f"CAGR={metrics.get('cagr', 0):.4f}, "
                        f"MaxDD={metrics.get('max_dd', 0):.2%}")

        except Exception as e:
            logger.error(f"  OTS evaluation failed: {e}")

    with open(results_dir / 'ots_results.json', 'w') as f:
        json.dump(ots_results, f, indent=2, default=str)

    logger.info(f"\nOTS results saved to {results_dir / 'ots_results.json'}")
    logger.info(f"Strategies evaluated on OTS: {len(ots_results)}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CriptoGA v2 — Evolutionary Trading Strategy Discovery'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # evolve
    sub_evolve = subparsers.add_parser('evolve', help='Run evolution')
    sub_evolve.add_argument('--config', default='config_v2.yaml', help='Config file')
    sub_evolve.add_argument('--seed', type=int, default=None, help='Override seed')

    # validate
    sub_validate = subparsers.add_parser('validate', help='Validate strategies')
    sub_validate.add_argument('--config', default='config_v2.yaml', help='Config file')
    sub_validate.add_argument('--seed', type=int, default=None, help='Override seed')
    sub_validate.add_argument('--results', required=True, help='Results directory')

    # ots
    sub_ots = subparsers.add_parser('ots', help='Final OTS evaluation')
    sub_ots.add_argument('--config', default='config_v2.yaml', help='Config file')
    sub_ots.add_argument('--seed', type=int, default=None, help='Override seed')
    sub_ots.add_argument('--results', required=True, help='Results directory')

    args = parser.parse_args()

    if args.command == 'evolve':
        cmd_evolve(args)
    elif args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'ots':
        cmd_ots(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
