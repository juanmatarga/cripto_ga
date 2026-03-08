#!/usr/bin/env python3
"""
Multi-asset evolution runner.

Runs the full evolve → validate → OTS pipeline for each asset independently.
Results are saved per asset and then aggregated into a cross-asset portfolio.

Usage:
    python3 run_multi_asset.py                    # All assets, seed 42
    python3 run_multi_asset.py --seed 123         # All assets, seed 123
    python3 run_multi_asset.py --symbols ETH SOL  # Specific assets only
    python3 run_multi_asset.py --aggregate-only   # Just aggregate existing results
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger('cripto_ga')

OTS_START = '2025-06-01'


def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'multi_asset_{timestamp}.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_single_asset(symbol: str, seed: int, base_config: dict) -> dict:
    """
    Run full pipeline for a single asset.

    Returns dict with results summary.
    """
    import random
    from data.multi_asset import make_asset_config
    from data.loader import load_data

    # Create per-asset config
    config = make_asset_config(symbol, base_config)
    safe_symbol = symbol.replace('/', '_')

    logger.info(f"\n{'='*60}")
    logger.info(f"ASSET: {symbol} | SEED: {seed}")
    logger.info(f"{'='*60}")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    df = load_data(config)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Split evolution / OTS
    ots_start = config.get('data', {}).get('ots_start', OTS_START)
    df_evo = df[df.index < pd.Timestamp(ots_start)]
    df_ots = df[df.index >= pd.Timestamp(ots_start)]

    logger.info(f"Evolution data: {len(df_evo)} bars")
    logger.info(f"OTS data: {len(df_ots)} bars")

    if len(df_evo) < 5000:
        logger.error(f"{symbol}: not enough evolution data ({len(df_evo)} bars)")
        return {'symbol': symbol, 'status': 'insufficient_data'}

    # ================================================================
    # EVOLVE
    # ================================================================
    from evolution.island import IslandModel

    # Reset seeds before evolution
    random.seed(seed)
    np.random.seed(seed)

    model = IslandModel(config, df_evo)
    total_pop = config.get('evolution', {}).get('population', 300)
    n_gens = config.get('evolution', {}).get('generations_max', 150)
    patience = config.get('evolution', {}).get('patience', 25)

    model.initialize(total_pop_size=total_pop)
    result = model.run(n_generations=n_gens, patience=patience)

    # Save results
    output_dir = Path(config.get('output', {}).get('results_dir', './results'))
    exp_dir = output_dir / f'experiment_{safe_symbol}_seed{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    strategies_data = [s.to_dict() for s in result['best_strategies']]
    with open(exp_dir / 'top_strategies.json', 'w') as f:
        json.dump(strategies_data, f, indent=2, default=str)

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

    archive_summary = result['archive'].summary()
    with open(exp_dir / 'archive.json', 'w') as f:
        json.dump(archive_summary, f, indent=2)

    metadata = {
        'symbol': symbol,
        'seed': seed,
        'total_evaluations': result['total_evaluations'],
        'unique_phenotypes': result.get('unique_phenotypes', result['total_evaluations']),
        'final_generation': result['final_generation'],
    }
    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Evolution done: {len(result['best_strategies'])} strategies, "
                f"{result['final_generation']} gens, "
                f"archive {result['archive'].n_occupied}/45")

    # ================================================================
    # VALIDATE
    # ================================================================
    from grammar.mapper import decode
    from validation.cpcv import cpcv_evaluate
    from validation.pbo import calculate_pbo
    from validation.deflated_sharpe import deflated_sharpe_ratio
    from validation.signal_permutation import signal_permutation_test
    from scipy import stats as scipy_stats

    strategies = []
    for sd in strategies_data:
        s = decode(sd['genome'])
        if s is not None:
            strategies.append(s)

    val_cfg = config.get('validation', {})
    n_trials = metadata['unique_phenotypes']
    validation_results = []

    for i, strategy in enumerate(strategies):
        logger.info(f"Validating {i+1}/{len(strategies)}: "
                    f"{strategy.expression_raw[:50]}...")

        cpcv = cpcv_evaluate(
            strategy, df_evo, config,
            n_groups=val_cfg.get('cpcv_groups', 10),
            purge_bars=val_cfg.get('cpcv_purge_bars', 96),
            embargo_bars=val_cfg.get('cpcv_embargo_bars', 48),
            max_splits=val_cfg.get('cpcv_max_splits', 252),
        )
        pbo = calculate_pbo(cpcv)
        dsr = deflated_sharpe_ratio(
            observed_sharpe=cpcv['mean_sortino'],
            n_trials=1, T=len(df_evo),
        )

        oos_sortinos = cpcv.get('oos_sortinos', [])
        if len(oos_sortinos) > 1:
            t_stat, t_pval_2sided = scipy_stats.ttest_1samp(oos_sortinos, 0)
            t_pval = t_pval_2sided / 2 if t_stat > 0 else 1.0 - t_pval_2sided / 2
        else:
            t_stat, t_pval = 0.0, 1.0

        perm = signal_permutation_test(
            strategy, df_evo, config,
            n_permutations=val_cfg.get('permutation_n', 1000),
        )

        passes_pbo = pbo['pbo'] < val_cfg.get('pbo_threshold', 0.50)
        passes_perm = perm['p_value'] < val_cfg.get('permutation_alpha', 0.05)
        passes_ttest = t_pval < val_cfg.get('permutation_alpha', 0.05)

        vr = {
            'strategy_index': i,
            'expression': strategy.expression_raw,
            'direction': strategy.direction,
            'cpcv_mean_sortino': cpcv['mean_sortino'],
            'cpcv_pct_positive': cpcv['pct_positive_sortino'],
            'cpcv_n_splits': cpcv['n_splits'],
            'pbo': pbo['pbo'],
            'cpcv_ttest_pval': float(t_pval),
            'perm_p_value': perm['p_value'],
            'passes_all': passes_pbo and passes_perm and passes_ttest,
        }
        validation_results.append(vr)

        status = 'PASS' if vr['passes_all'] else 'FAIL'
        logger.info(f"  [{status}] PBO={pbo['pbo']:.3f} t_p={t_pval:.4f} "
                    f"perm_p={perm['p_value']:.4f}")

    with open(exp_dir / 'validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    n_passed = sum(1 for r in validation_results if r['passes_all'])
    logger.info(f"Validation: {n_passed}/{len(validation_results)} passed")

    # ================================================================
    # OTS
    # ================================================================
    from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
    from backtest.metrics import calculate_all_metrics

    passed_indices = [v['strategy_index'] for v in validation_results
                      if v.get('passes_all')]
    ots_results = []

    if len(df_ots) > 0 and passed_indices:
        costs_config = config.get('costs', {})
        atr_period = config.get('exits', {}).get('atr_period', 14)

        for idx in passed_indices:
            sd = strategies_data[idx]
            strategy = decode(sd['genome'])
            if strategy is None:
                continue

            try:
                equity, trades = _run_single_window(
                    strategy, df_ots, costs_config, atr_period
                )
                metrics = calculate_all_metrics(equity, BARS_PER_YEAR_15M)
                ots_results.append({
                    'strategy_index': idx,
                    'expression': strategy.expression_raw,
                    'direction': strategy.direction,
                    'n_trades': len(trades),
                    'metrics': {k: float(v) if isinstance(v, (int, float, np.floating))
                               else v for k, v in metrics.items()},
                })
                cagr = metrics.get('cagr', 0)
                sortino = metrics.get('sortino', 0)
                logger.info(f"  OTS: {len(trades)} trades, "
                            f"Sortino={sortino:.3f}, CAGR={cagr:.4f}")
            except Exception as e:
                logger.error(f"  OTS failed: {e}")
    else:
        if not passed_indices:
            logger.info("No strategies passed validation — OTS skipped")
        else:
            logger.warning("No OTS data available")

    with open(exp_dir / 'ots_results.json', 'w') as f:
        json.dump(ots_results, f, indent=2, default=str)

    n_ots_pos = sum(1 for r in ots_results if r['metrics'].get('cagr', 0) > 0)

    summary = {
        'symbol': symbol,
        'seed': seed,
        'status': 'complete',
        'results_dir': str(exp_dir),
        'n_strategies': len(strategies_data),
        'n_validated': len(validation_results),
        'n_passed': n_passed,
        'n_ots_tested': len(ots_results),
        'n_ots_positive': n_ots_pos,
        'best_ots_cagr': max((r['metrics'].get('cagr', -999)
                              for r in ots_results), default=None),
        'best_ots_sortino': max((r['metrics'].get('sortino', -999)
                                 for r in ots_results), default=None),
    }

    logger.info(f"\n{symbol} SUMMARY: "
                f"val={n_passed}/{len(validation_results)}, "
                f"OTS={n_ots_pos}/{len(ots_results)}, "
                f"best CAGR={summary['best_ots_cagr']}")

    return summary


def aggregate_results(summaries: list, output_dir: Path):
    """Aggregate results across all assets into a cross-asset report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'assets': [],
        'totals': {
            'total_strategies': 0,
            'total_passed_validation': 0,
            'total_ots_tested': 0,
            'total_ots_positive': 0,
        },
    }

    for s in summaries:
        if s.get('status') != 'complete':
            continue
        report['assets'].append(s)
        report['totals']['total_strategies'] += s.get('n_strategies', 0)
        report['totals']['total_passed_validation'] += s.get('n_passed', 0)
        report['totals']['total_ots_tested'] += s.get('n_ots_tested', 0)
        report['totals']['total_ots_positive'] += s.get('n_ots_positive', 0)

    # Best strategies across assets
    all_best = []
    for s in summaries:
        if s.get('status') == 'complete' and s.get('best_ots_cagr') is not None:
            all_best.append({
                'symbol': s['symbol'],
                'cagr': s['best_ots_cagr'],
                'sortino': s.get('best_ots_sortino'),
                'results_dir': s.get('results_dir'),
            })
    all_best.sort(key=lambda x: x['cagr'], reverse=True)
    report['best_per_asset'] = all_best

    with open(output_dir / 'multi_asset_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-ASSET RESULTS")
    print(f"{'='*60}")
    t = report['totals']
    print(f"Total strategies evolved: {t['total_strategies']}")
    print(f"Passed validation: {t['total_passed_validation']}")
    print(f"OTS positive: {t['total_ots_positive']}/{t['total_ots_tested']}")
    print(f"\nBest per asset:")
    for b in all_best:
        cagr = b['cagr'] * 100 if b['cagr'] else 0
        sortino = b['sortino'] or 0
        print(f"  {b['symbol']}: CAGR={cagr:+.1f}%, Sortino={sortino:.3f}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='CriptoGA Multi-Asset Evolution'
    )
    parser.add_argument('--config', default='config_v2.yaml',
                        help='Base config file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to process (default: all)')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only aggregate existing results')

    args = parser.parse_args()
    setup_logging()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    from data.multi_asset import ASSETS
    symbols = args.symbols or list(ASSETS.keys())
    # Normalize symbol format
    symbols = [s if '/' in s else f"{s}/USDT" for s in symbols]

    output_dir = Path(base_config.get('output', {}).get('results_dir', './results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        # Find existing results and aggregate
        summaries = []
        for d in sorted(output_dir.iterdir()):
            if d.is_dir() and (d / 'metadata.json').exists():
                with open(d / 'metadata.json') as f:
                    meta = json.load(f)
                if meta.get('symbol') in symbols:
                    with open(d / 'ots_results.json') as f:
                        ots = json.load(f)
                    with open(d / 'validation.json') as f:
                        val = json.load(f)
                    summaries.append({
                        'symbol': meta['symbol'],
                        'seed': meta.get('seed'),
                        'status': 'complete',
                        'results_dir': str(d),
                        'n_strategies': meta.get('unique_phenotypes', 0),
                        'n_validated': len(val),
                        'n_passed': sum(1 for v in val if v.get('passes_all')),
                        'n_ots_tested': len(ots),
                        'n_ots_positive': sum(1 for r in ots
                                              if r['metrics'].get('cagr', 0) > 0),
                        'best_ots_cagr': max((r['metrics'].get('cagr', -999)
                                              for r in ots), default=None),
                        'best_ots_sortino': max((r['metrics'].get('sortino', -999)
                                                 for r in ots), default=None),
                    })
        aggregate_results(summaries, output_dir)
        return

    # Run each asset
    summaries = []
    t0 = time.time()

    for symbol in symbols:
        try:
            summary = run_single_asset(symbol, args.seed, base_config)
            summaries.append(summary)
        except Exception as e:
            logger.error(f"FAILED {symbol}: {e}", exc_info=True)
            summaries.append({'symbol': symbol, 'status': 'error', 'error': str(e)})

    elapsed = time.time() - t0
    logger.info(f"\nAll assets complete in {elapsed:.0f}s")

    # Aggregate
    aggregate_results(summaries, output_dir)


if __name__ == '__main__':
    main()
