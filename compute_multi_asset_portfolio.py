"""
Compute multi-asset portfolio metrics for the 10-strategy live portfolio.

Loads OTS data per symbol, backtests each strategy individually,
then combines into an equal-weight portfolio and computes combined metrics.
"""

import json
import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
from grammar.mapper import decode
from evolution.fitness import _run_single_window
from evolution.param_extractor import extract_params, rebuild_strategy
from data.loader import load_data
from data.multi_timeframe import prepare_multi_tf_data
from live.config import PORTFOLIO, _find_experiment_dir

OTS_START = '2025-06-01'
OTS_END = '2025-11-21'


def load_ots_for_symbol(symbol: str) -> pd.DataFrame:
    """Load OTS data for a specific symbol."""
    # Map ccxt symbol to base symbol for config
    base = symbol.replace('/USDT:USDT', '/USDT')
    config = {
        'data': {
            'exchange': 'binance',
            'symbol': base,
            'market_type': 'future',
            'timeframe': '15m',
            'start': '2022-01-01',
            'end': OTS_END,
        }
    }
    df = load_data(config)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index >= pd.Timestamp(OTS_START)]
    logger.info(f"{symbol}: {len(df)} OTS bars ({df.index.min()} to {df.index.max()})")
    return df


def load_and_backtest_strategy(entry: dict, ots_data: pd.DataFrame,
                                costs_config: dict, atr_period: int):
    """Load a strategy from results, apply CMA-ES if needed, and backtest on OTS."""
    results_dir = Path('results')
    exp_dir = _find_experiment_dir(results_dir, entry['results_dir'], entry['seed'])

    with open(exp_dir / 'top_strategies.json') as f:
        top_strats = json.load(f)

    idx = entry['strategy_index']
    sd = top_strats[idx]

    # Decode genome
    strategy = decode(sd['genome'])
    if strategy is None:
        raise ValueError(f"Failed to decode {entry['label']}")

    # Apply CMA-ES overrides
    cmaes = entry.get('cmaes_params', {})
    if cmaes:
        param_specs = extract_params(strategy)
        param_vector = [cmaes.get(ps.name, ps.value) for ps in param_specs]
        strategy = rebuild_strategy(strategy, param_vector, param_specs)
        logger.info(f"  {entry['label']} [CMA-ES]: {strategy.expression_raw[:80]}")
    else:
        logger.info(f"  {entry['label']}: {strategy.expression_raw[:80]}")

    # Prepare multi-TF data
    tf_data = prepare_multi_tf_data(ots_data)

    # Run backtest
    equity, trades = _run_single_window(strategy, ots_data, costs_config, atr_period,
                                         tf_data=tf_data)
    return equity, trades, strategy


def compute_portfolio_metrics(returns_series, periods=35040):
    """Compute CAGR, MaxDD, Sortino, Calmar, Sharpe from returns series."""
    r = returns_series.replace([np.inf, -np.inf], 0).fillna(0)
    eq = (1 + r).cumprod() * 100
    n = len(eq)
    if n < 2 or eq.iloc[-1] <= 0:
        return {}
    years = n / periods
    cagr_v = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-6)) - 1
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Sortino
    down = r[r < 0]
    if len(down) > 0:
        ds = np.sqrt((down ** 2).mean())
        sortino = (r.mean() / ds) * np.sqrt(periods) if ds > 0 else 999
    else:
        sortino = 999

    # Sharpe
    sharpe = (r.mean() / r.std()) * np.sqrt(periods) if r.std() > 0 else 999

    calmar = cagr_v / abs(max_dd) if abs(max_dd) > 1e-10 else 999

    # Profit factor from returns
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else 999

    return {
        'cagr': cagr_v,
        'max_dd': max_dd,
        'sortino': sortino,
        'sharpe': sharpe,
        'calmar': calmar,
        'profit_factor': pf,
        'final_equity': eq.iloc[-1],
        'total_return': eq.iloc[-1] / eq.iloc[0] - 1,
    }


def main():
    # Cost config
    costs_config = {
        'fees_bps_long': 1.0,
        'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0,
        'slippage_bps_short': 1.0,
    }
    atr_period = 14

    # Group strategies by symbol
    symbols = {}
    for entry in PORTFOLIO:
        sym = entry['symbol']
        if sym not in symbols:
            symbols[sym] = []
        symbols[sym].append(entry)

    # Load OTS data per symbol
    print("\n=== Loading OTS data per symbol ===")
    ots_data = {}
    for sym in symbols:
        ots_data[sym] = load_ots_for_symbol(sym)

    # Backtest each strategy
    print("\n=== Backtesting 10 strategies ===")
    equities = {}
    all_trades = {}
    strategy_results = []

    for sym, entries in symbols.items():
        df = ots_data[sym]
        bh_return = df['Close'].iloc[-1] / df['Close'].iloc[0] - 1

        for entry in entries:
            equity, trades, strategy = load_and_backtest_strategy(
                entry, df, costs_config, atr_period
            )
            key = entry['label']
            equities[key] = equity
            all_trades[key] = trades

            # Individual metrics
            ret = equity.pct_change().fillna(0)
            m = compute_portfolio_metrics(ret)
            n_trades = len(trades)
            winning = sum(1 for t in trades if t['pnl_pct'] > 0)
            win_rate = winning / n_trades if n_trades > 0 else 0

            strategy_results.append({
                'key': key,
                'symbol': sym.split('/')[0],
                'direction': strategy.direction,
                'cagr': m.get('cagr', 0),
                'max_dd': m.get('max_dd', 0),
                'sortino': m.get('sortino', 0),
                'sharpe': m.get('sharpe', 0),
                'profit_factor': m.get('profit_factor', 0),
                'n_trades': n_trades,
                'win_rate': win_rate,
                'total_return': m.get('total_return', 0),
                'has_cmaes': bool(entry.get('cmaes_params')),
            })

    # Print individual results
    print(f"\n{'='*100}")
    print(f"{'Strategy':<28} {'Sym':>4} {'Dir':>6} {'CAGR':>8} {'MaxDD':>8} {'Sort':>7} {'Sharp':>7} {'PF':>6} {'#Tr':>4} {'WR':>6}")
    print(f"{'='*100}")
    for r in strategy_results:
        cmaes_tag = '*' if r['has_cmaes'] else ' '
        print(f"{r['key']:<27}{cmaes_tag} {r['symbol']:>4} {r['direction']:>6} "
              f"{r['cagr']:>7.1%} {r['max_dd']:>7.2%} {r['sortino']:>7.2f} "
              f"{r['sharpe']:>7.2f} {r['profit_factor']:>5.2f} {r['n_trades']:>4} "
              f"{r['win_rate']:>5.1%}")

    # Build equal-weight portfolio
    print(f"\n{'='*100}")
    print("COMBINED PORTFOLIO (equal weight, 10 strategies)")
    print(f"{'='*100}")

    returns_dict = {}
    for k, eq in equities.items():
        returns_dict[k] = eq.pct_change().fillna(0)
    returns_df = pd.DataFrame(returns_dict)

    # Equal-weight portfolio returns
    portfolio_returns = returns_df.mean(axis=1)
    pm = compute_portfolio_metrics(portfolio_returns)

    print(f"\n  CAGR:            {pm['cagr']:.2%}")
    print(f"  Max Drawdown:    {pm['max_dd']:.2%}")
    print(f"  Sortino:         {pm['sortino']:.3f}")
    print(f"  Sharpe:          {pm['sharpe']:.3f}")
    print(f"  Calmar:          {pm['calmar']:.2f}")
    print(f"  Profit Factor:   {pm['profit_factor']:.2f}")
    print(f"  Total Return:    {pm['total_return']:.2%}")
    print(f"  Final Equity:    {pm['final_equity']:.2f}")

    # Buy & Hold comparison (average of 3 assets)
    print(f"\n--- Buy & Hold comparison ---")
    for sym, df in ots_data.items():
        bh_ret = df['Close'].iloc[-1] / df['Close'].iloc[0] - 1
        sym_short = sym.split('/')[0]
        print(f"  {sym_short} B&H: {bh_ret:.2%}")

    # Correlation matrix
    print(f"\n--- Return correlation matrix ---")
    corr = returns_df.corr()
    # Show abbreviated
    short_keys = [k.replace('_cmaes', '*') for k in returns_df.columns]
    corr.columns = short_keys
    corr.index = short_keys
    print(corr.round(2).to_string())

    # Average pairwise correlation
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    avg_corr = corr.where(mask).stack().mean()
    print(f"\n  Average pairwise correlation: {avg_corr:.3f}")

    # Total trades
    total_trades = sum(len(t) for t in all_trades.values())
    print(f"\n  Total trades across portfolio: {total_trades}")

    # Per-symbol portfolio
    print(f"\n--- Per-symbol sub-portfolios ---")
    for sym in symbols:
        sym_short = sym.split('/')[0]
        sym_keys = [e['label'] for e in symbols[sym]]
        sym_returns = returns_df[sym_keys].mean(axis=1)
        sm = compute_portfolio_metrics(sym_returns)
        bh_ret = ots_data[sym]['Close'].iloc[-1] / ots_data[sym]['Close'].iloc[0] - 1
        print(f"  {sym_short}: CAGR={sm['cagr']:.1%}, MaxDD={sm['max_dd']:.2%}, "
              f"Sortino={sm['sortino']:.2f}, B&H={bh_ret:.1%}")

    # Save results
    output = {
        'portfolio': pm,
        'strategies': strategy_results,
        'avg_correlation': float(avg_corr),
        'total_trades': total_trades,
    }
    output_path = Path('results/multi_asset_portfolio_ots.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
