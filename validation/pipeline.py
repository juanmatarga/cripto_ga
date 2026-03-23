"""
Automated Validation Pipeline.

A strategy must pass ALL gates to be portfolio-eligible.
No manual judgment needed — pass/fail is binary.

Gates (in order):
1. OTS Profitability: CAGR > 0% in OTS period (Jun-Nov 2025)
2. Cross-Regime: Positive in >= 3 of 4 market regimes
3. Signal Permutation: p < 0.10 (signal timing beats random)
4. Minimum Trades: >= 30 total across all periods
5. Max Drawdown: < 25% in any single period
6. No Symbol+Direction Conflict: 1 per combo in final portfolio
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from grammar.mapper import decode
from strategy.vectorized_eval import generate_signals
from evolution.fitness import _run_single_window, BARS_PER_YEAR_15M
from backtest.metrics import calculate_all_metrics
from backtest.exits import calculate_atr
from data.multi_timeframe import prepare_multi_tf_data
from data.loader import load_data

logger = logging.getLogger(__name__)

COSTS = {
    'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
    'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
}

PERIODS = {
    'OTS': ('2025-06-01', '2025-11-21'),
    'Bear': ('2022-01-01', '2022-07-01'),
    'Side': ('2023-07-01', '2024-01-01'),
    'Bull': ('2024-10-01', '2025-03-01'),
}

# Gate thresholds
MIN_OTS_CAGR = 0.0
MIN_POSITIVE_PERIODS = 3
SIGNAL_PERM_P_THRESHOLD = 0.10
SIGNAL_PERM_N = 300
MIN_TOTAL_TRADES = 30
MAX_PERIOD_DD = 0.25


@dataclass
class ValidationResult:
    """Result of validating a single strategy."""
    # Identity
    symbol: str
    direction: str
    n_nodes: int
    conditions: List[str]
    genome: List[int]
    logic: str
    tp_atr_mult: float
    sl_atr_mult: float
    trail_atr_mult: float

    # Gate results
    ots_cagr: float = 0.0
    ots_trades: int = 0
    ots_wr: float = 0.0
    ots_pf: float = 0.0
    ots_dd: float = 0.0

    bear_cagr: float = 0.0
    bear_trades: int = 0
    side_cagr: float = 0.0
    side_trades: int = 0
    bull_cagr: float = 0.0
    bull_trades: int = 0

    positive_periods: int = 0
    total_trades: int = 0
    worst_dd: float = 0.0
    signal_perm_p: float = 1.0

    # Gates passed
    gate_ots: bool = False
    gate_cross_regime: bool = False
    gate_signal_perm: bool = False
    gate_min_trades: bool = False
    gate_max_dd: bool = False
    passes_all: bool = False


def _load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load full data for a symbol."""
    config = {
        'data': {
            'symbol': f'{symbol}/USDT',
            'market_type': 'future',
            'timeframe': '15m',
            'start': '2022-01-01',
            'end': '2025-11-21',
        }
    }
    df = load_data(config)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _backtest_period(strategy, data, tf_data) -> Dict:
    """Run backtest on a period, return metrics dict."""
    eq, trades = _run_single_window(strategy, data, COSTS, 14, tf_data=tf_data)
    m = calculate_all_metrics(eq, BARS_PER_YEAR_15M)
    nt = len(trades)
    cagr = float(m.get('cagr', 0))
    dd = float(m.get('max_dd', 0))
    w = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = w / nt if nt > 0 else 0
    wp = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    lp = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    pf = wp / max(lp, 1e-10)
    return {
        'cagr': cagr, 'trades': nt, 'wr': wr, 'pf': pf, 'dd': dd,
        'equity': eq, 'trade_list': trades,
    }


def _signal_permutation(strategy, data, tf_data, n_perm=SIGNAL_PERM_N) -> float:
    """Run signal permutation test. Returns p-value."""
    signals = generate_signals(strategy, data, tf_data=tf_data)
    n_signals = int(signals.sum())
    if n_signals < 3:
        return 1.0

    eq, trades = _run_single_window(strategy, data, COSTS, 14, tf_data=tf_data)
    real_ret = eq.iloc[-1] / eq.iloc[0] - 1

    atr = calculate_atr(data, period=14)
    direction = strategy.direction
    tp_mult = strategy.tp_atr_mult
    sl_mult = strategy.sl_atr_mult
    trail_mult = strategy.trail_atr_mult
    has_tp = tp_mult > 0
    total_cost = 2.0 / 10000

    highs = data['High'].values
    lows = data['Low'].values
    closes = data['Close'].values
    atr_vals = atr.values
    sig_vals = signals.values.copy()

    perm_rets = []
    for _ in range(n_perm):
        shuffled = sig_vals.copy()
        np.random.shuffle(shuffled)
        equity = 100.0
        in_pos = False
        ep = sp = tp_l = bp = ae = 0.0
        eb = 0
        for i in range(len(data)):
            if in_pos:
                h, l = highs[i], lows[i]
                if trail_mult > 0 and ae > 0:
                    if direction == 'LONG':
                        bp = max(bp, h)
                        tl = bp - trail_mult * ae
                        if tl > sp: sp = tl
                    else:
                        bp = min(bp, l)
                        tl = bp + trail_mult * ae
                        if tl < sp: sp = tl
                if direction == 'LONG':
                    sh = l <= sp
                    th = has_tp and h >= tp_l
                else:
                    sh = h >= sp
                    th = has_tp and l <= tp_l
                te = (i - eb) >= 960
                exit_p = sp if sh else (tp_l if th else (closes[i] if te else None))
                if exit_p is not None:
                    if direction == 'LONG':
                        pnl = (exit_p * (1 - total_cost) - ep) / ep
                    else:
                        pnl = (ep - exit_p * (1 + total_cost)) / ep
                    equity *= (1 + pnl)
                    in_pos = False
            else:
                if shuffled[i] and not np.isnan(atr_vals[i]) and atr_vals[i] > 0:
                    ae = atr_vals[i]
                    if direction == 'LONG':
                        ep = closes[i] * (1 + total_cost)
                        sp = ep - sl_mult * ae
                        tp_l = ep + tp_mult * ae if has_tp else 0
                        bp = ep
                    else:
                        ep = closes[i] * (1 - total_cost)
                        sp = ep + sl_mult * ae
                        tp_l = ep - tp_mult * ae if has_tp else 0
                        bp = ep
                    eb = i
                    in_pos = True
        perm_rets.append(equity / 100.0 - 1)

    return float(np.mean(np.array(perm_rets) >= real_ret))


def validate_strategy(genome: List[int], symbol: str,
                      symbol_data: pd.DataFrame = None) -> Optional[ValidationResult]:
    """
    Run full validation pipeline on a single strategy.
    Returns ValidationResult or None if genome is invalid.
    """
    strategy = decode(genome)
    if strategy is None:
        return None

    if symbol_data is None:
        symbol_data = _load_symbol_data(symbol)

    result = ValidationResult(
        symbol=symbol,
        direction=strategy.direction,
        n_nodes=strategy.n_nodes,
        conditions=[str(c) for c in strategy.conditions],
        genome=genome,
        logic=strategy.logic,
        tp_atr_mult=strategy.tp_atr_mult,
        sl_atr_mult=strategy.sl_atr_mult,
        trail_atr_mult=strategy.trail_atr_mult,
    )

    # Compute metrics for each period
    period_results = {}
    for pname, (start, end) in PERIODS.items():
        pdata = symbol_data[
            (symbol_data.index >= pd.Timestamp(start)) &
            (symbol_data.index < pd.Timestamp(end))
        ]
        if len(pdata) < 500:
            continue
        tf = prepare_multi_tf_data(pdata)
        try:
            period_results[pname] = _backtest_period(strategy, pdata, tf)
        except Exception:
            period_results[pname] = {'cagr': -999, 'trades': 0, 'wr': 0,
                                      'pf': 0, 'dd': -1}

    # Fill result fields
    if 'OTS' in period_results:
        r = period_results['OTS']
        result.ots_cagr = r['cagr']
        result.ots_trades = r['trades']
        result.ots_wr = r['wr']
        result.ots_pf = r['pf']
        result.ots_dd = r['dd']
    if 'Bear' in period_results:
        result.bear_cagr = period_results['Bear']['cagr']
        result.bear_trades = period_results['Bear']['trades']
    if 'Side' in period_results:
        result.side_cagr = period_results['Side']['cagr']
        result.side_trades = period_results['Side']['trades']
    if 'Bull' in period_results:
        result.bull_cagr = period_results['Bull']['cagr']
        result.bull_trades = period_results['Bull']['trades']

    # Aggregate metrics
    result.positive_periods = sum(
        1 for p in ['OTS', 'Bear', 'Side', 'Bull']
        if period_results.get(p, {}).get('cagr', -999) > 0
    )
    result.total_trades = sum(
        period_results.get(p, {}).get('trades', 0)
        for p in ['OTS', 'Bear', 'Side', 'Bull']
    )
    result.worst_dd = max(
        abs(period_results.get(p, {}).get('dd', 0))
        for p in ['OTS', 'Bear', 'Side', 'Bull']
        if p in period_results
    )

    # Gate 1: OTS profitability
    result.gate_ots = result.ots_cagr > MIN_OTS_CAGR and result.ots_trades >= 5

    # Gate 2: Cross-regime
    result.gate_cross_regime = result.positive_periods >= MIN_POSITIVE_PERIODS

    # Gate 4: Minimum trades
    result.gate_min_trades = result.total_trades >= MIN_TOTAL_TRADES

    # Gate 5: Max drawdown
    result.gate_max_dd = result.worst_dd < MAX_PERIOD_DD

    # Gate 3: Signal permutation (expensive — only run if other gates pass)
    if result.gate_ots and result.gate_cross_regime and result.gate_min_trades and result.gate_max_dd:
        ots_data = symbol_data[
            (symbol_data.index >= pd.Timestamp('2025-06-01')) &
            (symbol_data.index < pd.Timestamp('2025-11-21'))
        ]
        tf_ots = prepare_multi_tf_data(ots_data)
        result.signal_perm_p = _signal_permutation(strategy, ots_data, tf_ots)
        result.gate_signal_perm = result.signal_perm_p < SIGNAL_PERM_P_THRESHOLD
    else:
        result.gate_signal_perm = False

    # Final verdict
    result.passes_all = (
        result.gate_ots and
        result.gate_cross_regime and
        result.gate_signal_perm and
        result.gate_min_trades and
        result.gate_max_dd
    )

    return result


def validate_pareto_front(pareto_path: str, symbol: str) -> List[ValidationResult]:
    """Validate all strategies in a pareto_front.json file."""
    with open(pareto_path) as f:
        strategies = json.load(f)

    symbol_data = _load_symbol_data(symbol)
    results = []

    for s in strategies:
        genome = s.get('genome', [])
        if not genome:
            continue
        result = validate_strategy(genome, symbol, symbol_data)
        if result is not None:
            results.append(result)

    return results


def build_portfolio(results: List[ValidationResult],
                    max_per_combo: int = 1) -> List[ValidationResult]:
    """
    Build portfolio from validated strategies.
    Only includes strategies that pass ALL gates.
    Picks best (by OTS CAGR) per (symbol, direction) combo.
    """
    passing = [r for r in results if r.passes_all]
    passing.sort(key=lambda r: r.ots_cagr, reverse=True)

    portfolio = []
    seen_combos = set()

    for r in passing:
        combo = (r.symbol, r.direction)
        if seen_combos.get(combo, 0) if isinstance(seen_combos, dict) else combo in seen_combos:
            continue
        portfolio.append(r)
        seen_combos.add(combo)

    return portfolio


def print_validation_summary(results: List[ValidationResult]):
    """Print a summary table of validation results."""
    print(f"\n{'='*100}")
    print(f"VALIDATION PIPELINE RESULTS")
    print(f"{'='*100}")
    print(f"{'Sym':>4} {'Dir':>5} {'N':>2} {'OTS':>7} {'Bear':>7} {'Side':>7} {'Bull':>7} "
          f"{'T':>4} {'DD':>6} {'p':>6} {'Pass':>5}  Conditions")
    print('-' * 100)

    for r in sorted(results, key=lambda x: x.ots_cagr, reverse=True):
        gates = ''
        if not r.gate_ots: gates += 'OTS '
        if not r.gate_cross_regime: gates += 'CR '
        if not r.gate_signal_perm: gates += 'SP '
        if not r.gate_min_trades: gates += 'T '
        if not r.gate_max_dd: gates += 'DD '

        status = 'PASS' if r.passes_all else f'fail({gates.strip()})'
        p_str = f'{r.signal_perm_p:.3f}' if r.signal_perm_p < 1.0 else '—'
        conds = '; '.join(r.conditions)[:45]

        print(f"{r.symbol:>4} {r.direction:>5} [{r.n_nodes}n] "
              f"{r.ots_cagr:>+6.1%} {r.bear_cagr:>+6.1%} {r.side_cagr:>+6.1%} "
              f"{r.bull_cagr:>+6.1%} {r.total_trades:>4} {r.worst_dd:>5.1%} "
              f"{p_str:>6} {status:>5}  {conds}")

    total = len(results)
    passing = sum(1 for r in results if r.passes_all)
    print(f"\nTotal: {total} evaluated, {passing} pass ALL gates")
