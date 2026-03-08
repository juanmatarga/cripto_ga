"""
Extract tunable parameters from a decoded Strategy for CMA-ES optimization.

Given a Strategy phenotype (decoded from a genome), this module extracts all
numeric parameters that can be optimized continuously:
  - Indicator periods (RSI period, MACD fast/slow/signal, etc.)
  - Bollinger Band std deviations
  - Thresholds (oscillator 0-100, normalized -5 to 5)
  - Exit multipliers (TP, SL, trail ATR multiples)

Sources (close/open/high/low) and timeframes (15m/1h/4h) are NOT optimizable
since they represent categorical structure decisions.
"""

import re
import copy
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

from strategy.phenotype import Strategy, Condition

logger = logging.getLogger(__name__)

# Known source names and timeframes (not optimizable)
SOURCES = {'close', 'open', 'high', 'low', 'volume'}
TIMEFRAMES = {'15m', '1h', '4h'}

# Indicator signatures: which args are what type
# Format: list of (arg_name, arg_type) tuples
# arg_type: 'source' (categorical), 'period' (int), 'std' (float), 'tf' (categorical)
INDICATOR_SIGS = {
    'RSI':       [('source', 'source'), ('period', 'period')],
    'STOCH_K':   [('period', 'period')],
    'STOCH_D':   [('period', 'period')],
    'ADX':       [('period', 'period')],
    'MFI':       [('period', 'period')],
    'PCT_B':     [('period', 'period'), ('std_dev', 'std')],
    'MACD_NORM': [('fast', 'period'), ('slow', 'period'), ('signal', 'period')],
    'PRICE_POS': [('period', 'period')],
    'ROC':       [('period', 'period')],
    'VOL_RATIO': [('period', 'period')],
    'BBWIDTH':   [('period', 'period'), ('std_dev', 'std')],
    'ATR_PCT':   [('period', 'period')],
}

# Bounds for each parameter type
PERIOD_BOUNDS = (2, 100)
STD_BOUNDS = (0.5, 5.0)
OSC_THRESHOLD_BOUNDS = (0.0, 100.0)
NORM_THRESHOLD_BOUNDS = (-5.0, 5.0)
TP_BOUNDS = (0.5, 15.0)
SL_BOUNDS = (0.2, 5.0)
TRAIL_BOUNDS = (0.5, 10.0)


@dataclass
class ParamSpec:
    """Specification for a single tunable parameter."""
    name: str           # Human-readable name (e.g. "c0_left_RSI_period")
    value: float        # Current value
    bounds: Tuple[float, float]  # (lower, upper) for CMA-ES
    param_type: str     # 'period', 'std', 'osc_threshold', 'norm_threshold', 'tp', 'sl', 'trail'
    # Location info for reconstruction
    condition_idx: int = -1     # Which condition (-1 for exits)
    side: str = ''              # 'left', 'right', or '' for exits
    arg_idx: int = -1           # Which argument in the indicator call


def _is_threshold(s: str) -> bool:
    """Check if a string is a numeric literal (threshold)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_indicator(indicator_str: str) -> Optional[Tuple[str, List[str]]]:
    """Parse 'FUNC(arg1, arg2, ...)' into (func_name, [args])."""
    match = re.match(r'^(\w+)\((.+)\)$', indicator_str.strip())
    if not match:
        return None
    func_name = match.group(1)
    args_str = match.group(2)
    args = [a.strip() for a in args_str.split(',')]
    return func_name, args


def _classify_threshold(indicator_str: str, condition: Condition) -> str:
    """Determine if a threshold is oscillator-type or normalized-type."""
    # Find the indicator side (the NON-threshold side)
    if _is_threshold(condition.left):
        indicator_side = condition.right
    else:
        indicator_side = condition.left

    if _is_threshold(indicator_side):
        # Both sides are thresholds — shouldn't happen normally
        return 'norm_threshold'

    parsed = _parse_indicator(indicator_side)
    if parsed is None:
        return 'norm_threshold'

    func_name = parsed[0]
    # Oscillator indicators (output 0-100)
    if func_name in ('RSI', 'STOCH_K', 'STOCH_D', 'ADX', 'MFI'):
        return 'osc_threshold'
    else:
        return 'norm_threshold'


def _extract_indicator_params(indicator_str: str, condition_idx: int,
                               side: str) -> List[ParamSpec]:
    """Extract tunable parameters from an indicator call string."""
    parsed = _parse_indicator(indicator_str)
    if parsed is None:
        return []

    func_name, args = parsed
    sig = INDICATOR_SIGS.get(func_name)
    if sig is None:
        return []

    params = []
    # Strip timeframe from args if present (last arg)
    actual_args = args[:]
    if actual_args and actual_args[-1] in TIMEFRAMES:
        actual_args = actual_args[:-1]

    for i, (arg_name, arg_type) in enumerate(sig):
        if i >= len(actual_args):
            break

        if arg_type == 'source':
            continue  # Not optimizable

        value_str = actual_args[i]
        try:
            value = float(value_str)
        except ValueError:
            continue

        if arg_type == 'period':
            bounds = PERIOD_BOUNDS
            ptype = 'period'
        elif arg_type == 'std':
            bounds = STD_BOUNDS
            ptype = 'std'
        else:
            continue

        params.append(ParamSpec(
            name=f"c{condition_idx}_{side}_{func_name}_{arg_name}",
            value=value,
            bounds=bounds,
            param_type=ptype,
            condition_idx=condition_idx,
            side=side,
            arg_idx=i,
        ))

    return params


def extract_params(strategy: Strategy) -> List[ParamSpec]:
    """
    Extract all tunable parameters from a decoded Strategy.

    Returns ordered list of ParamSpec. The order defines the CMA-ES vector.
    """
    params = []

    for ci, cond in enumerate(strategy.conditions):
        # Left side
        if _is_threshold(cond.left):
            ttype = _classify_threshold(cond.left, cond)
            bounds = OSC_THRESHOLD_BOUNDS if ttype == 'osc_threshold' else NORM_THRESHOLD_BOUNDS
            params.append(ParamSpec(
                name=f"c{ci}_left_threshold",
                value=float(cond.left),
                bounds=bounds,
                param_type=ttype,
                condition_idx=ci,
                side='left',
                arg_idx=-1,
            ))
        else:
            params.extend(_extract_indicator_params(cond.left, ci, 'left'))

        # Right side
        if _is_threshold(cond.right):
            ttype = _classify_threshold(cond.right, cond)
            bounds = OSC_THRESHOLD_BOUNDS if ttype == 'osc_threshold' else NORM_THRESHOLD_BOUNDS
            params.append(ParamSpec(
                name=f"c{ci}_right_threshold",
                value=float(cond.right),
                bounds=bounds,
                param_type=ttype,
                condition_idx=ci,
                side='right',
                arg_idx=-1,
            ))
        else:
            params.extend(_extract_indicator_params(cond.right, ci, 'right'))

    # Exit parameters
    if strategy.tp_atr_mult > 0:
        params.append(ParamSpec(
            name='tp_mult', value=strategy.tp_atr_mult,
            bounds=TP_BOUNDS, param_type='tp',
        ))
    params.append(ParamSpec(
        name='sl_mult', value=strategy.sl_atr_mult,
        bounds=SL_BOUNDS, param_type='sl',
    ))
    if strategy.trail_atr_mult > 0:
        params.append(ParamSpec(
            name='trail_mult', value=strategy.trail_atr_mult,
            bounds=TRAIL_BOUNDS, param_type='trail',
        ))

    return params


def tighten_bounds(params: List[ParamSpec]) -> List[ParamSpec]:
    """
    Tighten parameter bounds to be centered around original GE-discovered values.

    Global bounds (e.g. period [2, 100]) allow CMA-ES to wander too far.
    Adaptive bounds constrain search to a neighborhood of the original,
    treating CMA-ES as a LOCAL optimizer rather than a global explorer.

    Rules:
      - period:         ±60% of original, min range 4, clamp to [2, 100]
      - std:            ±40% of original, clamp to [0.5, 5.0]
      - osc_threshold:  ±25 points, clamp to [0, 100]
      - norm_threshold: ±2.0, clamp to [-5, 5]
      - tp:             ±50% of original, clamp to [0.5, 15]
      - sl:             ±50% of original, clamp to [0.2, 5]
      - trail:          ±50% of original, clamp to [0.5, 10]
    """
    tight = []
    for p in params:
        lo, hi = p.bounds  # global bounds as safety clamp

        if p.param_type == 'period':
            margin = max(p.value * 0.6, 2)  # ±60%, minimum margin of 2
            new_lo = max(lo, p.value - margin)
            new_hi = min(hi, p.value + margin)
        elif p.param_type == 'std':
            margin = max(p.value * 0.4, 0.3)
            new_lo = max(lo, p.value - margin)
            new_hi = min(hi, p.value + margin)
        elif p.param_type == 'osc_threshold':
            new_lo = max(lo, p.value - 25)
            new_hi = min(hi, p.value + 25)
        elif p.param_type == 'norm_threshold':
            new_lo = max(lo, p.value - 2.0)
            new_hi = min(hi, p.value + 2.0)
        elif p.param_type in ('tp', 'sl', 'trail'):
            margin = max(p.value * 0.5, 0.3)
            new_lo = max(lo, p.value - margin)
            new_hi = min(hi, p.value + margin)
        else:
            new_lo, new_hi = lo, hi

        tight.append(ParamSpec(
            name=p.name,
            value=p.value,
            bounds=(new_lo, new_hi),
            param_type=p.param_type,
            condition_idx=p.condition_idx,
            side=p.side,
            arg_idx=p.arg_idx,
        ))
    return tight


def _rebuild_indicator(indicator_str: str, new_values: dict,
                       condition_idx: int, side: str) -> str:
    """Rebuild an indicator string with new parameter values."""
    parsed = _parse_indicator(indicator_str)
    if parsed is None:
        return indicator_str

    func_name, args = parsed
    sig = INDICATOR_SIGS.get(func_name)
    if sig is None:
        return indicator_str

    # Detect timeframe suffix
    has_tf = args[-1] in TIMEFRAMES if args else False
    actual_args = args[:-1] if has_tf else args[:]
    tf_arg = args[-1] if has_tf else None

    new_args = actual_args[:]
    for i, (arg_name, arg_type) in enumerate(sig):
        if i >= len(new_args):
            break
        if arg_type == 'source':
            continue

        key = f"c{condition_idx}_{side}_{func_name}_{arg_name}"
        if key in new_values:
            val = new_values[key]
            if arg_type == 'period':
                new_args[i] = str(max(2, round(val)))
            elif arg_type == 'std':
                new_args[i] = f"{val:.1f}"

    result = f"{func_name}({', '.join(new_args)}"
    if tf_arg:
        result += f", {tf_arg}"
    result += ")"
    return result


def rebuild_strategy(strategy: Strategy, param_vector: List[float],
                     param_specs: List[ParamSpec]) -> Strategy:
    """
    Create a new Strategy with modified parameters from CMA-ES vector.

    The original strategy structure (which indicators, which comparators,
    which logic connectives, direction) is preserved. Only numeric parameters
    are modified.
    """
    # Build name→value mapping
    new_values = {}
    for spec, val in zip(param_specs, param_vector):
        # Clamp to bounds
        val = max(spec.bounds[0], min(spec.bounds[1], val))
        new_values[spec.name] = val

    # Deep copy conditions
    new_conditions = []
    for ci, cond in enumerate(strategy.conditions):
        new_left = cond.left
        new_right = cond.right

        # Rebuild left
        if _is_threshold(cond.left):
            key = f"c{ci}_left_threshold"
            if key in new_values:
                val = new_values[key]
                # Round nicely
                new_left = f"{val:.1f}" if abs(val) < 10 else str(round(val))
        else:
            new_left = _rebuild_indicator(cond.left, new_values, ci, 'left')

        # Rebuild right
        if _is_threshold(cond.right):
            key = f"c{ci}_right_threshold"
            if key in new_values:
                val = new_values[key]
                new_right = f"{val:.1f}" if abs(val) < 10 else str(round(val))
        else:
            new_right = _rebuild_indicator(cond.right, new_values, ci, 'right')

        new_conditions.append(Condition(new_left, cond.comparator, new_right))

    # Exit parameters
    tp = new_values.get('tp_mult', strategy.tp_atr_mult)
    sl = new_values.get('sl_mult', strategy.sl_atr_mult)
    trail = new_values.get('trail_mult', strategy.trail_atr_mult)

    # Build new expression string
    cond_strs = []
    for c in new_conditions:
        comp = c.comparator
        # Use _GT_ / _LT_ style for expression_raw to match original format
        if comp == '>':
            comp = '_GT_'
        elif comp == '<':
            comp = '_LT_'
        cond_strs.append(f"{c.left} {comp} {c.right}")

    logic_display = strategy.logic
    for i, cs in enumerate(cond_strs):
        logic_display = logic_display.replace(f"c{i}", cs)
    logic_display = logic_display.replace(' AND ', ' AND ').replace(' OR ', ' OR ')

    # Reconstruct expression_raw
    exit_parts = []
    if tp > 0:
        exit_parts.append(f"TP={tp:.1f}")
    exit_parts.append(f"SL={sl:.1f}")
    if trail > 0:
        exit_parts.append(f"TRAIL={trail:.1f}")
    exit_str = " ".join(exit_parts)

    expr_raw = f"{strategy.direction} WHEN {logic_display} EXIT {exit_str}"

    return Strategy(
        genome=strategy.genome,  # Keep original genome for reference
        direction=strategy.direction,
        conditions=new_conditions,
        logic=strategy.logic,
        tp_atr_mult=round(tp, 2),
        sl_atr_mult=round(sl, 2),
        trail_atr_mult=round(trail, 2),
        expression_raw=expr_raw,
        n_nodes=strategy.n_nodes,
        codons_used=strategy.codons_used,
        wrapping_count=strategy.wrapping_count,
    )
