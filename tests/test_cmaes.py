"""Tests for CMA-ES parameter extraction and strategy reconstruction."""

import pytest
from strategy.phenotype import Strategy, Condition
from evolution.param_extractor import (
    extract_params, rebuild_strategy, _parse_indicator, _is_threshold,
    _classify_threshold, ParamSpec, tighten_bounds
)


def _make_strategy(conditions, direction='LONG', logic=None,
                   tp=2.0, sl=1.0, trail=0.0):
    """Helper to build a Strategy from condition tuples."""
    conds = [Condition(left=l, comparator=c, right=r) for l, c, r in conditions]
    if logic is None:
        logic = ' AND '.join(f'c{i}' for i in range(len(conds)))
    return Strategy(
        genome=[0] * 50, direction=direction, conditions=conds,
        logic=logic, tp_atr_mult=tp, sl_atr_mult=sl, trail_atr_mult=trail,
        expression_raw='test', n_nodes=len(conds),
    )


class TestParseIndicator:
    def test_rsi(self):
        result = _parse_indicator("RSI(close, 14)")
        assert result == ('RSI', ['close', '14'])

    def test_macd_norm_with_tf(self):
        result = _parse_indicator("MACD_NORM(8, 21, 9, 1h)")
        assert result == ('MACD_NORM', ['8', '21', '9', '1h'])

    def test_stoch_k(self):
        result = _parse_indicator("STOCH_K(5, 4h)")
        assert result == ('STOCH_K', ['5', '4h'])

    def test_literal(self):
        assert _parse_indicator("30") is None
        assert _parse_indicator("-0.5") is None

    def test_pct_b(self):
        result = _parse_indicator("PCT_B(14, 2.5)")
        assert result == ('PCT_B', ['14', '2.5'])


class TestIsThreshold:
    def test_integers(self):
        assert _is_threshold("30") is True
        assert _is_threshold("0") is True

    def test_floats(self):
        assert _is_threshold("-0.5") is True
        assert _is_threshold("1.5") is True

    def test_indicators(self):
        assert _is_threshold("RSI(close, 14)") is False
        assert _is_threshold("STOCH_K(5)") is False


class TestClassifyThreshold:
    def test_osc_threshold(self):
        cond = Condition("RSI(close, 14)", ">", "30")
        assert _classify_threshold("30", cond) == 'osc_threshold'

    def test_norm_threshold(self):
        cond = Condition("MACD_NORM(8, 21, 9)", "<", "-0.5")
        assert _classify_threshold("-0.5", cond) == 'norm_threshold'

    def test_osc_threshold_left(self):
        cond = Condition("70", "<", "ADX(14)")
        assert _classify_threshold("70", cond) == 'osc_threshold'


class TestExtractParams:
    def test_simple_rsi_threshold(self):
        """RSI(close, 14) > 30 → period + threshold"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        names = [p.name for p in params]
        assert 'c0_left_RSI_period' in names
        assert 'c0_right_threshold' in names
        # Also exit params
        assert 'tp_mult' in names
        assert 'sl_mult' in names

    def test_rsi_period_value(self):
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        period_param = next(p for p in params if p.name == 'c0_left_RSI_period')
        assert period_param.value == 14.0
        assert period_param.param_type == 'period'

    def test_macd_norm_three_periods(self):
        """MACD_NORM(8, 21, 9, 1h) → 3 period params"""
        s = _make_strategy([("MACD_NORM(8, 21, 9, 1h)", "<", "-0.5")])
        params = extract_params(s)
        period_names = [p.name for p in params if p.param_type == 'period']
        assert len(period_names) == 3
        assert 'c0_left_MACD_NORM_fast' in period_names
        assert 'c0_left_MACD_NORM_slow' in period_names
        assert 'c0_left_MACD_NORM_signal' in period_names

    def test_pct_b_period_and_std(self):
        """PCT_B(14, 2.5) → period + std_dev params"""
        s = _make_strategy([("PCT_B(14, 2.5)", ">", "0.5")])
        params = extract_params(s)
        types = {p.name: p.param_type for p in params}
        assert types.get('c0_left_PCT_B_period') == 'period'
        assert types.get('c0_left_PCT_B_std_dev') == 'std'

    def test_indicator_vs_indicator(self):
        """RSI(close, 7) > STOCH_K(5) → both sides have periods"""
        s = _make_strategy([("RSI(close, 7)", ">", "STOCH_K(5)")])
        params = extract_params(s)
        names = [p.name for p in params if 'period' in p.name]
        assert 'c0_left_RSI_period' in names
        assert 'c0_right_STOCH_K_period' in names

    def test_multi_condition(self):
        """Two conditions → params from both"""
        s = _make_strategy([
            ("RSI(close, 14)", ">", "30"),
            ("MACD_NORM(8, 21, 9)", "<", "0.5"),
        ])
        params = extract_params(s)
        assert any(p.name.startswith('c0_') for p in params)
        assert any(p.name.startswith('c1_') for p in params)

    def test_exit_params(self):
        """Exit multipliers included"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")],
                           tp=3.0, sl=1.5, trail=2.0)
        params = extract_params(s)
        tp = next(p for p in params if p.name == 'tp_mult')
        sl = next(p for p in params if p.name == 'sl_mult')
        trail = next(p for p in params if p.name == 'trail_mult')
        assert tp.value == 3.0
        assert sl.value == 1.5
        assert trail.value == 2.0

    def test_no_trail_when_zero(self):
        """Trail=0 → no trail parameter"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")],
                           tp=2.0, sl=1.0, trail=0.0)
        params = extract_params(s)
        assert not any(p.name == 'trail_mult' for p in params)

    def test_no_tp_when_zero(self):
        """TP=0 → no TP parameter"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")],
                           tp=0.0, sl=1.0)
        params = extract_params(s)
        assert not any(p.name == 'tp_mult' for p in params)

    def test_source_not_extracted(self):
        """Source args (close, open) are NOT parameters"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        assert not any('source' in p.name for p in params)

    def test_timeframe_not_extracted(self):
        """Timeframe args (1h, 4h) are NOT parameters"""
        s = _make_strategy([("RSI(close, 14, 1h)", ">", "30")])
        params = extract_params(s)
        assert not any('timeframe' in p.name or 'tf' in p.name for p in params)
        # But period IS extracted
        assert any(p.name == 'c0_left_RSI_period' for p in params)


class TestRebuildStrategy:
    def test_roundtrip_unchanged(self):
        """Rebuilding with same values → same conditions"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")],
                           tp=2.0, sl=1.0)
        params = extract_params(s)
        values = [p.value for p in params]
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.conditions[0].left == "RSI(close, 14)"
        assert rebuilt.conditions[0].right == "30"  # Integer thresholds stay clean
        assert rebuilt.tp_atr_mult == 2.0
        assert rebuilt.sl_atr_mult == 1.0

    def test_period_change(self):
        """Changing RSI period from 14 to 21"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        values = [p.value for p in params]
        # Find and change the period
        for i, p in enumerate(params):
            if p.name == 'c0_left_RSI_period':
                values[i] = 21.0
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.conditions[0].left == "RSI(close, 21)"

    def test_threshold_change(self):
        """Changing threshold from 30 to 25"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        values = [p.value for p in params]
        for i, p in enumerate(params):
            if p.name == 'c0_right_threshold':
                values[i] = 25.0
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.conditions[0].right == "25"  # Integer thresholds stay clean

    def test_macd_periods_change(self):
        """Changing MACD fast/slow/signal"""
        s = _make_strategy([("MACD_NORM(8, 21, 9, 1h)", "<", "-0.5")])
        params = extract_params(s)
        values = [p.value for p in params]
        for i, p in enumerate(params):
            if 'fast' in p.name:
                values[i] = 10.0
            elif 'slow' in p.name:
                values[i] = 30.0
            elif 'signal' in p.name:
                values[i] = 7.0
        rebuilt = rebuild_strategy(s, values, params)
        assert "MACD_NORM(10, 30, 7, 1h)" in rebuilt.conditions[0].left

    def test_exit_params_change(self):
        """Changing TP and SL"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")],
                           tp=2.0, sl=1.0, trail=3.0)
        params = extract_params(s)
        values = [p.value for p in params]
        for i, p in enumerate(params):
            if p.name == 'tp_mult':
                values[i] = 4.5
            elif p.name == 'sl_mult':
                values[i] = 0.8
            elif p.name == 'trail_mult':
                values[i] = 2.5
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.tp_atr_mult == 4.5
        assert rebuilt.sl_atr_mult == 0.8
        assert rebuilt.trail_atr_mult == 2.5

    def test_preserves_direction(self):
        s = _make_strategy([("RSI(close, 14)", ">", "30")], direction='SHORT')
        params = extract_params(s)
        values = [p.value for p in params]
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.direction == 'SHORT'

    def test_preserves_logic(self):
        s = _make_strategy([
            ("RSI(close, 14)", ">", "30"),
            ("STOCH_K(5)", "<", "80"),
        ], logic="c0 AND c1")
        params = extract_params(s)
        values = [p.value for p in params]
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.logic == "c0 AND c1"

    def test_preserves_comparator(self):
        s = _make_strategy([("RSI(close, 14)", "CROSSES_ABOVE", "STOCH_K(5)")])
        params = extract_params(s)
        values = [p.value for p in params]
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.conditions[0].comparator == "CROSSES_ABOVE"

    def test_period_rounded_to_int(self):
        """Continuous period values get rounded to nearest integer"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        values = [p.value for p in params]
        for i, p in enumerate(params):
            if p.name == 'c0_left_RSI_period':
                values[i] = 11.7  # CMA-ES might output this
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.conditions[0].left == "RSI(close, 12)"

    def test_bounds_clamping(self):
        """Values outside bounds get clamped"""
        s = _make_strategy([("RSI(close, 14)", ">", "30")],
                           tp=2.0, sl=1.0)
        params = extract_params(s)
        values = [p.value for p in params]
        for i, p in enumerate(params):
            if p.name == 'sl_mult':
                values[i] = -5.0  # Way below minimum
        rebuilt = rebuild_strategy(s, values, params)
        assert rebuilt.sl_atr_mult == 0.2  # Clamped to minimum


class TestLiveStrategies:
    """Test parameter extraction on actual live strategy expressions."""

    def test_btc_short_macd(self):
        """SHORT WHEN MACD_NORM(8, 21, 9, 1h) < -0.5 EXIT TP=8.0 SL=3.0"""
        s = _make_strategy(
            [("MACD_NORM(8, 21, 9, 1h)", "<", "-0.5")],
            direction='SHORT', tp=8.0, sl=3.0,
        )
        params = extract_params(s)
        names = {p.name for p in params}
        assert 'c0_left_MACD_NORM_fast' in names
        assert 'c0_left_MACD_NORM_slow' in names
        assert 'c0_left_MACD_NORM_signal' in names
        assert 'c0_right_threshold' in names
        assert 'tp_mult' in names
        assert 'sl_mult' in names
        assert len(params) == 6  # 3 MACD periods + threshold + TP + SL

    def test_bnb_short_complex(self):
        """SHORT WHEN ROC(5, 1h) CROSSES_ABOVE 1.0 AND MACD_NORM(8, 26, 9, 1h) > ROC(3, 1h)
        AND RSI(high, 7, 4h) > 10 EXIT TP=5.0 SL=1.0"""
        s = _make_strategy(
            [
                ("ROC(5, 1h)", "CROSSES_ABOVE", "1.0"),
                ("MACD_NORM(8, 26, 9, 1h)", ">", "ROC(3, 1h)"),
                ("RSI(high, 7, 4h)", ">", "10"),
            ],
            direction='SHORT', tp=5.0, sl=1.0,
            logic="c0 AND c1 AND c2",
        )
        params = extract_params(s)
        names = {p.name for p in params}
        # c0: ROC period + threshold
        assert 'c0_left_ROC_period' in names
        assert 'c0_right_threshold' in names
        # c1: MACD 3 periods + ROC period
        assert 'c1_left_MACD_NORM_fast' in names
        assert 'c1_right_ROC_period' in names
        # c2: RSI period + threshold
        assert 'c2_left_RSI_period' in names
        assert 'c2_right_threshold' in names
        # Total: ROC_period + threshold + 3×MACD + ROC_period + RSI_period + threshold + TP + SL
        assert len(params) == 10


class TestTightenBounds:
    """Test adaptive bounds for CMA-ES local optimization."""

    def test_period_bounds_centered(self):
        """Period bounds should be ±60% of original."""
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        tight = tighten_bounds(params)
        period = next(p for p in tight if p.name == 'c0_left_RSI_period')
        # 14 ± 60% = 14 ± 8.4 → [5.6, 22.4], clamped to [2, 100]
        assert period.bounds[0] == pytest.approx(5.6, abs=0.1)
        assert period.bounds[1] == pytest.approx(22.4, abs=0.1)

    def test_small_period_minimum_margin(self):
        """Small periods still get minimum margin of 2."""
        s = _make_strategy([("ROC(3)", ">", "0.5")])
        params = extract_params(s)
        tight = tighten_bounds(params)
        period = next(p for p in tight if 'period' in p.name)
        # 3 ± max(3*0.6=1.8, 2) = 3 ± 2 → [1, 5], clamped to [2, 5]
        assert period.bounds[0] == 2.0  # Clamped to global min
        assert period.bounds[1] == 5.0

    def test_osc_threshold_bounds(self):
        """Oscillator threshold bounds: ±25 points."""
        s = _make_strategy([("RSI(close, 14)", ">", "30")])
        params = extract_params(s)
        tight = tighten_bounds(params)
        thresh = next(p for p in tight if p.name == 'c0_right_threshold')
        assert thresh.bounds[0] == pytest.approx(5.0, abs=0.1)  # 30-25
        assert thresh.bounds[1] == pytest.approx(55.0, abs=0.1)  # 30+25

    def test_norm_threshold_bounds(self):
        """Normalized threshold bounds: ±2.0."""
        s = _make_strategy([("MACD_NORM(8, 21, 9)", "<", "-0.5")])
        params = extract_params(s)
        tight = tighten_bounds(params)
        thresh = next(p for p in tight if p.name == 'c0_right_threshold')
        assert thresh.bounds[0] == pytest.approx(-2.5, abs=0.1)  # -0.5-2
        assert thresh.bounds[1] == pytest.approx(1.5, abs=0.1)   # -0.5+2

    def test_exit_bounds(self):
        """TP/SL bounds: ±50% of original."""
        s = _make_strategy([("RSI(close, 14)", ">", "30")],
                           tp=8.0, sl=3.0, trail=2.0)
        params = extract_params(s)
        tight = tighten_bounds(params)
        tp = next(p for p in tight if p.name == 'tp_mult')
        sl = next(p for p in tight if p.name == 'sl_mult')
        trail = next(p for p in tight if p.name == 'trail_mult')
        # TP=8 ± 4 → [4, 12]
        assert tp.bounds[0] == pytest.approx(4.0, abs=0.1)
        assert tp.bounds[1] == pytest.approx(12.0, abs=0.1)
        # SL=3 ± 1.5 → [1.5, 4.5]
        assert sl.bounds[0] == pytest.approx(1.5, abs=0.1)
        assert sl.bounds[1] == pytest.approx(4.5, abs=0.1)

    def test_bounds_respect_global_limits(self):
        """Tight bounds should never exceed global limits."""
        s = _make_strategy([("RSI(close, 14)", ">", "95")])
        params = extract_params(s)
        tight = tighten_bounds(params)
        thresh = next(p for p in tight if p.name == 'c0_right_threshold')
        # 95 + 25 = 120, but clamped to 100
        assert thresh.bounds[1] == 100.0

    def test_values_preserved(self):
        """Original values should be preserved after tightening."""
        s = _make_strategy([("RSI(close, 14)", ">", "30")], tp=5.0, sl=2.0)
        params = extract_params(s)
        tight = tighten_bounds(params)
        for orig, t in zip(params, tight):
            assert orig.value == t.value
            assert orig.name == t.name
            assert orig.param_type == t.param_type

    def test_macd_periods_tight(self):
        """MACD fast/slow/signal all get adaptive bounds."""
        s = _make_strategy([("MACD_NORM(8, 21, 9, 1h)", "<", "-0.5")])
        params = extract_params(s)
        tight = tighten_bounds(params)
        fast = next(p for p in tight if 'fast' in p.name)
        slow = next(p for p in tight if 'slow' in p.name)
        signal = next(p for p in tight if 'signal' in p.name)
        # fast=8 ± 4.8 → [3.2, 12.8]
        assert fast.bounds[0] < 8.0
        assert fast.bounds[1] > 8.0
        # slow=21 ± 12.6 → [8.4, 33.6]
        assert slow.bounds[0] < 21.0
        assert slow.bounds[1] > 21.0
        # Slow should have wider range than fast
        assert (slow.bounds[1] - slow.bounds[0]) > (fast.bounds[1] - fast.bounds[0])
