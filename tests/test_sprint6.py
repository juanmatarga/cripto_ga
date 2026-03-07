"""
Tests for Sprint 6: Multi-Timeframe + Alternative Data compute functions.

Grammar v5b: alt data removed from grammar (proven noise), but compute
functions kept in vectorized_eval.py for potential future use.
"""

import random
import numpy as np
import pandas as pd
import pytest

from grammar.bnf import GRAMMAR, validate_grammar
from grammar.mapper import decode
from strategy.parameters import random_genome
from strategy.vectorized_eval import (
    IndicatorCache, generate_signals,
    compute_funding_zscore, compute_oi_change, compute_oi_price_divergence,
    compute_ls_ratio_change, compute_taker_imbalance,
)
from data.multi_timeframe import (
    resample_ohlcv, prepare_multi_tf_data, align_higher_tf_to_15m,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def df_15m():
    """Create realistic 15m OHLCV data for testing (1000 bars = ~10 days)."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    price = 40000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        'Open': price,
        'High': price + np.abs(np.random.randn(n) * 30),
        'Low': price - np.abs(np.random.randn(n) * 30),
        'Close': price + np.random.randn(n) * 20,
        'Volume': np.abs(np.random.randn(n) * 100) + 50,
    }, index=dates)
    return df


@pytest.fixture
def df_with_alt(df_15m):
    """15m OHLCV data with alternative data columns."""
    n = len(df_15m)
    np.random.seed(42)
    df = df_15m.copy()
    df['funding_rate'] = np.random.randn(n) * 0.001 + 0.0001
    df['open_interest'] = np.abs(np.random.randn(n) * 10000) + 50000
    df['oi_value'] = df['open_interest'] * df['Close']
    df['ls_ratio'] = np.random.randn(n) * 0.1 + 1.0
    df['long_account'] = 0.5 + np.random.randn(n) * 0.05
    df['short_account'] = 1.0 - df['long_account']
    df['taker_buy_sell_ratio'] = np.abs(np.random.randn(n) * 0.2) + 0.9
    df['buy_vol'] = np.abs(np.random.randn(n) * 1000) + 5000
    df['sell_vol'] = np.abs(np.random.randn(n) * 1000) + 5000
    return df


# ============================================================================
# Grammar Tests
# ============================================================================

class TestGrammarV5b:
    """Test the v5b grammar with multi-timeframe (alt data removed)."""

    def test_grammar_validates(self):
        validate_grammar()

    def test_no_alt_in_grammar(self):
        """Alt data indicators should NOT be in grammar (proven noise)."""
        assert '<alt>' not in GRAMMAR
        assert '<alt_thresh>' not in GRAMMAR
        assert '<zscore_period>' not in GRAMMAR
        conditions = GRAMMAR['<condition>']
        alt_conditions = [c for c in conditions if '<alt>' in c]
        assert len(alt_conditions) == 0

    def test_timeframe_in_grammar(self):
        tf = GRAMMAR['<timeframe>']
        assert '15m' in tf
        assert '1h' in tf
        assert '4h' in tf

    def test_osc_with_timeframe_in_grammar(self):
        osc = GRAMMAR['<osc>']
        htf_osc = [o for o in osc if '<timeframe>' in o]
        assert len(htf_osc) >= 5  # RSI, STOCH_K, STOCH_D, ADX, MFI

    def test_norm_with_timeframe_in_grammar(self):
        norm = GRAMMAR['<norm>']
        htf_norm = [n for n in norm if '<timeframe>' in n]
        assert len(htf_norm) >= 7  # PCT_B, MACD_NORM, PRICE_POS, ROC, VOL_RATIO, BBWIDTH, ATR_PCT


# ============================================================================
# Decode Tests
# ============================================================================

class TestDecodeV5b:
    """Test decoding genomes with multi-timeframe grammar."""

    def test_decode_with_timeframe(self):
        """Some genomes should decode to strategies with HTF indicators."""
        random.seed(123)
        htf_found = False
        for _ in range(200):
            g = random_genome(50)
            s = decode(g)
            if s is not None and s.expression_raw:
                if ', 1h)' in s.expression_raw or ', 4h)' in s.expression_raw:
                    htf_found = True
                    break
        assert htf_found, "No HTF indicators found in 200 decoded genomes"

    def test_decode_rate_still_reasonable(self):
        """Grammar expansion shouldn't dramatically reduce decode success rate."""
        random.seed(42)
        valid = sum(1 for _ in range(500)
                    if decode(random_genome(50)) is not None)
        # Should be at least 50% success rate
        assert valid >= 250, f"Only {valid}/500 genomes decoded — too low"

    def test_no_alt_in_decoded_strategies(self):
        """No decoded strategy should contain alt data indicators."""
        random.seed(42)
        for _ in range(500):
            g = random_genome(50)
            s = decode(g)
            if s is not None and s.expression_raw:
                for alt in ['FUNDING_ZSCORE', 'OI_CHANGE', 'OI_PRICE_DIV',
                            'LS_RATIO_CHANGE', 'TAKER_IMBALANCE']:
                    assert alt not in s.expression_raw, \
                        f"Alt indicator {alt} found in decoded strategy: {s.expression_raw}"


# ============================================================================
# Alternative Data Indicator Tests (compute functions still work)
# ============================================================================

class TestAltIndicators:
    """Test alternative data indicator computations (kept for potential future use)."""

    def test_funding_zscore(self, df_with_alt):
        result = compute_funding_zscore(df_with_alt, period=48)
        assert len(result) == len(df_with_alt)
        valid = result.dropna()
        assert len(valid) > 0
        assert abs(valid.mean()) < 1.0

    def test_funding_zscore_missing_data(self, df_15m):
        result = compute_funding_zscore(df_15m, period=48)
        assert result.isna().all()

    def test_oi_change(self, df_with_alt):
        result = compute_oi_change(df_with_alt, period=8)
        assert len(result) == len(df_with_alt)
        valid = result.dropna()
        assert len(valid) > 0

    def test_oi_change_missing_data(self, df_15m):
        result = compute_oi_change(df_15m, period=8)
        assert result.isna().all()

    def test_oi_price_divergence(self, df_with_alt):
        result = compute_oi_price_divergence(df_with_alt, period=8)
        assert len(result) == len(df_with_alt)
        valid = result.dropna()
        assert len(valid) > 0

    def test_ls_ratio_change(self, df_with_alt):
        result = compute_ls_ratio_change(df_with_alt, period=8)
        assert len(result) == len(df_with_alt)
        valid = result.dropna()
        assert len(valid) > 0

    def test_taker_imbalance(self, df_with_alt):
        result = compute_taker_imbalance(df_with_alt, period=8)
        assert len(result) == len(df_with_alt)
        valid = result.dropna()
        assert len(valid) > 0

    def test_taker_imbalance_missing_data(self, df_15m):
        result = compute_taker_imbalance(df_15m, period=8)
        assert result.isna().all()


# ============================================================================
# Multi-Timeframe Tests
# ============================================================================

class TestMultiTimeframe:
    """Test multi-timeframe data preparation and alignment."""

    def test_resample_to_1h(self, df_15m):
        df_1h = resample_ohlcv(df_15m, '1h')
        assert len(df_1h) <= len(df_15m) // 4 + 1
        assert len(df_1h) > 0
        first_hour = df_15m.iloc[:4]
        assert abs(df_1h['Volume'].iloc[0] - first_hour['Volume'].sum()) < 1e-6

    def test_resample_to_4h(self, df_15m):
        df_4h = resample_ohlcv(df_15m, '4h')
        assert len(df_4h) <= len(df_15m) // 16 + 1
        assert len(df_4h) > 0

    def test_prepare_multi_tf_data(self, df_15m):
        tf_data = prepare_multi_tf_data(df_15m)
        assert '15m' in tf_data
        assert '1h' in tf_data
        assert '4h' in tf_data
        assert len(tf_data['15m']) == len(df_15m)

    def test_align_no_lookahead(self, df_15m):
        """Higher-TF values should not appear before the HTF candle closes."""
        df_1h = resample_ohlcv(df_15m, '1h')
        indicator_1h = df_1h['Close'].rolling(2).mean()
        aligned = align_higher_tf_to_15m(indicator_1h, df_15m.index, '1h')
        assert aligned.iloc[:4].isna().all()

    def test_unsupported_timeframe_raises(self, df_15m):
        with pytest.raises(ValueError):
            resample_ohlcv(df_15m, '2h')

    def test_resample_with_alt_data(self, df_with_alt):
        """Alternative data columns should be properly resampled."""
        df_1h = resample_ohlcv(df_with_alt, '1h')
        if 'funding_rate' in df_1h.columns:
            assert not df_1h['funding_rate'].isna().all()


# ============================================================================
# IndicatorCache with Multi-TF
# ============================================================================

class TestIndicatorCacheV5b:
    """Test IndicatorCache with multi-timeframe."""

    def test_cache_basic_still_works(self, df_15m):
        cache = IndicatorCache(df_15m)
        rsi = cache.get('RSI(close, 14)')
        assert len(rsi) == len(df_15m)
        assert not rsi.dropna().empty

    def test_cache_with_timeframe(self, df_15m):
        tf_data = prepare_multi_tf_data(df_15m)
        cache = IndicatorCache(df_15m, tf_data=tf_data)
        rsi_1h = cache.get('RSI(close, 14, 1h)')
        assert len(rsi_1h) == len(df_15m)
        rsi_15m = cache.get('RSI(close, 14)')
        assert rsi_1h.isna().sum() >= rsi_15m.isna().sum()

    def test_cache_alt_data_still_computes(self, df_with_alt):
        """Alt compute functions still work via cache (just not in grammar)."""
        cache = IndicatorCache(df_with_alt)
        fz = cache.get('FUNDING_ZSCORE(48)')
        assert len(fz) == len(df_with_alt)
        assert not fz.dropna().empty

    def test_cache_alt_data_missing_graceful(self, df_15m):
        cache = IndicatorCache(df_15m)
        fz = cache.get('FUNDING_ZSCORE(48)')
        assert fz.isna().all()


# ============================================================================
# Signal Generation with MTF
# ============================================================================

class TestSignalGenerationV5b:
    """Test signal generation with multi-timeframe."""

    def test_signals_with_htf_strategy(self, df_15m):
        """A strategy using HTF indicators should generate signals."""
        tf_data = prepare_multi_tf_data(df_15m)
        random.seed(200)
        for _ in range(500):
            g = random_genome(50)
            s = decode(g)
            if s and (', 1h)' in s.expression_raw or ', 4h)' in s.expression_raw):
                signals = generate_signals(s, df_15m, tf_data=tf_data)
                assert len(signals) == len(df_15m)
                assert signals.dtype == bool
                return
        pytest.skip("No HTF strategy decoded in 500 attempts")

    def test_signals_backward_compatible(self, df_15m):
        """Strategies without HTF should still work normally."""
        random.seed(42)
        for _ in range(100):
            g = random_genome(50)
            s = decode(g)
            if s and '1h' not in s.expression_raw and '4h' not in s.expression_raw:
                signals = generate_signals(s, df_15m)
                assert len(signals) == len(df_15m)
                return
        pytest.fail("Couldn't find a basic strategy")


# ============================================================================
# Integration: Full Pipeline with MTF
# ============================================================================

class TestIntegration:
    """Integration tests for full evaluation pipeline."""

    def test_evaluate_strategy_with_alt_data(self, df_with_alt):
        """Strategy evaluation should work with alternative data columns present."""
        from evolution.fitness import evaluate_strategy
        from backtest.sampling import sample_evolution_windows

        random.seed(42)
        np.random.seed(42)
        windows = sample_evolution_windows(df_with_alt, n_windows=3, window_bars=200)
        config = {
            'costs': {'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
                      'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0},
            'exits': {'atr_period': 14},
            'fitness': {'min_trades': 1, 'max_signal_rate': 0.50,
                        'min_win_rate': 0.0},
        }

        for _ in range(100):
            g = random_genome(50)
            s = decode(g)
            if s:
                evaluate_strategy(s, windows, config)
                assert s.fitness is not None
                return
        pytest.fail("No strategy decoded")

    def test_multi_tf_evaluation_no_crash(self, df_15m):
        """Evaluation with multi-TF data should not crash."""
        from evolution.fitness import _run_single_window
        from data.multi_timeframe import prepare_multi_tf_data

        tf_data = prepare_multi_tf_data(df_15m)
        costs = {'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
                 'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0}

        random.seed(42)
        for _ in range(50):
            g = random_genome(50)
            s = decode(g)
            if s:
                equity, trades = _run_single_window(s, df_15m, costs, 14,
                                                     tf_data=tf_data)
                assert len(equity) == len(df_15m)
                return
        pytest.fail("No strategy decoded")
