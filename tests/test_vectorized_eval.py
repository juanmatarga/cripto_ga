"""Tests for strategy/vectorized_eval.py"""

import random
import time
import numpy as np
import pandas as pd
import pytest

from grammar.mapper import decode
from strategy.parameters import random_genome
from strategy.phenotype import Condition, Strategy
from strategy.vectorized_eval import (
    generate_signals, evaluate_condition, evaluate_logic,
    IndicatorCache, compute_rsi, compute_sma, compute_ema,
    compute_atr, compute_macd, compute_stoch
)


@pytest.fixture
def sample_df():
    """1 month of synthetic 15m OHLCV data."""
    np.random.seed(42)
    n = 2880
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    df = pd.DataFrame({
        'Open': close + np.random.randn(n) * 10,
        'High': close + abs(np.random.randn(n) * 30),
        'Low': close - abs(np.random.randn(n) * 30),
        'Close': close,
        'Volume': np.random.exponential(1000, n),
    }, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    return df


class TestIndicators:
    def test_sma_matches_pandas(self, sample_df):
        result = compute_sma(sample_df['Close'], 20)
        expected = sample_df['Close'].rolling(20).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_ema_length(self, sample_df):
        result = compute_ema(sample_df['Close'], 20)
        assert len(result) == len(sample_df)

    def test_rsi_range(self, sample_df):
        rsi = compute_rsi(sample_df['Close'], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), \
            f"RSI out of range: min={valid.min()}, max={valid.max()}"

    def test_atr_positive(self, sample_df):
        atr = compute_atr(sample_df, 14)
        valid = atr.dropna()
        assert (valid >= 0).all()

    def test_macd_components(self, sample_df):
        line = compute_macd(sample_df['Close'], 12, 26, 'line')
        signal = compute_macd(sample_df['Close'], 12, 26, 'signal', 9)
        hist = compute_macd(sample_df['Close'], 12, 26, 'histogram', 9)
        assert len(line) == len(sample_df)
        assert len(signal) == len(sample_df)
        assert len(hist) == len(sample_df)
        # Histogram should be line - signal (approximately, due to NaN handling)
        valid_idx = line.notna() & signal.notna()
        diff = (hist[valid_idx] - (line[valid_idx] - signal[valid_idx])).abs()
        assert diff.max() < 1e-10

    def test_stoch_range(self, sample_df):
        k = compute_stoch(sample_df, 14, 'k')
        valid = k.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_stoch_d_smoother_than_k(self, sample_df):
        k = compute_stoch(sample_df, 14, 'k')
        d = compute_stoch(sample_df, 14, 'd')
        # D is SMA(3) of K, so should have smaller std
        assert d.std() < k.std()


class TestIndicatorCache:
    def test_raw_sources_cached(self, sample_df):
        cache = IndicatorCache(sample_df)
        pd.testing.assert_series_equal(cache.get('close'), sample_df['Close'])
        pd.testing.assert_series_equal(cache.get('open'), sample_df['Open'])
        pd.testing.assert_series_equal(cache.get('volume'), sample_df['Volume'])

    def test_indicator_cached_on_second_call(self, sample_df):
        cache = IndicatorCache(sample_df)
        r1 = cache.get('RSI(close, 14)')
        r2 = cache.get('RSI(close, 14)')
        assert r1 is r2  # Same object, not recomputed

    def test_numeric_constant(self, sample_df):
        cache = IndicatorCache(sample_df)
        result = cache.get('30')
        assert (result == 30.0).all()
        assert len(result) == len(sample_df)

    def test_sma_via_cache(self, sample_df):
        cache = IndicatorCache(sample_df)
        result = cache.get('SMA(close, 20)')
        expected = compute_sma(sample_df['Close'], 20)
        pd.testing.assert_series_equal(result, expected)


class TestConditionEvaluation:
    def test_gt_condition(self, sample_df):
        cache = IndicatorCache(sample_df)
        cond = Condition(left='close', comparator='>', right='open')
        result = evaluate_condition(cond, cache)
        expected = sample_df['Close'] > sample_df['Open']
        pd.testing.assert_series_equal(result, expected)

    def test_lt_condition(self, sample_df):
        cache = IndicatorCache(sample_df)
        cond = Condition(left='RSI(close, 14)', comparator='<', right='30')
        result = evaluate_condition(cond, cache)
        rsi = compute_rsi(sample_df['Close'], 14)
        expected = rsi < 30
        np.testing.assert_array_equal(result.values, expected.values)

    def test_crosses_above(self, sample_df):
        cache = IndicatorCache(sample_df)
        cond = Condition(
            left='SMA(close, 8)',
            comparator='CROSSES_ABOVE',
            right='SMA(close, 21)'
        )
        result = evaluate_condition(cond, cache)
        assert result.dtype == bool
        # Crosses should be relatively rare
        assert 0 < result.sum() < len(sample_df) // 2

    def test_crosses_below(self, sample_df):
        cache = IndicatorCache(sample_df)
        cond = Condition(
            left='SMA(close, 8)',
            comparator='CROSSES_BELOW',
            right='SMA(close, 21)'
        )
        result = evaluate_condition(cond, cache)
        assert result.dtype == bool
        assert 0 < result.sum() < len(sample_df) // 2


class TestLogicEvaluation:
    def test_single_condition(self, sample_df):
        s1 = pd.Series([True, False, True], index=[0, 1, 2])
        result = evaluate_logic([s1], "c0")
        pd.testing.assert_series_equal(result, s1)

    def test_and_logic(self):
        s0 = pd.Series([True, True, False, False])
        s1 = pd.Series([True, False, True, False])
        result = evaluate_logic([s0, s1], "c0 AND c1")
        expected = pd.Series([True, False, False, False])
        pd.testing.assert_series_equal(result, expected)

    def test_or_logic(self):
        s0 = pd.Series([True, True, False, False])
        s1 = pd.Series([True, False, True, False])
        result = evaluate_logic([s0, s1], "c0 OR c1")
        expected = pd.Series([True, True, True, False])
        pd.testing.assert_series_equal(result, expected)

    def test_mixed_logic(self):
        s0 = pd.Series([True, True, False, False])
        s1 = pd.Series([True, False, True, False])
        s2 = pd.Series([False, True, True, True])
        result = evaluate_logic([s0, s1, s2], "(c0 AND c1) OR c2")
        expected = pd.Series([True, True, True, True])
        pd.testing.assert_series_equal(result, expected)


class TestGenerateSignals:
    def test_returns_bool_series(self, sample_df):
        random.seed(42)
        for _ in range(50):
            s = decode(random_genome(50))
            if s is not None:
                signals = generate_signals(s, sample_df)
                assert signals.dtype == bool
                assert len(signals) == len(sample_df)
                break

    def test_empty_conditions_returns_false(self, sample_df):
        s = Strategy(
            genome=[1, 2, 3],
            direction='LONG',
            conditions=[],
            logic='',
            tp_atr_mult=2.0,
            sl_atr_mult=1.0,
            expression_raw='test',
            n_nodes=0,
            codons_used=0,
            wrapping_count=0,
        )
        signals = generate_signals(s, sample_df)
        assert signals.sum() == 0

    def test_benchmark_1000_strategies(self, sample_df):
        """1000 strategies evaluated on 1 month of data in < 10 seconds."""
        random.seed(42)
        genomes = [random_genome(50) for _ in range(1000)]
        strategies = [decode(g) for g in genomes]
        valid = [s for s in strategies if s is not None]

        assert len(valid) >= 600, f"Too few valid strategies: {len(valid)}"

        t0 = time.time()
        for s in valid:
            generate_signals(s, sample_df)
        elapsed = time.time() - t0

        assert elapsed < 10.0, f"Too slow: {elapsed:.1f}s for {len(valid)} strategies"
