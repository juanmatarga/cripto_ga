"""Tests for CPCV, PBO, DSR, and signal permutation."""

import random
import numpy as np
import pandas as pd
import pytest

from grammar.mapper import decode
from strategy.parameters import random_genome
from strategy.phenotype import Condition, Strategy
from validation.cpcv import (
    create_cpcv_groups, generate_cpcv_splits,
    apply_purge_embargo, cpcv_evaluate
)
from validation.pbo import calculate_pbo
from validation.deflated_sharpe import deflated_sharpe_ratio, expected_max_sharpe
from validation.signal_permutation import signal_permutation_test


@pytest.fixture
def sample_df():
    """6 months of synthetic 15m OHLCV data."""
    np.random.seed(42)
    n = 17280
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


@pytest.fixture
def config():
    return {
        'costs': {
            'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
            'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
        },
        'exits': {'atr_period': 14},
    }


# ============================================================================
# CPCV
# ============================================================================

class TestCPCVGroups:
    def test_correct_number_of_groups(self, sample_df):
        groups = create_cpcv_groups(sample_df, 10)
        assert len(groups) == 10

    def test_groups_cover_all_data(self, sample_df):
        groups = create_cpcv_groups(sample_df, 10)
        total_bars = sum(len(g) for g in groups)
        assert total_bars == len(sample_df)

    def test_groups_are_contiguous(self, sample_df):
        groups = create_cpcv_groups(sample_df, 5)
        for i in range(len(groups) - 1):
            last_of_current = groups[i].index[-1]
            first_of_next = groups[i + 1].index[0]
            # Next group should start right after current
            assert first_of_next > last_of_current


class TestCPCVSplits:
    def test_correct_number_of_splits(self):
        splits = generate_cpcv_splits(10, k=5)
        # C(10, 5) = 252
        assert len(splits) == 252

    def test_train_test_disjoint(self):
        splits = generate_cpcv_splits(8, k=4)
        for train_idx, test_idx in splits:
            assert set(train_idx).isdisjoint(set(test_idx))

    def test_train_test_cover_all_groups(self):
        splits = generate_cpcv_splits(8, k=4)
        for train_idx, test_idx in splits:
            assert set(train_idx) | set(test_idx) == set(range(8))

    def test_smaller_k(self):
        splits = generate_cpcv_splits(6, k=2)
        # C(6, 2) = 15
        assert len(splits) == 15


class TestPurgeEmbargo:
    def test_purge_removes_end_of_train(self, sample_df):
        groups = create_cpcv_groups(sample_df, 4)
        # Train: [0, 1], Test: [2, 3]
        # Group 1 is before test group 2, so it should be purged
        train_df, test_df = apply_purge_embargo(
            groups, (0, 1), (2, 3), purge_bars=96, embargo_bars=48
        )
        # Train should be shorter than groups[0] + groups[1]
        full_train_len = len(groups[0]) + len(groups[1])
        assert len(train_df) < full_train_len

    def test_embargo_removes_start_of_test(self, sample_df):
        groups = create_cpcv_groups(sample_df, 4)
        train_df, test_df = apply_purge_embargo(
            groups, (0, 1), (2, 3), purge_bars=96, embargo_bars=48
        )
        full_test_len = len(groups[2]) + len(groups[3])
        assert len(test_df) < full_test_len

    def test_no_overlap_after_purge(self, sample_df):
        groups = create_cpcv_groups(sample_df, 4)
        train_df, test_df = apply_purge_embargo(
            groups, (0, 1), (2, 3), purge_bars=96, embargo_bars=48
        )
        # Train and test should not share any timestamps
        train_times = set(train_df.index)
        test_times = set(test_df.index)
        assert train_times.isdisjoint(test_times)


class TestCPCVEvaluate:
    def test_returns_dict_with_required_keys(self, sample_df, config):
        random.seed(42)
        for _ in range(50):
            s = decode(random_genome(50))
            if s is not None:
                result = cpcv_evaluate(
                    s, sample_df, config,
                    n_groups=6, purge_bars=48, embargo_bars=24,
                    max_splits=15
                )
                assert 'oos_sortinos' in result
                assert 'n_splits' in result
                assert 'mean_sortino' in result
                assert 'pct_positive_sortino' in result
                return
        pytest.skip("No valid strategy found")


# ============================================================================
# PBO
# ============================================================================

class TestPBO:
    def test_all_negative_sortinos_gives_pbo_1(self):
        cpcv_results = {'oos_sortinos': [-1.0, -0.5, -2.0, -0.1, -3.0]}
        result = calculate_pbo(cpcv_results)
        assert result['pbo'] == 1.0

    def test_all_positive_sortinos_gives_pbo_0(self):
        cpcv_results = {'oos_sortinos': [1.0, 0.5, 2.0, 0.1, 3.0]}
        result = calculate_pbo(cpcv_results)
        assert result['pbo'] == 0.0

    def test_mixed_sortinos(self):
        cpcv_results = {'oos_sortinos': [1.0, -0.5, 2.0, -0.1, 3.0]}
        result = calculate_pbo(cpcv_results)
        assert 0.0 < result['pbo'] < 1.0
        assert result['pbo'] == 2 / 5  # 2 out of 5 are <= 0

    def test_empty_results(self):
        result = calculate_pbo({'oos_sortinos': []})
        assert result['pbo'] == 1.0

    def test_has_interpretation(self):
        cpcv_results = {'oos_sortinos': [1.0, 2.0, 3.0]}
        result = calculate_pbo(cpcv_results)
        assert 'interpretation' in result
        assert len(result['interpretation']) > 0


# ============================================================================
# DEFLATED SHARPE RATIO
# ============================================================================

class TestDSR:
    def test_high_sharpe_few_trials_high_dsr(self):
        # SR=3.0 with only 10 trials and long sample should be credible
        # Need large T to reduce SR estimator variance at 15m annualization
        result = deflated_sharpe_ratio(
            observed_sharpe=3.0, n_trials=10, T=100000, periods_per_year=35040
        )
        assert result['dsr'] > 0.8

    def test_low_sharpe_many_trials_low_dsr(self):
        # SR=0.5 with 10000 trials — likely data snooping
        result = deflated_sharpe_ratio(
            observed_sharpe=0.5, n_trials=10000, T=5000, periods_per_year=35040
        )
        assert result['dsr'] < 0.5

    def test_more_trials_reduces_dsr(self):
        dsr_10 = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=10, T=5000, periods_per_year=35040
        )['dsr']
        dsr_10000 = deflated_sharpe_ratio(
            observed_sharpe=1.5, n_trials=10000, T=5000, periods_per_year=35040
        )['dsr']
        assert dsr_10 > dsr_10000

    def test_higher_sharpe_increases_dsr(self):
        dsr_low = deflated_sharpe_ratio(
            observed_sharpe=0.5, n_trials=1000, T=5000, periods_per_year=35040
        )['dsr']
        dsr_high = deflated_sharpe_ratio(
            observed_sharpe=3.0, n_trials=1000, T=5000, periods_per_year=35040
        )['dsr']
        assert dsr_high > dsr_low

    def test_expected_max_sharpe_increases_with_trials(self):
        e10 = expected_max_sharpe(10, 5000, periods_per_year=35040)
        e1000 = expected_max_sharpe(1000, 5000, periods_per_year=35040)
        assert e1000 > e10

    def test_expected_max_sharpe_is_positive(self):
        e = expected_max_sharpe(100, 5000, periods_per_year=35040)
        assert e > 0

    def test_returns_interpretation(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=2.0, n_trials=100, T=5000, periods_per_year=35040
        )
        assert 'interpretation' in result


# ============================================================================
# SIGNAL PERMUTATION
# ============================================================================

class TestSignalPermutation:
    def test_returns_required_keys(self, sample_df, config):
        random.seed(42)
        for _ in range(50):
            s = decode(random_genome(50))
            if s is not None:
                result = signal_permutation_test(
                    s, sample_df.iloc[:2880], config,
                    n_permutations=20, seed=42
                )
                assert 'p_value' in result
                assert 'real_sortino' in result
                assert 'n_permutations' in result
                assert 'interpretation' in result
                return
        pytest.skip("No valid strategy found")

    def test_p_value_in_range(self, sample_df, config):
        random.seed(42)
        for _ in range(50):
            s = decode(random_genome(50))
            if s is not None:
                result = signal_permutation_test(
                    s, sample_df.iloc[:2880], config,
                    n_permutations=20, seed=42
                )
                assert 0.0 <= result['p_value'] <= 1.0
                return
        pytest.skip("No valid strategy found")

    def test_few_signals_returns_p1(self, sample_df, config):
        """Strategy with no signals should get p=1.0."""
        s = Strategy(
            genome=[1, 2, 3], direction='LONG', conditions=[],
            logic='', tp_atr_mult=2.0, sl_atr_mult=1.0,
            expression_raw='test', n_nodes=0, codons_used=0,
            wrapping_count=0,
        )
        result = signal_permutation_test(
            s, sample_df.iloc[:2880], config, n_permutations=10
        )
        assert result['p_value'] == 1.0
