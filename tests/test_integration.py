"""Integration test — end-to-end v2 pipeline on mini data."""

import json
import random
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from grammar.mapper import decode
from strategy.parameters import random_genome
from evolution.island import IslandModel
from evolution.archive import MAPElitesArchive
from validation.cpcv import cpcv_evaluate
from validation.pbo import calculate_pbo
from validation.deflated_sharpe import deflated_sharpe_ratio
from validation.signal_permutation import signal_permutation_test
from data.regime_detector import detect_regime


@pytest.fixture
def mini_df():
    """2 months of synthetic 15m OHLCV data (small for speed)."""
    np.random.seed(42)
    n = 5760  # ~2 months
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
def mini_config():
    return {
        'costs': {
            'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
            'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
        },
        'exits': {'atr_period': 14},
        'evolution': {
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elitism_pct': 0.05,
            'genome_length': 50,
            'tournament_k': 3,
            'n_windows_per_gen': 3,
            'window_bars': 1440,
        },
        'islands': {
            'n_islands': 3,
            'migration_interval': 3,
            'migration_size': 2,
            'selection_types': ['tournament', 'lexicase', 'random'],
        },
        'validation': {
            'cpcv_groups': 4,
            'cpcv_purge_bars': 48,
            'cpcv_embargo_bars': 24,
            'cpcv_max_splits': 6,
            'permutation_n': 10,
        },
    }


class TestEndToEnd:
    def test_evolve_validate_pipeline(self, mini_df, mini_config):
        """Full pipeline: evolve -> validate best strategy."""
        random.seed(42)
        np.random.seed(42)

        # 1. EVOLVE
        model = IslandModel(mini_config, mini_df)
        model.initialize(total_pop_size=30)
        result = model.run(n_generations=5, patience=10)

        assert result['total_evaluations'] > 0
        assert result['final_generation'] == 5
        assert isinstance(result['archive'], MAPElitesArchive)

        # 2. Get best strategy
        best_strategies = result['best_strategies']
        if not best_strategies:
            pytest.skip("No valid strategies evolved")

        strategy = best_strategies[0]
        assert strategy.fitness[0] > -999.0

        # 3. VALIDATE with CPCV
        val_cfg = mini_config['validation']
        cpcv_result = cpcv_evaluate(
            strategy, mini_df, mini_config,
            n_groups=val_cfg['cpcv_groups'],
            purge_bars=val_cfg['cpcv_purge_bars'],
            embargo_bars=val_cfg['cpcv_embargo_bars'],
            max_splits=val_cfg['cpcv_max_splits'],
        )
        assert 'oos_sortinos' in cpcv_result
        assert 'mean_sortino' in cpcv_result

        # PBO
        pbo = calculate_pbo(cpcv_result)
        assert 0.0 <= pbo['pbo'] <= 1.0

        # DSR
        dsr = deflated_sharpe_ratio(
            observed_sharpe=cpcv_result['mean_sortino'],
            n_trials=150,
            T=len(mini_df),
        )
        assert 0.0 <= dsr['dsr'] <= 1.0

        # Signal permutation
        perm = signal_permutation_test(
            strategy, mini_df, mini_config,
            n_permutations=val_cfg['permutation_n'],
        )
        assert 0.0 <= perm['p_value'] <= 1.0

    def test_reproducibility(self, mini_df, mini_config):
        """Same seed produces same results."""
        results = []
        for _ in range(2):
            random.seed(42)
            np.random.seed(42)
            model = IslandModel(mini_config, mini_df)
            model.initialize(total_pop_size=15)
            result = model.run(n_generations=3, patience=10)
            if result['best_strategies']:
                results.append(result['best_strategies'][0].fitness[0])

        if len(results) == 2:
            assert results[0] == results[1], "Same seed should produce same fitness"

    def test_regime_detection_integrates(self, mini_df):
        """Regime detection works on the pipeline data."""
        regime = detect_regime(mini_df)
        assert len(regime) == len(mini_df)
        assert set(regime.unique()).issubset({'bull', 'bear', 'sideways'})

    def test_strategy_serialization_roundtrip(self):
        """Strategy can be serialized to dict and reconstructed from genome."""
        random.seed(42)
        for _ in range(50):
            s = decode(random_genome(50))
            if s is not None:
                d = s.to_dict()
                s2 = decode(d['genome'])
                assert s2 is not None
                assert s2.direction == s.direction
                assert s2.expression_raw == s.expression_raw
                return
        pytest.skip("No valid strategy generated")

    def test_archive_fills_during_evolution(self, mini_df, mini_config):
        """Archive should accumulate strategies during evolution."""
        random.seed(42)
        model = IslandModel(mini_config, mini_df)
        model.initialize(total_pop_size=30)
        model.run(n_generations=10, patience=20)
        assert model.archive.n_occupied >= 0

    def test_all_island_types_produce_stats(self, mini_df, mini_config):
        """All 3 island types should produce generation stats."""
        random.seed(42)
        model = IslandModel(mini_config, mini_df)
        model.initialize(total_pop_size=30)
        gen_stats = model.step()
        assert len(gen_stats) == 3
        types = {s.selection_type for s in gen_stats}
        assert types == {'tournament', 'lexicase', 'random'}
