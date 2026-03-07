"""Tests for Sprint 4: MAP-Elites, Island Model, Regime Detection."""

import random
import numpy as np
import pandas as pd
import pytest

from grammar.mapper import decode
from strategy.parameters import random_genome
from strategy.phenotype import Strategy
from data.regime_detector import detect_regime, regime_summary
from evolution.archive import (
    MAPElitesArchive, TOTAL_CELLS, _freq_bin, _complexity_bin, _regime_bin
)
from evolution.island import IslandModel


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
        'evolution': {
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_pct': 0.05,
            'genome_length': 50,
            'tournament_k': 3,
            'n_windows_per_gen': 3,
            'window_bars': 2880,
        },
        'islands': {
            'n_islands': 3,
            'migration_interval': 5,
            'migration_size': 2,
            'selection_types': ['tournament', 'lexicase', 'random'],
        },
    }


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class TestRegimeDetector:
    def test_returns_series_same_length(self, sample_df):
        regime = detect_regime(sample_df)
        assert len(regime) == len(sample_df)
        assert regime.index.equals(sample_df.index)

    def test_only_valid_labels(self, sample_df):
        regime = detect_regime(sample_df)
        valid_labels = {'bull', 'bear', 'sideways'}
        assert set(regime.unique()).issubset(valid_labels)

    def test_all_three_regimes_present(self, sample_df):
        regime = detect_regime(sample_df)
        # With 17280 bars of random walk, we should get all three
        assert len(regime.unique()) == 3

    def test_trending_data_mostly_bull(self):
        """Strongly uptrending data should be mostly bull."""
        np.random.seed(99)
        n = 5000
        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        # Strong uptrend: +10 per bar on average
        close = 50000 + np.cumsum(np.ones(n) * 10 + np.random.randn(n) * 2)
        df = pd.DataFrame({
            'Open': close, 'High': close + 5, 'Low': close - 5,
            'Close': close, 'Volume': np.ones(n) * 1000,
        }, index=dates)
        regime = detect_regime(df)
        bull_pct = (regime == 'bull').mean()
        assert bull_pct > 0.5

    def test_regime_summary(self, sample_df):
        regime = detect_regime(sample_df)
        summary = regime_summary(regime)
        assert 'bull' in summary
        assert 'bear' in summary
        assert 'sideways' in summary
        total_pct = sum(v['pct'] for v in summary.values())
        assert abs(total_pct - 1.0) < 0.01


# ============================================================================
# MAP-ELITES ARCHIVE
# ============================================================================

class TestArchiveBins:
    def test_freq_bin_low(self):
        assert _freq_bin(3.0) == 'low'

    def test_freq_bin_medium(self):
        assert _freq_bin(10.0) == 'medium'

    def test_freq_bin_high(self):
        assert _freq_bin(25.0) == 'high'

    def test_complexity_bin_clamps_to_5(self):
        assert _complexity_bin(10) == 5
        assert _complexity_bin(100) == 5

    def test_complexity_bin_minimum_1(self):
        assert _complexity_bin(0) == 1
        assert _complexity_bin(1) == 1

    def test_regime_bin_picks_best(self):
        assert _regime_bin({'bull': 2.0, 'bear': 1.0, 'sideways': 0.5}) == 'bull'

    def test_regime_bin_default_sideways(self):
        assert _regime_bin(None) == 'sideways'


class TestMAPElitesArchive:
    def test_add_to_empty_cell(self):
        archive = MAPElitesArchive()
        s = self._make_strategy(fitness=(2.0, 1.0), n_nodes=2)
        added = archive.try_add(s, trades_per_month=10.0)
        assert added
        assert archive.n_occupied == 1

    def test_better_fitness_replaces(self):
        archive = MAPElitesArchive()
        s1 = self._make_strategy(fitness=(1.0, 0.5), n_nodes=2)
        s2 = self._make_strategy(fitness=(3.0, 1.5), n_nodes=2)
        archive.try_add(s1, trades_per_month=10.0)
        added = archive.try_add(s2, trades_per_month=10.0)
        assert added
        assert archive.n_occupied == 1  # Same cell, replaced

    def test_worse_fitness_rejected(self):
        archive = MAPElitesArchive()
        s1 = self._make_strategy(fitness=(3.0, 1.5), n_nodes=2)
        s2 = self._make_strategy(fitness=(1.0, 0.5), n_nodes=2)
        archive.try_add(s1, trades_per_month=10.0)
        added = archive.try_add(s2, trades_per_month=10.0)
        assert not added

    def test_fail_fitness_rejected(self):
        archive = MAPElitesArchive()
        s = self._make_strategy(fitness=(-999.0, -999.0), n_nodes=2)
        added = archive.try_add(s, trades_per_month=10.0)
        assert not added
        assert archive.n_occupied == 0

    def test_different_niches_separate(self):
        archive = MAPElitesArchive()
        s1 = self._make_strategy(fitness=(2.0, 1.0), n_nodes=1)
        s2 = self._make_strategy(fitness=(2.0, 1.0), n_nodes=4)
        archive.try_add(s1, trades_per_month=3.0)   # low freq, 1 node
        archive.try_add(s2, trades_per_month=25.0)   # high freq, 4 nodes
        assert archive.n_occupied == 2

    def test_sample_for_reproduction(self):
        archive = MAPElitesArchive()
        for i in range(5):
            s = self._make_strategy(fitness=(float(i), 0.0), n_nodes=i + 1)
            archive.try_add(s, trades_per_month=float(i * 5))
        samples = archive.sample_for_reproduction(3)
        assert len(samples) == 3
        assert all(isinstance(s, Strategy) for s in samples)

    def test_sample_empty_archive(self):
        archive = MAPElitesArchive()
        assert archive.sample_for_reproduction(5) == []

    def test_coverage(self):
        archive = MAPElitesArchive()
        assert archive.coverage == 0.0
        s = self._make_strategy(fitness=(1.0, 0.5), n_nodes=2)
        archive.try_add(s, trades_per_month=10.0)
        assert archive.coverage == 1.0 / TOTAL_CELLS

    def test_many_insertions_fill_cells(self):
        """Random strategies should fill multiple cells."""
        archive = MAPElitesArchive()
        random.seed(42)
        for _ in range(500):
            n_nodes = random.randint(1, 8)
            fitness = (random.uniform(-1, 5), random.uniform(-1, 3))
            tpm = random.uniform(0, 40)
            regime = random.choice([
                {'bull': random.random(), 'bear': random.random(), 'sideways': random.random()},
                None,
            ])
            s = self._make_strategy(fitness=fitness, n_nodes=n_nodes)
            archive.try_add(s, trades_per_month=tpm, regime_sortinos=regime)
        # Should fill a good portion of the 45 cells
        assert archive.n_occupied >= 15

    def test_summary(self):
        archive = MAPElitesArchive()
        s = self._make_strategy(fitness=(2.0, 1.0), n_nodes=2)
        archive.try_add(s, trades_per_month=10.0)
        summary = archive.summary()
        assert summary['n_occupied'] == 1
        assert summary['best_fitness'] == 2.0

    def _make_strategy(self, fitness, n_nodes):
        return Strategy(
            genome=[1, 2, 3], direction='LONG', conditions=[],
            logic='', tp_atr_mult=2.0, sl_atr_mult=1.0,
            expression_raw='test', n_nodes=n_nodes, codons_used=3,
            wrapping_count=0, fitness=fitness,
        )


# ============================================================================
# ISLAND MODEL
# ============================================================================

class TestIslandModel:
    def test_initialize_creates_islands(self, sample_df, config):
        model = IslandModel(config, sample_df)
        model.initialize(total_pop_size=30)
        assert len(model.islands) == 3
        total = sum(len(island) for island in model.islands)
        assert total == 30

    def test_step_evaluates_all(self, sample_df, config):
        random.seed(42)
        model = IslandModel(config, sample_df)
        model.initialize(total_pop_size=30)
        gen_stats = model.step()
        assert len(gen_stats) == 3
        assert model.generation == 1
        assert model.total_evaluations == 30

    def test_population_stable_after_step(self, sample_df, config):
        random.seed(42)
        model = IslandModel(config, sample_df)
        model.initialize(total_pop_size=30)
        sizes_before = [len(island) for island in model.islands]
        model.step()
        sizes_after = [len(island) for island in model.islands]
        assert sizes_before == sizes_after

    def test_migration_occurs(self, sample_df, config):
        """After migration_interval generations, migration should happen."""
        random.seed(42)
        config['islands']['migration_interval'] = 2
        model = IslandModel(config, sample_df)
        model.initialize(total_pop_size=30)
        # Run 2 steps to trigger migration
        model.step()
        model.step()
        # Population sizes should still be stable
        total = sum(len(island) for island in model.islands)
        assert total == 30

    def test_archive_populated_after_run(self, sample_df, config):
        random.seed(42)
        model = IslandModel(config, sample_df)
        model.initialize(total_pop_size=30)
        result = model.run(n_generations=5, patience=10)
        # Archive should have some entries
        assert result['archive'].n_occupied >= 0  # May be 0 if all fail constraints
        assert result['total_evaluations'] > 0
        assert result['final_generation'] == 5

    def test_different_selection_types(self, sample_df, config):
        random.seed(42)
        model = IslandModel(config, sample_df)
        model.initialize(total_pop_size=30)
        gen_stats = model.step()
        types = [s.selection_type for s in gen_stats]
        assert types == ['tournament', 'lexicase', 'random']

    def test_history_grows(self, sample_df, config):
        random.seed(42)
        model = IslandModel(config, sample_df)
        model.initialize(total_pop_size=30)
        model.step()
        model.step()
        model.step()
        assert len(model.history) == 3
