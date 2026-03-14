"""Tests for NSGA-II fitness functions."""

import pytest
from unittest.mock import MagicMock
from strategy.phenotype import Strategy


def _make_strategy(n_nodes=2):
    s = MagicMock(spec=Strategy)
    s.n_nodes = n_nodes
    s.genome = [1, 2, 3]
    s.objectives = (-999.0, -999.0)
    s.stability = -999.0
    s.constraint_violation = 0.0
    s.fitness = (-999.0, -999.0)
    s.metrics = None
    s.window_metrics = None
    s.n_trades = 0
    return s


class TestComputeObjectives:
    def test_median_of_5_windows(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy(n_nodes=2)
        metrics = [
            {'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10, 'n_trades': 20,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 2.0, 'return_pct': 10.0, 'max_dd': -0.15, 'n_trades': 25,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 3.0, 'return_pct': 15.0, 'max_dd': -0.20, 'n_trades': 30,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 1.5, 'return_pct': 8.0, 'max_dd': -0.12, 'n_trades': 22,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 2.5, 'return_pct': 12.0, 'max_dd': -0.18, 'n_trades': 28,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
        ]
        compute_objectives(s, metrics, parsimony_coeff=0.02)
        # Sorted sortinos: [1.0, 1.5, 2.0, 2.5, 3.0] → median = 2.0
        assert abs(s.objectives[0] - (2.0 - 0.02 * 2)) < 0.01
        assert abs(s.objectives[1] - 10.0) < 0.01

    def test_feasible_when_constraints_met(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [{'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.30,
                     'n_trades': 15, 'win_rate': 0.5, 'profit_factor': 1.5,
                     'expectancy': 0.01}]
        compute_objectives(s, metrics)
        assert s.constraint_violation == 0.0

    def test_infeasible_high_dd(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [{'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.50,
                     'n_trades': 15, 'win_rate': 0.5, 'profit_factor': 1.5,
                     'expectancy': 0.01}]
        compute_objectives(s, metrics)
        assert s.constraint_violation > 0

    def test_infeasible_low_trades(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [{'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10,
                     'n_trades': 5, 'win_rate': 0.5, 'profit_factor': 1.5,
                     'expectancy': 0.01}]
        compute_objectives(s, metrics)
        assert s.constraint_violation > 0

    def test_empty_metrics(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        compute_objectives(s, [])
        assert s.objectives == (-999.0, -999.0)
        assert s.constraint_violation > 0

    def test_stability_zero_variance(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [
            {'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10, 'n_trades': 20,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10, 'n_trades': 20,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
        ]
        compute_objectives(s, metrics)
        assert s.stability == 0.0

    def test_legacy_fitness_set(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [{'sortino': 2.0, 'return_pct': 10.0, 'max_dd': -0.10,
                     'n_trades': 20, 'win_rate': 0.5, 'profit_factor': 1.5,
                     'expectancy': 0.01}]
        compute_objectives(s, metrics, parsimony_coeff=0.02)
        assert s.fitness == s.objectives
