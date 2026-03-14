"""Tests for NSGA-II non-dominated sorting and selection."""

import pytest
from unittest.mock import MagicMock
from strategy.phenotype import Strategy


def _make_strategy(obj1, obj2, stability=0.0, cv=0.0):
    """Create a mock strategy with given objectives."""
    s = MagicMock(spec=Strategy)
    s.objectives = (obj1, obj2)
    s.stability = stability
    s.constraint_violation = cv
    s.rank = 999
    return s


class TestNonDominatedSort:
    def test_single_front(self):
        from evolution.nsga2 import non_dominated_sort
        strategies = [
            _make_strategy(3.0, 5.0),
            _make_strategy(1.0, 20.0),
            _make_strategy(2.0, 12.0),
        ]
        fronts = non_dominated_sort(strategies)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_two_fronts(self):
        from evolution.nsga2 import non_dominated_sort
        strategies = [
            _make_strategy(3.0, 20.0),
            _make_strategy(2.0, 10.0),
            _make_strategy(1.0, 5.0),
        ]
        fronts = non_dominated_sort(strategies)
        assert len(fronts) == 3
        assert strategies[0] in fronts[0]
        assert strategies[1] in fronts[1]
        assert strategies[2] in fronts[2]

    def test_rank_assigned(self):
        from evolution.nsga2 import non_dominated_sort
        s1 = _make_strategy(3.0, 20.0)
        s2 = _make_strategy(1.0, 5.0)
        non_dominated_sort([s1, s2])
        assert s1.rank == 1
        assert s2.rank == 2

    def test_empty_population(self):
        from evolution.nsga2 import non_dominated_sort
        assert non_dominated_sort([]) == []

    def test_equal_objectives_same_front(self):
        from evolution.nsga2 import non_dominated_sort
        s1 = _make_strategy(2.0, 10.0)
        s2 = _make_strategy(2.0, 10.0)
        fronts = non_dominated_sort([s1, s2])
        assert len(fronts) == 1
        assert len(fronts[0]) == 2


class TestConstrainedDomination:
    def test_feasible_dominates_infeasible(self):
        from evolution.nsga2 import non_dominated_sort
        feasible = _make_strategy(1.0, 5.0, cv=0.0)
        infeasible = _make_strategy(10.0, 50.0, cv=5.0)
        fronts = non_dominated_sort([feasible, infeasible])
        assert feasible.rank == 1
        assert infeasible.rank == 2

    def test_infeasible_ranked_by_violation(self):
        from evolution.nsga2 import non_dominated_sort
        low_violation = _make_strategy(1.0, 1.0, cv=1.0)
        high_violation = _make_strategy(1.0, 1.0, cv=10.0)
        fronts = non_dominated_sort([low_violation, high_violation])
        assert low_violation.rank < high_violation.rank


class TestStabilitySort:
    def test_within_front_sorted_by_stability(self):
        from evolution.nsga2 import select_by_stability
        s1 = _make_strategy(3.0, 5.0, stability=-0.1)
        s2 = _make_strategy(1.0, 20.0, stability=-0.5)
        s3 = _make_strategy(2.0, 12.0, stability=-2.0)
        front = [s3, s1, s2]
        selected = select_by_stability(front, n=2)
        assert selected == [s1, s2]


class TestBinaryTournament:
    def test_lower_rank_wins(self):
        from evolution.nsga2 import binary_tournament
        s1 = _make_strategy(1.0, 1.0, stability=-0.5)
        s1.rank = 1
        s2 = _make_strategy(5.0, 5.0, stability=-0.1)
        s2.rank = 2
        assert binary_tournament(s1, s2) is s1

    def test_same_rank_stability_wins(self):
        from evolution.nsga2 import binary_tournament
        s1 = _make_strategy(1.0, 1.0, stability=-0.1)
        s1.rank = 1
        s2 = _make_strategy(1.0, 1.0, stability=-2.0)
        s2.rank = 1
        assert binary_tournament(s1, s2) is s1


class TestSelectParents:
    def test_returns_correct_count(self):
        from evolution.nsga2 import non_dominated_sort, select_parents
        strategies = [_make_strategy(float(i), float(10-i)) for i in range(10)]
        non_dominated_sort(strategies)
        parents = select_parents(strategies, 5)
        assert len(parents) == 5
