"""
NSGA-II Multi-Objective Selection for Grammatical Evolution.

Implements:
- Non-dominated sorting with constrained domination (Deb 2002)
- Stability-based secondary sort (replaces standard crowding distance)
- Binary tournament on (rank, stability)
"""

import random
from typing import List
from strategy.phenotype import Strategy


def _dominates(a: Strategy, b: Strategy) -> bool:
    """True if a dominates b (all objectives >= and at least one >)."""
    a_obj = a.objectives
    b_obj = b.objectives
    at_least_one_better = False
    for ai, bi in zip(a_obj, b_obj):
        if ai < bi:
            return False
        if ai > bi:
            at_least_one_better = True
    return at_least_one_better


def _constrained_dominates(a: Strategy, b: Strategy) -> bool:
    """
    Constrained domination (Deb 2002):
    1. Feasible dominates infeasible
    2. Between feasible: standard dominance
    3. Between infeasible: lower constraint violation wins
    """
    a_feas = a.constraint_violation <= 0.0
    b_feas = b.constraint_violation <= 0.0

    if a_feas and not b_feas:
        return True
    if not a_feas and b_feas:
        return False
    if not a_feas and not b_feas:
        return a.constraint_violation < b.constraint_violation
    return _dominates(a, b)


def non_dominated_sort(population: List[Strategy]) -> List[List[Strategy]]:
    """
    Fast non-dominated sort (NSGA-II).

    Returns list of fronts: fronts[0] = Pareto front 1 (best),
    fronts[1] = front 2, etc.

    Sets strategy.rank for each individual (1-based).
    """
    if not population:
        return []

    n = len(population)
    domination_count = [0] * n
    dominated_set = [[] for _ in range(n)]

    fronts = []
    first_front_indices = []

    for i in range(n):
        for j in range(i + 1, n):
            if _constrained_dominates(population[i], population[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _constrained_dominates(population[j], population[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            population[i].rank = 1
            first_front_indices.append(i)

    fronts.append([population[i] for i in first_front_indices])
    current_front_indices = first_front_indices
    rank = 1

    while current_front_indices:
        next_front_indices = []
        rank += 1
        for i in current_front_indices:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = rank
                    next_front_indices.append(j)

        if next_front_indices:
            fronts.append([population[j] for j in next_front_indices])
        current_front_indices = next_front_indices

    return fronts


def select_by_stability(front: List[Strategy], n: int) -> List[Strategy]:
    """
    Select top n from a front by stability (highest = least negative std).
    """
    sorted_front = sorted(front, key=lambda s: s.stability, reverse=True)
    return sorted_front[:n]


def binary_tournament(a: Strategy, b: Strategy) -> Strategy:
    """Binary tournament: lower rank wins; ties broken by stability."""
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.stability > b.stability:
        return a
    return b


def select_parents(population: List[Strategy], n: int) -> List[Strategy]:
    """Select n parents via binary tournament from population."""
    parents = []
    for _ in range(n):
        a = random.choice(population)
        b = random.choice(population)
        parents.append(binary_tournament(a, b))
    return parents
