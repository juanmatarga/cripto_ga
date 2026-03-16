"""
NSGA-II Multi-Objective Selection for Grammatical Evolution.

Implements:
- Non-dominated sorting with constrained domination (Deb 2002)
- Crowding distance for diversity maintenance
- Binary tournament on (rank, crowding_distance)
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


def compute_crowding_distance(front: List[Strategy]) -> None:
    """
    Compute crowding distance for each individual in a front (in-place).
    Boundary individuals get infinite distance.
    """
    n = len(front)
    for s in front:
        s.crowding_distance = 0.0

    if n <= 2:
        for s in front:
            s.crowding_distance = float('inf')
        return

    n_obj = len(front[0].objectives)

    for m in range(n_obj):
        sorted_indices = sorted(range(n), key=lambda i: front[i].objectives[m])

        # Boundary points get infinity
        front[sorted_indices[0]].crowding_distance = float('inf')
        front[sorted_indices[-1]].crowding_distance = float('inf')

        obj_range = (front[sorted_indices[-1]].objectives[m] -
                     front[sorted_indices[0]].objectives[m])
        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            front[sorted_indices[i]].crowding_distance += (
                front[sorted_indices[i + 1]].objectives[m] -
                front[sorted_indices[i - 1]].objectives[m]
            ) / obj_range


def select_by_crowding(front: List[Strategy], n: int) -> List[Strategy]:
    """
    Select top n from a front by crowding distance (highest = most diverse).
    """
    compute_crowding_distance(front)
    sorted_front = sorted(front, key=lambda s: s.crowding_distance, reverse=True)
    return sorted_front[:n]


# Keep old name for backward compatibility
select_by_stability = select_by_crowding


def binary_tournament(a: Strategy, b: Strategy) -> Strategy:
    """Binary tournament: lower rank wins; ties broken by crowding distance."""
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding_distance > b.crowding_distance:
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
