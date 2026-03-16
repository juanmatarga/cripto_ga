"""
Selection operators for evolution.
"""

import random
from typing import List
from strategy.phenotype import Strategy


def tournament_select(population: List[Strategy], k: int = 3) -> Strategy:
    """
    Tournament selection. Pick k random individuals, return the best.

    Uses fitness[0] (Sortino) as the primary objective.
    """
    candidates = random.sample(population, min(k, len(population)))
    return max(candidates, key=lambda s: s.fitness[0])


def lexicase_select(population: List[Strategy]) -> Strategy:
    """
    Extended lexicase selection for multi-objective + per-window optimization.

    Shuffles a set of criteria, filters population at each step by keeping
    individuals >= median. Promotes diversity by favoring specialists in
    different metrics on different calls.

    Criteria pool:
    - objectives[0]: composite fitness
    - objectives[1]: consistency
    - stability: -std(sortino across windows)
    - n_trades: total trade count
    """
    candidates = list(population)
    if not candidates:
        return random.choice(population)

    # Build criteria functions
    criteria = [
        lambda s: s.objectives[0] if hasattr(s, 'objectives') else -999,
        lambda s: s.objectives[1] if hasattr(s, 'objectives') else -999,
        lambda s: s.stability if hasattr(s, 'stability') else -999,
        lambda s: s.n_trades if hasattr(s, 'n_trades') else 0,
    ]
    random.shuffle(criteria)

    for criterion in criteria:
        if len(candidates) <= 1:
            break
        values = [criterion(c) for c in candidates]
        valid_values = [v for v in values if v != -999]
        if not valid_values:
            continue
        med = sorted(valid_values)[len(valid_values) // 2]
        filtered = [c for c, v in zip(candidates, values) if v >= med]
        if filtered:
            candidates = filtered

    return random.choice(candidates)
