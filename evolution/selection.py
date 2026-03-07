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
    Lexicase selection for multi-objective optimization.

    Shuffles objectives, filters population at each step by keeping
    individuals >= median on that objective. Promotes diversity by
    favoring specialists in different objectives on different calls.
    """
    candidates = list(population)
    objectives = [0, 1]  # sortino, calmar
    random.shuffle(objectives)

    for obj_idx in objectives:
        if len(candidates) <= 1:
            break
        values = [c.fitness[obj_idx] for c in candidates]
        med = sorted(values)[len(values) // 2]
        filtered = [c for c in candidates if c.fitness[obj_idx] >= med]
        if filtered:
            candidates = filtered

    return random.choice(candidates)
