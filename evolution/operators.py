"""
Genetic operators for Grammatical Evolution.

v2: Two-point crossover + generation-dependent mutation schedule.
"""

import random
from typing import List, Tuple


def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Two-point crossover on codon vectors.

    Exchanges an interior block between parents.
    """
    min_len = min(len(parent1), len(parent2))
    if min_len <= 2:
        return parent1[:], parent2[:]

    p1, p2 = sorted(random.sample(range(1, min_len), 2))
    child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return child1, child2


def mutate(genome: List[int], rate: float = 0.1,
           generation: int = 0, max_generations: int = 100) -> List[int]:
    """
    Mutate with generation-dependent exploration schedule.

    Early generations: high exploration (random jumps).
    Later generations: high exploitation (fine-tuning ±1..3).
    """
    result = genome[:]

    progress = min(generation / max(max_generations, 1), 1.0)
    explore_ratio = 0.6 - 0.4 * progress

    for i in range(len(result)):
        if random.random() >= rate:
            continue

        if random.random() < explore_ratio:
            result[i] = random.randint(0, 255)
        else:
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            result[i] = (result[i] + delta) % 256

    return result
