"""
Genetic operators for Grammatical Evolution.

Operate on integer codon vectors (genomes). The grammar guarantees
that any integer vector produces a valid derivation (or fails cleanly).
"""

import random
from typing import List, Tuple


def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    One-point crossover on codon vectors.

    Returns two children. If parents differ in length, crossover
    point is chosen within the shorter length.
    """
    min_len = min(len(parent1), len(parent2))
    if min_len <= 1:
        return parent1[:], parent2[:]

    point = random.randint(1, min_len - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(genome: List[int], rate: float = 0.1) -> List[int]:
    """
    Mutate individual codons.

    Three mutation types applied per-codon:
    - increment (60%): ±1 (fine-tune parameters)
    - random (30%): new random value (structural exploration)
    - swap (10%): swap with neighbor (reorder conditions)
    """
    result = genome[:]
    for i in range(len(result)):
        if random.random() >= rate:
            continue

        r = random.random()
        if r < 0.60:
            # Increment: ±1 for fine parameter tuning
            result[i] = (result[i] + random.choice([-1, 1])) % 256
        elif r < 0.90:
            # Random: new value for structural change
            result[i] = random.randint(0, 255)
        else:
            # Swap with neighbor
            j = i + 1 if i < len(result) - 1 else i - 1
            if 0 <= j < len(result):
                result[i], result[j] = result[j], result[i]

    return result
