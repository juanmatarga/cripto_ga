"""
Genetic operators for Grammatical Evolution.

v3: Two-point crossover + improved mutation with active-codon targeting.
"""

import random
from typing import List, Tuple


def crossover(parent1: List[int], parent2: List[int],
              active_codons: int = 0) -> Tuple[List[int], List[int]]:
    """
    Two-point crossover on codon vectors.

    If active_codons > 0, biases cut points toward the active region
    to avoid neutral crossover in unused tail codons.
    """
    min_len = min(len(parent1), len(parent2))
    if min_len <= 2:
        return parent1[:], parent2[:]

    if active_codons > 0 and active_codons < min_len - 1:
        # 70% chance: at least one cut point in active region
        if random.random() < 0.7:
            p1 = random.randint(1, max(1, active_codons))
            p2 = random.randint(p1 + 1, min_len - 1) if p1 < min_len - 1 else p1
            if p1 == p2 and p2 < min_len - 1:
                p2 += 1
        else:
            p1, p2 = sorted(random.sample(range(1, min_len), 2))
    else:
        p1, p2 = sorted(random.sample(range(1, min_len), 2))

    child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
    return child1, child2


def mutate(genome: List[int], rate: float = 0.1,
           generation: int = 0, max_generations: int = 100,
           active_codons: int = 0) -> List[int]:
    """
    Mutate with generation-dependent exploration schedule.

    Improvements over v2:
    - Higher exploration floor (35% minimum, was 20%)
    - Wider exploitation deltas (±5 instead of ±3)
    - Active-codon targeting: 2x mutation rate on active region
    """
    result = genome[:]

    progress = min(generation / max(max_generations, 1), 1.0)
    # Exploration ratio: starts at 0.6, decays to 0.35 (was 0.2)
    explore_ratio = max(0.35, 0.6 - 0.25 * progress)

    for i in range(len(result)):
        # Active codons mutate at 2x rate (they actually affect phenotype)
        effective_rate = rate
        if active_codons > 0 and i < active_codons:
            effective_rate = min(rate * 2.0, 0.40)

        if random.random() >= effective_rate:
            continue

        if random.random() < explore_ratio:
            result[i] = random.randint(0, 255)
        else:
            # Wider deltas for exploitation (was ±3, now ±5)
            delta = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            result[i] = (result[i] + delta) % 256

    return result
