"""Tests for v2 operators: two-point crossover + mutation schedule."""

import random
import pytest


class TestTwoPointCrossover:
    def test_interior_block_exchanged(self):
        from evolution.operators import crossover
        random.seed(42)
        p1 = list(range(10))
        p2 = list(range(100, 110))
        c1, c2 = crossover(p1, p2)
        has_p2_block = any(v >= 100 for v in c1[1:-1])
        assert has_p2_block, "c1 should have interior block from p2"

    def test_children_same_length_equal_parents(self):
        from evolution.operators import crossover
        p1 = [1] * 20
        p2 = [2] * 20
        c1, c2 = crossover(p1, p2)
        assert len(c1) == 20
        assert len(c2) == 20

    def test_short_genome_returns_copies(self):
        from evolution.operators import crossover
        c1, c2 = crossover([1], [2])
        assert c1 == [1]
        assert c2 == [2]


class TestMutationSchedule:
    def test_early_gen_high_exploration(self):
        from evolution.operators import mutate
        random.seed(42)
        genome = [128] * 200
        mutated = mutate(genome, rate=1.0, generation=0, max_generations=100)
        changes = sum(1 for a, b in zip(genome, mutated) if a != b)
        big_changes = sum(1 for a, b in zip(genome, mutated) if abs(a - b) > 3 and abs(a-b) < 253 and a != b)
        ratio = big_changes / max(changes, 1)
        assert ratio > 0.4, f"Expected ~60% random jumps at gen 0, got {ratio:.1%}"

    def test_late_gen_low_exploration(self):
        from evolution.operators import mutate
        random.seed(42)
        genome = [128] * 200
        mutated = mutate(genome, rate=1.0, generation=100, max_generations=100)
        changes = sum(1 for a, b in zip(genome, mutated) if a != b)
        big_changes = sum(1 for a, b in zip(genome, mutated) if abs(a - b) > 3 and abs(a-b) < 253 and a != b)
        ratio = big_changes / max(changes, 1)
        assert ratio < 0.4, f"Expected ~20% random jumps at gen 100, got {ratio:.1%}"

    def test_no_mutation_at_zero_rate(self):
        from evolution.operators import mutate
        genome = [1, 2, 3, 4, 5]
        result = mutate(genome, rate=0.0, generation=0, max_generations=100)
        assert result == genome

    def test_finetune_range(self):
        from evolution.operators import mutate
        random.seed(42)
        genome = [128] * 1000
        mutated = mutate(genome, rate=1.0, generation=100, max_generations=100)
        small_deltas = set()
        for a, b in zip(genome, mutated):
            delta = b - a
            if 0 < abs(delta) <= 3:
                small_deltas.add(abs(delta))
        assert 1 in small_deltas
        assert 2 in small_deltas
        assert 3 in small_deltas
