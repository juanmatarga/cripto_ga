"""Tests for grammar/mapper.py"""

import random
import pytest
from grammar.mapper import decode
from strategy.parameters import random_genome


class TestDecode:
    def test_deterministic(self):
        """Same genome always produces same strategy."""
        genome = [100, 50, 200, 30, 150, 80, 20, 170, 90, 60,
                  40, 110, 75, 180, 10, 250, 35, 130, 220, 5,
                  140, 85, 195, 55, 160, 70, 25, 115, 210, 45,
                  95, 165, 15, 230, 125, 65, 185, 105, 245, 50,
                  75, 120, 155, 30, 200, 80, 145, 60, 190, 110]

        s1 = decode(genome)
        s2 = decode(genome)

        assert s1 is not None
        assert s2 is not None
        assert s1.direction == s2.direction
        assert s1.logic == s2.logic
        assert s1.tp_atr_mult == s2.tp_atr_mult
        assert s1.sl_atr_mult == s2.sl_atr_mult
        assert len(s1.conditions) == len(s2.conditions)
        for c1, c2 in zip(s1.conditions, s2.conditions):
            assert str(c1) == str(c2)

    def test_empty_genome_returns_none(self):
        assert decode([]) is None

    def test_valid_genome_returns_strategy(self):
        random.seed(123)
        genome = random_genome(50)
        s = decode(genome)
        # Might be None for some seeds, so try a few
        for seed in range(100):
            random.seed(seed)
            s = decode(random_genome(50))
            if s is not None:
                break
        assert s is not None

    def test_strategy_has_required_fields(self):
        random.seed(42)
        s = None
        for _ in range(100):
            s = decode(random_genome(50))
            if s is not None:
                break

        assert s is not None
        assert s.direction in ('LONG', 'SHORT')
        assert len(s.conditions) >= 1
        assert s.tp_atr_mult > 0
        assert s.sl_atr_mult > 0
        assert s.n_nodes == len(s.conditions)
        assert s.codons_used > 0
        assert s.genome is not None

    def test_favorable_rr_enforced(self):
        """TP should always be >= SL."""
        random.seed(42)
        for _ in range(200):
            s = decode(random_genome(50))
            if s is not None:
                assert s.tp_atr_mult >= s.sl_atr_mult, \
                    f"Unfavorable R:R: TP={s.tp_atr_mult} SL={s.sl_atr_mult}"

    def test_high_validity_rate(self):
        """At least 60% of random genomes should produce valid strategies."""
        random.seed(42)
        valid = sum(1 for _ in range(500) if decode(random_genome(50)) is not None)
        rate = valid / 500
        assert rate >= 0.60, f"Validity rate too low: {rate:.0%}"

    def test_different_genomes_produce_variety(self):
        """Different genomes should produce different strategies."""
        random.seed(42)
        expressions = set()
        for _ in range(100):
            s = decode(random_genome(50))
            if s is not None:
                expressions.add(s.expression_raw)
        # At least 80% unique
        assert len(expressions) >= 50, \
            f"Only {len(expressions)} unique strategies from 100 genomes"

    def test_direction_distribution(self):
        """Should produce roughly equal LONG and SHORT strategies."""
        random.seed(42)
        directions = []
        for _ in range(500):
            s = decode(random_genome(50))
            if s is not None:
                directions.append(s.direction)
        long_pct = directions.count('LONG') / len(directions)
        # Should be between 30% and 70% (grammar has 2 choices: LONG/SHORT)
        assert 0.30 <= long_pct <= 0.70, f"Direction imbalance: {long_pct:.0%} LONG"

    def test_conditions_have_valid_comparators(self):
        random.seed(42)
        for _ in range(200):
            s = decode(random_genome(50))
            if s is not None:
                for c in s.conditions:
                    assert c.comparator in ('>', '<', 'CROSSES_ABOVE', 'CROSSES_BELOW'), \
                        f"Invalid comparator: {c.comparator}"

    def test_wrapping_tracked(self):
        """Wrapping count should be >= 0."""
        random.seed(42)
        for _ in range(100):
            s = decode(random_genome(50))
            if s is not None:
                assert s.wrapping_count >= 0

    def test_short_genome_may_wrap(self):
        """A very short genome should either wrap or return None."""
        s = decode([100, 50, 200])
        # Short genome will need wrapping — either succeeds or returns None
        # Just verify no crash
        assert s is None or s.wrapping_count > 0

    def test_to_dict_roundtrip(self):
        random.seed(42)
        s = None
        for _ in range(100):
            s = decode(random_genome(50))
            if s is not None:
                break
        assert s is not None
        d = s.to_dict()
        assert d['direction'] == s.direction
        assert d['tp_atr_mult'] == s.tp_atr_mult
        assert d['sl_atr_mult'] == s.sl_atr_mult
        assert d['n_nodes'] == s.n_nodes
        assert len(d['conditions']) == len(s.conditions)
