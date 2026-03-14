# Evolution Pipeline Paradigm Shift — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scalar-fitness tournament-selection evolution engine with NSGA-II multi-objective optimization, fix grammar bias, add evaluation caching, and improve genetic operators — then re-evolve strategies on extended training data.

**Architecture:** NSGA-II (non-dominated sort + stability-based tiebreak) replaces tournament selection. Grammar decouples logical operators from structure. Evaluation uses 5 windows with median aggregation and per-window caching. MAP-Elites archive feeds 10% of parents back into selection.

**Tech Stack:** Python 3.9+, NumPy, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-14-evolution-paradigm-shift-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `evolution/nsga2.py` | Non-dominated sorting, stability sort, binary tournament, constrained domination |
| `evolution/cache.py` | Per-window evaluation cache with generation-based eviction |
| `tests/test_nsga2.py` | NSGA-II unit tests |
| `tests/test_cache.py` | Cache unit tests |
| `tests/test_operators_v2.py` | Mutation schedule + two-point crossover tests |

### Modified Files
| File | What Changes |
|------|-------------|
| `strategy/phenotype.py` | Add `objectives`, `stability`, `constraint_violation`, `rank` fields |
| `grammar/bnf.py` | Replace `<entry_rule>` with 4 balanced productions + `<logical_op>` |
| `grammar/mapper.py` | No change needed — mapper already handles any grammar structure via generic expansion |
| `evolution/operators.py` | Two-point crossover, mutation schedule (generation-dependent) |
| `evolution/fitness.py` | Return per-window metrics dict, remove scalar fitness composition, add constraint checks |
| `evolution/selection.py` | No changes needed — binary tournament lives in `nsga2.py`. Old functions kept for backward compat. |
| `evolution/archive.py` | Add `dominates()` comparison using objectives, not `fitness[0]` |
| `evolution/engine.py` | Complete rewrite of `step()` to NSGA-II loop with caching |
| `backtest/sampling.py` | 3-month windows (8640 bars), partial rotation (keep 3, replace 2) |
| `tests/test_grammar.py` | Update for new `<entry_rule>` + `<logical_op>` distribution tests |
| `tests/test_engine.py` | Update for NSGA-II based engine |

---

## Chunk 1: NSGA-II Core + Strategy Fields

### Task 1: Strategy Phenotype — Add Multi-Objective Fields

**Files:**
- Modify: `strategy/phenotype.py`

- [ ] **Step 1: Add new fields to Strategy dataclass**

In `strategy/phenotype.py`, add these fields after the existing `fitness` field (line 40):

```python
# NSGA-II fields (filled during evaluation/selection)
objectives: Tuple[float, float] = (-999.0, -999.0)  # (median_sortino, median_return)
stability: float = -999.0                             # -std(sortino across windows)
constraint_violation: float = 0.0                      # 0.0 = feasible, >0 = infeasible
rank: int = 999                                        # Pareto front rank (1 = best)
window_metrics: Optional[List[Dict]] = None            # Per-window metrics for analysis
```

Also update `to_dict()` to include the new fields:
```python
'objectives': list(self.objectives),
'rank': self.rank,
'stability': self.stability,
```

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_mapper.py tests/test_grammar.py -v`
Expected: All pass (new fields have defaults, so existing code is unaffected)

- [ ] **Step 3: Commit**

```bash
git add strategy/phenotype.py
git commit -m "feat: add NSGA-II fields to Strategy (objectives, stability, rank, constraint_violation)"
```

---

### Task 2: NSGA-II Core — Non-Dominated Sorting

**Files:**
- Create: `evolution/nsga2.py`
- Create: `tests/test_nsga2.py`

- [ ] **Step 1: Write failing tests for non_dominated_sort**

Create `tests/test_nsga2.py`:

```python
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
        """All non-dominated → single front."""
        from evolution.nsga2 import non_dominated_sort
        # (Sortino, Return) — all on Pareto front
        strategies = [
            _make_strategy(3.0, 5.0),   # high sortino, low return
            _make_strategy(1.0, 20.0),   # low sortino, high return
            _make_strategy(2.0, 12.0),   # middle
        ]
        fronts = non_dominated_sort(strategies)
        assert len(fronts) == 1
        assert len(fronts[0]) == 3

    def test_two_fronts(self):
        """Dominated strategies go to second front."""
        from evolution.nsga2 import non_dominated_sort
        strategies = [
            _make_strategy(3.0, 20.0),   # dominates all
            _make_strategy(2.0, 10.0),   # dominated by first
            _make_strategy(1.0, 5.0),    # dominated by both
        ]
        fronts = non_dominated_sort(strategies)
        assert len(fronts) == 3
        assert strategies[0] in fronts[0]
        assert strategies[1] in fronts[1]
        assert strategies[2] in fronts[2]

    def test_rank_assigned(self):
        """Each strategy gets its front rank."""
        from evolution.nsga2 import non_dominated_sort
        s1 = _make_strategy(3.0, 20.0)
        s2 = _make_strategy(1.0, 5.0)
        non_dominated_sort([s1, s2])
        assert s1.rank == 1
        assert s2.rank == 2

    def test_empty_population(self):
        from evolution.nsga2 import non_dominated_sort
        assert non_dominated_sort([]) == []


class TestConstrainedDomination:
    def test_feasible_dominates_infeasible(self):
        """Feasible always beats infeasible regardless of objectives."""
        from evolution.nsga2 import non_dominated_sort
        feasible = _make_strategy(1.0, 5.0, cv=0.0)
        infeasible = _make_strategy(10.0, 50.0, cv=5.0)  # better obj but infeasible
        fronts = non_dominated_sort([feasible, infeasible])
        assert feasible.rank == 1
        assert infeasible.rank == 2

    def test_infeasible_ranked_by_violation(self):
        """Among infeasible, lower violation is better."""
        from evolution.nsga2 import non_dominated_sort
        low_violation = _make_strategy(1.0, 1.0, cv=1.0)
        high_violation = _make_strategy(1.0, 1.0, cv=10.0)
        fronts = non_dominated_sort([low_violation, high_violation])
        assert low_violation.rank < high_violation.rank


class TestStabilitySort:
    def test_within_front_sorted_by_stability(self):
        """Within a Pareto front, most stable first."""
        from evolution.nsga2 import select_by_stability
        s1 = _make_strategy(3.0, 5.0, stability=-0.1)   # most stable
        s2 = _make_strategy(1.0, 20.0, stability=-0.5)
        s3 = _make_strategy(2.0, 12.0, stability=-2.0)   # least stable
        front = [s3, s1, s2]
        selected = select_by_stability(front, n=2)
        assert selected == [s1, s2]


class TestBinaryTournament:
    def test_lower_rank_wins(self):
        """Strategy with lower rank always wins tournament."""
        from evolution.nsga2 import binary_tournament
        s1 = _make_strategy(1.0, 1.0, stability=-0.5)
        s1.rank = 1
        s2 = _make_strategy(5.0, 5.0, stability=-0.1)
        s2.rank = 2
        # s1 should win (rank 1 < rank 2) regardless of objectives
        winner = binary_tournament(s1, s2)
        assert winner is s1

    def test_same_rank_stability_wins(self):
        """Same rank: higher stability wins."""
        from evolution.nsga2 import binary_tournament
        s1 = _make_strategy(1.0, 1.0, stability=-0.1)
        s1.rank = 1
        s2 = _make_strategy(1.0, 1.0, stability=-2.0)
        s2.rank = 1
        winner = binary_tournament(s1, s2)
        assert winner is s1  # stability -0.1 > -2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_nsga2.py -v`
Expected: ImportError — `evolution.nsga2` does not exist

- [ ] **Step 3: Implement NSGA-II core**

Create `evolution/nsga2.py`:

```python
"""
NSGA-II Multi-Objective Selection for Grammatical Evolution.

Implements:
- Non-dominated sorting with constrained domination (Deb 2002)
- Stability-based secondary sort (replaces standard crowding distance)
- Binary tournament on (rank, stability)
"""

from typing import List, Tuple
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
    domination_count = [0] * n    # how many dominate me
    dominated_set = [[] for _ in range(n)]  # who I dominate

    fronts = []
    first_front = []

    for i in range(n):
        for j in range(i + 1, n):
            if _constrained_dominates(population[i], population[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _constrained_dominates(population[j], population[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            population[i].rank = 1
            first_front.append(population[i])

    fronts.append(first_front)

    current_front_indices = [i for i in range(n) if domination_count[i] == 0]
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

    stability = -std(sortino across windows). Higher = more consistent.
    """
    sorted_front = sorted(front, key=lambda s: s.stability, reverse=True)
    return sorted_front[:n]


def binary_tournament(a: Strategy, b: Strategy) -> Strategy:
    """
    Binary tournament: lower rank wins; ties broken by stability.
    """
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    # Same rank — higher stability wins
    if a.stability > b.stability:
        return a
    return b


def select_parents(population: List[Strategy], n: int) -> List[Strategy]:
    """Select n parents via binary tournament from population."""
    import random
    parents = []
    for _ in range(n):
        a = random.choice(population)
        b = random.choice(population)
        parents.append(binary_tournament(a, b))
    return parents
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_nsga2.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add evolution/nsga2.py tests/test_nsga2.py
git commit -m "feat: NSGA-II non-dominated sort with constrained domination and stability tiebreak"
```

---

## Chunk 2: Grammar Fix + Operators

### Task 3: Grammar — Decouple Structure from Logic

**Files:**
- Modify: `grammar/bnf.py`
- Modify: `tests/test_grammar.py`

- [ ] **Step 1: Write failing test for new grammar distribution**

Add to `tests/test_grammar.py`:

```python
def test_entry_rule_distribution():
    """New grammar: ~25% per structural family (±5%)."""
    import random
    from grammar.mapper import decode
    from strategy.parameters import random_genome

    random.seed(42)
    families = {'simple': 0, 'binary': 0, 'ternary': 0, 'grouped': 0}
    total = 0

    for _ in range(10000):
        genome = random_genome()
        s = decode(genome)
        if s is None:
            continue
        total += 1
        n = len(s.conditions)
        has_parens = '(' in s.logic
        if n == 1:
            families['simple'] += 1
        elif n == 2 and not has_parens:
            families['binary'] += 1
        elif n == 3 and not has_parens:
            families['ternary'] += 1
        elif has_parens:
            families['grouped'] += 1

    for family, count in families.items():
        pct = count / total * 100
        assert 18 < pct < 35, f"{family}: {pct:.1f}% — expected ~25% (±7%)"


def test_logical_op_distribution():
    """<logical_op> should be ~50/50 AND/OR. Count individual operator occurrences."""
    import random
    import re
    from grammar.mapper import decode
    from strategy.parameters import random_genome

    random.seed(42)
    and_count = 0
    or_count = 0

    for _ in range(5000):
        genome = random_genome()
        s = decode(genome)
        if s is None or len(s.conditions) < 2:
            continue
        # Count individual AND/OR tokens in the logic string
        tokens = s.logic.split()
        and_count += tokens.count('AND')
        or_count += tokens.count('OR')

    total = and_count + or_count
    assert total > 0
    and_pct = and_count / total * 100
    assert 35 < and_pct < 65, f"AND: {and_pct:.1f}% — expected ~50% (±15%)"
```

- [ ] **Step 2: Run to verify it fails on current grammar**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_grammar.py::test_entry_rule_distribution -v`
Expected: FAIL — current grammar has ~75% AND bias

- [ ] **Step 3: Update grammar**

In `grammar/bnf.py`, replace the `<entry_rule>` block (lines 29-38) with:

```python
    # Entry rules: balanced structure families (25% each)
    "<entry_rule>": [
        "<condition>",                                                      # simple
        "<condition> <logical_op> <condition>",                             # binary
        "<condition> <logical_op> <condition> <logical_op> <condition>",    # ternary
        "(<condition> <logical_op> <condition>) <logical_op> <condition>",  # grouped
    ],

    # Logical operator: evolves independently of structure
    "<logical_op>": ["AND", "OR"],
```

- [ ] **Step 4: Run all grammar tests**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_grammar.py tests/test_mapper.py -v`
Expected: All pass including new distribution tests

- [ ] **Step 5: Commit**

```bash
git add grammar/bnf.py tests/test_grammar.py
git commit -m "feat: decouple entry_rule structure from logical operators — balanced 25% distribution"
```

---

### Task 4: Operators — Two-Point Crossover + Mutation Schedule

**Files:**
- Modify: `evolution/operators.py`
- Create: `tests/test_operators_v2.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_operators_v2.py`:

```python
"""Tests for v2 operators: two-point crossover + mutation schedule."""

import random
import pytest


class TestTwoPointCrossover:
    def test_interior_block_exchanged(self):
        from evolution.operators import crossover
        random.seed(42)
        p1 = list(range(10))          # [0,1,2,3,4,5,6,7,8,9]
        p2 = list(range(100, 110))    # [100,101,...,109]
        c1, c2 = crossover(p1, p2)
        # Children should have a contiguous block from the other parent
        # c1 should have some values from p2 in the middle
        has_p2_block = any(v >= 100 for v in c1[1:-1])
        assert has_p2_block, "c1 should have interior block from p2"

    def test_children_same_length(self):
        from evolution.operators import crossover
        p1 = [1] * 20
        p2 = [2] * 20
        c1, c2 = crossover(p1, p2)
        assert len(c1) == 20
        assert len(c2) == 20

    def test_short_genome_returns_copies(self):
        from evolution.operators import crossover
        p1 = [1, 2]
        p2 = [3, 4]
        c1, c2 = crossover(p1, p2)
        assert c1 == [1, 2] or c1 == [3, 4]  # with 2 elements, two-point degenerates


class TestMutationSchedule:
    def test_early_gen_high_exploration(self):
        """Gen 0: ~60% random jumps."""
        from evolution.operators import mutate
        random.seed(42)
        genome = [128] * 200
        mutated = mutate(genome, rate=1.0, generation=0, max_generations=100)
        changes = sum(1 for a, b in zip(genome, mutated) if a != b)
        # With rate=1.0, all should mutate. Count random jumps (big changes)
        big_changes = sum(1 for a, b in zip(genome, mutated) if abs(a - b) > 3 and a != b)
        ratio = big_changes / max(changes, 1)
        assert ratio > 0.4, f"Expected ~60% random jumps at gen 0, got {ratio:.1%}"

    def test_late_gen_low_exploration(self):
        """Gen 100: ~20% random jumps."""
        from evolution.operators import mutate
        random.seed(42)
        genome = [128] * 200
        mutated = mutate(genome, rate=1.0, generation=100, max_generations=100)
        changes = sum(1 for a, b in zip(genome, mutated) if a != b)
        big_changes = sum(1 for a, b in zip(genome, mutated) if abs(a - b) > 3 and a != b)
        ratio = big_changes / max(changes, 1)
        assert ratio < 0.4, f"Expected ~20% random jumps at gen 100, got {ratio:.1%}"

    def test_finetune_range(self):
        """Fine-tune produces changes in ±{1,2,3}."""
        from evolution.operators import mutate
        random.seed(42)
        genome = [128] * 1000
        # Late gen = mostly fine-tune
        mutated = mutate(genome, rate=1.0, generation=100, max_generations=100)
        small_deltas = []
        for a, b in zip(genome, mutated):
            delta = (b - a + 128) % 256 - 128  # signed delta with wrapping
            if 0 < abs(delta) <= 3:
                small_deltas.append(abs(delta))
        # Should have some of each
        assert 1 in small_deltas
        assert 2 in small_deltas
        assert 3 in small_deltas

    def test_no_swap_mutation(self):
        """Swap mutation removed — no adjacent swaps should happen deterministically."""
        from evolution.operators import mutate
        # Verify function signature accepts generation params
        genome = [1, 2, 3, 4, 5]
        result = mutate(genome, rate=0.0, generation=0, max_generations=100)
        assert result == genome  # rate=0 → no mutations
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_operators_v2.py -v`
Expected: TypeError — `mutate()` got unexpected keyword argument 'generation'

- [ ] **Step 3: Implement new operators**

Replace entire `evolution/operators.py`:

```python
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
    Returns two children of same length as respective parents.
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

    Args:
        genome: Integer codon vector
        rate: Per-codon mutation probability
        generation: Current generation number
        max_generations: Total planned generations
    """
    result = genome[:]

    # Exploration ratio decays linearly: 0.6 → 0.2
    progress = min(generation / max(max_generations, 1), 1.0)
    explore_ratio = 0.6 - 0.4 * progress

    for i in range(len(result)):
        if random.random() >= rate:
            continue

        if random.random() < explore_ratio:
            # Random jump — structural exploration
            result[i] = random.randint(0, 255)
        else:
            # Fine-tune — parameter refinement (±1, ±2, or ±3)
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            result[i] = (result[i] + delta) % 256

    return result
```

- [ ] **Step 4: Run all operator tests (old + new)**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_operators_v2.py tests/test_engine.py::TestCrossover tests/test_engine.py::TestMutate -v`
Expected: New tests pass. Old tests in test_engine.py may need the `generation` kwarg — check if they fail and fix.

- [ ] **Step 5: Fix any old tests that broke**

If `tests/test_engine.py` has crossover/mutate tests that call the old signatures, update them to pass `generation=0, max_generations=100` to `mutate()`.

- [ ] **Step 6: Commit**

```bash
git add evolution/operators.py tests/test_operators_v2.py tests/test_engine.py
git commit -m "feat: two-point crossover + generation-dependent mutation schedule"
```

---

## Chunk 3: Evaluation Pipeline (Cache + Fitness + Sampling)

### Task 5: Evaluation Cache

**Files:**
- Create: `evolution/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cache.py`:

```python
"""Tests for per-window evaluation cache."""

import pytest


class TestEvalCache:
    def test_cache_hit(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        genome = (1, 2, 3, 4, 5)
        window_id = "w_0_8640"
        metrics = {'sortino': 1.5, 'return_pct': 10.0, 'max_dd': -0.15, 'n_trades': 25}
        cache.put(genome, window_id, metrics)
        result = cache.get(genome, window_id)
        assert result == metrics

    def test_cache_miss(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        assert cache.get((1, 2, 3), "w_0") is None

    def test_evict_old_windows(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        cache.put((1,), "old_window", {'sortino': 1.0})
        cache.put((1,), "active_window", {'sortino': 2.0})
        cache.evict_except(active_windows={"active_window"})
        assert cache.get((1,), "old_window") is None
        assert cache.get((1,), "active_window") is not None

    def test_stats(self):
        from evolution.cache import EvalCache
        cache = EvalCache()
        cache.put((1,), "w1", {'sortino': 1.0})
        cache.record_hit()
        cache.record_miss()
        cache.record_miss()
        stats = cache.stats()
        assert stats['entries'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert abs(stats['hit_rate'] - 1/3) < 0.01
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_cache.py -v`
Expected: ImportError

- [ ] **Step 3: Implement cache**

Create `evolution/cache.py`:

```python
"""
Per-window evaluation cache for evolution.

Keys: (genome_tuple, window_id) → per-window metrics dict.
Eviction: purge entries for windows no longer in active set.
"""

from typing import Dict, Optional, Set, Tuple


class EvalCache:
    """Cache backtest results keyed by (genome, window_id)."""

    def __init__(self):
        self._cache: Dict[Tuple[tuple, str], dict] = {}
        self._hits = 0
        self._misses = 0

    def get(self, genome: tuple, window_id: str) -> Optional[dict]:
        """Look up cached metrics. Returns None on miss."""
        return self._cache.get((genome, window_id))

    def put(self, genome: tuple, window_id: str, metrics: dict):
        """Store metrics for a (genome, window) pair."""
        self._cache[(genome, window_id)] = metrics

    def evict_except(self, active_windows: Set[str]):
        """Remove entries for windows not in active set."""
        to_remove = [k for k in self._cache if k[1] not in active_windows]
        for k in to_remove:
            del self._cache[k]

    def record_hit(self):
        self._hits += 1

    def record_miss(self):
        self._misses += 1

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            'entries': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total > 0 else 0.0,
        }

    def reset_counters(self):
        self._hits = 0
        self._misses = 0
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_cache.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add evolution/cache.py tests/test_cache.py
git commit -m "feat: per-window evaluation cache with generation-based eviction"
```

---

### Task 6: Fitness — Multi-Objective Per-Window Evaluation

**Files:**
- Modify: `evolution/fitness.py`

This is the most critical change. We refactor `evaluate_strategy` to:
1. Return per-window metrics (not aggregate)
2. Compute objectives as median across windows
3. Compute constraint violations (soft, not hard FAIL_FITNESS)
4. Remove scalar fitness composition (CAGR bonus etc.)

- [ ] **Step 1: Add new function `evaluate_strategy_nsga2` alongside existing one**

We keep the old `evaluate_strategy` for backward compatibility and add the new one. Add at the end of `evolution/fitness.py`:

```python
def evaluate_single_window(strategy: Strategy, window_df: pd.DataFrame,
                           config: dict, tf_data=None) -> Optional[dict]:
    """
    Evaluate strategy on a single window. Returns metrics dict or None on error.

    This is the atomic building block for NSGA-II evaluation.
    """
    costs_config = config.get('costs', {
        'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
        'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0,
    })
    atr_period = config.get('exits', {}).get('atr_period', 14)

    try:
        equity, trades = _run_single_window(
            strategy, window_df, costs_config, atr_period, tf_data=tf_data
        )
    except Exception:
        return None

    if not trades:
        return {'sortino': 0.0, 'return_pct': 0.0, 'max_dd': 0.0, 'n_trades': 0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'expectancy': 0.0}

    n_trades = len(trades)
    returns = calculate_returns(equity).dropna()

    if len(returns) < 5:
        return None

    sortino = calculate_sortino_ratio(returns, BARS_PER_YEAR_15M)
    sortino = max(min(sortino, 10.0), -10.0)

    start_eq = equity.iloc[0]
    end_eq = equity.iloc[-1]
    return_pct = (end_eq - start_eq) / start_eq * 100.0

    dd = max_drawdown(equity)

    winning = sum(1 for t in trades if t['pnl_pct'] > 0)
    win_rate = winning / n_trades
    winning_pnl = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    losing_pnl = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    pf = winning_pnl / max(losing_pnl, 1e-10)
    avg_win = winning_pnl / max(winning, 1)
    avg_loss = losing_pnl / max(n_trades - winning, 1)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    return {
        'sortino': sortino,
        'return_pct': return_pct,
        'max_dd': dd,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': pf,
        'expectancy': expectancy,
    }


def compute_objectives(strategy: Strategy, window_metrics: List[dict],
                       parsimony_coeff: float = 0.02) -> None:
    """
    Compute NSGA-II objectives from per-window metrics. Sets fields in-place.

    Objectives:
      - obj1: median(sortino) - parsimony * n_nodes
      - obj2: median(return_pct)

    Constraints (violation > 0 means infeasible):
      - MaxDD > 40% in any window
      - n_trades < 10 in any window
    """
    if not window_metrics:
        strategy.objectives = (-999.0, -999.0)
        strategy.stability = -999.0
        strategy.constraint_violation = 100.0
        return

    sortinos = [m['sortino'] for m in window_metrics]
    returns = [m['return_pct'] for m in window_metrics]
    max_dds = [abs(m['max_dd']) for m in window_metrics]
    trade_counts = [m['n_trades'] for m in window_metrics]

    # Objectives: median across windows
    sortinos_sorted = sorted(sortinos)
    returns_sorted = sorted(returns)
    mid = len(sortinos_sorted) // 2

    median_sortino = sortinos_sorted[mid]
    median_return = returns_sorted[mid]

    # Parsimony on sortino objective
    obj1 = median_sortino - parsimony_coeff * strategy.n_nodes
    obj2 = median_return

    strategy.objectives = (obj1, obj2)

    # Stability = -std(sortino)
    if len(sortinos) > 1:
        mean_s = sum(sortinos) / len(sortinos)
        var_s = sum((s - mean_s) ** 2 for s in sortinos) / (len(sortinos) - 1)
        strategy.stability = -(var_s ** 0.5)
    else:
        strategy.stability = 0.0

    # Constraint violations
    cv = 0.0
    for dd in max_dds:
        if dd > 0.40:
            cv += (dd - 0.40)
    for tc in trade_counts:
        if tc < 10:
            cv += (10 - tc) * 0.1
    strategy.constraint_violation = cv

    # Store per-window metrics for analysis
    strategy.window_metrics = window_metrics
    strategy.n_trades = sum(trade_counts)

    # Keep legacy fitness tuple for backward compat (archive etc.)
    strategy.fitness = strategy.objectives
    strategy.metrics = {
        'sortino': median_sortino,
        'return_pct': median_return,
        'max_dd': max(max_dds) if max_dds else 0.0,
        'n_trades': sum(trade_counts),
        'stability': strategy.stability,
        'window_sortinos': sortinos,
        'window_returns': returns,
    }
```

- [ ] **Step 2: Write tests for new functions**

Add to a new file `tests/test_fitness_nsga2.py`:

```python
"""Tests for NSGA-II fitness functions."""

import pytest
from unittest.mock import MagicMock
from strategy.phenotype import Strategy


def _make_strategy(n_nodes=2):
    s = MagicMock(spec=Strategy)
    s.n_nodes = n_nodes
    s.genome = [1, 2, 3]
    s.objectives = (-999.0, -999.0)
    s.stability = -999.0
    s.constraint_violation = 0.0
    s.fitness = (-999.0, -999.0)
    s.metrics = None
    s.window_metrics = None
    s.n_trades = 0
    return s


class TestComputeObjectives:
    def test_median_of_5_windows(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy(n_nodes=2)
        metrics = [
            {'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10, 'n_trades': 20,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 2.0, 'return_pct': 10.0, 'max_dd': -0.15, 'n_trades': 25,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 3.0, 'return_pct': 15.0, 'max_dd': -0.20, 'n_trades': 30,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 1.5, 'return_pct': 8.0, 'max_dd': -0.12, 'n_trades': 22,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 2.5, 'return_pct': 12.0, 'max_dd': -0.18, 'n_trades': 28,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
        ]
        compute_objectives(s, metrics, parsimony_coeff=0.02)
        # Sorted sortinos: [1.0, 1.5, 2.0, 2.5, 3.0] → median = 2.0
        # Sorted returns: [5.0, 8.0, 10.0, 12.0, 15.0] → median = 10.0
        assert abs(s.objectives[0] - (2.0 - 0.02 * 2)) < 0.01  # median - parsimony
        assert abs(s.objectives[1] - 10.0) < 0.01

    def test_feasible_when_constraints_met(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [{'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.30,
                     'n_trades': 15, 'win_rate': 0.5, 'profit_factor': 1.5,
                     'expectancy': 0.01}]
        compute_objectives(s, metrics)
        assert s.constraint_violation == 0.0

    def test_infeasible_high_dd(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [{'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.50,
                     'n_trades': 15, 'win_rate': 0.5, 'profit_factor': 1.5,
                     'expectancy': 0.01}]
        compute_objectives(s, metrics)
        assert s.constraint_violation > 0  # DD 50% > 40% threshold

    def test_infeasible_low_trades(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [{'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10,
                     'n_trades': 5, 'win_rate': 0.5, 'profit_factor': 1.5,
                     'expectancy': 0.01}]
        compute_objectives(s, metrics)
        assert s.constraint_violation > 0  # 5 trades < 10 threshold

    def test_empty_metrics(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        compute_objectives(s, [])
        assert s.objectives == (-999.0, -999.0)
        assert s.constraint_violation > 0

    def test_stability_computed(self):
        from evolution.fitness import compute_objectives
        s = _make_strategy()
        metrics = [
            {'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10, 'n_trades': 20,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
            {'sortino': 1.0, 'return_pct': 5.0, 'max_dd': -0.10, 'n_trades': 20,
             'win_rate': 0.5, 'profit_factor': 1.5, 'expectancy': 0.01},
        ]
        compute_objectives(s, metrics)
        assert s.stability == 0.0  # zero variance → stability = 0
```

- [ ] **Step 3: Run all tests**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add evolution/fitness.py tests/test_fitness_nsga2.py
git commit -m "feat: add evaluate_single_window + compute_objectives for NSGA-II multi-objective eval"
```

---

### Task 7: Window Sampling — 3-Month Windows + Partial Rotation

**Files:**
- Modify: `backtest/sampling.py`

- [ ] **Step 1: Add new sampling function alongside existing one**

Add to `backtest/sampling.py`:

```python
def sample_windows_with_rotation(data: pd.DataFrame,
                                  n_windows: int = 5,
                                  window_bars: int = 8640,
                                  previous_windows: List[Tuple[int, str]] = None,
                                  keep_ratio: float = 0.6,
                                  ) -> List[Tuple[pd.DataFrame, str]]:
    """
    Sample windows with partial rotation for NSGA-II evaluation.

    Returns list of (DataFrame, window_id) tuples.
    window_id is "w_{start}_{bars}" — used as cache key.

    Args:
        data: Full training OHLCV
        n_windows: Total windows per generation
        window_bars: Bars per window (8640 = ~3 months at 15m)
        previous_windows: [(df, window_id), ...] from previous generation
        keep_ratio: Fraction of windows to keep from previous gen (default 0.6 = 3/5)
    """
    total_bars = len(data)
    if total_bars < window_bars:
        logger.warning(f"Data ({total_bars}) shorter than window ({window_bars})")
        wid = f"w_0_{window_bars}"
        return [(data, wid)]

    max_start = total_bars - window_bars

    # Build pool of all valid start positions
    all_starts = list(range(0, max_start + 1, window_bars))  # non-overlapping grid
    if not all_starts:
        all_starts = [0]

    # Determine which windows to keep from previous generation
    kept = []
    if previous_windows:
        n_keep = int(n_windows * keep_ratio)
        kept = random.sample(previous_windows, min(n_keep, len(previous_windows)))

    # Sample fresh windows for remaining slots
    kept_ids = {wid for _, wid in kept}
    n_fresh = n_windows - len(kept)

    available = [s for s in all_starts if f"w_{s}_{window_bars}" not in kept_ids]
    if len(available) < n_fresh:
        # Fall back to random starts if not enough grid positions
        fresh_starts = [random.randint(0, max_start) for _ in range(n_fresh)]
    else:
        fresh_starts = random.sample(available, n_fresh)

    fresh = [(data.iloc[s:s + window_bars].copy(), f"w_{s}_{window_bars}")
             for s in fresh_starts]

    result = kept + fresh
    logger.debug(f"Windows: {len(kept)} kept + {len(fresh)} fresh = {len(result)} total")
    return result
```

- [ ] **Step 2: Write test for partial rotation**

Add `tests/test_sampling_v2.py`:

```python
"""Tests for window sampling with partial rotation."""

import pandas as pd
import numpy as np
import pytest


def _make_data(n_bars=50000):
    """Create synthetic OHLCV data for sampling tests."""
    idx = pd.date_range('2022-01-01', periods=n_bars, freq='15min')
    return pd.DataFrame({
        'Open': np.random.randn(n_bars).cumsum() + 100,
        'High': np.random.randn(n_bars).cumsum() + 101,
        'Low': np.random.randn(n_bars).cumsum() + 99,
        'Close': np.random.randn(n_bars).cumsum() + 100,
        'Volume': np.random.rand(n_bars) * 1000,
    }, index=idx)


class TestSampleWindowsWithRotation:
    def test_returns_correct_count(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        result = sample_windows_with_rotation(data, n_windows=5, window_bars=8640)
        assert len(result) == 5

    def test_returns_tuples_with_ids(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        result = sample_windows_with_rotation(data, n_windows=3, window_bars=4320)
        for df, wid in result:
            assert isinstance(df, pd.DataFrame)
            assert isinstance(wid, str)
            assert wid.startswith("w_")

    def test_partial_rotation_keeps_some(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        # First call
        gen1 = sample_windows_with_rotation(data, n_windows=5, window_bars=4320)
        # Second call with previous windows
        gen2 = sample_windows_with_rotation(
            data, n_windows=5, window_bars=4320,
            previous_windows=gen1, keep_ratio=0.6
        )
        gen1_ids = {wid for _, wid in gen1}
        gen2_ids = {wid for _, wid in gen2}
        # Should keep ~3 from gen1
        kept = gen1_ids & gen2_ids
        assert len(kept) >= 2  # at least 2 kept (some randomness)

    def test_window_size_correct(self):
        from backtest.sampling import sample_windows_with_rotation
        data = _make_data(50000)
        result = sample_windows_with_rotation(data, n_windows=3, window_bars=8640)
        for df, _ in result:
            assert len(df) == 8640
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/test_sampling_v2.py -v`
Expected: All 4 pass

- [ ] **Step 4: Commit**

```bash
git add backtest/sampling.py
git commit -m "feat: add sample_windows_with_rotation — 3-month windows with partial rotation for caching"
```

---

## Chunk 4: Engine Integration

### Task 8: Engine Rewrite — NSGA-II Generation Loop

**Files:**
- Modify: `evolution/engine.py`

This is the integration task. The engine's `step()` method is completely rewritten.

- [ ] **Step 1: Rewrite engine.py**

Replace the content of `evolution/engine.py` with the NSGA-II-based engine. Keep the same class interface (`EvolutionEngine`, `initialize`, `step`, `run`) so callers don't break:

```python
"""
Evolution Engine v2 — NSGA-II with evaluation caching and MAP-Elites.

Replaces v1's scalar-fitness tournament-selection loop.
"""

import random
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from grammar.mapper import decode
from strategy.phenotype import Strategy
from strategy.parameters import random_genome, GENOME_LENGTH
from evolution.operators import crossover, mutate
from evolution.nsga2 import non_dominated_sort, select_parents, select_by_stability
from evolution.fitness import evaluate_single_window, compute_objectives, FAIL_FITNESS
from evolution.cache import EvalCache
from evolution.archive import MAPElitesArchive
from backtest.sampling import sample_windows_with_rotation
from data.multi_timeframe import prepare_multi_tf_data

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Stats for one generation."""
    generation: int
    best_sortino: float
    best_return: float
    median_sortino: float
    front1_size: int
    valid_count: int
    total_count: int
    cache_hit_rate: float
    archive_coverage: float
    elapsed_seconds: float


@dataclass
class EvolutionResult:
    """Result of a full evolution run."""
    pareto_front: List[Strategy]
    all_strategies: List[Strategy]
    archive: MAPElitesArchive
    history: List[GenerationStats]
    total_evaluations: int
    final_generation: int


class EvolutionEngine:
    """
    NSGA-II Grammatical Evolution engine.

    Usage:
        engine = EvolutionEngine(config, data)
        engine.initialize(pop_size=200)
        result = engine.run(n_generations=100, patience=30)
    """

    def __init__(self, config: dict, data):
        self.config = config
        self.data = data
        self.population: List[Strategy] = []
        self.generation: int = 0
        self.history: List[GenerationStats] = []
        self.total_evaluations: int = 0

        evo_cfg = config.get('evolution', {})
        self.mutation_rate = evo_cfg.get('mutation_rate', 0.1)
        self.crossover_rate = evo_cfg.get('crossover_rate', 0.8)
        self.genome_length = evo_cfg.get('genome_length', GENOME_LENGTH)
        self.n_windows = evo_cfg.get('n_windows_per_gen', 5)
        self.window_bars = evo_cfg.get('window_bars', 8640)
        self.max_generations = evo_cfg.get('max_generations', 100)
        self.archive_parent_pct = evo_cfg.get('archive_parent_pct', 0.10)
        self.parsimony_coeff = config.get('fitness', {}).get('parsimony_coefficient', 0.02)

        self.cache = EvalCache()
        self.archive = MAPElitesArchive()
        self._prev_windows: List[Tuple] = []

    def initialize(self, pop_size: int):
        """Generate initial random population."""
        self.population = []
        attempts = 0
        max_attempts = pop_size * 5

        while len(self.population) < pop_size and attempts < max_attempts:
            genome = random_genome(self.genome_length)
            strategy = decode(genome)
            if strategy is not None:
                self.population.append(strategy)
            attempts += 1

        if len(self.population) < pop_size:
            logger.warning(f"Only {len(self.population)}/{pop_size} valid strategies")

        logger.info(f"Initialized: {len(self.population)} strategies")

    def _evaluate_population(self, strategies: List[Strategy],
                             windows: List[Tuple]) -> int:
        """
        Evaluate strategies on windows using cache.
        Sets objectives, stability, constraint_violation in-place.
        Returns number of new evaluations (cache misses).
        """
        new_evals = 0
        window_ids = {wid for _, wid in windows}

        for strategy in strategies:
            genome_key = tuple(strategy.genome)
            window_metrics = []

            for window_df, wid in windows:
                cached = self.cache.get(genome_key, wid)
                if cached is not None:
                    self.cache.record_hit()
                    window_metrics.append(cached)
                else:
                    self.cache.record_miss()
                    tf_data = prepare_multi_tf_data(window_df)
                    metrics = evaluate_single_window(
                        strategy, window_df, self.config, tf_data=tf_data
                    )
                    if metrics is not None:
                        self.cache.put(genome_key, wid, metrics)
                        window_metrics.append(metrics)
                    new_evals += 1

            compute_objectives(strategy, window_metrics, self.parsimony_coeff)

        return new_evals

    def _breed_offspring(self, parents: List[Strategy], n: int) -> List[Strategy]:
        """Generate n offspring from parents via crossover + mutation."""
        offspring = []
        while len(offspring) < n:
            if random.random() < self.crossover_rate and len(parents) >= 2:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                c1_genome, c2_genome = crossover(p1.genome, p2.genome)
            else:
                p1 = random.choice(parents)
                c1_genome = p1.genome[:]
                c2_genome = None

            c1_genome = mutate(c1_genome, self.mutation_rate,
                               self.generation, self.max_generations)
            s1 = decode(c1_genome)
            if s1 is not None:
                offspring.append(s1)

            if c2_genome is not None and len(offspring) < n:
                c2_genome = mutate(c2_genome, self.mutation_rate,
                                   self.generation, self.max_generations)
                s2 = decode(c2_genome)
                if s2 is not None:
                    offspring.append(s2)

        return offspring[:n]

    def step(self) -> GenerationStats:
        """Execute one NSGA-II generation."""
        t0 = time.time()
        pop_size = len(self.population)

        # 1. Sample windows with partial rotation
        windows = sample_windows_with_rotation(
            self.data, n_windows=self.n_windows,
            window_bars=self.window_bars,
            previous_windows=self._prev_windows,
        )
        self._prev_windows = windows

        # Evict stale cache entries
        active_ids = {wid for _, wid in windows}
        self.cache.evict_except(active_ids)
        self.cache.reset_counters()

        # 2. Select parents: 90% via binary tournament, 10% from archive
        n_archive = max(1, int(pop_size * self.archive_parent_pct))
        n_tournament = pop_size - n_archive

        tournament_parents = select_parents(self.population, n_tournament)
        archive_parents = self.archive.sample_for_reproduction(n_archive)
        all_parents = tournament_parents + archive_parents

        # 3. Breed offspring (λ = μ)
        offspring = self._breed_offspring(all_parents, pop_size)

        # 4. Evaluate offspring
        new_evals = self._evaluate_population(offspring, windows)
        self.total_evaluations += new_evals

        # Re-evaluate parents on new windows (cache hits for kept windows)
        parent_evals = self._evaluate_population(self.population, windows)
        self.total_evaluations += parent_evals

        # 5. Merge parents + offspring (μ+λ)
        combined = list(self.population) + offspring

        # 6. Non-dominated sort
        fronts = non_dominated_sort(combined)

        # 7. Fill new population from fronts
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                remaining = pop_size - len(new_pop)
                selected = select_by_stability(front, remaining)
                new_pop.extend(selected)
                break

        # 8. Update MAP-Elites archive
        for s in new_pop:
            if s.constraint_violation <= 0 and s.metrics:
                trades_per_month = s.n_trades / max(len(windows), 1) / 3.0  # ~3 months per window
                # Regime sortinos not available in per-window eval (no regime labels in windows).
                # Archive bins by direction heuristic: positive return_pct on bull-like windows, etc.
                # For now pass None — archive defaults to 'sideways', reducing regime dimension.
                # Full regime labeling can be added later when regime_detector is integrated.
                self.archive.try_add(s, trades_per_month, regime_sortinos=None)

        # Stats
        cache_stats = self.cache.stats()
        front1 = fronts[0] if fronts else []
        feasible = [s for s in new_pop if s.constraint_violation <= 0]

        if feasible:
            best = max(feasible, key=lambda s: s.objectives[0])
            best_return = max(feasible, key=lambda s: s.objectives[1])
            sortinos = [s.objectives[0] for s in feasible]
            median_s = sorted(sortinos)[len(sortinos) // 2]
        else:
            best = new_pop[0] if new_pop else None
            best_return = best
            median_s = -999.0

        elapsed = time.time() - t0
        stats = GenerationStats(
            generation=self.generation,
            best_sortino=best.objectives[0] if best else -999.0,
            best_return=best_return.objectives[1] if best_return else -999.0,
            median_sortino=median_s,
            front1_size=len(front1),
            valid_count=len(feasible),
            total_count=len(new_pop),
            cache_hit_rate=cache_stats['hit_rate'],
            archive_coverage=self.archive.coverage,
            elapsed_seconds=elapsed,
        )
        self.history.append(stats)

        logger.info(
            f"Gen {self.generation:3d} | "
            f"F1={stats.front1_size} feasible={stats.valid_count}/{stats.total_count} | "
            f"best_S={stats.best_sortino:+.2f} best_R={stats.best_return:+.1f}% | "
            f"cache={stats.cache_hit_rate:.0%} archive={stats.archive_coverage:.0%} | "
            f"{elapsed:.1f}s"
        )

        self.population = new_pop
        self.generation += 1
        return stats

    def run(self, n_generations: int, patience: int = 30) -> EvolutionResult:
        """
        Run full NSGA-II evolution.

        Patience tracks stagnation of front1 size + best Sortino.
        """
        best_metric = -999.0
        stagnation = 0

        logger.info(f"Starting NSGA-II evolution: pop={len(self.population)}, "
                    f"max_gen={n_generations}, patience={patience}")

        for gen in range(n_generations):
            stats = self.step()

            current_metric = stats.best_sortino + stats.front1_size * 0.01
            if current_metric > best_metric + 0.01:
                best_metric = current_metric
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                logger.info(f"Early stop at gen {gen} (stagnation={patience})")
                break

        # Collect Pareto front
        fronts = non_dominated_sort(self.population)
        pareto = fronts[0] if fronts else []
        pareto_feasible = [s for s in pareto if s.constraint_violation <= 0]
        pareto_sorted = sorted(pareto_feasible, key=lambda s: s.objectives[0], reverse=True)

        logger.info(f"Evolution complete: {self.generation} gens, "
                    f"{self.total_evaluations} evals, "
                    f"Pareto front: {len(pareto_sorted)} strategies")

        return EvolutionResult(
            pareto_front=pareto_sorted,
            all_strategies=self.population,
            archive=self.archive,
            history=self.history,
            total_evaluations=self.total_evaluations,
            final_generation=self.generation,
        )
```

- [ ] **Step 2: Rewrite test_engine.py for NSGA-II interface**

Replace tests in `tests/test_engine.py` that reference old fields. The new engine has:
- `GenerationStats`: `best_sortino`, `best_return`, `front1_size` (not `best_fitness`, `mean_fitness`)
- `EvolutionResult`: `pareto_front` (not `best_strategies`)
- No `elitism_pct` or `tournament_k` in config

Key tests to keep/rewrite:

```python
"""Tests for NSGA-II Evolution Engine."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    """3 months of synthetic OHLCV data."""
    n = 8640  # ~3 months at 15m
    idx = pd.date_range('2024-01-01', periods=n, freq='15min')
    close = np.random.randn(n).cumsum() + 50000
    return pd.DataFrame({
        'Open': close + np.random.randn(n) * 10,
        'High': close + abs(np.random.randn(n) * 50),
        'Low': close - abs(np.random.randn(n) * 50),
        'Close': close,
        'Volume': np.random.rand(n) * 1e6,
    }, index=idx)


@pytest.fixture
def config():
    return {
        'evolution': {
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'genome_length': 50,
            'n_windows_per_gen': 3,
            'window_bars': 2880,
            'max_generations': 5,
            'archive_parent_pct': 0.10,
        },
        'costs': {'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
                  'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0},
        'exits': {'atr_period': 14},
        'fitness': {'parsimony_coefficient': 0.02},
    }


class TestInitialize:
    def test_creates_valid_population(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        assert len(engine.population) == 10
        for s in engine.population:
            assert s.direction in ('LONG', 'SHORT')
            assert len(s.conditions) > 0


class TestStep:
    def test_returns_stats(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        stats = engine.step()
        assert stats.generation == 0
        assert hasattr(stats, 'best_sortino')
        assert hasattr(stats, 'front1_size')
        assert stats.total_count == 10

    def test_population_size_maintained(self, sample_df, config):
        from evolution.engine import EvolutionEngine
        engine = EvolutionEngine(config, sample_df)
        engine.initialize(pop_size=10)
        engine.step()
        assert len(engine.population) == 10
```

Read existing test_engine.py first to preserve any other useful tests, then replace old field references.

- [ ] **Step 3: Run tests**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add evolution/engine.py tests/test_engine.py
git commit -m "feat: rewrite evolution engine with NSGA-II, evaluation caching, and MAP-Elites integration"
```

---

### Task 9: Archive Update — Use Objectives Instead of fitness[0]

**Files:**
- Modify: `evolution/archive.py`

- [ ] **Step 1: Update archive to use objectives**

In `evolution/archive.py`, the `try_add` method and `summary` currently compare `strategy.fitness[0]`. Since we set `strategy.fitness = strategy.objectives` in `compute_objectives`, this actually works as-is. But for clarity and future-proofing, update comparisons:

In `try_add` (line 80), change:
```python
if strategy.fitness[0] <= -999.0:
```
to:
```python
if strategy.objectives[0] <= -999.0:
```

In `try_add` (line 89), change:
```python
if cell not in self.grid or strategy.fitness[0] > self.grid[cell].fitness[0]:
```
to:
```python
if cell not in self.grid or strategy.objectives[0] > self.grid[cell].objectives[0]:
```

In `summary` (line 130), change:
```python
fitnesses = [s.fitness[0] for s in self.grid.values()]
```
to:
```python
fitnesses = [s.objectives[0] for s in self.grid.values()]
```

In `best_per_regime` (line 114), change:
```python
result[regime] = max(cells, key=lambda x: x[1].fitness[0])[1]
```
to:
```python
result[regime] = max(cells, key=lambda x: x[1].objectives[0])[1]
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/juanma/cripto_ga/cripto_ga && python3 -m pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add evolution/archive.py
git commit -m "refactor: archive uses objectives instead of fitness[0] for NSGA-II compatibility"
```

---

## Chunk 5: Data Split + Portfolio Cleanup + Integration Test

### Task 10: Data Split Configuration

**Files:**
- Modify: config files / engine initialization

- [ ] **Step 1: Verify data availability**

Check if we can load data from 2022-01 to 2026-02:
```bash
cd /Users/juanma/cripto_ga/cripto_ga && python3 -c "
from loader import load_data
df = load_data('BTC/USDT', '15m', start='2022-01-01', end='2026-02-28')
print(f'Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}')
"
```

Expected: Successfully loads ~140,000+ bars

- [ ] **Step 2: Create OTS exclusion assertion test**

Add to `tests/test_data_split.py`:

```python
"""Test that OTS data is never used during evolution."""

import pandas as pd
import pytest

OTS_START = '2025-10-01'
TRAIN_END = '2025-09-30'


def test_training_data_excludes_ots():
    """Training data must end before OTS start."""
    from loader import load_data
    data = load_data('BTC/USDT', '15m', start='2022-01-01', end=TRAIN_END)
    assert data.index[-1] <= pd.Timestamp(TRAIN_END + ' 23:59:59')
    assert data.index[-1] < pd.Timestamp(OTS_START)


def test_ots_data_starts_after_training():
    """OTS data must start at or after OTS_START."""
    from loader import load_data
    data = load_data('BTC/USDT', '15m', start=OTS_START, end='2026-02-28')
    assert data.index[0] >= pd.Timestamp(OTS_START)
```

- [ ] **Step 3: Document new dates in config**

The data split is controlled by what gets passed to `EvolutionEngine(config, data)`. The caller loads training data up to 2025-09-30 only. Key dates:
- Training: 2022-01-01 to 2025-09-30
- OTS: 2025-10-01 to 2026-02-28

- [ ] **Step 4: Commit**

```bash
git add tests/test_data_split.py
git commit -m "feat: add OTS exclusion assertion tests + update data split dates"
```

---

### Task 11: Portfolio Cleanup

**Files:**
- Modify: `live/config.py`

- [ ] **Step 1: Read current portfolio config**

Read `live/config.py` and identify the two strategies to remove:
- `bnb_seed42_s13_cmaes` (BNB L2*)
- `bnb_seed777_s25_cmaes` (BNB L3*)

- [ ] **Step 2: Remove overfit strategies**

Comment out or remove the two entries from the PORTFOLIO list in `live/config.py`. Add a comment explaining why.

- [ ] **Step 3: Commit**

```bash
git add live/config.py
git commit -m "fix: remove BNB CMA-ES L2*/L3* from portfolio — confirmed overfitting on extended OTS"
```

---

### Task 12: Integration Smoke Test

**Files:**
- No new files — run existing pipeline end-to-end

- [ ] **Step 1: Run a minimal evolution (5 gen, pop 20)**

```bash
cd /Users/juanma/cripto_ga/cripto_ga && python3 -c "
import logging
logging.basicConfig(level=logging.INFO)
from loader import load_data
from evolution.engine import EvolutionEngine

# Load small slice for smoke test
data = load_data('BTC/USDT', '15m', start='2024-01-01', end='2024-12-31')
print(f'Data: {len(data)} bars')

config = {
    'evolution': {
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'genome_length': 50,
        'n_windows_per_gen': 3,
        'window_bars': 4320,  # ~1 month for quick test
        'max_generations': 5,
        'archive_parent_pct': 0.10,
    },
    'costs': {'fees_bps_long': 1.0, 'fees_bps_short': 1.0,
              'slippage_bps_long': 1.0, 'slippage_bps_short': 1.0},
    'exits': {'atr_period': 14},
    'fitness': {'parsimony_coefficient': 0.02},
}

engine = EvolutionEngine(config, data)
engine.initialize(pop_size=20)
result = engine.run(n_generations=5, patience=10)

print(f'Pareto front: {len(result.pareto_front)} strategies')
print(f'Total evals: {result.total_evaluations}')
print(f'Archive coverage: {result.archive.coverage:.0%}')
for s in result.pareto_front[:3]:
    print(f'  {s.direction} | Sortino={s.objectives[0]:.2f} Return={s.objectives[1]:.1f}% | {s.n_nodes} nodes')
"
```

Expected: Completes without errors. Pareto front has at least 1 strategy. Cache hit rate > 0% after gen 2.

- [ ] **Step 2: Fix any integration issues**

If errors occur, diagnose and fix. Common issues:
- Import errors from refactored modules
- Type mismatches between old and new Strategy fields
- Window sampling returning unexpected format

- [ ] **Step 3: Final commit**

```bash
git add -A && git commit -m "feat: NSGA-II evolution pipeline complete — integration verified"
```

---

## Post-Implementation: Full Re-Evolution

After all tasks are complete and tests pass:

1. **Full evolution run** on BTC, ETH, BNB separately:
   ```bash
   python3 -c "
   # BTC evolution with full training data
   from loader import load_data
   from evolution.engine import EvolutionEngine
   data = load_data('BTC/USDT', '15m', start='2022-01-01', end='2025-09-30')
   # ... config with pop=200, gen=100 ...
   result = engine.run(n_generations=100, patience=30)
   # Save results to JSON
   "
   ```

2. **OTS validation** on Oct 2025 — Feb 2026 (never seen during evolution)

3. **Statistical tests**: CPCV, DSR, PBO, Hansen SPA on Pareto front strategies

4. **Portfolio construction**: Select strategies from Pareto front for live deployment

These are separate sessions — not part of this plan.
