# Evolution Pipeline Paradigm Shift — Design Spec

**Date**: 2026-03-14
**Status**: Approved
**Context**: 13 fundamental weaknesses identified in evolution pipeline. CMA-ES overfitting confirmed on extended OTS. Portfolio needs cleanup + re-evolution with fixed pipeline.

---

## Problem Statement

The current evolution engine is essentially doing random search due to:
1. Fitness dominated by unbounded CAGR bonus (×10 multiplier)
2. Grammar bias — 75% of random genomes produce 2-3 condition AND-rules
3. Hard constraints kill potentially good strategies on unlucky windows
4. Crossover produces ~50% invalid children
5. Mutation too local (60% is ±1) for structural exploration
6. MAP-Elites tracks diversity but doesn't influence selection
7. Re-evaluates ALL individuals every generation (no caching)
8. Window rotation causes excessive fitness variance (2-3 windows)
9. CMA-ES overfits on small OTS windows (BNB L2*, L3* collapsed)

Extended OTS (9 months) revealed that 2 of 4 CMA-ES strategies collapsed from +48-64% to +0.4-0.6%.

---

## Design

### 1. NSGA-II Multi-Objective Selection

Replace scalar fitness with true multi-objective optimization.

**Supersedes CLAUDE.md Section 5** (which defined Primary: Sortino, Secondary: Calmar). Calmar is dropped because it's CAGR/MaxDD — the CAGR component distorts on short windows (3mo +15% → 75% CAGR annualized). Both new objectives are measured directly, no annualization.

**Two objectives:**
- **Objective 1: Median Sortino** across evaluation windows — risk-adjusted return quality. Parsimony pressure (-0.02 * n_nodes) subtracted from this objective to penalize complexity in ranking.
- **Objective 2: Median Return (%)** across evaluation windows — raw growth

**Hard constraints (eligibility, not fitness — relaxed from CLAUDE.md to cast wider net during evolution; strict thresholds apply at Layer 2 CPCV validation):**
- MaxDD > 40% in any window → infeasible (evolution). Layer 2 still requires < 30%.
- n_trades < 10 per window → infeasible (evolution). Layer 2 still requires >= 30 total.
- Infeasible individuals handled via constrained domination (Deb 2002): feasible always dominates infeasible. Among infeasible, rank by total constraint violation magnitude.

**Secondary sort within each Pareto front:** stability = -std(Sortino across windows). This replaces standard crowding distance. Rationale: crowding distance preserves spread in objective space, but our problem is fitness noise from window variance. Sorting by stability within fronts rewards strategies that perform consistently, which is more important than spread for our use case.

**Rationale:** Sortino alone could produce timid strategies. Return alone overfits. Together, the Pareto front ranges from conservative-consistent to aggressive-profitable. User selects from front based on risk appetite.

**Implementation (μ+λ with λ=μ, standard NSGA-II):**
- Generate `pop_size` offspring from parents (no separate elitism — NSGA-II front preservation subsumes it)
- Merge parents + offspring → pool of `2 * pop_size`
- Non-dominated sorting into fronts (F1, F2, F3...)
- Fill new population from fronts; when last included front exceeds remaining slots, sort by stability (not crowding distance) and take the most stable
- Binary tournament for parent selection: compare (front_rank, stability); lower rank wins, ties broken by higher stability

**Generation loop pseudocode:**
```
1. Sample 5 windows from pool
2. Evaluate offspring on all 5 windows (cache hits for unchanged genomes)
3. Merge parents (N) + offspring (N) → pool (2N)
4. Non-dominated sort pool → fronts F1, F2, ...
5. Fill new population from F1, then F2, ...; within last partial front, sort by stability
6. Update MAP-Elites archive with best-per-niche from new population
7. Draw 10% of parents from MAP-Elites archive, 90% via binary tournament
8. Generate N offspring via crossover + mutation (with generation-dependent schedule)
```

**Files:** `evolution/nsga2.py` (new), `evolution/selection.py` (refactor)

### 2. Grammar: Decouple Structure from Logic

Replace biased entry_rule productions with orthogonal design.

**Current (biased):**
```
<entry_rule> → 8 productions, 75% AND-heavy, AND/OR baked into structure
```

**Proposed:**
```
<entry_rule>:
  [0] <condition>                                                    # simple
  [1] <condition> <logical_op> <condition>                           # binary
  [2] <condition> <logical_op> <condition> <logical_op> <condition>  # ternary
  [3] (<condition> <logical_op> <condition>) <logical_op> <condition># grouped

<logical_op>: ["AND", "OR"]
```

**Why:**
- 4 productions → `codon % 4` is perfectly uniform
- AND/OR evolves independently of structure count
- Mutation ±1 on `<logical_op>` codón toggles AND↔OR directly
- Each codón controls ONE decision → cleaner gradient for GA

**Files:** `grammar/bnf.py` (modify)

### 3. Mutation: Exploration Schedule

Replace fixed 60/30/10 ratio with generation-dependent schedule.

**Proposed:**
```python
explore_ratio = 0.6 - 0.4 * (generation / max_generations)
# Gen 0:   60% random jumps, 40% fine-tune
# Gen 100: 20% random jumps, 80% fine-tune

# Two mutation types only (swap removed — near no-op in GE):
# random jump: genome[i] = random.randint(0, 255)
# fine-tune:   genome[i] += random.choice([-3, -2, -1, +1, +2, +3])  # uniform
```

**Rationale:** Simulates annealing. Early generations explore structure space broadly, later generations refine parameters of promising structures. ±3 fine-tune allows jumping between nearby indicator types, not just parameter tweaks. Uniform distribution over {-3..+3}\{0} gives equal chance to each step size.

**Files:** `evolution/operators.py` (modify)

### 4. Crossover: Two-Point

Replace one-point with two-point crossover.

```python
p1, p2 = sorted(random.sample(range(1, min_len), 2))
child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
```

**Rationale:** Marginal improvement. Exchanges interior block instead of entire tail. Not the bottleneck — included for completeness. Crossover destruction rate in GE is inherently high (~50%) regardless of method; selection compensates.

**Files:** `evolution/operators.py` (modify)

### 5. Multi-Window Evaluation with Caching

Replace 2-3 noisy windows with 5-window median + per-window cache.

**Window pool:** All non-overlapping windows of `window_bars` size (default 8640 = ~3 months at 15m) from training data. With training 2022-01 to 2025-09 (~44 months), this yields ~14 non-overlapping windows. Window size increased from 2880 (1 month) to 8640 (3 months) to reduce fitness variance and give strategies enough bars for meaningful trade counts.

**Evaluation:**
```
Each generation:
1. Sample 5 windows from the pool of ~14 (same 5 for entire generation → comparable)
2. For each individual, evaluate on all 5 windows
3. Objectives = median(Sortino), median(Return) across windows
4. Between generations: keep 3 windows, replace 2 fresh (partial rotation)
   → anti-overfitting via rotation + caching benefit from overlap
```

**Cache:**
```python
window_cache = {}  # (genome_tuple, window_id) → metrics
# Per-window granularity: genome unchanged + window seen before → cache hit
# Partial rotation (3/5 kept): ~60% hit rate for surviving individuals
# Eviction: after each generation, purge entries for windows no longer in active set
# Max cache size: bounded to pop_size * n_active_windows (~1000 entries)
```

**Why median, not mean:** Robust to outlier windows. A strategy that scores [1.5, 1.8, 1.6, 1.7, -0.5] has median=1.6 (good) vs mean=1.22 (penalized by one bad window). The bad window might be a flash crash — median rewards the strategy for being consistent in 4/5 cases.

**Files:** `evolution/engine.py` (modify), `evolution/cache.py` (new)

### 6. MAP-Elites as Diversity Reservoir

Keep MAP-Elites archive but integrate with NSGA-II selection.

**Current:** Archive tracks best-per-niche but never feeds back into selection.

**Proposed:**
```
Each generation:
1. NSGA-II selects parents from population
2. 10% of parents are drawn from MAP-Elites archive instead
3. After evaluation, best-per-niche individuals update the archive
4. Archive dimensions: frequency × complexity × regime (unchanged from current)
   - frequency: [low, medium, high] (trades/month)
   - complexity: [1-2, 3-4, 5-6, 7-8, 9+] (n_nodes)
   - regime: [bull, bear, sideways] (best regime for strategy)
   - Total: 3 × 5 × 3 = 45 cells
```

**Rationale:** NSGA-II preserves diversity in objective space (Sortino vs Return). MAP-Elites preserves diversity in behavior space (trading frequency, rule complexity, market regime). These are complementary — two strategies with identical Sortino/Return can behave completely differently. Dimensions kept as frequency × complexity × regime (matching current archive.py) rather than direction, because regime captures more behavioral variation.

**Files:** `evolution/archive.py` (modify), `evolution/engine.py` (modify)

### 7. Data Split for Re-Evolution

**Current:** Train 2023-01 to 2025-05, OTS 2025-06 to 2025-11 (6mo)

**Proposed:** Train 2022-01 to 2025-09, OTS 2025-10 to 2026-02 (5mo real data)

**Data provenance note for paper:** This is a NEW experiment round (Round 2). The original OTS period (Jun-Nov 2025) was sacred for Round 1 and is reported separately. For Round 2, that period becomes training data. The paper reports both rounds independently — Round 1 with original split, Round 2 with extended split. This is standard practice: the original OTS results stand as-is; the new experiment uses more data for a fresh evolution. No data leaks because Round 2 strategies are entirely new (re-evolved from scratch).

**Changes:**
- Training extended by ~16 months (more data, more regimes covered)
- OTS window: 5 months of real data (Oct 2025 - Feb 2026)
- All data is real Binance data — no synthetic, no future data
- Today is 2026-03-14, latest available complete month is Feb 2026

**Files:** config, `evolution/engine.py` (data loading)

### 8. Portfolio Cleanup

**Remove from live:** BNB L2* (bnb_seed42_s13_cmaes) and BNB L3* (bnb_seed777_s25_cmaes)
- Extended OTS: +64.6% → +0.4% and +48.6% → +0.6%
- Clear CMA-ES overfitting on small sample

**Keep (8 strategies):** All original + ETH CMA-ES (eth_seed777_s26_cmaes held up at +23.7%) + BNB CMA-ES L1 (bnb_seed123_s18_cmaes held up at +30.1%)

**After re-evolution:** Replace portfolio with new strategies from fixed pipeline, validated on extended OTS.

---

## What We're NOT Doing

- **S6A (alt data):** Cancelled — adds complexity without proven benefit
- **S10 (RL hybrid):** Deferred — fix fundamentals first
- **CMA-ES post-optimization:** Suspended until overfitting issue is understood
- **Regime filter in live:** Stays OFF (commit ce03133 — hurts validated strategies)
- **Voting/ensemble grammar:** Interesting idea, deferred to after base pipeline works
- **3+ NSGA-II objectives:** Would inflate Pareto front with pop=200, reducing selection pressure

---

## Success Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| Pareto front size | 10+ | 30+ |
| OTS median Sortino (front strategies) | > 0.5 | > 1.5 |
| OTS median Return (5mo) | > 5% | > 20% |
| OTS MaxDD | < 30% | < 15% |
| Cross-window consistency | 3/5 positive | 5/5 positive |
| Strategies passing all stat tests | >= 3 | >= 10 |
| Evolution time (100 gen, pop 200) | < 2h | < 1h |

---

## Implementation Order (with acceptance tests)

1. **NSGA-II selection** (`evolution/nsga2.py`)
   - Test: known 10-individual set with predefined objectives → verify fronts F1/F2/F3 match hand-calculated result
   - Test: constrained domination — infeasible individuals always behind feasible
   - Test: stability sort within front — most stable individual selected first

2. **Grammar fix** (`grammar/bnf.py`)
   - Test: generate 10,000 random genomes → `<entry_rule>` distribution is ~25% per family (±3%)
   - Test: `<logical_op>` distribution is ~50/50 AND/OR (±5%)
   - Test: existing grammar decode still produces valid strategies (no regression)

3. **Mutation schedule + two-point crossover** (`evolution/operators.py`)
   - Test: at gen=0, ~60% of mutations are random jumps; at gen=max, ~20%
   - Test: two-point crossover produces two valid children with exchanged interior block
   - Test: fine-tune only produces values in ±{1,2,3} range

4. **Multi-window eval + cache** (`evolution/engine.py`, `evolution/cache.py`)
   - Test: same genome + same window → cache hit (no re-evaluation)
   - Test: cache eviction removes entries for windows no longer in active set
   - Test: median of 5 windows matches expected value for known input

5. **MAP-Elites integration** (`evolution/archive.py`)
   - Test: 10% of parent selections come from archive (±2%)
   - Test: archive updates correctly with best-per-niche after each generation

6. **Data split update + portfolio cleanup**
   - Test: training data ends at 2025-09-30, OTS starts at 2025-10-01
   - Test: assert OTS data is never loaded during evolution

7. **Full re-evolution run + OTS validation**
   - Exit: Pareto front has 10+ strategies after 100 generations
   - Exit: at least 1 strategy with Sortino > 0.5 and Return > 5% on OTS

8. **Statistical tests on new strategies** (CPCV, DSR, PBO, Hansen SPA)
   - Exit: report all metrics honestly, even if negative
