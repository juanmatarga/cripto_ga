# CriptoGA - Evolutionary Discovery of Trading Strategies

## Project Vision
Research project for discovering statistically robust trading strategies in crypto markets
using evolutionary algorithms. The final deliverable is a **reproducible research paper**,
not a trading bot. Every result must survive rigorous statistical validation.

**Null hypothesis awareness**: It is entirely possible that no exploitable alpha exists in
BTC/USDT 15m. The v1 statistical tests already suggest this (Hansen SPA p=0.44, White RC
p=0.58). If the redesigned system reaches the same conclusion with better methodology,
the paper becomes "a rigorous framework for searching + evidence that alpha is absent in
this market/timeframe" — which is still a publishable and honest result. We must NOT fall
into the bias of searching until we force-fit alpha that doesn't exist.

## ULTIMATE OBJECTIVE
**Find statistically verified profitable patterns.** Every decision, every sprint, every
line of code must serve this goal. If a pattern doesn't survive rigorous out-of-sample
validation with Monte Carlo, CPCV, signal permutation, and walk-forward testing, it is
NOT a real pattern — no matter how good it looks in-sample. Alpha is rare and fragile.
Our job is to find it honestly or prove it doesn't exist.

## Core Principles
- **Simple and robust over complex and fragile**
- **No curve fitting** — all strategies must pass out-of-sample validation
- **Reproducibility** — fixed seeds, deterministic pipelines, documented experiments
- **Think like a quant researcher**, not a trading bot developer
- **Statistical rigor** — DSR > 0.95, PBO < 0.50, Hansen SPA p < 0.05
- **Honest results** — a negative result (no alpha found) is still a valid result
- **Real data ONLY** — ALL testing, validation, and experiments use real BTC/USDT data from Binance. NEVER use synthetic/random data for results. Synthetic data is only acceptable in unit tests.
- **Beware CMA-ES overfitting** — CMA-ES parameter tuning can overfit to specific market regimes. Always validate CMA-ES results on extended OTS periods.

## Current State: v1 (Legacy — Being Redesigned)

The v1 system uses a building-blocks GA with 59 hardcoded trading modules combined
via AND/OR logic. It **does not work**: patterns stagnate at generation ~4, Hansen SPA
p=0.44, White RC p=0.58. Nine root causes diagnosed:

1. **Search space vs evaluations**: 5.5M combinations, only 1500 evaluations (0.03%)
2. **Hardcoded parameters**: RSI period=14, threshold=30 etc. — GA can't tune them
3. **eval() bottleneck**: Bar-by-bar string evaluation limits population and generations
4. **Fixed evaluation windows**: Same 10 windows all generations → subtle overfitting
5. **Compressed fitness**: All viable patterns score 0.55-0.67 (no gradient for selection)
6. **Excessive selection pressure**: 16% elitism kills diversity by generation 4
7. **Artificial fitness normalization**: Maps everything to [0,1], destroying signal
8. **Immigration is a band-aid**: 20 immigrants arrive with fitness=-999, get evaluated
   once, fail to beat elite (~0.65), and are eliminated next generation. Doesn't solve
   the structural diversity problem — just delays convergence by one generation.
9. **Semantic constraints too restrictive**: Operators filter everything with
   bias_score < 0.5. This prevents garbage patterns but also blocks lateral exploration.
   Example: a LONG pattern using a bearish RSI_overbought as a contrarian filter is
   legitimate but gets rejected. The GE grammar in v2 resolves this by not imposing
   directional constraints on indicator combinations — validity comes from backtest
   results, not from semantic pre-filtering.

### Known Bugs in v1
- **bootstrap.py**: Bare `except: continue` silently swallows all errors. Results show
  UPI mean=2.4M, CAGR mean=2.2M — clearly corrupt. Bug may be deeper than just the
  bare except (possible overflow or wrong metric calculation).
- **white_rc.py**: Uses simple bootstrap instead of block bootstrap, inconsistent with
  bootstrap.py which correctly uses blocks. Violates autocorrelation assumptions.
- **main.py**: `warnings.filterwarnings('ignore')` at top hides real problems.
- **No evaluation caching**: Patterns that survive unchanged between generations get
  re-evaluated from scratch. With vectorized eval this matters less, but still wasteful.
- **Circular import avoidance**: Several files use imports inside functions to avoid
  circular dependencies — a sign of entangled architecture.

## Architecture: v2 (Planned Redesign)

### Grammatical Evolution + MAP-Elites + CPCV

```
data/               Data loading and market regime detection
grammar/            BNF grammar definition and codon-to-phenotype mapping
strategy/           Strategy representation and vectorized evaluation
evolution/          Evolution engine, operators, selection, archive
backtest/           Backtesting engine, exits, metrics, position sizing
validation/         CPCV, DSR, PBO, bootstrap, Hansen SPA, White RC, signal permutation
analysis/           Evolution analytics, Monte Carlo, visualization
reports/            LaTeX, pattern explanation, report generation
tests/              Unit and integration tests
```

### Key Design Decisions

1. **Grammatical Evolution over Building Blocks**
   - Genome = vector of integers (codons)
   - BNF grammar maps codons to strategy phenotype
   - Parameters (periods, thresholds) evolve alongside structure
   - Genetic operators work on integers — always produce valid offspring
   - No directional semantic constraints — the grammar produces syntactically valid
     strategies; fitness (not pre-filtering) determines if they work

2. **Vectorized Evaluation over eval()**
   - Indicators precomputed as DataFrame columns
   - Signal generation via NumPy boolean operations
   - Target: 100-500x speedup, enabling pop=200+, gen=100+

3. **CPCV over Fixed Windows (post-evolution validation)**
   - Combinatorial Purged Cross-Validation with embargo
   - Generates distribution of performance (not single number)
   - Enables PBO calculation
   - Used for FINAL validation of evolved strategies, not during evolution

4. **Window Rotation During Evolution (anti-overfitting in-loop)**
   - Each generation re-samples different evaluation windows
   - Forces strategies to generalize rather than memorize specific periods
   - Distinct from CPCV: this is an in-evolution mechanism, CPCV is post-evolution

5. **Multi-objective Fitness (no normalization)**
   - Primary: Sortino ratio (real values, not [0,1])
   - Secondary: Calmar ratio
   - Constraints: min_trades >= 30, max_dd < 30%, win_rate > 35%
   - Parsimony pressure: -0.01 * n_nodes

6. **MAP-Elites for Diversity**
   - Archive grid: frequency x complexity x regime
   - Maintains best strategy per niche
   - Prevents convergence to single strategy type

7. **Out-of-Time Sample (OTS) — Sacred Holdout**
   - Last 6 months of data (approx 2025-06-01 to 2025-11-21) are NEVER touched
     during evolution or CPCV validation
   - Only used ONCE at the very end for final reported results
   - Evolution + CPCV use data from 2023-01-01 to 2025-05-31
   - This is non-negotiable for paper credibility

8. **No DEAP framework**
   - Decision: build evolution engine from scratch
   - Rationale: DEAP adds complexity without proportional benefit for our use case.
     Our genome is a simple integer vector, operators are trivial (one-point crossover,
     ±1 mutation on codons). DEAP's abstractions (Toolbox, halloffame, selectors) add
     indirection that makes debugging harder. We need ~200 lines of engine code total.
     Rolling our own keeps it transparent and paper-documentable.

9. **No intra-window timing in grammar**
   - Decision: the grammar does NOT model "when within the window" a signal fires
   - Rationale: the strategy generates a boolean signal per bar. The backtest engine
     handles entry timing (first True bar after position is flat). Adding timing
     complexity to the grammar would expand the search space without clear benefit
     at this stage. Can be revisited in future work if initial results are promising.

10. **Fixed position sizing (not evolved)**
    - Decision: position sizing stays fixed (2% risk per trade, 10x leverage)
    - Rationale: evolving position sizing alongside entry logic conflates two
      concerns — a bad entry with aggressive sizing could look profitable by luck.
      Keep them separate: evolve entry/exit logic, then optimize sizing independently.
      Adaptive sizing is deferred to future work.

### Alternatives Evaluated and Rejected

For the paper's methodology section — document WHY we chose GE + MAP-Elites:

| Approach | Verdict | Reason |
|----------|---------|--------|
| **Co-evolution** (strategies competing) | Rejected | Markets are already the adversary. Adds complexity without clear benefit. |
| **Reinforcement Learning hybrid** | Deferred | Powerful but opaque. GE's interpretability is our selling point. Potential follow-up paper. |
| **NEAT (neuroevolution)** | Rejected | Loses interpretability entirely. Black-box strategies can't be explained in paper. |
| **CMA-ES (pure)** | Deferred as local optimizer | Excellent for continuous parameters but can't evolve structure. PLAN: use as local optimizer AFTER GE finds good structure. See Future Work. |
| **Linear GP** | Rejected | Faster and cleaner but less expressive than GE's grammar-guided generation. |
| **GP Trees (standard)** | Rejected | Bloat problem, closure property issues with mixed types (prices vs booleans). |
| **Differential Evolution** | Not applicable | Continuous-only, can't handle discrete structural search. |

### Comparative Analysis of EA Approaches (Paper Reference)

| Approach | Structure | Parameters | Diversity | Multi-obj | Our Use |
|----------|-----------|------------|-----------|-----------|---------|
| GP Trees | Yes | No | Low | Via NSGA-II | Rejected (bloat) |
| **Grammatical Evolution** | **Yes** | **Yes** | **Medium** | **Via NSGA-II** | **PRIMARY** |
| Linear GP | Yes | No | Low | Possible | Rejected (less expressive) |
| CMA-ES | No | Yes | High | Via MO-CMA | LOCAL OPTIMIZER (future) |
| Differential Evolution | No | Yes | Medium | Via MODE | Not applicable |
| **MAP-Elites** | N/A | N/A | **Very High** | Implicit | **DIVERSITY ENGINE** |
| Novelty Search | N/A | N/A | Very High | No | Considered, too exploratory |
| **NSGA-II/III** | N/A | N/A | Medium | **Native** | **SELECTION** |
| Island Model | N/A | N/A | High | Possible | Sprint 4 |

### Key References (for paper bibliography)

- Vectorial GP (VGP) — ArXiv 2025: vectorized fitness evaluation for GP
- Strongly-Typed GP (STGP-SATA) — ScienceDirect 2025: type safety in GP for trading
- Multi-Objective GP with Directional Changes — Springer 2025: DC-based price representation
- Agent-Based GA for Crypto — ArXiv 2025: population dynamics in crypto GA
- GE Ensembles for Trading — ScienceDirect: grammatical evolution applied to trading rules
- MAP-Elites for Trade Execution — 2026: quality-diversity in execution optimization
- Bailey & Lopez de Prado: DSR (Deflated Sharpe Ratio), PBO (Probability of Backtest
  Overfitting), CPCV (Combinatorial Purged Cross-Validation)
- Hansen (2005): Superior Predictive Ability test
- White (2000): Reality Check for data snooping

## Validation Pipeline (ordered)

The validation pipeline has THREE distinct layers. Conflating them is a design error.

### Layer 1: In-Evolution Anti-Overfitting
- **Window rotation**: each generation evaluates on freshly sampled windows
- **Parsimony pressure**: complexity penalty in fitness
- **Diversity maintenance**: MAP-Elites archive ensures niche coverage

### Layer 2: Post-Evolution Statistical Validation (on evolution data, excl. OTS)
- **CPCV**: Combinatorial Purged Cross-Validation (N=10 groups, purge + embargo)
- **PBO**: Probability of Backtest Overfitting (from CPCV distribution)
- **DSR**: Deflated Sharpe Ratio (corrects for multiple testing, skew, kurtosis)
- **Hansen SPA**: Superior Predictive Ability test (p < 0.05)
- **White RC**: Reality Check with block bootstrap (p < 0.05)
- **Signal Permutation Test**: Shuffle the strategy's SIGNALS 1000x, recalculate
  metrics. If shuffled signals produce similar performance, the signal has no
  predictive power. This is MORE rigorous than trade shuffling (which the current
  monte_carlo.py does) because it tests the signal itself, not just trade ordering.
- **Cross-Regime Validation**: Verify strategies work in bull, bear, AND sideways
  regimes separately. A strategy that only works in one regime is fragile.

### Layer 3: Final Out-of-Time Validation (sacred holdout)
- **OTS backtest**: Run surviving strategies on 2025-06-01 to 2025-11-21
- **One shot**: No iteration, no tuning. Report whatever comes out.
- Results from this layer are what goes in the paper's results section.

## Data Split
```
|---- Evolution + CPCV (2023-01-01 to 2025-05-31) ----|-- OTS holdout (2025-06 to 2025-11) --|
|  Window rotation during evolution                    |  NEVER touched until final report     |
|  CPCV folds for post-evolution validation            |  One-shot evaluation only             |
```

## What to Conserve from v1
- `loader.py` — robust Binance data loading via ccxt
- `backtest/runner.py` — core backtest engine (adapt interface)
- `backtest/exits.py` — ATR-based exit logic
- `backtest/metrics.py` — Sortino, Calmar, Sharpe, drawdown, CAGR (add DSR)
- `backtest/futures_position_sizing.py` — futures position sizing
- `backtest/final_backtest.py` — walk-forward final backtest
- `backtest/correlation.py` — portfolio decorrelation
- `robustness/` — Hansen SPA, White RC, bootstrap (with bug fixes)
- `reports/` — LaTeX, visualizations, pattern explainer
- `analysis/` — evolution analytics, Monte Carlo (extend with signal permutation)

## What to Eliminate from v1
- `ga_patterns/` entire directory — replaced by grammar/ + strategy/ + evolution/
- `evaluator.py` with eval() — replaced by vectorized_eval.py
- `chromosome.py` (legacy v1) — dead code
- `templates.py` — generates legacy type
- `building_blocks.py` — hardcoded parameters, replaced by grammar
- `module_semantics.py` — unnecessary with GE (grammar doesn't impose directional bias)
- `simple_sampling.py` — replaced by CPCV + window rotation

## Implementation Plan

### Sprint 1: Foundation (Grammar + Vectorized Eval)
**Goal**: BNF grammar + codon mapping + vectorized signal generation working

Files to create:
- `grammar/bnf.py` — BNF grammar definition (rules, terminals, parameter ranges)
- `grammar/mapper.py` — Integer codon vector → phenotype (strategy expression)
- `grammar/simplifier.py` — Canonicalize expressions (deduplicate, simplify)
- `strategy/phenotype.py` — Strategy dataclass (decoded expression, parameters, metadata)
- `strategy/vectorized_eval.py` — Compile phenotype → vectorized NumPy signal generation
- `tests/test_grammar.py` — Grammar + mapper unit tests
- `tests/test_vectorized_eval.py` — Vectorized evaluation tests

**Exit criterion**: Generate 1000 random genomes, decode to strategies, evaluate
vectorized on 1 month of data, all in <10 seconds.

### Sprint 2: Evolution Engine
**Goal**: Functional GA loop with new representation, replacing inline main.py loop

Files to create:
- `evolution/engine.py` — EvolutionEngine class (run, step, testable)
- `evolution/operators.py` — Crossover (one-point on codons), mutation (±1, ±N, resample)
- `evolution/selection.py` — Tournament selection, lexicase selection
- `strategy/parameters.py` — Parameter ranges and constraints for grammar terminals
- Adapt `backtest/runner.py` interface to accept Strategy phenotype
- Rewrite fitness evaluation (multi-objective, real values, parsimony)
- `tests/test_engine.py`

**Exit criterion**: 50 generations with population 200 in <30 minutes. Fitness
improves monotonically until at least generation 20.

### Sprint 3: Anti-Overfitting Suite
**Goal**: Rigorous statistical validation pipeline

Files to create:
- `validation/cpcv.py` — Combinatorial Purged Cross-Validation
- `validation/deflated_sharpe.py` — Deflated Sharpe Ratio
- `validation/pbo.py` — Probability of Backtest Overfitting
- `validation/signal_permutation.py` — Signal shuffle permutation test
- `backtest/sampling.py` — Window rotation for in-evolution use + CPCV integration
- Fix `robustness/bootstrap.py` — Replace bare except, investigate corrupt UPI/CAGR values
- Fix `robustness/white_rc.py` — Switch to block bootstrap
- `tests/test_cpcv.py`

**Exit criterion**: Full pipeline evaluates a strategy with CPCV (10 groups), calculates
DSR and PBO. Strategy is "real" if DSR > 0.95 AND PBO < 0.50.

### Sprint 4: Quality-Diversity + Regime Awareness
**Goal**: MAP-Elites archive + island model + market regime detection

Files to create:
- `evolution/archive.py` — MAP-Elites grid (frequency x complexity x regime)
- `evolution/island.py` — Island model with migration between subpopulations
- `data/regime_detector.py` — Bull/bear/sideways classification (volatility + trend based)
- Cross-regime validation in validation pipeline
- `tests/test_archive.py`

**Exit criterion**: Archive >50% cells occupied after 100 generations. At least 3
niches with strategies passing DSR > 0.95.

### Sprint 5: Integration + Paper Experiments
**Goal**: End-to-end pipeline, reproducible results, paper-ready output

Files to create/modify:
- Thin CLI in `main.py` (argparse: evolve, validate, report modes)
- OTS holdout enforcement (hard-coded date split, assertion that OTS data is excluded)
- Adapt `reports/` for new strategy representation
- Reproducible experiment runner (seeds, logging, artifact saving)
- Generate LaTeX tables and figures for paper

**Exit criterion**: `python main.py evolve --seed 42` produces identical results across
runs. At least 1 strategy passes ALL statistical tests, OR we have documented evidence
that no alpha exists (also a valid paper result).

## Tech Stack
- Python 3.9+
- pandas, numpy, scipy, statsmodels (core computation)
- ccxt (data fetching)
- matplotlib, seaborn (visualization)
- numba (JIT for hot loops in backtest)
- pyyaml (configuration)
- pytest (testing)
- NO DEAP (see decision #8 above)

## How to Run
```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run v1 (legacy — for reference only)
python3 main.py

# Run v2 (after implementation)
python3 main.py evolve --config config.yaml --seed 42
python3 main.py validate --results results.json
python3 main.py report --results results.json
```

## Success Metrics
| Metric | Minimum | Target |
|--------|---------|--------|
| Deflated Sharpe Ratio | > 0.95 | > 0.99 |
| Prob. Backtest Overfitting | < 0.50 | < 0.30 |
| Hansen SPA p-value | < 0.05 | < 0.01 |
| Signal Permutation p-value | < 0.05 | < 0.01 |
| OOS Sortino (on OTS holdout) | > 0.5 | > 1.0 |
| OOS Max Drawdown | < 30% | < 15% |
| Min trades (total) | 30 | 100+ |
| Cross-regime consistency | Pass 2/3 regimes | Pass 3/3 |
| Reproducibility | 100% | 100% |

## Code Conventions
- No over-engineering: minimum complexity for current task
- Functions over classes unless state management is needed
- Type hints on public interfaces
- Logging over print statements
- Tests for any non-trivial logic
- Config-driven behavior (no hardcoded magic numbers)
- Spanish comments are OK (bilingual project)

## Lessons Learned

### From v1
1. Don't normalize fitness to [0,1] — it destroys the gradient
2. Don't hardcode indicator parameters — they must evolve
3. Don't use eval() on strings — vectorize everything
4. Don't use fixed evaluation windows across generations — rotate or use CPCV
5. Don't over-constrain the search space with semantic rules — let evolution explore
6. Population of 50 is too small for a 5.5M search space
7. 30 generations with patience=5 is too few — need 100+ with patience=20+
8. Elitism of 16% kills diversity — use 4-6% or MAP-Elites
9. If statistical tests fail (p > 0.05), the patterns are NOT real — no exceptions
10. Verbose logging (15 lines per evaluation) buries useful information in noise
11. Immigration with fitness=-999 is a band-aid — structural diversity mechanisms needed
12. Semantic pre-filtering prevents legitimate contrarian strategies from being explored
13. Bootstrap results can be silently corrupt — always sanity-check metric values
14. `warnings.filterwarnings('ignore')` hides real problems — never suppress globally

### From v2 (Sprints 6-9)
15. **CMA-ES overfitting is real**: BNB L2* went from +64.6% (6mo OTS) to +0.3% (9mo OTS).
    CMA-ES tuning overfits to the specific regime. Always validate on extended holdout.
16. **Original GE params are more robust than CMA-ES tuned**: Strategies without CMA-ES
    degrade gracefully; CMA-ES strategies can collapse entirely in a new regime.
17. **SHORT strategies are more robust in volatile crypto**: They profit from mean-reversion
    and panic selling. LONG strategies need bull regimes to work.
18. **Evolve once, deploy with regime filter**: Walk-forward re-evolution performs WORSE than
    fixed strategies + SMA regime gate. Re-evolution introduces new overfitting each time.
19. **HMM captures volatility, not direction**: Use SMA for direction, HMM for volatility.
    Don't conflate the two.
20. **Monte Carlo P(profitable)=100% doesn't mean strong alpha**: All trade orderings are
    profitable, but the specific ordering matters (MC percentile varies widely).
21. **9-month OTS is more revealing than 6-month**: Additional 3 months exposed CMA-ES
    overfitting that was invisible in the original 6-month window.

## Current State (as of 2026-03-14)

### Sprints Complete
- **Sprints 1-5**: v2 architecture (GE + MAP-Elites + CPCV). All DONE.
- **Sprint 6B**: Multi-timeframe (15m/1h/4h) in grammar v5b. DONE.
- **Sprint 6A**: Alt data (funding, OI, L/S ratio). DEFERRED — data quality.
- **Sprint 7**: Multi-asset (BTC/ETH/BNB). DONE. Live on Hetzner.
- **Sprint 8**: CMA-ES + Walk-forward validation. DONE.
- **Sprint 9**: HMM volatility detector + adaptive sizing. DONE.
- **Sprint 10**: RL hybrid. NOT STARTED — conditional on alpha evidence.

### Extended OTS Results (Jun 2025 – Feb 2026, 9 months)
- 10/10 strategies positive (100% survival rate)
- Average return: +8.7%, Average Sortino: 0.18
- **CMA-ES overfitting exposed**: BNB L2*/L3* collapsed from 50-64% to <1%
- Best: BNB L1* (+21.6%, Calmar 2.80), ETH S2* (+17.1%, CAGR +23.7%)
- Weakest: BNB L2* (+0.3%), BNB L3* (+0.5%) — candidates for removal

### Key Scripts
- `generate_strategy_analysis.py`: 4-panel analysis per strategy (equity+MC, dist, PnL, metrics)
- `generate_paper_figures.py`: Portfolio-level figures for the paper
- `compute_multi_asset_portfolio.py`: Portfolio metrics computation

## Future Work (deferred — material for paper's Future Work section)
- **CMA-ES with CPCV validation**: Instead of optimizing on full training set, apply CPCV
  to CMA-ES parameters. This prevents regime-specific overfitting.
- **Reinforcement Learning hybrid**: Use GE-discovered strategies as initial policy for
  RL fine-tuning. Combines interpretability of GE with adaptability of RL.
- **Alpha decay analysis**: Monthly sub-period performance breakdown to detect when
  strategies start degrading. Informs retraining frequency.
- **Portfolio optimization**: Replace equal-weight with mean-variance or risk parity.
  Remove weak strategies, increase weight on robust ones.
- **Regime-adaptive portfolio rotation**: Dynamically adjust weights based on detected
  regime (more SHORT in bear, more LONG in bull).
- **Walk-forward with ensemble rebalancing**: Periodic re-evaluation of which strategies
  get capital based on rolling 3-month performance.
- **Signal confirmation across timeframes**: Require multi-TF agreement before entry.
- **Adaptive position sizing**: Kelly fraction, volatility scaling, regime-dependent sizing.
- **Live performance tracking**: Structured comparison of live P&L vs OTS expectations.
