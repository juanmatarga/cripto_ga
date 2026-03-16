# Session 2026-03-15: Engine Overhaul & Portfolio Discovery

## Problem Statement
The NSGA-II engine (v3) was producing only 1-node strategies. The old island model
system found complex multi-condition strategies but was replaced by flat NSGA-II
which eliminated complexity through selection pressure.

## Root Causes Diagnosed

### 1. 1-Node Convergence (Fixed in v6)
**Root cause**: NSGA-II selection doesn't protect complexity niches. Simple strategies
naturally have higher per-trade sortino (they fire at extreme conditions) and outcompete
complex strategies. By gen 5, 88% of population was 1-node (from 25% at init).

**Fix**: Complexity-niched population. 3 islands by n_nodes (25%/37.5%/37.5%).
NSGA-II selection happens WITHIN each niche. Structure codon (genome[2]) protected
during mutation/crossover.

### 2. Missing Diversity Mechanisms (Fixed in v7)
**Root cause**: Old island model had lexicase selection, migration, archive injection.
NSGA-II only had binary tournament.

**Fix**: Hybrid selection (60% tournament + 30% lexicase + 10% random), cross-niche
migration every 10 gens, archive injection every 5 gens, stagnation-triggered
immigration, diversity-reactive mutation boost.

### 3. Low Trade Frequency (Fixed in v8)
**Root cause**: 60% of grammar conditions are CROSSES_ABOVE/BELOW (one-bar events).
For AND logic, two crossovers coinciding on the same bar is ~0.01% probability.

**Fix**: Signal persistence — CROSSES events stay "warm" for 4 bars (1 hour).
The condition must still hold (e.g., left > right) but the crossover event is
remembered for a window, allowing AND combinations to fire.

### 4. Grammar Imbalance (Fixed in v9)
**Root cause**: Grammar had 60% crossover, 40% persistent conditions. Even with
signal persistence, crossovers fire <0.5% of bars vs persistent at 5-30%.

**Fix**: Rebalanced grammar to 57% persistent / 43% crossover. Added weighted
duplicates of persistent condition types.

### 5. Sortino Cap Masking Fitness Gradient (Fixed in v7)
**Root cause**: Sortino capped at 10.0 in composite fitness. Most strategies hit the
cap, destroying differentiation. Engine couldn't distinguish good from great.

**Fix**: Cap lowered to 5.0 in composite, 50.0 in raw. Forces differentiation via
CAGR, calmar, PF, WL ratio.

## Engine Versions

| Version | Key Change | OTS Pass | Cross-regime | Top CAGR |
|---------|-----------|----------|-------------|----------|
| v3 | Flat NSGA-II | 9/25 | 0 | +21.5% |
| v6 | Complexity niches | 70/155 | ~30 | +24.6% |
| v7 | +Lexicase+migration | 18/72 | ~10 | +13.1% |
| v8 | +Signal persistence | 32/74 | ~15 | +89.5%* |
| v9 | +Grammar rebalance | 27/72 | 21 (78%) | +20.2% |

*v8 BNB +89.5% was directional beta (collapsed in bear market)

## Statistical Validation

### Signal Permutation (500 shuffles)
- ETH 2n SHORT MFI+ATR: p=0.000 (OTS), p=0.003 (Bear), p=0.000 (Bull)
- ETH 2n SHORT ATR+RSI: p=0.000 (OTS), p=0.003 (Bear), p=0.000 (Bull)
- ETH 1n LONG BBWIDTH: p=0.000 (OTS), p=0.007 (Bear), p=0.000 (Side)
- BNB 3n LONG RSI+STOCH+VOL: p=0.000

### Hansen SPA
0/10 reject H0 vs B&H (p_mean=0.611). Expected: strategies make few trades,
most bars have no position → per-bar outperformance diluted. Signal permutation
is the more appropriate test for low-frequency strategies.

### Cross-Regime Validation
Tested on 4 periods: OTS, Bear (Jan-Jun 2022), Sideways (Jul-Dec 2023),
Bull (Oct 2024-Mar 2025). 78% of v9 strategies are positive in 3+ periods.

## Final Portfolio (18 strategies)

- 6 BTC (4 SHORT, 2 LONG)
- 6 ETH (4 SHORT, 2 LONG)
- 6 BNB (2 SHORT, 4 LONG)
- 16/18 cross-regime validated
- Average pairwise correlation: +0.007
- Equal-weight Sharpe: 2.92, Max DD: -2.5%

Saved in `results/final_portfolio.json`

## Files Modified

### Core Engine
- `evolution/engine.py` — Complete rewrite: complexity niches, hybrid selection,
  migration, archive injection, stagnation immigration, diversity-reactive mutation
- `evolution/fitness.py` — Composite fitness with sortino cap 5.0, removed trade
  bonus, relaxed constraints for AND strategies
- `evolution/selection.py` — Extended lexicase with 4 criteria
- `evolution/operators.py` — Active-codon targeting (unchanged)
- `evolution/nsga2.py` — NSGA-II core (unchanged)

### Signal Generation
- `strategy/vectorized_eval.py` — Signal persistence (CROSS_PERSISTENCE_BARS=4)

### Grammar
- `grammar/bnf.py` — Rebalanced: 57% persistent / 43% crossover conditions

### Data
- `data/cache/` — Parquet cache for BTC/ETH/BNB 2022-2025
- `run_nsga2_evolution.py` — Cache-aware data loading

### Results
- `results/v6_ots_passing_strategies.json` (70 strategies)
- `results/v7_ots_passing_strategies.json` (18 strategies)
- `results/v8_ots_passing_strategies.json` (32 strategies)
- `results/v9_ots_passing_strategies.json` (27 strategies)
- `results/validated_portfolio_candidates.json` (3 ETH deeply validated)
- `results/final_portfolio.json` (18 production portfolio)
