# User Guide - Genetic Algorithm Pattern Discovery

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Understanding Outputs](#understanding-outputs)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Introduction

This system uses genetic algorithms to discover trading patterns in cryptocurrency data. This guide will walk you through setting up, configuring, running experiments, and interpreting results.

## Installation

### Prerequisites

- **Python 3.9+** (recommended: 3.10 or 3.11)
- **pip** (package manager)
- **Git** (for cloning repository)
- **4GB+ RAM** (8GB recommended for large datasets)

### Step-by-Step Setup
```bash
# 1. Clone repository
git clone https://github.com/yourusername/cripto_ga.git
cd cripto_ga

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
pytest tests/test_integration.py::test_pipeline_grammar_to_pattern -v
```

## Configuration

### Main Configuration File: `config.yaml`

#### Data Configuration
```yaml
data:
  symbol: "BTCUSDT"              # Trading pair
  timeframe: "15m"               # Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d)
  start: "2020-01-01"            # Start date (YYYY-MM-DD)
  end: "2025-01-01"              # End date
  exchange: "binance"            # Exchange name
```

#### Genetic Algorithm Parameters
```yaml
ga:
  population: 100                # Population size (50-200 recommended)
  generations_max: 150           # Maximum generations
  patience_no_improve: 20        # Early stopping patience
  elitism: 10                    # Elite patterns to preserve
  mutation_rate: 0.2             # Probability of mutation (0.1-0.3)
  crossover_rate: 0.8            # Probability of crossover (0.7-0.9)
  seed: 42                       # Random seed for reproducibility
  window_min: 2                  # Minimum lookback bars
  window_max: 8                  # Maximum lookback bars
```

**Parameter Guidelines**:
- **Population**: Small (50-75) = faster; Large (150-200) = thorough exploration
- **Mutation Rate**: Low (0.1) = conservative; High (0.3) = aggressive
- **Window Size**: Smaller = more signals, less context; Larger = fewer signals, more context

#### Exit Strategy
```yaml
exits:
  stop_loss: 0.02     # Stop loss percentage (2%)
  take_profit: 0.03   # Take profit percentage (3%)
  max_hold_bars: 100  # Maximum holding period
```

**Exit Presets**:
- Conservative: `stop_loss: 0.025, take_profit: 0.025`
- Balanced: `stop_loss: 0.02, take_profit: 0.03` (default)
- Aggressive: `stop_loss: 0.015, take_profit: 0.04`

## Running Experiments

### Basic Execution
```bash
python main.py
```

### Experiment Phases

**Phase 1: Data Loading** (2-5 min)
```
Downloading data from Binance...
✓ Downloaded 35,040 bars
```

**Phase 2: GA Evolution** (30-90 min)
```
Generation 10/150
Best: 0.3821 (LONG)
Mean: 0.2103
```
*Watch for*: Best fitness increasing, direction balance (LONG/SHORT)

**Phase 3: Portfolio Selection** (30-60 min)
```
Re-evaluating top patterns...
Pattern 1: Fitness = 0.4102 (LONG)
```

**Phase 4: Statistical Validation** (10-20 min)
```
Hansen SPA Test: p-value = 0.0180
White RC: p-value = 0.0340
```

**Phase 5: Reports** (2-5 min)
```
✓ Saved equity curve plot
✓ Report saved to output_reports/experiment_report.md
```

## Understanding Outputs

### Main Report: `experiment_report.md`

Contains:
- **Executive Summary**: Key metrics at a glance
- **Methodology**: Complete experimental setup
- **Evolution Analysis**: GA convergence
- **Portfolio Patterns**: Discovered patterns with explanations
- **Statistical Validation**: Test results interpretation

### Key Metrics

**UPI (Ulcer Performance Index)**
- Formula: CAGR / Ulcer Index
- Interpretation:
  - \>0.5: Excellent
  - 0.2-0.5: Good
  - <0.2: Poor

**Sharpe Ratio**
- Formula: (Return - RiskFreeRate) / Volatility
- Interpretation:
  - \>2.0: Excellent
  - 1.0-2.0: Good
  - <1.0: Poor

**Max Drawdown**
- Formula: Max peak-to-trough decline
- Interpretation:
  - <15%: Excellent
  - 15-30%: Good
  - \>30%: Poor

### Statistical Tests

**Hansen SPA (p-value)**
- p < 0.05: Portfolio outperforms benchmark (significant)
- p ≥ 0.05: Cannot conclude outperformance

**White's Reality Check (p-value)**
- p < 0.05: Results robust after multiple testing correction
- p ≥ 0.05: Possible data snooping

### Visualizations

**equity_performance.png**
- Blue line: Portfolio
- Purple line: Buy & Hold benchmark
- Look for: Portfolio above benchmark

**evolution_fitness.png**
- Shows GA convergence
- Look for: Upward trend, plateauing

**statistical_tests.png**
- Left: P-values (bars should be left of red line α = 0.05)
- Right: Confidence intervals

## Troubleshooting

### Common Issues

**1. "No valid patterns after filters"**

*Cause*: Filters too strict or insufficient data

*Solutions*:
```yaml
# Relax CAGR threshold
ga:
  fitness_weights:
    cagr_min_threshold: 0.0  # Was 0.05

# Reduce min trades requirement
selection:
  min_trades: 1  # Was 5
```

**2. "Experiment too slow"**

*Cause*: Large dataset or population

*Solutions*:
```yaml
# Reduce population
ga:
  population: 50  # Was 100

# Use larger timeframe
data:
  timeframe: "1h"  # Was "15m"
```

**3. "Hansen SPA p-value too high"**

*Cause*: Patterns not outperforming benchmark

*Solutions*:
- Run more generations (150 → 200)
- Adjust fitness weights to emphasize returns
- Try different time period

**4. "Memory error"**

*Cause*: Dataset too large

*Solutions*:
```yaml
# Reduce date range
data:
  start: "2022-01-01"  # Was "2020-01-01"

# Use larger timeframe
data:
  timeframe: "1h"  # Was "15m"
```

### Debugging

Enable debug logging:
```python
# In main.py, change:
logging.basicConfig(
    level=logging.DEBUG,  # Was INFO
    ...
)
```

## Advanced Usage

### Custom Predicates

Add custom indicators in `ga_patterns/grammar.py`:
```python
def my_custom_indicator(data: pd.DataFrame, bar_offset: int = 0) -> float:
    """Your custom logic here."""
    idx = -(bar_offset + 1)
    # Calculate your indicator
    return value

register_predicate(
    'my_custom',
    my_custom_indicator,
    threshold_range=(0.0, 1.0),
    description="My custom indicator",
    category='derived'
)
```

### Testing on Different Assets

```yaml
data:
  symbol: "ETHUSDT"  # Try Ethereum
  # or
  symbol: "SOLUSDT"  # Try Solana
```

**Note**: Results may vary significantly across assets.

## Tips for Academic Writing

### In Your Paper

1. **Methodology Section**: Copy from `experiment_report.md` → Methodology
2. **Results Section**: Use metrics from `metrics_table.tex`
3. **Figures**: Import all PNG visualizations
4. **Statistical Validation**: Use `statistical_tests_table.tex`

### Best Practices

✅ **Do**: "The algorithm identified a pattern with UPI of 0.45 (95% CI: [0.24, 0.51])"
❌ **Don't**: "The algorithm found the best pattern"

✅ **Do**: "The strategy demonstrated superior performance vs benchmark (Hansen SPA p=0.018)"
❌ **Don't**: "This strategy is profitable"

✅ **Do**: Report all experiments, discuss limitations
❌ **Don't**: Present only successful experiments

### Reproducibility

Always include in paper:
- Random seed used
- Exact config.yaml
- Data date range
- Software versions (from `requirements.txt`)

---

**Last updated**: January 2025
