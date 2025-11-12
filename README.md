# 🧬 Genetic Algorithm Pattern Discovery for Crypto Trading

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Evolutionary algorithm for discovering profitable trading patterns in cryptocurrency time series data with rigorous statistical validation.

## 📋 Overview

This project implements a sophisticated genetic algorithm system that:

- **Discovers** interpretable trading patterns in OHLCV data
- **Validates** patterns using walk-forward analysis (anti-lookahead guaranteed)
- **Evaluates** bidirectionally (LONG and SHORT) with adaptive exits
- **Tests** statistical significance (Hansen SPA, White's Reality Check, Bootstrap)
- **Generates** publication-ready reports and visualizations

Developed as an undergraduate thesis project for Advanced Business Analytics at UCEMA.

## 🎯 Key Features

### Genetic Algorithm
- **Multi-objective fitness**: UPI, Sharpe Ratio, CAGR
- **Bidirectional evaluation**: Automatically discovers LONG/SHORT patterns
- **Adaptive constraints**: Pattern complexity evolves with generations
- **Progressive grammar**: Direct comparisons → Ratios → Technical indicators

### Backtesting Engine
- **Walk-forward validation**: Rolling windows with strict anti-lookahead
- **Adaptive exits**: Stop-loss and take-profit based on market conditions
- **Realistic costs**: Binance Futures fees + slippage modeling
- **Portfolio decorrelation**: Selects diverse patterns

### Statistical Validation
- **Hansen's SPA Test**: Superior predictive ability vs benchmark
- **White's Reality Check**: Correction for data snooping
- **Block Bootstrap**: Confidence intervals preserving autocorrelation

### Pattern Grammar
```python
# Examples of discoverable patterns:
- close[0] > close[1] AND volume[0] > volume[1]
- price_change_pct[0] > 0.02 AND body_ratio[0] > 0.7
- RSI[0] < 30 OR price_vs_ma_pct[0] < -0.05
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/cripto_ga.git
cd cripto_ga

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml`:
```yaml
data:
  symbol: "BTCUSDT"
  timeframe: "15m"
  start: "2020-01-01"
  end: "2025-01-01"

ga:
  population: 100
  generations_max: 150
  mutation_rate: 0.2
  crossover_rate: 0.8
```

### Run Experiment
```bash
python main.py
```

**Expected runtime**: 2-4 hours (depending on hardware)

## 📊 Output Structure
```
output_reports/
├── experiment_report.md          # Complete experiment documentation
├── equity_performance.png        # Portfolio vs benchmark
├── evolution_fitness.png         # GA convergence
├── statistical_tests.png         # P-values visualization
├── drawdown_analysis.png         # Drawdown analysis
├── returns_distribution.png      # Returns histogram + Q-Q plot
├── patterns_table.tex           # LaTeX table for paper
├── metrics_table.tex            # Performance metrics
├── statistical_tests_table.tex  # Statistical results
├── hansen_spa_results.json      # Hansen SPA data
├── white_rc_results.json        # White RC data
├── bootstrap_results.json       # Bootstrap CI data
└── equity_curves.csv            # Raw equity data
```

## 📖 Documentation

- [User Guide](docs/USER_GUIDE.md) - Detailed usage and configuration
- [Deployment Guide](DEPLOYMENT.md) - Reproducibility checklist
- API Documentation - Module and function reference (inline docstrings)

## 🧪 Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run integration tests
pytest tests/test_integration.py -v

# Run benchmarks
pytest tests/test_performance.py -m benchmark
```

## 📈 Example Results

### Pattern Example
```
Pattern (LONG, window=5, fitness=0.4512)
  AND(
    close[0] > close[1],
    volume[0] > volume[2]
  )
```

### Performance Metrics
| Metric | Value |
|--------|-------|
| UPI | 0.3421 |
| Sharpe Ratio | 1.87 |
| CAGR | 21.34% |
| Max Drawdown | -12.45% |
| Win Rate | 58.3% |

### Statistical Validation
- **Hansen SPA**: p-value = 0.018 (✓ Reject H0)
- **White's RC**: p-value = 0.034 (✓ Robust)
- **Bootstrap UPI CI**: [0.24, 0.45]

## 🏗️ Architecture
```
cripto_ga/
├── loader.py               # Data loading (Binance API)
├── backtest/               # Backtesting engine
│   ├── metrics.py          # Performance metrics (UPI, Sharpe, CAGR)
│   ├── exits.py            # Exit strategies
│   ├── runner.py           # Backtest execution
│   ├── walkforward.py      # Walk-forward validation
│   └── correlation.py      # Portfolio selection
├── ga_patterns/            # Genetic algorithm
│   ├── grammar.py          # Pattern grammar (predicates)
│   ├── chromosome.py       # Expression trees
│   ├── generator.py        # Population initialization + operators
│   ├── fitness.py          # Fitness evaluation (bidirectional)
│   └── evolution_tracker.py # Evolution monitoring
├── robustness/            # Statistical validation
│   ├── bootstrap.py        # Block bootstrap
│   ├── hansen_spa.py       # Hansen SPA test
│   └── white_rc.py         # White's Reality Check
├── reports/               # Reporting and visualization
│   ├── pattern_explainer.py  # Natural language explanations
│   ├── visualizations.py     # 5 publication-quality plots
│   ├── report_generator.py   # Markdown report
│   └── latex_exporter.py     # LaTeX tables
├── tests/                 # Test suite (72 tests)
├── config.yaml           # Configuration
└── main.py              # Main execution script
```

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file.

## 🙏 Acknowledgments

- **UCEMA** - Advanced Business Analytics Program
- **Binance** - Historical market data via API
- **Hansen (2005)** - Superior Predictive Ability test methodology
- **White (2000)** - Reality Check for data snooping

## 📧 Contact

**Juan Manuel Targa**
Undergraduate Student - UCEMA
📧 [jmtarga26@ucema.edu.ar]
🔗 [@juanmanueltarga]

---

**Note**: This project is for educational and research purposes. Not financial advice. Always backtest thoroughly before live trading.
