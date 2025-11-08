# BTC/USDT Pattern Discovery - Genetic Algorithm

Sistema de descubrimiento automático de patrones de trading en futuros de BTC/USDT usando Algoritmos Genéticos con validación estadística rigurosa.

## 🎯 Características

- **Asset**: BTC/USDT Perpetual Futures (Binance)
- **Timeframe**: 15 minutos (configurable vía TIME_MAP)
- **Metodología**: Algoritmos genéticos + Walk-forward + Hansen SPA + White RC
- **Output**: Portafolio de 3-8 patrones decorrelacionados (ρ < 0.35)
- **Validación**: Zero lookahead bias + reproducibilidad total
- **Tracking**: Evolución de patrones por generación

## 📦 Instalación

```bash
git clone <repo>
cd btc-ga-patterns
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Editar configuración (opcional)
nano config.yaml

# 2. Ejecutar experimento completo
python main.py
```

El proceso tomará ~2-4 horas dependiendo de:
- Población GA (default: 100)
- Generaciones (default: 150)
- Ventanas walk-forward (default: ~30)

## ⚙️ Configuración

### Cambiar Timeframe

Edita `config.yaml`:

```yaml
data:
  timeframe: "15m"  # Opciones: "1m", "5m", "15m", "1h", "4h", "1d"
```

El TIME_MAP automáticamente ajusta `bars_per_year` para métricas anualizadas.

### Ajustar Costos

```yaml
costs:
  fees_bps_long: 4.0      # Binance Futures: 0.04% taker
  fees_bps_short: 4.0
  slippage_bps_long: 2.0  # BTC líquido: ~0.02%
  slippage_bps_short: 2.0
```

### Evolution Tracking

```yaml
ga:
  evolution_tracking:
    enabled: true
    sample_size_per_generation: 5   # Patrones por generación
    save_every_n_generations: 10    # Snapshot cada N gens
```

Outputs en: `./evolution_snapshots/`

## 📊 Estructura del Proyecto

```
btc-ga-patterns/
├── config.yaml              # Configuración única editable
├── main.py                  # Orquestador principal
├── loader.py                # Binance API con paginación
├── backtest/                # Motor de backtesting
│   ├── metrics.py          # UPI, Sharpe, CAGR, etc.
│   ├── runner.py           # Backtest engine
│   ├── walkforward.py      # Walk-forward analysis
│   └── exits.py            # ATR stops/targets
├── ga_patterns/             # Algoritmo genético
│   ├── grammar.py          # Predicados OHLCV
│   ├── chromosome.py       # Expression trees
│   ├── generator.py        # GA operators
│   ├── fitness.py          # Multi-objective evaluation
│   └── evolution_tracker.py # Tracking de evolución
├── robustness/              # Tests estadísticos
│   ├── hansen_spa.py       # Hansen SPA Test
│   ├── white_rc.py         # White's Reality Check
│   └── bootstrap.py        # Monte Carlo bootstrap
└── tests/                   # Tests unitarios
```

## 🧪 Testing

```bash
# Tests completos
pytest tests/ -v

# Coverage
pytest tests/ --cov=backtest --cov=ga_patterns --cov-report=html

# Solo test de anti-lookahead
pytest tests/test_leakage.py -v
```

## 📈 Outputs

Después de ejecutar, encontrarás:

- **Logs**: `./logs/experiment_YYYYMMDD_HHMMSS.log`
- **Patrones finales**: `./output_reports/final_portfolio.yaml`
- **Equity curves**: `./output_reports/equity_*.csv`
- **Matriz de correlación**: `./output_reports/correlation_matrix.png`
- **Evolution snapshots**: `./evolution_snapshots/generation_*.json`
- **Reporte HTML**: `./output_reports/experiment_report.html`

## 🔬 Reproducibilidad

Todos los componentes aleatorios usan seeds configurables:

```yaml
ga:
  seed: 42

robustness:
  seed: 42
```

Ejecutar 2 veces con el mismo config → resultados idénticos.

## 📖 Documentación Adicional

- **Metodología GA**: Ver `docs/genetic_algorithm.md`
- **Walk-Forward**: Ver `docs/walk_forward_validation.md`
- **Hansen SPA**: Ver `docs/statistical_tests.md`

## 🛠️ Desarrollo

**Status**:
- ✅ Sprint 0: Project Setup
- ⏳ Sprint 1: Core Infrastructure (Binance API + Metrics)
- ⏳ Sprint 2: Genetic Algorithm + Evolution Tracking
- ⏳ Sprint 3: Backtesting + Walk-Forward
- ⏳ Sprint 4: Statistical Validation
- ⏳ Sprint 5: Reports & Testing

## 📄 License

MIT

## 👤 Author

[Tu Nombre]
