# Plan: Encontrar Mejores Patrones

## Diagnóstico: Por qué los patrones actuales son limitados

1. **Pocos trades** (~2-3/mes): Las condiciones crossover son eventos de 1 bar.
   Con persistence=4 mejora, pero sigue siendo bajo.
2. **Solo ETH y BNB**: BTC es demasiado eficiente para la gramática actual.
3. **Dominan las mismas condiciones**: RSI extremos, BBWIDTH, PRICE_POS.
   La gramática no ofrece suficiente diversidad de conceptos.
4. **Exit parameters fijos por estrategia**: TP/SL no se adaptan a volatilidad.
5. **Sin datos alternativos**: Funding, OI, liquidations — señales de sentiment
   que otros traders no están usando con GE.

## Sprint 1: Gramática Expandida (2-3 días)

### 1A: Nuevos tipos de condiciones
Agregar condiciones que capturen conceptos que la gramática actual NO puede:

```
# Momentum divergence: precio hace nuevo high pero RSI no
"DIVERGENCE_BULL(<osc>, <period>)"    # bullish divergence
"DIVERGENCE_BEAR(<osc>, <period>)"    # bearish divergence

# Breakout de rango: precio rompe canal de N periodos
"BREAKOUT_UP(<period>)"    # close > max(high, N)
"BREAKOUT_DOWN(<period>)"  # close < min(low, N)

# Volatility squeeze → expansion
"SQUEEZE(<bb_period>, <bb_std>)"  # BBWIDTH en mínimo de N bars

# Multi-bar patterns (no solo 1 bar)
"TRENDING_UP(<indicator>, <period>)"   # indicator subiendo N bars
"TRENDING_DOWN(<indicator>, <period>)" # indicator bajando N bars
```

**Por qué importa**: Estos conceptos capturan dinámicas que 2-3 indicadores
con `>` o `CROSSES` no pueden. Un squeeze seguido de breakout es un setup
clásico que la gramática actual no puede representar.

### 1B: Evolved signal persistence
En vez de `CROSS_PERSISTENCE_BARS = 4` fijo, agregar `<lookback>` como
parámetro que evoluciona: `RSI(14) CROSSES_BELOW 30 WITHIN 8`.

### 1C: Datos alternativos (si están disponibles)
- Funding rate z-score (ya implementado, no evolucionado)
- Open interest rate of change
- Liquidation imbalance (buy vs sell liquidations)
- Taker buy/sell ratio

Estos están en `strategy/vectorized_eval.py` pero la gramática v5b los
removió porque "funding rate proven noise". Reconsiderar con la nueva
gramática y persistence.

## Sprint 2: BTC-Specific Alpha (2 días)

### 2A: Timeframe 1d para BTC
BTC se mueve en ciclos de semanas. Los indicadores de 15m/1h/4h son ruido.
Agregar `1d` como timeframe en la gramática (solo para BTC runs).

**Implementación**: Agregar '1d' a `<timeframe>` productions. En
`multi_timeframe.py`, agregar resample a 1d. Necesita más lookback bars
(1000+ para 1d indicators con period 30).

### 2B: Population 400, Generations 200
BTC es el mercado más eficiente. Necesita más fuerza bruta.
Correr con el doble de population y generaciones.

### 2C: Bitcoin Dominance como indicador
Si podemos obtener datos de BTC dominance (BTC.D), es un indicador macro
poderoso. Cuando dominance sube, BTC se fortalece vs altcoins.
Evaluar si ccxt puede obtener este dato.

## Sprint 3: Anti-Overfitting Avanzado (2 días)

### 3A: Walk-Forward Evolution
En vez de evolucionar en un bloque y validar OTS, hacer walk-forward:
- Evolucionar en 2022-2023, validar en 2024H1
- Evolucionar en 2023-2024H1, validar en 2024H2
- Evolucionar en 2024, validar en 2025H1
Si las mismas CONDICIONES aparecen en múltiples folds, es alpha real.

### 3B: Signal Permutation integrado en fitness
En vez de correr signal permutation post-hoc, integrarlo como constraint:
estrategias cuya señal no es significativamente mejor que random (p>0.10)
reciben penalidad en el fitness. Esto filtra noise DURANTE la evolución.

### 3C: CPCV con PBO
Correr CPCV (10 folds) en las top estrategias. Calcular PBO.
Target: PBO < 0.50 para incluir en portfolio.

## Sprint 4: Diversidad de Patrones (1-2 días)

### 4A: Anti-correlation objective
Agregar un tercer objetivo a NSGA-II: correlación negativa con las mejores
estrategias existentes. Fuerza al motor a buscar patrones decorrelacionados.

### 4B: Condition-type diversity enforcement
Después de evolución, clusterizar por tipo de indicador (momentum, volatility,
volume, trend). Forzar representación de cada cluster en el portfolio.

### 4C: Ensemble meta-strategies
Combinar 2-3 estrategias simples: "Strategy A AND Strategy B signal". Esto
crea meta-patrones con mayor confianza por señal y más trades.

## Sprint 5: Optimización de Portfolio (1 día)

### 5A: Mean-variance optimization
En vez de equal weight, usar Markowitz para optimizar weights basado en
returns y correlaciones históricas.

### 5B: Risk parity
Asignar peso inversamente proporcional a volatilidad de cada estrategia.
Las más volátiles reciben menos capital.

### 5C: Regime-gated portfolio
Activar/desactivar estrategias basado en régimen de mercado. SHORT strategies
se activan en bear/sideways, LONG en bull/sideways.

## Orden de Ejecución Recomendado

1. **Sprint 1A** (gramática expandida) — mayor impacto potencial
2. **Sprint 3A** (walk-forward) — mejora confianza en los patrones
3. **Sprint 2A** (BTC timeframe 1d) — llena el gap de BTC
4. **Sprint 4A** (anti-correlation) — mejora portfolio
5. **Sprint 1C** (datos alternativos) — nueva fuente de alpha
6. **Sprint 5** (optimización) — refina lo que tenemos

## Métricas de Éxito

| Métrica | Actual | Target |
|---------|--------|--------|
| Strategies en portfolio | 4 | 8-12 |
| Avg trades/mes/strategy | 2-3 | 5-10 |
| Cross-regime validated | 4/4 | 8+/12 |
| Portfolio Sharpe (1x) | ~3.0 | 4.0+ |
| Assets cubiertos | ETH+BNB | ETH+BNB+BTC |
| Signal perm p<0.05 | 2 | 6+ |
| Max OTS CAGR (1x) | +9.1% | +15%+ |
