# Plan de Mejora — Post Engine Overhaul

## Estado Actual
- 18 estrategias en portfolio final (16 cross-regime)
- Sharpe portfolio: 2.92, Max DD: -2.5%
- Trade frequency: ~2-3/mes por estrategia (~8 total/mes en portfolio)
- Signal permutation: 3 estrategias con p<0.01 multi-periodo
- Alpha encontrado principalmente en ETH SHORT y BNB SHORT

## Objetivos
1. Aumentar frecuencia de trades (target: 5-10/mes por estrategia)
2. Encontrar alpha en BTC (actualmente el más difícil)
3. Mejorar robustez estadística (más trades = mejor Hansen SPA)
4. Diversificar timeframes y tipos de condiciones

---

## Sprint A: Trade Frequency Boost (1-2 días)

### A1: Evolved Signal Persistence
**Problema**: Signal persistence es fijo en 4 bars. Diferentes condiciones necesitan
diferentes ventanas.

**Cambio**: Agregar `<persistence>` como parámetro de la gramática que evoluciona
junto con la condición. Rango: 1-16 bars. El genoma decidirá si un crossover necesita
1 bar (preciso) o 16 bars (permisivo).

**Impacto esperado**: Estrategias con persistence óptimo por condición → más trades.

### A2: OR Logic Exploitation
**Problema**: AND reduce señales. OR las aumenta. El 50/50 AND/OR actual no se
explota bien porque OR produce señales demasiado frecuentes y ruidosas.

**Cambio**: Agregar OR-with-confirmation: `(A OR B) AND C`. La gramática ya lo
soporta (entry_rule option 3: grouped), pero verificar que se explota. Considerar
agregar más variantes: `A AND (B OR C)`.

### A3: Multi-Entry per Signal Cluster
**Problema**: Una señal produce un trade. Si la señal persiste 10 bars y el trade
se cierra en 5 bars, los bars 6-10 de señal se desperdician.

**Cambio**: Permitir re-entrada inmediata si la señal sigue activa y la posición
se cerró (por TP/SL). Esto duplicaría trades para señales persistentes.

**Riesgo**: Puede generar overtrading. Agregar cooldown mínimo de 2-4 bars.

---

## Sprint B: BTC Alpha Discovery (2-3 días)

### B1: BTC-Specific Grammar Extensions
**Problema**: BTC es el mercado más eficiente en crypto. Los indicadores estándar
no capturan su dinámica.

**Cambio**: Agregar indicadores BTC-específicos a la gramática:
- `BTC_DOM_CHANGE(period)`: cambio en dominancia BTC (requiere datos nuevos)
- `FUNDING_ZSCORE(period)`: z-score de funding rate (ya existe en grammar, revisar)
- `OI_PRICE_DIV(period)`: divergencia OI vs precio
- `LIQUIDATION_IMBALANCE`: ratio de liquidaciones long/short

**Datos**: Necesita ccxt + datos adicionales de Binance futures.

### B2: Higher Timeframe Signals for BTC
**Problema**: BTC es lento (ciclos de semanas/meses). Los indicadores de 15m/1h/4h
pueden ser ruido para BTC.

**Cambio**: Agregar timeframe `1d` a la gramática para BTC. Condiciones como
`RSI(close, 14, 1d) < 30` capturan ciclos macro.

**Impacto**: Pocas señales pero potencialmente más robustas para BTC.

### B3: BTC-Specific Evolution Runs
**Cambio**: Correr evolución solo BTC con population 400, 200 gens, patience 100.
Más recursos computacionales para el mercado más difícil.

---

## Sprint C: Statistical Robustness (1-2 días)

### C1: CPCV (Combinatorial Purged Cross-Validation)
**Estado**: Implementado en `validation/cpcv.py` pero no usado en v6-v9.

**Cambio**: Correr CPCV con 10 folds en las 18 estrategias del portfolio.
Calcular PBO (Probability of Backtest Overfitting). Target: PBO < 0.50.

### C2: DSR (Deflated Sharpe Ratio)
**Estado**: Implementado en `validation/deflated_sharpe.py`.

**Cambio**: Calcular DSR para las 18 estrategias. Corrige por multiple testing,
skew, y kurtosis. Target: DSR > 0.95.

### C3: Walk-Forward Validation
**Cambio**: Dividir 2022-2025 en 6 periodos de 6 meses. Evolucionar en cada
periodo, validar en el siguiente. Si las mismas CONDICIONES aparecen en múltiples
periodos de evolución, eso es evidencia fuerte de alpha persistente.

---

## Sprint D: Advanced Diversity (2-3 días)

### D1: Semantic Diversity via Condition Clustering
**Problema**: Muchas estrategias usan las mismas condiciones base (RSI, STOCH, MFI)
con diferentes parámetros. Falta diversidad semántica.

**Cambio**: Después de evolución, clusterizar estrategias por tipo de indicadores
usados (momentum, volatilidad, volumen, trend). Forzar que el portfolio final
tenga representación de cada cluster.

### D2: Anti-Correlation Objective
**Cambio**: Agregar un tercer objetivo a NSGA-II: correlación negativa con las
mejores estrategias existentes. Esto forzaría al motor a buscar patterns
decorrelacionados.

### D3: Ensemble Strategies
**Cambio**: Combinar 2-3 estrategias simples en un ensemble que opera cuando
la mayoría señala. `Strategy_A.signal AND Strategy_B.signal` como nueva condición.
Esto crea meta-estrategias con mayor confianza por señal.

---

## Sprint E: Market Regime Adaptation (2-3 días)

### E1: Regime-Conditional Position Sizing
**Cambio**: En vez de fixed sizing, ajustar tamaño basado en régimen detectado.
Bull: más capital en LONG strategies. Bear: más en SHORT. Sideways: reducir todo.

### E2: Regime-Gated Strategies
**Cambio**: Cada estrategia solo opera en regímenes donde ha demostrado funcionar.
Si ETH SHORT S1 pierde en sideways, desactivarla durante sideways.
Usa SMA(200) o HMM para detectar régimen.

### E3: Dynamic Portfolio Rebalancing
**Cambio**: Cada mes, recalcular pesos del portfolio basado en rolling 3-month
performance de cada estrategia. Reducir peso de estrategias en drawdown.

---

## Sprint F: Production Hardening (1-2 días)

### F1: Live Signal Generator
**Cambio**: Script que carga las 18 estrategias, conecta a Binance websocket,
y genera señales en real-time. Output: JSON con {strategy_name, signal, direction,
entry_price, tp, sl} para cada señal activa.

### F2: Performance Dashboard
**Cambio**: Script que compara rendimiento live vs OTS expectations.
Alerta si alguna estrategia desvía >2σ de lo esperado.

### F3: Automatic Strategy Rotation
**Cambio**: Si una estrategia tiene drawdown > 1.5x max DD histórico durante
1 mes, se desactiva automáticamente. Se reactiva si el drawdown se recupera.

---

## Prioridad Recomendada

1. **Sprint C** (Statistical Robustness) — valida lo que tenemos
2. **Sprint A** (Trade Frequency) — más trades = más data = más robustez
3. **Sprint B** (BTC Alpha) — llenar el gap de BTC
4. **Sprint F** (Production) — poner en producción
5. **Sprint E** (Regime) — optimizar portfolio
6. **Sprint D** (Diversity) — explorar nuevos territories

## Métricas de Éxito

| Métrica | Actual | Target Sprint A-B | Target Final |
|---------|--------|-------------------|-------------|
| Strategies en portfolio | 18 | 25-30 | 30-40 |
| Avg trades/mes/strategy | 2-3 | 5-10 | 8-15 |
| Cross-regime % | 78% | 85% | 90% |
| Portfolio Sharpe | 2.92 | 3.5+ | 4.0+ |
| Portfolio Max DD | -2.5% | -3.0% | -5.0% |
| BTC strategies | 6 | 10+ | 12+ |
| Signal perm p<0.05 | 3 | 8+ | 12+ |
| PBO (CPCV) | N/A | < 0.50 | < 0.30 |
