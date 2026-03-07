# Plan de Mejora — CriptoGA v3

## Contexto

Los resultados de v2 son honestos pero mixtos: de 166 estrategias evolucionadas, 87 pasaron
validación estadística (PBO + permutación + t-test), pero en OTS (jun-nov 2025, BTC -16%):
- Las estrategias SHORT dominaron (100% éxito OTS vs 50% LONG)
- El ensemble logró +12.3% CAGR vs B&H -31.6%
- Hansen SPA no alcanzó significancia (pocas trades, ~20-50 por estrategia)
- Señales basadas en price×volume fueron espurias (escala-dependientes)
- Las estrategias son **régimen-dependientes**: funcionan en bear, fallan en bull

**Diagnóstico raíz**: el sistema actual busca alpha con un toolkit limitado (indicadores
técnicos clásicos en un solo activo y timeframe). Para avanzar, necesitamos expandir
en 3 ejes: **datos**, **representación**, y **optimización**.

---

## Prioridades (ordenadas por impacto esperado / esfuerzo)

| # | Mejora | Impacto | Esfuerzo | Sprint |
|---|--------|---------|----------|--------|
| 1 | Datos alternativos (funding, OI, sentiment) | Alto | Medio | S6 |
| 2 | Multi-activo (ETH, SOL + top líquidos) | Alto | Medio | S7 |
| 3 | Multi-timeframe (15m + 1h + 4h) | Alto | Medio | S6 |
| 4 | CMA-ES optimización local de parámetros | Medio-Alto | Bajo | S8 |
| 5 | Walk-forward adaptativo | Medio | Medio | S8 |
| 6 | Regime-switching dinámico | Medio | Medio | S9 |
| 7 | RL híbrido (GE→RL fine-tuning) | Alto (teórico) | Alto | S10 |
| 8 | Infraestructura de deploy multi-bot | Medio | Bajo | S7 |

---

## Sprint 6: Datos Alternativos + Multi-Timeframe

### 6A. Datos Alternativos

**Problema**: las señales actuales solo usan OHLCV — la misma información que tiene
cualquier trader retail. No hay edge informacional.

**Datos disponibles vía ccxt (confirmado)**:

| Dato | Método ccxt | Granularidad | Latencia |
|------|------------|--------------|----------|
| Funding Rate histórico | `fetch_funding_rate_history()` | 8h (Binance) | Tiempo real |
| Open Interest | `fetch_open_interest_history()` | 15m, 1h, 4h, 1d | ~15 min |
| Long/Short Ratio (global) | `fetch_long_short_ratio_history()` | 15m+ | ~15 min |
| Top Trader L/S (positions) | `fetch_long_short_ratio_history(params={'type':'position'})` | 15m+ | ~15 min |
| Top Trader L/S (accounts) | `fetch_long_short_ratio_history(params={'type':'account'})` | 15m+ | ~15 min |
| Taker Buy/Sell Volume | `fetch_taker_buy_sell_ratio()` (o REST directo) | 15m+ | ~15 min |

**Plan de implementación**:

1. **`data/alternative.py`**: módulo para descargar y cachear datos alternativos
   - Funciones: `fetch_funding_rate()`, `fetch_open_interest()`, `fetch_ls_ratio()`,
     `fetch_taker_volume()`
   - Cache en disco (parquet) con merge temporal al DataFrame principal
   - Resample funding rate de 8h → 15m (forward fill)
   - Normalización: OI como z-score rolling 48h, funding como valor absoluto

2. **Nuevos indicadores derivados**:
   - `FUNDING_RATE`: valor actual del funding (positivo = longs pagan)
   - `FUNDING_ZSCORE`: z-score del funding sobre ventana rolling
   - `OI_CHANGE`: cambio porcentual del OI vs N barras atrás
   - `OI_PRICE_DIV`: divergencia OI vs precio (OI sube + precio baja = bearish)
   - `LS_RATIO`: ratio long/short global
   - `LS_RATIO_CHANGE`: delta del L/S ratio
   - `TAKER_RATIO`: buy volume / sell volume (>1 = presión compradora)
   - `TAKER_IMBALANCE`: taker ratio - media rolling

3. **Extensión de la gramática BNF**:
   ```
   <indicator> ::= ... | FUNDING_RATE | FUNDING_ZSCORE | OI_CHANGE
                 | OI_PRICE_DIV | LS_RATIO | TAKER_RATIO | TAKER_IMBALANCE
   ```
   - Los nuevos terminales entran como columnas precalculadas en IndicatorCache
   - El mapper los trata igual que RSI, MACD, etc.
   - La gramática permite combinarlos con indicadores técnicos (e.g., RSI < 30 AND FUNDING > 0.01)

4. **Hipótesis a testear**:
   - Funding rate extremo predice reversión (crowded trade)
   - Divergencia OI/precio señala liquidaciones inminentes
   - Taker imbalance persistente confirma tendencia
   - Combinación de técnicos + alternativos > técnicos solos

**Riesgos**:
- Open Interest tiene historia limitada (~2020 en adelante)
- Funding rate solo se actualiza cada 8h — señal de baja frecuencia
- Posible data snooping si se añaden demasiados indicadores sin ajustar el DSR

**Mitigación**: aumentar la penalización por complejidad en fitness, y ajustar el
DSR por el número total de indicadores disponibles (no solo estrategias testeadas).

### 6B. Multi-Timeframe

**Problema**: las señales en 15m son ruidosas. Los traders profesionales confirman
señales en timeframes superiores (1h, 4h) antes de ejecutar en el inferior.

**Arquitectura**:

1. **Datos**: NO descargar 1h y 4h por separado. Resample 15m → 1h y 4h in-memory:
   ```python
   df_1h = df_15m.resample('1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
   df_4h = df_15m.resample('4h').agg(...)
   ```
   - Ventaja: un solo download, datos perfectamente alineados
   - Indicadores se calculan sobre cada timeframe independientemente

2. **Extensión de IndicatorCache**:
   - `IndicatorCache` pasa a recibir un dict de DataFrames: `{'15m': df_15m, '1h': df_1h, '4h': df_4h}`
   - Columnas nombradas con prefijo: `RSI_14_15m`, `RSI_14_1h`, `RSI_14_4h`
   - Todos los indicadores disponibles en todos los timeframes

3. **Extensión de la gramática**:
   ```
   <signal> ::= <condition> | <condition> AND <condition>
   <condition> ::= <value> <comparator> <value>
   <value> ::= <indicator>(<source>, <period>) [<timeframe>]
   <timeframe> ::= 15m | 1h | 4h
   ```
   - Default: 15m (backwards compatible)
   - El genoma necesita un codón extra para seleccionar timeframe

4. **Reglas de ejecución**:
   - Señal de ENTRY: evaluada en 15m (máxima granularidad)
   - Filtros de CONFIRMACIÓN: evaluados en 1h o 4h
   - Ejemplo evolucionado posible: "RSI_15m < 30 AND MACD_4h > 0" (sobreventa local + tendencia macro alcista)

5. **Backtester**: la ejecución sigue en 15m. Los indicadores de 1h/4h se alinean
   por timestamp al DataFrame de 15m (forward fill del último valor cerrado).

**Impacto esperado**: reduce señales falsas al requerir confirmación multi-escala.
Las estrategias SHORT que funcionaron en OTS podrían mejorar con filtro de tendencia 4h.

---

## Sprint 7: Multi-Activo + Deploy Multi-Bot

### 7A. Expansión Multi-Activo

**Problema**: BTC/USDT es un solo mercado. Las ineficiencias pueden existir en activos
menos eficientes (altcoins con menor liquidez/atención institucional).

**Activos candidatos** (por liquidez y disponibilidad de datos):

| Activo | Vol 24h (aprox) | Desde | Notas |
|--------|----------------|-------|-------|
| ETH/USDT | $15B+ | 2019 | Segundo más líquido, correlación alta con BTC |
| SOL/USDT | $3B+ | 2021 | Alta volatilidad, posible más alpha |
| BNB/USDT | $1B+ | 2019 | Ecosistema Binance |
| DOGE/USDT | $1B+ | 2021 | Retail-driven, patrones de sentiment |
| ARB/USDT | $500M+ | 2023 | Nuevo, menos eficiente |
| ~1,946 pares | Variable | Variable | Disponibles en Binance Futures |

**Plan**:

1. **`data/multi_asset.py`**: extensión de `loader.py` para múltiples activos
   - Descarga paralela (asyncio o threads)
   - Cache por activo en `data/cache/{symbol}_15m.parquet`
   - Validación: mínimo 2 años de datos, sin gaps > 24h

2. **Evolución por activo**:
   - Opción A: **evolución independiente** — correr el pipeline completo para cada activo
     por separado. Simple, paralelizable. Un genoma óptimo por activo.
   - Opción B: **evolución multi-activo** — fitness promedio sobre N activos.
     Encuentra estrategias universales pero puede ser mediocre en todas.
   - **Recomendación: Opción A primero** (un bot por activo), luego Opción B como
     experimento comparativo para el paper.

3. **Filtrado de activos**:
   - Volumen diario promedio > $500M (para ejecutabilidad)
   - Historia > 2 años (para validación estadística)
   - Spread < 0.05% (para que los costos no se coman el alpha)
   - Candidatos iniciales: ETH, SOL, BNB (3 activos además de BTC)

4. **Diversificación de portfolio**:
   - Portfolio final = mejores estrategias de cada activo
   - Decorrelación inter-activo (ya tenemos `backtest/correlation.py`)
   - Sizing por activo basado en volatilidad relativa (vol parity)

### 7B. Deploy Multi-Bot

**Problema**: actualmente un solo servicio systemd para BTC. Necesitamos escalar.

**Plan**:

1. **Template de servicio systemd parametrizado**:
   ```ini
   # cripto-ga-bot@.service
   [Service]
   ExecStart=/home/juanma/cripto_ga/venv/bin/python -m live --symbol %i
   ```
   - `systemctl start cripto-ga-bot@BTCUSDT`
   - `systemctl start cripto-ga-bot@ETHUSDT`
   - Cada instancia corre con su propia config y state

2. **`live/config.py`**: acepta `--symbol` como argumento CLI
   - Carga estrategias específicas para ese símbolo desde `results/{symbol}/`
   - State file separado por símbolo: `state_{symbol}.json`

3. **Monitoreo centralizado**:
   - Script `live/monitor.py`: consulta status de todas las instancias
   - Alerta simple por email/telegram si algún bot se detiene o pierde > X%

4. **Recursos del servidor**:
   - Hetzner actual (CX22: 2 vCPU, 4GB RAM) puede correr ~5 bots sin problemas
   - Cada bot usa <100MB RAM y casi nada de CPU (wake cada 15 min)

---

## Sprint 8: CMA-ES + Walk-Forward

### 8A. CMA-ES como Optimizador Local

**Problema**: GE es excelente para descubrir ESTRUCTURA (qué indicadores combinar),
pero ineficiente para optimizar PARÁMETROS continuos (períodos, thresholds). Los
codones discretos solo pueden hacer ±1 saltos, tardando muchas generaciones en
encontrar el RSI period óptimo de 21 si empezó en 14.

**Plan**:

1. **Pipeline de 2 fases**:
   ```
   Fase 1: GE + MAP-Elites → Top K estructuras (K=20)
   Fase 2: CMA-ES → Optimizar parámetros de cada estructura
   ```

2. **`evolution/cmaes.py`**: wrapper de CMA-ES (usando `cma` library de Nikolaus Hansen)
   - Input: estructura fija (qué indicadores, qué comparadores) + rangos de parámetros
   - Variables a optimizar: períodos de indicadores, thresholds, multiplicadores ATR
   - Fitness: mismo Sortino que en GE, evaluado con window rotation
   - Budget: ~500-1000 evaluaciones por estructura (CMA-ES converge rápido)

3. **Extracción de parámetros del genoma**:
   - Dado un genoma GE, identificar qué codones son "estructura" vs "parámetro"
   - Los codones de estructura se fijan
   - Los codones de parámetro se mapean a variables continuas para CMA-ES
   - Ejemplo: genoma [3, 7, 2, 14, 1, 30] donde 14=RSI_period y 30=threshold
     → CMA-ES optimiza [period ∈ (5,50), threshold ∈ (10,90)]

4. **Validación**: las estrategias optimizadas con CMA-ES pasan por el MISMO pipeline
   de validación (CPCV + PBO + permutación). No se asume que son mejores solo porque
   el fitness in-sample subió.

**Impacto esperado**: 10-30% mejora en fitness para estructuras que ya son prometedoras.
Relativamente barato de implementar (~200 líneas + library).

### 8B. Walk-Forward Adaptativo

**Problema**: actualmente la validación es estática (evolucionar una vez, validar una vez,
OTS una vez). En producción, los mercados cambian y las estrategias decaen.

**Plan**:

1. **Walk-Forward con re-evolución periódica**:
   ```
   Window 1: Train [2023-01 → 2024-06] → Test [2024-07 → 2024-09]
   Window 2: Train [2023-04 → 2024-09] → Test [2024-10 → 2024-12]
   Window 3: Train [2023-07 → 2024-12] → Test [2025-01 → 2025-03]
   Window 4: Train [2023-10 → 2025-03] → Test [2025-04 → 2025-06]
   ```
   - Cada ventana: evolución completa (50 gens) + selección del mejor
   - Test: backtest out-of-sample en la ventana siguiente
   - Resultado: secuencia de returns OOS concatenados → métrica realista

2. **Anchored vs Expanding window**:
   - **Expanding** (recomendado): el train siempre empieza en 2023-01, crece
   - **Rolling**: ventana fija de N meses, descarta datos antiguos
   - Empezar con expanding, comparar con rolling en el paper

3. **`validation/walk_forward.py`**:
   - Orquesta múltiples corridas de evolución con diferentes cortes temporales
   - Paralelizable (cada ventana es independiente)
   - Genera métricas agregadas: Sortino WF, Sharpe WF, max DD WF

4. **Re-evolución en producción**:
   - Cada mes (o cada N barras), el bot puede re-evolucionar con datos actualizados
   - Las nuevas estrategias reemplazan a las antiguas solo si pasan validación
   - Esto es material excelente para el paper: "adaptive strategy discovery"

---

## Sprint 9: Regime-Switching Dinámico

**Problema**: las estrategias SHORT dominaron el OTS porque fue bear market. En bull,
hubieran perdido. Necesitamos un meta-sistema que active/desactive estrategias
según el régimen actual.

**Plan**:

1. **Detector de régimen mejorado** (`data/regime_detector.py` v2):
   - Actual: volatilidad + tendencia → bull/bear/sideways
   - Mejorar con: Hidden Markov Model (HMM) de 3-4 estados
   - Features: returns, volatilidad, volumen, OI change, funding
   - Usar `hmmlearn` library

2. **Meta-estrategia de selección**:
   - Cada estrategia evolucionada tiene un perfil de régimen (en qué régimen funciona)
   - El meta-selector activa solo las estrategias apropiadas al régimen actual
   - Sizing proporcional a la confianza en la detección de régimen

3. **Evolución regime-aware**:
   - Fitness ponderado por régimen: penalizar estrategias que solo funcionan en 1 régimen
   - O alternativamente: evolucionar specialists por régimen + meta-selector
   - Comparar ambos enfoques en el paper

4. **Transiciones de régimen**:
   - El momento más peligroso es la transición (bull→bear)
   - Reducir sizing durante períodos de incertidumbre de régimen
   - Buffer de transición: no cambiar estrategias hasta N barras de confirmación

---

## Sprint 10: RL Híbrido (Exploratorio)

**Problema**: GE descubre reglas fijas. Los mercados son no-estacionarios. RL puede
adaptarse en tiempo real.

**Advertencia**: este sprint es especulativo. Solo vale la pena si los sprints 6-9
producen estrategias con alpha demostrable. Si no hay alpha con datos mejorados,
RL tampoco va a encontrarlo.

**Plan** (si se aprueba):

1. **GE→RL pipeline**:
   - GE descubre la estructura base (e.g., "comprar cuando RSI < X AND MACD cruza Y")
   - RL fine-tunea los parámetros X, Y en tiempo real usando PPO o SAC
   - La estructura GE actúa como inductive bias para el RL

2. **Observation space**: últimos N valores de los indicadores de la estrategia GE
3. **Action space**: continuo [0, 1] = sizing (0 = sin posición, 1 = full)
4. **Reward**: PnL ajustado por riesgo (Sortino incremental)

5. **Framework**: Stable-Baselines3 (PPO/SAC)
6. **Riesgo**: overfitting extremo con RL en finanzas. Necesita:
   - Train/test split estricto
   - Regularización fuerte
   - Evaluación estadística idéntica al pipeline GE (CPCV, PBO, etc.)

**Impacto esperado**: potencialmente alto, pero con riesgo de overfitting proporcionalmente
alto. Material excelente para el paper independientemente del resultado.

---

## Mejoras Menores (transversales, sin sprint dedicado)

### Corrección de Señales Espurias
- **Problema**: la señal "BBAND_LOWER(open,30) CROSSES volume" comparaba USD con BTC.
- **Fix**: normalizar todas las señales a la misma escala antes de comparar.
  En la gramática, `<source>` solo puede compararse con otro `<source>` del mismo tipo.
  O mejor: todas las comparaciones se hacen en z-score.
- **Implementar en S6** como parte de la extensión de gramática.

### Trade Count Mínimo
- **Problema**: Hansen SPA no alcanza significancia con 20-50 trades.
- **Análisis**: PBO < 0.20 predijo 63% éxito OTS; PBO > 0.30 predijo 0%.
- **Fix**: aumentar `min_trades` constraint de 30 a 50 en fitness.
  Penalización gradual: `trade_penalty = max(0, (50 - n_trades) * 0.1)`.
- **Implementar en S6** como ajuste de fitness.

### Penalización por Complejidad Adaptativa
- **Problema**: parsimony pressure fija no escala con el número de indicadores disponibles.
- **Fix**: `parsimony = -0.01 * n_nodes * (n_available_indicators / n_base_indicators)`.
  A medida que añadimos más indicadores, la penalización aumenta proporcionalmente.

### Cache de Datos Alternativos
- Los datos de funding/OI tienen rate limits en Binance.
- Descargar una vez y cachear en parquet. Update incremental diario.
- Para evolución, los datos históricos son estáticos.

---

## Orden de Ejecución Recomendado

```
S6: Datos Alternativos + Multi-Timeframe
    ├── 6A: data/alternative.py + nuevos indicadores en gramática
    ├── 6B: IndicatorCache multi-TF + gramática con <timeframe>
    └── Re-correr evolución con datos expandidos → comparar resultados
        (¿mejora el número de estrategias que pasan validación?)

S7: Multi-Activo + Deploy
    ├── 7A: ETH, SOL, BNB — evolución independiente por activo
    ├── 7B: Template systemd + live/config.py paramétrico
    └── Portfolio multi-activo decorrelacionado

S8: CMA-ES + Walk-Forward
    ├── 8A: Optimización local de parámetros post-GE
    ├── 8B: Walk-forward con re-evolución periódica
    └── Comparar: GE solo vs GE+CMA-ES (para el paper)

S9: Regime-Switching
    ├── HMM para detección de régimen
    ├── Meta-selector de estrategias
    └── Comparar: regime-agnostic vs regime-aware (para el paper)

S10: RL Híbrido (condicional a resultados de S6-S9)
    ├── Solo si hay alpha demostrable con datos mejorados
    ├── GE structure → RL parameter tuning
    └── Comparar: GE vs GE+RL (para el paper)
```

## Estimación de Impacto Acumulado

| Después de | Estrategias esperadas que pasan OTS | Mejora vs actual |
|------------|--------------------------------------|------------------|
| S6 (datos + MTF) | 5-15 (vs 3 actual) | +100-400% |
| S7 (multi-activo) | 15-40 (sumando todos los activos) | Escala lineal |
| S8 (CMA-ES + WF) | +20-30% sobre base | Optimización fina |
| S9 (regime) | +10-20% por adaptación | Robustez |
| S10 (RL) | Incierto — experimental | Potencial alto |

**Nota**: estas estimaciones son especulativas. La hipótesis nula sigue siendo posible:
puede que no exista alpha explotable en crypto 15m con indicadores técnicos + alternativos.
Si S6 no mejora significativamente los resultados, debemos considerar seriamente que
la conclusión del paper sea negativa (lo cual sigue siendo un resultado válido y publicable).

## Métricas de Decisión (Go/No-Go)

Después de cada sprint, evaluar:

1. **¿Aumentó el % de estrategias que pasan validación?** (PBO < 0.50 + perm p < 0.05)
2. **¿Mejoraron las métricas OOS promedio?** (Sortino, Calmar, win rate)
3. **¿Se redujo la dependencia de régimen?** (funciona en bull Y bear)

Si después de S6+S7 no hay mejora significativa → pivotar hacia paper de resultado
negativo y documentar por qué el alpha no existe en este dominio.

---

## Para el Paper

Cada sprint produce material para secciones específicas:

- S6 → "Extensión del espacio de features" + "Arquitectura multi-resolución"
- S7 → "Generalización multi-activo" + "Portfolio construction"
- S8 → "Optimización híbrida GE+CMA-ES" + "Walk-forward validation"
- S9 → "Adaptive regime detection" + "Dynamic strategy selection"
- S10 → "Hybrid evolutionary-RL approach" (si aplica)

Cada comparación (antes/después de la mejora) es una tabla o figura del paper.
Las comparaciones negativas también se documentan — esto es ciencia, no marketing.
