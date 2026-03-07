# CriptoGA v2 — Plan de Implementacion

## Resumen Ejecutivo

Reescritura del sistema de descubrimiento de estrategias de trading cripto.
Se reemplaza un GA basado en building blocks con parametros hardcodeados por
Grammatical Evolution (GE) con evaluacion vectorizada, MAP-Elites para
diversidad, y un pipeline de validacion estadistica riguroso (CPCV, DSR, PBO).

**Estado actual**: v1 no funciona. Estanca en generacion ~4. Tests estadisticos
fallan (SPA p=0.44, WRC p=0.58). Diagnosticadas 9 causas raiz.

**Objetivo**: Pipeline reproducible que descubra estrategias estadisticamente
robustas, o demuestre rigurosamente que no existe alpha explotable en BTC/USDT 15m.
Ambos resultados son publicables.

---

## Estructura de Directorios Final

```
cripto_ga/
├── CLAUDE.md
├── PLAN.md                    # Este archivo
├── config.yaml                # Config simplificado para v2
├── main.py                    # CLI thin: evolve | validate | report
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   ├── loader.py              # CONSERVAR — ccxt Binance loader
│   └── regime_detector.py     # NUEVO — bull/bear/sideways
│
├── grammar/
│   ├── __init__.py
│   ├── bnf.py                 # NUEVO — definicion BNF
│   ├── mapper.py              # NUEVO — codones → fenotipo
│   └── simplifier.py          # NUEVO — canonicalizacion
│
├── strategy/
│   ├── __init__.py
│   ├── phenotype.py           # NUEVO — Strategy dataclass
│   ├── vectorized_eval.py     # NUEVO — señales vectorizadas
│   └── parameters.py          # NUEVO — rangos de parametros
│
├── evolution/
│   ├── __init__.py
│   ├── engine.py              # NUEVO — loop de evolucion
│   ├── operators.py           # NUEVO — crossover/mutacion sobre codones
│   ├── selection.py           # NUEVO — tournament, lexicase
│   ├── fitness.py             # NUEVO — multi-objetivo, parsimonia
│   ├── archive.py             # NUEVO — MAP-Elites
│   └── island.py              # NUEVO — modelo de islas
│
├── backtest/
│   ├── __init__.py
│   ├── runner.py              # CONSERVAR — adaptar interface
│   ├── exits.py               # CONSERVAR tal cual
│   ├── metrics.py             # CONSERVAR — agregar DSR
│   ├── futures_position_sizing.py  # CONSERVAR tal cual
│   ├── final_backtest.py      # CONSERVAR tal cual
│   ├── correlation.py         # CONSERVAR tal cual
│   └── sampling.py            # NUEVO — window rotation + CPCV support
│
├── validation/
│   ├── __init__.py
│   ├── cpcv.py                # NUEVO
│   ├── deflated_sharpe.py     # NUEVO
│   ├── pbo.py                 # NUEVO
│   ├── signal_permutation.py  # NUEVO
│   ├── bootstrap.py           # CONSERVAR — fix bugs
│   ├── hansen_spa.py          # CONSERVAR
│   └── white_rc.py            # CONSERVAR — fix block bootstrap
│
├── analysis/                  # CONSERVAR todo
│   ├── evolution_analytics.py
│   ├── monte_carlo.py
│   └── final_visualization.py
│
├── reports/                   # CONSERVAR todo
│   ├── latex_exporter.py
│   ├── pattern_explainer.py
│   ├── report_generator.py
│   └── visualizations.py
│
└── tests/
    ├── __init__.py
    ├── test_grammar.py
    ├── test_mapper.py
    ├── test_vectorized_eval.py
    ├── test_engine.py
    ├── test_fitness.py
    ├── test_cpcv.py
    ├── test_signal_permutation.py
    └── test_integration.py
```

---

## Sprint 1: Foundation — Gramatica + Evaluacion Vectorizada

### Objetivo
Sistema capaz de: genoma aleatorio → decodificar a estrategia → generar señales
vectorizadas sobre datos reales. Sin evolucion todavia, solo la representacion.

### 1.1 grammar/bnf.py

Define la gramatica BNF como diccionario de Python. Cada regla tiene
alternativas de produccion. Los terminales incluyen parametros numericos
que van a evolucionar.

```
Reglas principales:
  <strategy>     → <entry_rule> con <exit_params>
  <entry_rule>   → combinaciones de <condition> via AND/OR (max 4 condiciones)
  <condition>    → <indicator> <comparator> <threshold>
                  | <indicator> <comparator> <indicator>
                  | <indicator> CROSSES_ABOVE <indicator>
                  | <indicator> CROSSES_BELOW <indicator>
  <indicator>    → SMA/EMA/RSI/BBAND/ATR/MACD/STOCH/VOLUME_SMA/raw OHLCV
  <source>       → close | open | high | low | volume
  <comparator>   → > | < | >= | <=
  <period>       → Fibonacci: 5, 8, 13, 21, 34, 55, 89
  <threshold>    → rangos dependientes del tipo de indicador
  <exit_params>  → TP y SL como multiplos de ATR
```

Estructura del diccionario:
```python
GRAMMAR = {
    "<strategy>": ["<entry_rule> <exit_params>"],
    "<entry_rule>": [
        "<condition>",
        "<condition> AND <condition>",
        "<condition> AND <condition> AND <condition>",
        "<condition> OR <condition>",
        "(<condition> AND <condition>) OR <condition>",
    ],
    # ... etc
}
```

Cada clave es un non-terminal. Cada valor es lista de producciones (strings
con non-terminales embebidos). El mapper usa modular arithmetic sobre los
codones para elegir que produccion tomar.

Datos a tener en cuenta:
- Limitar profundidad maxima de recursion (wrapping limit = 3)
- Si se agota el genoma, hacer wrapping (volver al inicio del vector de codones)
- Maximo ~50 codones por genoma (suficiente para estrategias de 3-4 condiciones)

### 1.2 grammar/mapper.py

Funcion principal: `decode(genome: List[int]) -> Strategy`

Algoritmo:
1. Empezar con el simbolo inicial `<strategy>`
2. Para cada non-terminal en la expresion actual:
   a. Tomar el siguiente codon del genoma
   b. `produccion = reglas[non_terminal][codon % len(reglas[non_terminal])]`
   c. Reemplazar el non-terminal por la produccion elegida
3. Repetir hasta que solo queden terminales
4. Si se agotan los codones, hacer wrapping (max 3 veces)
5. Si despues de wrapping quedan non-terminales, el genoma es invalido

Retorna un objeto Strategy (phenotype.py) o None si es invalido.

El mapper debe trackear:
- Cuantos codones se usaron (codones efectivos)
- Profundidad del arbol resultante (para parsimonia)
- Si hubo wrapping

### 1.3 grammar/simplifier.py

Canonicaliza la expresion decodificada:
- `RSI(close, 14) > 30 AND RSI(close, 14) > 30` → `RSI(close, 14) > 30` (dedup)
- `A AND TRUE` → `A` (simplificar tautologias)
- `SMA(close, 20) > SMA(close, 20)` → FALSE → genoma invalido
- Ordenar condiciones dentro de AND/OR para canonicalizar (A AND B == B AND A)

Esto reduce el search space efectivo y evita que el GA mantenga
duplicados funcionales como individuos distintos.

No necesita ser sofisticado. Reglas simples aplicadas post-decodificacion.

### 1.4 strategy/phenotype.py

```python
@dataclass
class Strategy:
    """Fenotipo decodificado de un genoma GE."""
    genome: List[int]              # Genoma original (codones)
    expression: str                # Expresion legible: "RSI(close,14) < 30 AND ..."
    conditions: List[Condition]    # Condiciones parseadas
    logic: str                     # 'AND' | 'OR' | expresion mixta
    tp_atr_mult: float             # Take profit en multiplos de ATR
    sl_atr_mult: float             # Stop loss en multiplos de ATR
    n_nodes: int                   # Complejidad (para parsimonia)
    codons_used: int               # Cuantos codones se consumieron
    wrapping_count: int            # Cuantas veces se hizo wrapping

    # Estos se llenan post-evaluacion
    fitness: Optional[Tuple[float, float]] = None  # (sortino, calmar) multi-obj
    metrics: Optional[Dict] = None
    n_trades: int = 0

@dataclass
class Condition:
    """Una condicion individual dentro de la estrategia."""
    left: str          # "RSI(close, 14)" o "close"
    comparator: str    # ">", "<", ">=", "<=", "CROSSES_ABOVE", "CROSSES_BELOW"
    right: str         # "30" (threshold) o "SMA(close, 50)" (indicator)
```

### 1.5 strategy/vectorized_eval.py

El corazon del speedup. Compila un Strategy a operaciones vectorizadas.

Flujo:
1. Recibir Strategy + DataFrame con OHLCV
2. Para cada Condition, calcular el indicador izquierdo y derecho como Series
3. Aplicar comparador → Series booleana por condicion
4. Combinar condiciones segun logica (AND/OR) → Series booleana final
5. Retornar signal Series (True = entrada)

Implementacion de indicadores:
```python
INDICATOR_FUNCTIONS = {
    'SMA':          lambda df, src, period: df[src].rolling(period).mean(),
    'EMA':          lambda df, src, period: df[src].ewm(span=period).mean(),
    'RSI':          lambda df, src, period: _compute_rsi(df[src], period),
    'BBAND_UPPER':  lambda df, src, period, std: _compute_bb(df[src], period, std, 'upper'),
    'BBAND_LOWER':  lambda df, src, period, std: _compute_bb(df[src], period, std, 'lower'),
    'ATR':          lambda df, period: _compute_atr(df, period),
    'MACD_LINE':    lambda df, fast, slow: _compute_macd(df['close'], fast, slow, 'line'),
    'MACD_SIGNAL':  lambda df, fast, slow, sig: _compute_macd(df['close'], fast, slow, 'signal', sig),
    'STOCH_K':      lambda df, period: _compute_stoch(df, period),
    'VOLUME_SMA':   lambda df, period: df['volume'].rolling(period).mean(),
}
```

Para CROSSES_ABOVE/BELOW:
```python
# A crosses above B cuando:
# - barra actual: A > B
# - barra anterior: A <= B
crosses_above = (a > b) & (a.shift(1) <= b.shift(1))
```

Caching de indicadores: si multiples condiciones usan el mismo indicador
(ej: RSI(close,14) aparece dos veces), calcularlo una sola vez.

### 1.6 strategy/parameters.py

Define rangos validos para cada terminal de la gramatica:
```python
PARAMETER_RANGES = {
    'period':    [5, 8, 13, 21, 34, 55, 89],      # Fibonacci
    'fast':      [8, 12, 16],
    'slow':      [21, 26, 34],
    'signal':    [5, 9, 13],
    'bb_std':    [1.5, 2.0, 2.5, 3.0],
    'rsi_thresh': list(range(20, 81, 5)),           # 20, 25, ..., 80
    'tp_mult':   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'sl_mult':   [0.8, 1.0, 1.2, 1.5, 2.0],
}
```

Tambien: constraint de que tp_mult > sl_mult (R:R favorable).

### 1.7 Tests Sprint 1

**test_grammar.py:**
- Gramatica es un diccionario valido (todos los non-terminales referenciados existen)
- Todas las producciones usan non-terminales definidos o terminales
- No hay recursion infinita (toda derivacion termina eventualmente)

**test_mapper.py:**
- Genoma aleatorio produce Strategy valida (o None si wrapping excede limite)
- Mismo genoma siempre produce misma Strategy (determinismo)
- Genomas distintos producen estrategias distintas (al menos frecuentemente)
- Wrapping funciona correctamente
- codons_used <= len(genome) * (1 + max_wrapping)

**test_vectorized_eval.py:**
- Indicadores individuales producen valores correctos vs calculo manual
- RSI en [0, 100] siempre
- SMA coincide con pandas rolling mean
- CROSSES_ABOVE/BELOW detectan cruces correctamente
- Signal es Series booleana del mismo largo que DataFrame
- 1000 estrategias random evaluadas en <10 segundos (benchmark)

### Exit Criterion Sprint 1
```python
# Este test debe pasar:
def test_full_pipeline_speed():
    df = load_btc_data('2024-01-01', '2024-02-01')  # 1 mes
    genomes = [random_genome(50) for _ in range(1000)]
    strategies = [decode(g) for g in genomes]
    valid = [s for s in strategies if s is not None]

    t0 = time.time()
    for s in valid:
        signals = vectorized_evaluate(s, df)
    elapsed = time.time() - t0

    assert len(valid) >= 800  # >80% de genomas producen estrategia valida
    assert elapsed < 10.0     # <10 segundos para 800+ evaluaciones
```

---

## Sprint 2: Evolution Engine

### Objetivo
GA funcional con la nueva representacion. Reemplaza el loop inline de main.py.

### 2.1 evolution/operators.py

Operadores geneticos sobre vectores de enteros (codones).

**Crossover (one-point):**
```python
def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """One-point crossover. Punto elegido al azar."""
    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
```

**Mutacion (3 tipos):**
```python
def mutate(genome: List[int], rate: float = 0.1) -> List[int]:
    """Muta codones individuales."""
    result = genome.copy()
    for i in range(len(result)):
        if random.random() < rate:
            mutation_type = random.choice(['increment', 'random', 'swap'])
            if mutation_type == 'increment':
                result[i] += random.choice([-1, 1])  # ±1 (ajuste fino)
            elif mutation_type == 'random':
                result[i] = random.randint(0, 255)     # nuevo valor (exploracion)
            elif mutation_type == 'swap' and i < len(result) - 1:
                result[i], result[i+1] = result[i+1], result[i]  # swap vecinos
    return result
```

Los codones son enteros 0-255. Mutacion ±1 hace ajuste fino de parametros.
Mutacion random hace exploracion de estructura. Swap reordena condiciones.

### 2.2 evolution/selection.py

**Tournament selection:**
```python
def tournament_select(population, k=3):
    """Selecciona individuo por torneo de tamaño k."""
    candidates = random.sample(population, k)
    return max(candidates, key=lambda s: s.fitness[0])  # Por Sortino
```

**Lexicase selection (para multi-objetivo):**
Evalua individuos en orden aleatorio de objetivos. En cada paso, filtra
los que no son mejores o iguales que la mediana en ese objetivo. Sobrevive
el ultimo. Promueve diversidad porque distintas ejecuciones privilegian
distintos objetivos.

```python
def lexicase_select(population, objectives):
    """Lexicase selection over multiple objectives."""
    candidates = list(population)
    shuffled_obj = random.sample(range(len(objectives)), len(objectives))
    for obj_idx in shuffled_obj:
        if len(candidates) == 1:
            break
        values = [get_objective(c, obj_idx) for c in candidates]
        threshold = np.median(values)
        candidates = [c for c, v in zip(candidates, values) if v >= threshold]
    return random.choice(candidates)
```

### 2.3 evolution/fitness.py

**Multi-objetivo, sin normalizacion:**

```python
def evaluate_fitness(strategy: Strategy, df: DataFrame, windows: List) -> Strategy:
    """
    Evalua estrategia en multiples ventanas. Retorna fitness como tupla real.

    Fitness = (sortino, calmar) — valores reales, NO normalizados.
    Parsimony pressure: resta 0.01 * n_nodes al sortino.

    Constraints (pass/fail, no son objetivos):
    - min_trades >= 30
    - max_drawdown < 0.30
    - win_rate > 0.35
    Si falla constraint: fitness = (-999, -999)
    """
```

El backtest se ejecuta via runner.py (conservado de v1, con interface adaptada).
Las metricas se calculan via metrics.py (conservado).

Cada generacion usa DISTINTAS ventanas (window rotation, ver sampling.py).

### 2.4 evolution/engine.py

```python
class EvolutionEngine:
    def __init__(self, config: dict, data: DataFrame):
        self.config = config
        self.data = data  # Solo datos de evolucion (excl. OTS)
        self.population: List[Strategy] = []
        self.generation: int = 0
        self.best_ever: Optional[Strategy] = None
        self.history: List[GenerationStats] = []

    def initialize(self, pop_size: int, genome_length: int = 50):
        """Genera poblacion inicial de genomas aleatorios."""
        self.population = []
        for _ in range(pop_size):
            genome = [random.randint(0, 255) for _ in range(genome_length)]
            strategy = decode(genome)
            if strategy is not None:
                self.population.append(strategy)
        # Rellenar si hay invalidos
        while len(self.population) < pop_size:
            genome = [random.randint(0, 255) for _ in range(genome_length)]
            strategy = decode(genome)
            if strategy is not None:
                self.population.append(strategy)

    def step(self) -> GenerationStats:
        """Una generacion. Testeable unitariamente."""
        # 1. Sample nuevas ventanas (window rotation)
        windows = sample_windows(self.data, self.config)

        # 2. Evaluar toda la poblacion
        for strategy in self.population:
            evaluate_fitness(strategy, self.data, windows)

        # 3. Registrar stats
        stats = self._compute_stats()
        self.history.append(stats)

        # 4. Elitismo (top 4-6%)
        n_elite = max(2, int(len(self.population) * 0.05))
        elite = sorted(self.population, key=lambda s: s.fitness[0], reverse=True)[:n_elite]

        # 5. Seleccion + reproduccion
        new_pop = list(elite)
        while len(new_pop) < len(self.population):
            p1 = tournament_select(self.population)
            p2 = tournament_select(self.population)
            c1_genome, c2_genome = crossover(p1.genome, p2.genome)
            c1_genome = mutate(c1_genome, rate=self.config['mutation_rate'])
            c2_genome = mutate(c2_genome, rate=self.config['mutation_rate'])
            for g in [c1_genome, c2_genome]:
                s = decode(g)
                if s is not None and len(new_pop) < len(self.population):
                    new_pop.append(s)

        self.population = new_pop
        self.generation += 1
        return stats

    def run(self, n_generations: int, patience: int = 20) -> List[Strategy]:
        """Loop completo de evolucion."""
        best_fitness = -999
        stagnation = 0

        for gen in range(n_generations):
            stats = self.step()

            if stats.best_fitness > best_fitness:
                best_fitness = stats.best_fitness
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                logger.info(f"Early stop at gen {gen} (patience={patience})")
                break

        return sorted(self.population, key=lambda s: s.fitness[0], reverse=True)
```

### 2.5 Adaptacion de backtest/runner.py

El runner actual acepta Pattern o PatternChromosome v2. Necesita aceptar
Strategy (phenotype v2 nuevo).

Cambio minimo: el runner necesita recibir:
- signal: Series booleana (ya calculada por vectorized_eval)
- direction: no aplica — la estrategia define si es long/short via sus condiciones
- tp/sl ATR multipliers

Opcion mas limpia: el runner recibe directamente el signal Series + exit params.
No necesita saber nada de la estrategia internamente.

```python
def run_backtest(df, signal, tp_atr_mult, sl_atr_mult, config):
    """
    Ejecuta backtest dado un signal booleano precalculado.
    Retorna dict con equity curve y metricas.
    """
```

Esto desacopla completamente la representacion de la estrategia del backtester.

### 2.6 backtest/sampling.py

Reemplaza simple_sampling.py. Dos funciones:

```python
def sample_evolution_windows(data, n_windows, window_months, exclude_after):
    """
    Samplea ventanas aleatorias del periodo de evolucion.
    Llamada CADA GENERACION con seed distinto → window rotation.
    exclude_after: fecha limite (2025-05-31) para no tocar OTS.
    """

def create_cpcv_folds(data, n_groups, purge_bars, embargo_bars):
    """
    Crea folds para CPCV (usado post-evolucion en Sprint 3).
    Retorna lista de (train_indices, test_indices) con purge/embargo.
    """
```

### 2.7 Tests Sprint 2

**test_engine.py:**
- Inicializacion produce poblacion del tamaño correcto
- step() incrementa generacion
- Fitness mejora o se mantiene (elitismo)
- Population size se mantiene constante entre generaciones
- Early stop funciona con patience

**test_fitness.py:**
- Constraints rechazan estrategias con <30 trades
- Constraints rechazan estrategias con drawdown >30%
- Parsimony penalty reduce fitness proporcional a n_nodes
- Fitness es tupla de floats reales (no normalizado a [0,1])

### Exit Criterion Sprint 2
50 generaciones con poblacion 200 en <30 minutos. Best fitness del Sortino
mejora monotonicamente (gracias a elitismo) al menos hasta generacion 20.

---

## Sprint 3: Anti-Overfitting Suite

### Objetivo
Pipeline de validacion estadistica que distingue alpha real de overfitting.

### 3.1 validation/cpcv.py

Implementacion de Combinatorial Purged Cross-Validation
(Bailey & Lopez de Prado, 2014).

```python
def cpcv_evaluate(strategy, data, n_groups=10, purge_bars=96, embargo_bars=48):
    """
    Evalua estrategia con CPCV.

    1. Dividir datos en N grupos contiguos
    2. Generar C(N, N//2) combinaciones de train/test
    3. Para cada combinacion:
       a. Purgar `purge_bars` entre train y test (eliminar lookahead)
       b. Aplicar embargo de `embargo_bars` despues de cada bloque de train
       c. Evaluar en test
    4. Retornar distribucion de performance OOS

    purge_bars = 96 (24h en 15m bars)
    embargo_bars = 48 (12h en 15m bars)
    """
```

Con N=10 y k=5, genera C(10,5)=252 combinaciones train/test.
Cada una produce un OOS Sharpe/Sortino. La distribucion resultante
permite calcular PBO.

### 3.2 validation/pbo.py

Probability of Backtest Overfitting.

```python
def calculate_pbo(cpcv_results):
    """
    Calcula PBO a partir de los resultados de CPCV.

    PBO = proporcion de combinaciones donde la mejor estrategia IS
    tiene peor performance OOS que la mediana.

    PBO < 0.50 → no hay evidencia fuerte de overfitting
    PBO < 0.30 → robustez solida
    """
```

### 3.3 validation/deflated_sharpe.py

Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

```python
def deflated_sharpe_ratio(observed_sharpe, n_trials, T, skew, kurtosis,
                          sharpe_benchmark=0):
    """
    Corrige Sharpe ratio por multiple testing.

    Args:
        observed_sharpe: Sharpe de la estrategia
        n_trials: numero TOTAL de estrategias evaluadas durante evolucion
        T: numero de observaciones (barras)
        skew: skewness de los retornos
        kurtosis: excess kurtosis de los retornos
        sharpe_benchmark: Sharpe del benchmark (default 0)

    Returns:
        DSR: probabilidad de que el Sharpe observado sea real
        DSR > 0.95 → el Sharpe es probablemente real
    """
```

Nota critica: `n_trials` debe contar TODAS las estrategias evaluadas
durante toda la evolucion (population * generations). Si evaluamos 200 * 100
= 20,000 estrategias, n_trials = 20,000. Esto penaliza fuertemente por
data snooping, que es exactamente lo que queremos.

### 3.4 validation/signal_permutation.py

Test de permutacion sobre señales.

```python
def signal_permutation_test(strategy, data, n_permutations=1000):
    """
    Testea si la señal tiene poder predictivo.

    1. Calcular metricas reales de la estrategia
    2. Repetir n_permutations veces:
       a. Shuffle el vector de señales (manteniendo misma cantidad de True)
       b. Correr backtest con señales shuffleadas
       c. Calcular metricas
    3. p-value = (# permutaciones con metrica >= real) / n_permutations

    DISTINTO de monte_carlo.py (que shufflea trades, no señales).
    Este test es mas riguroso porque testea la señal misma.
    """
```

### 3.5 Fixes a modulos existentes

**robustness/bootstrap.py:**
- Reemplazar `except: continue` por `except Exception as e: logger.warning(...)`
- Investigar valores corruptos (UPI=2.4M): probablemente overflow en calculo
  de UPI cuando hay drawdown ~0. Agregar sanity checks post-calculo.

**robustness/white_rc.py:**
- Cambiar de bootstrap simple a block bootstrap (block_size de config)
- Alinear con bootstrap.py que ya usa blocks

### 3.6 Tests Sprint 3

**test_cpcv.py:**
- Folds no se solapan
- Purge gap existe entre todo par train-test adyacente
- Embargo aplicado correctamente
- Numero de combinaciones = C(N, N//2)
- Resultados reproducibles con mismo seed

**test_signal_permutation.py:**
- Señal random tiene p-value ~0.50 (no significativo)
- Señal perfecta (siempre acierta) tiene p-value ~0.00

### Exit Criterion Sprint 3
Pipeline completo evalua estrategia con CPCV (10 groups, 252 combinaciones),
calcula DSR y PBO. Tomar una estrategia random y verificar que:
- DSR < 0.95 (una estrategia random no deberia pasar)
- PBO > 0.50 (deberia mostrar overfitting)

---

## Sprint 4: Quality-Diversity + Regimenes

### Objetivo
MAP-Elites para mantener diversidad de estrategias. Deteccion de regimenes
de mercado. Validacion cross-regime.

### 4.1 evolution/archive.py

MAP-Elites archive con 3 dimensiones:

```python
class MAPElitesArchive:
    """
    Grid 3D: frequency x complexity x regime

    Dimensiones:
    - Frequency: trades/mes → [low (0-5), medium (5-20), high (20+)]
    - Complexity: n_nodes → [1, 2, 3, 4, 5+]
    - Regime: donde la estrategia performa mejor → [bull, bear, sideways]

    Cada celda guarda la MEJOR estrategia para ese nicho.
    Total: 3 x 5 x 3 = 45 celdas.
    """

    def __init__(self):
        self.grid = {}  # (freq_bin, complexity_bin, regime_bin) → Strategy

    def try_add(self, strategy: Strategy) -> bool:
        """Agrega si la celda esta vacia o si supera al residente."""
        cell = self._get_cell(strategy)
        if cell not in self.grid or strategy.fitness > self.grid[cell].fitness:
            self.grid[cell] = strategy
            return True
        return False

    def sample_for_reproduction(self, n: int) -> List[Strategy]:
        """Samplea n estrategias del archive para reproduccion."""
        occupied = list(self.grid.values())
        return random.choices(occupied, k=n)
```

Integracion con engine.py:
- Cada generacion, las mejores estrategias se intentan agregar al archive
- Periodicamente (cada 5 gens), se samplean estrategias del archive y se
  inyectan en la poblacion como "immigrants de calidad"
- Esto reemplaza la inmigracion con fitness=-999 de v1

### 4.2 evolution/island.py

Modelo de islas simple:

```python
class IslandModel:
    """
    3 islas con politicas de seleccion distintas:
    - Isla 1: Tournament selection (explotacion)
    - Isla 2: Lexicase selection (diversidad multi-objetivo)
    - Isla 3: Random selection (exploracion pura)

    Migracion: cada M generaciones, los top K de cada isla migran a las otras.
    """
```

Cada isla tiene su propia sub-poblacion y evoluciona independientemente.
Migracion cada 10 generaciones, top 5 individuos migran.

### 4.3 data/regime_detector.py

Clasificacion simple de regimen de mercado:

```python
def detect_regime(df, window=100):
    """
    Clasifica cada barra en bull/bear/sideways.

    Metodo: SMA slope + volatility
    - Bull: SMA(100) slope > threshold AND vol < high_vol_threshold
    - Bear: SMA(100) slope < -threshold AND vol < high_vol_threshold
    - Sideways: abs(slope) < threshold OR vol > high_vol_threshold

    Retorna Series con valores 'bull', 'bear', 'sideways'.
    """
```

Usado para:
1. Dimension del MAP-Elites archive (en que regimen funciona mejor la estrategia)
2. Cross-regime validation (la estrategia debe funcionar en al menos 2 de 3 regimenes)

### 4.4 Cross-regime validation

```python
def validate_cross_regime(strategy, data, regime_labels):
    """
    Evalua estrategia en cada regimen por separado.
    Retorna dict: {'bull': metrics, 'bear': metrics, 'sideways': metrics}

    Criterio: pass si al menos 2 de 3 regimenes tienen Sortino > 0.
    Criterio ideal: pass si 3 de 3 tienen Sortino > 0.
    """
```

### 4.5 Tests Sprint 4

**test_archive.py:**
- try_add a celda vacia siempre retorna True
- try_add con fitness inferior retorna False
- try_add con fitness superior reemplaza residente
- sample_for_reproduction retorna estrategias del archive
- Despues de 1000 inserciones random, al menos 30% de celdas ocupadas

### Exit Criterion Sprint 4
Archive >50% de las 45 celdas ocupadas despues de 100 generaciones.
Al menos 3 nichos contienen estrategias con Sortino > 0 en evaluacion.

---

## Sprint 5: Integracion + Experimentos

### Objetivo
Pipeline end-to-end. CLI limpio. Experimentos reproducibles. Output para paper.

### 5.1 main.py (rewrite completo)

```python
"""CLI entry point para CriptoGA v2."""

def cmd_evolve(args):
    """Ejecuta evolucion + guarda mejores estrategias."""
    config = load_config(args.config)
    data = load_data(config, exclude_ots=True)  # Excluir OTS
    engine = EvolutionEngine(config, data)
    engine.initialize(pop_size=config['population'])
    results = engine.run(n_generations=config['generations'], patience=config['patience'])
    save_results(results, args.output)

def cmd_validate(args):
    """Valida estrategias con CPCV + DSR + PBO + SPA + WRC + permutation."""
    results = load_results(args.results)
    data = load_data(config, exclude_ots=True)  # Aun sin OTS
    for strategy in results:
        cpcv = cpcv_evaluate(strategy, data)
        pbo = calculate_pbo(cpcv)
        dsr = deflated_sharpe_ratio(...)
        spa = hansen_spa(...)
        wrc = white_rc(...)
        perm = signal_permutation_test(strategy, data)
    save_validation(...)

def cmd_ots(args):
    """Evaluacion final en Out-of-Time Sample. UNA VEZ."""
    # Solo estrategias que pasaron TODOS los tests de validacion
    validated = load_validated(args.validated)
    ots_data = load_data(config, ots_only=True)  # Solo 2025-06 a 2025-11
    for strategy in validated:
        ots_metrics = run_final_backtest(strategy, ots_data)
    save_ots_results(...)

def cmd_report(args):
    """Genera reporte para paper (tablas LaTeX, figuras, markdown)."""
    ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # evolve, validate, ots, report
    ...
```

### 5.2 Enforcement de OTS

El data split debe ser un hard assertion:

```python
OTS_START = '2025-06-01'

def load_data(config, exclude_ots=False, ots_only=False):
    df = loader.fetch_data(config)
    if exclude_ots:
        assert df.index.max() < pd.Timestamp(OTS_START), "OTS leak!"
        return df[df.index < OTS_START]
    if ots_only:
        return df[df.index >= OTS_START]
    return df
```

### 5.3 config.yaml v2

Simplificar la config actual (que tiene muchos parametros deprecados):

```yaml
data:
  exchange: binance
  symbol: BTC/USDT
  market_type: future
  timeframe: 15m
  start: "2023-01-01"
  end: "2025-11-21"
  ots_start: "2025-06-01"  # Sacred holdout

costs:
  fees_bps: 1.0
  slippage_bps: 1.0

evolution:
  population: 200
  generations_max: 100
  patience: 20
  genome_length: 50
  mutation_rate: 0.1      # Per-codon rate
  crossover_rate: 0.8
  elitism_pct: 0.05       # 5%
  seed: 42

  # Window rotation
  n_windows_per_gen: 10
  window_months: 1

  # MAP-Elites
  archive_enabled: true
  archive_injection_every: 5
  archive_injection_n: 5

  # Island model
  islands_enabled: true
  n_islands: 3
  migration_every: 10
  migration_top_k: 5

fitness:
  parsimony_coefficient: 0.01
  min_trades: 30
  max_drawdown: 0.30
  min_win_rate: 0.35

exits:
  atr_period: 14
  # TP/SL vienen del genoma (evolucionan con la estrategia)

validation:
  cpcv_groups: 10
  cpcv_purge_bars: 96     # 24h
  cpcv_embargo_bars: 48   # 12h
  dsr_threshold: 0.95
  pbo_threshold: 0.50
  spa_alpha: 0.05
  wrc_alpha: 0.05
  permutation_n: 1000
  bootstrap_n: 1000
  bootstrap_block_size: 20

output:
  results_dir: ./results
  reports_dir: ./reports
  logs_dir: ./logs
```

### 5.4 Reproducibilidad

```python
def set_global_seed(seed: int):
    """Fija TODOS los seeds para reproducibilidad total."""
    random.seed(seed)
    np.random.seed(seed)
    # No usamos torch/tf, asi que esto es suficiente
```

Cada experimento se guarda como:
```
results/
├── experiment_seed42_20260306/
│   ├── config.yaml          # Config usada
│   ├── evolution_log.json   # Stats por generacion
│   ├── archive.json         # MAP-Elites final
│   ├── top_strategies.json  # Mejores estrategias
│   ├── validation.json      # CPCV + DSR + PBO + SPA + WRC + perm
│   ├── ots_results.json     # Resultados holdout final
│   └── figures/             # Plots para paper
```

### 5.5 Tests Sprint 5

**test_integration.py:**
- Pipeline completo evolve → validate → report en mini-datos (1 semana, pop=10, gen=5)
- Config load + validation
- OTS assertion no permite filtrar datos post 2025-06-01 en modo evolve
- Mismo seed → mismos resultados

### Exit Criterion Sprint 5
`python main.py evolve --seed 42` produce resultados identicos en multiples ejecuciones.
Pipeline completo funciona end-to-end. Output incluye tablas LaTeX y figuras.

---

## Dependencias Entre Sprints

```
Sprint 1 (grammar + eval) ──→ Sprint 2 (engine) ──→ Sprint 3 (validation)
                                                  ──→ Sprint 4 (diversity)
                                                         │
Sprint 3 + Sprint 4 ──────────────────────────────→ Sprint 5 (integration)
```

Sprint 3 y 4 son parcialmente independientes. Se pueden paralelizar
si se trabaja con dos personas, pero para un solo desarrollador,
el orden secuencial 1→2→3→4→5 es el mas natural.

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|-------------|---------|------------|
| No existe alpha en BTC 15m | Alta | Medio | Paper documenta resultado negativo con metodologia rigurosa |
| Evaluacion vectorizada no es suficientemente rapida | Baja | Alto | numba JIT para hot loops, reducir poblacion si necesario |
| CPCV con 252 folds es muy lento | Media | Medio | Reducir a N=8 (C(8,4)=70 folds) si necesario |
| Gramatica produce estrategias demasiado simples | Media | Medio | Expandir gramatica iterativamente, monitorear distribucion de complejidad |
| MAP-Elites archive se llena rapido con mediocridad | Baja | Bajo | Solo aceptar estrategias con fitness > threshold minimo |
| Overfitting sobrevive todas las validaciones | Baja | Alto | OTS holdout es la ultima linea de defensa |

---

## Metricas de Exito Globales

El proyecto tiene exito si:

**Escenario A (alpha encontrado):**
Al menos 1 estrategia pasa TODOS los tests:
- DSR > 0.95
- PBO < 0.50
- Hansen SPA p < 0.05
- Signal permutation p < 0.05
- OOS Sortino > 0.5 en OTS holdout
- Funciona en al menos 2/3 regimenes

**Escenario B (no alpha, resultado negativo riguroso):**
- Pipeline funciona correctamente end-to-end
- Se evaluaron >10,000 estrategias con la metodologia
- Ninguna pasa la bateria completa de tests
- Resultado documentado con metricas y figuras
- Paper argumenta que BTC 15m es eficiente en el periodo 2023-2025

Ambos escenarios son resultados validos y publicables.
