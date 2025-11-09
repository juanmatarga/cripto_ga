"""Tests GA optimizado."""

import pytest
import pandas as pd
import numpy as np
from ga_patterns.grammar import PREDICATE_REGISTRY, get_available_predicates, close_price
from ga_patterns.chromosome import Pattern, PredicateNode, LogicalNode, validate_pattern, _count_predicates
from ga_patterns.generator import generate_random_pattern, _generate_random_predicate
from ga_patterns.fitness import evaluate_fitness_bidirectional

def test_predicate_registry():
    """Registry tiene predicados esperados."""
    assert 'close' in PREDICATE_REGISTRY
    assert 'open' in PREDICATE_REGISTRY
    assert 'high' in PREDICATE_REGISTRY
    assert 'low' in PREDICATE_REGISTRY
    assert 'volume' in PREDICATE_REGISTRY
    assert 'price_change_pct' in PREDICATE_REGISTRY
    assert 'body_pct' in PREDICATE_REGISTRY

def test_direct_predicates():
    """Predicados directos funcionan."""
    data = pd.DataFrame({
        'Close': [100, 105, 103],
        'Open': [99, 104, 102],
        'High': [106, 107, 105],
        'Low': [98, 103, 101],
        'Volume': [1000, 1100, 1050]
    })

    val = close_price(data, bar_offset=0)
    assert val == 103

def test_direct_comparison_node():
    """PredicateNode con comparación directa."""
    data = pd.DataFrame({
        'Close': [100, 105, 110],
        'Open': [99, 104, 108],
        'High': [106, 107, 112],
        'Low': [98, 103, 107],
        'Volume': [1000, 1100, 1200]
    })

    # C[0] > C[1] → 110 > 105 → True
    node = PredicateNode(
        predicate_name='close',
        operator='>',
        bar_offset=0,
        compare_with_bar=1
    )

    result = node.evaluate(data)
    assert result == True

def test_threshold_comparison_node():
    """PredicateNode con threshold."""
    data = pd.DataFrame({
        'Close': [100, 105, 110],
        'Open': [99, 104, 108],
        'High': [106, 107, 112],
        'Low': [98, 103, 107],
        'Volume': [1000, 1100, 1200]
    })

    # C[0] > 105 → 110 > 105 → True
    node = PredicateNode(
        predicate_name='close',
        operator='>',
        bar_offset=0,
        threshold=105.0
    )

    result = node.evaluate(data)
    assert result == True

def test_logical_node_and():
    """LogicalNode AND."""
    data = pd.DataFrame({
        'Close': [100, 105, 110],
        'Open': [99, 104, 108],
        'High': [106, 107, 112],
        'Low': [98, 103, 107],
        'Volume': [1000, 1100, 1200]
    })

    # C[0] > C[1] AND V[0] > V[1]
    # 110 > 105 AND 1200 > 1100 → True AND True → True
    node1 = PredicateNode('close', '>', bar_offset=0, compare_with_bar=1)
    node2 = PredicateNode('volume', '>', bar_offset=0, compare_with_bar=1)

    logical = LogicalNode('AND', [node1, node2])

    result = logical.evaluate(data)
    assert result == True

def test_available_predicates_by_generation():
    """Predicados disponibles cambian por generación."""
    # Gen 1-50: Solo directos
    preds_gen1 = get_available_predicates(1, allow_indicators=False)
    assert 'close' in preds_gen1
    assert 'price_change_pct' not in preds_gen1
    assert 'rsi' not in preds_gen1

    # Gen 51-100: Directos + ratios
    preds_gen60 = get_available_predicates(60, allow_indicators=False)
    assert 'close' in preds_gen60
    assert 'price_change_pct' in preds_gen60
    assert 'rsi' not in preds_gen60

    # Gen 101+: Todos (con allow_indicators=True)
    preds_gen110 = get_available_predicates(110, allow_indicators=True)
    assert 'close' in preds_gen110
    assert 'price_change_pct' in preds_gen110
    assert 'rsi' in preds_gen110

def test_adaptive_constraints(config_fixture):
    """Constraints adaptativos por generación."""
    # Gen 1: max 4 predicados
    pattern_gen1 = generate_random_pattern(1, config_fixture['ga'])
    assert _count_predicates(pattern_gen1.expression) <= 4

    # Gen 60: max 6
    pattern_gen60 = generate_random_pattern(60, config_fixture['ga'])
    assert _count_predicates(pattern_gen60.expression) <= 6

    # Gen 110: max 8
    pattern_gen110 = generate_random_pattern(110, config_fixture['ga'])
    assert _count_predicates(pattern_gen110.expression) <= 8

def test_pattern_validation(config_fixture):
    """Validación de patrones."""
    pattern = generate_random_pattern(1, config_fixture['ga'])

    # Window válido
    assert 2 <= pattern.window <= 8

    # Direction válido
    assert pattern.direction in ['LONG', 'SHORT']

    # Valida
    assert validate_pattern(pattern) == True

def test_bidirectional_metadata(config_fixture):
    """Pattern guarda fitness_long y fitness_short."""
    pattern = generate_random_pattern(1, config_fixture['ga'])

    assert hasattr(pattern, 'fitness_long')
    assert hasattr(pattern, 'fitness_short')
    assert pattern.fitness_long == -999.0  # Inicialmente
    assert pattern.fitness_short == -999.0

def test_bidirectional_evaluation(config_fixture):
    """Evaluación bidireccional funciona."""
    data = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'Open': np.random.randn(100).cumsum() + 99,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Volume': np.random.randint(1000, 2000, 100)
    })

    pattern = generate_random_pattern(1, config_fixture['ga'])

    fitness, direction = evaluate_fitness_bidirectional(pattern, data, config_fixture)

    # Debe retornar fitness válido
    assert isinstance(fitness, float)
    assert direction in ['LONG', 'SHORT']

    # Debe haber actualizado el patrón
    assert pattern.fitness_long != -999.0 or pattern.fitness_short != -999.0
    assert pattern.direction in ['LONG', 'SHORT']

def test_pattern_repr():
    """Representación de patrones."""
    # Comparación directa
    node_direct = PredicateNode('close', '>', bar_offset=0, compare_with_bar=1)
    repr_str = str(node_direct)
    assert 'close[0]' in repr_str
    assert 'close[1]' in repr_str

    # Comparación con threshold
    node_threshold = PredicateNode('price_change_pct', '>', bar_offset=0, threshold=0.02)
    repr_str2 = str(node_threshold)
    assert 'price_change_pct[0]' in repr_str2
    assert '0.020' in repr_str2
