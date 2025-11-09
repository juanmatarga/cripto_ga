"""
Genetic Operators - Advanced mutation with term substitution
"""

import random
import numpy as np
from typing import List
import copy
import logging

from ga_patterns.grammar import (
    get_available_predicates, PREDICATE_REGISTRY,
    get_common_offsets, get_random_offset
)
from ga_patterns.chromosome import Pattern, PredicateNode, LogicalNode, validate_pattern

logger = logging.getLogger(__name__)

def generate_random_pattern(generation: int, config: dict) -> Pattern:
    """Genera patrón random con profundidad controlada."""
    direction = random.choice(['LONG', 'SHORT'])
    window = random.randint(config['window_min'], config['window_max'])

    # Profundidad inicial: 70% simple, 30% con 1 nivel lógico
    if random.random() < 0.7:
        max_depth = 0  # Solo hoja
    else:
        max_depth = 1  # 1 nivel lógico

    available_predicates = get_available_predicates(generation, config.get('allow_indicators', False))

    expression = _generate_random_tree(available_predicates, depth=0, max_depth=max_depth)

    pattern = Pattern(
        direction=direction,
        window=window,
        expression=expression,
        generation_created=generation
    )

    if not validate_pattern(pattern):
        # Fallback a predicado simple
        expression = _generate_random_predicate(available_predicates)
        pattern.expression = expression

        if not validate_pattern(pattern):
            raise ValueError("Could not generate valid pattern")

    return pattern

def _generate_random_tree(available_predicates: List[str], depth: int, max_depth: int):
    """Genera árbol random."""
    if depth == max_depth:
        return _generate_random_predicate(available_predicates)

    # 30% terminar temprano
    if random.random() < 0.3:
        return _generate_random_predicate(available_predicates)

    operator = random.choice(['AND', 'OR'])
    num_children = random.randint(2, 3)

    children = [
        _generate_random_tree(available_predicates, depth + 1, max_depth)
        for _ in range(num_children)
    ]

    return LogicalNode(operator=operator, children=children)

def _generate_random_predicate(available_predicates: List[str]) -> PredicateNode:
    """
    Genera PredicateNode con offsets mixtos.

    Para predicados 'direct' (C, O, H, L, V):
    - 70% comparación con otra barra (C[0] > C[1])
    - 30% comparación con threshold

    Para predicados 'ratio':
    - 100% comparación con threshold
    """
    predicate_name = random.choice(available_predicates)
    predicate_info = PREDICATE_REGISTRY[predicate_name]

    operator = random.choice(['>', '>=', '<='])
    bar_offset = random.randint(0, 5)

    # Determinar tipo de comparación
    if predicate_info['allows_comparison'] and random.random() < 0.7:
        # Comparación con otra barra (offsets mixtos)
        if random.random() < 0.7:
            # 70%: Offset común
            compare_offset = bar_offset + random.choice(get_common_offsets())
        else:
            # 30%: Offset aleatorio
            compare_offset = bar_offset + get_random_offset()

        return PredicateNode(
            predicate_name=predicate_name,
            operator=operator,
            threshold=None,
            bar_offset=bar_offset,
            compare_with_bar=compare_offset
        )
    else:
        # Comparación con threshold
        threshold_min, threshold_max = predicate_info['threshold_range']
        threshold = random.uniform(threshold_min, threshold_max)

        return PredicateNode(
            predicate_name=predicate_name,
            operator=operator,
            threshold=threshold,
            bar_offset=bar_offset,
            compare_with_bar=None
        )

def initialize_population(population_size: int, generation: int, config: dict) -> List[Pattern]:
    """Genera población inicial."""
    logger.info(f"Initializing population of {population_size} patterns...")
    population = []

    for i in range(population_size):
        try:
            pattern = generate_random_pattern(generation, config)
            population.append(pattern)

            if (i + 1) % 20 == 0:
                logger.info(f"  Generated {i+1}/{population_size}")
        except Exception as e:
            logger.error(f"Failed to generate pattern {i}: {e}")
            continue

    logger.info(f"[OK] Population initialized: {len(population)}")
    return population

def tournament_selection(population: List[Pattern], tournament_size: int = 3) -> Pattern:
    """Selección por torneo."""
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda p: p.fitness)
    return winner

def subtree_crossover(parent1: Pattern, parent2: Pattern,
                     generation: int, config: dict) -> Pattern:
    """Crossover de subárboles."""
    offspring = copy.deepcopy(parent1)

    subtree_p2 = _select_random_subtree(copy.deepcopy(parent2.expression))
    offspring.expression = _replace_random_subtree(offspring.expression, subtree_p2)

    offspring.direction = random.choice([parent1.direction, parent2.direction])
    offspring.window = random.choice([parent1.window, parent2.window])
    offspring.generation_created = generation
    offspring.fitness = -999.0
    offspring.fitness_long = -999.0
    offspring.fitness_short = -999.0

    if not validate_pattern(offspring):
        return copy.deepcopy(parent1)

    return offspring

def _select_random_subtree(node):
    """Selecciona subtree random."""
    all_nodes = []

    def collect_nodes(n):
        all_nodes.append(n)
        if isinstance(n, LogicalNode):
            for child in n.children:
                collect_nodes(child)

    collect_nodes(node)
    selected = random.choice(all_nodes)
    return copy.deepcopy(selected)

def _replace_random_subtree(tree, new_subtree):
    """Reemplaza subtree random."""
    if isinstance(tree, PredicateNode):
        return new_subtree

    if random.random() < 0.3:
        return new_subtree

    if tree.children:
        child_idx = random.randint(0, len(tree.children) - 1)
        tree.children[child_idx] = _replace_random_subtree(tree.children[child_idx], new_subtree)

    return tree

def mutate_pattern(pattern: Pattern, generation: int, config: dict) -> Pattern:
    """
    Mutación con 4 tipos:
    1. Subtree (40%)
    2. Threshold (30%)
    3. Operator (20%)
    4. Simplify (10%) - NUEVO
    """
    mutated = copy.deepcopy(pattern)

    mutation_type = random.choices(
        ['subtree', 'threshold', 'operator', 'simplify'],
        weights=[0.4, 0.3, 0.2, 0.1]
    )[0]

    logger.debug(f"Applying {mutation_type} mutation")

    if mutation_type == 'subtree':
        # Reemplazar subtree con predicado random
        available_predicates = get_available_predicates(generation, config.get('allow_indicators', False))
        new_subtree = _generate_random_predicate(available_predicates)
        mutated.expression = _replace_random_subtree(mutated.expression, new_subtree)

    elif mutation_type == 'threshold':
        # Ajustar threshold o cambiar offset
        predicate = _select_random_predicate_node(mutated.expression)
        if predicate:
            if predicate.compare_with_bar is not None:
                # Cambiar offset de comparación
                if random.random() < 0.5:
                    predicate.compare_with_bar += random.choice([-1, 1])
                    predicate.compare_with_bar = max(0, predicate.compare_with_bar)
                else:
                    predicate.bar_offset += random.choice([-1, 1])
                    predicate.bar_offset = max(0, predicate.bar_offset)
            else:
                # Ajustar threshold
                adjustment = random.uniform(0.8, 1.2)
                predicate.threshold *= adjustment

                pred_info = PREDICATE_REGISTRY.get(predicate.predicate_name)
                if pred_info:
                    min_t, max_t = pred_info['threshold_range']
                    predicate.threshold = max(min_t, min(max_t, predicate.threshold))

    elif mutation_type == 'operator':
        # Flip operator
        node = _select_random_node(mutated.expression)
        if isinstance(node, PredicateNode):
            flip_map = {'>': '<=', '>=': '<=', '<=': '>='}
            node.operator = flip_map.get(node.operator, node.operator)
        elif isinstance(node, LogicalNode):
            node.operator = 'OR' if node.operator == 'AND' else 'AND'

    elif mutation_type == 'simplify':
        # NUEVO: Simplify - reduce complejidad
        # Si tiene LogicalNode, reemplazar con uno de sus hijos
        if isinstance(mutated.expression, LogicalNode):
            if mutated.expression.children:
                mutated.expression = copy.deepcopy(random.choice(mutated.expression.children))
                logger.debug("Simplified: replaced LogicalNode with child")

    mutated.generation_created = generation
    mutated.fitness = -999.0
    mutated.fitness_long = -999.0
    mutated.fitness_short = -999.0

    if not validate_pattern(mutated):
        return copy.deepcopy(pattern)

    return mutated

def _select_random_predicate_node(node) -> PredicateNode:
    """Selecciona PredicateNode random."""
    predicates = []

    def collect_predicates(n):
        if isinstance(n, PredicateNode):
            predicates.append(n)
        elif isinstance(n, LogicalNode):
            for child in n.children:
                collect_predicates(child)

    collect_predicates(node)
    return random.choice(predicates) if predicates else None

def _select_random_node(node):
    """Selecciona nodo random."""
    all_nodes = []

    def collect_all(n):
        all_nodes.append(n)
        if isinstance(n, LogicalNode):
            for child in n.children:
                collect_all(child)

    collect_all(node)
    return random.choice(all_nodes) if all_nodes else None
