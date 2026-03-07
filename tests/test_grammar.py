"""Tests for grammar/bnf.py"""

import re
import pytest
from grammar.bnf import GRAMMAR, START_SYMBOL, MAX_DEPTH, validate_grammar


def test_grammar_validates():
    """Grammar should pass its own validation (already runs at import time)."""
    validate_grammar()  # Should not raise


def test_start_symbol_exists():
    assert START_SYMBOL in GRAMMAR


def test_all_nonterminals_reachable():
    """Every non-terminal should be reachable from the start symbol."""
    reachable = set()
    to_visit = [START_SYMBOL]

    while to_visit:
        nt = to_visit.pop()
        if nt in reachable:
            continue
        reachable.add(nt)
        for prod in GRAMMAR.get(nt, []):
            for match in re.findall(r'<[^>]+>', prod):
                to_visit.append(match)

    all_nts = set(GRAMMAR.keys())
    unreachable = all_nts - reachable
    assert unreachable == set(), f"Unreachable non-terminals: {unreachable}"


def test_all_referenced_nonterminals_defined():
    """Every non-terminal referenced in a production must be defined."""
    all_nts = set(GRAMMAR.keys())
    for nt, productions in GRAMMAR.items():
        for prod in productions:
            for match in re.findall(r'<[^>]+>', prod):
                assert match in all_nts, f"{match} referenced in {nt} but not defined"


def test_every_rule_has_at_least_one_production():
    for nt, productions in GRAMMAR.items():
        assert len(productions) >= 1, f"{nt} has no productions"


def test_no_empty_productions():
    for nt, productions in GRAMMAR.items():
        for prod in productions:
            assert prod.strip() != "", f"{nt} has empty production"


def test_comparators_use_safe_tokens():
    """Comparators should use _GT_/_LT_ to avoid conflict with <> delimiters."""
    for prod in GRAMMAR["<comparator>"]:
        assert '<' not in prod and '>' not in prod, \
            f"Comparator '{prod}' uses raw < or >, should use _GT_/_LT_"


def test_derivation_terminates():
    """Every non-terminal should have at least one production that
    eventually leads to all terminals (no infinite recursion)."""
    # Find which non-terminals can terminate in one step (all-terminal production)
    can_terminate = set()
    changed = True
    while changed:
        changed = False
        for nt, productions in GRAMMAR.items():
            if nt in can_terminate:
                continue
            for prod in productions:
                refs = re.findall(r'<[^>]+>', prod)
                if all(r in can_terminate for r in refs):
                    can_terminate.add(nt)
                    changed = True
                    break

    non_terminating = set(GRAMMAR.keys()) - can_terminate
    assert non_terminating == set(), \
        f"Non-terminals that can't terminate: {non_terminating}"
