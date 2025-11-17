"""
SPRINT 14: Evolution Analytics Package

Post-run analytics and presentation generation for GA evolution.
"""

from .evolution_analytics import EvolutionAnalyzer
from .generate_presentation import generate_html_presentation

__all__ = ['EvolutionAnalyzer', 'generate_html_presentation']
