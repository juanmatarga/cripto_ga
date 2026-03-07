"""
Parameter ranges and genome utilities.
"""

import random
from typing import List

# Default genome length (number of codons)
GENOME_LENGTH = 50

# Codon range: 0-255 (8-bit integer)
CODON_MIN = 0
CODON_MAX = 255


def random_genome(length: int = GENOME_LENGTH) -> List[int]:
    """Generate a random genome (list of integer codons)."""
    return [random.randint(CODON_MIN, CODON_MAX) for _ in range(length)]
