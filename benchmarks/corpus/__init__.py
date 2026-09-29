"""
benchmarks.corpus
=================

Evaluation dataset corpus and manifest generation.
"""

from benchmarks.corpus.generator import generate_corpus_data, write_corpus

__all__ = [
    "generate_corpus_data",
    "write_corpus",
]
