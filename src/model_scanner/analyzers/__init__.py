"""
model_scanner.analyzers
=======================

Static security analyzers for AI model formats.

Subpackages:
  - pickle  — Raw Pickle stream disassembler, memo tracker, and anomaly engine
  - pytorch — PyTorch ZIP archive inspector and embedded Pickle extractor

SECURITY GUARANTEE
------------------
All analyzers operate strictly via static disassembly and binary parsing.
Zero dynamic code execution, model instantiation, or deserialization.
"""

from model_scanner.analyzers.base import BaseFormatAnalyzer

__all__ = ["BaseFormatAnalyzer"]
