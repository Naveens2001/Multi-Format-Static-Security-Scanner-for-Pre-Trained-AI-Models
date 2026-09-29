"""
model_scanner.analyzers.pytorch
===============================

Static PyTorch ZIP container inspector and embedded Pickle extractor.

Implements:
  - zip_reader.py — safe read-only ZIP central directory navigator and tolerant reader
  - analyzer.py   — PyTorchAnalyzer coordinating container unpacking and Pickle analysis
"""

from model_scanner.analyzers.pytorch.analyzer import PyTorchAnalyzer
from model_scanner.analyzers.pytorch.zip_reader import PyTorchZipReader, ZipMemberInfo

__all__ = [
    "PyTorchAnalyzer",
    "PyTorchZipReader",
    "ZipMemberInfo",
]
