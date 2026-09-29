"""
model_scanner.analyzers.gguf
============================

Static binary security analyzer for GGUF and GGML model files.

SECURITY GUARANTEE
------------------
- Pure standard library struct/byte parsing only.
- Never imports llama.cpp, transformers, or external runtime libraries.
- Never calls eval(), exec(), pickle.loads(), or torch.load().
- Never loads tensor payloads into memory.
"""

from model_scanner.analyzers.gguf.analyzer import GGUFAnalyzer
from model_scanner.analyzers.gguf.reader import GGUFHeader, GGUFKVEntry, GGUFReader, GGUFValueType
from model_scanner.analyzers.gguf.tensors import GGMLType, GGUFTensorDescriptor

__all__ = [
    "GGMLType",
    "GGUFAnalyzer",
    "GGUFHeader",
    "GGUFKVEntry",
    "GGUFReader",
    "GGUFTensorDescriptor",
    "GGUFValueType",
]
