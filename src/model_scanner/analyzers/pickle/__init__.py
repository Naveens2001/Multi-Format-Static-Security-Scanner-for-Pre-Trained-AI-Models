"""
model_scanner.analyzers.pickle
==============================

Static Pickle disassembly, memo tracking, and structural anomaly analysis.

Implements:
  - disassembler.py — linear opcode disassembler via pickletools.genops()
  - memo.py         — bounded 1-level memo slot tracker
  - structural.py   — PROTO, FRAME, and multi-stream anomaly validator
  - catalog.py      — dangerous and suspicious global symbol catalog
  - opcodes.py      — normalized opcode data structures
  - analyzer.py     — PickleAnalyzer implementation of BaseFormatAnalyzer
"""

from model_scanner.analyzers.pickle.analyzer import PickleAnalyzer
from model_scanner.analyzers.pickle.catalog import (
    GlobalCategory,
    classify_global,
)
from model_scanner.analyzers.pickle.disassembler import PickleDisassembler
from model_scanner.analyzers.pickle.memo import MemoTracker
from model_scanner.analyzers.pickle.opcodes import (
    BUILDRecord,
    GLOBALRecord,
    NEWOBJRecord,
    OpcodeRecord,
    REDUCERecord,
)
from model_scanner.analyzers.pickle.structural import StructuralAnomalyEngine

__all__ = [
    "BUILDRecord",
    "GLOBALRecord",
    "GlobalCategory",
    "MemoTracker",
    "NEWOBJRecord",
    "OpcodeRecord",
    "PickleAnalyzer",
    "PickleDisassembler",
    "REDUCERecord",
    "StructuralAnomalyEngine",
    "classify_global",
]
