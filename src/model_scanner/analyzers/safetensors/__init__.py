"""
model_scanner.analyzers.safetensors
=====================================

Pure-stdlib static security analyzer for SafeTensors model files.

SECURITY GUARANTEE
------------------
This package NEVER:
  - Loads tensor weights or deserializes tensor data
  - Imports safetensors, torch, tensorflow, or any ML framework
  - Calls pickle.loads(), torch.load(), eval(), or exec()
  - Allocates memory proportional to attacker-controlled tensor dimensions

All analysis is purely structural: 8-byte header unpacking, JSON parsing
of the bounded header section, and byte-range arithmetic.

Design Reference
----------------
Section 16 of docs/implementation_design.md
"""

from model_scanner.analyzers.safetensors.analyzer import SafeTensorsAnalyzer

__all__ = ["SafeTensorsAnalyzer"]
