"""
model_scanner.analyzers.repository
===================================

Supply chain and repository-level static analysis for Hugging Face-style
model repositories.

Work packages covered:
  A1  — Framework identification (detection/framework.py feeds this)
  A2  — Sharded model integrity (shard_checker.py)
  A3  — Repository metadata reading (metadata_reader.py)
  B1  — Remote code execution vectors in configs (config_scanner.py)
  B2  — Provenance indicators offline (provenance.py)
  B3  — Dependency vulnerability scanning offline (dep_scanner.py)
  B4  — RepositoryAnalyzer orchestrator (analyzer.py)

SECURITY GUARANTEE
------------------
This package NEVER:
  - Imports, instantiates, or executes model weights.
  - Calls eval(), exec(), pickle.loads(), torch.load().
  - Makes network calls (NFR-01: offline at scan time).
  - Imports YAML, PyYAML, ruamel, or any non-stdlib package.
"""

from model_scanner.analyzers.repository.analyzer import RepositoryAnalyzer

__all__ = ["RepositoryAnalyzer"]
