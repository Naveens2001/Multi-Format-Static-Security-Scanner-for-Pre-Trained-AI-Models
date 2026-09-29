"""
model_scanner
=============

Static Security Scanner for Pre-Trained AI Models.

This package provides a unified, multi-technique static analysis engine
for detecting supply-chain security threats embedded in AI model artifacts
(Pickle, PyTorch archives, SafeTensors, GGUF) without executing any model
code or loading any tensor weights.

SECURITY GUARANTEE
------------------
- No pickle.loads()  — streams are disassembled, never deserialized.
- No torch.load()    — PyTorch ZIPs are traversed as containers only.
- No eval() / exec() — prohibited throughout the codebase.
- No tensor weight loading — weight data regions are never read.

Primary Interface: Web UI (`python -m model_scanner serve`)
Supporting CLI: `python -m model_scanner scan <path>`
Python API: `from model_scanner import scan, serve`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "2.0.0"
__author__ = "Static Security Scanner Project"
__license__ = "MIT"

__all__ = [
    "__author__",
    "__license__",
    "__version__",
    "export_json",
    "export_json_dict",
    "scan",
    "serve",
]


def __getattr__(name: str) -> object:
    """Resolve public API functions lazily to prevent circular import cycles."""
    if name in ("export_json", "export_json_dict", "scan", "serve"):
        import model_scanner.api as api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from model_scanner.api import export_json, export_json_dict, scan, serve
