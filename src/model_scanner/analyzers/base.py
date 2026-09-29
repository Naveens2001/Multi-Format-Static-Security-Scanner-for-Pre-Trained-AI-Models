"""
model_scanner.analyzers.base
============================

Abstract base class for all static format analyzers.

Design Reference
----------------
Section 38 of docs/implementation_design.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from model_scanner.core.models import Artifact, Finding, ScanConfig


class BaseFormatAnalyzer(ABC):
    """Abstract base class for format-specific static security analyzers.

    All concrete analyzers (PickleAnalyzer, PyTorchAnalyzer, etc.) implement
    this interface.

    Security Guarantees
    -------------------
    - Pure static inspection of raw bytes and structural metadata.
    - Never imports, instantiates, or executes model weights or modules.
    - Produces strongly-typed Finding objects with precise byte coordinates.
    """

    @abstractmethod
    def analyze(
        self, artifact: Artifact, config: ScanConfig | None = None
    ) -> Sequence[Finding]:
        """Perform static analysis on the acquired Artifact.

        Parameters
        ----------
        artifact:
            The raw, SHA-256-fingerprinted Artifact to inspect.
        config:
            Optional scan configuration with limits and rule toggles.

        Returns
        -------
        Sequence[Finding]
            Collection of security findings with byte-accurate evidence.
        """
        ...


__all__ = ["BaseFormatAnalyzer"]
