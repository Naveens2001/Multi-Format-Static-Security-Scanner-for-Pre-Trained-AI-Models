"""
model_scanner.acquisition.base
===============================

Abstract base interface for all input acquisition providers.

Design Reference
----------------
Section 9 of docs/implementation_design.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from model_scanner.core.models import Artifact, ScanRequest


class InputAcquisitionProvider(ABC):
    """Abstract interface for model artifact acquisition.

    All concrete acquisition providers (local file, directory, public URL,
    Hugging Face) implement this interface. The Scan Orchestrator calls
    ``acquire()`` without knowing the specific origin strategy.

    Security Guarantees
    -------------------
    - Streaming SHA-256 fingerprinting during ingestion.
    - Zero dynamic code execution during acquisition.
    - Resource quota and timeout enforcement.
    - Artifacts returned are raw bytes only — never deserialized.
    """

    @abstractmethod
    def acquire(self, request: ScanRequest) -> Sequence[Artifact]:
        """Acquire one or more validated Artifacts under resource bounds.

        Parameters
        ----------
        request:
            The incoming scan request describing the target and configuration.

        Returns
        -------
        Sequence[Artifact]
            An ordered sequence of acquired, SHA-256-fingerprinted Artifacts
            ready for format detection and analysis.

        Raises
        ------
        model_scanner.core.exceptions.AcquisitionError
            If acquisition fails (timeout, quota, I/O error, SSRF rejection).
        model_scanner.core.exceptions.InputError
            If the request target is invalid (bad path, unsupported scheme).
        model_scanner.core.exceptions.ResourceLimitError
            If a resource ceiling is exceeded (file too large, etc.).
        """
        ...


__all__ = ["InputAcquisitionProvider"]
