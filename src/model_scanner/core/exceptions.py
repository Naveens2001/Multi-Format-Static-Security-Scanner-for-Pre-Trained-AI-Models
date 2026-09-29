"""
model_scanner.core.exceptions
==============================

Structured exception hierarchy for the static security scanner pipeline.

Design Reference
----------------
Section 28 of docs/implementation_design.md (authoritative).

Hierarchy
---------
ScannerException (base)
├── AcquisitionError          — URL fetch failure, HTTP timeout, quota exceeded
├── SecurityBoundaryViolationError — Attempt to invoke forbidden dynamic execution
├── FormatDetectionError      — Inconclusive or corrupted format identification
├── AnalysisError             — Generic analysis failure (base for format-specific)
│   ├── PickleAnalysisError   — Opcode stream parsing failure, framing corruption
│   ├── SafeTensorsAnalysisError — Header size violation, JSON token budget exceeded
│   └── GGUFAnalysisError     — Magic mismatch, unsupported version, KV overflow
├── ResourceLimitError        — DoS protection ceiling breached
└── ConfigurationError        — Invalid TOML/JSON schema, illegal threshold values

Notes
-----
- These are Python *exceptions* — not data transfer objects. They propagate
  through the pipeline and are caught by the Orchestrator, which converts
  them into ``ScanError`` data records for inclusion in ``ScanResult``.
- ``InputError`` is provided as an alias / subclass so that Phase 2's
  requirement (InputError) maps cleanly to the design hierarchy.
- ``ReportingError`` is provided for Phase 9 (Presentation Layer) errors.
- No exception is ever silently swallowed. If it is caught, it must be
  converted into a ``ScanError`` and attached to the result.

SECURITY GUARANTEE
------------------
This module NEVER executes model code, imports torch/safetensors,
or invokes pickle.loads() / eval() / exec().
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


class ScannerException(Exception):  # noqa: N818

    """Base class for all scanner-internal exceptions.

    All scanner errors inherit from this class, enabling callers to catch
    the complete set of scanner failures with a single ``except`` clause.

    Attributes
    ----------
    message : str
        Human-readable description of the failure.
    error_code : str
        Machine-readable category code (e.g. ``"ACQ_TIMEOUT"``).
    details : str | None
        Optional additional context (path, URL, byte offset, etc.).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SCANNER_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.error_code: str = error_code
        self.details: str | None = details

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={self.details!r})"
        )

    def __str__(self) -> str:
        if self.details:
            return f"[{self.error_code}] {self.message} | {self.details}"
        return f"[{self.error_code}] {self.message}"


# ---------------------------------------------------------------------------
# Acquisition errors
# ---------------------------------------------------------------------------


class AcquisitionError(ScannerException):
    """Raised when artifact acquisition fails.

    Covers: public URL fetch failures, HTTP timeouts, quota exceeded,
    SSRF-rejected URLs, missing local files, unreadable paths.

    Maps to ClamAV exit code 2 (operational error).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "ACQ_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


class InputError(AcquisitionError):
    """Raised when the scan target specification is invalid.

    Examples: empty path, invalid URL scheme, unsupported input_source_type.
    This is a specialisation of AcquisitionError because invalid input
    prevents acquisition from proceeding.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INPUT_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


# ---------------------------------------------------------------------------
# Security boundary violation
# ---------------------------------------------------------------------------


class SecurityBoundaryViolationError(ScannerException):
    """Raised if any code path attempts forbidden dynamic execution.

    This exception guards the zero-execution contract. It must be raised
    (not silently suppressed) if any internal code attempts to invoke
    ``pickle.loads()``, ``torch.load()``, ``eval()``, or ``exec()``.

    Raising this exception is a hard assertion failure — the scanner has
    encountered a code-path that violates the static-only guarantee.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SECURITY_BOUNDARY_VIOLATION",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


# ---------------------------------------------------------------------------
# Format detection errors
# ---------------------------------------------------------------------------


class FormatDetectionError(ScannerException):
    """Raised when format identification is inconclusive or corrupted.

    Occurs when neither magic bytes nor file extension can identify a
    supported model format, or when the file header is too short to read.

    Maps to ScanOutcome.UNSUPPORTED or ScanOutcome.ERROR depending on
    context.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "FORMAT_DETECTION_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


# ---------------------------------------------------------------------------
# Analysis errors — format-specific
# ---------------------------------------------------------------------------


class AnalysisError(ScannerException):
    """Base class for all format-specific analyzer failures.

    Covers any failure occurring during static binary inspection of an
    artifact after format detection. Subclasses provide format-specific
    context.

    Maps to ClamAV exit code 2. The orchestrator converts this to a
    ``ScanError`` DTO and sets ``scan_outcome = ERROR``.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "ANALYSIS_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


class PickleAnalysisError(AnalysisError):
    """Raised when Pickle / PyTorch opcode stream parsing fails.

    Examples: framing corruption, unsupported protocol version, malformed
    opcode byte, truncated argument, concatenated stream boundary error.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "PICKLE_ANALYSIS_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


class SafeTensorsAnalysisError(AnalysisError):
    """Raised when SafeTensors binary inspection fails.

    Examples: header length > 16 MB, JSON token budget exceeded, nesting
    depth exceeded, invalid tensor dtype, overlapping byte spans.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SAFETENSORS_ANALYSIS_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


class GGUFAnalysisError(AnalysisError):
    """Raised when GGUF/GGML binary inspection fails.

    Examples: magic byte mismatch, unsupported format version, KV entry
    count overflow, malformed length-prefixed string.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "GGUF_ANALYSIS_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


# ---------------------------------------------------------------------------
# Resource / DoS protection
# ---------------------------------------------------------------------------


class ResourceLimitError(ScannerException):
    """Raised when a configured resource ceiling is breached.

    Examples: SafeTensors header exceeds max_safetensors_header_mb,
    JSON token count exceeds max_json_tokens, GGUF KV count exceeds
    max_gguf_kv_count, download size exceeds quota.

    Maps to ClamAV exit code 2 (operational error).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "RESOURCE_LIMIT_ERROR",
        details: str | None = None,
        limit_name: str | None = None,
        limit_value: int | None = None,
        actual_value: int | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)
        self.limit_name: str | None = limit_name
        self.limit_value: int | None = limit_value
        self.actual_value: int | None = actual_value


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigurationError(ScannerException):
    """Raised when configuration parsing or validation fails.

    Examples: invalid TOML/JSON schema, illegal threshold values (negative
    download caps, oversized JSON token limits), unrecognised config keys.

    Maps to ClamAV exit code 2 (operational error).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "CONFIG_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


# ---------------------------------------------------------------------------
# Reporting / presentation errors
# ---------------------------------------------------------------------------


class ReportingError(ScannerException):
    """Raised when a presentation layer component fails to render output.

    Examples: JSON serialisation failure, PDF generation failure (when
    optional reportlab dependency is absent), dashboard render error.

    This does NOT affect ClamAV exit code mapping because reporting errors
    occur after the scan pipeline completes. Callers should degrade
    gracefully (e.g. fall back to plain text).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "REPORTING_ERROR",
        details: str | None = None,
    ) -> None:
        super().__init__(message=message, error_code=error_code, details=details)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "AcquisitionError",
    "AnalysisError",
    "ConfigurationError",
    "FormatDetectionError",
    "GGUFAnalysisError",
    "InputError",
    "PickleAnalysisError",
    "ReportingError",
    "ResourceLimitError",
    "SafeTensorsAnalysisError",
    "ScannerException",
    "SecurityBoundaryViolationError",
]
