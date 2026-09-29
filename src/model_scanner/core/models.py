"""
model_scanner.core.models
=========================

Strongly-typed, immutable data contracts for the static security scanner
pipeline. All models use Python standard library ``dataclasses`` and
``enum`` exclusively — zero third-party runtime dependencies.

SECURITY GUARANTEE
------------------
This module contains data structures only.
It NEVER imports or invokes:
  - torch / safetensors / transformers / tensorflow / onnx / h5py
  - pickle.loads() / torch.load() / eval() / exec()
  - Any model execution or deserialization primitive

Design Reference
----------------
Sections 4-7 of docs/implementation_design.md (authoritative).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

_RULE_ID_PATTERN: re.Pattern[str] = re.compile(
    r"^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1|MS1)\d{2}$"
)


class InputSourceType(StrEnum):
    """Origin category of a scan target.

    Used in ScanRequest and Artifact to record how the input was supplied.
    """

    LOCAL_FILE = "LOCAL_FILE"
    LOCAL_DIRECTORY = "LOCAL_DIRECTORY"
    PUBLIC_URL = "PUBLIC_URL"
    IN_MEMORY = "IN_MEMORY"


class ModelFormat(StrEnum):
    """Detected binary model format.

    Determined by the FormatDispatcher using 16-byte magic sniffing and
    secondary file-extension fallback. ``UNSUPPORTED`` indicates the file
    type is not handled by any registered analyzer.

    Formats with analyzers: RAW_PICKLE, PYTORCH_ZIP, SAFETENSORS, GGUF, GGML,
      ONNX (ONNXAnalyzer), HDF5/KERAS (KerasAnalyzer), TF_SAVEDMODEL (TensorFlowAnalyzer).
    Identified but unanalyzed (emit G002): NUMPY_NPY, NUMPY_NPZ, TFLITE, FLAX_MSGPACK.
    REPOSITORY: virtual format used for directory/repository-level findings.
    """

    RAW_PICKLE = "RAW_PICKLE"
    LEGACY_PYTORCH = "LEGACY_PYTORCH"
    PYTORCH_ZIP = "PYTORCH_ZIP"
    SAFETENSORS = "SAFETENSORS"
    GGUF = "GGUF"
    GGML = "GGML"
    UNSUPPORTED = "UNSUPPORTED"

    # Identified but unanalyzed formats (A5)
    ONNX = "ONNX"
    HDF5 = "HDF5"
    NUMPY_NPY = "NUMPY_NPY"
    NUMPY_NPZ = "NUMPY_NPZ"
    TFLITE = "TFLITE"
    FLAX_MSGPACK = "FLAX_MSGPACK"

    # Virtual format for repository/supply-chain findings (B1/B2/B3/B4)
    REPOSITORY = "REPOSITORY"

    # P2 new formats
    TF_SAVEDMODEL = "TF_SAVEDMODEL"
    KERAS = "KERAS"

    # MindSpore protobuf checkpoint format
    MINDSPORE_CKPT = "MINDSPORE_CKPT"

    # TF.js weight shard: raw float32 buffer referenced by model.json.
    # No ZIP or pickle magic; no analyzer is registered for this format.
    TFJS_WEIGHT_SHARD = "TFJS_WEIGHT_SHARD"


class ScanOutcome(StrEnum):
    """High-level outcome of scanning a single artifact.

    Scanner exit code semantics:
      COMPLETE     - exit 0 (if no findings) or exit 1 (if findings present)
      WARNING      - exit 1
      ERROR        - exit 2
      UNSUPPORTED  - exit 0 (informational; no analysis performed)
      INCONCLUSIVE - exit 2; artifact could not be confidently analyzed
                     (parse/format error); contributes no security risk points
    """

    COMPLETE = "COMPLETE"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNSUPPORTED = "UNSUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Severity(StrEnum):
    """Authoritative 5-level finding severity classification.

    Defined in project_scope.md Section 4.1 and enforced by the
    SeverityEngine (core/severity.py - Phase 8).
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Confidence(StrEnum):
    """Forensic confidence rating attached to every Finding.

    Reflects the certainty of the detection: HIGH means the match is
    deterministic and unambiguous; LOW means heuristic or incomplete
    symbolic resolution was required.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskBand(StrEnum):
    """Discrete risk classification bands derived from the 0-100 risk score.

    Score boundaries (implementation_design.md Section 21.3):
      CLEAN         - 0 (analysis completed, nothing found)
      UNANALYZED    - no analyzer ran; absence of findings proves nothing
      LOW_RISK      - 1-30
      MEDIUM_RISK   - 31-60
      HIGH_RISK     - 61-89
      CRITICAL_RISK - 90-100
    """

    CLEAN = "CLEAN"
    UNANALYZED = "UNANALYZED"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL_RISK = "CRITICAL_RISK"


class CoordinateSpace(StrEnum):
    """Byte-offset coordinate reference frame for Evidence.

    Distinguishes between offsets in the raw file, within an embedded
    archive member (PyTorch ZIP), and relative to a format header boundary
    (SafeTensors / GGUF).
    """

    ARTIFACT = "artifact"              # Raw byte offset from file start
    ARCHIVE_MEMBER = "archive_member"  # Decompressed offset within ZIP member
    HEADER = "header"                  # Offset relative to format header boundary
    GLOBAL = "global"                  # Cross-artifact coordinate space


# ---------------------------------------------------------------------------
# Configuration Model — Section 4.2
# ---------------------------------------------------------------------------


def _validate_scan_config(
    strict_mode: bool,
    max_safetensors_header_mb: int,
    max_json_tokens: int,
    max_json_depth: int,
    max_gguf_kv_count: int,
    max_gguf_value_bytes: int,
    max_download_repo_bytes: int,
    max_download_file_bytes: int,
    download_timeout_sec: int,
) -> None:
    """Validate ScanConfig field constraints defined in Section 4.2."""
    if max_safetensors_header_mb <= 0:
        raise ValueError(
            f"max_safetensors_header_mb must be positive, got {max_safetensors_header_mb}"
        )
    if max_safetensors_header_mb > 128:
        raise ValueError(
            f"max_safetensors_header_mb cannot exceed 128 MB, got {max_safetensors_header_mb}"
        )
    if max_json_tokens <= 0:
        raise ValueError(f"max_json_tokens must be positive, got {max_json_tokens}")
    if max_json_tokens > 1_000_000:
        raise ValueError(
            f"max_json_tokens cannot exceed 1,000,000, got {max_json_tokens}"
        )
    if max_json_depth <= 0:
        raise ValueError(f"max_json_depth must be positive, got {max_json_depth}")
    if max_gguf_kv_count <= 0:
        raise ValueError(f"max_gguf_kv_count must be positive, got {max_gguf_kv_count}")
    if max_gguf_value_bytes <= 0:
        raise ValueError(
            f"max_gguf_value_bytes must be positive, got {max_gguf_value_bytes}"
        )
    if max_download_repo_bytes <= 0:
        raise ValueError(
            f"max_download_repo_bytes must be positive, got {max_download_repo_bytes}"
        )
    if max_download_file_bytes <= 0:
        raise ValueError(
            f"max_download_file_bytes must be positive, got {max_download_file_bytes}"
        )
    if download_timeout_sec <= 0:
        raise ValueError(
            f"download_timeout_sec must be positive, got {download_timeout_sec}"
        )


@dataclass(frozen=True)
class ScanConfig:
    """Immutable runtime configuration controlling scanner behaviour.

    All fields have safe, conservative defaults. Consumers override via
    TOML/JSON config files or CLI flags (config/loader.py — Phase 3).

    Validation: implemented in __post_init__ per Section 4.2 rules.
    """

    strict_mode: bool = False
    max_safetensors_header_mb: int = 16
    max_json_tokens: int = 100_000
    max_json_depth: int = 128
    max_gguf_kv_count: int = 10_000
    max_gguf_value_bytes: int = 1_048_576          # 1 MB
    max_download_repo_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
    max_download_file_bytes: int = 1 * 1024 * 1024 * 1024  # 1 GB
    download_timeout_sec: int = 60
    allowlist_modules: tuple[str, ...] = ()
    blocklist_modules: tuple[str, ...] = ()
    suppressed_rule_ids: tuple[str, ...] = ()
    include_pattern: str | None = None
    exclude_pattern: str | None = None

    def __post_init__(self) -> None:
        _validate_scan_config(
            strict_mode=self.strict_mode,
            max_safetensors_header_mb=self.max_safetensors_header_mb,
            max_json_tokens=self.max_json_tokens,
            max_json_depth=self.max_json_depth,
            max_gguf_kv_count=self.max_gguf_kv_count,
            max_gguf_value_bytes=self.max_gguf_value_bytes,
            max_download_repo_bytes=self.max_download_repo_bytes,
            max_download_file_bytes=self.max_download_file_bytes,
            download_timeout_sec=self.download_timeout_sec,
        )


# ---------------------------------------------------------------------------
# ScanRequest — Section 4.1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanRequest:
    """Incoming scan request from CLI, Python API, or CI/CD workflow.

    ``target`` may be a filesystem path string, a Path object, raw bytes,
    or an in-memory BytesIO stream. The ``input_source_type`` field
    disambiguates the origin.

    Validation rules (Section 4.1):
      - target cannot be empty / None.
      - If input_source_type == PUBLIC_URL, target must be a str
        starting with http:// or https://.
    """

    target: str | Path | bytes | io.BytesIO
    input_source_type: InputSourceType
    config: ScanConfig | None = None
    strict: bool = False
    filename_hint: str | None = None

    def __post_init__(self) -> None:
        # target cannot be empty
        if self.target is None:
            raise ValueError("ScanRequest.target cannot be None")
        if isinstance(self.target, (str, bytes)) and len(self.target) == 0:
            raise ValueError("ScanRequest.target cannot be empty")
        # PUBLIC_URL requires a string URL with http/https scheme
        if self.input_source_type == InputSourceType.PUBLIC_URL:
            if not isinstance(self.target, str):
                raise ValueError(
                    "ScanRequest.target must be a str URL when "
                    "input_source_type is PUBLIC_URL"
                )
            if not (
                self.target.startswith("http://")
                or self.target.startswith("https://")
            ):
                raise ValueError(
                    f"ScanRequest.target URL must start with http:// or https://, "
                    f"got: {self.target!r}"
                )


# ---------------------------------------------------------------------------
# Artifact — Section 4.3
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """Concrete acquired binary artifact ready for static inspection.

    Produced by the Acquisition Layer (Phase 3). Consumed by the Format
    Detector and Format Analyzers.

    Validation rules (Section 4.3):
      - sha256 must be exactly 64 lowercase hex characters.
      - size_bytes >= 0.
      - Exactly one of local_path or stream must be set.
    """

    artifact_id: str        # UUIDv4
    source_path: str        # Absolute path, URL, or in-memory label
    input_source: InputSourceType
    size_bytes: int
    sha256: str             # 64-character lowercase hex digest

    # Mutually exclusive: exactly one must be present
    local_path: Path | None = None
    stream: io.BytesIO | None = None

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(
                f"Artifact.size_bytes must be >= 0, got {self.size_bytes}"
            )
        if len(self.sha256) != 64 or not all(
            c in "0123456789abcdef" for c in self.sha256
        ):
            raise ValueError(
                f"Artifact.sha256 must be exactly 64 lowercase hex characters, "
                f"got {self.sha256!r}"
            )
        if self.local_path is None and self.stream is None:
            raise ValueError(
                "Artifact: exactly one of local_path or stream must be set; "
                "both are None"
            )
        if self.local_path is not None and self.stream is not None:
            raise ValueError(
                "Artifact: exactly one of local_path or stream must be set; "
                "both are set"
            )


# ---------------------------------------------------------------------------
# FormatDetectionResult — Section 4.4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormatDetectionResult:
    """Result of the two-layer magic-byte and extension format identification.

    Produced by FormatDispatcher (detection/detector.py — Phase 3).
    Consumed by the Scan Orchestrator (core/engine.py — Phase 2/3).
    """

    detected_format: ModelFormat
    magic_matched: bool
    extension_matched: bool
    format_mismatch: bool   # True → Rule G001 (extension spoofing) should fire
    details: str            # Human-readable identification rationale


# ---------------------------------------------------------------------------
# Evidence / Coordinate Model — Section 7
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """Unambiguous byte coordinates and raw context for forensic validation.

    Enables third-party reproduction and audit of every Finding. All
    coordinate fields reference a specific CoordinateSpace.

    Validation rules (Section 7):
      - If coordinate_space == ARCHIVE_MEMBER, archive_member must be set.
      - byte_offset must be >= 0 if present.
      - byte_length must be >= 0 if present.
    """

    coordinate_space: CoordinateSpace
    byte_offset: int | None          # Absolute/relative byte position
    byte_length: int | None          # Length of the triggering region
    archive_member: str | None       # e.g. "archive/data.pkl" for PyTorch ZIP
    matched_value: str                  # e.g. "posix.system", "eval", "AKIA…"

    # Optional forensic supplements
    opcode: str | None = None        # e.g. "GLOBAL", "STACK_GLOBAL"
    raw_hex: str | None = None       # Hex dump of triggering bytes
    context_snippet: str | None = None  # Surrounding text/bytes context

    def __post_init__(self) -> None:
        if (
            self.coordinate_space == CoordinateSpace.ARCHIVE_MEMBER
            and self.archive_member is None
        ):
            raise ValueError(
                "Evidence.archive_member must be set when "
                "coordinate_space is ARCHIVE_MEMBER"
            )
        if self.byte_offset is not None and self.byte_offset < 0:
            raise ValueError(
                f"Evidence.byte_offset must be >= 0, got {self.byte_offset}"
            )
        if self.byte_length is not None and self.byte_length < 0:
            raise ValueError(
                f"Evidence.byte_length must be >= 0, got {self.byte_length}"
            )


# ---------------------------------------------------------------------------
# RiskScore — Section 4.7
# ---------------------------------------------------------------------------


def _risk_band_for_score(score: int) -> RiskBand:
    """Return the RiskBand that corresponds to a 0-100 risk score.

    Boundaries per Section 21.3:
      0       - CLEAN
      1-30    - LOW_RISK
      31-60   - MEDIUM_RISK
      61-89   - HIGH_RISK
      90-100  - CRITICAL_RISK
    """
    if score == 0:
        return RiskBand.CLEAN
    if score <= 30:
        return RiskBand.LOW_RISK
    if score <= 60:
        return RiskBand.MEDIUM_RISK
    if score <= 89:
        return RiskBand.HIGH_RISK
    return RiskBand.CRITICAL_RISK


# Ordinal ranking of RiskBand values for severity-floor comparison (R5).
# UNANALYZED is treated as order 0 — it is a coverage state, not a risk level,
# and must never prevent a genuine score from dominating.
_RISK_BAND_ORDER: dict[RiskBand, int] = {
    RiskBand.CLEAN: 0,
    RiskBand.UNANALYZED: 0,
    RiskBand.LOW_RISK: 1,
    RiskBand.MEDIUM_RISK: 2,
    RiskBand.HIGH_RISK: 3,
    RiskBand.CRITICAL_RISK: 4,
}


@dataclass(frozen=True)
class RiskScore:
    """Composite 0-100 quantitative risk assessment for a single artifact.

    The scoring formula and calibration weights are defined in
    implementation_design.md Section 21 and implemented by the Risk Scoring
    Engine (core/scoring.py — Phase 8).

    Validation rules (Section 4.7):
      - 0 <= final_score <= 100.
      - risk_band must match the score boundaries in Section 21.3.
      - raw_score, format_multiplier, obfuscation_multiplier,
        findings_weighted_sum must all be >= 0.
    """

    raw_score: float                   # Pre-clamp composite score
    final_score: int                   # Clamped to [0, 100]
    risk_band: RiskBand
    format_multiplier: float           # Format-specific risk weight
    obfuscation_multiplier: float      # Structural obfuscation weight
    findings_weighted_sum: float       # Sum of (severity_weight x count)

    def __post_init__(self) -> None:
        if not (0 <= self.final_score <= 100):
            raise ValueError(
                f"RiskScore.final_score must be in [0, 100], got {self.final_score}"
            )
        if self.raw_score < 0:
            raise ValueError(
                f"RiskScore.raw_score must be >= 0, got {self.raw_score}"
            )
        if self.format_multiplier < 0:
            raise ValueError(
                f"RiskScore.format_multiplier must be >= 0, got {self.format_multiplier}"
            )
        if self.obfuscation_multiplier < 0:
            raise ValueError(
                f"RiskScore.obfuscation_multiplier must be >= 0, "
                f"got {self.obfuscation_multiplier}"
            )
        if self.findings_weighted_sum < 0:
            raise ValueError(
                f"RiskScore.findings_weighted_sum must be >= 0, "
                f"got {self.findings_weighted_sum}"
            )
        # UNANALYZED is a coverage state, not a score band: no analyzer ran, so
        # the score is meaningless and must not be validated against the numeric
        # boundaries. Every other band must match or exceed its score's band.
        if self.risk_band == RiskBand.UNANALYZED:
            return
        expected_band = _risk_band_for_score(self.final_score)
        # R5: severity floor may raise the band above what the numeric score
        # alone implies (e.g. a single HIGH finding scores 20 → LOW_RISK by
        # arithmetic, but must report HIGH_RISK). Allow elevation; only reject
        # if the band is *below* the score-derived band.
        if _RISK_BAND_ORDER[self.risk_band] < _RISK_BAND_ORDER[expected_band]:
            raise ValueError(
                f"RiskScore.risk_band {self.risk_band!r} does not match the "
                f"expected band {expected_band!r} for final_score={self.final_score}"
            )

    @staticmethod
    def clean() -> RiskScore:
        """Return the canonical zero-risk score for clean artifacts."""
        return RiskScore(
            raw_score=0.0,
            final_score=0,
            risk_band=RiskBand.CLEAN,
            format_multiplier=1.0,
            obfuscation_multiplier=1.0,
            findings_weighted_sum=0.0,
        )

    @staticmethod
    def unanalyzed() -> RiskScore:
        """Return the score for an artifact no analyzer inspected.

        Distinct from :meth:`clean`. A CLEAN score asserts that analysis ran and
        found nothing; UNANALYZED asserts only that nothing was inspected, so
        the absence of findings carries no security meaning.
        """
        return RiskScore(
            raw_score=0.0,
            final_score=0,
            risk_band=RiskBand.UNANALYZED,
            format_multiplier=1.0,
            obfuscation_multiplier=1.0,
            findings_weighted_sum=0.0,
        )


# ---------------------------------------------------------------------------
# Finding — Section 6
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """Strongly-typed, verifiable security finding.

    Emitted when a security rule matches an extracted token or structural
    anomaly. Every Finding carries forensic Evidence to enable independent
    reproduction.

    Validation rules (Section 6):
      - rule_id must match ^(P1|P2|ST1|ST2|GU1|GU2|G0)\\d{2}$.
      - remediation must not be empty.
      - artifact_sha256 must be 64 lowercase hex characters.
    """

    finding_id: str                    # UUIDv4 or formatted ID e.g. FINDING-P101-001
    rule_id: str                       # e.g. "P101", "ST101", "GU101", "G001"
    rule_name: str
    severity: Severity
    confidence: Confidence
    category: str                      # code_execution | structural_anomaly |
                                       # credential_exposure | tampering | format_integrity
    format: ModelFormat
    artifact_sha256: str               # 64-character lowercase hex
    evidence: Evidence
    description: str
    remediation: str
    cve_ids: tuple[str, ...] = ()
    cwe_ids: tuple[str, ...] = ()
    owasp_category: str | None = None
    mitre_technique: str | None = None
    correlated_finding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _RULE_ID_PATTERN.match(self.rule_id):
            raise ValueError(
                f"Finding.rule_id {self.rule_id!r} does not match the required "
                r"pattern ^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1)\d{2}$"
            )
        if not self.remediation:
            raise ValueError("Finding.remediation must not be empty")
        if len(self.artifact_sha256) != 64 or not all(
            c in "0123456789abcdef" for c in self.artifact_sha256
        ):
            raise ValueError(
                f"Finding.artifact_sha256 must be exactly 64 lowercase hex "
                f"characters, got {self.artifact_sha256!r}"
            )


# ---------------------------------------------------------------------------
# ScanWarning & ScanError (data records) — Section 4.8
#
# These are DATA TRANSFER OBJECTS recorded inside ScanResult. They are
# distinct from the exception hierarchy in exceptions.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanWarning:
    """Non-fatal operational notice recorded in ScanResult.warnings.

    Examples: format ambiguity, unsupported archive member, non-critical
    acquisition degradation.
    """

    warning_code: str
    message: str
    context: str | None = None

    def __post_init__(self) -> None:
        if not self.warning_code:
            raise ValueError("ScanWarning.warning_code must not be empty")
        if not self.message:
            raise ValueError("ScanWarning.message must not be empty")


@dataclass(frozen=True)
class ScanError:
    """Fatal operational error record stored in ScanResult.errors.

    Maps to ClamAV exit code 2. Distinct from Python exceptions — this is
    a serialisable data object emitted into the final ScanResult for
    presentation layer consumption.
    """

    error_code: str
    message: str
    exception_type: str               # e.g. "PickleAnalysisError"
    traceback_summary: str | None = None

    def __post_init__(self) -> None:
        if not self.error_code:
            raise ValueError("ScanError.error_code must not be empty")
        if not self.message:
            raise ValueError("ScanError.message must not be empty")
        if not self.exception_type:
            raise ValueError("ScanError.exception_type must not be empty")


# ---------------------------------------------------------------------------
# ScanResult hierarchy — Section 5
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactScanResult:
    """Per-artifact scan result — the canonical output for one model file.

    Produced by the Scan Orchestrator after the complete pipeline runs
    for a single artifact. Consumed by all presentation layer components.
    """

    artifact_id: str
    name: str                          # Human-readable filename / label
    source_origin: str                 # Path, URL, or memory label
    input_source: InputSourceType
    detected_format: ModelFormat
    size_bytes: int
    sha256: str                        # 64-character lowercase hex
    scan_outcome: ScanOutcome
    risk_score: RiskScore
    findings: tuple[Finding, ...]
    warnings: tuple[ScanWarning, ...]
    errors: tuple[ScanError, ...]
    duration_ms: int                   # Wall-clock scan duration in milliseconds
    model_structure: dict[str, object] | None = None  # Static tensor inventory (informational)

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(
                f"ArtifactScanResult.size_bytes must be >= 0, got {self.size_bytes}"
            )
        if self.duration_ms < 0:
            raise ValueError(
                f"ArtifactScanResult.duration_ms must be >= 0, got {self.duration_ms}"
            )


@dataclass(frozen=True)
class ScanSummary:
    """Aggregate summary across all artifacts in a single scan run.

    Consumed by the Dashboard, JSON Exporter, and CLI exit dispatcher.

    exit_code semantics (Section 5, Section 26.2):
      0 → all artifacts clean, no operational errors
      1 → one or more security findings, no operational errors
      2 → one or more operational errors (acquisition/parsing failure)
    """

    total_artifacts: int
    scanned_successfully: int
    failed_artifacts: int
    unsupported_artifacts: int
    findings_by_severity: dict[str, int]   # Keys: CRITICAL HIGH MEDIUM LOW INFO
    artifacts_by_risk_band: dict[str, int]
    max_risk_score: int
    overall_outcome: ScanOutcome
    exit_code: int                         # 0, 1, or 2
    coverage_gap_artifacts: int = 0        # INCONCLUSIVE from unsupported format (no analyzer)

    def __post_init__(self) -> None:
        if self.exit_code not in (0, 1, 2):
            raise ValueError(
                f"ScanSummary.exit_code must be 0, 1, or 2, "
                f"got {self.exit_code}"
            )
        if self.max_risk_score < 0 or self.max_risk_score > 100:
            raise ValueError(
                f"ScanSummary.max_risk_score must be in [0, 100], "
                f"got {self.max_risk_score}"
            )
        if self.total_artifacts < 0:
            raise ValueError(
                f"ScanSummary.total_artifacts must be >= 0, got {self.total_artifacts}"
            )


@dataclass(frozen=True)
class ScanResult:
    """Canonical, single source of truth produced by the scanning pipeline.

    All presentation formats — Dashboard, JSON exporter, PDF exporter —
    consume this object directly. Produced exclusively by the Scan
    Orchestrator (core/engine.py — Phase 3).

    scan_id   : UUIDv4 uniquely identifying this scan run.
    timestamp_utc : ISO-8601 UTC timestamp of scan initiation.
    acquisition_warnings : scan-level warnings from the acquisition layer
        (e.g. blocked symlink escapes, A4). These are not artifact-level.
    """

    scan_id: str                           # UUIDv4
    scanner_version: str
    timestamp_utc: str                     # ISO-8601 UTC
    target_request: str                    # Original user-supplied target string
    strict_mode: bool
    artifacts: tuple[ArtifactScanResult, ...]
    summary: ScanSummary
    total_duration_ms: int
    acquisition_warnings: tuple[ScanWarning, ...] = ()

    def __post_init__(self) -> None:
        if self.total_duration_ms < 0:
            raise ValueError(
                f"ScanResult.total_duration_ms must be >= 0, "
                f"got {self.total_duration_ms}"
            )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Artifact",
    "ArtifactScanResult",
    "Confidence",
    "CoordinateSpace",
    "Evidence",
    "Finding",
    "FormatDetectionResult",
    "InputSourceType",
    "ModelFormat",
    "RiskBand",
    "RiskScore",
    "ScanConfig",
    "ScanError",
    "ScanOutcome",
    "ScanRequest",
    "ScanResult",
    "ScanSummary",
    "ScanWarning",
    "Severity",
    "risk_band_for_score",
]


def risk_band_for_score(score: int) -> RiskBand:
    """Public accessor for the score-to-band mapping (Section 21.3)."""
    return _risk_band_for_score(score)
