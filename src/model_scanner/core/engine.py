"""
model_scanner.core.engine
=========================

Unified scan orchestration engine — the central pipeline coordinator.

Pipeline
--------
1. ScanRequest → Input Acquisition Router → Artifact(s)
2. For each Artifact:
   a. Format Detection (FormatDispatcher)
   b. Analyzer Registry Lookup
   c. Format Analyzer Execution (PickleAnalyzer, SafeTensorsAnalyzer, etc.)
   d. Finding Collection
   e. ArtifactScanResult Assembly
3. ScanResult & Summary Aggregation

SECURITY GUARANTEE
------------------
This module NEVER:
  - Calls pickle.loads(), torch.load(), eval(), exec()
  - Imports torch, transformers, safetensors runtime, llama_cpp, ML libraries
  - Instantiates or executes model weights

The orchestrator remains STATIC ONLY — it coordinates pure inspection.

Design Reference
----------------
docs/phase7_design.md (authoritative)
Section 20 of docs/implementation_design.md
"""

from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from model_scanner import __version__
from model_scanner.acquisition.base import InputAcquisitionProvider
from model_scanner.acquisition.huggingface import HuggingFaceAcquisition, is_hf_url
from model_scanner.acquisition.local import (
    LocalDirectoryAcquisition,
    LocalFileAcquisition,
)
from model_scanner.acquisition.url import PublicUrlAcquisition
from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.analyzers.gguf.analyzer import GGUFAnalyzer
from model_scanner.analyzers.keras.analyzer import KerasAnalyzer
from model_scanner.analyzers.mindspore.analyzer import MindSporeAnalyzer
from model_scanner.analyzers.onnx.analyzer import ONNXAnalyzer
from model_scanner.analyzers.pickle.analyzer import PickleAnalyzer
from model_scanner.analyzers.pytorch.analyzer import PyTorchAnalyzer
from model_scanner.analyzers.safetensors.analyzer import SafeTensorsAnalyzer
from model_scanner.analyzers.tensorflow.analyzer import TensorFlowAnalyzer
from model_scanner.core.exceptions import ScannerException
from model_scanner.core.models import (
    Artifact,
    ArtifactScanResult,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    InputSourceType,
    ModelFormat,
    RiskScore,
    ScanConfig,
    ScanError,
    ScanOutcome,
    ScanRequest,
    ScanResult,
    ScanSummary,
    ScanWarning,
    Severity,
)
from model_scanner.core.normalizer import FindingCorrelator
from model_scanner.core.scoring import RiskEngine
from model_scanner.detection.detector import FormatDispatcher

# ---------------------------------------------------------------------------
# Analyzer Registry — Format → Analyzer Class Mapping
# ---------------------------------------------------------------------------

_ANALYZER_REGISTRY: dict[ModelFormat, type[BaseFormatAnalyzer]] = {
    ModelFormat.RAW_PICKLE: PickleAnalyzer,
    ModelFormat.LEGACY_PYTORCH: PickleAnalyzer,
    ModelFormat.PYTORCH_ZIP: PyTorchAnalyzer,
    ModelFormat.SAFETENSORS: SafeTensorsAnalyzer,
    ModelFormat.GGUF: GGUFAnalyzer,
    ModelFormat.GGML: GGUFAnalyzer,  # GGML uses the same analyzer as GGUF
    ModelFormat.ONNX: ONNXAnalyzer,
    ModelFormat.HDF5: KerasAnalyzer,
    ModelFormat.KERAS: KerasAnalyzer,
    ModelFormat.TF_SAVEDMODEL: TensorFlowAnalyzer,
    ModelFormat.MINDSPORE_CKPT: MindSporeAnalyzer,
}

# Formats that are identified by magic bytes but have no registered analyzer (A5).
# These emit G002 findings rather than producing UNSUPPORTED results silently.
_IDENTIFIED_UNANALYZED_FORMATS: frozenset[ModelFormat] = frozenset(
    [
        ModelFormat.NUMPY_NPY,
        ModelFormat.NUMPY_NPZ,
        ModelFormat.TFLITE,
        ModelFormat.FLAX_MSGPACK,
    ]
)

# Formats with no registered analyzer that produce INCONCLUSIVE (not WARNING) outcome.
# TFJS_WEIGHT_SHARD falls here: the .bin extension overlaps with PYTORCH_ZIP, so
# routing to PYTORCH_ZIP causes a P201 false-positive.  INCONCLUSIVE clearly signals
# the artifact was detected but not inspected, without inflating the security score.
_FORMAT_UNSUPPORTED_FORMATS: frozenset[ModelFormat] = frozenset(
    [
        ModelFormat.TFJS_WEIGHT_SHARD,
    ]
)

# Structural parse-failure rule IDs that signal the artifact could not be fully read.
# These are coverage/format errors, NOT security findings, and must not inflate scores.
_FATAL_FINDING_RULES: frozenset[str] = frozenset({
    "ST209",  # SafeTensors truncated/unreadable header
    "ST201",  # SafeTensors header size/token DoS (when early return)
    "GU201",  # GGUF header validation failure
    "P201",   # Pickle PROTO desynchronization / corrupt ZIP container
})

_FORMAT_DISPLAY_NAMES: dict[ModelFormat, str] = {
    ModelFormat.ONNX: "ONNX (Open Neural Network Exchange)",
    ModelFormat.HDF5: "HDF5/Keras (.h5)",
    ModelFormat.KERAS: "Keras (.keras ZIP)",
    ModelFormat.TF_SAVEDMODEL: "TensorFlow SavedModel",
    ModelFormat.NUMPY_NPY: "NumPy .npy array",
    ModelFormat.NUMPY_NPZ: "NumPy .npz archive",
    ModelFormat.TFLITE: "TensorFlow Lite",
    ModelFormat.FLAX_MSGPACK: "Flax MessagePack",
    ModelFormat.MINDSPORE_CKPT: "MindSpore Checkpoint (.ckpt)",
    ModelFormat.TFJS_WEIGHT_SHARD: "TF.js Weight Shard (.bin)",
}


def _make_g002_finding(
    artifact: Artifact,
    detected_format: ModelFormat,
    detection_details: str,
) -> Finding:
    """Create a G002 'format identified but no analyzer available' finding (A5).

    Emitted for formats that are identified by magic bytes but have no registered
    analyzer.  Severity is INFO — this is a coverage-gap notice, not a threat.
    """
    from model_scanner.core.normalizer import MAX_MATCHED_VALUE_LEN

    display_name = _FORMAT_DISPLAY_NAMES.get(detected_format, detected_format.value)
    matched_value = f"format={display_name}"[:MAX_MATCHED_VALUE_LEN]

    return Finding(
        finding_id=f"FINDING-G002-{artifact.sha256[:8]}-001",
        rule_id="G002",
        rule_name="Format Identified — No Analyzer Available",
        severity=Severity.INFO,
        confidence=Confidence.HIGH,
        category="format_integrity",
        format=detected_format,
        artifact_sha256=artifact.sha256,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=0,
            byte_length=8,
            archive_member=None,
            matched_value=matched_value,
            context_snippet=detection_details,
        ),
        description=(
            f"The artifact was identified as {display_name} but this scanner has "
            f"no analyzer for that format.  The artifact was not inspected; "
            f"security coverage is incomplete."
        ),
        remediation=(
            "Verify the artifact using a format-specific tool.  "
            "Do not assume a clean result from this scanner alone."
        ),
    )


def _is_coverage_gap_for_test(r: ArtifactScanResult) -> bool:
    """True when an INCONCLUSIVE result is a format-gap (no analyzer), not a parse error.

    Public for testing only.  Production callers should use the nested helper
    inside _build_summary, which uses the same logic.

    A coverage-gap artifact has:
    - scan_outcome == INCONCLUSIVE
    - no operational errors (no ANALYZER_FAILED, no MALFORMED_ARTIFACT)
    - no fatal parse-failure findings (no P201, ST209, etc.)
    """
    return (
        r.scan_outcome == ScanOutcome.INCONCLUSIVE
        and not r.errors
        and not any(f.rule_id in _FATAL_FINDING_RULES for f in r.findings)
    )


def get_analyzer(model_format: ModelFormat) -> BaseFormatAnalyzer | None:
    """Return an instantiated analyzer for the given ModelFormat, or None if unsupported.

    Parameters
    ----------
    model_format:
        The detected model format.

    Returns
    -------
    BaseFormatAnalyzer | None
        Instantiated analyzer, or None if format is UNSUPPORTED or not registered.
    """
    analyzer_class = _ANALYZER_REGISTRY.get(model_format)
    if analyzer_class is None:
        return None
    return analyzer_class()


# ---------------------------------------------------------------------------
# Scan Orchestrator
# ---------------------------------------------------------------------------


class ScanOrchestrator:
    """Central pipeline coordinator for the static security scanner.

    Responsibilities:
    - Route ScanRequest to the appropriate acquisition provider.
    - Dispatch each Artifact to format detection and the correct analyzer.
    - Collect findings and assemble per-artifact and aggregate scan results.
    - Enforce fail-closed semantics: errors become ScanError records.
    - Ensure per-artifact isolation: one artifact failure does not corrupt others.
    """

    def __init__(self, config: ScanConfig | None = None) -> None:
        """Initialize the orchestrator with optional global configuration.

        Parameters
        ----------
        config:
            Global scan configuration. If None, uses ScanConfig defaults.
        """
        self.config = config or ScanConfig()
        self._format_dispatcher = FormatDispatcher(self.config)

    def scan(
        self,
        request: ScanRequest,
        *,
        raise_on_acquisition_error: bool = False,
        scan_id: str | None = None,
    ) -> ScanResult:
        """Execute the complete scan pipeline for the given request.

        Parameters
        ----------
        request:
            The incoming scan request describing target and configuration.
        raise_on_acquisition_error:
            When False (the default, and the behaviour every existing caller
            relies on) a fatal acquisition failure is absorbed into an
            artifact-less ERROR summary. When True the original
            ``ScannerException`` propagates instead.

            The summary cannot carry the exception's code or message, so a
            caller that reports failures out-of-band — the web layer, which
            surfaces them through the job's ``error_code`` / ``error_message``
            — must opt in, or the reason for the failure is lost and the user
            sees an empty ERROR report.
        scan_id:
            Optional pre-supplied UUIDv4 string. When provided (e.g. by the
            web executor so its job ID and the result ID match), this value is
            used directly instead of generating a new UUID.

        Returns
        -------
        ScanResult
            Complete scan result with per-artifact details and aggregate summary.

        Raises
        ------
        ScannerException
            Only when ``raise_on_acquisition_error`` is True and acquisition fails.
        """
        start_time_ms = int(time.time() * 1000)
        scan_id = scan_id if scan_id is not None else str(uuid.uuid4())
        timestamp_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        # Merge request-level config with orchestrator-level config
        effective_config = request.config or self.config
        strict_mode = request.strict or effective_config.strict_mode

        # Determine target string for reporting
        if isinstance(request.target, (str, Path)):
            target_request = str(request.target)
        elif isinstance(request.target, bytes):
            target_request = f"<in-memory bytes: {len(request.target)} bytes>"
        else:
            target_request = "<in-memory stream>"

        # Step 1: Acquire artifacts
        acquisition_warnings: list[ScanWarning] = []
        try:
            artifacts, acquisition_warnings = self._acquire_artifacts(request)
        except ScannerException:
            # Fatal acquisition error — no artifacts to scan
            if raise_on_acquisition_error:
                raise
            summary = ScanSummary(
                total_artifacts=0,
                scanned_successfully=0,
                failed_artifacts=0,
                unsupported_artifacts=0,
                coverage_gap_artifacts=0,
                findings_by_severity={
                    "CRITICAL": 0,
                    "HIGH": 0,
                    "MEDIUM": 0,
                    "LOW": 0,
                    "INFO": 0,
                },
                artifacts_by_risk_band={
                    "CLEAN": 0,
                    "UNANALYZED": 0,
                    "LOW_RISK": 0,
                    "MEDIUM_RISK": 0,
                    "HIGH_RISK": 0,
                    "CRITICAL_RISK": 0,
                },
                max_risk_score=0,
                overall_outcome=ScanOutcome.ERROR,
                exit_code=2,
            )
            end_time_ms = int(time.time() * 1000)
            return ScanResult(
                scan_id=scan_id,
                scanner_version=__version__,
                timestamp_utc=timestamp_utc,
                target_request=target_request,
                strict_mode=strict_mode,
                artifacts=(),
                summary=summary,
                total_duration_ms=end_time_ms - start_time_ms,
            )

        # Step 2: Scan each artifact
        artifact_results: list[ArtifactScanResult] = []
        for artifact in artifacts:
            artifact_result = self.scan_artifact(artifact, effective_config)
            artifact_results.append(artifact_result)

        # Step 2b: For directory scans, run the supply-chain repository analyzer
        if request.input_source_type == InputSourceType.LOCAL_DIRECTORY and isinstance(
            request.target, (str, Path)
        ):
            repo_result = self._scan_repository(
                Path(request.target), artifacts, effective_config
            )
            if repo_result is not None:
                artifact_results.append(repo_result)

        # Step 2c: Cross-Artifact Correlation
        import dataclasses

        from model_scanner.core.correlation import CrossArtifactCorrelator

        correlated_findings = CrossArtifactCorrelator.correlate_scan(artifact_results)
        if correlated_findings:
            corr_by_sha: dict[str, list[Finding]] = {}
            for cf in correlated_findings:
                corr_by_sha.setdefault(cf.artifact_sha256, []).append(cf)

            # Point the source findings back at the correlations that cite them,
            # so a finding that is half of a detected chain no longer renders as
            # "Correlated Finding IDs: None".
            backlinks = CrossArtifactCorrelator.backlink_sources(
                artifact_results, correlated_findings
            )

            updated_results: list[ArtifactScanResult] = []
            for result in artifact_results:
                new_cfs = corr_by_sha.get(result.sha256)
                if new_cfs:
                    linked_existing = tuple(
                        dataclasses.replace(
                            f,
                            correlated_finding_ids=tuple(
                                dict.fromkeys(
                                    list(f.correlated_finding_ids)
                                    + list(backlinks[f.finding_id])
                                )
                            ),
                        )
                        if f.finding_id in backlinks
                        else f
                        for f in result.findings
                    )
                    combined_findings = tuple(list(linked_existing) + new_cfs)
                    new_risk = RiskEngine.calculate(combined_findings, result.detected_format)
                    updated_result = dataclasses.replace(
                        result,
                        findings=combined_findings,
                        risk_score=new_risk,
                    )
                    updated_results.append(updated_result)
                else:
                    updated_results.append(result)
            artifact_results = updated_results

        # Step 3: Sort results deterministically by source_origin (alphabetical)
        artifact_results_sorted = sorted(artifact_results, key=lambda r: r.source_origin)

        # Step 4: Aggregate summary
        summary = self._build_summary(tuple(artifact_results_sorted))

        end_time_ms = int(time.time() * 1000)
        return ScanResult(
            scan_id=scan_id,
            scanner_version=__version__,
            timestamp_utc=timestamp_utc,
            target_request=target_request,
            strict_mode=strict_mode,
            artifacts=tuple(artifact_results_sorted),
            summary=summary,
            total_duration_ms=end_time_ms - start_time_ms,
            acquisition_warnings=tuple(acquisition_warnings),
        )

    def scan_artifact(
        self, artifact: Artifact, config: ScanConfig | None = None
    ) -> ArtifactScanResult:
        """Scan a single artifact through the complete pipeline.

        Parameters
        ----------
        artifact:
            The acquired artifact to scan.
        config:
            Scan configuration (falls back to orchestrator config if None).

        Returns
        -------
        ArtifactScanResult
            Complete per-artifact scan result with findings, errors, warnings.
        """
        start_time_ms = int(time.time() * 1000)
        effective_config = config or self.config

        name = (
            artifact.local_path.name
            if artifact.local_path
            else Path(artifact.source_path).name
        )

        warnings: list[ScanWarning] = []
        errors: list[ScanError] = []

        # Step 1: Format Detection
        try:
            detection_result = self._format_dispatcher.detect_format(artifact)
            detected_format = detection_result.detected_format

            # Check for format mismatch (spoofing indicator)
            if detection_result.format_mismatch:
                warnings.append(
                    ScanWarning(
                        warning_code="FORMAT_MISMATCH",
                        message="File extension does not match detected content format",
                        context=detection_result.details,
                    )
                )

        except Exception as exc:
            # Format detection failure → ERROR outcome
            errors.append(
                ScanError(
                    error_code="FORMAT_DETECTION_FAILED",
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    traceback_summary=traceback.format_exc(limit=3),
                )
            )
            end_time_ms = int(time.time() * 1000)
            return ArtifactScanResult(
                artifact_id=artifact.artifact_id,
                name=name,
                source_origin=artifact.source_path,
                input_source=artifact.input_source,
                detected_format=ModelFormat.UNSUPPORTED,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                scan_outcome=ScanOutcome.ERROR,
                risk_score=RiskScore.unanalyzed(),
                findings=(),
                warnings=tuple(warnings),
                errors=tuple(errors),
                duration_ms=end_time_ms - start_time_ms,
            )

        # Step 2a: Identified-but-unanalyzed format (A5) — emit G002 + WARNING outcome
        if detected_format in _IDENTIFIED_UNANALYZED_FORMATS:
            g002 = _make_g002_finding(artifact, detected_format, detection_result.details)
            normalized_findings = FindingCorrelator.correlate(
                findings=[g002],
                detection_result=detection_result if detection_result.format_mismatch else None,
                sha256=artifact.sha256,
                strict_mode=effective_config.strict_mode,
            )
            # The format was identified but nothing inspected it. Scoring the lone
            # G002 finding yields 0, which would band this CLEAN and contradict
            # G002's own text ("the artifact was not inspected").
            risk_score = RiskScore.unanalyzed()
            end_time_ms = int(time.time() * 1000)
            return ArtifactScanResult(
                artifact_id=artifact.artifact_id,
                name=name,
                source_origin=artifact.source_path,
                input_source=artifact.input_source,
                detected_format=detected_format,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                scan_outcome=ScanOutcome.WARNING,
                risk_score=risk_score,
                findings=tuple(normalized_findings),
                warnings=tuple(warnings),
                errors=tuple(errors),
                duration_ms=end_time_ms - start_time_ms,
            )

        # Step 2aa: Format detected but no analyzer and INCONCLUSIVE semantics.
        # Unlike _IDENTIFIED_UNANALYZED_FORMATS (which give WARNING), these formats
        # give INCONCLUSIVE because they overlap with analyzable extensions and
        # routing them to the wrong analyzer would produce spurious HIGH findings.
        if detected_format in _FORMAT_UNSUPPORTED_FORMATS:
            g002 = _make_g002_finding(artifact, detected_format, detection_result.details)
            normalized_findings = FindingCorrelator.correlate(
                findings=[g002],
                detection_result=detection_result if detection_result.format_mismatch else None,
                sha256=artifact.sha256,
                strict_mode=effective_config.strict_mode,
            )
            risk_score = RiskScore.unanalyzed()
            end_time_ms = int(time.time() * 1000)
            return ArtifactScanResult(
                artifact_id=artifact.artifact_id,
                name=name,
                source_origin=artifact.source_path,
                input_source=artifact.input_source,
                detected_format=detected_format,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                scan_outcome=ScanOutcome.INCONCLUSIVE,
                risk_score=risk_score,
                findings=tuple(normalized_findings),
                warnings=tuple(warnings),
                errors=tuple(errors),
                duration_ms=end_time_ms - start_time_ms,
            )

        # Step 2b: Completely unknown / UNSUPPORTED format
        if detected_format == ModelFormat.UNSUPPORTED:
            warnings.append(
                ScanWarning(
                    warning_code="UNSUPPORTED_FORMAT",
                    message="File format is not supported by any registered analyzer",
                    context=detection_result.details,
                )
            )
            end_time_ms = int(time.time() * 1000)
            return ArtifactScanResult(
                artifact_id=artifact.artifact_id,
                name=name,
                source_origin=artifact.source_path,
                input_source=artifact.input_source,
                detected_format=ModelFormat.UNSUPPORTED,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                scan_outcome=ScanOutcome.UNSUPPORTED,
                risk_score=RiskScore.unanalyzed(),
                findings=(),
                warnings=tuple(warnings),
                errors=tuple(errors),
                duration_ms=end_time_ms - start_time_ms,
            )

        # Step 3: Analyzer Registry Lookup
        analyzer = get_analyzer(detected_format)
        if analyzer is None:
            # Format recognized but no analyzer registered (should not happen)
            warnings.append(
                ScanWarning(
                    warning_code="NO_ANALYZER",
                    message=f"No analyzer registered for format {detected_format.value}",
                    context=None,
                )
            )
            end_time_ms = int(time.time() * 1000)
            return ArtifactScanResult(
                artifact_id=artifact.artifact_id,
                name=name,
                source_origin=artifact.source_path,
                input_source=artifact.input_source,
                detected_format=detected_format,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                scan_outcome=ScanOutcome.UNSUPPORTED,
                risk_score=RiskScore.unanalyzed(),
                findings=(),
                warnings=tuple(warnings),
                errors=tuple(errors),
                duration_ms=end_time_ms - start_time_ms,
            )

        # Step 4: Execute Analyzer
        #
        # Artifact is a frozen dataclass produced by the acquisition layer, which
        # runs before detection and so cannot know the format. Analyzers that
        # behave differently per detected format (PickleAnalyzer, for legacy
        # non-ZIP PyTorch) need that verdict, so attach it here.
        object.__setattr__(artifact, "detected_format", detected_format)
        model_structure: dict[str, object] | None = None
        try:
            findings = analyzer.analyze(artifact, effective_config)
            model_structure = getattr(analyzer, "model_structure", None)
        except Exception as exc:
            # Analyzer failure → fail-closed ERROR outcome
            errors.append(
                ScanError(
                    error_code="ANALYZER_FAILED",
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    traceback_summary=traceback.format_exc(limit=3),
                )
            )
            end_time_ms = int(time.time() * 1000)
            return ArtifactScanResult(
                artifact_id=artifact.artifact_id,
                name=name,
                source_origin=artifact.source_path,
                input_source=artifact.input_source,
                detected_format=detected_format,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
                scan_outcome=ScanOutcome.INCONCLUSIVE,
                risk_score=RiskScore.unanalyzed(),
                findings=(),
                warnings=tuple(warnings),
                errors=tuple(errors),
                duration_ms=end_time_ms - start_time_ms,
            )

        # Step 5: Normalize and Correlate Findings
        strict_mode = effective_config.strict_mode
        normalized_findings = FindingCorrelator.correlate(
            findings=findings,
            detection_result=detection_result,
            sha256=artifact.sha256,
            strict_mode=strict_mode,
        )

        # Step 6: Check for fatal structural findings (fail-closed semantics).
        # These rules fire when the file is fundamentally malformed or is the
        # wrong format — they are parse/coverage errors, NOT security findings.
        has_fatal_finding = any(
            f.rule_id in _FATAL_FINDING_RULES for f in normalized_findings
        )

        if has_fatal_finding:
            errors.append(
                ScanError(
                    error_code="MALFORMED_ARTIFACT",
                    message=(
                        f"Artifact is malformed or unreadable: "
                        f"{detected_format.value} file failed structural validation"
                    ),
                    exception_type="StructuralIntegrityFailure",
                    traceback_summary=None,
                )
            )

        # Separate genuine security findings from parse-error indicators.
        # Fatal findings signal that the artifact could not be fully read;
        # they must not inflate the security risk score (R3).
        genuine_findings = tuple(
            f for f in normalized_findings if f.rule_id not in _FATAL_FINDING_RULES
        )

        # Step 7: Calculate RiskScore via RiskEngine
        if has_fatal_finding and not genuine_findings:
            # Parse/format error only — coverage is incomplete.
            # Use unanalyzed() so no spurious risk band is assigned.
            risk_score = RiskScore.unanalyzed()
        else:
            # Score genuine security findings only; exclude parse-error artifacts.
            findings_for_scoring = genuine_findings if has_fatal_finding else normalized_findings
            risk_score = RiskEngine.calculate(
                findings=findings_for_scoring,
                format=detected_format,
            )

        # Step 8: Assemble ArtifactScanResult
        if has_fatal_finding and not genuine_findings:
            # Only parse errors fired: artifact could not be confidently analyzed.
            scan_outcome = ScanOutcome.INCONCLUSIVE
        elif errors:
            scan_outcome = ScanOutcome.ERROR
        else:
            scan_outcome = ScanOutcome.COMPLETE
        end_time_ms = int(time.time() * 1000)

        return ArtifactScanResult(
            artifact_id=artifact.artifact_id,
            name=name,
            source_origin=artifact.source_path,
            input_source=artifact.input_source,
            detected_format=detected_format,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            scan_outcome=scan_outcome,
            risk_score=risk_score,
            findings=tuple(normalized_findings),
            warnings=tuple(warnings),
            errors=tuple(errors),
            duration_ms=end_time_ms - start_time_ms,
            model_structure=model_structure,
        )

    def _scan_repository(
        self,
        repo_root: Path,
        artifacts: Sequence[Artifact],
        config: ScanConfig,
    ) -> ArtifactScanResult | None:
        """Run the supply-chain RepositoryAnalyzer on a directory root.

        Returns an ArtifactScanResult whose source_origin is the repo path,
        or None if the directory does not exist or is not a valid model repo.
        """
        if not repo_root.is_dir():
            return None

        from model_scanner.analyzers.repository.analyzer import (
            RepositoryAnalyzer,
            make_repository_artifact,
        )

        start_time_ms = int(time.time() * 1000)
        repo_artifact = make_repository_artifact(repo_root)
        weight_files = [Path(a.source_path) for a in artifacts if a.source_path]

        repo_name = repo_root.name or str(repo_root)

        try:
            raw_findings = RepositoryAnalyzer().analyze_directory(
                repo_root, weight_files, config
            )
        except Exception as exc:
            end_time_ms = int(time.time() * 1000)
            return ArtifactScanResult(
                artifact_id=repo_artifact.artifact_id,
                name=repo_name,
                source_origin=str(repo_root),
                input_source=InputSourceType.LOCAL_DIRECTORY,
                detected_format=ModelFormat.REPOSITORY,
                size_bytes=0,
                sha256=repo_artifact.sha256,
                scan_outcome=ScanOutcome.ERROR,
                risk_score=RiskScore.unanalyzed(),
                findings=(),
                warnings=(),
                errors=(
                    ScanError(
                        error_code="REPO_ANALYZER_ERROR",
                        message=f"Repository analyzer failed: {exc}",
                        exception_type=type(exc).__name__,
                        traceback_summary=traceback.format_exc(),
                    ),
                ),
                duration_ms=end_time_ms - start_time_ms,
            )

        normalized = FindingCorrelator.correlate(
            findings=raw_findings,
            detection_result=None,
            sha256=repo_artifact.sha256,
            strict_mode=config.strict_mode,
        )
        risk_score = RiskEngine.calculate(
            findings=normalized,
            format=ModelFormat.REPOSITORY,
        )

        end_time_ms = int(time.time() * 1000)
        return ArtifactScanResult(
            artifact_id=repo_artifact.artifact_id,
            name=repo_name,
            source_origin=str(repo_root),
            input_source=InputSourceType.LOCAL_DIRECTORY,
            detected_format=ModelFormat.REPOSITORY,
            size_bytes=0,
            sha256=repo_artifact.sha256,
            scan_outcome=ScanOutcome.COMPLETE,
            risk_score=risk_score,
            findings=tuple(normalized),
            warnings=(),
            errors=(),
            duration_ms=end_time_ms - start_time_ms,
        )

    def _acquire_artifacts(
        self, request: ScanRequest
    ) -> tuple[Sequence[Artifact], list[ScanWarning]]:
        """Route the request to the appropriate acquisition provider.

        Parameters
        ----------
        request:
            The incoming scan request.

        Returns
        -------
        tuple[Sequence[Artifact], list[ScanWarning]]
            Acquired artifacts and any acquisition-level warnings (e.g. blocked
            symlink escapes from LocalDirectoryAcquisition).

        Raises
        ------
        ScannerException
            If acquisition fails.
        """
        provider: InputAcquisitionProvider
        acq_warnings: list[ScanWarning] = []

        if request.input_source_type == InputSourceType.LOCAL_FILE:
            provider = LocalFileAcquisition()
        elif request.input_source_type == InputSourceType.LOCAL_DIRECTORY:
            dir_provider = LocalDirectoryAcquisition()
            artifacts = dir_provider.acquire(request)
            acq_warnings = dir_provider.skipped_symlinks
            return artifacts, acq_warnings
        elif request.input_source_type == InputSourceType.PUBLIC_URL:
            if isinstance(request.target, str) and is_hf_url(request.target):
                provider = HuggingFaceAcquisition()
            else:
                provider = PublicUrlAcquisition()
        else:
            raise NotImplementedError(
                f"Acquisition provider for {request.input_source_type.value} "
                f"is not yet implemented"
            )

        return provider.acquire(request), acq_warnings

    def _build_summary(
        self, artifact_results: tuple[ArtifactScanResult, ...]
    ) -> ScanSummary:
        """Aggregate per-artifact results into a ScanSummary.

        Parameters
        ----------
        artifact_results:
            Tuple of all artifact scan results.

        Returns
        -------
        ScanSummary
            Aggregate summary with counts, max risk, and ClamAV exit code.
        """
        total_artifacts = len(artifact_results)
        if total_artifacts == 0:
            return ScanSummary(
                total_artifacts=0,
                scanned_successfully=0,
                failed_artifacts=0,
                unsupported_artifacts=0,
                coverage_gap_artifacts=0,
                findings_by_severity={
                    "CRITICAL": 0,
                    "HIGH": 0,
                    "MEDIUM": 0,
                    "LOW": 0,
                    "INFO": 0,
                },
                artifacts_by_risk_band={
                    "CLEAN": 0,
                    "LOW_RISK": 0,
                    "MEDIUM_RISK": 0,
                    "HIGH_RISK": 0,
                    "CRITICAL_RISK": 0,
                    "UNANALYZED": 0,
                },
                max_risk_score=0,
                overall_outcome=ScanOutcome.UNSUPPORTED,
                exit_code=0,
            )

        scanned_successfully = sum(
            1
            for r in artifact_results
            if r.scan_outcome in (ScanOutcome.COMPLETE, ScanOutcome.WARNING)
        )
        # Hard failures: operational errors and parse-failure INCONCLUSIVE.
        # These indicate analysis could not complete due to scanner/file errors.
        hard_failed_artifacts = sum(
            1
            for r in artifact_results
            if r.scan_outcome == ScanOutcome.ERROR
            or (r.scan_outcome == ScanOutcome.INCONCLUSIVE and not _is_coverage_gap_for_test(r))
        )
        # Coverage gaps: format detected but no analyzer registered.
        # Analysis was not attempted, but this is a scanner limitation, not a failure.
        coverage_gap_artifacts = sum(
            1 for r in artifact_results if _is_coverage_gap_for_test(r)
        )
        # failed_artifacts in the public summary = hard failures only
        failed_artifacts = hard_failed_artifacts
        unsupported_artifacts = sum(
            1 for r in artifact_results if r.scan_outcome == ScanOutcome.UNSUPPORTED
        )

        # Aggregate findings by severity
        findings_by_severity: dict[str, int] = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0,
        }
        for result in artifact_results:
            for finding in result.findings:
                findings_by_severity[finding.severity.value] += 1

        # Aggregate artifacts by risk band
        artifacts_by_risk_band: dict[str, int] = {
            "CLEAN": 0,
            "UNANALYZED": 0,
            "LOW_RISK": 0,
            "MEDIUM_RISK": 0,
            "HIGH_RISK": 0,
            "CRITICAL_RISK": 0,
        }
        for result in artifact_results:
            artifacts_by_risk_band[result.risk_score.risk_band.value] += 1

        # Determine max risk score
        max_risk_score = (
            max(r.risk_score.final_score for r in artifact_results)
            if artifact_results
            else 0
        )

        # Has_analyzed_findings: findings from non-gap artifacts only.
        # Coverage-gap findings (G002 INFO from TFJS_WEIGHT_SHARD) must not
        # trigger overall WARNING, since no security analysis was performed.
        has_analyzed_findings = any(
            len(r.findings) > 0
            for r in artifact_results
            if not _is_coverage_gap_for_test(r)
        )

        # Determine overall outcome
        # Priority: ERROR > WARNING > INCONCLUSIVE > UNSUPPORTED > COMPLETE
        if hard_failed_artifacts > 0:
            overall_outcome = ScanOutcome.ERROR
        elif has_analyzed_findings:
            overall_outcome = ScanOutcome.WARNING
        elif coverage_gap_artifacts > 0:
            # Format-gap artifacts: analysis was not possible, not a security failure.
            overall_outcome = ScanOutcome.INCONCLUSIVE
        elif unsupported_artifacts > 0 and scanned_successfully == 0:
            overall_outcome = ScanOutcome.UNSUPPORTED
        else:
            overall_outcome = ScanOutcome.COMPLETE

        # Determine scanner exit code
        # 0: clean (no findings, no errors, no gaps)
        # 1: findings present OR coverage gaps (incomplete coverage, not an error)
        # 2: operational errors (analysis failed)
        if hard_failed_artifacts > 0:
            exit_code = 2
        elif has_analyzed_findings or coverage_gap_artifacts > 0:
            exit_code = 1
        else:
            exit_code = 0

        return ScanSummary(
            total_artifacts=total_artifacts,
            scanned_successfully=scanned_successfully,
            failed_artifacts=failed_artifacts,
            unsupported_artifacts=unsupported_artifacts,
            coverage_gap_artifacts=coverage_gap_artifacts,
            findings_by_severity=findings_by_severity,
            artifacts_by_risk_band=artifacts_by_risk_band,
            max_risk_score=max_risk_score,
            overall_outcome=overall_outcome,
            exit_code=exit_code,
        )


__all__ = [
    "ScanOrchestrator",
    "get_analyzer",
    "_is_coverage_gap_for_test",
]
