"""
tests.unit.test_tfjs_shard_detection
=====================================

Regression tests for TF.js weight-shard false-positive fix.

Root cause (pre-fix):
    .bin extension → PYTORCH_ZIP via extension fallback in detector.py.
    TF.js weight shards have no ZIP or pickle magic → PyTorchAnalyzer opens file
    as ZIP → fails → P201 HIGH fires → INCONCLUSIVE/UNANALYZED → exit_code=2.

Expected behaviour (post-fix):
    .bin with no ZIP/pickle magic → TFJS_WEIGHT_SHARD format (not PYTORCH_ZIP).
    Engine route: Step 2aa (format detected, no analyzer) → G002 INFO + INCONCLUSIVE.
    INCONCLUSIVE with no errors and no fatal findings is a "coverage gap", not
    a hard failure, so overall_outcome=INCONCLUSIVE and exit_code=1.

Security constraint: Genuine P201 on malformed PyTorch ZIPs must still fire.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from model_scanner.core.models import (
    ModelFormat,
    ScanConfig,
    ScanOutcome,
    Severity,
)
from model_scanner.core.engine import ScanOrchestrator, _is_coverage_gap_for_test
from model_scanner.detection.detector import FormatDispatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZIP_MAGIC = b"PK\x03\x04" + b"\x00" * 60          # valid ZIP local-file header
_PICKLE_MAGIC = b"\x80\x02" + b"\x00" * 62          # pickle protocol 2 opcode
_RANDOM_BYTES = b"\x00\xde\xad\xbe\xef" + b"\x00" * 59  # no recognisable magic


def _make_artifact(
    name: str,
    content: bytes,
    source_path: str | None = None,
) -> object:
    """Build a minimal Artifact-compatible object suitable for scan_artifact."""
    from model_scanner.core.models import Artifact, InputSourceType
    import hashlib

    return Artifact(
        artifact_id=f"test-{name}",
        source_path=source_path or f"hf://test/{name}",
        local_path=None,
        stream=io.BytesIO(content),
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        input_source=InputSourceType.PUBLIC_URL,
    )


def _dispatch(header: bytes, filename: str) -> ModelFormat:
    """Run FormatDispatcher.detect_from_bytes and return the detected format."""
    dispatcher = FormatDispatcher()
    result = dispatcher.detect_from_bytes(header=header, filename_hint=filename)
    return result.detected_format


# ---------------------------------------------------------------------------
# Format detection tests (detector.py)
# ---------------------------------------------------------------------------


class TestTfjsShardFormatDetection:
    def test_bin_no_magic_detects_tfjs_weight_shard(self) -> None:
        """Core regression: .bin with no ZIP/pickle magic → TFJS_WEIGHT_SHARD."""
        assert _dispatch(_RANDOM_BYTES, "group1-shard1of2.bin") == ModelFormat.TFJS_WEIGHT_SHARD

    def test_bin_all_zeros_detects_tfjs_weight_shard(self) -> None:
        """Null-filled .bin (common for zeroed weight shards) → TFJS_WEIGHT_SHARD."""
        assert _dispatch(b"\x00" * 64, "weights.bin") == ModelFormat.TFJS_WEIGHT_SHARD

    def test_bin_with_zip_magic_stays_pytorch_zip(self) -> None:
        """Regression: .bin containing a real ZIP archive → PYTORCH_ZIP (genuine PyTorch)."""
        assert _dispatch(_ZIP_MAGIC, "model.bin") == ModelFormat.PYTORCH_ZIP

    def test_bin_with_pickle_magic_is_legacy_pytorch(self) -> None:
        """Regression: .bin starting with pickle opcode → LEGACY_PYTORCH."""
        assert _dispatch(_PICKLE_MAGIC, "model.bin") == ModelFormat.LEGACY_PYTORCH

    def test_pt_no_magic_still_falls_back_to_pytorch_zip(self) -> None:
        """.pt with no magic falls back via extension → PYTORCH_ZIP (not TFJS_WEIGHT_SHARD)."""
        assert _dispatch(_RANDOM_BYTES, "model.pt") == ModelFormat.PYTORCH_ZIP

    def test_pth_no_magic_still_falls_back_to_pytorch_zip(self) -> None:
        """.pth with no magic falls back via extension → PYTORCH_ZIP."""
        assert _dispatch(_RANDOM_BYTES, "weights.pth") == ModelFormat.PYTORCH_ZIP

    def test_bin_no_extension_falls_through_to_unsupported(self) -> None:
        """Binary file with no extension → UNSUPPORTED (no extension fallback applies)."""
        assert _dispatch(_RANDOM_BYTES, "blob") == ModelFormat.UNSUPPORTED

    def test_tfjs_shard_format_mismatch_flag_false(self) -> None:
        """TFJS_WEIGHT_SHARD detection must not set format_mismatch=True."""
        dispatcher = FormatDispatcher()
        result = dispatcher.detect_from_bytes(
            header=_RANDOM_BYTES, filename_hint="group1-shard1of2.bin"
        )
        assert result.detected_format == ModelFormat.TFJS_WEIGHT_SHARD
        assert result.format_mismatch is False


# ---------------------------------------------------------------------------
# Scan engine tests — single TFJS_WEIGHT_SHARD artifact
# ---------------------------------------------------------------------------


class TestTfjsShardScanArtifact:
    def setup_method(self) -> None:
        self.orchestrator = ScanOrchestrator(ScanConfig())
        self.artifact = _make_artifact(
            "shard1of2.bin", _RANDOM_BYTES, "hf://test/shard1of2.bin"
        )

    def _scan(self):
        return self.orchestrator.scan_artifact(self.artifact, ScanConfig())

    def test_detected_format_is_tfjs_weight_shard(self) -> None:
        result = self._scan()
        assert result.detected_format == ModelFormat.TFJS_WEIGHT_SHARD

    def test_outcome_is_inconclusive(self) -> None:
        """TFJS_WEIGHT_SHARD must produce INCONCLUSIVE, not ERROR."""
        result = self._scan()
        assert result.scan_outcome == ScanOutcome.INCONCLUSIVE

    def test_risk_band_is_unanalyzed(self) -> None:
        result = self._scan()
        assert result.risk_score.risk_band.value == "UNANALYZED"

    def test_no_high_severity_findings(self) -> None:
        """P201 HIGH must NOT fire for TF.js weight shards (core false-positive fix)."""
        result = self._scan()
        high_critical = [
            f for f in result.findings
            if f.severity in (Severity.HIGH, Severity.CRITICAL)
        ]
        assert high_critical == [], (
            f"Unexpected HIGH/CRITICAL findings: {[f.rule_id for f in high_critical]}"
        )

    def test_no_p201_finding(self) -> None:
        """P201 must specifically not be emitted for TF.js weight shards."""
        result = self._scan()
        p201 = [f for f in result.findings if f.rule_id == "P201"]
        assert p201 == []

    def test_g002_info_finding_present(self) -> None:
        """G002 INFO 'Format Identified — No Analyzer Available' must be emitted."""
        result = self._scan()
        g002 = [f for f in result.findings if f.rule_id == "G002"]
        assert len(g002) == 1
        assert g002[0].severity == Severity.INFO

    def test_no_operational_errors(self) -> None:
        """TFJS_WEIGHT_SHARD INCONCLUSIVE must have no errors (it's a coverage gap)."""
        result = self._scan()
        assert result.errors == ()

    def test_is_classified_as_coverage_gap(self) -> None:
        """The artifact must be classified as a coverage gap, not a hard failure."""
        result = self._scan()
        assert _is_coverage_gap_for_test(result) is True


# ---------------------------------------------------------------------------
# Scan summary tests — overall_outcome and exit_code
# ---------------------------------------------------------------------------


class TestTfjsShardSummary:
    def setup_method(self) -> None:
        self.orchestrator = ScanOrchestrator(ScanConfig())

    def _full_scan(self, artifact):
        """Run a single-artifact full scan pipeline."""
        from model_scanner.core.models import (
            InputSourceType,
            ScanRequest,
        )
        import io as _io

        # Write artifact bytes to a temp-like stream and re-scan
        content = artifact.stream.read()
        artifact.stream.seek(0)

        results = [self.orchestrator.scan_artifact(artifact, ScanConfig())]
        return self.orchestrator._build_summary(tuple(results))

    def test_exit_code_is_1_not_2_for_coverage_gap(self) -> None:
        """Coverage-gap INCONCLUSIVE must produce exit_code=1, not 2."""
        artifact = _make_artifact("shard.bin", _RANDOM_BYTES, "hf://test/shard.bin")
        summary = self._full_scan(artifact)
        assert summary.exit_code == 1, (
            f"Expected exit_code=1 for coverage-gap INCONCLUSIVE, got {summary.exit_code}"
        )

    def test_overall_outcome_is_inconclusive(self) -> None:
        """Overall outcome must be INCONCLUSIVE (not ERROR) for format-gap artifacts."""
        artifact = _make_artifact("shard.bin", _RANDOM_BYTES, "hf://test/shard.bin")
        summary = self._full_scan(artifact)
        assert summary.overall_outcome == ScanOutcome.INCONCLUSIVE

    def test_failed_artifacts_is_zero(self) -> None:
        """Coverage-gap INCONCLUSIVE must NOT count toward failed_artifacts."""
        artifact = _make_artifact("shard.bin", _RANDOM_BYTES, "hf://test/shard.bin")
        summary = self._full_scan(artifact)
        assert summary.failed_artifacts == 0

    def test_coverage_gap_artifacts_is_one(self) -> None:
        """coverage_gap_artifacts counter must reflect the TF.js shard."""
        artifact = _make_artifact("shard.bin", _RANDOM_BYTES, "hf://test/shard.bin")
        summary = self._full_scan(artifact)
        assert summary.coverage_gap_artifacts == 1

    def test_findings_by_severity_no_high(self) -> None:
        """findings_by_severity must not include any HIGH/CRITICAL for TF.js shards."""
        artifact = _make_artifact("shard.bin", _RANDOM_BYTES, "hf://test/shard.bin")
        summary = self._full_scan(artifact)
        assert summary.findings_by_severity.get("HIGH", 0) == 0
        assert summary.findings_by_severity.get("CRITICAL", 0) == 0

    def test_findings_by_severity_has_info(self) -> None:
        """findings_by_severity must record the G002 INFO finding."""
        artifact = _make_artifact("shard.bin", _RANDOM_BYTES, "hf://test/shard.bin")
        summary = self._full_scan(artifact)
        assert summary.findings_by_severity.get("INFO", 0) >= 1


# ---------------------------------------------------------------------------
# Regression: genuine P201 on malformed PyTorch ZIP still fires
# ---------------------------------------------------------------------------


class TestGenuineP201Regression:
    def setup_method(self) -> None:
        self.orchestrator = ScanOrchestrator(ScanConfig())

    def test_malformed_pytorch_zip_still_produces_p201(self) -> None:
        """A .bin file with ZIP magic but invalid content → PYTORCH_ZIP → P201 HIGH."""
        # ZIP magic header but truncated/corrupt ZIP body — this is not a TF.js shard.
        corrupt_zip = _ZIP_MAGIC[:4] + b"\xff" * 60
        artifact = _make_artifact(
            "corrupt_model.bin", corrupt_zip, "hf://test/corrupt_model.bin"
        )
        result = self.orchestrator.scan_artifact(artifact, ScanConfig())
        assert result.detected_format == ModelFormat.PYTORCH_ZIP
        rule_ids = [f.rule_id for f in result.findings]
        assert "P201" in rule_ids, (
            f"Expected P201 for malformed ZIP .bin; got findings: {rule_ids}"
        )

    def test_malformed_pytorch_zip_is_not_a_coverage_gap(self) -> None:
        """Genuine P201 (parse failure) must be a hard INCONCLUSIVE, not a coverage gap."""
        corrupt_zip = _ZIP_MAGIC[:4] + b"\xff" * 60
        artifact = _make_artifact(
            "corrupt_model.bin", corrupt_zip, "hf://test/corrupt_model.bin"
        )
        result = self.orchestrator.scan_artifact(artifact, ScanConfig())
        assert _is_coverage_gap_for_test(result) is False

    def test_malformed_pytorch_zip_exit_code_is_2(self) -> None:
        """Genuine parse failure must still produce exit_code=2, not 1."""
        corrupt_zip = _ZIP_MAGIC[:4] + b"\xff" * 60
        artifact = _make_artifact(
            "corrupt_model.bin", corrupt_zip, "hf://test/corrupt_model.bin"
        )
        results = [self.orchestrator.scan_artifact(artifact, ScanConfig())]
        summary = self.orchestrator._build_summary(tuple(results))
        assert summary.exit_code == 2
        assert summary.failed_artifacts == 1
        assert summary.coverage_gap_artifacts == 0
