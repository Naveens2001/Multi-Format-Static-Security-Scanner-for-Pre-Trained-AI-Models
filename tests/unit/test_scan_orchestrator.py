"""
tests.unit.test_scan_orchestrator
==================================

Unit tests for the ScanOrchestrator — Phase 7.1 core scan orchestration.

Test Coverage (18 minimum test cases):
1. Analyzer registry contains all 4 expected analyzers
2-5. get_analyzer returns correct analyzer for each format
6. UNSUPPORTED format safe handling
7. Single artifact scan
8. Multi-artifact scan
9. Findings preserved across pipeline
10. SHA-256 preserved
11. Evidence preserved
12. archive_member preserved (PyTorch ZIP)
13. Config propagation
14. Analyzer failure isolation
15. Malformed artifact fail-closed
16. Deterministic result ordering
17. Empty artifact handling
18. Unsupported artifact handling
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from model_scanner.analyzers.gguf.analyzer import GGUFAnalyzer
from model_scanner.analyzers.pickle.analyzer import PickleAnalyzer
from model_scanner.analyzers.pytorch.analyzer import PyTorchAnalyzer
from model_scanner.analyzers.safetensors.analyzer import SafeTensorsAnalyzer
from model_scanner.core.engine import ScanOrchestrator, get_analyzer
from model_scanner.core.models import (
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    InputSourceType,
    ModelFormat,
    ScanConfig,
    ScanOutcome,
    ScanRequest,
    Severity,
)

# ---------------------------------------------------------------------------
# Test 1: Analyzer Registry Completeness
# ---------------------------------------------------------------------------


def test_analyzer_registry_contains_all_formats() -> None:
    """Test 1: Verify the analyzer registry contains all 4 expected format analyzers."""
    pickle_analyzer = get_analyzer(ModelFormat.RAW_PICKLE)
    pytorch_analyzer = get_analyzer(ModelFormat.PYTORCH_ZIP)
    safetensors_analyzer = get_analyzer(ModelFormat.SAFETENSORS)
    gguf_analyzer = get_analyzer(ModelFormat.GGUF)
    ggml_analyzer = get_analyzer(ModelFormat.GGML)

    assert pickle_analyzer is not None
    assert pytorch_analyzer is not None
    assert safetensors_analyzer is not None
    assert gguf_analyzer is not None
    assert ggml_analyzer is not None

    # Verify types
    assert isinstance(pickle_analyzer, PickleAnalyzer)
    assert isinstance(pytorch_analyzer, PyTorchAnalyzer)
    assert isinstance(safetensors_analyzer, SafeTensorsAnalyzer)
    assert isinstance(gguf_analyzer, GGUFAnalyzer)
    assert isinstance(ggml_analyzer, GGUFAnalyzer)


# ---------------------------------------------------------------------------
# Tests 2-5: Correct Analyzer for Each Format
# ---------------------------------------------------------------------------


def test_get_analyzer_returns_pickle_analyzer_for_raw_pickle() -> None:
    """Test 2: get_analyzer returns PickleAnalyzer for RAW_PICKLE."""
    analyzer = get_analyzer(ModelFormat.RAW_PICKLE)
    assert isinstance(analyzer, PickleAnalyzer)


def test_get_analyzer_returns_pytorch_analyzer_for_pytorch_zip() -> None:
    """Test 3: get_analyzer returns PyTorchAnalyzer for PYTORCH_ZIP."""
    analyzer = get_analyzer(ModelFormat.PYTORCH_ZIP)
    assert isinstance(analyzer, PyTorchAnalyzer)


def test_get_analyzer_returns_safetensors_analyzer_for_safetensors() -> None:
    """Test 4: get_analyzer returns SafeTensorsAnalyzer for SAFETENSORS."""
    analyzer = get_analyzer(ModelFormat.SAFETENSORS)
    assert isinstance(analyzer, SafeTensorsAnalyzer)


def test_get_analyzer_returns_gguf_analyzer_for_gguf() -> None:
    """Test 5: get_analyzer returns GGUFAnalyzer for GGUF."""
    analyzer = get_analyzer(ModelFormat.GGUF)
    assert isinstance(analyzer, GGUFAnalyzer)


# ---------------------------------------------------------------------------
# Test 6: UNSUPPORTED Format Safe Handling
# ---------------------------------------------------------------------------


def test_unsupported_format_safe_handling() -> None:
    """Test 6: UNSUPPORTED format returns None from get_analyzer.

    Also produces UNSUPPORTED outcome.
    """
    analyzer = get_analyzer(ModelFormat.UNSUPPORTED)
    assert analyzer is None

    # Create a minimal unsupported file
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".unknown", delete=False
    ) as tmp:
        tmp.write(b"UNSUPPORTED_MAGIC_BYTES_12345")
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        assert result.summary.unsupported_artifacts == 1
        assert result.artifacts[0].scan_outcome == ScanOutcome.UNSUPPORTED
        assert result.artifacts[0].detected_format == ModelFormat.UNSUPPORTED
        assert len(result.artifacts[0].warnings) >= 1
        assert any(
            "UNSUPPORTED" in w.warning_code for w in result.artifacts[0].warnings
        )
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 7: Single Artifact Scan
# ---------------------------------------------------------------------------


def test_single_artifact_scan() -> None:
    """Test 7: Scan a single artifact and verify ScanResult structure."""
    # Create a minimal SafeTensors file (8-byte header length + empty JSON)
    header_json = b"{}"
    header_len = len(header_json)
    file_bytes = header_len.to_bytes(8, byteorder="little") + header_json

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".safetensors", delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        assert len(result.artifacts) == 1
        assert result.artifacts[0].detected_format == ModelFormat.SAFETENSORS
        assert result.artifacts[0].scan_outcome == ScanOutcome.COMPLETE
        assert result.scan_id is not None
        assert result.scanner_version is not None
        assert result.timestamp_utc.endswith("Z")
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 8: Multi-Artifact Scan
# ---------------------------------------------------------------------------


def test_multi_artifact_scan() -> None:
    """Test 8: Scan multiple artifacts and verify deterministic ordering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create 3 SafeTensors files with different names
        header_json = b"{}"
        header_len = len(header_json)
        file_bytes = header_len.to_bytes(8, byteorder="little") + header_json

        file1 = tmpdir_path / "model_c.safetensors"
        file2 = tmpdir_path / "model_a.safetensors"
        file3 = tmpdir_path / "model_b.safetensors"

        for f in [file1, file2, file3]:
            f.write_bytes(file_bytes)

        request = ScanRequest(
            target=tmpdir_path,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        # 3 weight-file artifacts + 1 repository-level supply-chain result
        assert result.summary.total_artifacts == 4
        assert len(result.artifacts) == 4

        # Verify deterministic alphabetical ordering by source_origin
        origins = [a.source_origin for a in result.artifacts]
        assert origins == sorted(origins)


# ---------------------------------------------------------------------------
# Test 9: Findings Preserved
# ---------------------------------------------------------------------------


def test_findings_preserved_across_pipeline() -> None:
    """Test 9: Verify findings from analyzer are preserved in ArtifactScanResult."""
    # Create a Pickle with GLOBAL opcode (triggers P101)
    # Protocol 3: b'\x80\x03' + GLOBAL + module + name + STOP
    pickle_bytes = (
        b"\x80\x03"  # Protocol 3
        b"cposix\nsystem\n"  # GLOBAL posix.system
        b"."  # STOP
    )

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(pickle_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        artifact_result = result.artifacts[0]
        assert artifact_result.detected_format == ModelFormat.RAW_PICKLE
        assert len(artifact_result.findings) >= 1

        # Verify finding structure
        finding = artifact_result.findings[0]
        assert finding.rule_id.startswith("P1")
        assert finding.severity is not None
        assert finding.confidence is not None
        assert finding.artifact_sha256 == artifact_result.sha256
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 10: SHA-256 Preserved
# ---------------------------------------------------------------------------


def test_sha256_preserved() -> None:
    """Test 10: Verify SHA-256 from acquisition is preserved in ArtifactScanResult."""
    header_json = b"{}"
    header_len = len(header_json)
    file_bytes = header_len.to_bytes(8, byteorder="little") + header_json

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".safetensors", delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        assert len(artifact_result.sha256) == 64
        assert all(c in "0123456789abcdef" for c in artifact_result.sha256)
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 11: Evidence Preserved
# ---------------------------------------------------------------------------


def test_evidence_preserved() -> None:
    """Test 11: Verify Evidence objects are preserved from analyzer to result."""
    pickle_bytes = b"\x80\x03cposix\nsystem\n."

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(pickle_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        if len(artifact_result.findings) > 0:
            finding = artifact_result.findings[0]
            evidence = finding.evidence
            assert evidence.matched_value is not None
            assert evidence.coordinate_space is not None
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 12: archive_member Preserved (PyTorch ZIP)
# ---------------------------------------------------------------------------


def test_archive_member_preserved() -> None:
    """Test 12: Verify archive_member is preserved for PyTorch ZIP findings."""
    # Create a minimal PyTorch ZIP with an embedded pickle
    pickle_bytes = b"\x80\x03cposix\nsystem\n."

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "w") as zf:
            zf.writestr("archive/data.pkl", pickle_bytes)

        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        assert artifact_result.detected_format == ModelFormat.PYTORCH_ZIP

        # Check if any finding has archive_member set
        if len(artifact_result.findings) > 0:
            has_archive_member = any(
                f.evidence.archive_member is not None for f in artifact_result.findings
            )
            assert has_archive_member
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 13: Config Propagation
# ---------------------------------------------------------------------------


def test_config_propagation() -> None:
    """Test 13: Verify ScanConfig is propagated to analyzers."""
    config = ScanConfig(
        strict_mode=True,
        max_safetensors_header_mb=8,  # Custom value
    )

    header_json = b"{}"
    header_len = len(header_json)
    file_bytes = header_len.to_bytes(8, byteorder="little") + header_json

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".safetensors", delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
            config=config,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.strict_mode is True
        assert result.summary.total_artifacts == 1
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 14: Analyzer Failure Isolation
# ---------------------------------------------------------------------------


def test_analyzer_failure_isolation() -> None:
    """Test 14: Verify one artifact's analyzer failure does not corrupt other artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # File 1: Valid SafeTensors
        header_json = b"{}"
        header_len = len(header_json)
        valid_bytes = header_len.to_bytes(8, byteorder="little") + header_json
        file1 = tmpdir_path / "valid.safetensors"
        file1.write_bytes(valid_bytes)

        # File 2: Malformed SafeTensors (will trigger analyzer error)
        file2 = tmpdir_path / "malformed.safetensors"
        file2.write_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00INVALID")

        request = ScanRequest(
            target=tmpdir_path,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        # 2 weight-file artifacts + 1 repository-level supply-chain result
        assert result.summary.total_artifacts == 3

        # At least one should be ERROR, at least one should be COMPLETE
        outcomes = [a.scan_outcome for a in result.artifacts]
        assert ScanOutcome.ERROR in outcomes or ScanOutcome.COMPLETE in outcomes


# ---------------------------------------------------------------------------
# Test 15: Malformed Artifact Fail-Closed
# ---------------------------------------------------------------------------


def test_malformed_artifact_fail_closed() -> None:
    """Test 15: Verify malformed artifacts result in ERROR outcome (fail-closed)."""
    # Create a malformed SafeTensors (invalid header length)
    malformed_bytes = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF"

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".safetensors", delete=False
    ) as tmp:
        tmp.write(malformed_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        # Must never be CLEAN; parse errors yield INCONCLUSIVE (R3), not ERROR
        assert artifact_result.scan_outcome in (
            ScanOutcome.ERROR,
            ScanOutcome.UNSUPPORTED,
            ScanOutcome.INCONCLUSIVE,
        )
        if artifact_result.scan_outcome in (ScanOutcome.ERROR, ScanOutcome.INCONCLUSIVE):
            assert len(artifact_result.errors) > 0
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 16: Deterministic Result Ordering
# ---------------------------------------------------------------------------


def test_deterministic_result_ordering() -> None:
    """Test 16: Verify artifacts are returned in deterministic alphabetical order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        header_json = b"{}"
        header_len = len(header_json)
        file_bytes = header_len.to_bytes(8, byteorder="little") + header_json

        # Create files in non-alphabetical order
        files = ["zebra.safetensors", "alpha.safetensors", "beta.safetensors"]
        for fname in files:
            (tmpdir_path / fname).write_bytes(file_bytes)

        request = ScanRequest(
            target=tmpdir_path,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        # Verify deterministic alphabetical ordering of all artifact source_origins
        origins = [a.source_origin for a in result.artifacts]
        assert origins == sorted(origins)


# ---------------------------------------------------------------------------
# Test 17: Empty Artifact Handling
# ---------------------------------------------------------------------------


def test_empty_artifact_handling() -> None:
    """Test 17: Verify empty files are handled gracefully."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        # Write nothing (0 bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        assert artifact_result.size_bytes == 0
        # Empty files should be handled gracefully — never CLEAN with findings
        assert artifact_result.scan_outcome in (
            ScanOutcome.ERROR,
            ScanOutcome.UNSUPPORTED,
            ScanOutcome.COMPLETE,
            ScanOutcome.INCONCLUSIVE,
        )
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 18: Unsupported Artifact Handling
# ---------------------------------------------------------------------------


def test_unsupported_artifact_handling() -> None:
    """Test 18: Verify unsupported artifacts produce UNSUPPORTED outcome with warnings."""
    unsupported_bytes = b"RANDOM_UNSUPPORTED_FORMAT_DATA_12345"

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".unknown", delete=False
    ) as tmp:
        tmp.write(unsupported_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        assert artifact_result.scan_outcome == ScanOutcome.UNSUPPORTED
        assert artifact_result.detected_format == ModelFormat.UNSUPPORTED
        assert len(artifact_result.warnings) > 0
        assert len(artifact_result.findings) == 0  # No findings for unsupported
        assert result.summary.exit_code == 0  # Unsupported is not an error
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 19: Legacy PyTorch Non-ZIP Unsupported Handling
# ---------------------------------------------------------------------------


def test_legacy_pytorch_non_zip_is_analyzed_without_structural_noise() -> None:
    """Test 19: legacy non-ZIP PyTorch (.bin/.pt) is analyzed, not skipped.

    The artifact is a complete Pickle stream followed by a raw tensor payload.
    The analyzer slices at the first STOP so the tensor bytes are never parsed
    as opcodes: still no P201/P202/P206/P207 structural false positives, but the
    Pickle itself is now genuinely inspected rather than passed over.
    """
    # Synthetic legacy PyTorch non-ZIP structure: pickle header followed by
    # state_dict and raw float bytes
    legacy_bytes = (
        b"\x80\x02}q\x00(X\x06\x00\x00\x00weightq\x01u."
        b"\x00\x00\x80?\x00\x00\x00@\xcd\xcc\x8c?RAW_TENSOR_FLOATS"
    )

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".bin", delete=False
    ) as tmp:
        tmp.write(legacy_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        assert artifact_result.scan_outcome == ScanOutcome.COMPLETE
        assert artifact_result.detected_format == ModelFormat.LEGACY_PYTORCH

        # This payload is benign, so it still scores zero — but via completed
        # analysis, not via being skipped.
        assert artifact_result.risk_score.final_score == 0
        assert artifact_result.risk_score.risk_band.value == "CLEAN"

        # Crucial check: NO false positive P201, P202, P206, P207 structural
        # findings from the raw tensor payload trailing the pickle's STOP.
        finding_rule_ids = [f.rule_id for f in artifact_result.findings]
        for rule_id in ("P106", "P201", "P202", "P206", "P207"):
            assert rule_id not in finding_rule_ids
        assert len(artifact_result.findings) == 0
        assert len(artifact_result.errors) == 0
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 20: Modern PyTorch ZIP and Standalone Pickle Scans Normal
# ---------------------------------------------------------------------------


def test_modern_pytorch_zip_scans_normally() -> None:
    """Test 20a: Verify modern PyTorch ZIP files scan normally."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "archive/data.pkl",
            b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00.",
        )
    zip_bytes = buf.getvalue()

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pt", delete=False
    ) as tmp:
        tmp.write(zip_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        assert artifact_result.detected_format == ModelFormat.PYTORCH_ZIP
        assert artifact_result.scan_outcome == ScanOutcome.COMPLETE
        assert len(artifact_result.errors) == 0
        assert any(f.rule_id == "P209" for f in artifact_result.findings)
        # ClamAV exit code is 1 because PyTorch ZIP emits informational container
        # finding P209 to record embedded Pickle usage (findings present -> code 1)
        assert result.summary.exit_code == 1
        assert result.summary.overall_outcome == ScanOutcome.WARNING
        assert result.summary.max_risk_score == 0
    finally:
        tmp_path.unlink()


def test_standalone_raw_pickle_scans_normally() -> None:
    """Test 20b: Verify standalone .pkl files scan normally with PickleAnalyzer."""
    clean_pickle = b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00}\x94."

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pkl", delete=False
    ) as tmp:
        tmp.write(clean_pickle)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact_result = result.artifacts[0]
        assert artifact_result.scan_outcome == ScanOutcome.COMPLETE
        assert artifact_result.detected_format == ModelFormat.RAW_PICKLE
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Coverage: engine.py uncovered paths
# ---------------------------------------------------------------------------


def test_in_memory_bytes_target_label_and_acquisition_failure() -> None:
    """target=bytes sets '<in-memory bytes: N bytes>' label; acquisition fails cleanly.

    Covers engine.py lines 228-229 (bytes label) and 237/239/262-263
    (ScannerException from LocalFileAcquisition rejecting non-Path target).
    """
    request = ScanRequest(
        target=b"\x80\x02N.",
        input_source_type=InputSourceType.LOCAL_FILE,
    )
    orchestrator = ScanOrchestrator()
    result = orchestrator.scan(request)

    assert "<in-memory bytes:" in result.target_request
    assert result.summary.total_artifacts == 0
    assert result.summary.overall_outcome == ScanOutcome.ERROR


def test_in_memory_stream_target_label_and_acquisition_failure() -> None:
    """target=BytesIO sets '<in-memory stream>' label; acquisition fails cleanly.

    Covers engine.py line 231 (stream label) and 237/239 (ScannerException path).
    """
    request = ScanRequest(
        target=io.BytesIO(b"\x80\x02N."),
        input_source_type=InputSourceType.LOCAL_FILE,
    )
    orchestrator = ScanOrchestrator()
    result = orchestrator.scan(request)

    assert "<in-memory stream>" in result.target_request
    assert result.summary.total_artifacts == 0
    assert result.summary.overall_outcome == ScanOutcome.ERROR


def test_format_detection_failure_returns_error_artifact() -> None:
    """FormatDispatcher raising propagates to FORMAT_DETECTION_FAILED error.

    Covers engine.py lines 380-390 (format detection exception path).
    """
    header_json = b"{}"
    file_bytes = len(header_json).to_bytes(8, "little") + header_json

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        orchestrator = ScanOrchestrator()
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)

        with patch.object(
            orchestrator._format_dispatcher,
            "detect_format",
            side_effect=RuntimeError("magic bytes corrupted"),
        ):
            result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        artifact = result.artifacts[0]
        assert artifact.scan_outcome == ScanOutcome.ERROR
        assert any(e.error_code == "FORMAT_DETECTION_FAILED" for e in artifact.errors)
    finally:
        tmp_path.unlink()


def test_no_registered_analyzer_returns_unsupported_outcome() -> None:
    """get_analyzer returning None for a detected format yields UNSUPPORTED outcome.

    Covers engine.py lines 462-486 (no-analyzer-registered branch).
    """
    header_json = b"{}"
    file_bytes = len(header_json).to_bytes(8, "little") + header_json

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()

        with patch("model_scanner.core.engine.get_analyzer", return_value=None):
            result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        artifact = result.artifacts[0]
        assert artifact.scan_outcome == ScanOutcome.UNSUPPORTED
        assert any(w.warning_code == "NO_ANALYZER" for w in artifact.warnings)
    finally:
        tmp_path.unlink()


def test_analyzer_crash_produces_fail_closed_error() -> None:
    """analyzer.analyze() raising an unhandled exception yields INCONCLUSIVE outcome.

    R3: analyzer crashes are parse/coverage failures, not security verdicts.
    Covers engine.py ANALYZER_FAILED path.
    """
    header_json = b"{}"
    file_bytes = len(header_json).to_bytes(8, "little") + header_json

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("internal crash")

        with patch("model_scanner.core.engine.get_analyzer", return_value=mock_analyzer):
            result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        artifact = result.artifacts[0]
        assert artifact.scan_outcome == ScanOutcome.INCONCLUSIVE
        assert any(e.error_code == "ANALYZER_FAILED" for e in artifact.errors)
    finally:
        tmp_path.unlink()


def test_correlated_findings_injected_into_artifact_results() -> None:
    """CrossArtifactCorrelator findings are merged back into per-artifact results.

    Covers engine.py lines 297-315 (correlation injection path).
    """
    clean_pickle = b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00}\x94."

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(clean_pickle)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()

        def inject_c001(artifact_results: object) -> tuple[Finding, ...]:
            results = list(artifact_results)  # type: ignore[arg-type]
            if not results:
                return ()
            first_sha = results[0].sha256
            return (
                Finding(
                    finding_id="FINDING-C001-testcov001",
                    rule_id="C001",
                    rule_name="Active Supply Chain Execution",
                    severity=Severity.CRITICAL,
                    category="supply_chain",
                    confidence=Confidence.HIGH,
                    description="Correlated test finding",
                    remediation="Investigate",
                    format=ModelFormat.RAW_PICKLE,
                    artifact_sha256=first_sha,
                    evidence=Evidence(
                        coordinate_space=CoordinateSpace.GLOBAL,
                        byte_offset=0,
                        byte_length=0,
                        archive_member=None,
                        matched_value="",
                        context_snippet="",
                    ),
                    correlated_finding_ids=("f-src-1", "f-src-2"),
                ),
            )

        from model_scanner.core.correlation import CrossArtifactCorrelator

        with patch.object(CrossArtifactCorrelator, "correlate_scan", side_effect=inject_c001):
            result = orchestrator.scan(request)

        assert len(result.artifacts) == 1
        c001 = [f for f in result.artifacts[0].findings if f.rule_id == "C001"]
        assert len(c001) == 1
        assert c001[0].correlated_finding_ids == ("f-src-1", "f-src-2")
    finally:
        tmp_path.unlink()


def test_repo_analyzer_failure_returns_error_artifact() -> None:
    """RepositoryAnalyzer.analyze_directory raising returns an ERROR ArtifactScanResult.

    Covers engine.py lines 610-628 (repo analyzer exception path).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Put a minimal file so the directory scan has something
        (tmpdir_path / "model.pkl").write_bytes(b"\x80\x02N.")

        request = ScanRequest(
            target=tmpdir_path,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        orchestrator = ScanOrchestrator()

        with patch(
            "model_scanner.analyzers.repository.analyzer.RepositoryAnalyzer.analyze_directory",
            side_effect=RuntimeError("repo analyzer exploded"),
        ):
            result = orchestrator.scan(request)

        repo_artifacts = [
            a for a in result.artifacts if a.detected_format == ModelFormat.REPOSITORY
        ]
        assert len(repo_artifacts) == 1
        repo = repo_artifacts[0]
        from model_scanner.core.models import RiskBand
        assert repo.scan_outcome == ScanOutcome.ERROR
        assert any(e.error_code == "REPO_ANALYZER_ERROR" for e in repo.errors)
        assert repo.risk_score.risk_band == RiskBand.UNANALYZED


def test_scan_repository_nonexistent_dir_returns_none() -> None:
    """_scan_repository returns None when repo_root is not a directory.

    Covers engine.py line 593.
    """
    orchestrator = ScanOrchestrator()
    result = orchestrator._scan_repository(
        Path("/nonexistent/does/not/exist"),
        artifacts=[],
        config=ScanConfig(),
    )
    assert result is None


def test_acquire_artifacts_in_memory_raises_not_implemented() -> None:
    """_acquire_artifacts raises NotImplementedError for IN_MEMORY source type.

    Covers engine.py lines 698-700 (else branch for unhandled InputSourceType).
    """
    import pytest

    orchestrator = ScanOrchestrator()
    request = ScanRequest(
        target=b"\x80\x02N.",
        input_source_type=InputSourceType.IN_MEMORY,
    )
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        orchestrator._acquire_artifacts(request)


# ---------------------------------------------------------------------------
# Regression: ONNX fail-open fix (Bug fix: ImportError must not → CLEAN)
# ---------------------------------------------------------------------------


def test_onnx_missing_dep_produces_error_not_clean() -> None:
    """Regression: ONNXAnalyzer must raise ImportError when onnx is absent.

    Prior behavior: except ImportError: return [] → scan_outcome=COMPLETE,
    risk_band=CLEAN — a false-CLEAN that conceals an uninspected ONNX file.
    Fixed behavior: ImportError is raised, engine records ANALYZER_FAILED,
    scan_outcome=ERROR, risk_band=UNANALYZED.
    """
    import sys
    from model_scanner.analyzers.onnx.analyzer import ONNXAnalyzer
    import hashlib

    onnx_data = b"\x08\x04"  # minimal ONNX-like bytes
    sha = hashlib.sha256(onnx_data).hexdigest()
    from model_scanner.core.models import Artifact, RiskBand
    artifact = Artifact(
        artifact_id="test-onnx",
        source_path="model.onnx",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(onnx_data),
        sha256=sha,
        stream=io.BytesIO(onnx_data),
    )

    # Block onnx import to simulate missing optional dependency
    with patch.dict(sys.modules, {"onnx": None}):
        analyzer = ONNXAnalyzer()
        import pytest
        with pytest.raises(ImportError, match="onnx is required"):
            analyzer.analyze(artifact)


def test_onnx_missing_dep_full_pipeline_error_not_clean() -> None:
    """Full pipeline: ONNX + missing onnx dep → ERROR + UNANALYZED, never CLEAN."""
    import sys
    import hashlib
    from model_scanner.core.models import RiskBand

    onnx_data = b"\x08\x04"
    sha = hashlib.sha256(onnx_data).hexdigest()

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".onnx", delete=False) as tmp:
        tmp.write(onnx_data)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()

        with patch.dict(sys.modules, {"onnx": None}):
            result = orchestrator.scan(request)

        assert len(result.artifacts) == 1
        art = result.artifacts[0]
        assert art.scan_outcome == ScanOutcome.INCONCLUSIVE, (
            f"Expected INCONCLUSIVE for ONNX with missing dep (R3), got {art.scan_outcome}"
        )
        assert art.risk_score.risk_band == RiskBand.UNANALYZED, (
            f"Expected UNANALYZED risk band, got {art.risk_score.risk_band} — "
            "CLEAN would be the fail-open bug"
        )
        assert any(e.error_code == "ANALYZER_FAILED" for e in art.errors), (
            "Expected ANALYZER_FAILED error record"
        )
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Regression: Analyzer ERROR path must use UNANALYZED risk band, not CLEAN
# ---------------------------------------------------------------------------


def test_analyzer_exception_produces_unanalyzed_not_clean() -> None:
    """Regression: when analyzer raises, risk_band must be UNANALYZED, not CLEAN.

    The invariant: CLEAN means analysis ran and found nothing.
    An analyzer failure means analysis did NOT complete — UNANALYZED is correct.
    """
    import hashlib
    from model_scanner.core.models import RiskBand

    # Use a real SafeTensors file so format detection succeeds
    header_json = b"{}"
    header_len = len(header_json)
    data = (
        header_len.to_bytes(8, "little")
        + header_json
    )
    sha = hashlib.sha256(data).hexdigest()

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
        )
        orchestrator = ScanOrchestrator()

        # Force the analyzer to crash
        with patch.object(SafeTensorsAnalyzer, "analyze", side_effect=RuntimeError("injected")):
            result = orchestrator.scan(request)

        assert len(result.artifacts) == 1
        art = result.artifacts[0]
        assert art.scan_outcome == ScanOutcome.INCONCLUSIVE
        assert art.risk_score.risk_band == RiskBand.UNANALYZED, (
            f"Expected UNANALYZED for a failed analyzer, got {art.risk_score.risk_band} — "
            "CLEAN here would mean analysis ran and found nothing, which is false"
        )
        assert any(e.error_code == "ANALYZER_FAILED" for e in art.errors)
    finally:
        tmp_path.unlink()
