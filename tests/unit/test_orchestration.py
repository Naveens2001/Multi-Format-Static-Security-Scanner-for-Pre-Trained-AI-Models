"""
tests.unit.test_orchestration
=============================

Unified end-to-end orchestration test suite for Phase 7.5.

Validates the full pipeline:
  Acquisition -> Format Detection -> Analyzer Dispatch -> Finding Normalization
  -> Multi-Signal Correlation -> Risk Scoring -> ScanResult Assembly -> JSON Export

Test Coverage:
1. Clean SafeTensors end-to-end scan (COMPLETE, CLEAN, score=0, ClamAV=0)
2. Malicious raw Pickle end-to-end scan (P101, score>=60, ClamAV=1)
3. PyTorch ZIP with embedded Pickle (archive_member coords, format multiplier 1.3x)
4. GGUF binary format scan (metadata parsing, format multiplier 1.1x)
5. Format spoofing detection and G001 correlation (extension mismatch -> G001)
6. Malformed artifact fail-closed behavior (PROTO desync -> ERROR, ClamAV=2)
7. Unsupported format graceful handling (UNSUPPORTED outcome, 0 findings, ClamAV=0)
8. Mixed directory scan (clean, infected, and unsupported artifacts aggregated)
9. In-memory stream scan (IN_MEMORY source type, zero filesystem touch)
10. Strict-mode behavior (P103 escalated from MEDIUM to HIGH)
11. ClamAV exit-code behavior matrix (0 Clean, 1 Infected, 2 Fatal Error)
12. Full ScanOrchestrator -> JSONReportExporter pipeline flow
13. Deterministic JSON output and schema validation
14. Finding normalization and evidence coordinates verification
15. Risk score and risk band propagation across pipeline
16. ScanSummary aggregation metrics verification
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
import zipfile
from pathlib import Path

from model_scanner.analyzers.gguf.reader import GGUF_MAGIC, GGUFValueType
from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import (
    Artifact,
    CoordinateSpace,
    InputSourceType,
    ModelFormat,
    RiskBand,
    ScanConfig,
    ScanOutcome,
    ScanRequest,
    Severity,
)
from model_scanner.presentation.json_export import JSONReportExporter

# ---------------------------------------------------------------------------
# Synthetic Test Fixture Helpers
# ---------------------------------------------------------------------------


def _create_clean_safetensors_bytes() -> bytes:
    """Construct a minimal valid, clean SafeTensors binary byte sequence."""
    header_json = b"{}"
    header_len = len(header_json)
    return header_len.to_bytes(8, byteorder="little") + header_json


def _create_malicious_pickle_bytes() -> bytes:
    """Construct a raw Pickle protocol 3 stream importing dangerous posix.system."""
    return b"\x80\x03cposix\nsystem\n."


def _create_stack_global_pickle_bytes() -> bytes:
    """Construct a raw Pickle protocol 4 stream with STACK_GLOBAL opcode (triggers P103)."""
    return b"\x80\x04\x8c\x0bunknown_mod\x8c\x0cunknown_func\x93."


def _create_pytorch_zip_bytes(inner_pickle: bytes) -> bytes:
    """Construct a containerized PyTorch ZIP archive with an embedded Pickle stream."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", inner_pickle)
    return buf.getvalue()


def _create_clean_gguf_bytes() -> bytes:
    """Construct a minimal valid synthetic GGUF v3 binary."""
    magic = GGUF_MAGIC
    version = struct.pack("<I", 3)
    counts = struct.pack("<QQ", 0, 1)  # 0 tensors, 1 KV pair
    header = magic + version + counts

    # KV: "general.architecture" -> "llama"
    key = b"general.architecture"
    val = b"llama"
    header += struct.pack("<Q", len(key)) + key
    header += struct.pack("<I", GGUFValueType.STRING)
    header += struct.pack("<Q", len(val)) + val

    # Alignment padding (32 bytes)
    pad_len = (32 - (len(header) % 32)) % 32
    return header + bytes(pad_len)


# ---------------------------------------------------------------------------
# Test 1: Clean SafeTensors Scan
# ---------------------------------------------------------------------------


def test_clean_safetensors_end_to_end() -> None:
    """Verify clean SafeTensors scan yields COMPLETE, score=0, CLEAN band, ClamAV=0."""
    content = _create_clean_safetensors_bytes()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        assert result.summary.scanned_successfully == 1
        assert result.summary.failed_artifacts == 0
        assert result.summary.max_risk_score == 0
        assert result.summary.overall_outcome == ScanOutcome.COMPLETE
        assert result.summary.exit_code == 0

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.SAFETENSORS
        assert artifact.scan_outcome == ScanOutcome.COMPLETE
        assert len(artifact.findings) == 0
        assert artifact.risk_score.final_score == 0
        assert artifact.risk_score.risk_band == RiskBand.CLEAN
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 2: Malicious Raw Pickle Scan
# ---------------------------------------------------------------------------


def test_malicious_raw_pickle_end_to_end() -> None:
    """Verify malicious raw Pickle yields P101 finding, risk score >= 60, ClamAV=1."""
    content = _create_malicious_pickle_bytes()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        assert result.summary.scanned_successfully == 1
        # When security findings are present, overall summary outcome is WARNING
        assert result.summary.overall_outcome == ScanOutcome.WARNING
        assert result.summary.exit_code == 1
        assert result.summary.findings_by_severity["CRITICAL"] >= 1

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.RAW_PICKLE
        assert artifact.scan_outcome == ScanOutcome.COMPLETE
        assert len(artifact.findings) >= 1

        finding = artifact.findings[0]
        assert finding.rule_id == "P101"
        assert finding.severity == Severity.CRITICAL
        assert finding.category == "code_execution"
        assert finding.evidence.coordinate_space == CoordinateSpace.ARTIFACT
        assert finding.evidence.byte_offset is not None

        # 40.0 * 1.5 format multiplier = 60.0 -> numeric score 60, but CRITICAL
        # severity floor (R5) elevates the band to CRITICAL_RISK.
        assert artifact.risk_score.final_score >= 60
        assert artifact.risk_score.risk_band == RiskBand.CRITICAL_RISK
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 3: PyTorch ZIP Container with Embedded Pickle
# ---------------------------------------------------------------------------


def test_pytorch_zip_with_embedded_pickle() -> None:
    """Verify PyTorch ZIP sets archive coordinate space and 1.3x format multiplier."""
    inner_pickle = _create_malicious_pickle_bytes()
    zip_bytes = _create_pytorch_zip_bytes(inner_pickle)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as tmp:
        tmp.write(zip_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        assert result.summary.exit_code == 1

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.PYTORCH_ZIP
        assert len(artifact.findings) >= 1

        finding = artifact.findings[0]
        assert finding.evidence.coordinate_space == CoordinateSpace.ARCHIVE_MEMBER
        assert finding.evidence.archive_member == "archive/data.pkl"

        # PyTorch format multiplier is 1.3
        assert artifact.risk_score.format_multiplier == 1.3
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 4: GGUF Binary Scanning
# ---------------------------------------------------------------------------


def test_gguf_binary_scanning() -> None:
    """Verify valid GGUF binary format is scanned with 1.1x format multiplier."""
    gguf_bytes = _create_clean_gguf_bytes()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".gguf", delete=False) as tmp:
        tmp.write(gguf_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.GGUF
        assert artifact.scan_outcome == ScanOutcome.COMPLETE
        assert artifact.risk_score.format_multiplier == 1.1
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 5: Format Spoofing and G001 Correlation
# ---------------------------------------------------------------------------


def test_format_spoofing_detection_and_g001_correlation() -> None:
    """Verify .safetensors file containing Pickle bytes triggers G001 + P101."""
    pickle_bytes = _create_malicious_pickle_bytes()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(pickle_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        artifact = result.artifacts[0]
        # Magic sniff routes to RAW_PICKLE
        assert artifact.detected_format == ModelFormat.RAW_PICKLE

        # Both G001 and P101 should be present
        rule_ids = [f.rule_id for f in artifact.findings]
        assert "G001" in rule_ids
        assert "P101" in rule_ids

        # G001 (20.0) + P101 (40.0) = 60.0 * 1.5 = 90 -> CRITICAL_RISK
        assert artifact.risk_score.final_score >= 90
        assert artifact.risk_score.risk_band == RiskBand.CRITICAL_RISK
        assert result.summary.exit_code == 1
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 6: Malformed Artifact Fail-Closed Behavior
# ---------------------------------------------------------------------------


def test_malformed_artifact_fail_closed() -> None:
    """Verify malformed SafeTensors (truncated header) yields INCONCLUSIVE and ClamAV=2.

    R3: parse/format errors produce INCONCLUSIVE per-artifact (not a security verdict),
    but still count as failed_artifacts so summary stays ERROR with exit code 2.
    """
    # Truncated SafeTensors: claims 1000 byte header but has only 10 bytes
    bad_bytes = (1000).to_bytes(8, byteorder="little") + b'{"test":'
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(bad_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.total_artifacts == 1
        assert result.summary.failed_artifacts == 1
        assert result.summary.overall_outcome == ScanOutcome.ERROR
        assert result.summary.exit_code == 2

        artifact = result.artifacts[0]
        assert artifact.scan_outcome == ScanOutcome.INCONCLUSIVE
        assert len(artifact.errors) >= 1
        assert artifact.errors[0].error_code == "MALFORMED_ARTIFACT"
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 7: Unsupported Format Handling
# ---------------------------------------------------------------------------


def test_unsupported_format_handling() -> None:
    """Verify unsupported file format yields UNSUPPORTED outcome and ClamAV=0."""
    random_bytes = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".xyz_unknown", delete=False) as tmp:
        tmp.write(random_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.summary.unsupported_artifacts == 1
        assert result.summary.overall_outcome == ScanOutcome.UNSUPPORTED
        assert result.summary.exit_code == 0

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.UNSUPPORTED
        assert artifact.scan_outcome == ScanOutcome.UNSUPPORTED
        assert len(artifact.warnings) >= 1
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 7b: Identified-but-Unanalyzed Format Handling (A5 / G002)
# ---------------------------------------------------------------------------


def test_identified_unanalyzed_format_hdf5() -> None:
    """HDF5 magic bytes → detected as HDF5, analyzed by KerasAnalyzer (P2)."""
    hdf5_magic = b"\x89HDF\r\n\x1a\n" + b"\x00" * 56
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".h5", delete=False) as tmp:
        tmp.write(hdf5_magic)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        result = ScanOrchestrator().scan(request)

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.HDF5
        # P2: HDF5 now has a KerasAnalyzer — no longer unanalyzed (no G002)
        assert artifact.scan_outcome in (ScanOutcome.COMPLETE, ScanOutcome.ERROR)
        rule_ids = [f.rule_id for f in artifact.findings]
        assert "G002" not in rule_ids, f"G002 should not appear for analyzed HDF5: {rule_ids}"
    finally:
        tmp_path.unlink()


def test_identified_unanalyzed_format_numpy_npy() -> None:
    """NumPy .npy magic bytes → WARNING outcome + G002 INFO finding."""
    npy_magic = b"\x93NUMPY" + b"\x00" * 58
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".npy", delete=False) as tmp:
        tmp.write(npy_magic)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        result = ScanOrchestrator().scan(request)

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.NUMPY_NPY
        assert artifact.scan_outcome == ScanOutcome.WARNING
        assert any(f.rule_id == "G002" for f in artifact.findings)
    finally:
        tmp_path.unlink()


def test_identified_unanalyzed_format_tflite() -> None:
    """TFLite FlatBuffer identifier at offset 4 → WARNING outcome + G002 INFO finding."""
    tflite_bytes = b"\x00\x00\x00\x00" + b"TFL3" + b"\x00" * 56
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".tflite", delete=False) as tmp:
        tmp.write(tflite_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        result = ScanOrchestrator().scan(request)

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.TFLITE
        assert artifact.scan_outcome == ScanOutcome.WARNING
        assert any(f.rule_id == "G002" for f in artifact.findings)
    finally:
        tmp_path.unlink()


def test_identified_unanalyzed_format_onnx_by_extension() -> None:
    """ONNX file detected by extension → analyzed by ONNXAnalyzer (P2)."""
    arbitrary_bytes = b"\x08\x01\x12\x04test" + b"\x00" * 55
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".onnx", delete=False) as tmp:
        tmp.write(arbitrary_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        result = ScanOrchestrator().scan(request)

        artifact = result.artifacts[0]
        assert artifact.detected_format == ModelFormat.ONNX
        # P2: ONNX now has an ONNXAnalyzer — no longer unanalyzed (no G002)
        assert artifact.scan_outcome in (ScanOutcome.COMPLETE, ScanOutcome.ERROR)
        assert not any(f.rule_id == "G002" for f in artifact.findings), \
            "G002 should not appear for analyzed ONNX format"
    finally:
        tmp_path.unlink()


def test_g002_finding_has_catalog_entry() -> None:
    """G002 rule ID emitted by the engine must exist in the RuleCatalog."""
    from model_scanner.rules.catalog import RuleCatalog

    assert RuleCatalog.has_rule("G002"), "G002 must be registered in RuleCatalog"
    rule = RuleCatalog.get_rule("G002")
    assert rule.severity == Severity.INFO
    assert rule.category == "format_integrity"


# ---------------------------------------------------------------------------
# Test 8: Mixed Artifacts Aggregation & Directory Scanning
# ---------------------------------------------------------------------------


def test_mixed_artifacts_aggregation() -> None:
    """Verify aggregate summary across clean, malicious, and genuinely unsupported artifacts."""
    clean_bytes = _create_clean_safetensors_bytes()
    pickle_bytes = _create_malicious_pickle_bytes()
    unsupported_bytes = b"UNKNOWN_BINARY_FORMAT_DATA_12345"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p_clean = tmp_path / "model_a.safetensors"
        p_infected = tmp_path / "model_b.pkl"
        p_unsupported = tmp_path / "model_c.xyz_unknown"

        p_clean.write_bytes(clean_bytes)
        p_infected.write_bytes(pickle_bytes)
        p_unsupported.write_bytes(unsupported_bytes)

        orchestrator = ScanOrchestrator()
        r1 = orchestrator.scan(
            ScanRequest(target=p_clean, input_source_type=InputSourceType.LOCAL_FILE)
        )
        r2 = orchestrator.scan(
            ScanRequest(target=p_infected, input_source_type=InputSourceType.LOCAL_FILE)
        )
        r3 = orchestrator.scan(
            ScanRequest(target=p_unsupported, input_source_type=InputSourceType.LOCAL_FILE)
        )

        art1 = r1.artifacts[0]
        art2 = r2.artifacts[0]
        art3 = r3.artifacts[0]

        summary = orchestrator._build_summary((art1, art2, art3))

        assert summary.total_artifacts == 3
        assert summary.scanned_successfully == 2
        assert summary.unsupported_artifacts == 1
        assert summary.failed_artifacts == 0
        assert summary.overall_outcome == ScanOutcome.WARNING
        assert summary.exit_code == 1
        assert summary.max_risk_score >= 60


def test_directory_scan_supported_artifacts() -> None:
    """Verify recursive directory scan discovers supported files and skips non-models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)

        (dir_path / "model_a.safetensors").write_bytes(_create_clean_safetensors_bytes())
        (dir_path / "model_b.pkl").write_bytes(_create_malicious_pickle_bytes())
        (dir_path / "readme.txt").write_bytes(b"non-model text")

        request = ScanRequest(
            target=dir_path,
            input_source_type=InputSourceType.LOCAL_DIRECTORY,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        # 2 weight-file artifacts + 1 repository-level supply-chain result
        assert result.summary.total_artifacts == 3
        assert result.summary.scanned_successfully == 3
        assert result.summary.failed_artifacts == 0
        assert result.summary.exit_code == 1
        assert result.summary.max_risk_score >= 60

        # Filter to weight-file artifacts only for name check
        from model_scanner.core.models import ModelFormat
        weight_artifacts = [
            a for a in result.artifacts if a.detected_format != ModelFormat.REPOSITORY
        ]
        names = [Path(a.source_origin).name for a in weight_artifacts]
        assert names == ["model_a.safetensors", "model_b.pkl"]


# ---------------------------------------------------------------------------
# Test 9: In-Memory Stream Scan
# ---------------------------------------------------------------------------


def test_in_memory_stream_scan() -> None:
    """Verify scanning in-memory byte stream directly without filesystem persistence."""
    stream_content = _create_clean_safetensors_bytes()
    stream = io.BytesIO(stream_content)

    artifact = Artifact(
        artifact_id="mem-001",
        source_path="memory://clean_model.safetensors",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(stream_content),
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        stream=stream,
    )
    orchestrator = ScanOrchestrator()
    artifact_result = orchestrator.scan_artifact(artifact)

    assert artifact_result.input_source == InputSourceType.IN_MEMORY
    assert artifact_result.detected_format == ModelFormat.SAFETENSORS
    assert artifact_result.scan_outcome == ScanOutcome.COMPLETE
    assert artifact_result.risk_score.final_score == 0
    assert artifact_result.risk_score.risk_band == RiskBand.CLEAN


# ---------------------------------------------------------------------------
# Test 10: Strict Mode Escalation
# ---------------------------------------------------------------------------


def test_strict_mode_escalation_across_pipeline() -> None:
    """Verify strict mode promotes P103 unknown global finding from MEDIUM to HIGH."""
    pickle_bytes = _create_stack_global_pickle_bytes()

    # Normal mode
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(pickle_bytes)
        tmp_path = Path(tmp.name)

    try:
        req_normal = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        res_normal = ScanOrchestrator().scan(req_normal)
        art_normal = res_normal.artifacts[0]
        p103_normal = next((f for f in art_normal.findings if f.rule_id == "P103"), None)
        assert p103_normal is not None
        assert p103_normal.severity == Severity.MEDIUM

        # Strict mode
        config_strict = ScanConfig(strict_mode=True)
        req_strict = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
            config=config_strict,
        )
        res_strict = ScanOrchestrator().scan(req_strict)
        art_strict = res_strict.artifacts[0]
        p103_strict = next((f for f in art_strict.findings if f.rule_id == "P103"), None)
        assert p103_strict is not None
        assert p103_strict.severity == Severity.HIGH

        # Strict mode score must be strictly greater than normal mode score
        assert art_strict.risk_score.final_score > art_normal.risk_score.final_score
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test 11: ClamAV Exit-Code Verification Matrix
# ---------------------------------------------------------------------------


def test_exit_code_matrix() -> None:
    """Verify ClamAV exit codes 0 (Clean), 1 (Infected), and 2 (Error) across cases."""
    orchestrator = ScanOrchestrator()

    # 1. Clean -> 0
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write(_create_clean_safetensors_bytes())
        p_clean = Path(tmp.name)

    # 2. Infected -> 1
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(_create_malicious_pickle_bytes())
        p_infected = Path(tmp.name)

    # 3. Error -> 2
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".safetensors", delete=False) as tmp:
        tmp.write((1000).to_bytes(8, byteorder="little") + b"{}")
        p_error = Path(tmp.name)

    try:
        r0 = orchestrator.scan(
            ScanRequest(target=p_clean, input_source_type=InputSourceType.LOCAL_FILE)
        )
        assert r0.summary.exit_code == 0

        r1 = orchestrator.scan(
            ScanRequest(target=p_infected, input_source_type=InputSourceType.LOCAL_FILE)
        )
        assert r1.summary.exit_code == 1

        r2 = orchestrator.scan(
            ScanRequest(target=p_error, input_source_type=InputSourceType.LOCAL_FILE)
        )
        assert r2.summary.exit_code == 2
    finally:
        p_clean.unlink()
        p_infected.unlink()
        p_error.unlink()


# ---------------------------------------------------------------------------
# Test 12 & 13: Full Pipeline Scan -> JSON Export & Determinism
# ---------------------------------------------------------------------------


def test_full_pipeline_scan_to_json_export() -> None:
    """Verify complete pipeline from ScanOrchestrator to JSONReportExporter."""
    pickle_bytes = _create_malicious_pickle_bytes()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(pickle_bytes)
        tmp_path = Path(tmp.name)

    try:
        request = ScanRequest(target=tmp_path, input_source_type=InputSourceType.LOCAL_FILE)
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        # Export to JSON
        json_str1 = JSONReportExporter.to_json(result, indent=2)
        json_str2 = JSONReportExporter.to_json(result, indent=2)

        # Deterministic byte-for-byte reproducibility
        assert json_str1 == json_str2

        data = json.loads(json_str1)
        assert "$schema" not in data, "$schema removed — URL pointed to non-existent repo"
        assert data["summary"]["exit_code"] == 1
        assert data["summary"]["findings_by_severity"]["CRITICAL"] >= 1
        assert len(data["artifacts"]) == 1
        assert data["artifacts"][0]["detected_format"] == "RAW_PICKLE"
        assert len(data["artifacts"][0]["findings"]) >= 1
        assert data["artifacts"][0]["findings"][0]["rule_id"] == "P101"
    finally:
        tmp_path.unlink()
