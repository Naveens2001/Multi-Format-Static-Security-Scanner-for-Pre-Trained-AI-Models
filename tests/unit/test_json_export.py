"""
tests.unit.test_json_export
===========================

Unit tests for JSONReportExporter (Phase 7.4).

Test Coverage:
1. Clean ScanResult produces canonical JSON conforming to Section 10 schema
2. ScanResult containing findings serializes all finding & evidence fields
3. Multiple artifacts serialized in deterministic order
4. All enum values converted to canonical strings
5. Nullable evidence & error fields handled properly
6. Summary metrics (findings_by_severity, artifacts_by_risk_band, exit_code)
7. RiskScore nested dictionary serialization
8. ScanWarning serialization
9. ScanError serialization
10. $schema URI construction and version formatting
11. json.loads() round-trip validation
12. File export to Path and string path
13. TextIO (StringIO) stream export
14. Deterministic repeated serialization
15. End-to-end ScanOrchestrator -> JSONReportExporter pipeline
16. Invalid input validation (TypeError on non-ScanResult and invalid destination)
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import (
    ArtifactScanResult,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    InputSourceType,
    ModelFormat,
    RiskBand,
    RiskScore,
    ScanError,
    ScanOutcome,
    ScanRequest,
    ScanResult,
    ScanSummary,
    ScanWarning,
    Severity,
)
from model_scanner.presentation.json_export import (
    JSONReportExporter,
    JsonReportExporter,
)


def _make_sample_scan_result(
    findings: list[Finding] | None = None,
    warnings: list[ScanWarning] | None = None,
    errors: list[ScanError] | None = None,
    detected_format: ModelFormat = ModelFormat.RAW_PICKLE,
    scan_outcome: ScanOutcome = ScanOutcome.COMPLETE,
) -> ScanResult:
    """Construct a valid mock ScanResult for testing JSON serialization."""
    sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    actual_findings = tuple(findings or [])
    actual_warnings = tuple(warnings or [])
    actual_errors = tuple(errors or [])

    risk_score = RiskScore(
        raw_score=60.0 if actual_findings else 0.0,
        final_score=60 if actual_findings else 0,
        risk_band=RiskBand.MEDIUM_RISK if actual_findings else RiskBand.CLEAN,
        format_multiplier=1.5 if detected_format == ModelFormat.RAW_PICKLE else 1.0,
        obfuscation_multiplier=1.0,
        findings_weighted_sum=40.0 if actual_findings else 0.0,
    )

    artifact = ArtifactScanResult(
        artifact_id="3a7b1c4e-1234-5678-9abc-def012345678",
        name="test_model.pkl",
        source_origin="models/test_model.pkl",
        input_source=InputSourceType.LOCAL_FILE,
        detected_format=detected_format,
        size_bytes=1024,
        sha256=sha256,
        scan_outcome=scan_outcome,
        risk_score=risk_score,
        findings=actual_findings,
        warnings=actual_warnings,
        errors=actual_errors,
        duration_ms=42,
    )

    findings_by_severity = {
        "CRITICAL": sum(1 for f in actual_findings if f.severity == Severity.CRITICAL),
        "HIGH": sum(1 for f in actual_findings if f.severity == Severity.HIGH),
        "MEDIUM": sum(1 for f in actual_findings if f.severity == Severity.MEDIUM),
        "LOW": sum(1 for f in actual_findings if f.severity == Severity.LOW),
        "INFO": sum(1 for f in actual_findings if f.severity == Severity.INFO),
    }

    artifacts_by_risk_band = {
        "CLEAN": 0 if actual_findings else 1,
        "LOW_RISK": 0,
        "MEDIUM_RISK": 1 if actual_findings else 0,
        "HIGH_RISK": 0,
        "CRITICAL_RISK": 0,
    }

    summary = ScanSummary(
        total_artifacts=1,
        scanned_successfully=1 if scan_outcome == ScanOutcome.COMPLETE else 0,
        failed_artifacts=1 if scan_outcome == ScanOutcome.ERROR else 0,
        unsupported_artifacts=1 if detected_format == ModelFormat.UNSUPPORTED else 0,
        findings_by_severity=findings_by_severity,
        artifacts_by_risk_band=artifacts_by_risk_band,
        max_risk_score=risk_score.final_score,
        overall_outcome=scan_outcome,
        exit_code=1 if actual_findings else 0,
    )

    return ScanResult(
        scan_id="9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        scanner_version="2.0.0",
        timestamp_utc="2026-08-17T12:00:00Z",
        target_request="models/test_model.pkl",
        strict_mode=False,
        artifacts=(artifact,),
        summary=summary,
        total_duration_ms=42,
    )


# ---------------------------------------------------------------------------
# Test 1: Clean ScanResult
# ---------------------------------------------------------------------------


def test_clean_scan_result_to_dict() -> None:
    """Verify serialization of a clean scan result with no findings."""
    scan_result = _make_sample_scan_result(findings=[])
    data = JSONReportExporter.to_dict(scan_result)

    assert "$schema" not in data, "$schema removed — URL pointed to non-existent repo"
    assert data["scan_id"] == "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"
    assert data["scanner_version"] == "2.0.0"
    assert data["strict_mode"] is False
    assert data["summary"]["total_artifacts"] == 1
    assert data["summary"]["exit_code"] == 0
    assert data["summary"]["overall_outcome"] == "COMPLETE"
    assert len(data["artifacts"]) == 1
    assert data["artifacts"][0]["findings"] == []
    assert data["artifacts"][0]["risk_score"]["final_score"] == 0
    assert data["artifacts"][0]["risk_score"]["risk_band"] == "CLEAN"


# ---------------------------------------------------------------------------
# Test 2: ScanResult with Findings
# ---------------------------------------------------------------------------


def test_scan_result_with_findings_to_dict() -> None:
    """Verify serialization of a finding with full evidence payload."""
    finding = Finding(
        finding_id="FINDING-P101-e3b0c442-001",
        rule_id="P101",
        rule_name="Dangerous Global Import",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=42,
            byte_length=18,
            archive_member=None,
            matched_value="posix.system",
            opcode="GLOBAL",
            raw_hex="63706f7369780a73797374656d0a",
            context_snippet="cposix\nsystem\n",
        ),
        description="Pickle imports dangerous callable",
        remediation="Do not deserialize this model.",
        cve_ids=(),
    )

    scan_result = _make_sample_scan_result(findings=[finding])
    data = JSONReportExporter.to_dict(scan_result)

    assert len(data["artifacts"][0]["findings"]) == 1
    f_dict = data["artifacts"][0]["findings"][0]

    assert f_dict["finding_id"] == "FINDING-P101-e3b0c442-001"
    assert f_dict["rule_id"] == "P101"
    assert f_dict["severity"] == "CRITICAL"
    assert f_dict["confidence"] == "HIGH"
    assert f_dict["category"] == "code_execution"
    assert f_dict["format"] == "RAW_PICKLE"
    assert f_dict["cve_ids"] == []

    ev_dict = f_dict["evidence"]
    assert ev_dict is not None
    assert ev_dict["coordinate_space"] == "artifact"
    assert ev_dict["byte_offset"] == 42
    assert ev_dict["byte_length"] == 18
    assert ev_dict["archive_member"] is None
    assert ev_dict["matched_value"] == "posix.system"
    assert ev_dict["opcode"] == "GLOBAL"
    assert ev_dict["raw_hex"] == "63706f7369780a73797374656d0a"


# ---------------------------------------------------------------------------
# Test 3: Multiple Artifacts
# ---------------------------------------------------------------------------


def test_multi_artifact_serialization() -> None:
    """Verify multi-artifact scan results serialize all artifacts in order."""
    base = _make_sample_scan_result()
    art1 = base.artifacts[0]
    art2 = ArtifactScanResult(
        artifact_id="4b8c2d5f-2345-6789-abcd-ef0123456789",
        name="model2.safetensors",
        source_origin="models/model2.safetensors",
        input_source=InputSourceType.LOCAL_FILE,
        detected_format=ModelFormat.SAFETENSORS,
        size_bytes=2048,
        sha256="a" * 64,
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=RiskScore.clean(),
        findings=(),
        warnings=(),
        errors=(),
        duration_ms=10,
    )

    multi_result = ScanResult(
        scan_id=base.scan_id,
        scanner_version=base.scanner_version,
        timestamp_utc=base.timestamp_utc,
        target_request="models/",
        strict_mode=base.strict_mode,
        artifacts=(art1, art2),
        summary=base.summary,
        total_duration_ms=52,
    )

    data = JSONReportExporter.to_dict(multi_result)
    assert len(data["artifacts"]) == 2
    assert data["artifacts"][0]["name"] == "test_model.pkl"
    assert data["artifacts"][1]["name"] == "model2.safetensors"
    assert data["artifacts"][1]["detected_format"] == "SAFETENSORS"


# ---------------------------------------------------------------------------
# Test 4: Enum Conversions
# ---------------------------------------------------------------------------


def test_all_enum_conversions() -> None:
    """Verify all enums serialize to their standard string representations."""
    finding = Finding(
        finding_id="FINDING-ST201-00000000-001",
        rule_id="ST201",
        rule_name="Header Size Exceeded",
        severity=Severity.HIGH,
        confidence=Confidence.MEDIUM,
        category="structural_anomaly",
        format=ModelFormat.SAFETENSORS,
        artifact_sha256="0" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.HEADER,
            byte_offset=0,
            byte_length=8,
            archive_member=None,
            matched_value="size_exceeded",
        ),
        description="Header exceeds safe size budget",
        remediation="Ensure header size < 100MB",
    )

    scan_result = _make_sample_scan_result(
        findings=[finding],
        detected_format=ModelFormat.SAFETENSORS,
    )
    data = JSONReportExporter.to_dict(scan_result)

    assert data["artifacts"][0]["input_source"] == "LOCAL_FILE"
    assert data["artifacts"][0]["detected_format"] == "SAFETENSORS"
    assert data["artifacts"][0]["scan_outcome"] == "COMPLETE"
    assert data["artifacts"][0]["findings"][0]["severity"] == "HIGH"
    assert data["artifacts"][0]["findings"][0]["confidence"] == "MEDIUM"
    assert data["artifacts"][0]["findings"][0]["evidence"]["coordinate_space"] == "header"


# ---------------------------------------------------------------------------
# Test 5: Nullable Fields Handling
# ---------------------------------------------------------------------------


def test_nullable_evidence_and_errors() -> None:
    """Verify None values remain JSON null in serialized output."""
    finding = Finding(
        finding_id="FINDING-P101-00000000-001",
        rule_id="P101",
        rule_name="Test Rule",
        severity=Severity.LOW,
        confidence=Confidence.LOW,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256="0" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=None,
            byte_length=None,
            archive_member=None,
            matched_value="test",
            opcode=None,
            raw_hex=None,
            context_snippet=None,
        ),
        description="Test description",
        remediation="Test remediation",
    )

    scan_result = _make_sample_scan_result(findings=[finding])
    data = JSONReportExporter.to_dict(scan_result)
    ev = data["artifacts"][0]["findings"][0]["evidence"]

    assert ev["byte_offset"] is None
    assert ev["byte_length"] is None
    assert ev["archive_member"] is None
    assert ev["opcode"] is None
    assert ev["raw_hex"] is None
    assert ev["context_snippet"] is None


# ---------------------------------------------------------------------------
# Test 6: Warning & Error Serialization
# ---------------------------------------------------------------------------


def test_warning_and_error_serialization() -> None:
    """Verify warnings and errors are serialized into their respective arrays."""
    warning = ScanWarning(
        warning_code="FORMAT_MISMATCH",
        message="Extension mismatch",
        context="Detected RAW_PICKLE magic",
    )
    error = ScanError(
        error_code="MALFORMED_ARTIFACT",
        message="Header parsing failed",
        exception_type="SafeTensorsAnalysisError",
        traceback_summary="Traceback: line 10...",
    )

    scan_result = _make_sample_scan_result(
        warnings=[warning],
        errors=[error],
        scan_outcome=ScanOutcome.ERROR,
    )
    data = JSONReportExporter.to_dict(scan_result)

    art = data["artifacts"][0]
    assert len(art["warnings"]) == 1
    assert art["warnings"][0]["warning_code"] == "FORMAT_MISMATCH"
    assert art["warnings"][0]["context"] == "Detected RAW_PICKLE magic"

    assert len(art["errors"]) == 1
    assert art["errors"][0]["error_code"] == "MALFORMED_ARTIFACT"
    assert art["errors"][0]["exception_type"] == "SafeTensorsAnalysisError"
    assert art["errors"][0]["traceback_summary"] == "Traceback: line 10..."


# ---------------------------------------------------------------------------
# Test 7: json.loads() Round-Trip
# ---------------------------------------------------------------------------


def test_json_loads_round_trip() -> None:
    """Verify to_json produces valid parseable JSON identical to to_dict."""
    scan_result = _make_sample_scan_result()
    json_str = JSONReportExporter.to_json(scan_result, indent=2)

    parsed = json.loads(json_str)
    expected = JSONReportExporter.to_dict(scan_result)

    assert parsed == expected


# ---------------------------------------------------------------------------
# Test 8: File Export
# ---------------------------------------------------------------------------


def test_export_to_file_path() -> None:
    """Verify export() correctly writes UTF-8 JSON to a file path."""
    scan_result = _make_sample_scan_result()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "nested" / "report.json"
        exported_str = JSONReportExporter.export(scan_result, out_file)

        assert out_file.exists()
        disk_content = out_file.read_text(encoding="utf-8")
        assert disk_content == exported_str

        parsed = json.loads(disk_content)
        assert parsed["scan_id"] == scan_result.scan_id


# ---------------------------------------------------------------------------
# Test 9: Stream Export
# ---------------------------------------------------------------------------


def test_export_to_string_io() -> None:
    """Verify export() correctly writes JSON to a TextIO stream."""
    scan_result = _make_sample_scan_result()
    stream = io.StringIO()

    exported_str = JSONReportExporter.export(scan_result, stream)
    stream_content = stream.getvalue()

    assert stream_content == exported_str
    parsed = json.loads(stream_content)
    assert parsed["scanner_version"] == "2.0.0"


# ---------------------------------------------------------------------------
# Test 10: Determinism
# ---------------------------------------------------------------------------


def test_deterministic_serialization() -> None:
    """Verify multiple export calls on same ScanResult produce identical output."""
    scan_result = _make_sample_scan_result()

    json1 = JSONReportExporter.to_json(scan_result)
    json2 = JSONReportExporter.to_json(scan_result)

    assert json1 == json2


# ---------------------------------------------------------------------------
# Test 11: Invalid Input Handling
# ---------------------------------------------------------------------------


def test_invalid_exporter_input_raises_type_error() -> None:
    """Verify passing non-ScanResult objects raises TypeError."""
    with pytest.raises(TypeError, match=r"Expected ScanResult"):
        JSONReportExporter.to_dict("not a scan result")  # type: ignore[arg-type]

    scan_result = _make_sample_scan_result()
    with pytest.raises(TypeError, match=r"Destination must be a str, Path, or writable TextIO"):
        JSONReportExporter.export(scan_result, 12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 12: Backward Compatibility Alias
# ---------------------------------------------------------------------------


def test_json_report_exporter_alias() -> None:
    """Verify JsonReportExporter is an exact alias of JSONReportExporter."""
    assert JsonReportExporter is JSONReportExporter


# ---------------------------------------------------------------------------
# Test 13: End-to-End ScanOrchestrator -> JSONReportExporter Pipeline
# ---------------------------------------------------------------------------


def test_end_to_end_orchestrator_to_json_export() -> None:
    """Verify end-to-end pipeline: scan artifact, then export to JSON."""
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

        # Export result to JSON
        json_output = JSONReportExporter.to_json(result)
        data = json.loads(json_output)

        assert data["summary"]["total_artifacts"] == 1
        assert data["summary"]["exit_code"] == 1
        assert data["artifacts"][0]["detected_format"] == "RAW_PICKLE"
        assert len(data["artifacts"][0]["findings"]) >= 1
        assert data["artifacts"][0]["findings"][0]["rule_id"].startswith("P1")
        assert data["artifacts"][0]["risk_score"]["final_score"] >= 60
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test: correlated_finding_ids in JSON serialization
# ---------------------------------------------------------------------------


def test_finding_with_correlated_ids_serialized() -> None:
    """correlated_finding_ids must appear in the JSON output for correlated findings."""
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    finding = Finding(
        finding_id="FINDING-C001-e3b0c442-001",
        rule_id="C001",
        rule_name="Active Supply Chain Execution Intent",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        category="supply_chain",
        format=ModelFormat.REPOSITORY,
        artifact_sha256=sha,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.GLOBAL,
            byte_offset=0,
            byte_length=0,
            archive_member=None,
            matched_value="SC102,SC101",
        ),
        description="Correlated supply chain execution finding.",
        remediation="Investigate correlated signals immediately.",
        correlated_finding_ids=(
            "FINDING-SC102-aabbccdd-001",
            "FINDING-SC101-aabbccdd-002",
        ),
    )
    scan_result = _make_sample_scan_result(findings=[finding])
    data = JSONReportExporter.to_dict(scan_result)
    f_dict = data["artifacts"][0]["findings"][0]

    assert "correlated_finding_ids" in f_dict
    assert f_dict["correlated_finding_ids"] == [
        "FINDING-SC102-aabbccdd-001",
        "FINDING-SC101-aabbccdd-002",
    ]


def test_finding_without_correlation_has_empty_correlated_ids_in_json() -> None:
    """A plain (non-correlated) finding must have an empty correlated_finding_ids list in JSON."""
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    finding = Finding(
        finding_id="FINDING-P101-e3b0c442-001",
        rule_id="P101",
        rule_name="Dangerous Global Import",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256=sha,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=42,
            byte_length=12,
            archive_member=None,
            matched_value="posix.system",
        ),
        description="Pickle imports dangerous callable.",
        remediation="Do not deserialize this model.",
    )
    scan_result = _make_sample_scan_result(findings=[finding])
    data = JSONReportExporter.to_dict(scan_result)
    f_dict = data["artifacts"][0]["findings"][0]

    assert "correlated_finding_ids" in f_dict
    assert f_dict["correlated_finding_ids"] == []
