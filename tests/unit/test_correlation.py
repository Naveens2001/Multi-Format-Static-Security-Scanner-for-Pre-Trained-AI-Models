"""Unit tests for the cross-artifact correlation engine."""

import dataclasses

from model_scanner.core.correlation import CrossArtifactCorrelator
from model_scanner.core.models import (
    ArtifactScanResult,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    ModelFormat,
    RiskScore,
    ScanOutcome,
    Severity,
)


def make_finding(rule_id: str, finding_id: str) -> Finding:
    return Finding(
        finding_id=finding_id,
        rule_id=rule_id,
        rule_name="Test Rule",
        severity=Severity.HIGH,
        category="test",
        confidence=Confidence.HIGH,
        description="Test",
        remediation="Test",
        format=ModelFormat.REPOSITORY,
        artifact_sha256="a" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=0,
            byte_length=0,
            archive_member=None,
            matched_value="",
            context_snippet="",
        ),
    )


def test_correlation_c001_active_supply_chain() -> None:
    sc102 = make_finding("SC102", "1")
    sc101 = make_finding("SC101", "2")

    result = ArtifactScanResult(
        artifact_id="art-1",
        name="repo",
        source_origin="repo",
        input_source="LOCAL_DIRECTORY",  # type: ignore
        detected_format=ModelFormat.REPOSITORY,
        size_bytes=0,
        sha256="a" * 64,
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=RiskScore.clean(),
        findings=(sc102, sc101),
        warnings=(),
        errors=(),
        duration_ms=0,
    )

    correlated = CrossArtifactCorrelator.correlate_scan([result])
    assert len(correlated) == 1
    assert correlated[0].rule_id == "C001"
    assert correlated[0].severity == Severity.CRITICAL
    assert correlated[0].correlated_finding_ids == ("1", "2")
    assert correlated[0].artifact_sha256 == "a" * 64


def test_correlation_c002_networked_payload() -> None:
    p101 = make_finding("P101", "1")
    sc105 = make_finding("SC105", "2")

    # Simulate finding coming from different artifacts
    p101 = dataclasses.replace(p101, artifact_sha256="b" * 64, format=ModelFormat.RAW_PICKLE)
    sc105 = dataclasses.replace(sc105, artifact_sha256="c" * 64)

    r1 = ArtifactScanResult(
        artifact_id="art-1",
        name="model.pkl",
        source_origin="model.pkl",
        input_source="LOCAL_FILE",  # type: ignore
        detected_format=ModelFormat.RAW_PICKLE,
        size_bytes=0,
        sha256="sha-pickle",
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=RiskScore.clean(),
        findings=(p101,),
        warnings=(),
        errors=(),
        duration_ms=0,
    )

    r2 = ArtifactScanResult(
        artifact_id="art-2",
        name="repo",
        source_origin="repo",
        input_source="LOCAL_DIRECTORY",  # type: ignore
        detected_format=ModelFormat.REPOSITORY,
        size_bytes=0,
        sha256="c" * 64,
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=RiskScore.clean(),
        findings=(sc105,),
        warnings=(),
        errors=(),
        duration_ms=0,
    )

    correlated = CrossArtifactCorrelator.correlate_scan([r1, r2])
    assert len(correlated) == 1
    assert correlated[0].rule_id == "C002"
    assert correlated[0].correlated_finding_ids == ("1", "2")
    assert correlated[0].artifact_sha256 == "c" * 64


def test_no_correlation() -> None:
    p101 = make_finding("P101", "1")

    r1 = ArtifactScanResult(
        artifact_id="art-1",
        name="model.pkl",
        source_origin="model.pkl",
        input_source="LOCAL_FILE",  # type: ignore
        detected_format=ModelFormat.RAW_PICKLE,
        size_bytes=0,
        sha256="sha-pickle",
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=RiskScore.clean(),
        findings=(p101,),
        warnings=(),
        errors=(),
        duration_ms=0,
    )

    correlated = CrossArtifactCorrelator.correlate_scan([r1])
    assert len(correlated) == 0
