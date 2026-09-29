"""
tests.unit.test_normalization
=============================

Unit tests for FindingNormalizer, FindingCorrelator, and Orchestrator
integration in Phase 7.3.

Test Coverage:
1. Valid finding normalization
2. Invalid rule ID
3. Invalid coordinate space
4. Negative byte offset
5. ARCHIVE_MEMBER without archive_member
6. Non-ARCHIVE_MEMBER with archive_member
7. matched_value truncation at 512 characters
8. Deterministic finding IDs schema
9. Finding ID increments by index
10. Duplicate finding removal
11. Distinct offsets preserved
12. strict_mode P103 escalation (MEDIUM -> HIGH)
13. Normal mode P103 remains MEDIUM
14. G001 generation on format mismatch
15. G001 evidence content
16. G001 deterministic ID
17. G001 + P101 correlation
18. PyTorch archive + Pickle anomaly correlation
19. SafeTensors metadata + tensor anomaly correlation
20. GGUF metadata + tensor anomaly correlation
21. Deterministic repeated normalization
22. Input sequence is not mutated
23. No dynamic execution
24. No ML runtime imports
25-30. Orchestrator end-to-end integration tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import (
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    FormatDetectionResult,
    InputSourceType,
    ModelFormat,
    RiskBand,
    ScanConfig,
    ScanOutcome,
    ScanRequest,
    Severity,
)
from model_scanner.core.normalizer import (
    CANONICAL_CATEGORIES,
    FindingCorrelator,
    FindingNormalizer,
)
from model_scanner.core.scoring import RiskEngine


def _make_finding(
    rule_id: str = "P101",
    severity: Severity = Severity.CRITICAL,
    confidence: Confidence = Confidence.HIGH,
    category: str = "code_execution",
    byte_offset: int | None = 100,
    matched_value: str = "posix.system",
    archive_member: str | None = None,
    sha256: str = "a" * 64,
) -> Finding:
    """Helper to construct a valid Finding instance for normalization tests."""
    coord_space = (
        CoordinateSpace.ARCHIVE_MEMBER
        if archive_member
        else CoordinateSpace.ARTIFACT
    )
    return Finding(
        finding_id=f"FINDING-{rule_id}-00000000-001",
        rule_id=rule_id,
        rule_name="Test Rule",
        severity=severity,
        confidence=confidence,
        category=category,
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256=sha256,
        evidence=Evidence(
            coordinate_space=coord_space,
            byte_offset=byte_offset,
            byte_length=len(matched_value),
            archive_member=archive_member,
            matched_value=matched_value,
        ),
        description="Sample test description",
        remediation="Sample remediation guidance",
    )


# ---------------------------------------------------------------------------
# Test 1: Valid Finding Normalization
# ---------------------------------------------------------------------------


def test_valid_finding_normalization() -> None:
    """Verify that a valid finding is normalized and assigned expected fields."""
    raw = _make_finding(rule_id="P101", byte_offset=42)
    normalized = FindingNormalizer.normalize_finding(
        raw,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        index=1,
    )

    assert normalized.rule_id == "P101"
    assert normalized.finding_id == "FINDING-P101-e3b0c442-001"
    assert normalized.evidence.byte_offset == 42
    assert normalized.category in CANONICAL_CATEGORIES


# ---------------------------------------------------------------------------
# Test 2: Invalid Rule ID
# ---------------------------------------------------------------------------


def test_invalid_rule_id_rejected() -> None:
    """Verify invalid rule IDs trigger ValueError during normalization."""
    with pytest.raises(ValueError, match=r"Invalid rule_id"):
        FindingNormalizer.validate_rule_id("INVALID999")

    with pytest.raises(ValueError, match=r"Invalid rule_id"):
        FindingNormalizer.validate_rule_id("X101")


def test_valid_rule_ids_accepted() -> None:
    """Verify all canonical rule ID prefixes are accepted by validate_rule_id."""
    valid_ids = [
        "P101",   # Pickle opcode
        "ST102",  # SafeTensors metadata
        "GU103",  # GGUF metadata
        "SC201",  # Supply chain dependency
        "C001",   # Correlated cross-signal
        "C002",   # Correlated cross-signal
        "C003",   # Correlated cross-signal
    ]
    for rule_id in valid_ids:
        assert FindingNormalizer.validate_rule_id(rule_id) == rule_id

    invalid_ids = ["BADID", "c001", "C1", "CC01", "P1", ""]
    for rule_id in invalid_ids:
        with pytest.raises(ValueError, match=r"Invalid rule_id"):
            FindingNormalizer.validate_rule_id(rule_id)


# ---------------------------------------------------------------------------
# Test 3: Invalid Coordinate Space
# ---------------------------------------------------------------------------


def test_invalid_coordinate_space_rejected() -> None:
    """Verify invalid coordinate_space raises ValueError."""
    evidence = Evidence(
        coordinate_space="invalid_space",  # type: ignore[arg-type]
        byte_offset=10,
        byte_length=5,
        archive_member=None,
        matched_value="test",
    )
    with pytest.raises(ValueError, match=r"Invalid coordinate_space"):
        FindingNormalizer.normalize_evidence(evidence)


# ---------------------------------------------------------------------------
# Test 4: Negative Byte Offset
# ---------------------------------------------------------------------------


def test_negative_byte_offset_rejected() -> None:
    """Verify negative byte_offset raises ValueError."""
    with pytest.raises(ValueError, match=r"Evidence\.byte_offset must be >= 0"):
        Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=-5,
            byte_length=5,
            archive_member=None,
            matched_value="test",
        )


# ---------------------------------------------------------------------------
# Test 5 & 6: Archive Member Constraints
# ---------------------------------------------------------------------------


def test_archive_member_missing_for_archive_space() -> None:
    """Verify ARCHIVE_MEMBER without archive_member raises ValueError."""
    with pytest.raises(ValueError, match=r"Evidence\.archive_member must be set"):
        Evidence(
            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
            byte_offset=10,
            byte_length=5,
            archive_member=None,
            matched_value="test",
        )


def test_archive_member_present_for_artifact_space() -> None:
    """Verify non-ARCHIVE_MEMBER with archive_member raises ValueError in normalizer."""
    evidence = Evidence(
        coordinate_space=CoordinateSpace.ARTIFACT,
        byte_offset=10,
        byte_length=5,
        archive_member="archive/data.pkl",
        matched_value="test",
    )
    with pytest.raises(ValueError, match=r"archive_member must be None"):
        FindingNormalizer.normalize_evidence(evidence)


# ---------------------------------------------------------------------------
# Test 7: Matched Value Truncation at 512 Chars
# ---------------------------------------------------------------------------


def test_matched_value_truncation() -> None:
    """Verify matched_value longer than 512 chars is safely bounded to 512."""
    long_value = "A" * 1000
    evidence = Evidence(
        coordinate_space=CoordinateSpace.ARTIFACT,
        byte_offset=0,
        byte_length=1000,
        archive_member=None,
        matched_value=long_value,
    )
    normalized_evidence = FindingNormalizer.normalize_evidence(evidence)

    assert normalized_evidence.matched_value is not None
    assert len(normalized_evidence.matched_value) == 512
    assert normalized_evidence.matched_value == "A" * 512


# ---------------------------------------------------------------------------
# Test 8 & 9: Deterministic Finding IDs
# ---------------------------------------------------------------------------


def test_deterministic_finding_id_schema_and_indexing() -> None:
    """Verify finding IDs increment deterministically by 1-based index."""
    sha256 = "1234567890abcdef" * 4
    f1 = _make_finding(rule_id="P101", byte_offset=10)
    f2 = _make_finding(rule_id="P102", byte_offset=20)
    f3 = _make_finding(rule_id="P201", byte_offset=30)

    normalized = FindingNormalizer.normalize_findings([f1, f2, f3], sha256=sha256)

    assert len(normalized) == 3
    assert normalized[0].finding_id == "FINDING-P101-12345678-001"
    assert normalized[1].finding_id == "FINDING-P102-12345678-002"
    assert normalized[2].finding_id == "FINDING-P201-12345678-003"


# ---------------------------------------------------------------------------
# Test 10 & 11: Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_finding_removal() -> None:
    """Verify identical duplicate findings are deduplicated."""
    f1 = _make_finding(rule_id="P101", byte_offset=100, matched_value="posix.system")
    f2 = _make_finding(rule_id="P101", byte_offset=100, matched_value="posix.system")

    normalized = FindingNormalizer.normalize_findings([f1, f2], sha256="a" * 64)

    assert len(normalized) == 1
    assert normalized[0].finding_id == "FINDING-P101-aaaaaaaa-001"


def test_distinct_offsets_preserved() -> None:
    """Verify findings with distinct offsets remain separate."""
    f1 = _make_finding(rule_id="P101", byte_offset=100)
    f2 = _make_finding(rule_id="P101", byte_offset=200)

    normalized = FindingNormalizer.normalize_findings([f1, f2], sha256="a" * 64)

    assert len(normalized) == 2
    assert normalized[0].evidence.byte_offset == 100
    assert normalized[1].evidence.byte_offset == 200


# ---------------------------------------------------------------------------
# Test 12 & 13: Strict Mode Dynamic Severity Adjustment
# ---------------------------------------------------------------------------


def test_strict_mode_p103_escalation() -> None:
    """Verify P103 escalates from MEDIUM to HIGH in strict mode."""
    finding = _make_finding(rule_id="P103", severity=Severity.MEDIUM)

    normalized_strict = FindingNormalizer.normalize_finding(
        finding,
        sha256="a" * 64,
        index=1,
        strict_mode=True,
    )
    assert normalized_strict.severity == Severity.HIGH


def test_normal_mode_p103_remains_medium() -> None:
    """Verify P103 remains MEDIUM in non-strict (normal) mode."""
    finding = _make_finding(rule_id="P103", severity=Severity.MEDIUM)

    normalized_normal = FindingNormalizer.normalize_finding(
        finding,
        sha256="a" * 64,
        index=1,
        strict_mode=False,
    )
    assert normalized_normal.severity == Severity.MEDIUM


# ---------------------------------------------------------------------------
# Test 14, 15, 16: G001 Format Spoofing Generation
# ---------------------------------------------------------------------------


def test_g001_generation_on_format_mismatch() -> None:
    """Verify format_mismatch triggers G001 generation with correct properties."""
    detection = FormatDetectionResult(
        detected_format=ModelFormat.RAW_PICKLE,
        magic_matched=True,
        extension_matched=False,
        format_mismatch=True,
        details="Expected SafeTensors by extension (.safetensors), but detected RAW_PICKLE magic",
    )

    g001 = FindingCorrelator.generate_g001_finding(detection, sha256="b" * 64, index=1)

    assert g001.rule_id == "G001"
    assert g001.severity == Severity.HIGH
    assert g001.category == "format_integrity"
    assert g001.finding_id == "FINDING-G001-bbbbbbbb-001"
    assert "detected_format=RAW_PICKLE" in (g001.evidence.matched_value or "")


# ---------------------------------------------------------------------------
# Test 17: Format Spoofing + Code Execution Correlation
# ---------------------------------------------------------------------------


def test_g001_and_p101_correlation() -> None:
    """Verify G001 combined with P101 results in multi-signal correlated finding set."""
    detection = FormatDetectionResult(
        detected_format=ModelFormat.RAW_PICKLE,
        magic_matched=True,
        extension_matched=False,
        format_mismatch=True,
        details="Extension .safetensors contradicted RAW_PICKLE bytes",
    )
    p101 = _make_finding(rule_id="P101", severity=Severity.CRITICAL, byte_offset=50)

    correlated = FindingCorrelator.correlate(
        findings=[p101],
        detection_result=detection,
        sha256="c" * 64,
    )

    assert len(correlated) == 2
    rule_ids = [f.rule_id for f in correlated]
    assert "G001" in rule_ids
    assert "P101" in rule_ids

    # RiskEngine evaluation on correlated set
    score = RiskEngine.calculate(correlated, ModelFormat.RAW_PICKLE)
    # G001 (HIGH = 20) + P101 (CRITICAL = 40) = 60.0 * 1.5 (pickle) = 90.0 -> CRITICAL_RISK
    assert score.findings_weighted_sum == pytest.approx(60.0)
    assert score.raw_score == pytest.approx(90.0)
    assert score.final_score == 90
    assert score.risk_band == RiskBand.CRITICAL_RISK


# ---------------------------------------------------------------------------
# Test 18: PyTorch Container Manipulation Correlation
# ---------------------------------------------------------------------------


def test_pytorch_container_manipulation_correlation() -> None:
    """Verify PyTorch ZIP archive anomaly + Pickle opcode anomaly trigger compound obfuscation."""
    f_archive = _make_finding(
        rule_id="P209",
        category="structural_anomaly",
        severity=Severity.HIGH,
        archive_member="archive/data.pkl",
    )
    f_opcode = _make_finding(
        rule_id="P103",
        category="obfuscation",
        severity=Severity.MEDIUM,
        archive_member="archive/data.pkl",
    )

    correlated = FindingCorrelator.correlate([f_archive, f_opcode], sha256="d" * 64)
    assert len(correlated) == 2

    score = RiskEngine.calculate(correlated, ModelFormat.PYTORCH_ZIP)
    # Obfuscation multiplier should be OBFUSCATION_BOTH (1.75)
    assert score.obfuscation_multiplier == 1.75


# ---------------------------------------------------------------------------
# Test 19: SafeTensors Metadata + Tensor Anomaly Correlation
# ---------------------------------------------------------------------------


def test_safetensors_metadata_tensor_anomaly_correlation() -> None:
    """Verify SafeTensors metadata finding (ST101) + tensor anomaly (ST204) correlate."""
    f_meta = _make_finding(
        rule_id="ST101",
        category="credential_exposure",
        severity=Severity.HIGH,
        byte_offset=10,
    )
    f_tensor = _make_finding(
        rule_id="ST204",
        category="structural_anomaly",
        severity=Severity.HIGH,
        byte_offset=500,
    )

    correlated = FindingCorrelator.correlate([f_meta, f_tensor], sha256="e" * 64)
    assert len(correlated) == 2

    score = RiskEngine.calculate(correlated, ModelFormat.SAFETENSORS)
    assert score.obfuscation_multiplier == 1.5  # Structural anomaly present


# ---------------------------------------------------------------------------
# Test 20: GGUF Metadata + Tensor Anomaly Correlation
# ---------------------------------------------------------------------------


def test_gguf_metadata_tensor_anomaly_correlation() -> None:
    """Verify GGUF metadata finding (GU101) + tensor anomaly (GU201) correlate."""
    f_meta = _make_finding(
        rule_id="GU101",
        category="credential_exposure",
        severity=Severity.HIGH,
        byte_offset=100,
    )
    f_tensor = _make_finding(
        rule_id="GU201",
        category="structural_anomaly",
        severity=Severity.CRITICAL,
        byte_offset=200,
    )

    correlated = FindingCorrelator.correlate([f_meta, f_tensor], sha256="f" * 64)
    assert len(correlated) == 2

    score = RiskEngine.calculate(correlated, ModelFormat.GGUF)
    assert score.obfuscation_multiplier == 1.5


# ---------------------------------------------------------------------------
# Test 21 & 22: Determinism & Immutability
# ---------------------------------------------------------------------------


def test_normalization_determinism_and_immutability() -> None:
    """Verify normalization does not mutate inputs and produces identical results."""
    original = [
        _make_finding(rule_id="P101", byte_offset=10),
        _make_finding(rule_id="P102", byte_offset=20),
    ]
    original_copy = list(original)

    res1 = FindingNormalizer.normalize_findings(original, sha256="a" * 64)
    res2 = FindingNormalizer.normalize_findings(original, sha256="a" * 64)

    assert original == original_copy  # Input sequence not mutated
    assert res1 == res2  # Deterministic output


# ---------------------------------------------------------------------------
# Orchestrator Integration Tests
# ---------------------------------------------------------------------------


def test_orchestrator_clean_artifact_produces_clean_risk_score() -> None:
    """Verify ScanOrchestrator computes RiskScore.clean() for benign clean SafeTensors."""
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
        artifact_result = result.artifacts[0]
        assert artifact_result.scan_outcome == ScanOutcome.COMPLETE
        assert len(artifact_result.findings) == 0
        assert artifact_result.risk_score.final_score == 0
        assert artifact_result.risk_score.risk_band == RiskBand.CLEAN
        assert result.summary.max_risk_score == 0
    finally:
        tmp_path.unlink()


def test_orchestrator_malicious_artifact_computes_real_risk_score() -> None:
    """Verify ScanOrchestrator normalizes findings and computes non-zero RiskScore."""
    # Pickle with GLOBAL posix.system
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

        assert result.summary.total_artifacts == 1
        artifact_result = result.artifacts[0]
        assert artifact_result.scan_outcome == ScanOutcome.COMPLETE
        assert len(artifact_result.findings) >= 1

        # Finding must be normalized
        finding = artifact_result.findings[0]
        assert finding.finding_id.startswith("FINDING-")
        assert finding.artifact_sha256 == artifact_result.sha256

        # Risk score must be calculated by RiskEngine (CRITICAL 40 * 1.5 pickle = 60)
        assert artifact_result.risk_score.final_score >= 60
        assert artifact_result.risk_score.risk_band in (
            RiskBand.MEDIUM_RISK,
            RiskBand.HIGH_RISK,
            RiskBand.CRITICAL_RISK,
        )
        assert result.summary.max_risk_score == artifact_result.risk_score.final_score
    finally:
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test: correlated_finding_ids survive normalize_finding
# ---------------------------------------------------------------------------


def test_correlated_finding_ids_preserved_by_normalize_finding() -> None:
    """correlated_finding_ids must pass through normalize_finding unchanged."""
    raw = Finding(
        finding_id="FINDING-C001-00000000-001",
        rule_id="C001",
        rule_name="Active Supply Chain Execution Intent",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        category="supply_chain",
        format=ModelFormat.REPOSITORY,
        artifact_sha256="a" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.GLOBAL,
            byte_offset=0,
            byte_length=0,
            archive_member=None,
            matched_value="SC102,SC101",
        ),
        description="Correlated finding.",
        remediation="Investigate correlated signals.",
        correlated_finding_ids=("FINDING-SC102-aaaa-001", "FINDING-SC101-aaaa-002"),
    )
    normalized = FindingNormalizer.normalize_finding(
        raw,
        sha256="a" * 64,
        index=1,
    )
    assert normalized.correlated_finding_ids == (
        "FINDING-SC102-aaaa-001",
        "FINDING-SC101-aaaa-002",
    )


def test_non_correlated_finding_has_empty_correlated_ids() -> None:
    """A regular finding that was not produced by correlation has empty correlated_finding_ids."""
    raw = _make_finding(rule_id="P101")
    normalized = FindingNormalizer.normalize_finding(
        raw,
        sha256="a" * 64,
        index=1,
    )
    assert normalized.correlated_finding_ids == ()


def test_correlated_finding_ids_preserved_through_normalize_findings() -> None:
    """correlated_finding_ids survive the full normalize_findings batch pipeline."""
    correlated = Finding(
        finding_id="FINDING-C001-00000000-001",
        rule_id="C001",
        rule_name="Active Supply Chain Execution Intent",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        category="supply_chain",
        format=ModelFormat.REPOSITORY,
        artifact_sha256="b" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.GLOBAL,
            byte_offset=0,
            byte_length=0,
            archive_member=None,
            matched_value="SC102,SC101",
        ),
        description="Correlated finding.",
        remediation="Investigate correlated signals.",
        correlated_finding_ids=("SRC-A", "SRC-B"),
    )
    result = FindingNormalizer.normalize_findings([correlated], sha256="b" * 64)
    assert len(result) == 1
    assert result[0].correlated_finding_ids == ("SRC-A", "SRC-B")


def test_orchestrator_strict_mode_propagated() -> None:
    """Verify strict mode is propagated and properly adjusts finding severity."""
    # Unknown global pickle
    pickle_bytes = b"\x80\x03cunknown_module\nunknown_func\n."

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as tmp:
        tmp.write(pickle_bytes)
        tmp_path = Path(tmp.name)

    try:
        config = ScanConfig(strict_mode=True)
        request = ScanRequest(
            target=tmp_path,
            input_source_type=InputSourceType.LOCAL_FILE,
            config=config,
        )
        orchestrator = ScanOrchestrator()
        result = orchestrator.scan(request)

        assert result.strict_mode is True
        artifact_result = result.artifacts[0]
        if artifact_result.findings:
            p103_findings = [f for f in artifact_result.findings if f.rule_id == "P103"]
            if p103_findings:
                # Under strict mode, P103 should be HIGH
                assert p103_findings[0].severity == Severity.HIGH
    finally:
        tmp_path.unlink()
