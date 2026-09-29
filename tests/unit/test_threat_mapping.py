"""
tests.unit.test_threat_mapping
==============================

Focused tests for Threat Mapping data pipeline (CWE / OWASP / MITRE).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from model_scanner.core.engine import ScanOrchestrator
from model_scanner.core.models import (
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    InputSourceType,
    ModelFormat,
    ScanRequest,
    Severity,
)
from model_scanner.core.normalizer import FindingNormalizer
from model_scanner.presentation.json_export import JSONReportExporter
from model_scanner.rules.catalog import RuleCatalog


def _make_dummy_finding(rule_id: str = "P101", cwe_ids: tuple[str, ...] = ()) -> Finding:
    return Finding(
        finding_id="FINDING-P101-00000000-001",
        rule_id=rule_id,
        rule_name="Test Rule",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256="a" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=0,
            byte_length=0,
            archive_member=None,
            matched_value="test",
        ),
        description="Test",
        remediation="Test",
        cwe_ids=cwe_ids,
    )


def test_1_rule_metadata_produces_finding_with_metadata() -> None:
    """TEST 1: A rule with metadata produces a Finding containing the same metadata."""
    finding = Finding(
        finding_id="FINDING-TEST-000",
        rule_id="P101",
        rule_name="Test",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256="a" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=0,
            byte_length=0,
            archive_member=None,
            matched_value="test",
        ),
        description="Test",
        remediation="Test",
        cwe_ids=("CWE-78", "CWE-94"),
        owasp_category="A03:2021-Injection",
        mitre_technique="T1059",
    )

    assert finding.cwe_ids == ("CWE-78", "CWE-94")
    assert finding.owasp_category == "A03:2021-Injection"
    assert finding.mitre_technique == "T1059"


def test_2_normalizer_preserves_and_enriches_metadata() -> None:
    """TEST 2: The normalizer preserves and enriches the metadata."""
    # Finding from analyzer without the threat mapping fields set
    raw_finding = _make_dummy_finding(rule_id="P101")
    assert raw_finding.cwe_ids == ()
    assert raw_finding.owasp_category is None
    assert raw_finding.mitre_technique is None

    normalized = FindingNormalizer.normalize_finding(
        raw_finding,
        sha256="a" * 64,
        index=1,
    )

    # It should now be enriched from RuleCatalog
    rule = RuleCatalog.get_rule("P101")
    assert normalized.cwe_ids == tuple(rule.cwe_ids)
    assert normalized.owasp_category == rule.owasp_category
    assert normalized.mitre_technique == rule.mitre_technique

    # Also test preservation if already set
    raw_populated = _make_dummy_finding(rule_id="P101", cwe_ids=("CWE-999",))
    normalized_populated = FindingNormalizer.normalize_finding(
        raw_populated,
        sha256="a" * 64,
        index=1,
    )
    assert normalized_populated.cwe_ids == ("CWE-999",)


def test_3_json_output_contains_threat_metadata() -> None:
    """TEST 3: JSON output contains "cwe_ids", "owasp_category", "mitre_technique"."""
    finding = Finding(
        finding_id="FINDING-TEST-000",
        rule_id="P101",
        rule_name="Test",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        category="code_execution",
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256="a" * 64,
        evidence=Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=0,
            byte_length=0,
            archive_member=None,
            matched_value="test",
        ),
        description="Test",
        remediation="Test",
        cwe_ids=("CWE-78",),
        owasp_category="A03:2021-Injection",
        mitre_technique="T1059",
    )

    serialized = JSONReportExporter._serialize_finding(finding)

    assert "cwe_ids" in serialized
    assert "owasp_category" in serialized
    assert "mitre_technique" in serialized

    assert serialized["cwe_ids"] == ["CWE-78"]
    assert serialized["owasp_category"] == "A03:2021-Injection"
    assert serialized["mitre_technique"] == "T1059"


def test_4_real_rule_scan_preserves_metadata() -> None:
    """TEST 4: An existing real rule from the catalog is scanned and its threat metadata appears."""
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

        assert len(result.artifacts[0].findings) > 0
        finding = result.artifacts[0].findings[0]

        # Ensure normalizer attached it
        assert finding.rule_id == "P101"
        rule = RuleCatalog.get_rule("P101")
        assert finding.cwe_ids == tuple(rule.cwe_ids)
        assert finding.owasp_category == rule.owasp_category
        assert finding.mitre_technique == rule.mitre_technique

        # Ensure JSON exporter keeps it
        json_str = JSONReportExporter.to_json(result)
        doc = json.loads(json_str)

        json_finding = doc["artifacts"][0]["findings"][0]
        assert "cwe_ids" in json_finding
        assert "owasp_category" in json_finding
        assert "mitre_technique" in json_finding

        assert json_finding["cwe_ids"] == list(rule.cwe_ids)
        assert json_finding["owasp_category"] == rule.owasp_category
        assert json_finding["mitre_technique"] == rule.mitre_technique
    finally:
        tmp_path.unlink()


def test_5_rules_without_optional_metadata_work() -> None:
    """TEST 5: Rules without optional metadata continue to work correctly."""
    from unittest.mock import patch

    raw = _make_dummy_finding(rule_id="P101")

    with patch("model_scanner.core.normalizer.RuleCatalog.has_rule", return_value=False):
        normalized = FindingNormalizer.normalize_finding(raw, sha256="a" * 64, index=1)

    assert normalized.cwe_ids == ()
    assert normalized.owasp_category is None
    assert normalized.mitre_technique is None

    serialized = JSONReportExporter._serialize_finding(normalized)
    assert serialized["cwe_ids"] == []
    assert serialized["owasp_category"] is None
    assert serialized["mitre_technique"] is None
