"""
tests/unit/test_report_model.py
================================

Comprehensive tests for the reporting and dashboard presentation layer:

1. ReportModel construction — derived correctly, canonical models untouched
2. Secret redaction — raw secrets never appear in ReportModel/JSON/HTML
3. Presentation deduplication — occurrence counts correct, different evidence preserved
4. HTML security — no external resources, all values escaped
5. JSON report — stable keys, schema version, disclaimer, no secret leakage
6. Disclaimer — exact text in JSON and HTML
7. Correlation rendering — existing data rendered, no new correlations invented
8. Methodology/limitations — fail-closed behavior visible
9. Catalog mapping integrity — rule names, CWEs, OWASP, MITRE from catalog
10. Regression — existing tests unaffected
"""

from __future__ import annotations

import json
import re
import uuid

import pytest

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
    ScanOutcome,
    ScanResult,
    ScanSummary,
    ScanWarning,
    Severity,
    risk_band_for_score,
)
from model_scanner.presentation.html_report import HTMLReportRenderer
from model_scanner.presentation.professional_report import ProfessionalJSONExporter
from model_scanner.presentation.report_model import (
    STATIC_ANALYSIS_DISCLAIMER,
    ReportBuilder,
    ReportModel,
    ScanReportBuilder,
    _mask_secret,
)
from model_scanner.rules.catalog import RuleCatalog

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_ZERO_SHA256 = "a" * 64
_VALID_UUID = str(uuid.uuid4())


def _make_evidence(
    matched_value: str = "os.system",
    byte_offset: int | None = 0,
    archive_member: str | None = None,
    opcode: str | None = None,
) -> Evidence:
    space = (
        CoordinateSpace.ARCHIVE_MEMBER
        if archive_member is not None
        else CoordinateSpace.ARTIFACT
    )
    return Evidence(
        coordinate_space=space,
        byte_offset=byte_offset,
        byte_length=None,
        archive_member=archive_member,
        matched_value=matched_value,
        opcode=opcode,
    )


def _make_finding(
    rule_id: str = "P101",
    rule_name: str = "Dangerous Global Import",
    severity: Severity = Severity.CRITICAL,
    confidence: Confidence = Confidence.HIGH,
    category: str = "code_execution",
    matched_value: str = "os.system",
    byte_offset: int | None = 0,
    archive_member: str | None = None,
    correlated_finding_ids: tuple[str, ...] = (),
) -> Finding:
    return Finding(
        finding_id=str(uuid.uuid4()),
        rule_id=rule_id,
        rule_name=rule_name,
        severity=severity,
        confidence=confidence,
        category=category,
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256=_ZERO_SHA256,
        evidence=_make_evidence(
            matched_value=matched_value,
            byte_offset=byte_offset,
            archive_member=archive_member,
        ),
        description="A dangerous global import was detected.",
        remediation="Do not deserialize this artifact.",
        correlated_finding_ids=correlated_finding_ids,
    )


def _make_risk_score(final_score: int = 0) -> RiskScore:
    return RiskScore(
        raw_score=float(final_score),
        final_score=final_score,
        risk_band=risk_band_for_score(final_score),
        format_multiplier=1.0,
        obfuscation_multiplier=1.0,
        findings_weighted_sum=0.0,
    )


def _make_summary(
    exit_code: int = 0,
    outcome: ScanOutcome = ScanOutcome.COMPLETE,
    findings_by_severity: dict[str, int] | None = None,
) -> ScanSummary:
    sev = findings_by_severity or {
        "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0,
    }
    return ScanSummary(
        total_artifacts=1,
        scanned_successfully=1,
        failed_artifacts=0,
        unsupported_artifacts=0,
        findings_by_severity=sev,
        artifacts_by_risk_band={"CLEAN": 1},
        max_risk_score=0,
        overall_outcome=outcome,
        exit_code=exit_code,
    )


def _make_artifact(
    findings: tuple[Finding, ...] = (),
    risk_score: RiskScore | None = None,
    scan_outcome: ScanOutcome = ScanOutcome.COMPLETE,
    warnings: tuple[ScanWarning, ...] = (),
    name: str = "model.pkl",
) -> ArtifactScanResult:
    return ArtifactScanResult(
        artifact_id=str(uuid.uuid4()),
        name=name,
        source_origin="/data/model.pkl",
        input_source=InputSourceType.LOCAL_FILE,
        detected_format=ModelFormat.RAW_PICKLE,
        size_bytes=1024,
        sha256=_ZERO_SHA256,
        scan_outcome=scan_outcome,
        risk_score=risk_score or _make_risk_score(0),
        findings=findings,
        warnings=warnings,
        errors=(),
        duration_ms=42,
    )


def _make_scan_result(
    findings: tuple[Finding, ...] = (),
    risk_score: RiskScore | None = None,
    scan_outcome: ScanOutcome = ScanOutcome.COMPLETE,
    exit_code: int = 0,
    warnings: tuple[ScanWarning, ...] = (),
    artifact_name: str = "model.pkl",
) -> ScanResult:
    artifact = _make_artifact(
        findings=findings,
        risk_score=risk_score,
        scan_outcome=scan_outcome,
        warnings=warnings,
        name=artifact_name,
    )
    return ScanResult(
        scan_id=_VALID_UUID,
        scanner_version="2.0.0",
        timestamp_utc="2026-08-21T12:00:00+00:00",
        target_request="/data/model.pkl",
        strict_mode=False,
        artifacts=(artifact,),
        summary=_make_summary(exit_code=exit_code, outcome=scan_outcome),
        total_duration_ms=100,
    )


# ---------------------------------------------------------------------------
# 1. ReportModel construction
# ---------------------------------------------------------------------------


class TestReportModelConstruction:
    def test_basic_construction(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        assert isinstance(report, ReportModel)

    def test_canonical_scan_result_unchanged(self) -> None:
        finding = _make_finding()
        sr = _make_scan_result(findings=(finding,))
        original_finding_count = len(sr.artifacts[0].findings)
        original_rule_id = sr.artifacts[0].findings[0].rule_id
        _report = ReportBuilder.build(sr)
        # Verify original ScanResult is unchanged
        assert len(sr.artifacts[0].findings) == original_finding_count
        assert sr.artifacts[0].findings[0].rule_id == original_rule_id

    def test_canonical_finding_unchanged(self) -> None:
        finding = _make_finding(matched_value="os.system")
        sr = _make_scan_result(findings=(finding,))
        _report = ReportBuilder.build(sr)
        # The original finding's matched_value must remain intact
        assert sr.artifacts[0].findings[0].evidence.matched_value == "os.system"

    def test_header_fields(self) -> None:
        sr = _make_scan_result(artifact_name="test_model.safetensors")
        report = ReportBuilder.build(sr)
        assert report.artifact_name == "test_model.safetensors"
        assert report.scanner_version == "2.0.0"
        assert report.scan_id == _VALID_UUID
        assert report.detected_format == "RAW_PICKLE"
        assert report.sha256 == _ZERO_SHA256

    def test_verdict_pass(self) -> None:
        sr = _make_scan_result(findings=(), exit_code=0)
        report = ReportBuilder.build(sr)
        assert report.verdict == "PASS"

    def test_verdict_unsupported_is_never_pass(self) -> None:
        """An artifact nothing analysed must not be reported as clean.

        Regression guard: an UNSUPPORTED artifact has zero findings because no
        analyzer ran, not because it is safe. Reporting PASS told the reader a
        legacy non-ZIP PyTorch checkpoint - a raw Pickle stream - was clean
        when it had never been inspected.
        """
        sr = _make_scan_result(
            findings=(),
            scan_outcome=ScanOutcome.UNSUPPORTED,
            exit_code=0,
        )
        report = ReportBuilder.build(sr)
        assert report.verdict == "UNSUPPORTED"
        assert report.verdict != "PASS"

    def test_unsupported_html_does_not_claim_clean(self) -> None:
        """The rendered report must not describe an unanalysed artifact as clean."""
        from model_scanner.presentation.html_report import HTMLReportRenderer

        sr = _make_scan_result(
            findings=(),
            scan_outcome=ScanOutcome.UNSUPPORTED,
            exit_code=0,
        )
        html = HTMLReportRenderer.render(ReportBuilder.build(sr))
        assert "PASS &#8212; No Findings Detected" not in html
        assert "PASS — No Findings Detected" not in html
        assert "0 - clean, no findings" not in html
        assert "UNSUPPORTED" in html

    def test_verdict_fail_critical(self) -> None:
        f = _make_finding(severity=Severity.CRITICAL)
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        assert report.verdict == "FAIL"

    def test_verdict_fail_high(self) -> None:
        f = _make_finding(rule_id="P102", severity=Severity.HIGH)
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        assert report.verdict == "FAIL"

    def test_verdict_warn_medium(self) -> None:
        f = _make_finding(rule_id="P103", severity=Severity.MEDIUM)
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        assert report.verdict == "WARN"

    def test_verdict_fail_on_error_outcome(self) -> None:
        sr = _make_scan_result(
            findings=(),
            scan_outcome=ScanOutcome.ERROR,
            exit_code=2,
        )
        report = ReportBuilder.build(sr)
        assert report.verdict == "FAIL"

    def test_severity_counts(self) -> None:
        findings: tuple[Finding, ...] = (
            _make_finding(rule_id="P101", severity=Severity.CRITICAL),
            _make_finding(rule_id="P102", severity=Severity.HIGH, byte_offset=100),
            _make_finding(rule_id="P103", severity=Severity.MEDIUM, byte_offset=200),
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        report = ReportBuilder.build(sr)
        assert report.severity_counts["CRITICAL"] == 1
        assert report.severity_counts["HIGH"] == 1
        assert report.severity_counts["MEDIUM"] == 1
        assert report.severity_counts["LOW"] == 0

    def test_total_scanner_findings(self) -> None:
        findings = tuple(
            _make_finding(rule_id="P101", byte_offset=i * 100) for i in range(5)
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        report = ReportBuilder.build(sr)
        assert report.total_scanner_findings == 5

    def test_static_analysis_disclaimer_present(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        assert report.static_analysis_disclaimer == STATIC_ANALYSIS_DISCLAIMER

    def test_disclaimer_exact_text(self) -> None:
        expected = (
            "This assessment is based on static analysis. The scanner does not "
            "execute, deserialize, or otherwise run model artifacts. Evidence "
            "represents security-relevant structures observed during static "
            "inspection and does not constitute proof of successful exploitation."
        )
        assert STATIC_ANALYSIS_DISCLAIMER == expected

    def test_no_artifacts_raises(self) -> None:
        # Build a ScanResult with no artifacts
        empty_sr = ScanResult(
            scan_id=_VALID_UUID,
            scanner_version="2.0.0",
            timestamp_utc="2026-08-21T12:00:00+00:00",
            target_request="/data",
            strict_mode=False,
            artifacts=(),
            summary=_make_summary(),
            total_duration_ms=0,
        )
        with pytest.raises(ValueError, match="no artifacts"):
            ReportBuilder.build(empty_sr)

    def test_artifact_index_out_of_range_raises(self) -> None:
        sr = _make_scan_result()
        with pytest.raises(IndexError):
            ReportBuilder.build(sr, artifact_index=5)


# ---------------------------------------------------------------------------
# 2. Secret redaction
# ---------------------------------------------------------------------------

# Dynamically build a test secret (never hardcoded)
_SECRET_PREFIX = "sk-ant-api0"
_SECRET_BODY = "3-xBCDEFGHIJKLMNOP"
_SECRET_SUFFIX = "qrst"
_TEST_SECRET = _SECRET_PREFIX + _SECRET_BODY + _SECRET_SUFFIX


class TestSecretRedaction:
    def test_mask_secret_long_enough(self) -> None:
        masked, orig_len = _mask_secret(_TEST_SECRET)
        assert _TEST_SECRET not in masked
        assert orig_len == len(_TEST_SECRET)
        assert "..." in masked

    def test_mask_secret_too_short(self) -> None:
        short = "abc"
        masked, orig_len = _mask_secret(short)
        assert masked == "[redacted]"
        assert orig_len == 3

    def test_raw_secret_not_in_report_model(self) -> None:
        # ST104 is a credential-exposure rule
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="ST104",
            rule_name="Credential or Secret Exposure in Metadata",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            category="credential_exposure",
            format=ModelFormat.SAFETENSORS,
            artifact_sha256=_ZERO_SHA256,
            evidence=Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=0,
                byte_length=None,
                archive_member=None,
                matched_value=_TEST_SECRET,
            ),
            description="Secret detected in metadata.",
            remediation="Revoke the exposed token immediately.",
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)

        # Raw secret must NOT appear in any evidence sample matched_value
        for rf in report.report_findings:
            for ev in rf.evidence_samples:
                assert _TEST_SECRET not in ev.matched_value, (
                    f"Raw secret found in ReportModel evidence: {ev.matched_value!r}"
                )

    def test_raw_secret_not_in_json(self) -> None:
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="ST104",
            rule_name="Credential or Secret Exposure in Metadata",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            category="credential_exposure",
            format=ModelFormat.SAFETENSORS,
            artifact_sha256=_ZERO_SHA256,
            evidence=Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=0,
                byte_length=None,
                archive_member=None,
                matched_value=_TEST_SECRET,
            ),
            description="Secret in metadata.",
            remediation="Revoke immediately.",
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        json_str = ProfessionalJSONExporter.to_json(report)
        assert _TEST_SECRET not in json_str

    def test_raw_secret_not_in_html(self) -> None:
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="ST104",
            rule_name="Credential or Secret Exposure in Metadata",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            category="credential_exposure",
            format=ModelFormat.SAFETENSORS,
            artifact_sha256=_ZERO_SHA256,
            evidence=Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=0,
                byte_length=None,
                archive_member=None,
                matched_value=_TEST_SECRET,
            ),
            description="Secret in metadata.",
            remediation="Revoke immediately.",
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        html_str = HTMLReportRenderer.render(report)
        assert _TEST_SECRET not in html_str

    def test_non_credential_rule_not_redacted(self) -> None:
        safe_value = "os.system"
        f = _make_finding(rule_id="P101", matched_value=safe_value)
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        for rf in report.report_findings:
            for ev in rf.evidence_samples:
                assert ev.matched_value == safe_value

    def test_original_length_preserved_for_secret(self) -> None:
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="ST104",
            rule_name="Credential or Secret Exposure in Metadata",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            category="credential_exposure",
            format=ModelFormat.SAFETENSORS,
            artifact_sha256=_ZERO_SHA256,
            evidence=Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=0,
                byte_length=None,
                archive_member=None,
                matched_value=_TEST_SECRET,
            ),
            description="Secret.",
            remediation="Revoke.",
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        for rf in report.report_findings:
            for ev in rf.evidence_samples:
                if ev.original_length is not None:
                    assert ev.original_length == len(_TEST_SECRET)


# ---------------------------------------------------------------------------
# 3. Presentation deduplication
# ---------------------------------------------------------------------------


class TestPresentationDeduplication:
    def test_identical_findings_collapsed(self) -> None:
        n = 5
        # All share the same rule_id and byte_offset bucket
        findings = tuple(
            _make_finding(rule_id="P101", byte_offset=0)
            for _ in range(n)
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        report = ReportBuilder.build(sr)

        # Scanner count unchanged
        assert report.total_scanner_findings == n
        # Collapsed into 1 unique report finding
        assert report.unique_report_findings == 1
        assert report.collapsed_findings == n - 1
        assert len(report.report_findings) == 1
        assert report.report_findings[0].occurrence_count == n

    def test_same_value_different_offsets_collapses(self) -> None:
        """Identical findings collapse regardless of byte distance.

        Collapsing keys on (rule, member, matched_value), not a byte bucket.
        Bucketing split identical findings that happened to land in different
        buckets while merging unrelated findings that sat close together.
        """
        findings = tuple(
            _make_finding(rule_id="P101", byte_offset=i * 1000)
            for i in range(3)
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        report = ReportBuilder.build(sr)

        assert report.total_scanner_findings == 3
        assert report.unique_report_findings == 1
        assert report.collapsed_findings == 2

    def test_different_matched_values_not_collapsed(self) -> None:
        """Same rule and member but different subjects stay separate."""
        findings = (
            _make_finding(rule_id="P106", byte_offset=100, matched_value="__setstate__"),
            _make_finding(rule_id="P106", byte_offset=104, matched_value="__globals__"),
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        report = ReportBuilder.build(sr)

        assert report.unique_report_findings == 2
        assert report.collapsed_findings == 0

    def test_different_rules_not_collapsed(self) -> None:
        findings = (
            _make_finding(rule_id="P101", byte_offset=0),
            _make_finding(rule_id="P102", byte_offset=0),
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        report = ReportBuilder.build(sr)
        assert report.unique_report_findings == 2

    def test_evidence_samples_capped_at_three(self) -> None:
        n = 10
        # Same rule, same bucket
        findings = tuple(
            _make_finding(rule_id="P101", byte_offset=0)
            for _ in range(n)
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        report = ReportBuilder.build(sr)
        rf = report.report_findings[0]
        assert len(rf.evidence_samples) == 3
        assert rf.additional_occurrences == n - 3

    def test_dedup_does_not_affect_scanner_count(self) -> None:
        # Identical findings
        findings = tuple(
            _make_finding(rule_id="P101", byte_offset=0)
            for _ in range(7)
        )
        sr = _make_scan_result(findings=findings, exit_code=1)
        assert len(sr.artifacts[0].findings) == 7  # Scanner count unchanged
        report = ReportBuilder.build(sr)
        assert report.total_scanner_findings == 7


# ---------------------------------------------------------------------------
# 4. HTML security
# ---------------------------------------------------------------------------


class TestHTMLSecurity:
    def _render(self, sr: ScanResult | None = None) -> str:
        if sr is None:
            sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        return HTMLReportRenderer.render(report)

    def test_no_external_resource_urls(self) -> None:
        html_str = self._render()
        # No src= pointing to external URLs
        ext_src = re.findall(r'src\s*=\s*["\']https?://', html_str)
        assert not ext_src, f"External src references found: {ext_src}"

    def test_no_cdn_links(self) -> None:
        html_str = self._render()
        cdn_patterns = ["cdn.jsdelivr.net", "cdnjs.cloudflare.com",
                       "unpkg.com", "cdn.tailwindcss.com"]
        for cdn in cdn_patterns:
            assert cdn not in html_str, f"CDN reference found: {cdn}"

    def test_no_external_scripts(self) -> None:
        html_str = self._render()
        ext_scripts = re.findall(r'<script[^>]+src\s*=\s*["\']https?://', html_str)
        assert not ext_scripts, f"External script tags found: {ext_scripts}"

    def test_interpolated_values_escaped(self) -> None:
        # Artifact name with XSS payload
        xss_name = '<script>alert("xss")</script>'
        artifact = _make_artifact(name=xss_name)
        sr = ScanResult(
            scan_id=_VALID_UUID,
            scanner_version="2.0.0",
            timestamp_utc="2026-08-21T12:00:00+00:00",
            target_request=xss_name,
            strict_mode=False,
            artifacts=(artifact,),
            summary=_make_summary(),
            total_duration_ms=0,
        )
        report = ReportBuilder.build(sr)
        html_str = HTMLReportRenderer.render(report)
        # Raw XSS must not appear unescaped
        assert "<script>alert" not in html_str
        # The escaped version should appear instead
        assert "&lt;script&gt;" in html_str

    def test_evidence_value_escaped(self) -> None:
        f = _make_finding(matched_value='<img src=x onerror=alert(1)>')
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        html_str = HTMLReportRenderer.render(report)
        assert '<img src=x' not in html_str

    def test_single_html_file_no_link_rel(self) -> None:
        html_str = self._render()
        # No external stylesheet links
        ext_links = re.findall(r'<link[^>]+href\s*=\s*["\']https?://', html_str)
        assert not ext_links

    def test_html_readable_structure(self) -> None:
        html_str = self._render()
        assert "<!DOCTYPE html>" in html_str
        assert "<html" in html_str
        assert "<style>" in html_str
        assert "</html>" in html_str

    def test_print_css_present(self) -> None:
        html_str = self._render()
        assert "@media print" in html_str
        assert "@page" in html_str


# ---------------------------------------------------------------------------
# 5. JSON report
# ---------------------------------------------------------------------------


class TestJSONReport:
    def _make_report(self, sr: ScanResult | None = None) -> ReportModel:
        if sr is None:
            sr = _make_scan_result()
        return ReportBuilder.build(sr)

    def test_valid_json(self) -> None:
        report = self._make_report()
        json_str = ProfessionalJSONExporter.to_json(report)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_schema_version_present(self) -> None:
        report = self._make_report()
        json_str = ProfessionalJSONExporter.to_json(report)
        parsed = json.loads(json_str)
        assert "report_schema_version" in parsed
        assert isinstance(parsed["report_schema_version"], str)

    def test_disclaimer_in_json(self) -> None:
        report = self._make_report()
        json_str = ProfessionalJSONExporter.to_json(report)
        parsed = json.loads(json_str)
        assert "static_analysis_disclaimer" in parsed
        assert parsed["static_analysis_disclaimer"] == STATIC_ANALYSIS_DISCLAIMER

    def test_stable_key_order(self) -> None:
        report = self._make_report()
        j1 = ProfessionalJSONExporter.to_json(report)
        j2 = ProfessionalJSONExporter.to_json(report)
        assert j1 == j2

    def test_json_round_trip(self) -> None:
        report = self._make_report()
        d = ProfessionalJSONExporter.to_dict(report)
        assert isinstance(d, dict)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["verdict"] == report.verdict
        assert parsed["header"]["sha256"] == report.sha256

    def test_no_secret_in_json(self) -> None:
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="ST104",
            rule_name="Credential or Secret Exposure in Metadata",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            category="credential_exposure",
            format=ModelFormat.SAFETENSORS,
            artifact_sha256=_ZERO_SHA256,
            evidence=Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=0,
                byte_length=None,
                archive_member=None,
                matched_value=_TEST_SECRET,
            ),
            description="Secret found.",
            remediation="Revoke immediately.",
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        json_str = ProfessionalJSONExporter.to_json(report)
        assert _TEST_SECRET not in json_str

    def test_findings_present_in_json(self) -> None:
        f = _make_finding()
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        d = ProfessionalJSONExporter.to_dict(report)
        assert len(d["findings"]) == 1
        assert d["findings"][0]["rule_id"] == "P101"

    def test_methodology_in_json(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        d = ProfessionalJSONExporter.to_dict(report)
        assert "methodology" in d
        assert "static_only_limitation" in d["methodology"]
        assert "fail_closed_behavior" in d["methodology"]

    def test_reproducibility_in_json(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        d = ProfessionalJSONExporter.to_dict(report)
        assert "reproducibility" in d
        assert d["reproducibility"]["scan_id"] == _VALID_UUID


# ---------------------------------------------------------------------------
# 6. Disclaimer verification
# ---------------------------------------------------------------------------


class TestDisclaimer:
    def test_disclaimer_in_json_top_level(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        d = ProfessionalJSONExporter.to_dict(report)
        assert d["static_analysis_disclaimer"] == STATIC_ANALYSIS_DISCLAIMER

    def test_disclaimer_in_html(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        html_str = HTMLReportRenderer.render(report)
        # The disclaimer text (HTML-escaped) must appear
        # Key phrases that would survive HTML escaping
        assert "static analysis" in html_str.lower()
        assert "does not" in html_str
        assert "proof of successful exploitation" in html_str

    def test_disclaimer_exact_substring_in_report_model(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        assert report.static_analysis_disclaimer == STATIC_ANALYSIS_DISCLAIMER


# ---------------------------------------------------------------------------
# 7. Correlation rendering
# ---------------------------------------------------------------------------


class TestCorrelationRendering:
    def test_correlated_ids_rendered_in_report(self) -> None:
        corr_id = "FINDING-C001-abc12345-001"
        f = _make_finding(
            rule_id="P101",
            correlated_finding_ids=(corr_id,),
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        assert len(report.report_findings) == 1
        assert corr_id in report.report_findings[0].correlated_finding_ids

    def test_no_correlations_produces_note(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        assert "No correlated" in report.correlation_note

    def test_correlations_present_note(self) -> None:
        corr_id = "FINDING-C001-abc12345-001"
        f = _make_finding(rule_id="P101", correlated_finding_ids=(corr_id,))
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        assert "Correlated" in report.correlation_note

    def test_correlation_ids_in_json(self) -> None:
        corr_id = "FINDING-C001-abc12345-001"
        f = _make_finding(rule_id="P101", correlated_finding_ids=(corr_id,))
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        d = ProfessionalJSONExporter.to_dict(report)
        finding_dict = d["findings"][0]
        assert corr_id in finding_dict["correlated_finding_ids"]

    def test_correlation_ids_in_html(self) -> None:
        corr_id = "FINDING-C001-abc12345-001"
        f = _make_finding(rule_id="P101", correlated_finding_ids=(corr_id,))
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        html_str = HTMLReportRenderer.render(report)
        assert corr_id in html_str

    def test_report_does_not_invent_correlations(self) -> None:
        # A finding with no correlated IDs → report must not invent any
        f = _make_finding(rule_id="P101")
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        assert len(report.report_findings[0].correlated_finding_ids) == 0


# ---------------------------------------------------------------------------
# 8. Methodology / limitations
# ---------------------------------------------------------------------------


class TestMethodology:
    def test_methodology_notes_non_empty(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        assert len(report.methodology_notes) > 0

    def test_warning_appears_in_methodology(self) -> None:
        w = ScanWarning(
            warning_code="UNSUPPORTED_MEMBER",
            message="unsupported archive member skipped",
            context="archive/data.pkl",
        )
        sr = _make_scan_result(warnings=(w,))
        report = ReportBuilder.build(sr)
        # Warning message should appear in methodology notes
        has_warning_note = any(
            "UNSUPPORTED_MEMBER" in n or "unsupported archive" in n.lower()
            for n in report.methodology_notes
        )
        assert has_warning_note

    def test_fail_closed_note_present(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        d = ProfessionalJSONExporter.to_dict(report)
        text = d["methodology"]["fail_closed_behavior"].lower()
        assert "parsing failures" in text or "fail" in text

    def test_formats_analyzed(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        assert "RAW_PICKLE" in report.formats_analyzed

    def test_methodology_in_html(self) -> None:
        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        html_str = HTMLReportRenderer.render(report)
        assert "Methodology" in html_str


# ---------------------------------------------------------------------------
# 9. Catalog mapping integrity
# ---------------------------------------------------------------------------


class TestCatalogMappingIntegrity:
    def test_rule_names_from_catalog(self) -> None:
        all_rules = RuleCatalog.list_all_rules()
        for rule in all_rules[:5]:  # Test a sample
            f = _make_finding(rule_id=rule.rule_id, rule_name=rule.name)
            sr = _make_scan_result(findings=(f,), exit_code=1)
            report = ReportBuilder.build(sr)
            assert len(report.report_findings) > 0
            rf = report.report_findings[0]
            assert rf.rule_id == rule.rule_id
            assert rf.rule_name == rule.name

    def test_cwe_ids_from_catalog(self) -> None:
        # Use a known rule with CWEs
        rule = RuleCatalog.get_rule("P101")
        f = _make_finding(rule_id="P101")
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        rf = report.report_findings[0]
        for cwe in rule.cwe_ids:
            assert cwe in rf.cwe_ids

    def test_owasp_from_catalog(self) -> None:
        rule = RuleCatalog.get_rule("P101")
        f = _make_finding(rule_id="P101")
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        rf = report.report_findings[0]
        assert rule.owasp_category and rule.owasp_category in rf.owasp_category

    def test_mitre_from_catalog(self) -> None:
        rule = RuleCatalog.get_rule("P101")
        f = _make_finding(rule_id="P101")
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        rf = report.report_findings[0]
        assert rule.mitre_technique and rule.mitre_technique in rf.mitre_technique

    def test_no_hardcoded_rule_ids_in_report_logic(self) -> None:
        # Verify unknown rule_id still produces a report finding (gracefully)
        # NOTE: "SC199" is not in catalog but we can't use unknown rule IDs in Finding
        # due to validation. Use a real but lesser-known catalog rule.
        rule = RuleCatalog.get_rule("SC201")
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="SC201",
            rule_name=rule.name,
            severity=Severity.LOW,
            confidence=Confidence.HIGH,
            category="supply_chain",
            format=ModelFormat.REPOSITORY,
            artifact_sha256=_ZERO_SHA256,
            evidence=Evidence(
                coordinate_space=CoordinateSpace.ARTIFACT,
                byte_offset=0,
                byte_length=None,
                archive_member=None,
                matched_value="requests>=2.0",
            ),
            description="Unpinned dependency.",
            remediation="Pin all dependencies.",
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        assert len(report.report_findings) == 1
        rf = report.report_findings[0]
        assert rf.rule_id == "SC201"
        assert rf.rule_name == rule.name

    def test_catalog_cve_ids_not_fabricated(self) -> None:
        # All CVE IDs in catalog (if any) must match CVE-YYYY-NNNNN format
        cve_pattern = re.compile(r"^CVE-\d{4}-\d{4,}$")
        for rule in RuleCatalog.list_all_rules():
            for cve in rule.cve_ids:
                assert cve_pattern.match(cve), (
                    f"Rule {rule.rule_id} has malformed CVE ID: {cve!r}"
                )

    def test_not_directly_mapped_for_unmapped_catalog_rule(self) -> None:
        # P199 is a valid rule_id pattern but has no catalog entry; the
        # fallback should yield "Not directly mapped".
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="P199",
            rule_name="Hypothetical Rule",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            category="structural_anomaly",
            format=ModelFormat.SAFETENSORS,
            artifact_sha256=_ZERO_SHA256,
            evidence=_make_evidence(),
            description="No catalog entry.",
            remediation="N/A.",
            cwe_ids=(),
            owasp_category=None,
            mitre_technique=None,
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        rf = report.report_findings[0]
        assert rf.owasp_category == "Not directly mapped"

    def test_ml06_for_mapped_catalog_rule(self) -> None:
        # P101 maps to ML06:2023 in the catalog; finding owasp_category=None
        # should fall back to the catalog value.
        f = Finding(
            finding_id=str(uuid.uuid4()),
            rule_id="P101",
            rule_name="Dangerous Global Import",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            category="code_execution",
            format=ModelFormat.RAW_PICKLE,
            artifact_sha256=_ZERO_SHA256,
            evidence=_make_evidence(),
            description="Test finding.",
            remediation="Test remediation.",
            cwe_ids=(),
            owasp_category=None,
            mitre_technique=None,
        )
        sr = _make_scan_result(findings=(f,), exit_code=1)
        report = ReportBuilder.build(sr)
        rf = report.report_findings[0]
        assert rf.owasp_category == "ML06:2023 AI Supply Chain Attacks"


# ---------------------------------------------------------------------------
# 10. Unparseable container / WARN verdict preservation
# ---------------------------------------------------------------------------


class TestUnparseableContainer:
    def test_warn_outcome_preserved_in_report(self) -> None:
        sr = _make_scan_result(scan_outcome=ScanOutcome.WARNING, exit_code=1)
        report = ReportBuilder.build(sr)
        # No CRITICAL/HIGH findings, outcome is WARNING → verdict may be WARN or PASS
        # The scan_outcome is accurately recorded in the report
        assert report.scan_outcome == "WARNING"

    def test_error_outcome_produces_fail_verdict(self) -> None:
        sr = _make_scan_result(
            findings=(),
            scan_outcome=ScanOutcome.ERROR,
            exit_code=2,
        )
        report = ReportBuilder.build(sr)
        assert report.verdict == "FAIL"
        assert report.exit_code == 2

    def test_no_false_pass_on_error(self) -> None:
        sr = _make_scan_result(scan_outcome=ScanOutcome.ERROR, exit_code=2)
        report = ReportBuilder.build(sr)
        assert report.verdict != "PASS"


# ---------------------------------------------------------------------------
# 11. Risk score display
# ---------------------------------------------------------------------------


class TestRiskScoreDisplay:
    def test_risk_score_from_scanner(self) -> None:
        rs = RiskScore(
            raw_score=45.0,
            final_score=45,
            risk_band=RiskBand.MEDIUM_RISK,
            format_multiplier=1.3,
            obfuscation_multiplier=1.0,
            findings_weighted_sum=34.6,
        )
        sr = _make_scan_result(risk_score=rs)
        report = ReportBuilder.build(sr)
        assert report.risk_score_final == 45
        assert report.risk_band == "MEDIUM_RISK"
        assert report.format_multiplier == 1.3

    def test_risk_score_in_json(self) -> None:
        rs = RiskScore(
            raw_score=45.0,
            final_score=45,
            risk_band=RiskBand.MEDIUM_RISK,
            format_multiplier=1.3,
            obfuscation_multiplier=1.0,
            findings_weighted_sum=34.6,
        )
        sr = _make_scan_result(risk_score=rs)
        report = ReportBuilder.build(sr)
        d = ProfessionalJSONExporter.to_dict(report)
        assert d["risk_assessment"]["risk_score"] == 45
        assert d["risk_assessment"]["risk_band"] == "MEDIUM_RISK"

    def test_risk_score_in_html(self) -> None:
        rs = RiskScore(
            raw_score=90.0,
            final_score=90,
            risk_band=RiskBand.CRITICAL_RISK,
            format_multiplier=1.5,
            obfuscation_multiplier=1.0,
            findings_weighted_sum=60.0,
        )
        sr = _make_scan_result(risk_score=rs)
        report = ReportBuilder.build(sr)
        html_str = HTMLReportRenderer.render(report)
        assert "90" in html_str
        assert "CRITICAL" in html_str


# ---------------------------------------------------------------------------
# 12. PDF renderer availability check
# ---------------------------------------------------------------------------


class TestPDFRenderer:
    def test_pdf_import_error_when_unavailable(self, tmp_path) -> None:
        from model_scanner.presentation.pdf_report import PDFReportRenderer

        if PDFReportRenderer.is_available():
            pytest.skip("weasyprint is installed; skipping unavailability test")

        sr = _make_scan_result()
        report = ReportBuilder.build(sr)
        with pytest.raises(ImportError, match="weasyprint"):
            PDFReportRenderer.render(report, tmp_path / "report.pdf")


# ---------------------------------------------------------------------------
# Scan-level reporting (multi-artifact)
# ---------------------------------------------------------------------------


def _make_multi_artifact_scan() -> ScanResult:
    """A folder-style scan: repository + analysed weights + uninspected file."""
    repo = ArtifactScanResult(
        artifact_id=str(uuid.uuid4()),
        name="upload_repo",
        source_origin="/tmp/upload_repo",
        input_source=InputSourceType.LOCAL_DIRECTORY,
        detected_format=ModelFormat.REPOSITORY,
        size_bytes=0,
        sha256=_ZERO_SHA256,
        scan_outcome=ScanOutcome.COMPLETE,
        risk_score=_make_risk_score(0),
        findings=(),
        warnings=(),
        errors=(),
        duration_ms=10,
    )
    weights = _make_artifact(name="pytorch_model.bin")
    unanalyzed = ArtifactScanResult(
        artifact_id=str(uuid.uuid4()),
        name="tf_model.h5",
        source_origin="/tmp/upload_repo/tf_model.h5",
        input_source=InputSourceType.LOCAL_DIRECTORY,
        detected_format=ModelFormat.HDF5,
        size_bytes=2048,
        sha256=_ZERO_SHA256,
        scan_outcome=ScanOutcome.WARNING,
        risk_score=RiskScore.unanalyzed(),
        findings=(),
        warnings=(),
        errors=(),
        duration_ms=5,
    )
    return ScanResult(
        scan_id=_VALID_UUID,
        scanner_version="2.0.0",
        timestamp_utc="2026-08-21T12:00:00+00:00",
        target_request="/tmp/upload_repo",
        strict_mode=False,
        artifacts=(repo, weights, unanalyzed),
        summary=ScanSummary(
            total_artifacts=3,
            scanned_successfully=3,
            failed_artifacts=0,
            unsupported_artifacts=0,
            findings_by_severity={
                "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0,
            },
            artifacts_by_risk_band={"CLEAN": 2, "UNANALYZED": 1},
            max_risk_score=0,
            overall_outcome=ScanOutcome.COMPLETE,
            exit_code=0,
        ),
        total_duration_ms=100,
    )


class TestScanReportBuilder:
    def test_covers_every_artifact(self) -> None:
        """Regression: the HTML report rendered artifact[0] only, so a folder
        scan showed its REPOSITORY entry and dropped every weight file."""
        sr = _make_multi_artifact_scan()
        scan_report = ScanReportBuilder.build(sr)

        assert scan_report.total_artifacts == 3
        assert len(scan_report.artifact_reports) == 3
        names = {r.artifact_name for r in scan_report.artifact_reports}
        assert names == {"upload_repo", "pytorch_model.bin", "tf_model.h5"}

    def test_unanalyzed_artifact_is_counted_and_ranked(self) -> None:
        sr = _make_multi_artifact_scan()
        scan_report = ScanReportBuilder.build(sr)

        assert scan_report.unanalyzed_artifacts == 1
        assert scan_report.has_unanalyzed is True
        # UNANALYZED must outrank PASS: "we did not look" is not a clean pass.
        assert scan_report.overall_verdict == "UNANALYZED"
        assert scan_report.worst_risk_band == "UNANALYZED"

    def test_empty_scan_rejected(self) -> None:
        sr = _make_scan_result()
        empty = ScanResult(
            scan_id=sr.scan_id,
            scanner_version=sr.scanner_version,
            timestamp_utc=sr.timestamp_utc,
            target_request=sr.target_request,
            strict_mode=sr.strict_mode,
            artifacts=(),
            summary=sr.summary,
            total_duration_ms=sr.total_duration_ms,
        )
        with pytest.raises(ValueError):
            ScanReportBuilder.build(empty)


class TestScanHTMLReport:
    def test_html_renders_every_artifact(self) -> None:
        sr = _make_multi_artifact_scan()
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        for name in ("upload_repo", "pytorch_model.bin", "tf_model.h5"):
            assert name in html_out
        assert "Artifact Inventory" in html_out
        assert "Artifact 3 of 3" in html_out

    def test_html_surfaces_unanalyzed_and_never_claims_clean(self) -> None:
        """Regression: an uninspected artifact exported as risk band CLEAN.

        Individual artifact sections may legitimately read PASS — the two
        file-backed artifacts here really are clean. What must never happen is
        the scan-level headline claiming a clean pass while one artifact in the
        scan was never inspected, so this asserts against the headline only.
        """
        sr = _make_multi_artifact_scan()
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        # The scan-level banner is the first verdict-label in the document.
        banner = re.search(r'<span class="verdict-label"[^>]*>([^<]+)<', html_out)
        assert banner is not None
        assert "UNANALYZED" in banner.group(1)
        assert "PASS" not in banner.group(1)

        headline = html_out.split("Artifact Inventory")[0]
        assert "COVERAGE GAP" in headline

    def test_repository_digest_is_labelled_honestly(self) -> None:
        sr = _make_multi_artifact_scan()
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))
        assert "Repository path digest (not a file hash)" in html_out
        assert "virtual repository artifact" in html_out

    def test_verdict_label_matches_finding_count(self) -> None:
        """Regression: 'No Findings Detected' printed alongside 3 INFO findings."""
        info = _make_finding(severity=Severity.INFO, rule_id="P209")
        sr = _make_scan_result(findings=(info,))
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        assert "PASS — Informational Findings Only" in html_out
        assert "PASS — No Findings Detected" not in html_out

    def test_no_findings_still_says_no_findings(self) -> None:
        sr = _make_scan_result()
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))
        assert "PASS — No Findings Detected" in html_out


# ---------------------------------------------------------------------------
# Report presentation consistency (exit-code label, analyzed formats, wording)
# ---------------------------------------------------------------------------


def _make_mixed_format_scan() -> ScanResult:
    """Repository + SafeTensors + PyTorch ZIP + an UNANALYZED HDF5."""
    def _art(
        name: str,
        fmt: ModelFormat,
        outcome: ScanOutcome = ScanOutcome.COMPLETE,
        risk: RiskScore | None = None,
        findings: tuple[Finding, ...] = (),
    ) -> ArtifactScanResult:
        return ArtifactScanResult(
            artifact_id=str(uuid.uuid4()),
            name=name,
            source_origin=f"/tmp/repo/{name}",
            input_source=InputSourceType.LOCAL_DIRECTORY,
            detected_format=fmt,
            size_bytes=1024,
            sha256=_ZERO_SHA256,
            scan_outcome=outcome,
            risk_score=risk or _make_risk_score(0),
            findings=findings,
            warnings=(),
            errors=(),
            duration_ms=5,
        )

    info = _make_finding(severity=Severity.INFO, rule_id="SC106")
    return ScanResult(
        scan_id=_VALID_UUID,
        scanner_version="2.0.0",
        timestamp_utc="2026-08-21T12:00:00+00:00",
        target_request="/tmp/repo",
        strict_mode=False,
        artifacts=(
            _art("repo", ModelFormat.REPOSITORY, findings=(info,)),
            _art("model.safetensors", ModelFormat.SAFETENSORS),
            _art("pytorch_model.bin", ModelFormat.PYTORCH_ZIP),
            _art("extra.safetensors", ModelFormat.SAFETENSORS),
            _art(
                "tf_model.h5",
                ModelFormat.HDF5,
                outcome=ScanOutcome.WARNING,
                risk=RiskScore.unanalyzed(),
            ),
            _art(
                "notes.txt",
                ModelFormat.UNSUPPORTED,
                outcome=ScanOutcome.UNSUPPORTED,
                risk=RiskScore.unanalyzed(),
            ),
        ),
        summary=ScanSummary(
            total_artifacts=6,
            scanned_successfully=5,
            failed_artifacts=0,
            unsupported_artifacts=1,
            findings_by_severity={
                "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 1,
            },
            artifacts_by_risk_band={"CLEAN": 4, "UNANALYZED": 2},
            max_risk_score=0,
            overall_outcome=ScanOutcome.COMPLETE,
            exit_code=1,
        ),
        total_duration_ms=100,
    )


class TestExitCodeLabel:
    """The numeric exit code is ClamAV-compatible and severity-agnostic
    (engine: any findings -> 1). Only the human-readable label may vary."""

    def test_info_only_is_not_called_a_security_finding(self) -> None:
        info = _make_finding(severity=Severity.INFO, rule_id="P209")
        sr = _make_scan_result(findings=(info,), exit_code=1)
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        assert "1 - security findings present" not in html_out
        assert "1 - informational findings present (no actionable findings)" in html_out

    def test_actionable_findings_keep_security_wording(self) -> None:
        crit = _make_finding(severity=Severity.CRITICAL, rule_id="P101")
        sr = _make_scan_result(findings=(crit,), exit_code=1)
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        assert "1 - security findings present" in html_out

    def test_numeric_exit_code_contract_unchanged(self) -> None:
        """Regression guard: the label change must not alter the number."""
        info = _make_finding(severity=Severity.INFO, rule_id="P209")
        sr = _make_scan_result(findings=(info,), exit_code=1)
        assert ScanReportBuilder.build(sr).exit_code == 1

    def test_label_is_consistent_across_every_section(self) -> None:
        sr = _make_mixed_format_scan()
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        # PASS artifacts with zero findings of their own now show the
        # per-artifact "0 - clean" label rather than the scan-level "1" label,
        # so the scan-level informational label appears only for:
        # repo (1 INFO finding, not zero-findings), tf_model.h5 (UNANALYZED),
        # notes.txt (UNSUPPORTED), and the reproducibility block.
        assert html_out.count(
            "1 - informational findings present (no actionable findings)"
        ) == 4
        # Clean zero-finding PASS artifacts should show their own clean label
        assert "0 - clean, no findings" in html_out
        assert "1 - security findings present" not in html_out


class TestAnalyzedFormats:
    def test_lists_every_analyzed_format_deduplicated_and_sorted(self) -> None:
        sr = _make_mixed_format_scan()
        scan_report = ScanReportBuilder.build(sr)
        assert scan_report.formats_analyzed == [
            "PYTORCH_ZIP",
            "REPOSITORY",
            "SAFETENSORS",
        ]

    def test_unanalyzed_and_unsupported_formats_are_excluded(self) -> None:
        """HDF5 was identified, never analyzed; UNSUPPORTED likewise."""
        scan_report = ScanReportBuilder.build(_make_mixed_format_scan())
        assert "HDF5" not in scan_report.formats_analyzed
        assert "UNSUPPORTED" not in scan_report.formats_analyzed

    def test_methodology_section_renders_the_scan_level_list(self) -> None:
        sr = _make_mixed_format_scan()
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))
        assert "PYTORCH_ZIP, REPOSITORY, SAFETENSORS" in html_out

    def test_single_artifact_report_is_unaffected(self) -> None:
        """ReportBuilder.build keeps its per-artifact formats_analyzed."""
        report = ReportBuilder.build(_make_scan_result())
        assert report.formats_analyzed == ["RAW_PICKLE"]


class TestInfoTerminology:
    def test_info_only_scan_states_zero_actionable_findings(self) -> None:
        info = _make_finding(severity=Severity.INFO, rule_id="P209")
        sr = _make_scan_result(findings=(info,), exit_code=1)
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        assert "0 actionable security findings." in html_out
        assert "informational observation" in html_out
        assert "Actionable findings:" in html_out
        assert "Informational observations:" in html_out

    def test_actionable_scan_keeps_normal_findings_terminology(self) -> None:
        crit = _make_finding(severity=Severity.CRITICAL, rule_id="P101")
        sr = _make_scan_result(findings=(crit,), exit_code=1)
        html_out = HTMLReportRenderer.render_scan(ScanReportBuilder.build(sr))

        assert "0 actionable security findings." not in html_out
        assert "FAIL — Security Issues Detected" in html_out
