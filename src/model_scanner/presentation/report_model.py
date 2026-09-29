"""
model_scanner.presentation.report_model
========================================

Read-only derived reporting structure built from a canonical ScanResult.

This module is the single transformation point between the scanner's
authoritative data contract (ScanResult / Finding) and the presentation
layer.  Once a ReportModel is produced, all three output formats —
JSON, HTML, PDF — consume it identically; no format touches ScanResult
directly.

Constraints
-----------
- ScanResult and Finding are NEVER mutated.
- Secret / credential evidence is redacted HERE, before entering ReportModel.
  It is therefore impossible for a secret to appear in any downstream output.
- Deduplication for presentation happens here (scanner finding count unchanged).
- Every rule name / CWE / OWASP / MITRE value is read from the live
  RuleCatalog; nothing is hardcoded.

SECURITY GUARANTEE
------------------
This module NEVER executes model code, deserializes binary, or makes
network requests.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Final

from model_scanner.core.models import (
    ArtifactScanResult,
    Confidence,
    Evidence,
    Finding,
    RiskBand,
    ScanOutcome,
    ScanResult,
    Severity,
)
from model_scanner.rules.catalog import RuleCatalog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATIC_ANALYSIS_DISCLAIMER: Final[str] = (
    "This assessment is based on static analysis. The scanner does not "
    "execute, deserialize, or otherwise run model artifacts. Evidence "
    "represents security-relevant structures observed during static "
    "inspection and does not constitute proof of successful exploitation."
)

REPORT_SCHEMA_VERSION: Final[str] = "1.0.0"

# Maximum evidence samples rendered per deduplicated finding
_MAX_EVIDENCE_SAMPLES: Final[int] = 3

# Rules whose matched_value may contain a secret (credential_exposure category)
_CREDENTIAL_RULE_IDS: Final[frozenset[str]] = frozenset(
    {"ST104", "GU103"}
)

# Minimum total characters required to produce a masked prefix+suffix display
_MASK_MIN_LENGTH: Final[int] = 12

# How many characters to show before / after the mask
_MASK_PREFIX: Final[int] = 6
_MASK_SUFFIX: Final[int] = 4

# Bucket size for approximate-location-based presentation deduplication


# ---------------------------------------------------------------------------
# RedactedEvidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedactedEvidence:
    """Forensic evidence with secrets fully redacted before report entry.

    All fields mirror the canonical Evidence model except ``matched_value``
    which may be masked.  The original value NEVER enters this structure.
    """

    coordinate_space: str
    archive_member: str | None
    byte_offset: int | None
    byte_length: int | None
    matched_value: str           # Possibly masked, never raw secret
    original_length: int | None  # Byte length of the original value
    opcode: str | None
    context_snippet: str | None


# ---------------------------------------------------------------------------
# ReportFinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportFinding:
    """Presentation-layer finding: deduplicated, enriched, secrets-free.

    ``occurrence_count`` is the number of raw scanner findings collapsed into
    this entry.  The first ``_MAX_EVIDENCE_SAMPLES`` evidence objects are
    preserved; further occurrences are summarised via ``additional_occurrences``.
    """

    rule_id: str
    rule_name: str
    severity: str
    confidence: str
    category: str
    description: str
    security_issue: str          # "Why this is security-relevant"
    impact: str
    remediation: str
    cwe_ids: tuple[str, ...]
    owasp_category: str
    mitre_technique: str
    references: tuple[str, ...]
    confidence_rationale: str
    occurrence_count: int
    evidence_samples: tuple[RedactedEvidence, ...]
    additional_occurrences: int  # occurrence_count - len(evidence_samples)
    correlated_finding_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# ReportModel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportModel:
    """Single canonical presentation model consumed by JSON, HTML, and PDF."""

    # ---- report metadata ----
    report_schema_version: str
    static_analysis_disclaimer: str

    # ---- artifact header ----
    artifact_name: str
    artifact_source: str
    input_source_type: str
    sha256: str
    size_bytes: int
    detected_format: str
    scan_outcome: str
    scan_timestamp: str
    scanner_version: str
    target_request: str
    strict_mode: bool

    # ---- verdict ----
    verdict: str            # PASS | WARN | FAIL

    # ---- executive summary ----
    total_scanner_findings: int
    unique_report_findings: int
    collapsed_findings: int
    severity_counts: dict[str, int]
    confidence_distribution: dict[str, int]
    dominant_risk_themes: list[str]

    # ---- risk score ----
    risk_score_final: int
    risk_band: str
    risk_score_raw: float
    format_multiplier: float
    obfuscation_multiplier: float
    findings_weighted_sum: float

    # ---- findings ----
    report_findings: tuple[ReportFinding, ...]

    # ---- correlation summary ----
    correlation_note: str

    # ---- methodology / limitations ----
    formats_analyzed: list[str]
    warnings_count: int
    errors_count: int
    methodology_notes: list[str]
    unparsed_members: list[str]

    # ---- reproducibility ----
    scan_id: str
    total_duration_ms: int
    artifact_duration_ms: int | None
    exit_code: int

    # ---- structural inventory (informational, not a security finding) ----
    model_structure: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mask_secret(value: str) -> tuple[str, int | None]:
    """Return (masked_value, original_length).

    If the value is long enough, produce ``prefix...suffix``.
    Otherwise return ``[redacted]`` with the original length preserved.
    """
    original_length = len(value)
    if original_length >= _MASK_MIN_LENGTH:
        masked = value[:_MASK_PREFIX] + "..." + value[-_MASK_SUFFIX:]
        return masked, original_length
    return "[redacted]", original_length


def _redact_evidence(evidence: Evidence, rule_id: str) -> RedactedEvidence:
    """Convert Evidence to RedactedEvidence, masking secrets where needed."""
    is_secret = rule_id in _CREDENTIAL_RULE_IDS
    if is_secret:
        masked_value, orig_len = _mask_secret(evidence.matched_value)
    else:
        masked_value = evidence.matched_value
        orig_len = None

    return RedactedEvidence(
        coordinate_space=evidence.coordinate_space.value,
        archive_member=evidence.archive_member,
        byte_offset=evidence.byte_offset,
        byte_length=evidence.byte_length,
        matched_value=masked_value,
        original_length=orig_len,
        opcode=evidence.opcode,
        context_snippet=evidence.context_snippet,
    )


def _dedup_key(finding: Finding) -> tuple[str, str | None, str]:
    """Presentation deduplication key.

    Two findings collapse when they are the same rule, in the same archive
    member, about the same matched value. Bucketing by byte offset instead
    merged genuinely different findings that happened to sit near each other
    while splitting identical findings that landed in different buckets.
    """
    ev = finding.evidence
    return (
        finding.rule_id,
        ev.archive_member if ev is not None else None,
        str(ev.matched_value or "") if ev is not None else "",
    )


def _severity_order(sev: str) -> int:
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    return order.get(sev, 99)


def _verdict_for_artifact(artifact: ArtifactScanResult) -> str:
    """Derive PASS / WARN / FAIL / UNSUPPORTED from an artifact result.

    UNSUPPORTED is reported as its own verdict, never as PASS. An artifact
    whose format has no registered analyzer was never inspected, so "no
    findings" carries no assurance - it means no rules were evaluated. Folding
    that into PASS told the reader a legacy non-ZIP PyTorch checkpoint (a raw
    Pickle stream, the highest-risk format there is) was clean when nothing
    had looked at it.
    """
    if artifact.scan_outcome == ScanOutcome.ERROR:
        return "FAIL"
    if artifact.scan_outcome == ScanOutcome.UNSUPPORTED:
        return "UNSUPPORTED"
    # An artifact whose format was identified but for which no analyzer ran
    # (e.g. HDF5, which emits G002 and completes with a WARNING outcome) is
    # banded UNANALYZED by the engine. Its findings say nothing about the
    # artifact's safety, so it must not be graded on them.
    if artifact.risk_score.risk_band == RiskBand.UNANALYZED:
        return "UNANALYZED"
    if not artifact.findings:
        return "PASS"
    severities = {f.severity for f in artifact.findings}
    if Severity.CRITICAL in severities or Severity.HIGH in severities:
        return "FAIL"
    if Severity.MEDIUM in severities or Severity.LOW in severities:
        return "WARN"
    # INFO-only. P209 fires on the archive/data.pkl member present in EVERY
    # PyTorch checkpoint, so returning WARN here reported every legitimate
    # model as "Medium or low severity findings are present" when there
    # none at all.
    return "PASS"


def _dominant_themes(findings: tuple[Finding, ...]) -> list[str]:
    """Extract the top (≤3) security themes by finding count."""
    counter: dict[str, int] = defaultdict(int)
    for f in findings:
        counter[f.category] += 1
    sorted_themes = sorted(counter.items(), key=lambda t: t[1], reverse=True)
    return [t[0] for t in sorted_themes[:3]]


def _severity_counts(findings: tuple[Finding, ...]) -> dict[str, int]:
    counts: dict[str, int] = {
        "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0,
    }
    for f in findings:
        counts[f.severity.value] += 1
    return counts


def _confidence_distribution(findings: tuple[Finding, ...]) -> dict[str, int]:
    dist: dict[str, int] = {
        "HIGH": 0, "MEDIUM": 0, "LOW": 0,
    }
    for f in findings:
        dist[f.confidence.value] += 1
    return dist


# Category-based impact statements distinct from rule descriptions.
# Each entry answers "what harm can result?" for a security category, not
# "what did the scanner detect?" (that is the description / detection_condition).
_CATEGORY_IMPACT: dict[str, str] = {
    "code_execution": (
        "Successful exploitation enables arbitrary code execution on the host system "
        "at model-load time, with the privileges of the loading process."
    ),
    "structural_anomaly": (
        "The structural anomaly may indicate a crafted or corrupted artifact; "
        "loading behavior is undefined and may crash the runtime or enable further exploitation."
    ),
    "credential_exposure": (
        "Credentials embedded in model files can be extracted by any party who obtains "
        "the artifact, enabling unauthorized access to downstream systems."
    ),
    "tampering": (
        "Evidence of tampering indicates the artifact may not match its published source "
        "and could contain malicious modifications to weights or configuration."
    ),
    "format_integrity": (
        "Format integrity violations may crash the loading runtime, trigger memory-safety "
        "bugs, or bypass security checks that rely on well-formed inputs."
    ),
    "supply_chain": (
        "Supply-chain indicators suggest unverified provenance; the artifact may have been "
        "substituted or altered between publication and download."
    ),
    "provenance": (
        "Missing or invalid provenance records prevent verification that the artifact "
        "has not been modified after publication."
    ),
    "weight_integrity": (
        "Altered model weights affect both safety and accuracy; adversarial weight "
        "perturbations can introduce subtle backdoors that activate on specific inputs."
    ),
}


_RULE_SPECIFIC_IMPACT: dict[str, str] = {
    "KR104": (
        "Successful exploitation enables arbitrary file reads from attacker-controlled "
        "paths on the host system at model-load time, potentially exposing sensitive "
        "files accessible to the loading process (CWE-73, CWE-200)."
    ),
}


def _catalog_impact(rule_id: str, finding_description: str = "", category: str = "") -> str:
    """Return a category-scoped impact statement distinct from the description.

    The impact answers "what harm can result?" while the description answers
    "what did the scanner observe?". Generating impact from the category rather
    than repeating the description prevents the three-field Description /
    Security Issue / Impact repetition visible in earlier report versions.
    """
    if rule_id in _RULE_SPECIFIC_IMPACT:
        return _RULE_SPECIFIC_IMPACT[rule_id]
    cat = category
    if not cat and RuleCatalog.has_rule(rule_id):
        cat = RuleCatalog.get_rule(rule_id).category
    impact = _CATEGORY_IMPACT.get(cat)
    if impact:
        return impact
    if finding_description:
        return finding_description
    return "Impact assessment not available."


def _catalog_security_issue(rule_id: str, finding_description: str = "") -> str:
    """Return the detection condition — *what* the scanner observed, not the risk.

    Using the catalog's detection_condition keeps this field distinct from the
    description (which describes the vulnerability) and the impact (which
    describes the consequence), eliminating the three-way repetition where all
    three sections showed identical text.
    """
    if RuleCatalog.has_rule(rule_id):
        return RuleCatalog.get_rule(rule_id).detection_condition
    if finding_description:
        return finding_description
    return "Not available."


def _confidence_rationale(finding: Finding) -> str:
    """Return a short rationale for the confidence rating."""
    conf = finding.confidence
    if conf == Confidence.HIGH:
        return (
            "Deterministic pattern match — the indicator was unambiguously "
            "identified from static byte inspection."
        )
    if conf == Confidence.MEDIUM:
        return (
            "Pattern likely matches the expected indicator; partial context "
            "may limit absolute certainty."
        )
    return "Heuristic or incomplete symbolic resolution was required."


def _catalog_cwe_ids(rule_id: str, finding_cwes: tuple[str, ...]) -> tuple[str, ...]:
    if finding_cwes:
        return finding_cwes
    if RuleCatalog.has_rule(rule_id):
        return tuple(RuleCatalog.get_rule(rule_id).cwe_ids)
    return ()


def _catalog_owasp(rule_id: str, finding_owasp: str | None) -> str:
    if finding_owasp:
        return finding_owasp
    if RuleCatalog.has_rule(rule_id):
        cat = RuleCatalog.get_rule(rule_id).owasp_category
        return cat if cat else "Not directly mapped"
    return "Not directly mapped"


def _catalog_mitre(rule_id: str, finding_mitre: str | None) -> str:
    if finding_mitre:
        return finding_mitre
    if RuleCatalog.has_rule(rule_id):
        tech = RuleCatalog.get_rule(rule_id).mitre_technique
        return tech if tech else "Not mapped"
    return "Not mapped"


def _catalog_references(rule_id: str, cve_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return only CVE identifiers for external reference links.

    CWE, OWASP, and MITRE identifiers belong exclusively in Threat Mapping and
    are excluded here to prevent the References section from duplicating it.
    """
    refs: list[str] = []
    for cve in cve_ids:
        if re.match(r"^CVE-\d{4}-\d{4,}$", cve):
            refs.append(cve)
    if RuleCatalog.has_rule(rule_id):
        rule = RuleCatalog.get_rule(rule_id)
        refs.extend(c for c in rule.cve_ids if c not in refs and re.match(r"^CVE-\d{4}-\d{4,}$", c))
    return tuple(refs)


def _build_report_finding(
    rule_id: str,
    grouped_findings: list[Finding],
) -> ReportFinding:
    """Build a ReportFinding from a group of raw findings with the same dedup key."""
    # Use first finding as representative
    rep = grouped_findings[0]
    count = len(grouped_findings)

    # Collect evidence samples (up to _MAX_EVIDENCE_SAMPLES)
    evidence_samples: list[RedactedEvidence] = []
    for f in grouped_findings[:_MAX_EVIDENCE_SAMPLES]:
        evidence_samples.append(_redact_evidence(f.evidence, rule_id))

    additional = max(0, count - _MAX_EVIDENCE_SAMPLES)

    # Gather all correlated finding IDs across the group
    all_correlated: list[str] = []
    for f in grouped_findings:
        all_correlated.extend(f.correlated_finding_ids)
    unique_correlated = list(dict.fromkeys(all_correlated))  # preserve order, deduplicate

    # Catalog enrichment
    cwe_ids = _catalog_cwe_ids(rule_id, rep.cwe_ids)
    owasp_category = _catalog_owasp(rule_id, rep.owasp_category)
    mitre_technique = _catalog_mitre(rule_id, rep.mitre_technique)
    references = _catalog_references(rule_id, rep.cve_ids)

    return ReportFinding(
        rule_id=rule_id,
        rule_name=rep.rule_name,
        severity=rep.severity.value,
        confidence=rep.confidence.value,
        category=rep.category,
        description=rep.description,
        security_issue=_catalog_security_issue(rule_id, rep.description),
        impact=_catalog_impact(rule_id, rep.description, rep.category),
        remediation=rep.remediation,
        cwe_ids=cwe_ids,
        owasp_category=owasp_category,
        mitre_technique=mitre_technique,
        references=references,
        confidence_rationale=_confidence_rationale(rep),
        occurrence_count=count,
        evidence_samples=tuple(evidence_samples),
        additional_occurrences=additional,
        correlated_finding_ids=tuple(unique_correlated),
    )


_PARSE_ERROR_FINDING_RULES: frozenset[str] = frozenset(
    {"P201", "ST201", "ST209", "GU201"}
)


def _methodology_notes(artifact: ArtifactScanResult) -> list[str]:
    """Collect methodology / limitation notes from warnings and errors."""
    notes: list[str] = [
        "Analysis is static only — model weights and tensors are never loaded.",
        "Format detection uses 64-byte magic-byte sniffing with extension fallback.",
        "Unknown metadata/KV types and unparsed stream regions are skipped (fail-closed).",
        "Pickle streams are disassembled into symbolic opcodes; pickle.loads() is never called.",
        "Detection is blocklist/pattern-based, not behavioral.",
    ]
    for w in artifact.warnings:
        notes.append(f"Warning [{w.warning_code}]: {w.message}")
    # Structural parse errors that produced a finding are already shown in the
    # findings section; avoid adding a second mention in methodology notes.
    has_parse_finding = any(
        f.rule_id in _PARSE_ERROR_FINDING_RULES for f in artifact.findings
    )
    for e in artifact.errors:
        if has_parse_finding and e.error_code == "MALFORMED_ARTIFACT":
            notes.append(
                "Parse/structural error: recorded as a finding above "
                "(see Detailed Security Findings section)."
            )
        else:
            notes.append(f"Error [{e.error_code}]: {e.message}")
    return notes


# ---------------------------------------------------------------------------
# ReportBuilder — public API
# ---------------------------------------------------------------------------


class ReportBuilder:
    """Convert a ScanResult (single-artifact) into a ReportModel.

    For multi-artifact scans the caller must choose which artifact to
    report.  Pass ``artifact_index=0`` for the first (or only) artifact.
    """

    @staticmethod
    def build(
        scan_result: ScanResult,
        artifact_index: int = 0,
    ) -> ReportModel:
        """Derive a read-only ReportModel from a ScanResult.

        The canonical ScanResult and its Findings are NEVER mutated.

        Parameters
        ----------
        scan_result:
            The authoritative scan output from ScanOrchestrator.
        artifact_index:
            Zero-based index of the artifact to report (default: 0).

        Raises
        ------
        IndexError
            If artifact_index is out of range.
        ValueError
            If scan_result has no artifacts.
        """
        if not scan_result.artifacts:
            raise ValueError("ScanResult has no artifacts to report.")
        if artifact_index >= len(scan_result.artifacts):
            raise IndexError(
                f"artifact_index {artifact_index} out of range "
                f"(scan has {len(scan_result.artifacts)} artifact(s))"
            )

        artifact = scan_result.artifacts[artifact_index]
        all_findings = artifact.findings

        # ---- presentation deduplication ----
        grouped: dict[tuple[str, str | None, str], list[Finding]] = defaultdict(list)
        for f in all_findings:
            grouped[_dedup_key(f)].append(f)

        # Build report findings sorted by severity then rule_id
        report_findings_list: list[ReportFinding] = []
        for key, group in grouped.items():
            rule_id = key[0]
            report_findings_list.append(_build_report_finding(rule_id, group))

        report_findings_list.sort(
            key=lambda rf: (_severity_order(rf.severity), rf.rule_id)
        )

        total_scanner = len(all_findings)
        unique_report = len(report_findings_list)
        collapsed = total_scanner - unique_report

        # ---- verdict ----
        verdict = _verdict_for_artifact(artifact)

        # ---- risk score ----
        rs = artifact.risk_score

        # ---- dominant themes ----
        themes = _dominant_themes(all_findings)

        # ---- correlation note ----
        correlated_any = any(f.correlated_finding_ids for f in all_findings)
        correlation_note = (
            "Correlated security findings are present (see individual finding details)."
            if correlated_any
            else "No correlated security findings were observed."
        )

        # ---- methodology ----
        notes = _methodology_notes(artifact)
        unparsed: list[str] = [
            w.context or w.message
            for w in artifact.warnings
            if "unparsed" in w.message.lower() or "unsupported" in w.message.lower()
        ]

        return ReportModel(
            report_schema_version=REPORT_SCHEMA_VERSION,
            static_analysis_disclaimer=STATIC_ANALYSIS_DISCLAIMER,
            artifact_name=artifact.name,
            artifact_source=artifact.source_origin,
            input_source_type=artifact.input_source.value,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            detected_format=artifact.detected_format.value,
            scan_outcome=artifact.scan_outcome.value,
            scan_timestamp=scan_result.timestamp_utc,
            scanner_version=scan_result.scanner_version,
            target_request=scan_result.target_request,
            strict_mode=scan_result.strict_mode,
            verdict=verdict,
            total_scanner_findings=total_scanner,
            unique_report_findings=unique_report,
            collapsed_findings=collapsed,
            severity_counts=_severity_counts(all_findings),
            confidence_distribution=_confidence_distribution(all_findings),
            dominant_risk_themes=themes,
            risk_score_final=rs.final_score,
            risk_band=rs.risk_band.value,
            risk_score_raw=rs.raw_score,
            format_multiplier=rs.format_multiplier,
            obfuscation_multiplier=rs.obfuscation_multiplier,
            findings_weighted_sum=rs.findings_weighted_sum,
            report_findings=tuple(report_findings_list),
            correlation_note=correlation_note,
            formats_analyzed=[artifact.detected_format.value],
            warnings_count=len(artifact.warnings),
            errors_count=len(artifact.errors),
            methodology_notes=notes,
            unparsed_members=unparsed,
            scan_id=scan_result.scan_id,
            total_duration_ms=scan_result.total_duration_ms,
            artifact_duration_ms=artifact.duration_ms,
            exit_code=scan_result.summary.exit_code,
            model_structure=artifact.model_structure,
        )


# ---------------------------------------------------------------------------
# Scan-level report — multi-artifact reporting
# ---------------------------------------------------------------------------

# Worst-first ordering. UNANALYZED outranks PASS deliberately: "we did not
# look" is a weaker statement than "we looked and found nothing", and a scan
# containing an uninspected artifact must never present as a clean pass.
_VERDICT_RANK: Final[dict[str, int]] = {
    "PASS": 0,
    "UNANALYZED": 1,
    "UNSUPPORTED": 2,
    "WARN": 3,
    "FAIL": 4,
}

_BAND_RANK: Final[dict[str, int]] = {
    "CLEAN": 0,
    "UNANALYZED": 1,
    "LOW_RISK": 2,
    "MEDIUM_RISK": 3,
    "HIGH_RISK": 4,
    "CRITICAL_RISK": 5,
}


@dataclass(frozen=True)
class ScanReportModel:
    """Whole-scan presentation model wrapping one ReportModel per artifact.

    ReportModel is per-artifact by construction. Reporting a multi-artifact
    scan through a single ReportModel silently dropped every artifact after
    the first, so a folder scan rendered only its REPOSITORY artifact and the
    model weight files never appeared at all.
    """

    report_schema_version: str
    static_analysis_disclaimer: str

    scan_id: str
    scanner_version: str
    scan_timestamp: str
    target_request: str
    strict_mode: bool
    total_duration_ms: int
    exit_code: int

    artifact_reports: tuple[ReportModel, ...]

    overall_verdict: str
    worst_risk_band: str
    max_risk_score: int

    total_artifacts: int
    scanned_successfully: int
    unsupported_artifacts: int
    failed_artifacts: int
    unanalyzed_artifacts: int

    total_scanner_findings: int
    severity_counts: dict[str, int]
    formats_analyzed: list[str]

    @property
    def has_unanalyzed(self) -> bool:
        """True when at least one artifact was never inspected."""
        return self.unanalyzed_artifacts > 0


class ScanReportBuilder:
    """Build a ScanReportModel covering every artifact in a ScanResult."""

    @staticmethod
    def build(scan_result: ScanResult) -> ScanReportModel:
        """Derive a whole-scan ReportModel from a ScanResult.

        Every artifact is rendered through the existing per-artifact
        ReportBuilder, so per-artifact behaviour is unchanged.

        Raises
        ------
        ValueError
            If scan_result has no artifacts.
        """
        if not scan_result.artifacts:
            raise ValueError("ScanResult has no artifacts to report.")

        reports = tuple(
            ReportBuilder.build(scan_result, i)
            for i in range(len(scan_result.artifacts))
        )

        overall_verdict = max(
            (r.verdict for r in reports),
            key=lambda v: _VERDICT_RANK.get(v, 0),
        )
        worst_band = max(
            (r.risk_band for r in reports),
            key=lambda b: _BAND_RANK.get(b, 0),
        )

        severity_counts: dict[str, int] = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0,
        }
        for r in reports:
            for sev, count in r.severity_counts.items():
                if sev in severity_counts:
                    severity_counts[sev] += count

        summary = scan_result.summary
        unanalyzed = sum(
            1 for r in reports if r.risk_band == RiskBand.UNANALYZED.value
        )

        # Formats an analyzer actually inspected. ReportModel.formats_analyzed
        # is per-artifact, so a multi-artifact scan previously reported only the
        # first artifact's format and claimed a folder containing SafeTensors
        # and PyTorch weights had analyzed "REPOSITORY" alone. Artifacts banded
        # UNANALYZED are excluded: their format was identified, not analyzed.
        formats_analyzed = sorted(
            {
                r.detected_format
                for r in reports
                if r.risk_band != RiskBand.UNANALYZED.value
            }
        )

        return ScanReportModel(
            report_schema_version=REPORT_SCHEMA_VERSION,
            static_analysis_disclaimer=STATIC_ANALYSIS_DISCLAIMER,
            scan_id=scan_result.scan_id,
            scanner_version=scan_result.scanner_version,
            scan_timestamp=scan_result.timestamp_utc,
            target_request=scan_result.target_request,
            strict_mode=scan_result.strict_mode,
            total_duration_ms=scan_result.total_duration_ms,
            exit_code=summary.exit_code,
            artifact_reports=reports,
            overall_verdict=overall_verdict,
            worst_risk_band=worst_band,
            max_risk_score=summary.max_risk_score,
            total_artifacts=summary.total_artifacts,
            scanned_successfully=summary.scanned_successfully,
            unsupported_artifacts=summary.unsupported_artifacts,
            failed_artifacts=summary.failed_artifacts,
            unanalyzed_artifacts=unanalyzed,
            total_scanner_findings=sum(r.total_scanner_findings for r in reports),
            severity_counts=severity_counts,
            formats_analyzed=formats_analyzed,
        )


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "STATIC_ANALYSIS_DISCLAIMER",
    "RedactedEvidence",
    "ReportBuilder",
    "ReportFinding",
    "ReportModel",
    "ScanReportBuilder",
    "ScanReportModel",
]
