"""
model_scanner.presentation.professional_report
===============================================

Professional-grade JSON report exporter that uses the ReportModel as the
single source of truth.  This is SEPARATE from the canonical
JSONReportExporter (json_export.py), which remains the authoritative
forensic scanner output.

This module produces a human-readable, report-grade JSON document
with stable key ordering, a schema version, the static-analysis disclaimer,
and redacted evidence.  It does NOT change or replace the canonical
scan-result JSON contract.

SECURITY GUARANTEE
------------------
This module performs PURE IN-MEMORY DATA SERIALIZATION.
It NEVER executes model code, deserializes binary, or makes network requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from model_scanner.presentation.report_model import (
    RedactedEvidence,
    ReportFinding,
    ReportModel,
)


class ProfessionalJSONExporter:
    """Export a ReportModel as a professional-grade JSON assessment report."""

    @staticmethod
    def _serialize_evidence(ev: RedactedEvidence) -> dict[str, Any]:
        return {
            "coordinate_space": ev.coordinate_space,
            "archive_member": ev.archive_member,
            "byte_offset": ev.byte_offset,
            "byte_length": ev.byte_length,
            "matched_value": ev.matched_value,
            "original_length": ev.original_length,
            "opcode": ev.opcode,
            "context_snippet": ev.context_snippet,
        }

    @classmethod
    def _serialize_finding(cls, rf: ReportFinding) -> dict[str, Any]:
        return {
            "rule_id": rf.rule_id,
            "rule_name": rf.rule_name,
            "severity": rf.severity,
            "confidence": rf.confidence,
            "confidence_rationale": rf.confidence_rationale,
            "category": rf.category,
            "occurrence_count": rf.occurrence_count,
            "description": rf.description,
            "security_issue": rf.security_issue,
            "impact": rf.impact,
            "remediation": rf.remediation,
            "threat_mapping": {
                "cwe_ids": list(rf.cwe_ids) if rf.cwe_ids else ["Not mapped"],
                "owasp_category": rf.owasp_category or "Not directly mapped",
                "mitre_technique": rf.mitre_technique or "Not mapped",
            },
            "references": list(rf.references) if rf.references else [],
            "evidence_samples": [
                cls._serialize_evidence(ev) for ev in rf.evidence_samples
            ],
            "additional_occurrences": rf.additional_occurrences,
            "correlated_finding_ids": list(rf.correlated_finding_ids),
        }

    @classmethod
    def to_dict(cls, report: ReportModel) -> dict[str, Any]:
        """Convert a ReportModel to a deterministic, stable-ordered dict."""
        return {
            "report_schema_version": report.report_schema_version,
            "static_analysis_disclaimer": report.static_analysis_disclaimer,
            "header": {
                "scan_id": report.scan_id,
                "scanner_version": report.scanner_version,
                "scan_timestamp": report.scan_timestamp,
                "target_request": report.target_request,
                "strict_mode": report.strict_mode,
                "artifact_name": report.artifact_name,
                "artifact_source": report.artifact_source,
                "input_source_type": report.input_source_type,
                "sha256": report.sha256,
                "size_bytes": report.size_bytes,
                "detected_format": report.detected_format,
                "scan_outcome": report.scan_outcome,
                "exit_code": report.exit_code,
                "total_duration_ms": report.total_duration_ms,
            },
            "verdict": report.verdict,
            "risk_assessment": {
                "risk_score": report.risk_score_final,
                "risk_band": report.risk_band,
                "risk_score_raw": report.risk_score_raw,
                "format_multiplier": report.format_multiplier,
                "obfuscation_multiplier": report.obfuscation_multiplier,
                "findings_weighted_sum": report.findings_weighted_sum,
            },
            "executive_summary": {
                "total_scanner_findings": report.total_scanner_findings,
                "unique_report_findings": report.unique_report_findings,
                "collapsed_findings": report.collapsed_findings,
                "severity_counts": dict(report.severity_counts),
                "confidence_distribution": dict(report.confidence_distribution),
                "dominant_risk_themes": list(report.dominant_risk_themes),
                "correlation_note": report.correlation_note,
            },
            "findings": [cls._serialize_finding(rf) for rf in report.report_findings],
            "methodology": {
                "formats_analyzed": list(report.formats_analyzed),
                "warnings_count": report.warnings_count,
                "errors_count": report.errors_count,
                "notes": list(report.methodology_notes),
                "unparsed_members": list(report.unparsed_members),
                "static_only_limitation": (
                    "The scanner performs static inspection only. Model weights are "
                    "never loaded, deserialized, or executed. Coverage is limited to "
                    "detectable static indicators in the inspected format."
                ),
                "fail_closed_behavior": (
                    "Parsing failures result in a structured error Finding rather than "
                    "a silent skip. Unrecognized types, corrupted streams, and resource "
                    "limit violations all produce explicit findings."
                ),
            },
            "reproducibility": {
                "scan_id": report.scan_id,
                "scanner_version": report.scanner_version,
                "scan_timestamp": report.scan_timestamp,
                "target_request": report.target_request,
                "artifact_sha256": report.sha256,
                "exit_code": report.exit_code,
                "strict_mode": report.strict_mode,
                "total_duration_ms": report.total_duration_ms,
            },
            "model_structure": report.model_structure,
        }

    @classmethod
    def to_json(
        cls,
        report: ReportModel,
        indent: int | None = 2,
    ) -> str:
        """Serialize a ReportModel to a deterministic professional JSON report."""
        payload = cls.to_dict(report)
        return json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=False)

    @classmethod
    def export(
        cls,
        report: ReportModel,
        destination: str | Path | TextIO,
        indent: int | None = 2,
    ) -> str:
        """Export a ReportModel to a file path or writable text stream."""
        json_content = cls.to_json(report, indent=indent)

        if hasattr(destination, "write") and callable(destination.write):
            destination.write(json_content)
        elif isinstance(destination, (str, Path)):
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(json_content, encoding="utf-8")
        else:
            raise TypeError(
                f"Destination must be a str, Path, or writable TextIO stream, "
                f"got {type(destination).__name__}"
            )

        return json_content


__all__ = [
    "ProfessionalJSONExporter",
]
