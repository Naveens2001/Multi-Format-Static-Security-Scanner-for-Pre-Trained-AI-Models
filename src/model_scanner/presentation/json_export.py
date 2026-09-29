"""
model_scanner.presentation.json_export
======================================

Deterministic Canonical JSON forensic report exporter conforming directly
to the ScanResult data contract and Section 10 of docs/phase7_design.md.

SECURITY GUARANTEE
------------------
This module performs PURE IN-MEMORY DATA SERIALIZATION and FILE I/O.
It NEVER:
  - Invokes pickle.loads(), torch.load(), eval(), exec()
  - Imports torch, transformers, safetensors, gguf
  - Performs network or dynamic execution

Design Reference
----------------
- Section 10 (JSON Report Schema) of docs/phase7_design.md
- Section 24 (JSON Report Exporter) of docs/implementation_design.md
- Section 39.1 (F-11) of docs/implementation_design.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from model_scanner.core.models import (
    ArtifactScanResult,
    Evidence,
    Finding,
    RiskScore,
    ScanError,
    ScanResult,
    ScanSummary,
    ScanWarning,
)



class JSONReportExporter:
    """Canonical JSON forensic report exporter for ScanResult data models."""

    @staticmethod
    def _serialize_evidence(evidence: Evidence | None) -> dict[str, Any] | None:
        """Serialize an Evidence dataclass into a JSON-compatible dictionary."""
        if evidence is None:
            return None

        return {
            "coordinate_space": evidence.coordinate_space.value,
            "byte_offset": evidence.byte_offset,
            "byte_length": evidence.byte_length,
            "archive_member": evidence.archive_member,
            "matched_value": evidence.matched_value,
            "opcode": evidence.opcode,
            "raw_hex": evidence.raw_hex,
            "context_snippet": evidence.context_snippet,
        }

    @classmethod
    def _serialize_finding(cls, finding: Finding) -> dict[str, Any]:
        """Serialize a Finding dataclass into a JSON-compatible dictionary."""
        return {
            "finding_id": finding.finding_id,
            "rule_id": finding.rule_id,
            "rule_name": finding.rule_name,
            "severity": finding.severity.value,
            "confidence": finding.confidence.value,
            "category": finding.category,
            "format": finding.format.value,
            "artifact_sha256": finding.artifact_sha256,
            "description": finding.description,
            "remediation": finding.remediation,
            "cve_ids": list(finding.cve_ids),
            "cwe_ids": list(finding.cwe_ids),
            "owasp_category": finding.owasp_category,
            "mitre_technique": finding.mitre_technique,
            "correlated_finding_ids": list(finding.correlated_finding_ids),
            "evidence": cls._serialize_evidence(finding.evidence),
        }

    @staticmethod
    def _serialize_risk_score(risk_score: RiskScore) -> dict[str, Any]:
        """Serialize a RiskScore dataclass into a JSON-compatible dictionary."""
        return {
            "raw_score": risk_score.raw_score,
            "final_score": risk_score.final_score,
            "risk_band": risk_score.risk_band.value,
            "format_multiplier": risk_score.format_multiplier,
            "obfuscation_multiplier": risk_score.obfuscation_multiplier,
            "findings_weighted_sum": risk_score.findings_weighted_sum,
        }

    @staticmethod
    def _serialize_warning(warning: ScanWarning) -> dict[str, Any]:
        """Serialize a ScanWarning dataclass into a JSON-compatible dictionary."""
        return {
            "warning_code": warning.warning_code,
            "message": warning.message,
            "context": warning.context,
        }

    @staticmethod
    def _serialize_error(error: ScanError) -> dict[str, Any]:
        """Serialize a ScanError dataclass into a JSON-compatible dictionary."""
        return {
            "error_code": error.error_code,
            "message": error.message,
            "exception_type": error.exception_type,
            "traceback_summary": error.traceback_summary,
        }

    @classmethod
    def _serialize_artifact(cls, artifact: ArtifactScanResult) -> dict[str, Any]:
        """Serialize an ArtifactScanResult dataclass into a JSON-compatible dictionary."""
        return {
            "artifact_id": artifact.artifact_id,
            "name": artifact.name,
            "source_origin": artifact.source_origin,
            "input_source": artifact.input_source.value,
            "detected_format": artifact.detected_format.value,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "scan_outcome": artifact.scan_outcome.value,
            "duration_ms": artifact.duration_ms,
            "risk_score": cls._serialize_risk_score(artifact.risk_score),
            "findings": [cls._serialize_finding(f) for f in artifact.findings],
            "warnings": [cls._serialize_warning(w) for w in artifact.warnings],
            "errors": [cls._serialize_error(e) for e in artifact.errors],
        }

    @staticmethod
    def _serialize_summary(summary: ScanSummary) -> dict[str, Any]:
        """Serialize a ScanSummary dataclass into a JSON-compatible dictionary."""
        return {
            "total_artifacts": summary.total_artifacts,
            "scanned_successfully": summary.scanned_successfully,
            "failed_artifacts": summary.failed_artifacts,
            "unsupported_artifacts": summary.unsupported_artifacts,
            "coverage_gap_artifacts": summary.coverage_gap_artifacts,
            "max_risk_score": summary.max_risk_score,
            "overall_outcome": summary.overall_outcome.value,
            "exit_code": summary.exit_code,
            "findings_by_severity": dict(summary.findings_by_severity),
            "artifacts_by_risk_band": dict(summary.artifacts_by_risk_band),
        }

    @classmethod
    def to_dict(cls, scan_result: ScanResult) -> dict[str, Any]:
        """Convert a ScanResult into a deterministic dictionary conforming to Section 10.

        Parameters
        ----------
        scan_result:
            The ScanResult instance to serialize.

        Returns
        -------
        dict[str, Any]
            Canonical JSON-compatible dictionary.
        """
        if not isinstance(scan_result, ScanResult):
            raise TypeError(
                f"Expected ScanResult instance, got {type(scan_result).__name__}"
            )

        return {
            "scan_id": scan_result.scan_id,
            "scanner_version": scan_result.scanner_version,
            "timestamp_utc": scan_result.timestamp_utc,
            "target_request": scan_result.target_request,
            "strict_mode": scan_result.strict_mode,
            "total_duration_ms": scan_result.total_duration_ms,
            "summary": cls._serialize_summary(scan_result.summary),
            "artifacts": [cls._serialize_artifact(a) for a in scan_result.artifacts],
        }

    @classmethod
    def to_json(
        cls,
        scan_result: ScanResult,
        indent: int | None = 2,
    ) -> str:
        """Serialize a ScanResult into a deterministic canonical JSON string.

        Parameters
        ----------
        scan_result:
            The ScanResult instance to serialize.
        indent:
            Number of spaces for indentation (defaults to 2).

        Returns
        -------
        str
            Formatted UTF-8 JSON string.
        """
        payload = cls.to_dict(scan_result)
        return json.dumps(payload, indent=indent, ensure_ascii=False)

    @classmethod
    def export(
        cls,
        scan_result: ScanResult,
        destination: str | Path | TextIO,
        indent: int | None = 2,
    ) -> str:
        """Export a ScanResult to a file path or writable text stream.

        Parameters
        ----------
        scan_result:
            The ScanResult instance to serialize.
        destination:
            A file path (str or Path) or a writable text stream (TextIO).
        indent:
            JSON indentation level (defaults to 2).

        Returns
        -------
        str
            The exported JSON string.

        Raises
        ------
        TypeError
            If destination is neither a path nor a writable text stream.
        """
        json_content = cls.to_json(scan_result, indent=indent)

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


# Alias for backward/forward naming compatibility
JsonReportExporter = JSONReportExporter

__all__ = [
    "JSONReportExporter",
    "JsonReportExporter",
]
