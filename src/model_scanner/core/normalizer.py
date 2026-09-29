"""
model_scanner.core.normalizer
=============================

Finding normalization and multi-signal correlation engine for the static
security scanner pipeline.

Pipeline Role
-------------
1. Validates and normalizes raw analyzer findings against the canonical
   data contracts (Rule ID schema, CoordinateSpace, bounded Evidence).
2. Assigns deterministic finding IDs: FINDING-{rule_id}-{sha256[:8]}-{index:03d}.
3. Applies strict-mode severity adjustments (e.g. P103 MEDIUM -> HIGH).
4. Generates G001 format spoofing findings when format mismatch is detected.
5. Performs multi-signal finding correlation across detection layers.
6. Enforces deterministic coordinate-based deduplication.

SECURITY GUARANTEE
------------------
This module performs PURE IN-MEMORY DATA TRANSFORMATIONS.
It NEVER:
  - Calls pickle.loads(), torch.load(), eval(), exec()
  - Imports torch, transformers, safetensors, gguf
  - Performs network, socket, or filesystem I/O

Design Reference
----------------
- Section 5 (Finding Normalization), Section 6 (Severity Normalization),
  Section 8 (Finding Correlation) of docs/phase7_design.md
- Section 4, Section 20 of docs/implementation_design.md
- Section 4.1, SC-09 of docs/project_scope.md
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from model_scanner.core.models import (
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    FormatDetectionResult,
    Severity,
)
from model_scanner.rules.catalog import RuleCatalog

# ---------------------------------------------------------------------------
# Canonical Taxonomies & Validation Patterns
# ---------------------------------------------------------------------------

_RULE_ID_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1|MS1)\d{2}$"
)

CANONICAL_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "code_execution",
        "structural_anomaly",
        "credential_exposure",
        "tampering",
        "format_integrity",
        "supply_chain",   # SC1xx/SC2xx supply chain config and dependency findings
        "provenance",     # SC1xx provenance indicator findings
        "weight_integrity",  # P2: WI1xx rules
    }
)

MAX_MATCHED_VALUE_LEN: Final[int] = 512


# ---------------------------------------------------------------------------
# Finding Normalizer — Phase 7.3
# ---------------------------------------------------------------------------


class FindingNormalizer:
    """Validates, sanitizes, and normalizes findings to standard contracts."""

    @staticmethod
    def validate_rule_id(rule_id: str) -> str:
        """Validate that rule_id conforms to the canonical pattern.

        Parameters
        ----------
        rule_id:
            The rule ID string to validate.

        Returns
        -------
        str
            Validated rule_id.

        Raises
        ------
        ValueError
            If rule_id does not match the canonical regex.
        """
        if not isinstance(rule_id, str) or not _RULE_ID_REGEX.match(rule_id):
            raise ValueError(
                f"Invalid rule_id {rule_id!r}: "
                f"must match pattern ^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1)\\d{{2}}$"
            )
        return rule_id

    @staticmethod
    def normalize_evidence(evidence: Evidence) -> Evidence:
        """Validate and sanitize Evidence forensic coordinates.

        Parameters
        ----------
        evidence:
            The Evidence instance to validate.

        Returns
        -------
        Evidence
            Normalized and sanitized Evidence instance.

        Raises
        ------
        ValueError
            If evidence coordinate constraints are violated.
        """
        if not isinstance(evidence.coordinate_space, CoordinateSpace):
            raise ValueError(
                f"Invalid coordinate_space {evidence.coordinate_space!r}: "
                f"must be an instance of CoordinateSpace"
            )

        if evidence.byte_offset is not None and evidence.byte_offset < 0:
            raise ValueError(
                f"Invalid byte_offset {evidence.byte_offset}: must be non-negative"
            )

        # Archive member invariant: must be populated iff ARCHIVE_MEMBER
        if evidence.coordinate_space == CoordinateSpace.ARCHIVE_MEMBER:
            if not evidence.archive_member:
                raise ValueError(
                    "archive_member must be populated when coordinate_space is ARCHIVE_MEMBER"
                )
        else:
            if evidence.archive_member is not None:
                raise ValueError(
                    f"archive_member must be None when coordinate_space is "
                    f"{evidence.coordinate_space.value}, got {evidence.archive_member!r}"
                )

        # Matched value length bounding
        matched_value = evidence.matched_value
        if matched_value is not None and len(matched_value) > MAX_MATCHED_VALUE_LEN:
            matched_value = matched_value[:MAX_MATCHED_VALUE_LEN]

        if matched_value == evidence.matched_value:
            return evidence

        return Evidence(
            coordinate_space=evidence.coordinate_space,
            byte_offset=evidence.byte_offset,
            byte_length=evidence.byte_length,
            archive_member=evidence.archive_member,
            matched_value=matched_value,
            opcode=evidence.opcode,
            raw_hex=evidence.raw_hex,
            context_snippet=evidence.context_snippet,
        )

    @staticmethod
    def adjust_severity(
        rule_id: str,
        severity: Severity,
        strict_mode: bool = False,
    ) -> Severity:
        """Apply strict-mode dynamic severity adjustments.

        Parameters
        ----------
        rule_id:
            Rule ID of the finding.
        severity:
            Baseline severity.
        strict_mode:
            Whether strict mode is enabled.

        Returns
        -------
        Severity
            Adjusted severity.
        """
        if strict_mode:
            # P103 (unresolvable / unknown global) escalates from MEDIUM to HIGH in strict mode
            if rule_id == "P103" and severity == Severity.MEDIUM:
                return Severity.HIGH
        return severity

    @classmethod
    def deduplicate(cls, findings: Sequence[Finding]) -> tuple[Finding, ...]:
        """Deduplicate findings deterministically based on rule and forensic coordinates.

        Preserves encounter order. Two findings at distinct byte offsets remain separate.

        Parameters
        ----------
        findings:
            Input sequence of findings.

        Returns
        -------
        tuple[Finding, ...]
            Deduplicated sequence of findings.
        """
        seen: set[tuple[str, str, int | None, str | None, str | None]] = set()
        unique: list[Finding] = []

        for f in findings:
            key = (
                f.rule_id,
                f.evidence.coordinate_space.value if f.evidence else "",
                f.evidence.byte_offset if f.evidence else None,
                f.evidence.archive_member if f.evidence else None,
                f.evidence.matched_value if f.evidence else None,
            )
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return tuple(unique)

    @classmethod
    def normalize_finding(
        cls,
        finding: Finding,
        sha256: str,
        index: int,
        strict_mode: bool = False,
    ) -> Finding:
        """Normalize and validate a single Finding instance.

        Parameters
        ----------
        finding:
            The Finding to normalize.
        sha256:
            The artifact SHA-256 hash (64 hex characters).
        index:
            The 1-based deterministic index of the finding.
        strict_mode:
            Whether strict mode severity adjustments apply.

        Returns
        -------
        Finding
            Validated and normalized Finding.
        """
        rule_id = cls.validate_rule_id(finding.rule_id)
        evidence = cls.normalize_evidence(finding.evidence)
        severity = cls.adjust_severity(rule_id, finding.severity, strict_mode=strict_mode)

        # Deterministic finding ID schema: FINDING-{rule_id}-{sha256[:8]}-{index:03d}
        sha_prefix = sha256[:8].lower() if sha256 else "00000000"
        expected_finding_id = f"FINDING-{rule_id}-{sha_prefix}-{index:03d}"

        category = finding.category
        if category not in CANONICAL_CATEGORIES:
            category = "structural_anomaly" if "struct" in category.lower() else category

        cwe_ids = finding.cwe_ids
        owasp_category = finding.owasp_category
        mitre_technique = finding.mitre_technique

        if RuleCatalog.has_rule(rule_id):
            rule = RuleCatalog.get_rule(rule_id)
            if not cwe_ids:
                cwe_ids = tuple(rule.cwe_ids)
            if not owasp_category:
                owasp_category = rule.owasp_category
            if not mitre_technique:
                mitre_technique = rule.mitre_technique

        return Finding(
            finding_id=expected_finding_id,
            rule_id=rule_id,
            rule_name=finding.rule_name,
            severity=severity,
            confidence=finding.confidence,
            category=category,
            format=finding.format,
            artifact_sha256=sha256 if sha256 else finding.artifact_sha256,
            evidence=evidence,
            description=finding.description,
            remediation=finding.remediation,
            cve_ids=finding.cve_ids,
            cwe_ids=cwe_ids,
            owasp_category=owasp_category,
            mitre_technique=mitre_technique,
            correlated_finding_ids=finding.correlated_finding_ids,
        )

    @classmethod
    def normalize_findings(
        cls,
        findings: Sequence[Finding],
        sha256: str,
        strict_mode: bool = False,
    ) -> tuple[Finding, ...]:
        """Normalize, deduplicate, and re-index a sequence of findings.

        Parameters
        ----------
        findings:
            Raw findings sequence from format analyzers.
        sha256:
            Artifact SHA-256 hash string.
        strict_mode:
            Whether strict mode is enabled.

        Returns
        -------
        tuple[Finding, ...]
            Normalized and deterministically ordered findings.
        """
        unique_findings = cls.deduplicate(findings)
        normalized: list[Finding] = []

        for index, raw_finding in enumerate(unique_findings, start=1):
            normalized_finding = cls.normalize_finding(
                raw_finding,
                sha256=sha256,
                index=index,
                strict_mode=strict_mode,
            )
            normalized.append(normalized_finding)

        return tuple(normalized)


# ---------------------------------------------------------------------------
# Finding Correlator — Phase 7.3 Multi-Signal Rule Correlation
# ---------------------------------------------------------------------------


class FindingCorrelator:
    """Performs multi-signal finding correlation across disparate layers."""

    @staticmethod
    def generate_g001_finding(
        detection_result: FormatDetectionResult,
        sha256: str,
        index: int = 1,
    ) -> Finding:
        """Generate a G001 Format Spoofing finding for format mismatches.

        Parameters
        ----------
        detection_result:
            Result from FormatDispatcher indicating format mismatch.
        sha256:
            Artifact SHA-256 hex string.
        index:
            Index for deterministic finding ID formatting.

        Returns
        -------
        Finding
            Canonical G001 finding.
        """
        sha_prefix = sha256[:8].lower() if sha256 else "00000000"
        finding_id = f"FINDING-G001-{sha_prefix}-{index:03d}"

        matched_str = (
            f"detected_format={detection_result.detected_format.value}, "
            f"details={detection_result.details}"
        )
        if len(matched_str) > MAX_MATCHED_VALUE_LEN:
            matched_str = matched_str[:MAX_MATCHED_VALUE_LEN]

        evidence = Evidence(
            coordinate_space=CoordinateSpace.ARTIFACT,
            byte_offset=0,
            byte_length=16,
            archive_member=None,
            matched_value=matched_str,
            context_snippet=detection_result.details or "File extension contradicts magic bytes",
        )

        return Finding(
            finding_id=finding_id,
            rule_id="G001",
            rule_name="Format Mismatch / Extension Spoofing",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            category="format_integrity",
            format=detection_result.detected_format,
            artifact_sha256=sha256,
            evidence=evidence,
            description=(
                f"File extension does not match detected binary format "
                f"({detection_result.detected_format.value}). Possible format spoofing attack."
            ),
            remediation=(
                "Do not rely on file extensions for security boundaries. "
                "Inspect file headers and verify origin."
            ),
        )

    @classmethod
    def correlate(
        cls,
        findings: Sequence[Finding],
        detection_result: FormatDetectionResult | None = None,
        sha256: str = "",
        strict_mode: bool = False,
    ) -> tuple[Finding, ...]:
        """Correlate, augment with format findings, and normalize all findings.

        Parameters
        ----------
        findings:
            Raw findings emitted by format analyzers.
        detection_result:
            Format detection result from FormatDispatcher.
        sha256:
            Artifact SHA-256 fingerprint.
        strict_mode:
            Whether strict mode is enabled.

        Returns
        -------
        tuple[Finding, ...]
            Normalized and correlated sequence of findings.
        """
        combined_findings: list[Finding] = list(findings)

        # 1. Format Spoofing Correlation: G001 generation
        if detection_result and detection_result.format_mismatch:
            g001_finding = cls.generate_g001_finding(
                detection_result=detection_result,
                sha256=sha256,
                index=1,
            )
            # Insert G001 at the beginning so high-priority format integrity is prominent
            combined_findings.insert(0, g001_finding)

        # 2. Normalize, deduplicate, and re-index all findings deterministically
        normalized = FindingNormalizer.normalize_findings(
            combined_findings,
            sha256=sha256,
            strict_mode=strict_mode,
        )

        return normalized


__all__ = [
    "CANONICAL_CATEGORIES",
    "MAX_MATCHED_VALUE_LEN",
    "FindingCorrelator",
    "FindingNormalizer",
]
