"""
model_scanner.rules.models
==========================

Core data models for the centralized security rule catalog (Phase 8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from model_scanner.core.models import Severity

_RULE_ID_PATTERN = re.compile(r"^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1|MS1)\d{2}$")

# Kept in sync with ``core.normalizer.CANONICAL_CATEGORIES`` (plus
# ``format_integrity``). Duplicated rather than imported because the normalizer
# imports RuleCatalog, and importing it back here would be circular.
_CANONICAL_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "code_execution",
        "structural_anomaly",
        "credential_exposure",
        "tampering",
        "format_integrity",
        "supply_chain",
        "provenance",
        "weight_integrity",  # P2: WI1xx rules
    }
)

_CWE_PATTERN = re.compile(r"^CWE-\d+$")
_CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$")


@dataclass(frozen=True)
class SecurityRule:
    r"""Immutable definition of a static analysis security rule.

    Attributes
    ----------
    rule_id:
        Unique rule identifier conforming to ``^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0)\d{2}$``.
    name:
        Human-readable title describing the security issue.
    severity:
        Default severity an analyzer assigns to a finding for this rule. Some
        rules escalate per finding (e.g. P103/P107 on a blocklisted symbol), and
        strict mode escalates further during normalization.
    category:
        Security category classification.
    description:
        Detailed technical explanation of the security vulnerability.
    detection_condition:
        Specific opcode pattern, metadata signature, or structural condition triggering the rule.
    remediation:
        Actionable guidance for fixing or mitigating the vulnerability.
    affected_format:
        Target format(s) (e.g. 'RAW_PICKLE', 'PYTORCH_ZIP', 'SAFETENSORS', 'GGUF', 'ALL').
    cve_ids:
        Associated Common Vulnerabilities and Exposures (CVE) identifiers. Empty
        for every rule in the shipped catalog: these are detection heuristics for
        malicious artifacts, not implementation flaws with an NVD entry. Do not
        populate it with a placeholder.
    cwe_ids:
        Associated Common Weakness Enumeration (CWE) identifiers.
    owasp_category:
        Mapped OWASP Machine Learning Security Top Ten (2023 edition) category,
        e.g. ``"ML06:2023 AI Supply Chain Attacks"``. ``None`` when no ML OWASP
        category applies to the rule.
    mitre_technique:
        Mapped MITRE ATT&CK technique, formatted ``"<ID>: <official technique
        name>"``. ``None`` when no technique honestly applies.
    """

    rule_id: str
    name: str
    severity: Severity
    category: str
    description: str
    detection_condition: str
    remediation: str
    affected_format: str
    cve_ids: tuple[str, ...] = ()
    cwe_ids: tuple[str, ...] = ()
    owasp_category: str | None = None
    mitre_technique: str | None = None

    def __post_init__(self) -> None:
        if not _RULE_ID_PATTERN.match(self.rule_id):
            raise ValueError(
                f"SecurityRule.rule_id {self.rule_id!r} does not match required pattern "
                r"^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1)\d{2}$"
            )
        if not isinstance(self.severity, Severity):
            raise ValueError(
                f"SecurityRule.severity for {self.rule_id} must be a Severity enum member, "
                f"got {self.severity!r}"
            )
        if self.category not in _CANONICAL_CATEGORIES:
            raise ValueError(
                f"SecurityRule.category {self.category!r} for {self.rule_id} is not one of "
                f"{sorted(_CANONICAL_CATEGORIES)}"
            )
        for cwe in self.cwe_ids:
            if not _CWE_PATTERN.match(cwe):
                raise ValueError(
                    f"SecurityRule.cwe_ids for {self.rule_id} contains {cwe!r}; "
                    "expected the form 'CWE-<number>'"
                )
        for cve in self.cve_ids:
            if not _CVE_PATTERN.match(cve):
                raise ValueError(
                    f"SecurityRule.cve_ids for {self.rule_id} contains {cve!r}; "
                    "expected the form 'CVE-<year>-<number>'. Leave the tuple empty "
                    "rather than recording a placeholder identifier."
                )
        if not self.name.strip():
            raise ValueError("SecurityRule.name cannot be empty")
        if not self.description.strip():
            raise ValueError("SecurityRule.description cannot be empty")
        if not self.detection_condition.strip():
            raise ValueError("SecurityRule.detection_condition cannot be empty")
        if not self.remediation.strip():
            raise ValueError("SecurityRule.remediation cannot be empty")
        if not self.affected_format.strip():
            raise ValueError("SecurityRule.affected_format cannot be empty")


__all__ = [
    "SecurityRule",
]
