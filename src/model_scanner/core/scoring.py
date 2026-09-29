"""
model_scanner.core.scoring
==========================

Deterministic composite risk scoring engine for static model security findings.

Mathematical Formulation
------------------------
1. WeightedSum = SUM_{f in UniqueFindings}(
     SeverityWeight(f.severity) * ConfidenceFactor(f.confidence)
   )
2. RawScore = WeightedSum * FormatMultiplier(format) * ObfuscationMultiplier(findings)
3. FinalScore = min(100, max(0, floor(RawScore + 0.5)))
4. RiskBand = _risk_band_for_score(FinalScore)

Calibration Parameters (PROPOSED / INITIAL - Subject to Empirical Calibration)
-----------------------------------------------------------------------------
Severity Weights:
  - CRITICAL: 40.0
  - HIGH:     20.0
  - MEDIUM:    5.0
  - LOW:       1.0
  - INFO:      0.0

Confidence Factors:
  - HIGH:   1.0
  - MEDIUM: 0.8
  - LOW:    0.5

Format Multipliers:
  - RAW_PICKLE:      1.5
  - LEGACY_PYTORCH:  1.5
  - PYTORCH_ZIP:     1.3
  - GGUF:        1.1
  - GGML:        1.1
  - SAFETENSORS: 1.0
  - UNSUPPORTED: 1.0

Obfuscation Multipliers:
  - Baseline (Clean / Standard):              1.0
  - Unresolvable / obfuscated symbols:        1.25
  - Structural anomaly detected:              1.5
  - Both structural anomaly & unresolvable:   1.75

Risk Score Bands:
  - 0:       CLEAN
  - 1-30:    LOW_RISK
  - 31-60:   MEDIUM_RISK
  - 61-89:   HIGH_RISK
  - 90-100:  CRITICAL_RISK

SECURITY GUARANTEE
------------------
This module is a PURE MATHEMATICAL and DETERMINISTIC calculation component.
It NEVER:
  - Loads models or deserializes binary formats
  - Executes model weights or arbitrary code
  - Imports ML libraries (torch, transformers, safetensors, gguf)
  - Uses eval(), exec(), or dynamic code invocation
  - Makes network or filesystem requests

Design Reference
----------------
- Section 21 of docs/implementation_design.md (authoritative)
- Section 7 of docs/phase7_design.md
- Section 4.7 of docs/project_scope.md
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from model_scanner.core.models import (
    Confidence,
    Finding,
    ModelFormat,
    RiskBand,
    RiskScore,
    Severity,
    _RISK_BAND_ORDER,
    risk_band_for_score,
)

# ---------------------------------------------------------------------------
# Calibration Constants (PROPOSED / INITIAL - Section 21.2)
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: Final[dict[Severity, float]] = {
    Severity.CRITICAL: 40.0,
    Severity.HIGH: 20.0,
    Severity.MEDIUM: 5.0,
    Severity.LOW: 1.0,
    Severity.INFO: 0.0,
}

CONFIDENCE_FACTORS: Final[dict[Confidence, float]] = {
    Confidence.HIGH: 1.0,
    Confidence.MEDIUM: 0.8,
    Confidence.LOW: 0.5,
}

FORMAT_MULTIPLIERS: Final[dict[ModelFormat, float]] = {
    ModelFormat.RAW_PICKLE: 1.5,
    ModelFormat.LEGACY_PYTORCH: 1.5,
    ModelFormat.PYTORCH_ZIP: 1.3,
    ModelFormat.GGUF: 1.1,
    ModelFormat.GGML: 1.1,
    ModelFormat.SAFETENSORS: 1.0,
    ModelFormat.ONNX: 1.0,
    ModelFormat.HDF5: 1.0,
    ModelFormat.TF_SAVEDMODEL: 1.0,
    ModelFormat.KERAS: 1.0,
    ModelFormat.UNSUPPORTED: 1.0,
    ModelFormat.MINDSPORE_CKPT: 1.3,  # Same risk tier as PYTORCH_ZIP
}

OBFUSCATION_BASE: Final[float] = 1.0
OBFUSCATION_UNRESOLVABLE: Final[float] = 1.25
OBFUSCATION_STRUCTURAL: Final[float] = 1.5
OBFUSCATION_BOTH: Final[float] = 1.75

# Severity ordering for floor computation — higher integer = more severe.
_SEVERITY_ORDER: Final[dict[Severity, int]] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

# Minimum RiskBand guaranteed when ANY finding of this severity is present (R5).
# INFO findings carry no minimum band (they are informational, not threats).
_SEVERITY_BAND_FLOOR: Final[dict[Severity, RiskBand]] = {
    Severity.INFO: RiskBand.CLEAN,
    Severity.LOW: RiskBand.LOW_RISK,
    Severity.MEDIUM: RiskBand.MEDIUM_RISK,
    Severity.HIGH: RiskBand.HIGH_RISK,
    Severity.CRITICAL: RiskBand.CRITICAL_RISK,
}


# ---------------------------------------------------------------------------
# Risk Engine Implementation — F-09
# ---------------------------------------------------------------------------


class RiskEngine:
    """Deterministic composite risk scoring engine.

    Calculates an immutable, strongly-typed RiskScore from a sequence of
    findings and the detected model format.
    """

    @staticmethod
    def deduplicate_findings(findings: Sequence[Finding]) -> tuple[Finding, ...]:
        """Filter duplicate findings deterministically preserving original order.

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

    @staticmethod
    def determine_obfuscation_multiplier(findings: Sequence[Finding]) -> float:
        """Determine the obfuscation multiplier based on finding characteristics.

        Parameters
        ----------
        findings:
            Sequence of findings to evaluate.

        Returns
        -------
        float
            Obfuscation multiplier: 1.0, 1.25, 1.5, or 1.75.
        """
        has_structural = False
        has_unresolvable = False

        for f in findings:
            # Informational findings describe expected structure, not evasion.
            # P209 ("Embedded Pickle Stream in PyTorch ZIP") fires on the
            # archive/data.pkl member present in EVERY PyTorch checkpoint; it
            # shares the P2xx prefix but is not an anomaly, and letting it set
            # the structural flag inflated every PyTorch scan by 1.5x.
            if f.severity is Severity.INFO:
                continue
            # Structural anomalies: rules P2xx, ST2xx, GU2xx or category
            # structural_anomaly / tampering
            if (
                f.rule_id.startswith(("P2", "ST2", "GU2"))
                or f.category in ("structural_anomaly", "tampering")
            ):
                has_structural = True

            # Unresolvable / obfuscated symbols: P103/P104/P105 or category
            # obfuscation / unresolvable
            if (
                f.rule_id in ("P103", "P104", "P105")
                or f.category in ("obfuscation", "unresolvable")
                or "unresolvable" in f.rule_name.lower()
                or "obfuscation" in f.rule_name.lower()
                or "blocked" in f.rule_name.lower()
            ):
                has_unresolvable = True

        if has_structural and has_unresolvable:
            return OBFUSCATION_BOTH
        if has_structural:
            return OBFUSCATION_STRUCTURAL
        if has_unresolvable:
            return OBFUSCATION_UNRESOLVABLE
        return OBFUSCATION_BASE

    @classmethod
    def calculate(
        cls,
        findings: Sequence[Finding],
        format: ModelFormat = ModelFormat.SAFETENSORS,
    ) -> RiskScore:
        """Calculate the deterministic RiskScore for the given findings and format.

        Parameters
        ----------
        findings:
            Sequence of security findings.
        format:
            Detected format of the artifact.

        Returns
        -------
        RiskScore
            Immutable, validated RiskScore dataclass instance.
        """
        # Correlation findings (C-series) are derived from primary findings that
        # are already scored. Including them would double-count the underlying signal.
        primary_findings = tuple(f for f in findings if not f.rule_id.startswith("C"))
        unique_findings = cls.deduplicate_findings(primary_findings)

        if not unique_findings:
            format_mult = FORMAT_MULTIPLIERS.get(format, 1.0)
            return RiskScore(
                raw_score=0.0,
                final_score=0,
                risk_band=risk_band_for_score(0),
                format_multiplier=format_mult,
                obfuscation_multiplier=OBFUSCATION_BASE,
                findings_weighted_sum=0.0,
            )

        # Step 1: Weighted sum of unique findings
        weighted_sum = sum(
            SEVERITY_WEIGHTS[f.severity] * CONFIDENCE_FACTORS[f.confidence]
            for f in unique_findings
        )

        # Step 2: Format multiplier
        format_multiplier = FORMAT_MULTIPLIERS.get(format, 1.0)

        # Step 3: Obfuscation multiplier
        obfuscation_multiplier = cls.determine_obfuscation_multiplier(unique_findings)

        # Step 4: Raw score
        raw_score = weighted_sum * format_multiplier * obfuscation_multiplier

        # Step 5: Final score (clamped to [0, 100], half-up rounded)
        final_score = min(100, max(0, math.floor(raw_score + 0.5)))

        # Step 6: Risk band derivation from numeric score
        risk_band = risk_band_for_score(final_score)

        # Step 7: Severity floor (R5) — the worst finding's severity sets the
        # minimum band regardless of the arithmetic score.  A lone HIGH/CRITICAL
        # finding must never be reported as LOW/MEDIUM risk just because the
        # weighted sum is small (e.g. single HIGH on safe format → score 20).
        worst_severity = max(unique_findings, key=lambda f: _SEVERITY_ORDER[f.severity]).severity
        floor_band = _SEVERITY_BAND_FLOOR[worst_severity]
        if _RISK_BAND_ORDER[floor_band] > _RISK_BAND_ORDER[risk_band]:
            risk_band = floor_band

        return RiskScore(
            raw_score=round(raw_score, 4),
            final_score=final_score,
            risk_band=risk_band,
            format_multiplier=format_multiplier,
            obfuscation_multiplier=obfuscation_multiplier,
            findings_weighted_sum=round(weighted_sum, 4),
        )


__all__ = [
    "CONFIDENCE_FACTORS",
    "FORMAT_MULTIPLIERS",
    "OBFUSCATION_BASE",
    "OBFUSCATION_BOTH",
    "OBFUSCATION_STRUCTURAL",
    "OBFUSCATION_UNRESOLVABLE",
    "SEVERITY_WEIGHTS",
    "RiskEngine",
]
