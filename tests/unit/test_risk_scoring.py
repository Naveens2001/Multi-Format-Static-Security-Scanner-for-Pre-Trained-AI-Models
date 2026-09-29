"""
tests.unit.test_risk_scoring
============================

Unit tests for the RiskEngine — Phase 7.2 deterministic risk scoring.

Test Coverage:
1. Zero findings produces CLEAN RiskScore with 0 final score
2. Single CRITICAL finding with HIGH confidence produces correct score
3. Severity weights: CRITICAL (40), HIGH (20), MEDIUM (5), LOW (1), INFO (0)
4. Confidence factors: HIGH (1.0), MEDIUM (0.8), LOW (0.5)
5. Format multipliers: RAW_PICKLE (1.5), PYTORCH_ZIP (1.3), GGUF (1.1), GGML (1.1),
   SAFETENSORS (1.0), UNSUPPORTED (1.0)
6. Obfuscation multipliers: Base (1.0), Unresolvable (1.25), Structural (1.5), Both (1.75)
7. Score clamping at 100 for severe multiple findings
8. Score rounding (half-up)
9. Risk band boundary transitions:
   - 0: CLEAN
   - 1-30: LOW_RISK
   - 31-60: MEDIUM_RISK
   - 61-89: HIGH_RISK
   - 90-100: CRITICAL_RISK
10. Finding deduplication
11. Mathematical determinism (same input produces identical output)
12. RiskScore validation invariant compliance
"""

from __future__ import annotations

import pytest

from model_scanner.core.models import (
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    ModelFormat,
    RiskBand,
    Severity,
    risk_band_for_score,
)
from model_scanner.core.scoring import (
    OBFUSCATION_BASE,
    OBFUSCATION_BOTH,
    OBFUSCATION_STRUCTURAL,
    OBFUSCATION_UNRESOLVABLE,
    RiskEngine,
)


def _make_finding(
    rule_id: str = "P101",
    severity: Severity = Severity.CRITICAL,
    confidence: Confidence = Confidence.HIGH,
    category: str = "code_execution",
    byte_offset: int = 100,
    matched_value: str = "posix.system",
    archive_member: str | None = None,
) -> Finding:
    """Helper to create valid mock Finding instances for tests."""
    coord_space = (
        CoordinateSpace.ARCHIVE_MEMBER
        if archive_member
        else CoordinateSpace.ARTIFACT
    )
    return Finding(
        finding_id=f"FINDING-{rule_id}-001",
        rule_id=rule_id,
        rule_name="Test Rule",
        severity=severity,
        confidence=confidence,
        category=category,
        format=ModelFormat.RAW_PICKLE,
        artifact_sha256="a" * 64,
        evidence=Evidence(
            coordinate_space=coord_space,
            byte_offset=byte_offset,
            byte_length=len(matched_value),
            archive_member=archive_member,
            matched_value=matched_value,
        ),
        description="Test finding description",
        remediation="Do not load this model.",
    )


# ---------------------------------------------------------------------------
# Test 1: Zero Findings / Clean Artifact
# ---------------------------------------------------------------------------


def test_zero_findings_produces_clean_score() -> None:
    """Verify that an empty finding list produces a clean risk score."""
    score = RiskEngine.calculate([], ModelFormat.SAFETENSORS)

    assert score.final_score == 0
    assert score.raw_score == 0.0
    assert score.risk_band == RiskBand.CLEAN
    assert score.findings_weighted_sum == 0.0
    assert score.format_multiplier == 1.0
    assert score.obfuscation_multiplier == 1.0


def test_zero_findings_on_raw_pickle() -> None:
    """Verify clean score records format multiplier even on pickle."""
    score = RiskEngine.calculate([], ModelFormat.RAW_PICKLE)

    assert score.final_score == 0
    assert score.raw_score == 0.0
    assert score.risk_band == RiskBand.CLEAN
    assert score.findings_weighted_sum == 0.0
    assert score.format_multiplier == 1.5


# ---------------------------------------------------------------------------
# Test 2: Severity Weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("severity", "expected_weight"),
    [
        (Severity.CRITICAL, 40.0),
        (Severity.HIGH, 20.0),
        (Severity.MEDIUM, 5.0),
        (Severity.LOW, 1.0),
        (Severity.INFO, 0.0),
    ],
)
def test_severity_weights(severity: Severity, expected_weight: float) -> None:
    """Verify individual severity weights with SafeTensors format (1.0 multiplier)."""
    finding = _make_finding(severity=severity, confidence=Confidence.HIGH)
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)

    assert score.findings_weighted_sum == expected_weight
    assert score.raw_score == expected_weight
    assert score.final_score == int(expected_weight)


# ---------------------------------------------------------------------------
# Test 3: Confidence Factors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "expected_factor"),
    [
        (Confidence.HIGH, 1.0),
        (Confidence.MEDIUM, 0.8),
        (Confidence.LOW, 0.5),
    ],
)
def test_confidence_factors(confidence: Confidence, expected_factor: float) -> None:
    """Verify confidence factors scale the severity weight appropriately."""
    finding = _make_finding(severity=Severity.CRITICAL, confidence=confidence)
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)

    expected_weighted_sum = 40.0 * expected_factor
    assert score.findings_weighted_sum == pytest.approx(expected_weighted_sum)
    assert score.raw_score == pytest.approx(expected_weighted_sum)
    assert score.final_score == round(expected_weighted_sum)


# ---------------------------------------------------------------------------
# Test 4: Format Multipliers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_format", "expected_mult"),
    [
        (ModelFormat.RAW_PICKLE, 1.5),
        (ModelFormat.PYTORCH_ZIP, 1.3),
        (ModelFormat.GGUF, 1.1),
        (ModelFormat.GGML, 1.1),
        (ModelFormat.SAFETENSORS, 1.0),
        (ModelFormat.UNSUPPORTED, 1.0),
    ],
)
def test_format_multipliers(model_format: ModelFormat, expected_mult: float) -> None:
    """Verify format multipliers scale the score correctly."""
    finding = _make_finding(severity=Severity.HIGH, confidence=Confidence.HIGH)
    score = RiskEngine.calculate([finding], model_format)

    expected_raw = 20.0 * expected_mult * 1.0
    assert score.format_multiplier == expected_mult
    assert score.raw_score == pytest.approx(expected_raw)
    assert score.final_score == round(expected_raw)


# ---------------------------------------------------------------------------
# Test 5: Obfuscation Multipliers
# ---------------------------------------------------------------------------


def test_obfuscation_clean_baseline() -> None:
    """Verify clean standard finding uses base obfuscation multiplier (1.0)."""
    finding = _make_finding(rule_id="P101", category="code_execution")
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)
    assert score.obfuscation_multiplier == OBFUSCATION_BASE


def test_obfuscation_unresolvable_symbols() -> None:
    """Verify unresolvable symbol rule P103 triggers 1.25 obfuscation multiplier."""
    finding = _make_finding(
        rule_id="P103",
        category="obfuscation",
        severity=Severity.MEDIUM,
    )
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)
    assert score.obfuscation_multiplier == OBFUSCATION_UNRESOLVABLE

    # 5.0 * 1.0 * 1.25 = 6.25 -> final 6
    assert score.raw_score == pytest.approx(6.25)
    assert score.final_score == 6


def test_obfuscation_structural_anomaly() -> None:
    """Verify structural anomaly rule (e.g. ST201 or P201) triggers 1.5 multiplier."""
    finding = _make_finding(
        rule_id="ST201",
        category="structural_anomaly",
        severity=Severity.HIGH,
    )
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)
    assert score.obfuscation_multiplier == OBFUSCATION_STRUCTURAL

    # 20.0 * 1.0 * 1.5 = 30.0 -> final 30
    assert score.raw_score == pytest.approx(30.0)
    assert score.final_score == 30


def test_obfuscation_both_structural_and_unresolvable() -> None:
    """Verify presence of both structural and unresolvable triggers 1.75 multiplier."""
    finding1 = _make_finding(
        rule_id="P201",
        category="structural_anomaly",
        severity=Severity.HIGH,
        byte_offset=10,
    )
    finding2 = _make_finding(
        rule_id="P103",
        category="obfuscation",
        severity=Severity.MEDIUM,
        byte_offset=50,
    )

    score = RiskEngine.calculate([finding1, finding2], ModelFormat.SAFETENSORS)
    assert score.obfuscation_multiplier == OBFUSCATION_BOTH

    # Weighted sum: 20.0 + 5.0 = 25.0
    # Raw score: 25.0 * 1.0 * 1.75 = 43.75 -> final 44
    assert score.findings_weighted_sum == pytest.approx(25.0)
    assert score.raw_score == pytest.approx(43.75)
    assert score.final_score == 44


# ---------------------------------------------------------------------------
# Test 6: Composite Realistic Scenario (Phase 7 Design Spec Example)
# ---------------------------------------------------------------------------


def test_phase7_design_example_scenario() -> None:
    """Verify example scenario from phase7_design.md."""
    f1 = _make_finding(
        rule_id="P101",
        severity=Severity.CRITICAL,
        confidence=Confidence.HIGH,
        byte_offset=42,
    )
    f2 = _make_finding(
        rule_id="P102",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        byte_offset=100,
    )

    score = RiskEngine.calculate([f1, f2], ModelFormat.PYTORCH_ZIP)
    # Weighted sum: 40.0 + 20.0 = 60.0
    # Raw score: 60.0 * 1.3 * 1.0 = 78.0
    # Final score: 78 -> numeric band HIGH_RISK, but CRITICAL severity floor
    # (R5) elevates to CRITICAL_RISK.
    assert score.findings_weighted_sum == pytest.approx(60.0)
    assert score.format_multiplier == 1.3
    assert score.obfuscation_multiplier == 1.0
    assert score.raw_score == pytest.approx(78.0)
    assert score.final_score == 78
    assert score.risk_band == RiskBand.CRITICAL_RISK


# ---------------------------------------------------------------------------
# Test 7: Score Clamping at 100
# ---------------------------------------------------------------------------


def test_score_clamped_at_100() -> None:
    """Verify that multiple critical findings do not exceed 100."""
    findings = [
        _make_finding(
            rule_id="P101",
            severity=Severity.CRITICAL,
            byte_offset=i * 50,
        )
        for i in range(1, 6)
    ]
    # 5 * 40.0 = 200.0 * 1.5 (pickle) * 1.0 (baseline) = 300.0 -> clamped to 100
    score = RiskEngine.calculate(findings, ModelFormat.RAW_PICKLE)

    assert score.findings_weighted_sum == pytest.approx(200.0)
    assert score.obfuscation_multiplier == 1.0
    assert score.raw_score == pytest.approx(300.0)
    assert score.final_score == 100
    assert score.risk_band == RiskBand.CRITICAL_RISK


def test_score_clamped_with_obfuscation() -> None:
    """Verify multiple critical findings with obfuscation rules (P103) clamp to 100."""
    findings = [
        _make_finding(
            rule_id=f"P10{i}",
            severity=Severity.CRITICAL,
            byte_offset=i * 50,
        )
        for i in range(1, 6)
    ]
    # 5 * 40.0 = 200.0 * 1.5 (pickle) * 1.25 (unresolvable P103/P104/P105) = 375.0
    score = RiskEngine.calculate(findings, ModelFormat.RAW_PICKLE)

    assert score.findings_weighted_sum == pytest.approx(200.0)
    assert score.obfuscation_multiplier == 1.25
    assert score.raw_score == pytest.approx(375.0)
    assert score.final_score == 100
    assert score.risk_band == RiskBand.CRITICAL_RISK


# ---------------------------------------------------------------------------
# Test 8: Risk Band Boundary Mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score_val", "expected_band"),
    [
        (0, RiskBand.CLEAN),
        (1, RiskBand.LOW_RISK),
        (15, RiskBand.LOW_RISK),
        (30, RiskBand.LOW_RISK),
        (31, RiskBand.MEDIUM_RISK),
        (45, RiskBand.MEDIUM_RISK),
        (60, RiskBand.MEDIUM_RISK),
        (61, RiskBand.HIGH_RISK),
        (75, RiskBand.HIGH_RISK),
        (89, RiskBand.HIGH_RISK),
        (90, RiskBand.CRITICAL_RISK),
        (95, RiskBand.CRITICAL_RISK),
        (100, RiskBand.CRITICAL_RISK),
    ],
)
def test_risk_band_boundaries(score_val: int, expected_band: RiskBand) -> None:
    """Verify that score-to-band derivation matches Section 21.3 exact boundaries."""
    assert risk_band_for_score(score_val) == expected_band


# ---------------------------------------------------------------------------
# Test 9: Finding Deduplication
# ---------------------------------------------------------------------------


def test_duplicate_findings_deduplicated() -> None:
    """Verify identical duplicate findings are only counted once in weighted sum."""
    finding1 = _make_finding(rule_id="P101", severity=Severity.CRITICAL, byte_offset=100)
    finding2 = _make_finding(rule_id="P101", severity=Severity.CRITICAL, byte_offset=100)

    score = RiskEngine.calculate([finding1, finding2], ModelFormat.SAFETENSORS)

    assert score.findings_weighted_sum == pytest.approx(40.0)
    assert score.final_score == 40


def test_distinct_offset_findings_not_deduplicated() -> None:
    """Verify findings with different byte offsets are treated as distinct."""
    finding1 = _make_finding(rule_id="P101", severity=Severity.CRITICAL, byte_offset=100)
    finding2 = _make_finding(rule_id="P101", severity=Severity.CRITICAL, byte_offset=200)

    score = RiskEngine.calculate([finding1, finding2], ModelFormat.SAFETENSORS)

    assert score.findings_weighted_sum == pytest.approx(80.0)
    assert score.final_score == 80


# ---------------------------------------------------------------------------
# Test 10: Determinism
# ---------------------------------------------------------------------------


def test_scoring_determinism() -> None:
    """Verify multiple invocations with same inputs produce identical results."""
    findings = [
        _make_finding(
            rule_id="ST101",
            severity=Severity.MEDIUM,
            confidence=Confidence.MEDIUM,
            byte_offset=50,
        ),
        _make_finding(
            rule_id="ST201",
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            byte_offset=120,
        ),
    ]

    score1 = RiskEngine.calculate(findings, ModelFormat.SAFETENSORS)
    score2 = RiskEngine.calculate(findings, ModelFormat.SAFETENSORS)

    assert score1.raw_score == score2.raw_score
    assert score1.final_score == score2.final_score
    assert score1.risk_band == score2.risk_band
    assert score1.findings_weighted_sum == score2.findings_weighted_sum
    assert score1.format_multiplier == score2.format_multiplier
    assert score1.obfuscation_multiplier == score2.obfuscation_multiplier


# ---------------------------------------------------------------------------
# Test 11: Severity Floor (R5)
# ---------------------------------------------------------------------------
# A single HIGH or CRITICAL finding must never resolve to a band BELOW its
# severity, even when the arithmetic score is small (e.g. one HIGH finding
# on a low-risk format scores only 20 → LOW_RISK without the floor).


def test_severity_floor_single_high_finding() -> None:
    """Single HIGH finding must produce at least HIGH_RISK band (R5 floor)."""
    # ON101 doesn't start with P2/ST2/GU2 and uses default category "code_execution",
    # so no obfuscation multiplier is triggered — pure arithmetic score is 20.
    finding = _make_finding(rule_id="ON101", severity=Severity.HIGH, byte_offset=0)
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)

    # Numeric score: 20.0 * 1.0 (format) * 1.0 (no obfuscation) = 20 → LOW_RISK by arithmetic.
    # Severity floor raises band to HIGH_RISK.
    assert score.final_score == 20
    assert score.risk_band == RiskBand.HIGH_RISK


def test_severity_floor_single_critical_finding() -> None:
    """Single CRITICAL finding must produce CRITICAL_RISK band (R5 floor)."""
    finding = _make_finding(rule_id="P101", severity=Severity.CRITICAL, byte_offset=0)
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)

    # Numeric score: 40.0 * 1.0 * 1.0 = 40 → MEDIUM_RISK by arithmetic.
    # CRITICAL severity floor raises band to CRITICAL_RISK.
    assert score.final_score == 40
    assert score.risk_band == RiskBand.CRITICAL_RISK


def test_severity_floor_high_on_safetensors_with_obfuscation() -> None:
    """HIGH finding with structural obfuscation floors at HIGH_RISK, score 30 → LOW by arithmetic."""
    finding = _make_finding(
        rule_id="ST201",
        category="structural_anomaly",
        severity=Severity.HIGH,
        byte_offset=0,
    )
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)

    # 20 * 1.0 (ST) * 1.5 (structural) = 30.0 → LOW_RISK numerically.
    # HIGH severity floor raises to HIGH_RISK.
    assert score.final_score == 30
    assert score.risk_band == RiskBand.HIGH_RISK


def test_severity_floor_medium_finding() -> None:
    """MEDIUM severity finding floors at MEDIUM_RISK band."""
    finding = _make_finding(rule_id="ON103", severity=Severity.MEDIUM, byte_offset=0)
    score = RiskEngine.calculate([finding], ModelFormat.SAFETENSORS)

    # 5.0 * 1.0 * 1.0 = 5 → LOW_RISK numerically; MEDIUM floor raises to MEDIUM_RISK.
    assert score.final_score == 5
    assert score.risk_band == RiskBand.MEDIUM_RISK


def test_severity_floor_does_not_lower_high_score() -> None:
    """Floor must not reduce a band that is already correct — no ceiling effect."""
    findings = [
        _make_finding(rule_id="P101", severity=Severity.CRITICAL, byte_offset=i * 20)
        for i in range(5)
    ]
    score = RiskEngine.calculate(findings, ModelFormat.RAW_PICKLE)

    # 5 * 40.0 * 1.5 = 300.0 → clamped to 100 → CRITICAL_RISK; floor agrees.
    assert score.final_score == 100
    assert score.risk_band == RiskBand.CRITICAL_RISK


def test_severity_floor_info_only_stays_clean() -> None:
    """INFO-only findings produce CLEAN band; INFO floor is CLEAN (no elevation)."""
    finding = _make_finding(rule_id="P209", severity=Severity.INFO, byte_offset=0)
    score = RiskEngine.calculate([finding], ModelFormat.PYTORCH_ZIP)

    assert score.final_score == 0
    assert score.risk_band == RiskBand.CLEAN


# ---------------------------------------------------------------------------
# P2 regression: LEGACY_PYTORCH format multiplier
# ---------------------------------------------------------------------------


def test_legacy_pytorch_multiplier_equals_raw_pickle() -> None:
    """LEGACY_PYTORCH uses PickleAnalyzer and must carry the same 1.5 multiplier as RAW_PICKLE.

    Before the P2 fix, LEGACY_PYTORCH fell through to the default (.get fallback = 1.0),
    underscoring identical threat surface by 33%.
    """
    from model_scanner.core.scoring import FORMAT_MULTIPLIERS

    assert ModelFormat.LEGACY_PYTORCH in FORMAT_MULTIPLIERS, (
        "LEGACY_PYTORCH must be an explicit entry in FORMAT_MULTIPLIERS"
    )
    assert FORMAT_MULTIPLIERS[ModelFormat.LEGACY_PYTORCH] == 1.5, (
        "LEGACY_PYTORCH multiplier must equal RAW_PICKLE (1.5) — same PickleAnalyzer threat surface"
    )
    assert FORMAT_MULTIPLIERS[ModelFormat.LEGACY_PYTORCH] == FORMAT_MULTIPLIERS[ModelFormat.RAW_PICKLE]


def test_legacy_pytorch_score_matches_raw_pickle_score() -> None:
    """A finding on LEGACY_PYTORCH produces the same numeric score as on RAW_PICKLE."""
    finding = _make_finding(rule_id="P101", severity=Severity.HIGH, byte_offset=0)

    score_legacy = RiskEngine.calculate([finding], ModelFormat.LEGACY_PYTORCH)
    score_raw = RiskEngine.calculate([finding], ModelFormat.RAW_PICKLE)

    assert score_legacy.format_multiplier == 1.5
    assert score_legacy.format_multiplier == score_raw.format_multiplier
    assert score_legacy.final_score == score_raw.final_score
