"""
tests.unit.test_rules
=====================

Unit test suite for Phase 8: Centralized Security Rule Catalog and Remediation Engine.

Test Coverage:
1. Exactly 86 rules registered in RuleCatalog.
2. Every rule ID conforms to canonical regex
   ^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1|MS1)\\d{2}$.
3. No duplicate rule IDs in catalog.
4. Every rule has a valid Severity enum.
5. Every rule has a valid category.
6. Every rule has non-empty description.
7. Every rule has non-empty detection_condition.
8. Every rule has non-empty remediation guidance.
9. Every rule has an affected_format specification.
10. Rule lookup by ID via get_rule().
11. has_rule() validation for existing and missing rules.
12. Category-based rule filtering.
13. Format-based rule filtering.
14. Severity-based rule filtering.
15. Unknown rule lookups raise KeyError.
16. Remediation guidance available for all 86 rules via RemediationRegistry.
17. CVE and CWE mappings preserved accurately.
18. SecurityRule immutability (frozen dataclass enforcement).
19. Deterministic ordering of list_all_rules().
20. Analyzer-emitted rule IDs cross-checked against RuleCatalog.
"""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from model_scanner.core.models import Severity
from model_scanner.core.normalizer import CANONICAL_CATEGORIES
from model_scanner.rules.catalog import RuleCatalog
from model_scanner.rules.models import SecurityRule
from model_scanner.rules.remediation import RemediationRegistry

_RULE_ID_PATTERN = re.compile(r"^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1|MS1)\d{2}$")


# ---------------------------------------------------------------------------
# Test 1 & 2 & 3: Catalog Count and Rule ID Integrity
# ---------------------------------------------------------------------------


def test_catalog_exact_rule_count() -> None:
    """Verify that exactly 87 rules are registered in the RuleCatalog.

    87 = 86 (prior baseline incl. KR105) + 1 (MS101: MindSpore allocation bomb).
    """
    all_rules = RuleCatalog.list_all_rules()
    assert len(all_rules) == 87, f"Expected 87 rules, got {len(all_rules)}"


def test_all_rule_ids_conform_to_canonical_pattern() -> None:
    """Verify that all 87 rule IDs match the canonical pattern."""
    for rule in RuleCatalog.list_all_rules():
        assert _RULE_ID_PATTERN.match(rule.rule_id), (
            f"Rule ID {rule.rule_id!r} does not match canonical pattern"
        )


def test_no_duplicate_rule_ids() -> None:
    """Verify that every rule in the catalog has a unique rule_id."""
    all_rules = RuleCatalog.list_all_rules()
    rule_ids = [r.rule_id for r in all_rules]
    assert len(rule_ids) == len(set(rule_ids)), "Duplicate rule IDs detected in RuleCatalog"


# ---------------------------------------------------------------------------
# Test 4 & 5: Severity and Category Validation
# ---------------------------------------------------------------------------


def test_every_rule_has_valid_severity() -> None:
    """Verify that every registered rule has a valid Severity enum value."""
    for rule in RuleCatalog.list_all_rules():
        assert isinstance(rule.severity, Severity)


def test_every_rule_has_valid_category() -> None:
    """Verify that every rule category belongs to the canonical category taxonomy."""
    valid_categories = CANONICAL_CATEGORIES | {"format_integrity"}
    for rule in RuleCatalog.list_all_rules():
        assert rule.category in valid_categories, (
            f"Rule {rule.rule_id} has invalid category {rule.category!r}"
        )


# ---------------------------------------------------------------------------
# Test 6, 7, 8, 9: Field Completeness
# ---------------------------------------------------------------------------


def test_every_rule_has_complete_non_empty_fields() -> None:
    """Verify that name, description, detection_condition, remediation, and format are populated."""
    for rule in RuleCatalog.list_all_rules():
        assert rule.name.strip(), f"Rule {rule.rule_id} has empty name"
        assert rule.description.strip(), f"Rule {rule.rule_id} has empty description"
        assert rule.detection_condition.strip(), (
            f"Rule {rule.rule_id} has empty detection_condition"
        )
        assert rule.remediation.strip(), f"Rule {rule.rule_id} has empty remediation"
        assert rule.affected_format.strip(), f"Rule {rule.rule_id} has empty affected_format"


# ---------------------------------------------------------------------------
# Test 10 & 11: Rule Lookup and has_rule
# ---------------------------------------------------------------------------


def test_rule_lookup_and_has_rule() -> None:
    """Verify get_rule and has_rule for existing and non-existing rule IDs."""
    assert RuleCatalog.has_rule("P101") is True
    assert RuleCatalog.has_rule("ST101") is True
    assert RuleCatalog.has_rule("GU101") is True
    assert RuleCatalog.has_rule("G001") is True
    assert RuleCatalog.has_rule("G002") is True
    assert RuleCatalog.has_rule("NON_EXISTENT") is False

    rule = RuleCatalog.get_rule("P101")
    assert rule.rule_id == "P101"
    assert rule.name == "Dangerous Global Import"
    assert rule.severity == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Test 12, 13, 14: Filtering Capabilities
# ---------------------------------------------------------------------------


def test_filtering_by_category() -> None:
    """Verify filtering rules by category."""
    code_exec_rules = RuleCatalog.get_rules_by_category("code_execution")
    assert len(code_exec_rules) >= 8  # P101-P107, ST101, GU101, etc.
    for r in code_exec_rules:
        assert r.category == "code_execution"

    anomaly_rules = RuleCatalog.get_rules_by_category("structural_anomaly")
    assert len(anomaly_rules) >= 15
    for r in anomaly_rules:
        assert r.category == "structural_anomaly"


def test_filtering_by_format() -> None:
    """Verify filtering rules by format."""
    pickle_rules = RuleCatalog.get_rules_by_format("RAW_PICKLE")
    assert len(pickle_rules) >= 15  # P1xx + P2xx + C004 + the ALL-format rules
    for r in pickle_rules:
        assert "RAW_PICKLE" in r.affected_format or r.affected_format == "ALL"

    st_rules = RuleCatalog.get_rules_by_format("SAFETENSORS")
    assert len(st_rules) >= 12  # ST1xx + ST2xx + the ALL-format rules
    for r in st_rules:
        assert "SAFETENSORS" in r.affected_format or r.affected_format == "ALL"


def test_format_filter_matches_whole_tokens_only() -> None:
    """affected_format is a token list, so a substring must not match a format.

    A substring test reported every RAW_PICKLE rule for the non-existent format
    "PICKLE" and every "GGUF, GGML" rule for "GG".
    """
    for bogus in ("PICKLE", "GG", "TENSORS"):
        for rule in RuleCatalog.get_rules_by_format(bogus):
            assert rule.affected_format == "ALL", (
                f"{rule.rule_id} ({rule.affected_format!r}) matched bogus format {bogus!r}"
            )

    ggml_ids = {r.rule_id for r in RuleCatalog.get_rules_by_format("GGML")}
    assert "GU101" in ggml_ids
    assert "ST101" not in ggml_ids


def test_filtering_by_severity() -> None:
    """Verify filtering rules by severity level."""
    critical_rules = RuleCatalog.get_rules_by_severity(Severity.CRITICAL)
    critical_ids = {r.rule_id for r in critical_rules}
    assert "P101" in critical_ids
    assert "P104" in critical_ids
    assert "ST101" in critical_ids
    assert "GU101" in critical_ids

    # P204 (FRAME desync) is documented HIGH, not CRITICAL: the pickle analyzer's
    # structural branch emits HIGH for it, and the catalog mirrors what is emitted.
    high_ids = {r.rule_id for r in RuleCatalog.get_rules_by_severity(Severity.HIGH)}
    assert "P204" in high_ids


# ---------------------------------------------------------------------------
# Test 15: KeyError on Missing Rule
# ---------------------------------------------------------------------------


def test_unknown_rule_raises_key_error() -> None:
    """Verify querying non-existent rule IDs raises KeyError."""
    with pytest.raises(KeyError, match=r"Rule ID 'INVALID_RULE' not found"):
        RuleCatalog.get_rule("INVALID_RULE")

    with pytest.raises(KeyError, match=r"Rule ID 'INVALID_RULE' not found"):
        RemediationRegistry.get_guidance("INVALID_RULE")


# ---------------------------------------------------------------------------
# Test 16: Remediation Guidance for All Rules
# ---------------------------------------------------------------------------


def test_remediation_guidance_for_all_rules() -> None:
    """Verify RemediationRegistry provides non-empty guidance for all 87 rules."""
    for rule in RuleCatalog.list_all_rules():
        guidance = RemediationRegistry.get_guidance(rule.rule_id)
        assert guidance.strip(), f"Empty remediation guidance for {rule.rule_id}"
        assert guidance == rule.remediation


# ---------------------------------------------------------------------------
# Test 17: CVE & CWE Mappings
# ---------------------------------------------------------------------------


def test_cve_and_cwe_mappings() -> None:
    """Verify CWE mappings are preserved in the catalog.

    No fabricated CVE IDs are asserted — catalog.py uses cve_ids=() for rules
    that have no verified NVD / GHSA entry.
    """
    p101 = RuleCatalog.get_rule("P101")
    assert "CWE-502" in p101.cwe_ids
    assert p101.cve_ids == ()  # no verified CVE assigned

    p204 = RuleCatalog.get_rule("P204")
    assert "CWE-125" in p204.cwe_ids
    assert p204.cve_ids == ()  # no verified CVE assigned

    g001 = RuleCatalog.get_rule("G001")
    assert "CWE-436" in g001.cwe_ids

    # Verify RemediationRegistry mapping lookups
    assert RemediationRegistry.get_cve_mappings("P101") == ()
    assert "CWE-502" in RemediationRegistry.get_cwe_mappings("P101")


# ---------------------------------------------------------------------------
# Test 18: SecurityRule Immutability
# ---------------------------------------------------------------------------


def test_security_rule_immutability() -> None:
    """Verify SecurityRule instances are frozen and cannot be mutated at runtime."""
    rule = RuleCatalog.get_rule("P101")
    with pytest.raises(FrozenInstanceError):
        rule.severity = Severity.INFO  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        rule.name = "Mutated Name"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 19: Deterministic Ordering
# ---------------------------------------------------------------------------


def test_deterministic_catalog_ordering() -> None:
    """Verify list_all_rules() returns rules in deterministic fixed order."""
    rules_first = RuleCatalog.list_all_rules()
    rules_second = RuleCatalog.list_all_rules()

    assert [r.rule_id for r in rules_first] == [r.rule_id for r in rules_second]
    # P101 must be the first rule and WI103 must be the last rule (P2 additions)
    assert rules_first[0].rule_id == "P101"
    assert rules_first[-1].rule_id == "WI103"  # WI1xx still last — MS1xx inserted before WI1xx


# ---------------------------------------------------------------------------
# Test 20: Cross-Check Analyzer-Emitted Rules
# ---------------------------------------------------------------------------


def test_analyzer_emitted_rules_exist_in_catalog() -> None:
    """Verify every rule emitted across the analyzers is present in the catalog."""
    expected_rule_ids = {
        # Pickle & PyTorch (18)
        "P101",
        "P102",
        "P103",
        "P104",
        "P105",
        "P106",
        "P107",
        "P201",
        "P202",
        "P203",
        "P204",
        "P205",
        "P206",
        "P207",
        "P208",
        "P209",
        "P210",
        "P211",
        # SafeTensors (14)
        "ST101",
        "ST102",
        "ST103",
        "ST104",
        "ST105",
        "ST201",
        "ST202",
        "ST203",
        "ST204",
        "ST205",
        "ST206",
        "ST207",
        "ST208",
        "ST209",
        # GGUF (16)
        "GU101",
        "GU102",
        "GU103",
        "GU104",
        "GU105",
        "GU106",
        "GU201",
        "GU202",
        "GU203",
        "GU204",
        "GU205",
        "GU206",
        "GU207",
        "GU208",
        "GU209",
        "GU210",
        # Cross-format shard integrity (4)
        "G005",
        "G006",
        "G007",
        "G008",
        # Supply chain — config & code execution (9)
        "SC101",
        "SC102",
        "SC103",
        "SC104",
        "SC105",
        "SC106",
        "SC107",
        "SC108",
        "SC109",
        # Supply chain — dependency security (4)
        "SC201",
        "SC202",
        "SC203",
        "SC204",
        # Cross-format (2)
        "G001",
        "G002",
        # Correlated cross-signal rules (4)
        "C001",
        "C002",
        "C003",
        "C004",
        # P2: Weight integrity (3)
        "WI101",
        "WI102",
        "WI103",
        # P2: ONNX (4)
        "ON101",
        "ON102",
        "ON103",
        "ON104",
        # P2: TensorFlow SavedModel (3)
        "TF101",
        "TF102",
        "TF103",
        # P2: Keras (5)
        "KR101",
        "KR102",
        "KR103",
        "KR104",
        "KR105",
        # MindSpore (1)
        "MS101",
    }

    catalog_rule_ids = {r.rule_id for r in RuleCatalog.list_all_rules()}
    assert catalog_rule_ids == expected_rule_ids, (
        f"Mismatch between expected and catalog rule IDs: "
        f"missing={expected_rule_ids - catalog_rule_ids}, "
        f"extra={catalog_rule_ids - expected_rule_ids}"
    )


# ---------------------------------------------------------------------------
# Test 21: Migration Advice
# ---------------------------------------------------------------------------


def test_migration_advice() -> None:
    """Verify format migration advice generation across formats."""
    advice_pickle = RemediationRegistry.get_migration_advice("RAW_PICKLE")
    assert "MIGRATION TO SAFETENSORS" in advice_pickle

    advice_gguf = RemediationRegistry.get_migration_advice("GGUF")
    assert "GGUF" in advice_gguf

    advice_unknown = RemediationRegistry.get_migration_advice("UNKNOWN_FORMAT")
    assert "GENERAL MODEL REMEDIATION" in advice_unknown


# ---------------------------------------------------------------------------
# Test 22: SecurityRule Validation Checks
# ---------------------------------------------------------------------------


def test_invalid_security_rule_construction() -> None:
    """Verify SecurityRule rejects invalid parameters during construction."""
    with pytest.raises(ValueError, match=r"does not match required pattern"):
        SecurityRule(
            rule_id="INVALID_ID",
            name="Test",
            severity=Severity.LOW,
            category="code_execution",
            description="desc",
            detection_condition="cond",
            remediation="rem",
            affected_format="RAW_PICKLE",
        )

    with pytest.raises(ValueError, match=r"name cannot be empty"):
        SecurityRule(
            rule_id="P101",
            name="",
            severity=Severity.LOW,
            category="code_execution",
            description="desc",
            detection_condition="cond",
            remediation="rem",
            affected_format="RAW_PICKLE",
        )

    with pytest.raises(ValueError, match=r"description cannot be empty"):
        SecurityRule(
            rule_id="P101",
            name="Valid Name",
            severity=Severity.LOW,
            category="code_execution",
            description="",
            detection_condition="cond",
            remediation="rem",
            affected_format="RAW_PICKLE",
        )

    with pytest.raises(ValueError, match=r"detection_condition cannot be empty"):
        SecurityRule(
            rule_id="P101",
            name="Valid Name",
            severity=Severity.LOW,
            category="code_execution",
            description="desc",
            detection_condition="",
            remediation="rem",
            affected_format="RAW_PICKLE",
        )

    with pytest.raises(ValueError, match=r"remediation cannot be empty"):
        SecurityRule(
            rule_id="P101",
            name="Valid Name",
            severity=Severity.LOW,
            category="code_execution",
            description="desc",
            detection_condition="cond",
            remediation="",
            affected_format="RAW_PICKLE",
        )

    with pytest.raises(ValueError, match=r"affected_format cannot be empty"):
        SecurityRule(
            rule_id="P101",
            name="Valid Name",
            severity=Severity.LOW,
            category="code_execution",
            description="desc",
            detection_condition="cond",
            remediation="rem",
            affected_format="",
        )
