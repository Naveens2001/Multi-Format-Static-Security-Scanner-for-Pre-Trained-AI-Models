"""
tests.unit.test_catalog_analyzer_sync
=====================================

Guard against the catalog and the analyzers drifting apart.

``rules/catalog.py`` is not consulted at scan time — every analyzer carries its
own severity/name/category table and emits findings from that. The catalog is
what the Web UI's Rule Catalog view, ``docs/rule_catalog.md`` and the
Impact / CWE / OWASP / MITRE fields of a generated report are built from.

So when the two disagree, a report shows one severity next to a finding and a
different one in the rule reference beside it. That has happened before
(P106/P107 documented HIGH while the analyzer emitted MEDIUM; ST203 and ST208
documented with each other's detection conditions). These tests make that class
of defect fail the build instead of reaching a report.

Two sources of truth are checked:

1. The module-level ``_RULE_SEVERITY`` / ``_RULE_NAMES`` / ``_RULE_CATEGORY`` /
   ``_RULE_METADATA`` tables of each analyzer.
2. Every ``Finding(...)`` constructed anywhere under ``src/`` with a literal
   ``rule_id``; non-literal fields (severity chosen by a conditional) are
   skipped, since there is no single default to compare against.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from model_scanner.analyzers.gguf import analyzer as gguf_analyzer
from model_scanner.analyzers.pickle import analyzer as pickle_analyzer
from model_scanner.analyzers.repository import analyzer as repo_analyzer
from model_scanner.analyzers.safetensors import analyzer as st_analyzer
from model_scanner.core.models import Severity
from model_scanner.rules.catalog import RuleCatalog

_SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "model_scanner"


# ---------------------------------------------------------------------------
# 1. Analyzer lookup tables vs catalog
# ---------------------------------------------------------------------------

_SEVERITY_TABLES: list[tuple[str, dict[str, Severity]]] = [
    ("safetensors", st_analyzer._RULE_SEVERITY),
    ("gguf", gguf_analyzer._RULE_SEVERITY),
    ("repository", repo_analyzer._RULE_SEVERITY),
]

_NAME_TABLES: list[tuple[str, dict[str, str]]] = [
    ("safetensors", st_analyzer._RULE_NAMES),
    ("gguf", gguf_analyzer._RULE_NAMES),
    ("repository", repo_analyzer._RULE_NAMES),
]

_CATEGORY_TABLES: list[tuple[str, dict[str, str]]] = [
    ("safetensors", st_analyzer._RULE_CATEGORY),
    ("gguf", gguf_analyzer._RULE_CATEGORY),
    ("repository", repo_analyzer._RULE_CATEGORIES),
]


def test_analyzer_severity_tables_match_catalog() -> None:
    """Each analyzer's default severity must equal the catalog's severity."""
    mismatches: list[str] = []
    for source, table in _SEVERITY_TABLES:
        for rule_id, severity in table.items():
            rule = RuleCatalog.get_rule(rule_id)
            if rule.severity is not severity:
                mismatches.append(
                    f"{rule_id} ({source}): analyzer emits {severity.value}, "
                    f"catalog documents {rule.severity.value}"
                )
    if mismatches:
        pytest.fail("Catalog severity disagrees with analyzer:\n  " + "\n  ".join(mismatches))


def test_analyzer_name_tables_match_catalog() -> None:
    """The rule_name carried on a finding must equal the catalog's name."""
    mismatches: list[str] = []
    for source, table in _NAME_TABLES:
        for rule_id, name in table.items():
            rule = RuleCatalog.get_rule(rule_id)
            if rule.name != name:
                mismatches.append(
                    f"{rule_id} ({source}): analyzer emits {name!r}, catalog says {rule.name!r}"
                )
    for rule_id, meta in pickle_analyzer._RULE_METADATA.items():
        rule = RuleCatalog.get_rule(rule_id)
        if rule.name != meta["name"]:
            mismatches.append(
                f"{rule_id} (pickle): analyzer emits {meta['name']!r}, "
                f"catalog says {rule.name!r}"
            )
    if mismatches:
        pytest.fail("Catalog name disagrees with analyzer:\n  " + "\n  ".join(mismatches))


def test_analyzer_category_tables_match_catalog() -> None:
    """The category carried on a finding must equal the catalog's category."""
    mismatches: list[str] = []
    for source, table in _CATEGORY_TABLES:
        for rule_id, category in table.items():
            rule = RuleCatalog.get_rule(rule_id)
            if rule.category != category:
                mismatches.append(
                    f"{rule_id} ({source}): analyzer emits {category!r}, "
                    f"catalog says {rule.category!r}"
                )
    for rule_id, meta in pickle_analyzer._RULE_METADATA.items():
        rule = RuleCatalog.get_rule(rule_id)
        if rule.category != meta["category"]:
            mismatches.append(
                f"{rule_id} (pickle): analyzer emits {meta['category']!r}, "
                f"catalog says {rule.category!r}"
            )
    if mismatches:
        pytest.fail("Catalog category disagrees with analyzer:\n  " + "\n  ".join(mismatches))


def test_every_analyzer_table_rule_is_registered() -> None:
    """No analyzer may carry a rule the catalog has never heard of."""
    known = {r.rule_id for r in RuleCatalog.list_all_rules()}
    unknown: list[str] = []
    for source, table in _SEVERITY_TABLES + _NAME_TABLES + _CATEGORY_TABLES:  # type: ignore[operator]
        unknown.extend(f"{rid} ({source})" for rid in table if rid not in known)
    unknown.extend(f"{rid} (pickle)" for rid in pickle_analyzer._RULE_METADATA if rid not in known)
    assert not unknown, f"Rule IDs used by analyzers but absent from the catalog: {sorted(unknown)}"


# ---------------------------------------------------------------------------
# 2. Inline Finding(...) constructions vs catalog
# ---------------------------------------------------------------------------


def _literal_findings() -> list[tuple[str, str, dict[str, ast.expr]]]:
    """Return (file, rule_id, keyword-nodes) for every literal-rule_id Finding call."""
    out: list[tuple[str, str, dict[str, ast.expr]]] = []
    for py_file in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Finding"):
                continue
            kws = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            rid_node = kws.get("rule_id")
            if isinstance(rid_node, ast.Constant) and isinstance(rid_node.value, str):
                out.append((py_file.name, rid_node.value, kws))
    return out


def test_inline_findings_match_catalog() -> None:
    """Findings built inline (PyTorch, normalizer, engine) must match the catalog too."""
    problems: list[str] = []
    checked = 0

    for filename, rule_id, kws in _literal_findings():
        if not RuleCatalog.has_rule(rule_id):
            problems.append(f"{filename}: {rule_id} is not registered in the catalog")
            continue
        rule = RuleCatalog.get_rule(rule_id)
        checked += 1

        name_node = kws.get("rule_name")
        if isinstance(name_node, ast.Constant) and name_node.value != rule.name:
            problems.append(
                f"{filename}: {rule_id} rule_name {name_node.value!r} != catalog {rule.name!r}"
            )

        cat_node = kws.get("category")
        if isinstance(cat_node, ast.Constant) and cat_node.value != rule.category:
            problems.append(
                f"{filename}: {rule_id} category {cat_node.value!r} != catalog {rule.category!r}"
            )

        sev_node = kws.get("severity")
        if (
            isinstance(sev_node, ast.Attribute)
            and isinstance(sev_node.value, ast.Name)
            and sev_node.value.id == "Severity"
            and sev_node.attr != rule.severity.value
        ):
            problems.append(
                f"{filename}: {rule_id} severity {sev_node.attr} != catalog {rule.severity.value}"
            )

    assert checked, "No literal Finding(...) constructions were found — has the AST scan broken?"
    if problems:
        pytest.fail("Inline findings disagree with the catalog:\n  " + "\n  ".join(problems))
