"""
tests.unit.test_rule_catalog_integrity
=======================================

AST-walk each analyzer module and verify every string literal that looks like
a rule ID (P1xx/P2xx/ST1xx/ST2xx/GU1xx/GU2xx/G0xx/SC1xx/SC2xx) is registered
in the RuleCatalog.

This test catches the class of bug where a new rule is hard-coded inside an
analyzer's local table but never added to rules/catalog.py.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

from model_scanner.rules.catalog import RuleCatalog

# Pattern for valid rule IDs
_RULE_ID_RE = re.compile(r"^(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0|WI1|ON1|TF1|KR1|MS1)\d{2}$")

# Source root of the analyzer packages to walk
_ANALYZER_DIRS = [
    Path(__file__).parent.parent.parent / "src" / "model_scanner" / "analyzers",
]


def _collect_rule_id_strings(source: str) -> list[str]:
    """Extract string literals that match the rule-ID pattern from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _RULE_ID_RE.match(node.value):
                hits.append(node.value)
    return hits


def _all_analyzer_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _ANALYZER_DIRS:
        files.extend(root.rglob("*.py"))
    return sorted(files)


def test_all_analyzer_rule_ids_in_catalog() -> None:
    """Every rule ID string literal in an analyzer file must exist in the catalog."""
    catalog_ids = {r.rule_id for r in RuleCatalog.list_all_rules()}
    missing: dict[str, list[str]] = {}  # rule_id -> [file paths]

    for py_file in _all_analyzer_py_files():
        source = py_file.read_text(encoding="utf-8", errors="replace")
        for rule_id in _collect_rule_id_strings(source):
            if rule_id not in catalog_ids:
                missing.setdefault(rule_id, []).append(str(py_file))

    if missing:
        lines = [f"  {rid}: {', '.join(files)}" for rid, files in sorted(missing.items())]
        pytest.fail(
            "Rule IDs found in analyzers but missing from RuleCatalog:\n" + "\n".join(lines)
        )


def test_catalog_ids_match_canonical_pattern() -> None:
    """Every catalog rule ID must match the canonical regex."""
    for rule in RuleCatalog.list_all_rules():
        assert _RULE_ID_RE.match(rule.rule_id), (
            f"Catalog rule {rule.rule_id!r} does not match canonical pattern"
        )


def test_rule_catalog_doc_matches_generator() -> None:
    """docs/rule_catalog.md must match what scripts/generate_rule_catalog.py produces.

    If the committed doc is stale (a new rule was added to the catalog but the
    doc was not regenerated), this test fails, prompting the developer to run
    ``python scripts/generate_rule_catalog.py`` and commit the output.
    """
    project_root = Path(__file__).parent.parent.parent
    generator_script = project_root / "scripts" / "generate_rule_catalog.py"
    committed_doc = project_root / "docs" / "rule_catalog.md"

    if not generator_script.is_file():
        pytest.skip("scripts/generate_rule_catalog.py does not exist — skipping doc sync check")

    if not committed_doc.is_file():
        pytest.skip("docs/rule_catalog.md does not exist — skipping doc sync check")

    # Run the generator and call generate_markdown() directly to avoid subprocess overhead.
    spec = importlib.util.spec_from_file_location("generate_rule_catalog", generator_script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    generated: str = module.generate_markdown()  # type: ignore[attr-defined]

    committed: str = committed_doc.read_text(encoding="utf-8")

    if generated != committed:
        pytest.fail(
            "docs/rule_catalog.md is stale — run `python scripts/generate_rule_catalog.py` "
            "and commit the updated file.\n"
            f"Committed lines: {len(committed.splitlines())}, "
            f"Generated lines: {len(generated.splitlines())}"
        )


def test_rule_catalog_doc_owasp_row_present_for_all_rules() -> None:
    """Every rule in docs/rule_catalog.md must have an OWASP ML row.

    Unmapped rules must show 'Not directly mapped'; mapped rules must show
    their ML06:2023 category. No rule may silently omit the OWASP row.
    """
    import re

    project_root = Path(__file__).parent.parent.parent
    committed_doc = project_root / "docs" / "rule_catalog.md"

    if not committed_doc.is_file():
        pytest.skip("docs/rule_catalog.md does not exist")

    text = committed_doc.read_text(encoding="utf-8")
    sections = re.split(r"(?=### [A-Z])", text)

    missing_owasp: list[str] = []
    for sec in sections:
        m = re.match(r"### ([A-Z0-9]+)", sec)
        if not m:
            continue
        rid = m.group(1)
        if "**OWASP ML**" not in sec:
            missing_owasp.append(rid)

    assert not missing_owasp, (
        f"{len(missing_owasp)} rule(s) are missing an OWASP ML row in "
        f"docs/rule_catalog.md: {missing_owasp}"
    )
