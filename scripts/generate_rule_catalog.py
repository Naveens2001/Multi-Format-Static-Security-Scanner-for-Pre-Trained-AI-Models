"""
scripts/generate_rule_catalog.py
==================================

Generate a human-readable Markdown rule catalog from the centralized
RuleCatalog in model_scanner.rules.catalog.

Usage:
    python scripts/generate_rule_catalog.py [--output PATH]

Outputs the file to docs/rule_catalog.md by default.

SECURITY NOTE: This script only reads metadata from the in-memory catalog —
it performs no I/O other than writing the output file, and imports nothing
that could execute model code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src layout is importable when run from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from model_scanner.core.models import Severity
from model_scanner.rules.catalog import RuleCatalog
from model_scanner.rules.models import SecurityRule


_SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "⚪",
}

_SECTION_HEADERS: list[tuple[str, str]] = [
    ("P1", "Pickle — Code Execution (P1xx)"),
    ("P2", "Pickle / PyTorch ZIP — Structural Anomalies (P2xx)"),
    ("ST1", "SafeTensors — Metadata (ST1xx)"),
    ("ST2", "SafeTensors — Structural Anomalies (ST2xx)"),
    ("GU1", "GGUF — Metadata (GU1xx)"),
    ("GU2", "GGUF — Structural Anomalies (GU2xx)"),
    ("G0", "Cross-Format Integrity (G0xx)"),
    ("SC1", "Supply Chain — Config & Code Execution (SC1xx)"),
    ("SC2", "Supply Chain — Dependency Security (SC2xx)"),
    ("C0", "Correlated Cross-Signal Rules (C0xx)"),
    ("WI1", "Cross-Format Weight Integrity (WI1xx)"),
    ("ON1", "ONNX Security Rules (ON1xx)"),
    ("TF1", "TensorFlow SavedModel Security Rules (TF1xx)"),
    ("KR1", "Keras Security Rules (KR1xx)"),
]


def _group_rules() -> dict[str, list[SecurityRule]]:
    groups: dict[str, list[SecurityRule]] = {prefix: [] for prefix, _ in _SECTION_HEADERS}
    for rule in RuleCatalog.list_all_rules():
        for prefix, _ in _SECTION_HEADERS:
            if rule.rule_id.startswith(prefix):
                groups[prefix].append(rule)
                break
    return groups


def generate_markdown() -> str:
    rules = RuleCatalog.list_all_rules()
    groups = _group_rules()
    total = len(rules)

    lines: list[str] = [
        "# Static Security Rule Catalog",
        "",
        f"**Total rules:** {total}  ",
        "**Scanner version:** 2.0.0  ",
        "**Format:** `RULE_ID — Name — Severity — Category`",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]

    for prefix, header in _SECTION_HEADERS:
        count = len(groups[prefix])
        if count == 0:
            continue
        anchor = header.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("/", "").replace("&", "")
        anchor = "-".join(w for w in anchor.split("-") if w)
        lines.append(f"- [{header}](#{anchor}) ({count} rules)")

    lines += ["", "---", ""]

    for prefix, header in _SECTION_HEADERS:
        section_rules = groups[prefix]
        if not section_rules:
            continue

        lines += [f"## {header}", ""]

        for rule in sorted(section_rules, key=lambda r: r.rule_id):
            emoji = _SEVERITY_EMOJI.get(rule.severity, "⚫")
            lines += [
                f"### {rule.rule_id} — {rule.name}",
                "",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| **Severity** | {emoji} `{rule.severity.value}` |",
                f"| **Category** | `{rule.category}` |",
                f"| **Affected Format** | `{rule.affected_format}` |",
            ]

            if rule.cwe_ids:
                cwe_str = ", ".join(f"`{c}`" for c in rule.cwe_ids)
                lines.append(f"| **CWE** | {cwe_str} |")

            owasp_display = rule.owasp_category if rule.owasp_category else "Not directly mapped"
            lines.append(f"| **OWASP ML** | {owasp_display} |")

            if rule.mitre_technique:
                lines.append(f"| **MITRE ATT&CK** | `{rule.mitre_technique}` |")

            if rule.cve_ids:
                cve_str = ", ".join(f"`{c}`" for c in rule.cve_ids)
                lines.append(f"| **CVE** | {cve_str} |")

            lines += [
                "",
                f"**Description:** {rule.description}",
                "",
                f"**Detection Condition:** {rule.detection_condition}",
                "",
                f"**Remediation:** {rule.remediation}",
                "",
                "---",
                "",
            ]

    lines += [
        "_Generated automatically by `scripts/generate_rule_catalog.py`._",
        "_Do not edit manually — update `src/model_scanner/rules/catalog.py` and regenerate._",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown rule catalog from RuleCatalog."
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "docs" / "rule_catalog.md"),
        help="Output file path (default: docs/rule_catalog.md)",
    )
    args = parser.parse_args()

    content = generate_markdown()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Rule catalog written to {out_path} ({len(RuleCatalog.list_all_rules())} rules).")


if __name__ == "__main__":
    main()
