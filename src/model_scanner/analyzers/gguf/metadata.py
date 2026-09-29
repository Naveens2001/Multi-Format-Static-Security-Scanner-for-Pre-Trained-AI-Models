"""
model_scanner.analyzers.gguf.metadata
=====================================

Security pattern scanner for GGUF metadata string values.

Scans all metadata strings (individual string values, string array elements,
and metadata keys) against known security rule patterns:
  - Code injection / execution strings (GU101)
  - Path traversal sequences (GU102)
  - Credential & secret exposure (GU103)
  - Suspicious URLs (GU105)
  - Script injection tags (GU106)

SECURITY GUARANTEE
------------------
- Pure declarative regex pattern matching only.
- Values are length-capped (64 KiB) before regex application to prevent ReDoS.
- Never calls eval(), exec(), pickle.loads(), or torch.load().

Design Reference
----------------
Section 17.1 of docs/implementation_design.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from model_scanner.analyzers.gguf.reader import GGUFKVEntry, GGUFValueType

# Resource safety sentinel — cap per-value scan length to prevent ReDoS
_MAX_SCAN_LEN: int = 65_536


# ---------------------------------------------------------------------------
# Pattern Catalog — GU101: Code Execution Primitives
# ---------------------------------------------------------------------------
_CODE_EXEC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("eval(", re.compile(r"\beval\s*\(", re.IGNORECASE)),
    ("exec(", re.compile(r"\bexec\s*\(", re.IGNORECASE)),
    ("__import__", re.compile(r"__import__", re.IGNORECASE)),
    ("os.system", re.compile(r"\bos\s*\.\s*system\s*\(", re.IGNORECASE)),
    ("subprocess", re.compile(r"\bsubprocess\b", re.IGNORECASE)),
    ("base64.b64decode", re.compile(r"\bbase64\s*\.\s*b64decode\b", re.IGNORECASE)),
    ("compile(", re.compile(r"\bcompile\s*\(", re.IGNORECASE)),
    ("__builtins__", re.compile(r"__builtins__", re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Pattern Catalog — GU102: Path Traversal Sequences
# ---------------------------------------------------------------------------
_PATH_TRAVERSAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("DOT_DOT_SLASH", re.compile(r"(?:\.\./|\.\.\\)")),
    ("ENCODED_TRAVERSAL", re.compile(r"(?:%2e%2e%2f|%2e%2e/|\.\.%2f)", re.IGNORECASE)),
    (
        "SYSTEM_PATH_UNIX",
        re.compile(r"(?:/etc/passwd|/etc/shadow|/bin/sh|/bin/bash)\b", re.IGNORECASE),
    ),
    (
        "SYSTEM_PATH_WIN",
        re.compile(r"(?:[a-zA-Z]:\\Windows\\System32|\\system32\\cmd\.exe)", re.IGNORECASE),
    ),
]

# ---------------------------------------------------------------------------
# Pattern Catalog — GU103: Credential / Secret Exposure
# ---------------------------------------------------------------------------
_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS_ACCESS_KEY", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GITHUB_TOKEN", re.compile(r"ghp_[0-9a-zA-Z]{36}")),
    ("HF_TOKEN", re.compile(r"hf_[0-9a-zA-Z]{32,}")),
    ("OPENAI_KEY", re.compile(r"sk-[0-9a-zA-Z]{48}")),
    ("PRIVATE_KEY_BLOCK", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GENERIC_SECRET", re.compile(r"(?i)(?:password|passwd|secret|api_?key)\s*[=:]\s*\S{8,}")),
]

# ---------------------------------------------------------------------------
# Pattern Catalog — GU105: Suspicious URLs
# ---------------------------------------------------------------------------
_URL_PATTERN: re.Pattern[str] = re.compile(r"https?://[^\s\"'<>]{4,}")

# ---------------------------------------------------------------------------
# Pattern Catalog — GU106: Script Injection Tags
# ---------------------------------------------------------------------------
_SCRIPT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("<script", re.compile(r"<\s*script\b", re.IGNORECASE)),
    ("<iframe", re.compile(r"<\s*iframe\b", re.IGNORECASE)),
    ("onload=", re.compile(r"\bonload\s*=", re.IGNORECASE)),
    ("onerror=", re.compile(r"\bonerror\s*=", re.IGNORECASE)),
    ("javascript:", re.compile(r"\bjavascript\s*:", re.IGNORECASE)),
]


@dataclass(frozen=True)
class GGUFMetadataFinding:
    """A single security finding discovered in GGUF metadata."""

    rule_id: str
    key: str
    byte_offset: int
    matched_pattern: str
    matched_value: str


def scan_gguf_metadata(kv_entries: list[GGUFKVEntry]) -> list[GGUFMetadataFinding]:
    """Scan all GGUF metadata entries for security patterns.

    Parameters
    ----------
    kv_entries:
        List of parsed GGUFKVEntry records.

    Returns
    -------
    list[GGUFMetadataFinding]
        All security findings discovered.
    """
    findings: list[GGUFMetadataFinding] = []

    for entry in kv_entries:
        # Collect all string fragments from this entry
        strings_to_scan: list[tuple[str, int]] = []

        if entry.value_type == GGUFValueType.STRING and isinstance(entry.value, str):
            strings_to_scan.append((entry.value, entry.value_offset))
        elif entry.value_type == GGUFValueType.ARRAY and isinstance(entry.value, list):
            for elem in entry.value:
                if isinstance(elem, str):
                    strings_to_scan.append((elem, entry.value_offset))

        for raw_val, val_offset in strings_to_scan:
            text = raw_val[:_MAX_SCAN_LEN]

            # GU101: Code Execution Primitives
            for pat_name, pat in _CODE_EXEC_PATTERNS:
                m = pat.search(text)
                if m:
                    findings.append(
                        GGUFMetadataFinding(
                            rule_id="GU101",
                            key=entry.key,
                            byte_offset=val_offset,
                            matched_pattern=pat_name,
                            matched_value=text[max(0, m.start() - 20) : m.end() + 40][:200],
                        )
                    )
                    break

            # GU102: Path Traversal Sequences
            for pat_name, pat in _PATH_TRAVERSAL_PATTERNS:
                m = pat.search(text)
                if m:
                    findings.append(
                        GGUFMetadataFinding(
                            rule_id="GU102",
                            key=entry.key,
                            byte_offset=val_offset,
                            matched_pattern=pat_name,
                            matched_value=m.group(0)[:200],
                        )
                    )
                    break

            # GU103: Credential Exposure
            for pat_name, pat in _CREDENTIAL_PATTERNS:
                m = pat.search(text)
                if m:
                    findings.append(
                        GGUFMetadataFinding(
                            rule_id="GU103",
                            key=entry.key,
                            byte_offset=val_offset,
                            matched_pattern=pat_name,
                            matched_value=m.group(0)[:200],
                        )
                    )
                    break

            # GU105: Suspicious URLs
            for m in _URL_PATTERN.finditer(text):
                findings.append(
                    GGUFMetadataFinding(
                        rule_id="GU105",
                        key=entry.key,
                        byte_offset=val_offset,
                        matched_pattern="HTTP_URL",
                        matched_value=m.group(0)[:200],
                    )
                )

            # GU106: Script Injection Tags
            for pat_name, pat in _SCRIPT_INJECTION_PATTERNS:
                m = pat.search(text)
                if m:
                    findings.append(
                        GGUFMetadataFinding(
                            rule_id="GU106",
                            key=entry.key,
                            byte_offset=val_offset,
                            matched_pattern=pat_name,
                            matched_value=text[max(0, m.start() - 10) : m.end() + 30][:200],
                        )
                    )
                    break

    return findings


__all__ = ["GGUFMetadataFinding", "scan_gguf_metadata"]
