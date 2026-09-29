"""
model_scanner.analyzers.safetensors.metadata
=============================================

Security pattern scanner for the SafeTensors ``__metadata__`` dictionary.

The SafeTensors format allows an optional ``__metadata__`` key at the root of
the JSON header.  Its value must be a ``dict[str, str]`` (string keys mapping
to string values).  This module scans those string values against patterns
associated with:

- Code execution primitives (ST101)
- Credential / secret exposure (ST102)
- Embedded HTTP/HTTPS URLs (ST103)
- Script injection tags (ST104)

SECURITY GUARANTEE
------------------
- All pattern matching is purely declarative (``re`` module only).
- Each metadata value is capped at ``_MAX_METADATA_VALUE_BYTES`` characters
  before regex application to prevent catastrophic backtracking (ReDoS).
- Never calls pickle.loads(), torch.load(), eval(), or exec().
- Does not import torch, safetensors, or any ML framework.

Design Reference
----------------
Section 16.4 of docs/implementation_design.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Resource safety sentinel — cap per-value scan length to prevent ReDoS
# ---------------------------------------------------------------------------
_MAX_METADATA_VALUE_BYTES: int = 65_536  # 64 KiB per value


# ---------------------------------------------------------------------------
# Pattern Catalog — ST101: Code Execution Primitives
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
# Pattern Catalog — ST102: Path Traversal Sequences
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
# Pattern Catalog — ST103: Suspicious URL / C2 Link
# ---------------------------------------------------------------------------
_URL_PATTERN: re.Pattern[str] = re.compile(r"https?://[^\s\"'<>]{4,}")

# ---------------------------------------------------------------------------
# Pattern Catalog — ST104: Credential / Secret Exposure
# ---------------------------------------------------------------------------
_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS_ACCESS_KEY", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GITHUB_TOKEN", re.compile(r"ghp_[0-9a-zA-Z]{36}")),
    ("HF_TOKEN", re.compile(r"hf_[0-9a-zA-Z]{32,}")),
    ("SK_API_KEY", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("PRIVATE_KEY_BLOCK", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GENERIC_SECRET", re.compile(r"(?i)(?:password|passwd|secret|api_?key)\s*[=:]\s*\S{8,}")),
]

# ---------------------------------------------------------------------------
# Pattern Catalog — ST105: Script Injection Tags
# ---------------------------------------------------------------------------
_SCRIPT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("<script", re.compile(r"<\s*script\b", re.IGNORECASE)),
    ("<iframe", re.compile(r"<\s*iframe\b", re.IGNORECASE)),
    ("onload=", re.compile(r"\bonload\s*=", re.IGNORECASE)),
    ("onerror=", re.compile(r"\bonerror\s*=", re.IGNORECASE)),
    ("javascript:", re.compile(r"\bjavascript\s*:", re.IGNORECASE)),
]


@dataclass(frozen=True)
class MetadataFinding:
    """A single security pattern match found in ``__metadata__``.

    Attributes
    ----------
    rule_id : str
        ``ST101``, ``ST102``, ``ST103``, ``ST104``, or ``ST105``.
    key : str
        The ``__metadata__`` dictionary key where the pattern was found.
    matched_pattern : str
        Human-readable label for the matched pattern.
    matched_value : str
        The portion of the value that triggered the match (truncated to 200
        characters for safe embedding in ``Evidence.matched_value``).
    """

    rule_id: str
    key: str
    matched_pattern: str
    matched_value: str


def scan_metadata(metadata: object) -> list[MetadataFinding]:
    """Scan the ``__metadata__`` dictionary for security-relevant patterns.

    Parameters
    ----------
    metadata:
        The value associated with the ``__metadata__`` key in the parsed
        JSON header.  Must be a ``dict[str, str]`` per the SafeTensors
        format specification.

    Returns
    -------
    list[MetadataFinding]
        All findings discovered.  Empty list if nothing was found.
    """
    findings: list[MetadataFinding] = []

    if not isinstance(metadata, dict):
        # Non-dict __metadata__ is a structural violation handled separately.
        return findings

    for key, raw_value in metadata.items():
        if not isinstance(key, str):
            continue
        if not isinstance(raw_value, str):
            # Non-string values are unexpected; skip pattern scanning.
            continue

        # Cap the value length to prevent ReDoS on adversarial inputs.
        value = raw_value[:_MAX_METADATA_VALUE_BYTES]

        # ST101: Code execution patterns
        for pattern_label, pattern in _CODE_EXEC_PATTERNS:
            m = pattern.search(value)
            if m:
                findings.append(
                    MetadataFinding(
                        rule_id="ST101",
                        key=key,
                        matched_pattern=pattern_label,
                        matched_value=value[max(0, m.start() - 20) : m.end() + 40][
                            :200
                        ],
                    )
                )
                break  # One ST101 per key is sufficient; avoid duplicate findings

        # ST102: Path traversal sequences
        for pattern_label, pattern in _PATH_TRAVERSAL_PATTERNS:
            m = pattern.search(value)
            if m:
                findings.append(
                    MetadataFinding(
                        rule_id="ST102",
                        key=key,
                        matched_pattern=pattern_label,
                        matched_value=m.group(0)[:200],
                    )
                )
                break  # One path-traversal finding per key

        # ST103: URL patterns
        for m in _URL_PATTERN.finditer(value):
            findings.append(
                MetadataFinding(
                    rule_id="ST103",
                    key=key,
                    matched_pattern="HTTP_URL",
                    matched_value=m.group(0)[:200],
                )
            )

        # ST104: Credential patterns
        for pattern_label, pattern in _CREDENTIAL_PATTERNS:
            m = pattern.search(value)
            if m:
                findings.append(
                    MetadataFinding(
                        rule_id="ST104",
                        key=key,
                        matched_pattern=pattern_label,
                        matched_value=m.group(0)[:200],
                    )
                )
                break  # One credential finding per key

        # ST105: Script injection patterns
        for pattern_label, pattern in _SCRIPT_INJECTION_PATTERNS:
            m = pattern.search(value)
            if m:
                findings.append(
                    MetadataFinding(
                        rule_id="ST105",
                        key=key,
                        matched_pattern=pattern_label,
                        matched_value=value[max(0, m.start() - 10) : m.end() + 30][
                            :200
                        ],
                    )
                )
                break  # One injection finding per key

    return findings


def scan_tensor_names(tensor_names: list[str]) -> list[MetadataFinding]:
    """Scan SafeTensors tensor names (top-level JSON keys) for path traversal patterns.

    Legitimate tensor names never contain file-system traversal sequences
    (``../``, ``..\\``, ``%2e%2e%2f``, ``/etc/passwd``, etc.).  Their
    presence in a tensor name is an unambiguous indicator of tampering or
    malicious intent and fires ST102 regardless of the tensor's data_offsets.

    Parameters
    ----------
    tensor_names:
        All top-level JSON keys from the parsed header, excluding
        ``__metadata__``.

    Returns
    -------
    list[MetadataFinding]
        One ST102 finding per tensor name that matches a traversal pattern.
    """
    findings: list[MetadataFinding] = []
    for name in tensor_names:
        text = name[:_MAX_METADATA_VALUE_BYTES]
        for pat_name, pat in _PATH_TRAVERSAL_PATTERNS:
            m = pat.search(text)
            if m:
                findings.append(
                    MetadataFinding(
                        rule_id="ST102",
                        key=name,
                        matched_pattern=pat_name,
                        matched_value=m.group(0)[:200],
                    )
                )
                break  # One ST102 per tensor name is sufficient
    return findings


__all__ = ["MetadataFinding", "scan_metadata", "scan_tensor_names"]
