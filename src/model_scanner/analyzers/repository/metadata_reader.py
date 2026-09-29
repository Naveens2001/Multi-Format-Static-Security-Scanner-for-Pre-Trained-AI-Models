"""
model_scanner.analyzers.repository.metadata_reader
====================================================

Bounded, stdlib-only readers for Hugging Face repository metadata files (A3).

Reads ``config.json``, ``tokenizer_config.json``, and the YAML front matter
in ``README.md``.  All reads are strictly bounded; no YAML library is used.

SECURITY GUARANTEE
------------------
- All file reads are bounded (max 256 KB per file).
- JSON is parsed with stdlib ``json`` only.
- YAML front matter uses a minimal key: value reader — no YAML library,
  no anchors, no aliases, no nested structures.
- This module NEVER imports or calls torch, yaml, transformers, or any
  non-stdlib package.
- No network calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Maximum number of bytes to read from any single metadata file.
_MAX_METADATA_BYTES: int = 256 * 1024  # 256 KB

# Maximum YAML front-matter size (bytes) to consider valid.
_MAX_YAML_FRONTMATTER_BYTES: int = 16 * 1024  # 16 KB

# Regex to detect YAML anchors / aliases — reject these.
_YAML_ANCHOR_OR_ALIAS_RE: re.Pattern[str] = re.compile(r"[&*]")

# Simple key: value parser — scalar values on a single line only.
_YAML_KV_RE: re.Pattern[str] = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_\-]*)\s*:\s*(.*)$")

# Keys we extract from YAML front matter.
_YAML_EXTRACT_KEYS: frozenset[str] = frozenset(
    ["license", "base_model", "library_name", "tags"]
)


@dataclass
class RepoMetadata:
    """Metadata extracted from a Hugging Face-style model repository (A3).

    All fields are ``None`` when the corresponding file is absent or unparseable.
    """

    # From config.json
    model_type: str | None = None
    architectures: list[str] = field(default_factory=list)
    transformers_version: str | None = None
    torch_dtype: str | None = None

    # From tokenizer_config.json
    tokenizer_class: str | None = None
    tokenizer_version: str | None = None

    # From README.md YAML front matter
    license: str | None = None
    base_model: str | None = None
    library_name: str | None = None
    tags: list[str] = field(default_factory=list)

    # Parse errors (non-fatal — each is a string describing what failed)
    parse_errors: list[str] = field(default_factory=list)


def _read_bounded(path: Path) -> str | None:
    """Read a text file, capping at ``_MAX_METADATA_BYTES``.

    Returns ``None`` if the file cannot be read.
    """
    try:
        raw = path.read_bytes()[:_MAX_METADATA_BYTES]
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return None


def _parse_json_bounded(text: str, source: str) -> tuple[dict[str, Any], str | None]:
    """Parse JSON text with a depth guard.

    Returns ``(parsed_dict, error_string)``.  On failure the dict is empty.
    """
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return {}, f"{source}: JSON root is not an object"
        return obj, None
    except json.JSONDecodeError as exc:
        return {}, f"{source}: JSON parse error: {exc}"


def _parse_yaml_frontmatter(text: str) -> dict[str, str | list[str]]:
    """Extract a minimal key/value set from YAML front matter.

    Supports only:
    - Simple scalar: ``key: value``
    - List items: lines starting with ``  - item`` under a list-typed key

    Rejects:
    - Anchors (``&`` or ``*``)
    - Nested mappings (indented key: value lines)
    - Multi-line scalars (block / flow indicators)

    Returns an empty dict if the front matter cannot be found or is malformed.
    """
    # Front matter is enclosed by ``---`` at the very start.
    stripped = text.lstrip("﻿")  # strip BOM
    if not stripped.startswith("---"):
        return {}

    # Find the closing ``---`` delimiter.
    rest = stripped[3:].lstrip("\n\r")
    end_marker = re.search(r"^---", rest, re.MULTILINE)
    if end_marker is None:
        return {}

    frontmatter_text = rest[: end_marker.start()]
    if len(frontmatter_text.encode("utf-8")) > _MAX_YAML_FRONTMATTER_BYTES:
        return {}

    # Reject anchors / aliases.
    if _YAML_ANCHOR_OR_ALIAS_RE.search(frontmatter_text):
        return {}

    result: dict[str, str | list[str]] = {}
    current_list_key: str | None = None

    for raw_line in frontmatter_text.splitlines():
        line = raw_line.rstrip()

        # List continuation: "  - item"
        if current_list_key is not None and line.startswith("  - "):
            item = line[4:].strip()
            if item and not _YAML_ANCHOR_OR_ALIAS_RE.search(item):
                existing = result.get(current_list_key, [])
                if isinstance(existing, list):
                    existing.append(item)
                    result[current_list_key] = existing
            continue

        current_list_key = None  # reset on any non-list line

        # Skip blank lines and comment lines
        if not line or line.lstrip().startswith("#"):
            continue

        # Nested / indented structure — skip the whole key
        if line.startswith(" ") or line.startswith("\t"):
            continue

        m = _YAML_KV_RE.match(line)
        if not m:
            continue

        key, value = m.group(1), m.group(2).strip()

        if key not in _YAML_EXTRACT_KEYS:
            continue

        # Reject anchors in values
        if _YAML_ANCHOR_OR_ALIAS_RE.search(value):
            continue

        if value == "" or value == "null":
            # List mode: next indented lines are list items
            result[key] = []
            current_list_key = key
        elif value.startswith("[") and value.endswith("]"):
            # Inline flow sequence: [a, b, c]
            inner = value[1:-1]
            items = [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
            result[key] = items
        else:
            result[key] = value.strip("\"'")

    return result


def _str_or_none(v: object) -> str | None:
    """Return ``v`` as a str if it is a non-empty string, else ``None``."""
    if isinstance(v, str) and v:
        return v
    return None


def _list_of_str(v: object) -> list[str]:
    """Coerce ``v`` to a list of strings, filtering non-strings."""
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    if isinstance(v, str):
        return [v]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_repo_metadata(repo_root: Path) -> RepoMetadata:
    """Read and merge metadata from a model repository directory (A3).

    Reads (in order, all bounded):
    1. ``config.json``         — model_type, architectures, transformers_version, torch_dtype
    2. ``tokenizer_config.json`` — tokenizer class and version
    3. ``README.md``           — YAML front matter: license, base_model, library_name, tags

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.

    Returns
    -------
    RepoMetadata
        Merged metadata.  Fields absent in the repo are ``None`` or empty lists.
    """
    meta = RepoMetadata()

    # 1. config.json
    config_path = repo_root / "config.json"
    if config_path.is_file():
        text = _read_bounded(config_path)
        if text is not None:
            parsed, err = _parse_json_bounded(text, "config.json")
            if err:
                meta.parse_errors.append(err)
            else:
                meta.model_type = _str_or_none(parsed.get("model_type"))
                meta.architectures = _list_of_str(parsed.get("architectures", []))
                meta.transformers_version = _str_or_none(parsed.get("transformers_version"))
                meta.torch_dtype = _str_or_none(parsed.get("torch_dtype"))

    # 2. tokenizer_config.json
    tok_path = repo_root / "tokenizer_config.json"
    if tok_path.is_file():
        text = _read_bounded(tok_path)
        if text is not None:
            parsed, err = _parse_json_bounded(text, "tokenizer_config.json")
            if err:
                meta.parse_errors.append(err)
            else:
                meta.tokenizer_class = _str_or_none(parsed.get("tokenizer_class"))
                meta.tokenizer_version = _str_or_none(parsed.get("tokenizer_version"))

    # 3. README.md YAML front matter
    for readme_name in ("README.md", "README.rst", "README"):
        readme_path = repo_root / readme_name
        if readme_path.is_file():
            text = _read_bounded(readme_path)
            if text is not None:
                fm = _parse_yaml_frontmatter(text)
                meta.license = _str_or_none(fm.get("license"))
                meta.base_model = _str_or_none(fm.get("base_model"))
                meta.library_name = _str_or_none(fm.get("library_name"))
                raw_tags = fm.get("tags", [])
                if isinstance(raw_tags, list):
                    meta.tags = raw_tags
                elif isinstance(raw_tags, str):
                    meta.tags = [raw_tags]
            break

    return meta


__all__ = ["RepoMetadata", "read_repo_metadata"]
