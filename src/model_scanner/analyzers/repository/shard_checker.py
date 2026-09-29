"""
model_scanner.analyzers.repository.shard_checker
=================================================

Sharded model integrity checking (A2).

Real Hugging Face repos ship ``model-00001-of-00003.safetensors`` alongside
``model.safetensors.index.json`` (or ``pytorch_model.bin.index.json``).
This module groups shards, reads the index, and flags:

  G005 — Shard declared in the index but missing on disk.
  G006 — Shard file present on disk but NOT declared in the index.
  G007 — Declared shard size disagrees with the actual file size.
  G008 — YYYYY in the shard filename disagrees with the actual total count.

SECURITY GUARANTEE
------------------
- Reads bounded JSON with stdlib ``json``.
- No network calls.
- No dynamic execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_INDEX_BYTES: int = 4 * 1024 * 1024   # 4 MB

# Matches Hugging Face shard filenames like:
#   model-00001-of-00003.safetensors
#   pytorch_model-00001-of-00002.bin
_SHARD_NAME_RE: re.Pattern[str] = re.compile(
    r"^(.+)-(\d+)-of-(\d+)(\.[a-zA-Z0-9]+)$"
)


@dataclass(frozen=True)
class ShardGroup:
    """A logical group of shard files sharing the same base name.

    Attributes
    ----------
    base_name:
        Filename prefix before the shard index, e.g. ``model``.
    extension:
        File extension, e.g. ``.safetensors``.
    claimed_total:
        The YYYYY total count from the filename (or 0 if mixed/inconsistent).
    shard_files:
        Sorted list of (shard_index, path) tuples for files found on disk.
    """

    base_name: str
    extension: str
    claimed_total: int
    shard_files: tuple[tuple[int, Path], ...]


@dataclass(frozen=True)
class ShardFinding:
    """A shard integrity finding.

    Attributes
    ----------
    rule_id:
        G005 / G006 / G007 / G008.
    description:
        Human-readable description.
    shard_filename:
        The relevant shard filename.
    detail:
        Extra diagnostic detail.
    """

    rule_id: str
    description: str
    shard_filename: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Shard file grouping
# ---------------------------------------------------------------------------


def group_shard_files(paths: list[Path]) -> list[ShardGroup]:
    """Group a list of file paths by shard naming convention.

    Parameters
    ----------
    paths:
        All file paths in the repository (or at least the weight files).

    Returns
    -------
    list[ShardGroup]
        One ``ShardGroup`` per detected shard group.  Non-shard files are
        ignored.
    """
    # base_key → list of (shard_index, path, claimed_total, ext)
    grouped: dict[tuple[str, str], list[tuple[int, Path, int]]] = {}

    for p in paths:
        m = _SHARD_NAME_RE.match(p.name)
        if not m:
            continue
        base, idx_str, total_str, ext = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            idx = int(idx_str)
            total = int(total_str)
        except ValueError:
            continue
        key = (base, ext)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((idx, p, total))

    result: list[ShardGroup] = []
    for (base, ext), entries in grouped.items():
        totals = {e[2] for e in entries}
        claimed_total = totals.pop() if len(totals) == 1 else 0
        shard_files = tuple(sorted((idx, p) for idx, p, _ in entries))
        result.append(
            ShardGroup(
                base_name=base,
                extension=ext,
                claimed_total=claimed_total,
                shard_files=shard_files,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------


def _read_index(index_path: Path) -> dict[str, Any] | None:
    """Read and parse a shard index JSON file."""
    try:
        raw = index_path.read_bytes()[:_MAX_INDEX_BYTES]
        obj: Any = json.loads(raw.decode("utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_shard_integrity(
    repo_root: Path,
    weight_files: list[Path],
) -> list[ShardFinding]:
    """Check shard integrity for all shard groups in the repository (A2).

    Reads ``model.safetensors.index.json`` and ``pytorch_model.bin.index.json``
    to compare declared shards against what is actually present on disk.

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.
    weight_files:
        Weight files discovered by the acquisition layer.

    Returns
    -------
    list[ShardFinding]
        Integrity findings.  Empty if no shard groups or index files found.
    """
    findings: list[ShardFinding] = []

    # Collect all filenames present in the repo (within weight files)
    present_names: set[str] = {wf.name for wf in weight_files}
    present_rel: dict[str, Path] = {}
    for wf in weight_files:
        try:
            rel = str(wf.relative_to(repo_root))
            present_rel[rel] = wf
            present_rel[wf.name] = wf
        except ValueError:
            pass

    # Process each available index file
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = repo_root / index_name
        if not index_path.is_file():
            continue

        index_data = _read_index(index_path)
        if index_data is None:
            continue

        weight_map: Any = index_data.get("weight_map") or index_data.get("tensor_map")
        if not isinstance(weight_map, dict):
            continue

        # Declared shards: unique shard filenames from the weight_map values
        declared_shards: set[str] = {
            v for v in weight_map.values() if isinstance(v, str)
        }

        # G005: Declared shards missing on disk
        for shard_name in sorted(declared_shards):
            if shard_name not in present_names and shard_name not in present_rel:
                findings.append(
                    ShardFinding(
                        rule_id="G005",
                        description=(
                            f"Shard {shard_name!r} is declared in {index_name} "
                            f"but is not present on disk."
                        ),
                        shard_filename=shard_name,
                        detail=f"Source index: {index_name}",
                    )
                )

        # G006: Weight files with shard-like names that are NOT in the index
        for _rel_name, wf in present_rel.items():
            fname = wf.name
            if _SHARD_NAME_RE.match(fname) and fname not in declared_shards:
                findings.append(
                    ShardFinding(
                        rule_id="G006",
                        description=(
                            f"Shard file {fname!r} is present on disk but is NOT "
                            f"declared in {index_name}."
                        ),
                        shard_filename=fname,
                        detail=f"Source index: {index_name}",
                    )
                )
                break  # Report once per undeclared shard

        # G007: Check sizes if the index declares them
        shard_sizes: Any = index_data.get("file_sizes") or index_data.get("shard_sizes")
        if isinstance(shard_sizes, dict):
            for shard_name, declared_size in shard_sizes.items():
                if not isinstance(declared_size, int):
                    continue
                actual_path = present_rel.get(shard_name)
                if actual_path is None:
                    continue
                try:
                    actual_size = actual_path.stat().st_size
                except OSError:
                    continue
                if actual_size != declared_size:
                    findings.append(
                        ShardFinding(
                            rule_id="G007",
                            description=(
                                f"Shard {shard_name!r} size mismatch: "
                                f"declared={declared_size} bytes, "
                                f"actual={actual_size} bytes."
                            ),
                            shard_filename=shard_name,
                            detail=(
                                f"declared={declared_size}, actual={actual_size}, "
                                f"delta={actual_size - declared_size}"
                            ),
                        )
                    )

    # G008: Shard count in filename disagrees with actual count in the group
    shard_groups = group_shard_files(weight_files)
    for group in shard_groups:
        if group.claimed_total == 0:
            # Inconsistent totals across shards in the same group
            findings.append(
                ShardFinding(
                    rule_id="G008",
                    description=(
                        f"Shard group {group.base_name!r} has inconsistent "
                        f"YYYYY total counts across shard filenames."
                    ),
                    shard_filename=f"{group.base_name}-*-of-*{group.extension}",
                    detail="YYYYY total does not agree across all shard filenames.",
                )
            )
        elif group.claimed_total != len(group.shard_files):
            actual_count = len(group.shard_files)
            findings.append(
                ShardFinding(
                    rule_id="G008",
                    description=(
                        f"Shard group {group.base_name!r} claims YYYYY={group.claimed_total} "
                        f"but only {actual_count} shard file(s) are present on disk."
                    ),
                    shard_filename=f"{group.base_name}-*-of-{group.claimed_total}{group.extension}",
                    detail=(
                        f"claimed_total={group.claimed_total}, "
                        f"files_on_disk={actual_count}"
                    ),
                )
            )

    return findings


__all__ = [
    "ShardFinding",
    "ShardGroup",
    "check_shard_integrity",
    "group_shard_files",
]
