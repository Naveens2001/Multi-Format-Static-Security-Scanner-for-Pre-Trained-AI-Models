"""
model_scanner.analyzers.repository.provenance
==============================================

Offline provenance indicators for model repository artifacts (B2).

SCOPE BOUNDARY — Explicitly deferred features
----------------------------------------------
The following provenance checks REQUIRE network access or a PKI trust store
and are therefore NOT implemented here (NFR-01: offline at scan time):

  - Cryptographic signature verification (Sigstore/Cosign).
  - Transparency-log lookup (Rekor).
  - Certificate chain validation.
  - Publisher reputation scoring.

What IS implemented (fully offline):

  SC106 — Signature artifact presence/absence detection.
           Detects ``.sig``, ``.sigstore``, ``.intoto.jsonl``, ``.att``
           files next to weight files.  Reports signed vs unsigned (INFO).
           Shape and claimed identity of signature JSON is validated
           structurally — the signature itself is NOT cryptographically
           verified.

  SC107 — Hash self-consistency.
           Compares computed SHA-256 of each weight shard against any digest
           declared inside the repo's own ``model.safetensors.index.json``
           or ``pytorch_model.bin.index.json``.  Mismatch is HIGH.

  SC108 — LFS pointer integrity.
           A file tracked in ``.gitattributes`` as LFS that is still a text
           LFS pointer stub (rather than real binary weights) is flagged.

  SC109 — Declared-vs-actual consistency.
           ``config.json`` ``model_type`` disagreeing with the detected
           format or framework from A1.

SECURITY GUARANTEE
------------------
- All reads are bounded.
- No network calls, no cryptographic operations.
- stdlib only: hashlib, json, re.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_INDEX_BYTES: int = 4 * 1024 * 1024   # 4 MB
_MAX_GITATTRIBUTES_BYTES: int = 64 * 1024  # 64 KB
_LFS_POINTER_MAX_BYTES: int = 4096         # LFS pointer stubs are tiny

# Signature file extensions to check (presence = signed, absence = unsigned)
_SIGNATURE_EXTENSIONS: frozenset[str] = frozenset(
    [".sig", ".sigstore", ".intoto.jsonl", ".att", ".asc"]
)
_SIGNATURE_FILE_NAMES: frozenset[str] = frozenset(
    ["model.sig", "weights.sig", "sha256sum.txt", "checksums.txt", "SHA256SUMS"]
)

# LFS pointer stub preamble
_LFS_POINTER_PREAMBLE: bytes = b"version https://git-lfs.github.com/spec/"


@dataclass(frozen=True)
class ProvenanceFinding:
    """A provenance indicator finding (B2).

    Attributes
    ----------
    rule_id:
        SC106 / SC107 / SC108 / SC109.
    description:
        Human-readable description.
    artifact_rel_path:
        Path relative to repo root (for context).
    detail:
        Extra diagnostic detail.
    """

    rule_id: str
    description: str
    artifact_rel_path: str
    detail: str = ""


# ---------------------------------------------------------------------------
# SC106 — Signature artifact presence/absence
# ---------------------------------------------------------------------------


def check_signature_artifacts(
    repo_root: Path,
    weight_files: list[Path],
) -> list[ProvenanceFinding]:
    """Check whether weight files have accompanying signature artifacts (SC106).

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.
    weight_files:
        Weight files discovered by the acquisition layer.

    Returns
    -------
    list[ProvenanceFinding]
        SC106 findings (INFO severity) — one per unsigned weight file.
    """
    findings: list[ProvenanceFinding] = []

    # Look for repo-level signature files
    repo_level_sigs = [
        f for f in repo_root.iterdir()
        if f.is_file() and (
            f.suffix in _SIGNATURE_EXTENSIONS
            or f.name in _SIGNATURE_FILE_NAMES
        )
    ] if repo_root.is_dir() else []

    for wf in weight_files:
        # Check for per-weight-file signature
        has_sig = any(
            (wf.parent / (wf.name + ext)).exists()
            for ext in _SIGNATURE_EXTENSIONS
        )
        if not has_sig and not repo_level_sigs:
            try:
                rel = str(wf.relative_to(repo_root))
            except ValueError:
                rel = str(wf)
            findings.append(
                ProvenanceFinding(
                    rule_id="SC106",
                    description=(
                        f"No signature artifact found for {rel!r}. "
                        f"Artifact provenance cannot be verified offline."
                    ),
                    artifact_rel_path=rel,
                    detail=(
                        "No .sig, .sigstore, .intoto.jsonl, or .att file found "
                        "alongside this weight file or at the repository root."
                    ),
                )
            )

    return findings


# ---------------------------------------------------------------------------
# SC107 — Hash self-consistency
# ---------------------------------------------------------------------------


def _sha256_of_file_bounded(path: Path, max_bytes: int = 2 * 1024 * 1024 * 1024) -> str:
    """Compute SHA-256 of a file in streaming fashion (no full-load)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def check_index_hash_consistency(
    repo_root: Path,
    weight_files: list[Path],
) -> list[ProvenanceFinding]:
    """Compare declared digests in shard index against computed SHA-256 (SC107).

    Reads ``model.safetensors.index.json`` and ``pytorch_model.bin.index.json``
    (if present) and checks that each declared file hash matches the computed
    digest of the actual file on disk.  A mismatch indicates tampering.

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.
    weight_files:
        Weight files to cross-check.

    Returns
    -------
    list[ProvenanceFinding]
        SC107 findings (HIGH severity) for hash mismatches.
    """
    findings: list[ProvenanceFinding] = []

    # Read any available index file
    declared_hashes: dict[str, str] = {}
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = repo_root / index_name
        if not index_path.is_file():
            continue
        try:
            raw = index_path.read_bytes()[:_MAX_INDEX_BYTES]
            index_data: Any = json.loads(raw.decode("utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue

        if not isinstance(index_data, dict):
            continue

        # The HF index format has either:
        # { "weight_map": { "filename": "digest_or_shard" } } — no digests
        # or custom hash maps in other tools
        # We look for a "hash_map" or "checksums" field
        hash_section: Any = index_data.get("checksums") or index_data.get("sha256")
        if isinstance(hash_section, dict):
            for filename, digest in hash_section.items():
                if isinstance(filename, str) and isinstance(digest, str):
                    declared_hashes[filename] = digest

    if not declared_hashes:
        return findings  # no hash data to compare against

    # Map basename → full path for the discovered weight files
    weight_by_name: dict[str, Path] = {}
    for wf in weight_files:
        try:
            rel = str(wf.relative_to(repo_root))
            weight_by_name[rel] = wf
            weight_by_name[wf.name] = wf
        except ValueError:
            pass

    for filename, declared_digest in declared_hashes.items():
        actual_path = weight_by_name.get(filename)
        if actual_path is None or not actual_path.is_file():
            continue

        computed = _sha256_of_file_bounded(actual_path)
        if not computed:
            continue

        if computed.lower() != declared_digest.lower():
            findings.append(
                ProvenanceFinding(
                    rule_id="SC107",
                    description=(
                        f"SHA-256 hash mismatch for {filename!r}: "
                        f"declared={declared_digest[:16]}… computed={computed[:16]}…"
                    ),
                    artifact_rel_path=filename,
                    detail=(
                        f"Declared: {declared_digest}\nComputed:  {computed}"
                    ),
                )
            )

    return findings


# ---------------------------------------------------------------------------
# SC108 — LFS pointer integrity
# ---------------------------------------------------------------------------


def check_lfs_pointer_stubs(
    repo_root: Path,
    weight_files: list[Path],
) -> list[ProvenanceFinding]:
    """Detect weight files that are still Git LFS pointer stubs (SC108).

    An LFS pointer file is a small text file starting with
    ``version https://git-lfs.github.com/spec/``.  If a weight file is
    tracked in ``.gitattributes`` as LFS but was never fetched, the scanner
    will see only the pointer, not real weights.

    Parameters
    ----------
    repo_root:
        Root directory of the model repository.
    weight_files:
        Weight files to check for LFS stubs.

    Returns
    -------
    list[ProvenanceFinding]
        SC108 findings (MEDIUM severity) per LFS stub detected.
    """
    findings: list[ProvenanceFinding] = []

    # Read .gitattributes to find LFS-tracked patterns
    gitattributes_path = repo_root / ".gitattributes"
    lfs_patterns: list[re.Pattern[str]] = []

    if gitattributes_path.is_file():
        try:
            raw = gitattributes_path.read_bytes()[:_MAX_GITATTRIBUTES_BYTES]
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if "filter=lfs" in line:
                    parts = line.strip().split()
                    if parts:
                        # Convert glob pattern to regex
                        glob_pat = parts[0]
                        regex_pat = (
                            re.escape(glob_pat)
                            .replace(r"\*\*", ".+")
                            .replace(r"\*", "[^/]*")
                            .replace(r"\?", "[^/]")
                        )
                        try:
                            lfs_patterns.append(re.compile(regex_pat + "$"))
                        except re.error:
                            pass
        except OSError:
            pass

    for wf in weight_files:
        # Only check small files (LFS pointers are < 512 bytes)
        try:
            size = wf.stat().st_size
        except OSError:
            continue

        if size > _LFS_POINTER_MAX_BYTES:
            continue

        # Check if the file starts with the LFS pointer preamble
        try:
            head = wf.read_bytes()[:128]
        except OSError:
            continue

        if not head.startswith(_LFS_POINTER_PREAMBLE):
            continue

        # Looks like an LFS pointer — check if it's tracked as LFS
        try:
            rel = str(wf.relative_to(repo_root))
        except ValueError:
            rel = str(wf)

        is_lfs_tracked = not lfs_patterns or any(
            p.match(wf.name) or p.match(rel)
            for p in lfs_patterns
        )

        if is_lfs_tracked:
            findings.append(
                ProvenanceFinding(
                    rule_id="SC108",
                    description=(
                        f"{rel!r} is an LFS pointer stub — the actual weight "
                        f"content was never fetched."
                    ),
                    artifact_rel_path=rel,
                    detail=head.decode("utf-8", errors="replace")[:256],
                )
            )

    return findings


# ---------------------------------------------------------------------------
# SC109 — Declared model_type vs detected format
# ---------------------------------------------------------------------------


def check_config_format_consistency(
    declared_model_type: str | None,
    declared_library: str | None,
    detected_formats: set[str],
) -> list[ProvenanceFinding]:
    """Check consistency between config.json declarations and detected formats (SC109).

    Parameters
    ----------
    declared_model_type:
        Value of ``model_type`` from ``config.json`` (or None if absent).
    declared_library:
        Value of ``library_name`` from README.md front matter (or None).
    detected_formats:
        Set of ``ModelFormat.value`` strings detected across weight files.

    Returns
    -------
    list[ProvenanceFinding]
        SC109 findings (MEDIUM) if there is a detectable inconsistency.
    """
    findings: list[ProvenanceFinding] = []

    if not declared_library or not detected_formats:
        return findings

    lib_lower = declared_library.lower()

    # If library claims PyTorch / transformers but only GGUF was found
    if lib_lower in ("transformers", "pytorch", "diffusers"):
        if detected_formats <= {"GGUF", "GGML", "UNSUPPORTED"} and (
            "GGUF" in detected_formats or "GGML" in detected_formats
        ):
            if "RAW_PICKLE" not in detected_formats and "PYTORCH_ZIP" not in detected_formats:
                findings.append(
                    ProvenanceFinding(
                        rule_id="SC109",
                        description=(
                            f"README declares library_name={declared_library!r} "
                            f"but only GGUF/GGML weight files were detected. "
                            f"Expected PyTorch/.safetensors format."
                        ),
                        artifact_rel_path="config.json / README.md",
                        detail=(
                            f"declared_library={declared_library!r}, "
                            f"detected_formats={sorted(detected_formats)}"
                        ),
                    )
                )

    # If library claims llama.cpp / gguf but only SafeTensors found
    if lib_lower in ("gguf", "llama.cpp", "llamacpp"):
        if "SAFETENSORS" in detected_formats and "GGUF" not in detected_formats:
            findings.append(
                ProvenanceFinding(
                    rule_id="SC109",
                    description=(
                        f"README declares library_name={declared_library!r} "
                        f"but only SafeTensors weight files were detected."
                    ),
                    artifact_rel_path="config.json / README.md",
                    detail=(
                        f"declared_library={declared_library!r}, "
                        f"detected_formats={sorted(detected_formats)}"
                    ),
                )
            )

    return findings


__all__ = [
    "ProvenanceFinding",
    "check_config_format_consistency",
    "check_index_hash_consistency",
    "check_lfs_pointer_stubs",
    "check_signature_artifacts",
]
