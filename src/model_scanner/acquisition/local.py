"""
model_scanner.acquisition.local
================================

Local file and recursive directory acquisition providers.

Implements:
  - LocalFileAcquisition      — validates and fingerprints a single local file
  - LocalDirectoryAcquisition — recursively discovers and acquires candidate
                                model files under a directory tree

SECURITY GUARANTEE
------------------
This module NEVER deserializes, executes, or loads model weights.
All file handling is read-only raw binary I/O.

Design Reference
----------------
Sections 9-10 of docs/implementation_design.md
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

from model_scanner.acquisition.base import InputAcquisitionProvider
from model_scanner.core.exceptions import AcquisitionError, InputError, ResourceLimitError
from model_scanner.core.models import (
    Artifact,
    InputSourceType,
    ScanConfig,
    ScanRequest,
    ScanWarning,
)
from model_scanner.utils.hashing import sha256_of_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognised model file extensions — used for directory candidate filtering.
# Content-first detection (magic bytes) takes precedence during analysis;
# this list only controls which files are picked up during directory traversal.
_MODEL_EXTENSIONS: frozenset[str] = frozenset(
    [
        ".pkl",
        ".pickle",
        ".joblib",
        ".dat",
        ".data",
        ".dill",
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".safetensors",
        ".gguf",
        ".ggml",
        # Identified-but-unanalyzed formats (A5): emit G002 finding
        ".h5",
        ".hdf5",
        ".keras",
        ".npy",
        ".npz",
        ".onnx",
        ".tflite",
        ".msgpack",
    ]
)

# Hidden directory names to skip during recursive traversal.
_HIDDEN_DIR_PREFIXES: frozenset[str] = frozenset(
    [
        ".git",
        ".svn",
        ".hg",
        ".venv",
        "venv",
        ".env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
    ]
)

# Maximum number of files to acquire from a single directory scan (DoS guard).
_MAX_DIR_FILES: int = 10_000


def _is_hidden_dir(name: str) -> bool:
    """Return True if a directory name should be skipped during traversal."""
    return name in _HIDDEN_DIR_PREFIXES or name.startswith(".")


def _make_artifact_id() -> str:
    """Generate a new UUIDv4 artifact ID."""
    return str(uuid.uuid4())


def _validate_include_exclude(
    path: Path,
    include_pattern: str | None,
    exclude_pattern: str | None,
) -> bool:
    """Return True if path passes the include/exclude regex filters.

    Parameters
    ----------
    path:
        File path to evaluate.
    include_pattern:
        If set, the file's name must match this regex to be included.
    exclude_pattern:
        If set, the file's name must NOT match this regex to be included.
    """
    name = path.name
    if include_pattern is not None and not re.search(include_pattern, name):
        return False
    if exclude_pattern is not None and re.search(exclude_pattern, name):
        return False
    return True


# ---------------------------------------------------------------------------
# LocalFileAcquisition
# ---------------------------------------------------------------------------


class LocalFileAcquisition(InputAcquisitionProvider):
    """Acquires a single local model file as a raw binary Artifact.

    Validates existence, file type, and readability before computing a
    streaming SHA-256 digest and returning an Artifact.

    Raises
    ------
    InputError
        If the path does not exist or is not a regular file.
    AcquisitionError
        If the file cannot be read (permission denied, I/O error).
    ResourceLimitError
        If the file size exceeds ``config.max_download_file_bytes``.
    """

    def acquire(self, request: ScanRequest) -> Sequence[Artifact]:
        """Acquire a single local file Artifact.

        Parameters
        ----------
        request:
            Must have ``input_source_type == LOCAL_FILE`` and ``target``
            as a ``str`` or ``Path``.

        Returns
        -------
        Sequence[Artifact]
            A single-element list containing the acquired Artifact.
        """
        target = request.target
        if not isinstance(target, (str, Path)):
            raise InputError(
                f"LocalFileAcquisition requires a str or Path target, "
                f"got {type(target).__name__}",
                details=repr(target),
            )

        path = Path(target).resolve()
        config = request.config or ScanConfig()

        # --- Existence and type validation ---
        if not path.exists():
            raise InputError(
                f"Target path does not exist: {path}",
                error_code="FILE_NOT_FOUND",
                details=str(path),
            )
        if not path.is_file():
            raise InputError(
                f"Target path is not a regular file: {path}",
                error_code="NOT_A_FILE",
                details=str(path),
            )

        # --- Readability check ---
        if not os.access(path, os.R_OK):
            raise AcquisitionError(
                f"File is not readable (permission denied): {path}",
                error_code="PERMISSION_DENIED",
                details=str(path),
            )

        # --- Size check ---
        stat = path.stat()
        size_bytes = stat.st_size
        if size_bytes > config.max_download_file_bytes:
            raise ResourceLimitError(
                f"File exceeds maximum allowed size: {path}",
                error_code="FILE_TOO_LARGE",
                details=f"size={size_bytes}, limit={config.max_download_file_bytes}",
                limit_name="max_download_file_bytes",
                limit_value=config.max_download_file_bytes,
                actual_value=size_bytes,
            )

        # --- SHA-256 fingerprinting ---
        try:
            sha256 = sha256_of_file(path)
        except OSError as exc:
            raise AcquisitionError(
                f"Failed to read file for hashing: {path}",
                error_code="READ_ERROR",
                details=str(exc),
            ) from exc

        artifact = Artifact(
            artifact_id=_make_artifact_id(),
            source_path=str(path),
            input_source=InputSourceType.LOCAL_FILE,
            size_bytes=size_bytes,
            sha256=sha256,
            local_path=path,
        )
        return [artifact]


# ---------------------------------------------------------------------------
# LocalDirectoryAcquisition
# ---------------------------------------------------------------------------


class LocalDirectoryAcquisition(InputAcquisitionProvider):
    """Recursively discovers and acquires candidate model files under a directory.

    Discovery rules (Section 10.2):
    - Walks directory recursively using ``os.walk``.
    - Skips hidden directory names (e.g., ``.git``, ``.venv``).
    - Filters candidates by recognised model file extensions.
    - Applies optional include/exclude regex patterns from ``ScanConfig``.
    - Returns results in deterministic alphabetical order by resolved path.

    Symlink safety (A4):
    - Resolves every discovered path with ``Path.resolve()``.
    - Out-of-root symlinks are **skipped** (never followed) and recorded in
      ``skipped_symlinks`` for the caller to surface as warnings.
    - Symlink loops (OSError on resolve) are also skipped and recorded.

    Raises
    ------
    InputError
        If the path does not exist or is not a directory.
    AcquisitionError
        If any individual file read fails.
    ResourceLimitError
        If a discovered file exceeds the per-file size limit, or if the
        directory contains more than ``_MAX_DIR_FILES`` candidates.
    """

    def __init__(self) -> None:
        # Populated during acquire(); callers may read after acquire() returns.
        self.skipped_symlinks: list[ScanWarning] = []

    def acquire(self, request: ScanRequest) -> Sequence[Artifact]:
        """Acquire all candidate model file Artifacts under a directory tree.

        Parameters
        ----------
        request:
            Must have ``input_source_type == LOCAL_DIRECTORY`` and ``target``
            as a ``str`` or ``Path``.

        Returns
        -------
        Sequence[Artifact]
            Sorted sequence of acquired Artifacts (alphabetical by resolved path).
        """
        target = request.target
        if not isinstance(target, (str, Path)):
            raise InputError(
                f"LocalDirectoryAcquisition requires a str or Path target, "
                f"got {type(target).__name__}",
                details=repr(target),
            )

        root = Path(target).resolve()
        config = request.config or ScanConfig()

        if not root.exists():
            raise InputError(
                f"Directory does not exist: {root}",
                error_code="DIR_NOT_FOUND",
                details=str(root),
            )
        if not root.is_dir():
            raise InputError(
                f"Target path is not a directory: {root}",
                error_code="NOT_A_DIRECTORY",
                details=str(root),
            )

        self.skipped_symlinks = []  # reset on each call
        candidates = self._discover(root, config)
        artifacts: list[Artifact] = []

        for candidate_path in sorted(candidates):
            try:
                stat = candidate_path.stat()
            except OSError as exc:
                raise AcquisitionError(
                    f"Cannot stat candidate file: {candidate_path}",
                    error_code="STAT_ERROR",
                    details=str(exc),
                ) from exc

            size_bytes = stat.st_size
            if size_bytes > config.max_download_file_bytes:
                raise ResourceLimitError(
                    f"File exceeds maximum allowed size: {candidate_path}",
                    error_code="FILE_TOO_LARGE",
                    details=f"size={size_bytes}, limit={config.max_download_file_bytes}",
                    limit_name="max_download_file_bytes",
                    limit_value=config.max_download_file_bytes,
                    actual_value=size_bytes,
                )

            try:
                sha256 = sha256_of_file(candidate_path)
            except OSError as exc:
                raise AcquisitionError(
                    f"Failed to read file for hashing: {candidate_path}",
                    error_code="READ_ERROR",
                    details=str(exc),
                ) from exc

            artifacts.append(
                Artifact(
                    artifact_id=_make_artifact_id(),
                    source_path=str(candidate_path),
                    input_source=InputSourceType.LOCAL_DIRECTORY,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    local_path=candidate_path,
                )
            )

        return artifacts

    def _discover(self, root: Path, config: ScanConfig) -> list[Path]:
        """Return all candidate file paths under root, excluding hidden dirs.

        Symlinks that resolve outside ``root`` are skipped and recorded in
        ``self.skipped_symlinks``.  Symlink loops (``OSError`` on resolve) are
        also skipped.  No path is ever followed silently.

        Parameters
        ----------
        root:
            Resolved directory to walk.
        config:
            ScanConfig carrying optional include/exclude patterns.

        Returns
        -------
        list[Path]
            Unsorted list of candidate file paths (all within ``root``).
        """
        candidates: list[Path] = []
        file_count = 0

        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            # Prune hidden directories in-place to prevent os.walk from descending
            dirnames[:] = sorted(
                d for d in dirnames if not _is_hidden_dir(d)
            )

            for filename in filenames:
                file_path = Path(dirpath) / filename
                suffix = file_path.suffix.lower()

                # Extension filter: only pick up recognised model extensions.
                # saved_model.pb is the TensorFlow SavedModel protobuf; it is
                # identified by filename (not extension) in FormatDispatcher, so
                # we admit it explicitly rather than adding .pb to the allowlist.
                if suffix not in _MODEL_EXTENSIONS and file_path.name != "saved_model.pb":
                    continue

                # A4: Symlink escape guard — resolve and check containment.
                # OSError catches symlink loops and unresolvable paths.
                try:
                    resolved = file_path.resolve()
                except OSError:
                    self.skipped_symlinks.append(
                        ScanWarning(
                            warning_code="SYMLINK_LOOP_SKIPPED",
                            message=f"Symlink loop or unresolvable path skipped: {file_path}",
                            context=str(file_path),
                        )
                    )
                    continue

                try:
                    resolved.relative_to(root)
                except ValueError:
                    # Path resolves outside the scan root — skip, never follow.
                    self.skipped_symlinks.append(
                        ScanWarning(
                            warning_code="SYMLINK_ESCAPE_BLOCKED",
                            message=(
                                f"Symlink escaping scan root blocked: "
                                f"{file_path} -> {resolved}"
                            ),
                            context=str(file_path),
                        )
                    )
                    continue

                # Optional include/exclude pattern filter
                if not _validate_include_exclude(
                    file_path,
                    config.include_pattern,
                    config.exclude_pattern,
                ):
                    continue

                candidates.append(file_path)
                file_count += 1

                if file_count > _MAX_DIR_FILES:
                    raise ResourceLimitError(
                        f"Directory contains more than {_MAX_DIR_FILES} candidate "
                        f"model files — possible DoS attempt",
                        error_code="DIR_TOO_MANY_FILES",
                        limit_name="max_dir_files",
                        limit_value=_MAX_DIR_FILES,
                        actual_value=file_count,
                    )

        return candidates


__all__ = [
    "LocalDirectoryAcquisition",
    "LocalFileAcquisition",
]
