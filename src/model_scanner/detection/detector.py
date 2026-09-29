"""
model_scanner.detection.detector
=================================

Two-layer format identification and routing engine (FormatDispatcher).

Layer 1: Content-first magic byte and header structure inspection.
Layer 2: Secondary file extension fallback and format mismatch detection.

SECURITY GUARANTEE
------------------
This module reads only a small bounded header prefix (up to 64 bytes).
It NEVER imports, executes, deserializes, or loads model weights.

Design Reference
----------------
Section 13 of docs/implementation_design.md
"""

from __future__ import annotations

import io
from pathlib import Path

from model_scanner.core.models import Artifact, FormatDetectionResult, ModelFormat, ScanConfig
from model_scanner.detection.magic import (
    DEFAULT_MAX_SAFETENSORS_HEADER_BYTES,
    check_pickle_protocol_0_heuristic,
    sniff_magic,
)

# ---------------------------------------------------------------------------
# Extension Mapping Catalog
# ---------------------------------------------------------------------------

_EXTENSION_FORMAT_MAP: dict[str, ModelFormat] = {
    ".pkl": ModelFormat.RAW_PICKLE,
    ".pickle": ModelFormat.RAW_PICKLE,
    ".joblib": ModelFormat.RAW_PICKLE,
    ".dill": ModelFormat.RAW_PICKLE,
    ".dat": ModelFormat.RAW_PICKLE,
    ".data": ModelFormat.RAW_PICKLE,
    ".pt": ModelFormat.PYTORCH_ZIP,
    ".pth": ModelFormat.PYTORCH_ZIP,
    ".bin": ModelFormat.PYTORCH_ZIP,
    ".ckpt": ModelFormat.PYTORCH_ZIP,
    ".safetensors": ModelFormat.SAFETENSORS,
    ".gguf": ModelFormat.GGUF,
    ".ggml": ModelFormat.GGML,
    # Analyzed formats (P2)
    ".h5": ModelFormat.HDF5,
    ".hdf5": ModelFormat.HDF5,
    ".keras": ModelFormat.KERAS,
    ".npy": ModelFormat.NUMPY_NPY,
    ".npz": ModelFormat.NUMPY_NPZ,
    ".onnx": ModelFormat.ONNX,
    ".tflite": ModelFormat.TFLITE,
    ".msgpack": ModelFormat.FLAX_MSGPACK,
}

_LEGACY_PYTORCH_EXTENSIONS: frozenset[str] = frozenset(
    [".pt", ".pth", ".bin", ".ckpt"]
)

# Formats that are structurally compatible under common extensions:
# MINDSPORE_CKPT shares the .ckpt extension with PYTORCH_ZIP and LEGACY_PYTORCH.
# Detecting a MindSpore protobuf in a .ckpt file is expected, not extension-spoofing.
_COMPATIBLE_FORMAT_SETS: list[frozenset[ModelFormat]] = [
    frozenset({ModelFormat.MINDSPORE_CKPT, ModelFormat.PYTORCH_ZIP, ModelFormat.LEGACY_PYTORCH}),
]

_HEADER_READ_BYTES: int = 64  # Bounded prefix read for format sniffing


def _extension_for_name(name_or_path: str) -> str:
    """Extract lowercase file extension from filename, path, or URL string."""
    if not name_or_path:
        return ""
    # Strip URL queries and fragments
    path_part = name_or_path.split("?", 1)[0].split("#", 1)[0]
    name = path_part.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in name:
        return "." + name.rsplit(".", 1)[-1].lower()
    return ""


def _is_format_compatible(fmt1: ModelFormat, fmt2: ModelFormat) -> bool:
    """Return True if fmt1 and fmt2 are considered compatible variants."""
    if fmt1 == fmt2:
        return True
    for comp_set in _COMPATIBLE_FORMAT_SETS:
        if fmt1 in comp_set and fmt2 in comp_set:
            return True
    return False


def _read_artifact_header(artifact: Artifact, num_bytes: int = _HEADER_READ_BYTES) -> bytes:
    """Read a bounded header prefix from an Artifact without loading the full file."""
    if artifact.local_path is not None:
        try:
            with open(artifact.local_path, "rb") as fh:
                return fh.read(num_bytes)
        except OSError:
            return b""
    elif artifact.stream is not None:
        stream = artifact.stream
        try:
            if stream.seekable():
                pos = stream.tell()
                header = stream.read(num_bytes)
                stream.seek(pos)
                return header
            else:
                return stream.read(num_bytes)
        except (OSError, io.UnsupportedOperation):
            return b""
    return b""


class FormatDispatcher:
    """Two-layer format detector and analyzer router.

    Workflow:
    1. Reads leading bounded header bytes (<= 64 bytes).
    2. Performs magic byte and structural header sniffing.
    3. Examines secondary filename extension evidence.
    4. Evaluates format mismatch (e.g. extension indicates SafeTensors,
       but content is actually Pickle or PyTorch ZIP).
    5. Returns a strongly-typed FormatDetectionResult.
    """

    def __init__(self, config: ScanConfig | None = None) -> None:
        self.config = config or ScanConfig()

    def detect_format(self, artifact: Artifact) -> FormatDetectionResult:
        """Detect the model format of an acquired Artifact.

        Parameters
        ----------
        artifact:
            The acquired model artifact.

        Returns
        -------
        FormatDetectionResult
            Strongly-typed detection result detailing detected format,
            match layer, and whether a format mismatch was detected.
        """
        header = _read_artifact_header(artifact, _HEADER_READ_BYTES)

        # Inspect both source_path and local_path for authoritative extension evidence
        source_ext = _extension_for_name(artifact.source_path)
        local_ext = (
            _extension_for_name(str(artifact.local_path))
            if artifact.local_path
            else ""
        )

        if source_ext in _EXTENSION_FORMAT_MAP:
            filename_hint = artifact.source_path
        elif local_ext in _EXTENSION_FORMAT_MAP:
            filename_hint = str(artifact.local_path)
        else:
            filename_hint = artifact.source_path or str(artifact.local_path or "")

        max_st_bytes = (
            self.config.max_safetensors_header_mb * 1024 * 1024
            if self.config
            else DEFAULT_MAX_SAFETENSORS_HEADER_BYTES
        )

        return self.detect_from_bytes(
            header=header,
            filename_hint=filename_hint,
            total_size=artifact.size_bytes,
            max_safetensors_header_bytes=max_st_bytes,
        )

    def detect_from_path(self, path: str | Path) -> FormatDetectionResult:
        """Detect format directly from a local file path.

        Parameters
        ----------
        path:
            Local file path to inspect.

        Returns
        -------
        FormatDetectionResult
        """
        p = Path(path)
        try:
            with open(p, "rb") as fh:
                header = fh.read(_HEADER_READ_BYTES)
            total_size = p.stat().st_size
        except OSError:
            header = b""
            total_size = None

        max_st_bytes = (
            self.config.max_safetensors_header_mb * 1024 * 1024
            if self.config
            else DEFAULT_MAX_SAFETENSORS_HEADER_BYTES
        )

        return self.detect_from_bytes(
            header=header,
            filename_hint=str(p),
            total_size=total_size,
            max_safetensors_header_bytes=max_st_bytes,
        )

    def detect_from_bytes(
        self,
        header: bytes,
        filename_hint: str | None = None,
        total_size: int | None = None,
        max_safetensors_header_bytes: int = DEFAULT_MAX_SAFETENSORS_HEADER_BYTES,
    ) -> FormatDetectionResult:
        """Detect format from raw header bytes with optional filename hint.

        Parameters
        ----------
        header:
            Leading header bytes (up to 64 bytes).
        filename_hint:
            Optional filename or path string for extension inference.
        total_size:
            Total size of the artifact in bytes if known.
        max_safetensors_header_bytes:
            Maximum allowed SafeTensors header size.

        Returns
        -------
        FormatDetectionResult
        """
        # Special case: saved_model.pb filename → TF_SAVEDMODEL
        if filename_hint and (
            filename_hint.endswith("saved_model.pb") or
            filename_hint.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] == "saved_model.pb"
        ):
            return FormatDetectionResult(
                detected_format=ModelFormat.TF_SAVEDMODEL,
                magic_matched=False,
                extension_matched=True,
                format_mismatch=False,
                details="TensorFlow SavedModel protobuf identified by filename 'saved_model.pb'",
            )

        # Step 1: Content-first magic byte sniffing
        magic_format, magic_matched, magic_details = sniff_magic(
            header=header,
            total_size=total_size,
            max_safetensors_header_bytes=max_safetensors_header_bytes,
        )

        # Step 2: Extension lookup
        ext = _extension_for_name(filename_hint) if filename_hint else ""
        ext_format = _EXTENSION_FORMAT_MAP.get(ext)
        extension_matched = ext_format is not None

        # ZIP magic but .keras extension → KERAS format
        if magic_matched and magic_format == ModelFormat.PYTORCH_ZIP and ext == ".keras":
            return FormatDetectionResult(
                detected_format=ModelFormat.KERAS,
                magic_matched=True,
                extension_matched=True,
                format_mismatch=False,
                details="Keras .keras format (ZIP-based) detected by magic bytes and extension",
            )

        # Step 2.5: Check for legacy non-ZIP PyTorch serialization
        # Files with PyTorch extensions (.pt, .pth, .bin, .ckpt) starting with Pickle
        # opcodes are legacy non-ZIP PyTorch files (pickle metadata + raw tensor bytes),
        # which cannot be disassembled as pure standalone pickles.
        if ext in _LEGACY_PYTORCH_EXTENSIONS:
            _legacy_details = (
                "Legacy non-ZIP PyTorch serialization (pickle metadata followed by raw "
                "tensor payload). The leading pickle stream is disassembled statically; "
                "the trailing tensor payload is not parsed."
            )
            if magic_matched and magic_format == ModelFormat.RAW_PICKLE:
                return FormatDetectionResult(
                    detected_format=ModelFormat.LEGACY_PYTORCH,
                    magic_matched=True,
                    extension_matched=True,
                    format_mismatch=False,
                    details=_legacy_details,
                )
            is_p0_heuristic, _ = check_pickle_protocol_0_heuristic(header)
            if is_p0_heuristic:
                return FormatDetectionResult(
                    detected_format=ModelFormat.LEGACY_PYTORCH,
                    magic_matched=False,
                    extension_matched=True,
                    format_mismatch=False,
                    details=_legacy_details,
                )

        # Step 3: Format resolution and mismatch evaluation
        if magic_matched:
            detected_format = magic_format
            format_mismatch = False

            if extension_matched and ext_format is not None:
                # Check for format mismatch / spoofing
                if not _is_format_compatible(detected_format, ext_format):
                    format_mismatch = True
                    details = (
                        f"Format mismatch detected: Header magic indicates "
                        f"{detected_format.value}, but file extension {ext!r} "
                        f"indicates {ext_format.value}. {magic_details}"
                    )
                else:
                    details = f"{magic_details}; extension {ext!r} is consistent."
            else:
                details = magic_details

            return FormatDetectionResult(
                detected_format=detected_format,
                magic_matched=True,
                extension_matched=extension_matched,
                format_mismatch=format_mismatch,
                details=details,
            )

        # Step 4: Strong heuristic evidence (e.g. multi-byte Protocol 0 Pickle structure)
        is_p0_heuristic, p0_details = check_pickle_protocol_0_heuristic(header)
        if is_p0_heuristic:
            detected_format = ModelFormat.RAW_PICKLE
            format_mismatch = False

            if extension_matched and ext_format is not None:
                if not _is_format_compatible(detected_format, ext_format):
                    format_mismatch = True
                    details = (
                        f"Format mismatch detected: Strong heuristic indicates "
                        f"RAW_PICKLE, but file extension {ext!r} indicates "
                        f"{ext_format.value}. {p0_details}"
                    )
                else:
                    details = f"{p0_details}; extension {ext!r} is consistent."
            else:
                details = p0_details

            return FormatDetectionResult(
                detected_format=detected_format,
                magic_matched=False,
                extension_matched=extension_matched,
                format_mismatch=format_mismatch,
                details=details,
            )

        # Step 5: Extension fallback if magic and heuristics are inconclusive
        if extension_matched and ext_format is not None:
            # .bin with no ZIP or pickle magic is a raw binary weight buffer
            # (e.g. TF.js shards).  Routing it to PYTORCH_ZIP would send it to
            # PyTorchAnalyzer, which fails ZIP validation → P201 HIGH false-positive.
            if ext == ".bin" and ext_format == ModelFormat.PYTORCH_ZIP:
                return FormatDetectionResult(
                    detected_format=ModelFormat.TFJS_WEIGHT_SHARD,
                    magic_matched=False,
                    extension_matched=True,
                    format_mismatch=False,
                    details=(
                        "No ZIP or pickle magic in .bin file; classified as TF.js "
                        "weight shard (raw float32 buffer). No analyzer is registered "
                        "for this format — coverage is incomplete."
                    ),
                )
            return FormatDetectionResult(
                detected_format=ext_format,
                magic_matched=False,
                extension_matched=True,
                format_mismatch=False,
                details=(
                    f"Magic sniffing and heuristic detection inconclusive; "
                    f"fell back to file extension {ext!r} -> {ext_format.value}"
                ),
            )

        # Step 6: Completely unsupported / unknown
        return FormatDetectionResult(
            detected_format=ModelFormat.UNSUPPORTED,
            magic_matched=False,
            extension_matched=False,
            format_mismatch=False,
            details=f"Unrecognized format: {magic_details}; unknown extension {ext!r}",
        )


__all__ = [
    "FormatDispatcher",
]
