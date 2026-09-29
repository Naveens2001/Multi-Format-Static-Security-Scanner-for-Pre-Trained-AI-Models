"""
model_scanner.detection.magic
==============================

Magic byte signatures and header prefix sniffing for AI model formats.

SECURITY GUARANTEE
------------------
This module inspects only raw byte sequences (up to a small bounded prefix).
It NEVER imports, executes, deserializes, or loads model weights.

Design Reference
----------------
Section 13 of docs/implementation_design.md
"""

from __future__ import annotations

import re
import struct

from model_scanner.core.models import ModelFormat

# ---------------------------------------------------------------------------
# Additional magic bytes for identified-but-unanalyzed formats (A5)
# ---------------------------------------------------------------------------

# HDF5 magic: b'\x89HDF\r\n\x1a\n' — covers Keras .h5 and .hdf5 files.
# Specification: https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html
_HDF5_MAGIC: bytes = b"\x89HDF\r\n\x1a\n"

# NumPy .npy magic: b'\x93NUMPY'
# Specification: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
_NUMPY_NPY_MAGIC: bytes = b"\x93NUMPY"

# TFLite FlatBuffer: first 4 bytes are file size (LE uint32), next 4 are the
# FlatBuffer identifier "TFL3" (bytes 4-7).
# Specification: TensorFlow Lite flatbuffer schema (file_identifier = "TFL3")
_TFLITE_IDENTIFIER: bytes = b"TFL3"
_TFLITE_IDENTIFIER_OFFSET: int = 4  # file_identifier starts at byte 4

# ---------------------------------------------------------------------------
# Magic Byte Signatures
# ---------------------------------------------------------------------------

# PKZip magic bytes (Local file header: PK\x03\x04)
_ZIP_LOCAL_HEADER: bytes = b"PK\x03\x04"
_ZIP_EMPTY_ARCHIVE: bytes = b"PK\x05\x06"
_ZIP_SPANNED: bytes = b"PK\x07\x08"

# GGUF magic bytes: 'GGUF' (0x47, 0x47, 0x55, 0x46)
_GGUF_MAGIC: bytes = b"GGUF"

# GGML magic family
_GGML_MAGICS: frozenset[bytes] = frozenset(
    [
        b"ggml",
        b"ggmf",
        b"ggjt",
        b"ggla",
        b"lmgg",
        b"fmg",
        b"tjgg",
    ]
)

# Pickle Protocol 2-5 opcodes at byte 0 (0x80) followed by protocol version (0x02 - 0x05)
# or explicitly framed protocols (\x80\x00, \x80\x01)
_PICKLE_PROTO_PREFIX: bytes = b"\x80"
_PICKLE_BINARY_PROTO_VERSIONS: frozenset[int] = frozenset([0, 1, 2, 3, 4, 5])

# Regex for strong multi-byte structural evidence of Protocol 0 / 1 GLOBAL opcode
# e.g., cos\nsystem\n or c__builtin__\neval\n or ctorch._utils\n_rebuild_tensor\n
_PICKLE_PROTO_0_GLOBAL_RE: re.Pattern[bytes] = re.compile(
    rb"^c[a-zA-Z0-9_\.]+\n[a-zA-Z0-9_\.]+\n"
)

# Structural pattern for Protocol 0 marker / memo dictionary / tuple initiation
# e.g., (dp0\n or (lp0\n or (tp0\n or (dp1\n
_PICKLE_PROTO_0_STRUCTURE_RE: re.Pattern[bytes] = re.compile(
    rb"^\([dlt](?:p\d+\n|q\d+)?(?:[cISVNFdp0-9]|\n)"
)

# Default upper bound for SafeTensors header size
DEFAULT_MAX_SAFETENSORS_HEADER_BYTES: int = 16 * 1024 * 1024  # 16 MB


def is_pytorch_zip_magic(header: bytes) -> bool:
    """Return True if header starts with standard ZIP magic bytes."""
    return (
        header.startswith(_ZIP_LOCAL_HEADER)
        or header.startswith(_ZIP_EMPTY_ARCHIVE)
        or header.startswith(_ZIP_SPANNED)
    )


def is_gguf_magic(header: bytes) -> bool:
    """Return True if header starts with GGUF magic bytes (b'GGUF')."""
    return header.startswith(_GGUF_MAGIC)


def is_ggml_magic(header: bytes) -> bool:
    """Return True if header starts with any recognised GGML magic signature."""
    for magic in _GGML_MAGICS:
        if header.startswith(magic):
            return True
    return False


def is_pickle_magic(header: bytes) -> bool:
    """Return True ONLY if header starts with unambiguous binary Pickle magic.

    Protocols 2-5: \\x80 followed by protocol version byte (\\x02, \\x03, \\x04, \\x05)
    or framed protocols (\\x80\\x00, \\x80\\x01).

    Single-byte ASCII opcodes (c, (, I, S, }, etc.) are NOT treated as binary magic.
    """
    if len(header) < 2:
        return False

    if header.startswith(_PICKLE_PROTO_PREFIX):
        return header[1] in _PICKLE_BINARY_PROTO_VERSIONS

    return False


def check_pickle_protocol_0_heuristic(header: bytes) -> tuple[bool, str]:
    """Check for strong multi-byte structural evidence of ASCII Pickle Protocol 0 / 1.

    Does NOT rely on a single ASCII character. Requires multi-byte structural
    confirmation such as:
      - 'c<module>\\n<name>\\n' (GLOBAL opcode structure)
      - '(d...\\n' or '(l...\\n' (MARK + structured container opcodes with newlines/memos)

    Returns
    -------
    tuple[bool, str]
        - (True, explanation) if strong structural evidence is found.
        - (False, "") otherwise.
    """
    if len(header) < 4:
        return False, ""

    if _PICKLE_PROTO_0_GLOBAL_RE.match(header):
        matched = header.split(b"\n")[:2]
        mod_name = b"\n".join(matched).decode("ascii", errors="replace")
        return (
            True,
            f"Strong heuristic evidence for Pickle protocol 0 GLOBAL opcode ({mod_name!r})",
        )

    if _PICKLE_PROTO_0_STRUCTURE_RE.match(header):
        return (
            True,
            "Strong heuristic evidence for Pickle protocol 0 container initiation structure",
        )

    return False, ""


def is_safetensors_header(
    header: bytes,
    total_size: int | None = None,
    max_header_bytes: int = DEFAULT_MAX_SAFETENSORS_HEADER_BYTES,
) -> bool:
    """Return True if header matches the SafeTensors binary specification.

    SafeTensors specification:
    - First 8 bytes: 64-bit unsigned integer in little-endian format (N = header size).
    - 0 < N <= max_header_bytes.
    - If total_size is known, N + 8 <= total_size.
    - Byte at offset 8 (first byte of JSON header) must be '{' (0x7B) or ASCII whitespace.
    """
    if len(header) < 9:
        return False

    try:
        (header_len,) = struct.unpack("<Q", header[:8])
    except struct.error:
        return False

    if header_len <= 0 or header_len > max_header_bytes:
        return False

    if total_size is not None and (header_len + 8) > total_size:
        return False

    # Check the first byte of JSON metadata (at offset 8)
    first_json_byte = header[8]
    # '{' is 0x7B; also permit standard ASCII whitespace before '{'
    valid_prefix_bytes = (ord("{"), ord(" "), ord("\t"), ord("\n"), ord("\r"))
    return first_json_byte in valid_prefix_bytes


def is_hdf5_magic(header: bytes) -> bool:
    """Return True if header starts with HDF5 magic bytes (covers Keras .h5/.keras)."""
    return header.startswith(_HDF5_MAGIC)


def is_numpy_npy_magic(header: bytes) -> bool:
    """Return True if header starts with NumPy .npy magic bytes (b'\\x93NUMPY')."""
    return header.startswith(_NUMPY_NPY_MAGIC)


def is_tflite_magic(header: bytes) -> bool:
    """Return True if header contains TFLite FlatBuffer identifier 'TFL3' at offset 4."""
    return (
        len(header) >= 8
        and header[_TFLITE_IDENTIFIER_OFFSET : _TFLITE_IDENTIFIER_OFFSET + 4]
        == _TFLITE_IDENTIFIER
    )


def _read_pb_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read a protobuf LEB128 varint from data starting at offset.

    Returns (value, new_offset).  Raises ValueError if the varint is
    truncated or exceeds 10 continuation bytes (protobuf spec maximum).
    Never allocates memory proportional to the decoded value.
    """
    value = 0
    shift = 0
    for _ in range(10):  # protobuf spec: max 10 bytes for a 64-bit varint
        if offset >= len(data):
            raise ValueError("Truncated varint: unexpected end of data")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):  # MSB clear → last byte
            return value, offset
    raise ValueError("Varint too long: exceeds 10 continuation bytes")


def is_mindspore_ckpt_header(header: bytes) -> bool:
    """Return True if header matches a MindSpore protobuf checkpoint structure.

    MindSpore .ckpt files are serialised as ``mindspore_pb2.Checkpoint``
    protobufs.  The wire structure is:

        Checkpoint (field 1, LEN):
          Parameter (field 1, LEN):
            name   (field 1, LEN): <tensor name string>
            tensor (field 2, LEN):
              dims  (field 1, VARINT): <dimension value>
              ...

    We require all of the following to be visible in the supplied header
    slice to avoid false-positives on unrelated files that happen to start
    with 0x0a:

    1. Byte 0 == 0x0a  (field 1, wire type 2 = LEN — Checkpoint.parameter)
    2. Valid varint follows (parameter message length)
    3. Next byte == 0x0a  (field 1 of Parameter = name, wire type 2)
    4. Valid varint follows (name length, 1–512 bytes)
    5. After the name, byte == 0x12  (field 2 of Parameter = tensor, wire type 2)

    This five-step check rules out all formats whose magic bytes are already
    tested earlier in sniff_magic() and is specific enough that accidental
    matches from unrelated protobufs are vanishingly unlikely within .ckpt
    files.
    """
    if len(header) < 6:
        return False
    # Requirement 1: Checkpoint.parameter field tag
    if header[0] != 0x0A:
        return False
    try:
        param_len, pos = _read_pb_varint(header, 1)
    except ValueError:
        return False
    # Sanity: parameter message should be at least 4 bytes (name tag + len + 1 char + tensor tag)
    if param_len < 4:
        return False
    # Requirement 3: Parameter.name field tag
    if pos >= len(header) or header[pos] != 0x0A:
        return False
    try:
        name_len, pos = _read_pb_varint(header, pos + 1)
    except ValueError:
        return False
    # Requirement 4: name length must be plausible (1–512 bytes)
    if name_len == 0 or name_len > 512:
        return False
    # Skip name bytes
    pos += name_len
    # Requirement 5: Parameter.tensor field tag (field 2, wire type 2 = 0x12)
    if pos >= len(header):
        # Name extended past our header window — structure is consistent so far;
        # treat as a probable match to avoid missing large-named tensors.
        return True
    return header[pos] == 0x12


def sniff_magic(
    header: bytes,
    total_size: int | None = None,
    max_safetensors_header_bytes: int = DEFAULT_MAX_SAFETENSORS_HEADER_BYTES,
) -> tuple[ModelFormat, bool, str]:
    """Sniff the leading header bytes to identify model format via magic/structure.

    Parameters
    ----------
    header:
        At least 16 (preferably 32+) leading bytes of the file.
    total_size:
        Total size of the artifact in bytes if known.
    max_safetensors_header_bytes:
        Maximum allowed SafeTensors header size.

    Returns
    -------
    tuple[ModelFormat, bool, str]
        - detected_format: ModelFormat (or ModelFormat.UNSUPPORTED)
        - magic_matched: bool (True if matched by content magic/structure)
        - details: str explaining the match rationale
    """
    if not header:
        return ModelFormat.UNSUPPORTED, False, "Empty header (0 bytes)"

    # 1. Check PyTorch ZIP (PKZip)
    if is_pytorch_zip_magic(header):
        return (
            ModelFormat.PYTORCH_ZIP,
            True,
            "Matched ZIP container magic (PK\\x03\\x04)",
        )

    # 2. Check GGUF
    if is_gguf_magic(header):
        return (
            ModelFormat.GGUF,
            True,
            "Matched GGUF binary magic header ('GGUF')",
        )

    # 3. Check GGML
    if is_ggml_magic(header):
        matched_magic = header[:4].decode("latin-1", errors="replace")
        return (
            ModelFormat.GGML,
            True,
            f"Matched GGML binary magic header ({matched_magic!r})",
        )

    # 4. Check SafeTensors (8-byte LE header length + JSON '{')
    if is_safetensors_header(
        header,
        total_size=total_size,
        max_header_bytes=max_safetensors_header_bytes,
    ):
        (header_len,) = struct.unpack("<Q", header[:8])
        return (
            ModelFormat.SAFETENSORS,
            True,
            f"Matched SafeTensors header (8-byte LE len={header_len}, JSON prefix '{{')",
        )

    # 5. Check deterministic binary Pickle magic (\x80\x00 - \x80\x05)
    if is_pickle_magic(header):
        proto = header[1]
        return (
            ModelFormat.RAW_PICKLE,
            True,
            f"Matched Pickle binary protocol {proto} opcode prefix (\\x80\\x{proto:02x})",
        )

    # 6. HDF5 (Keras .h5, .hdf5, .keras)
    if is_hdf5_magic(header):
        return (
            ModelFormat.HDF5,
            True,
            "Matched HDF5 magic header (\\x89HDF\\r\\n\\x1a\\n) — Keras/HDF5 format identified",
        )

    # 7. NumPy .npy
    if is_numpy_npy_magic(header):
        return (
            ModelFormat.NUMPY_NPY,
            True,
            "Matched NumPy .npy magic header (\\x93NUMPY) — NumPy array format identified",
        )

    # 8. TFLite FlatBuffer
    if is_tflite_magic(header):
        return (
            ModelFormat.TFLITE,
            True,
            "Matched TFLite FlatBuffer identifier 'TFL3' at offset 4",
        )

    # 9. MindSpore protobuf checkpoint
    if is_mindspore_ckpt_header(header):
        return (
            ModelFormat.MINDSPORE_CKPT,
            True,
            "Matched MindSpore protobuf checkpoint structure (Checkpoint.parameter field sequence)",
        )

    return (
        ModelFormat.UNSUPPORTED,
        False,
        "No recognized binary magic signature or header structure found",
    )


__all__ = [
    "DEFAULT_MAX_SAFETENSORS_HEADER_BYTES",
    "check_pickle_protocol_0_heuristic",
    "is_ggml_magic",
    "is_gguf_magic",
    "is_hdf5_magic",
    "is_mindspore_ckpt_header",
    "is_numpy_npy_magic",
    "is_pickle_magic",
    "is_pytorch_zip_magic",
    "is_safetensors_header",
    "is_tflite_magic",
    "sniff_magic",
]
