"""
model_scanner.analyzers.gguf.reader
====================================

Binary header unpacker and length-prefixed Key-Value parser for GGUF model files.

GGUF Binary Specification
-------------------------
Header structure:
  - 4 bytes magic: b"GGUF" (0x46554747 in LE)
  - uint32 version: 1, 2, or 3
  - uint64/uint32 tensor_count (uint32 in v1, uint64 in v2/v3)
  - uint64/uint32 metadata_kv_count (uint32 in v1, uint64 in v2/v3)

Key-Value structure:
  - key: length-prefixed UTF-8 string (uint32 len in v1, uint64 len in v2/v3)
  - value_type: uint32 enum (0..12)
  - value: type-dependent encoding

Supported GGUF Value Types:
  0: UINT8    (1 byte)
  1: INT8     (1 byte)
  2: UINT16   (2 bytes LE)
  3: INT16    (2 bytes LE)
  4: UINT32   (4 bytes LE)
  5: INT32    (4 bytes LE)
  6: FLOAT32  (4 bytes LE float)
  7: BOOL     (1 byte boolean)
  8: STRING   (length-prefixed UTF-8)
  9: ARRAY    (element_type: uint32, count: uint64/uint32, elements: ...)
 10: UINT64   (8 bytes LE)
 11: INT64    (8 bytes LE)
 12: FLOAT64  (8 bytes LE double)

SECURITY GUARANTEE
------------------
- Pure standard library struct/byte parsing only.
- Strict bounded reading: string lengths, array element counts, and KV counts
  are checked before memory allocations.
- Never loads tensor payloads into memory.
- Never calls eval(), exec(), pickle.loads(), or torch.load().

Design Reference
----------------
Section 17 of docs/implementation_design.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

from model_scanner.core.exceptions import (
    GGUFAnalysisError,
    ResourceLimitError,
)

# Magic bytes for GGUF
GGUF_MAGIC: bytes = b"GGUF"

# Maximum allowable nesting depth for array metadata values (DoS defense)
MAX_ARRAY_NESTING_DEPTH: int = 10

# Maximum string length for keys (64 KiB)
MAX_KEY_LENGTH: int = 65_536


class GGUFValueType(IntEnum):
    """Canonical GGUF metadata value types."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# Simple scalar format specs (LE) and byte sizes
_SCALAR_TYPE_FORMATS: dict[int, tuple[str, int]] = {
    GGUFValueType.UINT8: ("<B", 1),
    GGUFValueType.INT8: ("<b", 1),
    GGUFValueType.UINT16: ("<H", 2),
    GGUFValueType.INT16: ("<h", 2),
    GGUFValueType.UINT32: ("<I", 4),
    GGUFValueType.INT32: ("<i", 4),
    GGUFValueType.FLOAT32: ("<f", 4),
    GGUFValueType.BOOL: ("<B", 1),
    GGUFValueType.UINT64: ("<Q", 8),
    GGUFValueType.INT64: ("<q", 8),
    GGUFValueType.FLOAT64: ("<d", 8),
}


@dataclass(frozen=True)
class GGUFHeader:
    """Parsed GGUF header information."""

    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int
    header_offset: int  # Offset where KV pairs begin


@dataclass(frozen=True)
class GGUFKVEntry:
    """Parsed GGUF Key-Value metadata entry."""

    key: str
    key_offset: int
    value_type: int
    value: object
    value_offset: int
    raw_size_bytes: int


@dataclass(frozen=True)
class GGUFReaderIssue:
    """A structural issue discovered during header / KV parsing."""

    rule_id: str
    byte_offset: int
    matched_value: str
    detail: str


class GGUFReader:
    """Zero-deserialization binary reader for GGUF headers and metadata."""

    def __init__(
        self,
        data: bytes,
        max_kv_count: int = 10_000,
        max_value_bytes: int = 1_048_576,
    ) -> None:
        self.data = data
        self.total_size = len(data)
        self.max_kv_count = max_kv_count
        self.max_value_bytes = max_value_bytes
        self.issues: list[GGUFReaderIssue] = []

    def parse_header(self) -> tuple[GGUFHeader | None, list[GGUFKVEntry], int]:
        """Parse GGUF header prefix and all Key-Value metadata entries.

        Returns
        -------
        tuple[GGUFHeader | None, list[GGUFKVEntry], int]
            (header, kv_entries, offset_after_kv)
        """
        if self.total_size < 4:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU203",
                    byte_offset=0,
                    matched_value="TRUNCATED_HEADER",
                    detail=(
                        f"File too short for GGUF magic (got {self.total_size} "
                        "bytes, minimum 4 bytes)"
                    ),
                )
            )
            return None, [], 0

        magic = self.data[:4]
        if magic != GGUF_MAGIC:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU202",
                    byte_offset=0,
                    matched_value=magic.hex(),
                    detail=(
                        f"Invalid GGUF magic bytes: expected b'GGUF' "
                        f"({GGUF_MAGIC.hex()}), got {magic!r}"
                    ),
                )
            )
            return None, [], 4

        if self.total_size < 8:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU203",
                    byte_offset=4,
                    matched_value="TRUNCATED_VERSION",
                    detail=(
                        f"File truncated before GGUF version (got {self.total_size} "
                        "bytes, minimum 8 bytes)"
                    ),
                )
            )
            return None, [], 4

        (version,) = struct.unpack_from("<I", self.data, 4)
        if version not in (1, 2, 3):
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU202",
                    byte_offset=4,
                    matched_value=str(version),
                    detail=(
                        f"Unsupported GGUF version: {version} "
                        "(supported versions: 1, 2, 3)"
                    ),
                )
            )
            return None, [], 8

        # Unpack tensor_count and metadata_kv_count
        offset = 8
        if version == 1:
            if self.total_size < offset + 8:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU203",
                        byte_offset=offset,
                        matched_value="TRUNCATED_COUNTS",
                        detail="File truncated before GGUF v1 tensor/KV counts",
                    )
                )
                return None, [], offset
            tensor_count, kv_count = struct.unpack_from("<II", self.data, offset)
            offset += 8
        else:  # v2, v3
            if self.total_size < offset + 16:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU203",
                        byte_offset=offset,
                        matched_value="TRUNCATED_COUNTS",
                        detail=f"File truncated before GGUF v{version} tensor/KV counts",
                    )
                )
                return None, [], offset
            tensor_count, kv_count = struct.unpack_from("<QQ", self.data, offset)
            offset += 16

        # Resource limits on metadata KV count and tensor count
        if kv_count > self.max_kv_count:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU201",
                    byte_offset=offset - 8 if version != 1 else offset - 4,
                    matched_value=f"kv_count={kv_count}",
                    detail=(
                        f"GGUF metadata KV count ({kv_count}) exceeds configured "
                        f"maximum limit ({self.max_kv_count})"
                    ),
                )
            )
            raise ResourceLimitError(
                f"GGUF metadata KV count {kv_count} exceeds limit {self.max_kv_count}",
                error_code="GGUF_KV_OVERFLOW",
                limit_name="max_gguf_kv_count",
                limit_value=self.max_kv_count,
                actual_value=kv_count,
            )

        header = GGUFHeader(
            magic=magic,
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=kv_count,
            header_offset=offset,
        )

        kv_entries: list[GGUFKVEntry] = []
        for i in range(kv_count):
            if offset >= self.total_size:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU203",
                        byte_offset=offset,
                        matched_value=f"entry_index={i}",
                        detail=f"File truncated while reading KV metadata entry {i}/{kv_count}",
                    )
                )
                break

            entry, next_offset = self._read_kv_entry(offset, version)
            if entry is not None:
                kv_entries.append(entry)
            if next_offset <= offset:
                # Failed to make forward progress
                break
            offset = next_offset

        return header, kv_entries, offset

    def _read_string(self, offset: int, version: int) -> tuple[str | None, int]:
        """Read a length-prefixed UTF-8 string at offset.

        Returns (string_val, next_offset). If reading fails, returns (None, offset).
        """
        len_size = 4 if version == 1 else 8
        len_fmt = "<I" if version == 1 else "<Q"

        if offset + len_size > self.total_size:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    matched_value="TRUNCATED_STR_LEN",
                    detail="File truncated before string length prefix",
                )
            )
            return None, offset

        (str_len,) = struct.unpack_from(len_fmt, self.data, offset)
        offset += len_size

        if offset + str_len > self.total_size:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    matched_value=f"declared_len={str_len}",
                    detail=(
                        f"String length ({str_len} bytes) extends beyond end of file "
                        f"(available: {self.total_size - offset} bytes)"
                    ),
                )
            )
            return None, offset

        raw_str_bytes = self.data[offset : offset + str_len]
        try:
            val = raw_str_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    matched_value=raw_str_bytes[:32].hex(),
                    detail=f"Invalid UTF-8 string at offset {offset}: {exc}",
                )
            )
            val = raw_str_bytes.decode("utf-8", errors="replace")

        return val, offset + str_len

    def _read_kv_entry(self, offset: int, version: int) -> tuple[GGUFKVEntry | None, int]:
        """Read a single Key-Value metadata entry at offset."""
        key_start = offset
        key, offset = self._read_string(offset, version)
        if key is None:
            return None, offset

        if len(key) > MAX_KEY_LENGTH:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU104",
                    byte_offset=key_start,
                    matched_value=f"key_length={len(key)}",
                    detail=(
                        f"GGUF metadata key length ({len(key)} bytes) exceeds "
                        f"maximum key limit ({MAX_KEY_LENGTH})"
                    ),
                )
            )

        if offset + 4 > self.total_size:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    matched_value="TRUNCATED_VALUE_TYPE",
                    detail=f"File truncated before value type for key {key!r}",
                )
            )
            return None, offset

        (value_type,) = struct.unpack_from("<I", self.data, offset)
        offset += 4
        value_start = offset

        try:
            val, next_offset = self._read_value(
                offset, value_type, version, key=key, depth=0
            )
        except GGUFAnalysisError as exc:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU204",
                    byte_offset=offset,
                    matched_value=f"type={value_type}",
                    detail=f"Error reading value for key {key!r}: {exc}",
                )
            )
            return None, offset

        if next_offset <= offset and val is None:
            return None, offset

        entry = GGUFKVEntry(
            key=key,
            key_offset=key_start,
            value_type=value_type,
            value=val,
            value_offset=value_start,
            raw_size_bytes=next_offset - value_start,
        )
        return entry, next_offset

    def _read_value(
        self,
        offset: int,
        value_type: int,
        version: int,
        key: str = "",
        depth: int = 0,
    ) -> tuple[object, int]:
        """Read a GGUF value of the specified type."""
        if depth > MAX_ARRAY_NESTING_DEPTH:
            self.issues.append(
                GGUFReaderIssue(
                    rule_id="GU201",
                    byte_offset=offset,
                    matched_value=f"depth={depth}",
                    detail=f"Array nesting depth {depth} exceeds limit {MAX_ARRAY_NESTING_DEPTH}",
                )
            )
            return None, offset

        # Scalar types
        if value_type in _SCALAR_TYPE_FORMATS:
            fmt, size = _SCALAR_TYPE_FORMATS[value_type]
            if offset + size > self.total_size:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU203",
                        byte_offset=offset,
                        matched_value="TRUNCATED_SCALAR",
                        detail=(
                            f"File truncated before reading scalar type {value_type} "
                            f"(needs {size} bytes)"
                        ),
                    )
                )
                return None, offset
            (raw_val,) = struct.unpack_from(fmt, self.data, offset)
            if value_type == GGUFValueType.BOOL:
                return bool(raw_val), offset + size
            return raw_val, offset + size

        # String type
        if value_type == GGUFValueType.STRING:
            str_start = offset
            val, next_offset = self._read_string(offset, version)
            if val is not None:
                str_byte_len = next_offset - str_start - (4 if version == 1 else 8)
                if str_byte_len > self.max_value_bytes:
                    self.issues.append(
                        GGUFReaderIssue(
                            rule_id="GU104",
                            byte_offset=str_start,
                            matched_value=f"size={str_byte_len}",
                            detail=(
                                f"Metadata string value for key {key!r} ({str_byte_len} bytes) "
                                f"exceeds maximum limit ({self.max_value_bytes} bytes)"
                            ),
                        )
                    )
            return val, next_offset

        # Array type
        if value_type == GGUFValueType.ARRAY:
            if offset + 4 > self.total_size:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU203",
                        byte_offset=offset,
                        matched_value="TRUNCATED_ARRAY_ELEM_TYPE",
                        detail=f"File truncated before array element type for key {key!r}",
                    )
                )
                return None, offset

            (elem_type,) = struct.unpack_from("<I", self.data, offset)
            offset += 4

            len_size = 4 if version == 1 else 8
            len_fmt = "<I" if version == 1 else "<Q"
            if offset + len_size > self.total_size:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU203",
                        byte_offset=offset,
                        matched_value="TRUNCATED_ARRAY_COUNT",
                        detail=f"File truncated before array element count for key {key!r}",
                    )
                )
                return None, offset

            (elem_count,) = struct.unpack_from(len_fmt, self.data, offset)
            offset += len_size

            # Basic sanity check on element count
            if elem_count > 10_000_000:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU201",
                        byte_offset=offset - len_size,
                        matched_value=f"array_count={elem_count}",
                        detail=f"Array element count ({elem_count}) is impossibly large",
                    )
                )
                return None, offset

            # Check if elem_type is valid
            if elem_type > 12:
                self.issues.append(
                    GGUFReaderIssue(
                        rule_id="GU204",
                        byte_offset=offset - len_size - 4,
                        matched_value=f"elem_type={elem_type}",
                        detail=f"Invalid GGUF array element type: {elem_type} (must be 0..12)",
                    )
                )
                return None, offset

            # If scalar, we can check byte bounds quickly
            if elem_type in _SCALAR_TYPE_FORMATS:
                _, elem_size = _SCALAR_TYPE_FORMATS[elem_type]
                total_array_bytes = elem_count * elem_size
                if offset + total_array_bytes > self.total_size:
                    self.issues.append(
                        GGUFReaderIssue(
                            rule_id="GU203",
                            byte_offset=offset,
                            matched_value=f"needed={total_array_bytes}",
                            detail=(
                                f"Array of {elem_count} elements of type {elem_type} "
                                f"requires {total_array_bytes} bytes, but only "
                                f"{self.total_size - offset} bytes remain"
                            ),
                        )
                    )
                    return None, offset

                if total_array_bytes > self.max_value_bytes:
                    self.issues.append(
                        GGUFReaderIssue(
                            rule_id="GU104",
                            byte_offset=offset,
                            matched_value=f"array_bytes={total_array_bytes}",
                            detail=(
                                f"Array value for key {key!r} ({total_array_bytes} bytes) "
                                f"exceeds maximum limit ({self.max_value_bytes} bytes)"
                            ),
                        )
                    )

                # Parse scalar elements (up to bounded limit to prevent memory bloat)
                elements: list[object] = []
                fmt, size = _SCALAR_TYPE_FORMATS[elem_type]
                for _ in range(elem_count):
                    (v,) = struct.unpack_from(fmt, self.data, offset)
                    elements.append(bool(v) if elem_type == GGUFValueType.BOOL else v)
                    offset += size
                return elements, offset

            # If array of strings or nested arrays
            elements_list: list[object] = []
            for i in range(elem_count):
                if offset >= self.total_size:
                    self.issues.append(
                        GGUFReaderIssue(
                            rule_id="GU203",
                            byte_offset=offset,
                            matched_value=f"elem_index={i}",
                            detail=f"File truncated while reading array element {i}/{elem_count}",
                        )
                    )
                    break
                elem_val, next_offset = self._read_value(
                    offset, elem_type, version, key=f"{key}[{i}]", depth=depth + 1
                )
                elements_list.append(elem_val)
                if next_offset <= offset:
                    break
                offset = next_offset

            return elements_list, offset

        # Unknown / Unsupported Value Type
        self.issues.append(
            GGUFReaderIssue(
                rule_id="GU204",
                byte_offset=offset,
                matched_value=str(value_type),
                detail=(
                    f"Unsupported or invalid GGUF metadata value type: {value_type} "
                    "(must be 0..12)"
                ),
            )
        )
        return None, offset


__all__ = [
    "GGUF_MAGIC",
    "MAX_ARRAY_NESTING_DEPTH",
    "MAX_KEY_LENGTH",
    "GGUFHeader",
    "GGUFKVEntry",
    "GGUFReader",
    "GGUFReaderIssue",
    "GGUFValueType",
]
