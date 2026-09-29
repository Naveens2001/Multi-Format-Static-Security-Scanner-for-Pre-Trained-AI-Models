"""
model_scanner.analyzers.safetensors.header
==========================================

SafeTensors 8-byte little-endian header length unpacker and bounds verifier.

The SafeTensors format begins with an 8-byte unsigned 64-bit little-endian
integer declaring the byte-length of the JSON header that immediately follows.
This module reads that prefix and applies resource-safety guards before any
JSON allocation occurs.

SECURITY GUARANTEE
------------------
- Reads exactly 8 bytes before any further memory allocation.
- Enforces maximum header size before reading the JSON body.
- Never calls pickle.loads(), torch.load(), eval(), or exec().
- Does not import torch, safetensors, or any ML framework.

Design Reference
----------------
Section 16.1, 16.2, and 30.1 of docs/implementation_design.md
"""

from __future__ import annotations

import struct

from model_scanner.core.exceptions import ResourceLimitError, SafeTensorsAnalysisError

# Struct format for the 8-byte LE uint64 header-length prefix
_HEADER_LEN_FMT: str = "<Q"
_HEADER_PREFIX_SIZE: int = struct.calcsize(_HEADER_LEN_FMT)  # 8 bytes

# Default resource ceiling — overridden by ScanConfig.max_safetensors_header_mb
DEFAULT_MAX_HEADER_BYTES: int = 16 * 1024 * 1024  # 16 MiB


def unpack_header_length(data: bytes, max_header_bytes: int) -> int:
    """Unpack the 8-byte LE uint64 header-length prefix and validate bounds.

    Parameters
    ----------
    data:
        Raw file bytes (at minimum the first 8 bytes; may be longer).
    max_header_bytes:
        Maximum permitted header size in bytes.  Derived from
        ``ScanConfig.max_safetensors_header_mb * 1024 * 1024``.

    Returns
    -------
    int
        The declared header length (number of bytes the JSON occupies).

    Raises
    ------
    SafeTensorsAnalysisError
        If the file is too short to contain the 8-byte prefix, or the
        declared header length is zero.
    ResourceLimitError
        If the declared header length exceeds *max_header_bytes*.
    """
    if len(data) < _HEADER_PREFIX_SIZE:
        raise SafeTensorsAnalysisError(
            message=(
                f"SafeTensors file too short to contain 8-byte header-length prefix "
                f"(got {len(data)} bytes)"
            ),
            error_code="ST_HEADER_TOO_SHORT",
            details=f"file_size={len(data)}",
        )

    (header_len,) = struct.unpack_from(_HEADER_LEN_FMT, data, 0)

    if header_len == 0:
        raise SafeTensorsAnalysisError(
            message="SafeTensors header length is zero — invalid format",
            error_code="ST_HEADER_LEN_ZERO",
            details="header_len=0",
        )

    if header_len > max_header_bytes:
        raise ResourceLimitError(
            message=(
                f"SafeTensors header length {header_len:,} bytes exceeds the "
                f"configured limit of {max_header_bytes:,} bytes "
                f"({max_header_bytes // (1024 * 1024)} MiB)"
            ),
            error_code="ST_HEADER_TOO_LARGE",
            details=f"declared_len={header_len}, limit={max_header_bytes}",
            limit_name="max_safetensors_header_bytes",
            limit_value=max_header_bytes,
            actual_value=header_len,
        )

    return int(header_len)


def extract_header_bytes(data: bytes, header_len: int) -> bytes:
    """Extract the JSON header bytes following the 8-byte length prefix.

    Parameters
    ----------
    data:
        Full or partial file bytes.
    header_len:
        Previously validated header length from :func:`unpack_header_length`.

    Returns
    -------
    bytes
        The raw JSON bytes (exactly ``header_len`` bytes).

    Raises
    ------
    SafeTensorsAnalysisError
        If the file is truncated before the full header is available.
    """
    required = _HEADER_PREFIX_SIZE + header_len
    if len(data) < required:
        raise SafeTensorsAnalysisError(
            message=(
                f"SafeTensors file truncated: header declares {header_len:,} bytes "
                f"but only {len(data) - _HEADER_PREFIX_SIZE:,} are available"
            ),
            error_code="ST_HEADER_TRUNCATED",
            details=(
                f"declared_len={header_len}, "
                f"available={len(data) - _HEADER_PREFIX_SIZE}, "
                f"file_size={len(data)}"
            ),
        )
    return data[_HEADER_PREFIX_SIZE : _HEADER_PREFIX_SIZE + header_len]


__all__ = [
    "DEFAULT_MAX_HEADER_BYTES",
    "extract_header_bytes",
    "unpack_header_length",
]
