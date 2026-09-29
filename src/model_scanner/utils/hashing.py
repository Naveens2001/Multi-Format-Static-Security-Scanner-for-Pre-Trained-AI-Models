"""
model_scanner.utils.hashing
============================

Streaming SHA-256 fingerprinting utilities.

SECURITY GUARANTEE
------------------
This module computes cryptographic digests only.
It NEVER deserializes, executes, or loads model weights.

Design Reference
----------------
Section 10.1, 11.2, 12 of docs/implementation_design.md
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

# 64 KB streaming chunk — matches the download chunk size defined in Section 11.2
_CHUNK_SIZE: int = 65_536  # 64 KB


def sha256_of_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of an in-memory bytes object.

    Returns exactly 64 lowercase hexadecimal characters.

    Parameters
    ----------
    data:
        Raw bytes to hash.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256(data).hexdigest()
    assert len(digest) == 64
    return digest


def sha256_of_file(path: str | Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Compute a streaming SHA-256 digest of a local file.

    Reads the file in bounded chunks so that arbitrarily large model files
    are never fully loaded into memory at once.

    Parameters
    ----------
    path:
        Absolute or relative path to the file to hash.
    chunk_size:
        Number of bytes to read per iteration (default: 64 KB).

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    assert len(digest) == 64
    return digest


def sha256_of_stream(
    stream: io.RawIOBase | io.BufferedIOBase,
    chunk_size: int = _CHUNK_SIZE,
) -> str:
    """Compute a streaming SHA-256 digest from a readable binary stream.

    Reads the stream from its current position to EOF in bounded chunks.
    The stream position is NOT reset before or after reading.

    Parameters
    ----------
    stream:
        A readable binary stream (e.g., ``io.BytesIO``, an open file in
        binary mode, or a socket-backed stream).
    chunk_size:
        Number of bytes to read per iteration (default: 64 KB).

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        h.update(chunk)
    digest = h.hexdigest()
    assert len(digest) == 64
    return digest


def sha256_incremental() -> hashlib._Hash:
    """Return a fresh SHA-256 hash object for incremental (streaming) updates.

    Callers update the returned object with ``h.update(chunk)`` and retrieve
    the final digest with ``h.hexdigest()``.

    Example
    -------
    ::

        h = sha256_incremental()
        for chunk in download_chunks():
            h.update(chunk)
        digest = h.hexdigest()

    Returns
    -------
    hashlib._Hash
        A fresh ``hashlib.sha256()`` hash object.
    """
    return hashlib.sha256()


__all__ = [
    "sha256_incremental",
    "sha256_of_bytes",
    "sha256_of_file",
    "sha256_of_stream",
]
