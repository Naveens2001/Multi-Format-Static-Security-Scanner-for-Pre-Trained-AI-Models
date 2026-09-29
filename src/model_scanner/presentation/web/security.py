"""
model_scanner.presentation.web.security
=======================================

Security controls, path sanitization, upload size guards, and streaming multipart
parser for the Web UI (Phase 9).
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import uuid
from email.message import EmailMessage
from email.parser import BytesFeedParser
from email.policy import default
from pathlib import Path
from typing import IO, BinaryIO

_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_\-\.]")
_DEFAULT_MAX_UPLOAD_BYTES: int = 1024 * 1024 * 1024  # 1 GB default

#: Ceilings for folder uploads. A model repository is a handful of weight files
#: plus config and code; anything past these is not a model folder.
_MAX_UPLOAD_FILES: int = 512
_MAX_PATH_DEPTH: int = 12


class WebSecurityError(Exception):
    """Base exception for web security violations."""

    def __init__(self, message: str, error_code: str = "SECURITY_VIOLATION") -> None:
        super().__init__(message)
        self.error_code = error_code


class PayloadTooLargeError(WebSecurityError):
    """Raised when an uploaded file or request exceeds size budget."""

    def __init__(
        self,
        message: str = "Uploaded file exceeds maximum allowable size (1 GB).",
    ) -> None:
        super().__init__(message, error_code="PAYLOAD_TOO_LARGE")


def sanitize_filename(raw_filename: str) -> str:
    """Sanitize an untrusted user-supplied filename against path traversal.

    Parameters
    ----------
    raw_filename:
        The raw filename string from multipart header or URL.

    Returns
    -------
    str
        Safe filename containing only [a-zA-Z0-9_\\-\\.] characters.
    """
    clean_name = os.path.basename(raw_filename.replace("\\", "/"))
    safe_name = _SAFE_FILENAME_RE.sub("_", clean_name)
    safe_name = safe_name.lstrip(".")

    if not safe_name or safe_name in (".", ".."):
        safe_name = f"model_{uuid.uuid4().hex[:8]}.bin"

    return safe_name


def sanitize_relative_path(raw_path: str) -> str:
    """Sanitize an untrusted relative path from a folder upload.

    Browsers send ``webkitRelativePath`` (e.g. ``my-model/configs/config.json``)
    as the multipart filename so the folder's structure can be rebuilt. That
    structure matters: the repository analyzer needs ``config.json`` at the
    root and bundled ``.py`` files where they actually sit. But an attacker
    controls this string, so every segment is sanitized independently and any
    traversal, absolute prefix, or drive letter is discarded rather than
    resolved.

    Parameters
    ----------
    raw_path:
        The raw relative path from the multipart Content-Disposition header.

    Returns
    -------
    str
        A safe relative path using forward slashes, or a generated safe
        filename if nothing usable survives sanitization.

    Raises
    ------
    WebSecurityError
        If the path nests deeper than the permitted limit.
    """
    normalized = raw_path.replace("\\", "/")

    # Strip a Windows drive letter or a leading root before splitting.
    normalized = re.sub(r"^[a-zA-Z]:", "", normalized)
    segments = [seg for seg in normalized.split("/") if seg not in ("", ".")]

    if any(seg == ".." for seg in segments):
        raise WebSecurityError(
            f"Upload path {raw_path!r} contains a parent-directory reference.",
            error_code="PATH_TRAVERSAL_BLOCKED",
        )

    if len(segments) > _MAX_PATH_DEPTH:
        raise WebSecurityError(
            f"Upload path nests deeper than {_MAX_PATH_DEPTH} levels.",
            error_code="PATH_TOO_DEEP",
        )

    safe_segments: list[str] = []
    for seg in segments:
        clean = _SAFE_FILENAME_RE.sub("_", seg).lstrip(".")
        if clean and clean not in (".", ".."):
            safe_segments.append(clean)

    if not safe_segments:
        return f"model_{uuid.uuid4().hex[:8]}.bin"

    return "/".join(safe_segments)


def create_isolated_upload_dir(base_dir: Path | None = None) -> Path:
    """Create a private, UUID-isolated temporary directory for an upload session.

    Parameters
    ----------
    base_dir:
        Optional root temporary directory path.

    Returns
    -------
    Path
        Isolated directory path.
    """
    root = base_dir or (Path(tempfile.gettempdir()) / "model_scanner_uploads")
    root.mkdir(parents=True, exist_ok=True)

    isolated = root / f"upload_{uuid.uuid4().hex}"
    isolated.mkdir(parents=True, exist_ok=False)
    return isolated


def parse_multipart_upload(
    stream: io.BufferedIOBase | BinaryIO | IO[bytes],
    content_type_header: str,
    content_length: int,
    destination_dir: Path,
    max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
) -> tuple[list[Path], dict[str, str]]:
    """Parse multipart/form-data upload, streaming file parts directly to disk.

    Parameters
    ----------
    stream:
        Raw input stream from HTTP request (rfile).
    content_type_header:
        Content-Type header value containing boundary parameter.
    content_length:
        Content-Length integer value.
    destination_dir:
        Directory where uploaded files will be written.
    max_upload_bytes:
        Maximum allowed total byte ceiling.

    Returns
    -------
    tuple[list[Path], dict[str, str]]
        A tuple of (saved_file_paths, form_fields_dict).

    Raises
    ------
    PayloadTooLargeError
        If content_length or actual stream size exceeds max_upload_bytes.
    WebSecurityError
        If content type or boundary is missing or invalid.
    """
    if content_length > max_upload_bytes:
        raise PayloadTooLargeError(
            f"Upload payload size ({content_length} bytes) exceeds limit "
            f"({max_upload_bytes} bytes)."
        )

    if not content_type_header or "multipart/form-data" not in content_type_header:
        raise WebSecurityError(
            "Expected 'multipart/form-data' Content-Type.",
            error_code="INVALID_CONTENT_TYPE",
        )

    raw_data = stream.read(content_length)
    if len(raw_data) > max_upload_bytes:
        raise PayloadTooLargeError("Stream exceeded upload budget during read.")

    msg_header = (
        f"Content-Type: {content_type_header}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1")
    )
    feed_parser = BytesFeedParser(policy=default)
    feed_parser.feed(msg_header + raw_data)
    message = feed_parser.close()

    saved_files: list[Path] = []
    fields: dict[str, str] = {}

    if isinstance(message, EmailMessage) and message.is_multipart():
        for part in message.iter_parts():
            content_disposition = str(part.get("Content-Disposition", ""))
            filename = part.get_filename()

            if filename:
                if len(saved_files) >= _MAX_UPLOAD_FILES:
                    raise WebSecurityError(
                        f"Upload contains more than {_MAX_UPLOAD_FILES} files; "
                        f"that is not a model folder.",
                        error_code="TOO_MANY_FILES",
                    )
                # A folder upload sends webkitRelativePath as the filename, so
                # the tree is rebuilt under destination_dir. A single-file
                # upload has no separator and behaves exactly as before.
                if "/" in filename or "\\" in filename:
                    rel = sanitize_relative_path(filename)
                else:
                    rel = sanitize_filename(filename)
                target_path = destination_dir / rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                raw_payload = part.get_payload(decode=True)
                payload_bytes: bytes | None = None
                if isinstance(raw_payload, bytes):
                    payload_bytes = raw_payload
                elif isinstance(raw_payload, str):
                    payload_bytes = raw_payload.encode("utf-8")
                else:
                    content = part.get_content()
                    if isinstance(content, bytes):
                        payload_bytes = content
                    elif isinstance(content, str):
                        payload_bytes = content.encode("utf-8")

                if payload_bytes is not None:
                    target_path.write_bytes(payload_bytes)
                    saved_files.append(target_path)
            else:
                name_match = re.search(r'name="([^"]+)"', content_disposition)
                if name_match:
                    field_name = name_match.group(1)
                    field_value = part.get_content()
                    if isinstance(field_value, str):
                        fields[field_name] = field_value.strip()

    return saved_files, fields


__all__ = [
    "PayloadTooLargeError",
    "WebSecurityError",
    "create_isolated_upload_dir",
    "parse_multipart_upload",
    "sanitize_filename",
    "sanitize_relative_path",
]
