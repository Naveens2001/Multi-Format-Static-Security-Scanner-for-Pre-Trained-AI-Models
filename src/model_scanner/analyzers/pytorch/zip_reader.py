"""
model_scanner.analyzers.pytorch.zip_reader
==========================================

Read-only, bounded PyTorch ZIP container navigator with CRC-32 tolerance.

SECURITY GUARANTEE
------------------
Uses standard library zipfile strictly in read-only mode.
Never extracts archives to filesystem, never loads tensor weights, never executes code.

Design Reference
----------------
Section 15, FR-12, FR-14 of docs/implementation_design.md
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from model_scanner.core.exceptions import AnalysisError, ResourceLimitError

DEFAULT_MAX_ZIP_MEMBERS: int = 5_000
DEFAULT_MAX_CUMULATIVE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_MAX_MEMBER_BYTES: int = 1 * 1024 * 1024 * 1024      # 1 GB


@dataclass(frozen=True)
class ZipMemberInfo:
    """Metadata for a single member inside a PyTorch ZIP archive."""

    filename: str
    file_size: int
    compress_size: int
    is_candidate_pickle: bool
    is_tensor_data: bool = False


def _is_pickle_filename(name: str) -> bool:
    """Return True if filename matches standard PyTorch Pickle member conventions."""
    clean = name.replace("\\", "/").rstrip("/")
    base = clean.rsplit("/", 1)[-1].lower()

    if base in ("data.pkl", "constants.pkl"):
        return True
    if base.endswith((".pkl", ".pickle", ".joblib")):
        return True
    return False


def _is_pytorch_tensor_storage(name: str) -> bool:
    """Return True if the member is PyTorch raw tensor storage or metadata."""
    clean = name.replace("\\", "/").rstrip("/")
    base = clean.rsplit("/", 1)[-1].lower()

    if base == "version":
        return True
    if "/data/" in clean.lower():
        if base.isdigit():
            return True
    return False


class PyTorchZipReader:
    """Safe read-only inspector for PyTorch ZIP archives."""

    def __init__(
        self,
        max_members: int = DEFAULT_MAX_ZIP_MEMBERS,
        max_cumulative_bytes: int = DEFAULT_MAX_CUMULATIVE_BYTES,
        max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    ) -> None:
        self.max_members = max_members
        self.max_cumulative_bytes = max_cumulative_bytes
        self.max_member_bytes = max_member_bytes
        self._cumulative_bytes: int = 0

    def list_members(self, archive_input: bytes | Path) -> list[ZipMemberInfo]:
        """Enumerate all archive members and identify candidate Pickle streams.

        Parameters
        ----------
        archive_input:
            Raw bytes or Path to the ZIP archive.

        Returns
        -------
        list[ZipMemberInfo]
            List of member metadata records.

        Raises
        ------
        ResourceLimitError
            If member count exceeds max_members.
        AnalysisError
            If the archive is not a valid ZIP file.
        """
        fh = self._open_archive(archive_input)
        try:
            with zipfile.ZipFile(fh, "r") as zf:
                infolist = zf.infolist()

                if len(infolist) > self.max_members:
                    raise ResourceLimitError(
                        f"ZIP archive contains {len(infolist)} members, "
                        f"exceeding ceiling of {self.max_members}",
                        error_code="ZIP_TOO_MANY_MEMBERS",
                        limit_name="max_zip_members",
                        limit_value=self.max_members,
                        actual_value=len(infolist),
                    )

                members: list[ZipMemberInfo] = []
                for info in infolist:
                    if info.is_dir():
                        continue

                    is_candidate = _is_pickle_filename(info.filename)
                    is_tensor = _is_pytorch_tensor_storage(info.filename)
                    if is_tensor:
                        is_candidate = False
                    members.append(
                        ZipMemberInfo(
                            filename=info.filename,
                            file_size=info.file_size,
                            compress_size=info.compress_size,
                            is_candidate_pickle=is_candidate,
                            is_tensor_data=is_tensor,
                        )
                    )
                return members

        except zipfile.BadZipFile as exc:
            raise AnalysisError(
                f"Corrupted or invalid PyTorch ZIP archive: {exc}",
                error_code="INVALID_ZIP_ARCHIVE",
                details=str(exc),
            ) from exc

    def extract_member(self, archive_input: bytes | Path, member_name: str) -> bytes:
        """Extract a single member's bytes under strict size limits with CRC tolerance.

        Parameters
        ----------
        archive_input:
            Raw bytes or Path to the ZIP archive.
        member_name:
            Target member filename to extract.

        Returns
        -------
        bytes
            Uncompressed bytes of the target member.

        Raises
        ------
        ResourceLimitError
            If member size exceeds max_member_bytes or cumulative bytes
            exceed max_cumulative_bytes.
        AnalysisError
            If member is missing or cannot be read.
        """
        fh = self._open_archive(archive_input)
        try:
            with zipfile.ZipFile(fh, "r") as zf:
                try:
                    info = zf.getinfo(member_name)
                except KeyError as exc:
                    raise AnalysisError(
                        f"Member {member_name!r} not found in ZIP archive",
                        error_code="MEMBER_NOT_FOUND",
                        details=member_name,
                    ) from exc

                if info.file_size > self.max_member_bytes:
                    raise ResourceLimitError(
                        f"ZIP member {member_name!r} size ({info.file_size} bytes) "
                        f"exceeds limit of {self.max_member_bytes} bytes",
                        error_code="MEMBER_TOO_LARGE",
                        limit_name="max_member_bytes",
                        limit_value=self.max_member_bytes,
                        actual_value=info.file_size,
                    )

                # Cumulative pre-check using the declared uncompressed size.
                projected_cumulative = self._cumulative_bytes + info.file_size
                if projected_cumulative > self.max_cumulative_bytes:
                    raise ResourceLimitError(
                        f"Extracting {member_name!r} ({info.file_size} bytes) would "
                        f"exceed cumulative extraction ceiling of "
                        f"{self.max_cumulative_bytes} bytes "
                        f"(already extracted {self._cumulative_bytes} bytes)",
                        error_code="CUMULATIVE_LIMIT_EXCEEDED",
                        limit_name="max_cumulative_bytes",
                        limit_value=self.max_cumulative_bytes,
                        actual_value=projected_cumulative,
                    )

                try:
                    with zf.open(info, "r") as mf:
                        data = mf.read(self.max_member_bytes + 1)
                        if len(data) > self.max_member_bytes:
                            raise ResourceLimitError(
                                f"Extracted bytes exceeded ceiling for {member_name!r}",
                                error_code="MEMBER_TOO_LARGE",
                                limit_name="max_member_bytes",
                                limit_value=self.max_member_bytes,
                                actual_value=len(data),
                            )
                        self._cumulative_bytes += len(data)
                        return data
                except (zipfile.BadZipFile, OSError) as exc:
                    # Tolerant fallback (FR-14): Attempt to read uncompressed STORED data directly
                    if info.compress_type == zipfile.ZIP_STORED:
                        try:
                            # Re-read raw stored segment
                            header_offset = info.header_offset
                            raw_stream: BinaryIO = (
                                io.BytesIO(archive_input)
                                if isinstance(archive_input, bytes)
                                else open(archive_input, "rb")
                            )
                            try:
                                raw_stream.seek(header_offset)
                                # Read local header (30 bytes + filename + extra)
                                local_hdr = raw_stream.read(30)
                                if len(local_hdr) >= 30 and local_hdr.startswith(b"PK\x03\x04"):
                                    fn_len = int.from_bytes(local_hdr[26:28], "little")
                                    extra_len = int.from_bytes(local_hdr[28:30], "little")
                                    raw_stream.seek(header_offset + 30 + fn_len + extra_len)
                                    read_len = min(info.file_size, self.max_member_bytes)
                                    raw_data = raw_stream.read(read_len)
                                    self._cumulative_bytes += len(raw_data)
                                    return raw_data
                            finally:
                                raw_stream.close()
                        except OSError:
                            pass
                    raise AnalysisError(
                        f"Failed to extract ZIP member {member_name!r}: {exc}",
                        error_code="ZIP_EXTRACTION_ERROR",
                        details=str(exc),
                    ) from exc

        except zipfile.BadZipFile as exc:
            raise AnalysisError(
                f"Corrupted or invalid ZIP archive: {exc}",
                error_code="INVALID_ZIP_ARCHIVE",
                details=str(exc),
            ) from exc

    @staticmethod
    def _open_archive(archive_input: bytes | Path) -> io.BytesIO | Path:
        if isinstance(archive_input, bytes):
            return io.BytesIO(archive_input)
        return archive_input


__all__ = [
    "DEFAULT_MAX_CUMULATIVE_BYTES",
    "DEFAULT_MAX_MEMBER_BYTES",
    "DEFAULT_MAX_ZIP_MEMBERS",
    "PyTorchZipReader",
    "ZipMemberInfo",
]
