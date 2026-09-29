"""
model_scanner.analyzers.pytorch.analyzer
=======================================

Static security analyzer for PyTorch ZIP archives and embedded Pickle streams.

SECURITY GUARANTEE
------------------
Zero dynamic execution, zero model loading, zero tensor decompression.
Uses standard library zipfile and PickleDisassembler only.

Design Reference
----------------
Section 15 of docs/implementation_design.md
"""

from __future__ import annotations

from collections.abc import Sequence

from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.analyzers.pickle.analyzer import PickleAnalyzer
from model_scanner.analyzers.pytorch.zip_reader import PyTorchZipReader
from model_scanner.core.exceptions import AnalysisError, ResourceLimitError
from model_scanner.core.models import (
    Artifact,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    ModelFormat,
    ScanConfig,
    Severity,
)
from model_scanner.detection.magic import (
    check_pickle_protocol_0_heuristic,
    is_pickle_magic,
)


class PyTorchAnalyzer(BaseFormatAnalyzer):
    """Static security analyzer for PyTorch ZIP container archives."""

    def __init__(self) -> None:
        self._zip_reader = PyTorchZipReader()
        self._pickle_analyzer = PickleAnalyzer()

    def analyze(
        self, artifact: Artifact, config: ScanConfig | None = None
    ) -> Sequence[Finding]:
        """Inspect a PyTorch ZIP archive and analyze all embedded Pickle streams.

        Parameters
        ----------
        artifact:
            The acquired PyTorch ZIP artifact.
        config:
            Optional ScanConfig.

        Returns
        -------
        Sequence[Finding]
            Aggregated findings from container analysis and embedded Pickle disassembly.
        """
        active_config = config or ScanConfig()
        archive_input = (
            artifact.local_path
            if artifact.local_path is not None
            else self._read_artifact_bytes(artifact)
        )

        findings: list[Finding] = []

        try:
            members = self._zip_reader.list_members(archive_input)
        except AnalysisError as exc:
            # Corrupted ZIP container
            findings.append(
                Finding(
                    finding_id=f"FINDING-P201-{artifact.sha256[:8]}-001",
                    rule_id="P201",
                    rule_name="Malformed or Truncated Opcode Stream",
                    severity=Severity.HIGH,
                    confidence=Confidence.HIGH,
                    category="structural_anomaly",
                    format=ModelFormat.PYTORCH_ZIP,
                    artifact_sha256=artifact.sha256,
                    evidence=Evidence(
                        coordinate_space=CoordinateSpace.ARTIFACT,
                        byte_offset=0,
                        byte_length=None,
                        archive_member=None,
                        matched_value="INVALID_ZIP_ARCHIVE",
                        opcode=None,
                        context_snippet=str(exc),
                    ),
                    description=(
                        "File identified as PYTORCH_ZIP format but failed ZIP archive "
                        "validation — the file is not a valid ZIP archive. "
                        "Security analysis could not be completed."
                        if exc.error_code == "INVALID_ZIP_ARCHIVE"
                        else f"Corrupted or invalid PyTorch ZIP archive: {exc}"
                    ),
                    remediation="Re-serialize or re-download the PyTorch model archive.",
                )
            )
            return findings

        # Discover candidate Pickle members
        pickle_members = [m for m in members if m.is_candidate_pickle]

        if not pickle_members:
            # Check all members if none matched by name using the same magic-detection
            # policy as Phase 3 (magic.py):
            #   - Binary protocols 2-5: deterministic via \x80 prefix.
            #   - Protocol 0/1: require strong multi-byte structural heuristic.
            # Single-byte ASCII opcodes alone are NOT sufficient evidence.
            for m in members:
                if m.is_tensor_data:
                    continue
                try:
                    head = self._zip_reader.extract_member(archive_input, m.filename)[:32]
                    if is_pickle_magic(head):
                        pickle_members.append(m)
                    else:
                        matched, _ = check_pickle_protocol_0_heuristic(head)
                        if matched:
                            pickle_members.append(m)
                except (AnalysisError, ResourceLimitError):
                    continue

        # A6: Structure profile checks — compression ratio anomaly (P211) and
        # nested archive detection (P210).
        finding_count = len(findings)
        for member in members:
            # P211: Extreme compression ratio → zip-bomb indicator
            # Threshold: compressed_size / uncompressed_size < 0.001 (1000:1 expansion)
            if (
                member.compress_size > 0
                and member.file_size > 0
                and member.compress_size / member.file_size < 0.001
            ):
                finding_count += 1
                ratio_str = f"{member.file_size}/{member.compress_size}"
                findings.append(
                    Finding(
                        finding_id=(
                            f"FINDING-P211-{artifact.sha256[:8]}-{finding_count:03d}"
                        ),
                        rule_id="P211",
                        rule_name="Extreme Compression Ratio (Zip-Bomb Indicator)",
                        severity=Severity.HIGH,
                        confidence=Confidence.MEDIUM,
                        category="structural_anomaly",
                        format=ModelFormat.PYTORCH_ZIP,
                        artifact_sha256=artifact.sha256,
                        evidence=Evidence(
                            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                            byte_offset=0,
                            byte_length=member.compress_size,
                            archive_member=member.filename,
                            matched_value=ratio_str[:512],
                            opcode=None,
                            context_snippet=(
                                f"uncompressed={member.file_size} bytes, "
                                f"compressed={member.compress_size} bytes, "
                                f"ratio={member.file_size / member.compress_size:.0f}:1"
                            ),
                        ),
                        description=(
                            f"ZIP member {member.filename!r} has an extreme "
                            f"compression ratio ({member.file_size}/{member.compress_size} "
                            f"= {member.file_size // member.compress_size}:1), "
                            f"a potential zip-bomb indicator."
                        ),
                        remediation=(
                            "Reject or quarantine this archive. Verify artifact provenance "
                            "before expanding."
                        ),
                    )
                )

        # P210: Nested ZIP detection — check first bytes of non-pickle/non-tensor members
        for member in members:
            if member.is_candidate_pickle or member.is_tensor_data:
                continue
            try:
                head = self._zip_reader.extract_member(archive_input, member.filename)[:4]
            except (AnalysisError, ResourceLimitError):
                continue
            if head[:4] == b"PK\x03\x04":
                finding_count += 1
                findings.append(
                    Finding(
                        finding_id=(
                            f"FINDING-P210-{artifact.sha256[:8]}-{finding_count:03d}"
                        ),
                        rule_id="P210",
                        rule_name="Nested Archive in PyTorch ZIP Container",
                        severity=Severity.MEDIUM,
                        confidence=Confidence.HIGH,
                        category="structural_anomaly",
                        format=ModelFormat.PYTORCH_ZIP,
                        artifact_sha256=artifact.sha256,
                        evidence=Evidence(
                            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                            byte_offset=0,
                            byte_length=member.compress_size,
                            archive_member=member.filename,
                            matched_value=member.filename,
                            opcode=None,
                            context_snippet=(
                                f"Member {member.filename!r} starts with ZIP magic "
                                f"(PK\\x03\\x04) — nested archive at depth 2"
                            ),
                        ),
                        description=(
                            f"PyTorch ZIP contains a nested ZIP archive "
                            f"({member.filename!r}). Legitimate model archives do not "
                            f"embed nested archives."
                        ),
                        remediation=(
                            "Inspect the nested archive contents. Reject if unexpected "
                            "or from an untrusted source."
                        ),
                    )
                )

        finding_count = len(findings)

        for member in pickle_members:
            finding_count += 1
            # Add informational container finding: P209
            findings.append(
                Finding(
                    finding_id=f"FINDING-P209-{artifact.sha256[:8]}-{finding_count:03d}",
                    rule_id="P209",
                    rule_name="Embedded Pickle Stream in PyTorch ZIP Container",
                    severity=Severity.INFO,
                    confidence=Confidence.HIGH,
                    category="format_integrity",
                    format=ModelFormat.PYTORCH_ZIP,
                    artifact_sha256=artifact.sha256,
                    evidence=Evidence(
                        coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                        byte_offset=0,
                        byte_length=member.file_size,
                        archive_member=member.filename,
                        matched_value=member.filename,
                        opcode=None,
                        context_snippet=f"Member size: {member.file_size} bytes",
                    ),
                    description=(
                        f"PyTorch ZIP contains embedded Pickle stream: {member.filename!r}"
                    ),
                    remediation=(
                        "Inspect embedded Pickle opcodes. Migrate model weights to SafeTensors."
                    ),
                )
            )

            try:
                member_bytes = self._zip_reader.extract_member(archive_input, member.filename)
                embedded_findings = self._pickle_analyzer.analyze_bytes(
                    data=member_bytes,
                    artifact_sha256=artifact.sha256,
                    coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                    archive_member=member.filename,
                    model_format=ModelFormat.PYTORCH_ZIP,
                    config=active_config,
                )
                findings.extend(embedded_findings)
            except ResourceLimitError as exc:
                # Cumulative or per-member extraction limit reached — stop further
                # member analysis and report as a resource exhaustion finding.
                finding_count += 1
                findings.append(
                    Finding(
                        finding_id=(
                            f"FINDING-P208-{artifact.sha256[:8]}-{finding_count:03d}"
                        ),
                        rule_id="P208",
                        rule_name="Excessive Opcode or Memo Resource Usage",
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        category="structural_anomaly",
                        format=ModelFormat.PYTORCH_ZIP,
                        artifact_sha256=artifact.sha256,
                        evidence=Evidence(
                            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                            byte_offset=0,
                            byte_length=None,
                            archive_member=member.filename,
                            matched_value=exc.error_code,
                            opcode=None,
                            context_snippet=str(exc),
                        ),
                        description=(
                            f"ZIP extraction resource limit exceeded: {exc}"
                        ),
                        remediation=(
                            "Check archive size or increase scanner resource ceilings."
                        ),
                    )
                )
                # Stop processing further members after cumulative limit breach.
                break
            except AnalysisError as exc:
                finding_count += 1
                findings.append(
                    Finding(
                        finding_id=f"FINDING-P201-{artifact.sha256[:8]}-{finding_count:03d}",
                        rule_id="P201",
                        rule_name="Malformed or Truncated Opcode Stream",
                        severity=Severity.HIGH,
                        confidence=Confidence.HIGH,
                        category="structural_anomaly",
                        format=ModelFormat.PYTORCH_ZIP,
                        artifact_sha256=artifact.sha256,
                        evidence=Evidence(
                            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                            byte_offset=0,
                            byte_length=None,
                            archive_member=member.filename,
                            matched_value=member.filename,
                            opcode=None,
                            context_snippet=str(exc),
                        ),
                        description=f"Failed to extract member {member.filename!r}: {exc}",
                        remediation="Verify archive integrity and extract valid streams.",
                    )
                )

        return findings

    @staticmethod
    def _read_artifact_bytes(artifact: Artifact) -> bytes:
        if artifact.local_path is not None:
            return artifact.local_path.read_bytes()
        elif artifact.stream is not None:
            pos = artifact.stream.tell() if artifact.stream.seekable() else 0
            data = artifact.stream.read()
            if artifact.stream.seekable():
                artifact.stream.seek(pos)
            return data
        return b""


__all__ = ["PyTorchAnalyzer"]
