"""
model_scanner.analyzers.gguf.analyzer
======================================

Composite static security analyzer for GGUF model files.

Pipeline
--------
1. Read file bytes from Artifact (local_path or stream).
2. Unpack binary header (magic, version, counts) via GGUFReader (reader.py).
3. Enforce resource bounds (max_gguf_kv_count, max_gguf_value_bytes).
4. Parse Key-Value metadata entries.
5. Parse and validate tensor descriptors: type, shape, offsets, alignment, bounds,
   overlaps (tensors.py).
6. Scan metadata keys and string values for security patterns (metadata.py).
7. Emit strongly-typed Finding objects using the standard data models.

SECURITY GUARANTEE
------------------
This module NEVER:
  - Loads tensor payloads or invokes llama.cpp / ML runtimes
  - Calls torch.load(), pickle.loads(), eval(), or exec()
  - Imports torch, transformers, safetensors, or external GGUF libraries
  - Allocates buffers proportional to tensor dimensions

Design Reference
----------------
Section 17 of docs/implementation_design.md
"""

from __future__ import annotations

import io
from collections.abc import Sequence

from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.analyzers.gguf.integrity import check_gguf_integrity, _FLOAT_GGML_TYPES
from model_scanner.analyzers.gguf.metadata import scan_gguf_metadata
from model_scanner.analyzers.gguf.reader import GGUFReader
from model_scanner.analyzers.gguf.tensors import (
    DEFAULT_GGUF_ALIGNMENT,
    parse_and_validate_tensors,
)
from model_scanner.core.exceptions import (
    ResourceLimitError,
)
from model_scanner.core.models import (
    Artifact,
    Confidence,
    CoordinateSpace,
    Evidence,
    Finding,
    InputSourceType,
    ModelFormat,
    ScanConfig,
    Severity,
)

# ---------------------------------------------------------------------------
# Rule Metadata Tables
# ---------------------------------------------------------------------------

_RULE_SEVERITY: dict[str, Severity] = {
    "GU101": Severity.CRITICAL,
    "GU102": Severity.HIGH,
    "GU103": Severity.HIGH,
    "GU104": Severity.MEDIUM,
    "GU105": Severity.MEDIUM,
    "GU106": Severity.HIGH,
    "GU201": Severity.HIGH,
    "GU202": Severity.HIGH,
    "GU203": Severity.HIGH,
    "GU204": Severity.MEDIUM,
    "GU205": Severity.HIGH,
    "GU206": Severity.MEDIUM,
    "GU207": Severity.MEDIUM,
    "GU208": Severity.HIGH,
    "GU209": Severity.HIGH,
    "GU210": Severity.HIGH,
}

_RULE_CONFIDENCE: dict[str, Confidence] = {
    "GU101": Confidence.HIGH,
    "GU102": Confidence.HIGH,
    "GU103": Confidence.HIGH,
    "GU104": Confidence.HIGH,
    "GU105": Confidence.MEDIUM,
    "GU106": Confidence.HIGH,
    "GU201": Confidence.HIGH,
    "GU202": Confidence.HIGH,
    "GU203": Confidence.HIGH,
    "GU204": Confidence.HIGH,
    "GU205": Confidence.HIGH,
    "GU206": Confidence.HIGH,
    "GU207": Confidence.HIGH,
    "GU208": Confidence.HIGH,
    "GU209": Confidence.HIGH,
    "GU210": Confidence.HIGH,
}

_RULE_NAMES: dict[str, str] = {
    "GU101": "Metadata Code Injection",
    "GU102": "Path Traversal Sequence in Metadata",
    "GU103": "Metadata Credential Exposure",
    "GU104": "Oversized Metadata Staging Buffer",
    "GU105": "Suspicious URL in Metadata",
    "GU106": "Script Injection Tag in Metadata",
    "GU201": "GGUF Resource Limit Exceeded",
    "GU202": "Invalid Magic or Unsupported GGUF Version",
    "GU203": "Truncated or Malformed GGUF Stream",
    "GU204": "Invalid Metadata Value Type",
    "GU205": "Invalid or Duplicate Tensor Name",
    "GU206": "Invalid Tensor Dimensions",
    "GU207": "Invalid Tensor Type",
    "GU208": "Invalid Tensor Offset or Alignment",
    "GU209": "Tensor Range Outside File Bounds",
    "GU210": "Overlapping Tensor Data Ranges",
}

_RULE_CATEGORY: dict[str, str] = {
    "GU101": "code_execution",
    "GU102": "tampering",
    "GU103": "credential_exposure",
    "GU104": "structural_anomaly",
    "GU105": "credential_exposure",
    "GU106": "code_execution",
    "GU201": "structural_anomaly",
    "GU202": "format_integrity",
    "GU203": "structural_anomaly",
    "GU204": "structural_anomaly",
    "GU205": "structural_anomaly",
    "GU206": "structural_anomaly",
    "GU207": "structural_anomaly",
    "GU208": "structural_anomaly",
    "GU209": "structural_anomaly",
    "GU210": "structural_anomaly",
}

_RULE_REMEDIATION: dict[str, str] = {
    "GU101": (
        "Remove executable code primitives from model metadata. GGUF metadata "
        "must contain only configuration and descriptive strings. Re-export the "
        "model from a trusted conversion tool."
    ),
    "GU102": (
        "Remove path traversal sequences (e.g. '../', '/etc/passwd') from metadata "
        "keys and values. Ensure all file references use safe relative basenames."
    ),
    "GU103": (
        "Revoke exposed credentials immediately. Strip API keys, tokens, and "
        "private keys from model metadata before publishing or distributing."
    ),
    "GU104": (
        "The GGUF metadata string or array value exceeds the maximum allowable "
        "buffer limit. Remove oversized staging buffers and re-export the model."
    ),
    "GU105": (
        "Review embedded URLs in model metadata. Verify that links point to "
        "authorized and verified endpoints. Remove unnecessary URLs."
    ),
    "GU106": (
        "Remove HTML/script injection tags and event handlers from model metadata. "
        "GGUF metadata must contain plain text values only."
    ),
    "GU201": (
        "The GGUF header contains an excessive number of metadata KV entries, "
        "tensor descriptors, or nested array elements. Re-export the model with "
        "standard configuration parameters."
    ),
    "GU202": (
        "The file does not start with valid GGUF magic bytes (b'GGUF') or uses "
        "an unsupported format version. Re-convert the model using a standard "
        "GGUF v2/v3 exporter."
    ),
    "GU203": (
        "The GGUF binary stream is truncated or malformed. Re-download or "
        "re-export the complete file from a trusted source."
    ),
    "GU204": (
        "The GGUF metadata specifies an unknown or invalid value type enum. "
        "Verify converter integrity and re-export the model."
    ),
    "GU205": (
        "Duplicate or empty tensor names detected in GGUF descriptors. Tensor "
        "names must be unique non-empty UTF-8 strings. Re-export the model."
    ),
    "GU206": (
        "Tensor dimensions are negative, excessive, or cause integer overflow. "
        "Verify model architecture and re-export from a trusted conversion pipeline."
    ),
    "GU207": (
        "Tensor descriptor specifies an unknown GGML quantization/data type. "
        "Use only standard GGML quantization formats."
    ),
    "GU208": (
        "Tensor data offset is negative or data section alignment is invalid. "
        "Ensure alignment is a positive power of 2 and re-export the model."
    ),
    "GU209": (
        "Tensor byte range extends beyond the physical end of the file. "
        "The file is truncated or the header has been tampered with. Reject the file."
    ),
    "GU210": (
        "Two or more tensors share overlapping byte ranges in the data section. "
        "This indicates header manipulation, payload smuggling, or memory aliasing. "
        "Reject this file."
    ),
}


class GGUFAnalyzer(BaseFormatAnalyzer):
    """Pure standard-library static security analyzer for GGUF model files."""

    model_structure: dict[str, object] | None = None

    def analyze(
        self, artifact: Artifact, config: ScanConfig | None = None
    ) -> Sequence[Finding]:
        """Perform static analysis on a GGUF artifact.

        Parameters
        ----------
        artifact:
            The raw, SHA-256-fingerprinted GGUF artifact.
        config:
            Optional ScanConfig; uses defaults when not provided.

        Returns
        -------
        Sequence[Finding]
            All security findings, including structural anomalies and
            metadata security violations.
        """
        active_config = config or ScanConfig()
        data = self._read_artifact_bytes(artifact)
        findings: list[Finding] = []
        counter = [0]

        def _next_id(rule_id: str) -> str:
            counter[0] += 1
            return f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}"

        def _make_finding(
            rule_id: str,
            byte_offset: int | None,
            matched_value: str,
            description: str,
            *,
            raw_hex: str | None = None,
            context_snippet: str | None = None,
        ) -> Finding:
            return Finding(
                finding_id=_next_id(rule_id),
                rule_id=rule_id,
                rule_name=_RULE_NAMES[rule_id],
                severity=_RULE_SEVERITY[rule_id],
                confidence=_RULE_CONFIDENCE[rule_id],
                category=_RULE_CATEGORY[rule_id],
                format=ModelFormat.GGUF,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.HEADER,
                    byte_offset=byte_offset,
                    byte_length=None,
                    archive_member=None,
                    matched_value=matched_value[:512],
                    raw_hex=raw_hex,
                    context_snippet=context_snippet,
                ),
                description=description,
                remediation=_RULE_REMEDIATION[rule_id],
            )

        # ------------------------------------------------------------------
        # Stage 1: Header and KV Metadata Parsing
        # ------------------------------------------------------------------
        reader = GGUFReader(
            data,
            max_kv_count=active_config.max_gguf_kv_count,
            max_value_bytes=active_config.max_gguf_value_bytes,
        )

        try:
            header, kv_entries, tensor_start_offset = reader.parse_header()
        except ResourceLimitError as exc:
            findings.append(
                _make_finding(
                    "GU201",
                    byte_offset=8,
                    matched_value=exc.error_code,
                    description=str(exc),
                )
            )
            return findings

        # Collect issues from reader
        for issue in reader.issues:
            findings.append(
                _make_finding(
                    issue.rule_id,
                    byte_offset=issue.byte_offset,
                    matched_value=issue.matched_value,
                    description=issue.detail,
                )
            )

        if header is None:
            return findings

        # ------------------------------------------------------------------
        # Stage 2: Alignment Discovery & Tensor Descriptors Parsing
        # ------------------------------------------------------------------
        alignment = DEFAULT_GGUF_ALIGNMENT
        for entry in kv_entries:
            if entry.key == "general.alignment" and isinstance(entry.value, int):
                alignment = entry.value
                break

        tensor_descriptors, tensor_issues, data_section_start = parse_and_validate_tensors(
            data,
            start_offset=tensor_start_offset,
            tensor_count=header.tensor_count,
            version=header.version,
            alignment=alignment,
        )

        for t_issue in tensor_issues:
            findings.append(
                _make_finding(
                    t_issue.rule_id,
                    byte_offset=t_issue.byte_offset,
                    matched_value=t_issue.tensor_name,
                    description=t_issue.detail,
                )
            )

        # Weight Integrity Check (WI101/WI102/WI103)
        # check_gguf_integrity is imported at module level (Finding 6 — no deferred import)
        integrity_findings = check_gguf_integrity(
            data=data,
            descriptors=tensor_descriptors,
            data_section_start=data_section_start,
            artifact=artifact,
            counter_start=len(findings),
        )
        findings.extend(integrity_findings)

        # Structural inventory from parsed descriptors (informational metadata)
        # _FLOAT_GGML_TYPES imported from gguf/integrity.py (Finding 8 — no local duplicate)
        self.model_structure = {
            "tensor_count": len(tensor_descriptors),
            "tensors": [
                {
                    "name": d.name,
                    "dtype": _FLOAT_GGML_TYPES.get(d.tensor_type, f"ggml_type_{d.tensor_type}"),
                    "shape": list(d.dimensions),
                }
                for d in tensor_descriptors[:1000]
            ],
        }

        # ------------------------------------------------------------------
        # Stage 3: Metadata Security Pattern Scan
        # ------------------------------------------------------------------
        meta_findings = scan_gguf_metadata(kv_entries)
        for mf in meta_findings:
            findings.append(
                _make_finding(
                    mf.rule_id,
                    byte_offset=mf.byte_offset,
                    matched_value=mf.matched_value,
                    description=(
                        f"GGUF metadata key {mf.key!r} contains {mf.matched_pattern}: "
                        f"{mf.matched_value[:100]!r}"
                    ),
                    context_snippet=f"key={mf.key!r}, pattern={mf.matched_pattern}",
                )
            )

        return findings

    @staticmethod
    def _read_artifact_bytes(artifact: Artifact) -> bytes:
        """Read all bytes from an Artifact's local_path or stream."""
        if artifact.local_path is not None:
            return artifact.local_path.read_bytes()
        if artifact.stream is not None:
            pos = artifact.stream.tell() if artifact.stream.seekable() else 0
            data = artifact.stream.read()
            if artifact.stream.seekable():
                artifact.stream.seek(pos)
            return data
        return b""

    def analyze_bytes(
        self,
        data: bytes,
        artifact_sha256: str,
        config: ScanConfig | None = None,
    ) -> Sequence[Finding]:
        """Convenience method for analyzing raw bytes directly.

        Parameters
        ----------
        data:
            Raw GGUF file bytes.
        artifact_sha256:
            SHA-256 hex digest of the data (64 lowercase hex characters).
        config:
            Optional ScanConfig.

        Returns
        -------
        Sequence[Finding]
            All security findings.
        """
        stream = io.BytesIO(data)
        artifact = Artifact(
            artifact_id="inline",
            source_path="<bytes>",
            input_source=InputSourceType.IN_MEMORY,
            size_bytes=len(data),
            sha256=artifact_sha256,
            stream=stream,
        )
        return self.analyze(artifact, config)


__all__ = ["GGUFAnalyzer"]
