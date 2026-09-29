"""
model_scanner.analyzers.safetensors.analyzer
=============================================

Composite static security analyzer for SafeTensors model files.

Pipeline
--------
1. Read file bytes from Artifact (local_path or stream).
2. Unpack the 8-byte LE header-length prefix (header.py).
3. Run the lexical token/depth/duplicate-key guard (token_guard.py).
4. Parse the bounded JSON header with json.loads().
5. Validate tensor descriptors: dtype, shape, offsets, bounds, overlap
   (tensors.py).
6. Scan __metadata__ string values for security patterns (metadata.py).
7. Emit strongly-typed Finding objects using the existing models.

SECURITY GUARANTEE
------------------
This module NEVER:
  - Loads tensor weights or calls safetensors.torch.load_file()
  - Calls torch.load(), pickle.loads(), eval(), or exec()
  - Imports torch, safetensors, tensorflow, or any ML framework
  - Allocates buffers proportional to attacker-controlled tensor dimensions

The only memory allocations are:
  - The bounded JSON header bytes (capped at max_safetensors_header_mb)
  - The parsed JSON dict (bounded by token budget)
  - Finding / Evidence objects

Design Reference
----------------
Section 16 of docs/implementation_design.md
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.analyzers.safetensors.header import (
    extract_header_bytes,
    unpack_header_length,
)
from model_scanner.analyzers.safetensors.metadata import scan_metadata, scan_tensor_names
from model_scanner.analyzers.safetensors.tensors import validate_tensors
from model_scanner.analyzers.safetensors.token_guard import scan_json_tokens
from model_scanner.core.exceptions import (
    ResourceLimitError,
    SafeTensorsAnalysisError,
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
# Severity / Confidence tables keyed by rule_id
# ---------------------------------------------------------------------------
_RULE_SEVERITY: dict[str, Severity] = {
    "ST101": Severity.CRITICAL,
    "ST102": Severity.HIGH,
    "ST103": Severity.MEDIUM,
    "ST104": Severity.HIGH,
    "ST105": Severity.HIGH,
    "ST201": Severity.HIGH,
    "ST202": Severity.HIGH,
    "ST203": Severity.HIGH,
    "ST204": Severity.MEDIUM,
    "ST205": Severity.MEDIUM,
    "ST206": Severity.HIGH,
    "ST207": Severity.HIGH,
    "ST208": Severity.HIGH,
    "ST209": Severity.HIGH,
}

_RULE_CONFIDENCE: dict[str, Confidence] = {
    "ST101": Confidence.HIGH,
    "ST102": Confidence.HIGH,
    "ST103": Confidence.MEDIUM,
    "ST104": Confidence.HIGH,
    "ST105": Confidence.HIGH,
    "ST201": Confidence.HIGH,
    "ST202": Confidence.HIGH,
    "ST203": Confidence.HIGH,
    "ST204": Confidence.HIGH,
    "ST205": Confidence.HIGH,
    "ST206": Confidence.HIGH,
    "ST207": Confidence.HIGH,
    "ST208": Confidence.HIGH,
    "ST209": Confidence.HIGH,
}

_RULE_NAMES: dict[str, str] = {
    "ST101": "Metadata Code Execution String",
    "ST102": "Path Traversal Sequence in Metadata",
    "ST103": "Suspicious URL in Metadata",
    "ST104": "Credential or Secret Exposure in Metadata",
    "ST105": "Script Injection Tag in Metadata",
    "ST201": "Header Size or JSON Token DoS",
    "ST202": "Tensor Data Range Overlap",
    "ST203": "Invalid Tensor Data Offsets",
    "ST204": "Invalid Tensor dtype",
    "ST205": "Invalid Tensor shape",
    "ST206": "Missing Required Tensor Fields",
    "ST207": "Duplicate Tensor Name",
    "ST208": "Tensor Range Outside File Bounds",
    "ST209": "Truncated or Unreadable Header",
}

_RULE_REMEDIATION: dict[str, str] = {
    "ST101": (
        "Remove code-execution primitives from model metadata. "
        "SafeTensors metadata must contain only informational key-value strings. "
        "Re-serialize the file from a trusted source."
    ),
    "ST102": (
        "Remove path traversal sequences from model metadata. SafeTensors metadata "
        "must not reference file-system paths. Re-serialize from a trusted source."
    ),
    "ST103": (
        "Review the embedded URL in model metadata. Verify that the link is "
        "intentional and does not point to an unauthorized or malicious host. "
        "Remove URLs from metadata if they are not required."
    ),
    "ST104": (
        "Revoke exposed credentials immediately. Remove all API keys, tokens, "
        "and secrets from model metadata before publishing or distributing. "
        "Rotate any compromised credentials."
    ),
    "ST105": (
        "Remove all HTML/script tags and event-handler attributes from model "
        "metadata. SafeTensors metadata must contain plain text strings only. "
        "Re-serialize the file from a trusted source."
    ),
    "ST201": (
        "The SafeTensors header is oversized or contains an adversarially large "
        "JSON structure. Reject this file. Re-export the model with a correctly "
        "sized header from a trusted serialization tool."
    ),
    "ST202": (
        "Two or more tensors share overlapping byte regions in the data section. "
        "This indicates header manipulation, memory aliasing, or a smuggled "
        "payload. Reject and re-export the model from a trusted serializer."
    ),
    "ST203": (
        "Tensor data_offsets are invalid (negative, start > end, or non-integer). "
        "This indicates a malformed or tampered header. Re-export the model from "
        "a trusted serialization tool."
    ),
    "ST204": (
        "Unknown or invalid tensor dtype encountered. Only canonical SafeTensors "
        "dtypes are permitted. Re-export the model with a compatible dtype."
    ),
    "ST205": (
        "Tensor shape contains an invalid dimension (negative or non-integer). "
        "Verify the model export process and re-serialize from a trusted source."
    ),
    "ST206": (
        "Tensor descriptor is missing required fields (dtype, shape, or "
        "data_offsets). The SafeTensors header is malformed. Re-export the model."
    ),
    "ST207": (
        "Duplicate tensor names detected in the header. The SafeTensors format "
        "requires unique tensor names. This may indicate header tampering or a "
        "malformed serializer. Re-export the model."
    ),
    "ST208": (
        "Tensor data region extends beyond the end of the file data section. "
        "This indicates a malformed or tampered header. Reject this file and "
        "re-export from a trusted serialization tool."
    ),
    "ST209": (
        "The SafeTensors file is truncated or its header is unreadable. "
        "The file may be corrupted or only partially downloaded. "
        "Re-acquire the file from a trusted source."
    ),
}

_RULE_CATEGORY: dict[str, str] = {
    "ST101": "code_execution",
    "ST102": "tampering",
    "ST103": "credential_exposure",
    "ST104": "credential_exposure",
    "ST105": "code_execution",
    "ST201": "structural_anomaly",
    "ST202": "structural_anomaly",
    "ST203": "structural_anomaly",
    "ST204": "structural_anomaly",
    "ST205": "structural_anomaly",
    "ST206": "structural_anomaly",
    "ST207": "structural_anomaly",
    "ST208": "structural_anomaly",
    "ST209": "structural_anomaly",
}


class SafeTensorsAnalyzer(BaseFormatAnalyzer):
    """Pure-stdlib static security analyzer for SafeTensors model files.

    Implements the full six-stage inspection pipeline described in
    ``docs/implementation_design.md`` Section 16 without loading model
    weights or invoking any dynamic deserialization.
    """

    model_structure: dict[str, object] | None = None

    def analyze(
        self, artifact: Artifact, config: ScanConfig | None = None
    ) -> Sequence[Finding]:
        """Perform static analysis on a SafeTensors artifact.

        Parameters
        ----------
        artifact:
            The raw, SHA-256-fingerprinted SafeTensors artifact.
        config:
            Optional ScanConfig; uses defaults when not provided.

        Returns
        -------
        Sequence[Finding]
            All security findings, including structural issues and
            metadata security violations.
        """
        active_config = config or ScanConfig()
        data = self._read_artifact_bytes(artifact)
        file_size = len(data)
        findings: list[Finding] = []
        counter = [0]  # mutable counter for unique finding IDs

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
                format=ModelFormat.SAFETENSORS,
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
        # Stage 1: Header length unpack
        # ------------------------------------------------------------------
        max_header_bytes = active_config.max_safetensors_header_mb * 1024 * 1024

        try:
            header_len = unpack_header_length(data, max_header_bytes)
        except ResourceLimitError as exc:
            findings.append(
                _make_finding(
                    "ST201",
                    byte_offset=0,
                    matched_value=str(exc.actual_value or "unknown"),
                    description=str(exc),
                    raw_hex=data[:8].hex(),
                )
            )
            return findings
        except SafeTensorsAnalysisError as exc:
            findings.append(
                _make_finding(
                    "ST209",
                    byte_offset=0,
                    matched_value="TRUNCATED_HEADER",
                    description=str(exc),
                    raw_hex=data[:min(8, len(data))].hex(),
                )
            )
            return findings

        # ------------------------------------------------------------------
        # Stage 2: Extract JSON header bytes (truncation check)
        # ------------------------------------------------------------------
        try:
            raw_json = extract_header_bytes(data, header_len)
        except SafeTensorsAnalysisError as exc:
            findings.append(
                _make_finding(
                    "ST209",
                    byte_offset=0,
                    matched_value="TRUNCATED_HEADER",
                    description=str(exc),
                    raw_hex=data[:min(16, len(data))].hex(),
                )
            )
            return findings

        # ------------------------------------------------------------------
        # Stage 3: Lexical token / depth / duplicate-key guard
        # ------------------------------------------------------------------
        try:
            guard_result = scan_json_tokens(
                raw_json,
                max_tokens=active_config.max_json_tokens,
                max_depth=active_config.max_json_depth,
            )
        except ResourceLimitError as exc:
            findings.append(
                _make_finding(
                    "ST201",
                    byte_offset=8,
                    matched_value=exc.error_code,
                    description=str(exc),
                )
            )
            return findings
        except SafeTensorsAnalysisError as exc:
            findings.append(
                _make_finding(
                    "ST209",
                    byte_offset=8,
                    matched_value="INVALID_JSON_HEADER",
                    description=str(exc),
                )
            )
            return findings

        # Report ST207: duplicate top-level keys
        for dup_key in guard_result.duplicate_keys:
            findings.append(
                _make_finding(
                    "ST207",
                    byte_offset=8,
                    matched_value=dup_key,
                    description=(
                        f"Duplicate tensor name detected in SafeTensors header: {dup_key!r}"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # Stage 4: JSON parsing
        # ------------------------------------------------------------------
        try:
            header_dict: dict[str, object] = json.loads(raw_json)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            findings.append(
                _make_finding(
                    "ST209",
                    byte_offset=8,
                    matched_value="INVALID_JSON",
                    description=(
                        f"SafeTensors header is not valid JSON: {exc}"
                    ),
                    context_snippet=raw_json[:200].decode("utf-8", errors="replace"),
                )
            )
            return findings

        if not isinstance(header_dict, dict):
            findings.append(
                _make_finding(
                    "ST209",
                    byte_offset=8,
                    matched_value="NON_DICT_ROOT",
                    description=(
                        "SafeTensors JSON header root is not a JSON object "
                        f"(got {type(header_dict).__name__})"
                    ),
                )
            )
            return findings

        # ------------------------------------------------------------------
        # Stage 5: Tensor descriptor validation
        # ------------------------------------------------------------------
        # Data section size = file_size - 8 (prefix) - header_len
        data_section_size = max(0, file_size - 8 - header_len)

        valid_descriptors, tensor_issues = validate_tensors(header_dict, data_section_size)

        _tensor_rule_description: dict[str, str] = {
            "ST201": "Excessive tensor count in SafeTensors header",
            "ST202": "Tensor data regions overlap",
            "ST203": "Invalid tensor data_offsets",
            "ST204": "Invalid tensor dtype",
            "ST205": "Invalid tensor shape",
            "ST206": "Missing required tensor fields",
            "ST207": "Duplicate tensor key",
            "ST208": "Tensor range exceeds file data section",
        }

        for issue in tensor_issues:
            rule_id = issue.rule_id
            findings.append(
                _make_finding(
                    rule_id,
                    byte_offset=issue.byte_offset if issue.byte_offset is not None else 8,
                    matched_value=issue.tensor_name,
                    description=issue.detail,
                )
            )

        # ------------------------------------------------------------------
        # Stage 5b: Tensor name security scan (path traversal in tensor keys)
        # ------------------------------------------------------------------
        tensor_name_hits = scan_tensor_names(
            [k for k in header_dict if k != "__metadata__"]
        )
        for tnf in tensor_name_hits:
            findings.append(
                _make_finding(
                    "ST102",
                    byte_offset=8,
                    matched_value=tnf.matched_value,
                    description=(
                        f"SafeTensors tensor name {tnf.key!r} contains path traversal "
                        f"pattern {tnf.matched_pattern}"
                    ),
                    context_snippet=f"tensor_name={tnf.key!r}, pattern={tnf.matched_pattern}",
                )
            )

        # ------------------------------------------------------------------
        # Stage 5c: Weight Integrity Check (WI101/WI102)
        # ------------------------------------------------------------------
        from model_scanner.analyzers.safetensors.integrity import check_safetensors_integrity
        integrity_findings = check_safetensors_integrity(
            data=data,
            descriptors=valid_descriptors,
            artifact=artifact,
            header_len=header_len,
            counter_start=counter[0],
        )
        findings.extend(integrity_findings)
        counter[0] += len(integrity_findings)

        # Structural inventory from parsed descriptors (informational metadata)
        self.model_structure = {
            "tensor_count": len(valid_descriptors),
            "tensors": [
                {"name": d.name, "dtype": d.dtype, "shape": list(d.shape)}
                for d in valid_descriptors[:1000]
            ],
        }

        # ------------------------------------------------------------------
        # Stage 6: __metadata__ security pattern scan
        # ------------------------------------------------------------------
        raw_metadata = header_dict.get("__metadata__")
        if raw_metadata is not None:
            meta_findings = scan_metadata(raw_metadata)
            for mf in meta_findings:
                rule_id = mf.rule_id
                findings.append(
                    _make_finding(
                        rule_id,
                        byte_offset=8,  # Offset of JSON body within file
                        matched_value=mf.matched_value,
                        description=(
                            f"SafeTensors __metadata__[{mf.key!r}] contains "
                            f"{mf.matched_pattern}: {mf.matched_value[:100]!r}"
                        ),
                        context_snippet=f"key={mf.key!r}, pattern={mf.matched_pattern}",
                    )
                )

        return findings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            Raw SafeTensors file bytes.
        artifact_sha256:
            SHA-256 hex digest of the data (64 lowercase hex characters).
        config:
            Optional ScanConfig.

        Returns
        -------
        Sequence[Finding]
            All security findings.
        """
        import io

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


__all__ = ["SafeTensorsAnalyzer"]
