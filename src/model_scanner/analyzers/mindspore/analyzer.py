"""
model_scanner.analyzers.mindspore.analyzer
==========================================

Static security analyzer for MindSpore protobuf checkpoint files (.ckpt).

SECURITY GUARANTEE
------------------
- Never imports mindspore, protobuf runtime, or any ML library.
- Never calls mindspore.load_checkpoint() or any deserializer.
- Never allocates memory proportional to declared tensor dimensions.
- All dimension arithmetic uses Python arbitrary-precision integers.
- Parsing is bounded (see proto_reader.py for hard limits).

Design
------
The MindSpore .ckpt format is a raw protobuf serialization of
``mindspore_pb2.Checkpoint``.  Each ``Parameter`` message carries:
  - a tensor name (string)
  - a TensorProto with declared ``dims``, ``data_type``, and ``tensor_content``

Rule MS101 fires when the declared allocation (product(dims) * bytes/elem)
significantly exceeds the actual tensor data present in the file.  This is
the CWE-789 ``Uncontrolled Memory Allocation`` pattern: a tiny checkpoint
file claims an enormous tensor shape, causing mindspore.load_checkpoint() to
attempt a multi-hundred-GB allocation when executed.

Thresholds
----------
HIGH severity  : declared_bytes >=  10 GB  AND  sparsity < 1e-6
HIGH severity  : declared_bytes >= 100 GB  (regardless of sparsity)
MEDIUM severity: declared_bytes >=   1 GB  AND  sparsity < 1e-6
"""

from __future__ import annotations

from collections.abc import Sequence

from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.analyzers.mindspore.proto_reader import (
    MalformedProtoError,
    TensorMeta,
    parse_checkpoint_metadata,
)
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

# ---------------------------------------------------------------------------
# Rule metadata (MS101)
# ---------------------------------------------------------------------------

_RULE_ID = "MS101"
_RULE_NAME = "Excessive Memory Allocation in MindSpore Checkpoint"
_RULE_CATEGORY = "structural_anomaly"
_RULE_REMEDIATION = (
    "Do not load this checkpoint with mindspore.load_checkpoint(). "
    "Re-acquire model weights from a trusted source and verify the declared "
    "tensor shapes match the actual data present in the file. "
    "Convert to SafeTensors format for format-verified supply chains."
)

# ---------------------------------------------------------------------------
# Allocation thresholds (bytes)
# ---------------------------------------------------------------------------

_1_GB: int = 1 * 1024 ** 3       # 1 073 741 824 bytes
_10_GB: int = 10 * 1024 ** 3     # 10 737 418 240 bytes
_100_GB: int = 100 * 1024 ** 3   # 107 374 182 400 bytes

# Fraction of declared data that must actually be present to suppress the rule.
# A ratio below this value (e.g. 1 byte out of 400 GB) indicates a shape bomb.
_SPARSITY_THRESHOLD: float = 1e-6   # less than 1 ppm present → suspicious


def _severity_and_confidence(
    declared_bytes: int, actual_data_bytes: int
) -> tuple[Severity, Confidence] | None:
    """Return (Severity, Confidence) if MS101 should fire, else None."""
    if declared_bytes < _1_GB:
        return None

    # Compute sparsity: fraction of declared space actually stored
    sparsity = actual_data_bytes / declared_bytes if declared_bytes > 0 else 0.0

    if declared_bytes >= _100_GB:
        # Any checkpoint this size with no real data is clearly an allocation bomb
        return Severity.HIGH, Confidence.HIGH

    if declared_bytes >= _10_GB and sparsity < _SPARSITY_THRESHOLD:
        return Severity.HIGH, Confidence.HIGH

    if declared_bytes >= _1_GB and sparsity < _SPARSITY_THRESHOLD:
        return Severity.MEDIUM, Confidence.HIGH

    return None


def _human_bytes(n: int) -> str:
    """Return a human-readable byte count string."""
    for unit, divisor in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= divisor:
            return f"{n / divisor:.1f} {unit}"
    return f"{n} B"


class MindSporeAnalyzer(BaseFormatAnalyzer):
    """Static security analyzer for MindSpore protobuf checkpoint files.

    Parses the protobuf wire format without importing mindspore or any ML
    library.  Emits MS101 findings for CWE-789 allocation bombs — checkpoints
    that declare enormous tensor shapes with minimal actual data.
    """

    model_structure: dict[str, object] | None = None

    def analyze(
        self, artifact: Artifact, config: ScanConfig | None = None
    ) -> Sequence[Finding]:
        """Perform static analysis on a MindSpore checkpoint artifact."""
        data = self._read_artifact_bytes(artifact)
        if not data:
            return []

        try:
            tensor_metas = parse_checkpoint_metadata(data)
        except MalformedProtoError:
            # Structurally invalid — format detector already confirmed protobuf
            # structure from the header; a parse failure deeper in means the
            # bulk of the file is corrupted.  Return empty: P201 is a PyTorch
            # concept; a separate malformed-checkpoint rule could be added later.
            return []
        except Exception:
            return []

        findings: list[Finding] = []
        counter = 0

        # Build structural inventory for report_model
        tensor_inventory: list[dict[str, object]] = []

        for meta in tensor_metas:
            tensor_inventory.append(
                {
                    "name": meta.name,
                    "dtype": meta.data_type,
                    "shape": meta.dims,
                    "declared_bytes": meta.declared_allocation_bytes,
                    "actual_bytes": meta.actual_data_bytes,
                }
            )

            result = _severity_and_confidence(
                meta.declared_allocation_bytes, meta.actual_data_bytes
            )
            if result is None:
                continue

            severity, confidence = result
            counter += 1

            declared_str = _human_bytes(meta.declared_allocation_bytes)
            actual_str = _human_bytes(meta.actual_data_bytes)
            dims_str = str(meta.dims)

            matched_value = (
                f"tensor={meta.name!r} dims={dims_str} dtype={meta.data_type} "
                f"declared={declared_str} actual={actual_str}"
            )[:512]

            description = (
                f"MindSpore checkpoint parameter '{meta.name}' declares tensor "
                f"shape {dims_str} ({meta.data_type}), which requires "
                f"{declared_str} to load, but the file contains only "
                f"{actual_str} of actual tensor data. "
                f"Calling mindspore.load_checkpoint() on this file will attempt "
                f"to allocate {declared_str} of memory, causing a system crash "
                f"or denial of service (CWE-789: Uncontrolled Memory Allocation)."
            )

            finding = Finding(
                finding_id=(
                    f"FINDING-{_RULE_ID}-{artifact.sha256[:8]}-{counter:03d}"
                ),
                rule_id=_RULE_ID,
                rule_name=_RULE_NAME,
                severity=severity,
                confidence=confidence,
                category=_RULE_CATEGORY,
                format=ModelFormat.MINDSPORE_CKPT,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARTIFACT,
                    byte_offset=meta.tensor_field_offset,
                    byte_length=meta.tensor_field_length,
                    archive_member=None,
                    matched_value=matched_value,
                    raw_hex=None,
                    context_snippet=(
                        f"Declared allocation: {declared_str} | "
                        f"Actual data in file: {actual_str} | "
                        f"Dims: {dims_str} | "
                        f"Dtype: {meta.data_type}"
                    ),
                ),
                description=description,
                remediation=_RULE_REMEDIATION,
                cve_ids=(),
                cwe_ids=("CWE-789",),
                owasp_category="ML06:2023 AI Supply Chain Attacks",
                mitre_technique="T1499: Endpoint Denial of Service",
            )
            findings.append(finding)

        self.model_structure = {
            "tensor_count": len(tensor_metas),
            "tensors": tensor_inventory,
        }

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


__all__ = ["MindSporeAnalyzer"]
