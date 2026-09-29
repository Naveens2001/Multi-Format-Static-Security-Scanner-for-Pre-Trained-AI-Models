"""Weight integrity analysis for GGUF tensor data."""
from __future__ import annotations
from model_scanner.analyzers._weight_utils import (
    check_tensor,
    WI_RULE_SEVERITY,
    WI_RULE_CONFIDENCE,
    WI_RULE_NAMES,
    WI_RULE_CATEGORY,
    WI_RULE_REMEDIATION,
    WI_KIND_TO_RULE,
)
from model_scanner.analyzers.gguf.tensors import GGUFTensorDescriptor
from model_scanner.core.models import (
    Artifact, CoordinateSpace, Evidence, Finding, ModelFormat,
)

_FLOAT_GGML_TYPES: dict[int, str] = {0: "F32", 1: "F16", 28: "F64", 30: "BF16"}

# Local aliases for backward compatibility — pointing to shared constants
_RULE_SEVERITY = WI_RULE_SEVERITY
_RULE_CONFIDENCE = WI_RULE_CONFIDENCE
_RULE_NAMES = WI_RULE_NAMES
_RULE_CATEGORY = WI_RULE_CATEGORY
_RULE_REMEDIATION = WI_RULE_REMEDIATION
_KIND_TO_RULE = WI_KIND_TO_RULE


def check_gguf_integrity(
    data: bytes,
    descriptors: list[GGUFTensorDescriptor],
    data_section_start: int,
    artifact: Artifact,
    counter_start: int = 0,
) -> list[Finding]:
    """Check GGUF float tensor data for NaN/Inf/extreme-finite values."""
    findings: list[Finding] = []
    counter = [counter_start]

    def _next_id(rule_id: str) -> str:
        counter[0] += 1
        return f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}"

    for desc in descriptors:
        dtype_str = _FLOAT_GGML_TYPES.get(desc.tensor_type)
        if dtype_str is None or desc.size_bytes == 0:
            continue

        abs_start = data_section_start + desc.offset
        abs_end = abs_start + desc.size_bytes
        if abs_end > len(data) or abs_start >= abs_end:
            continue

        tensor_bytes = data[abs_start:abs_end]
        hits = check_tensor(tensor_bytes, dtype_str)

        for hit in hits:
            rule_id = _KIND_TO_RULE[hit.kind]
            findings.append(Finding(
                finding_id=_next_id(rule_id),
                rule_id=rule_id,
                rule_name=_RULE_NAMES[rule_id],
                severity=_RULE_SEVERITY[rule_id],
                confidence=_RULE_CONFIDENCE[rule_id],
                category=_RULE_CATEGORY[rule_id],
                format=ModelFormat.GGUF,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARTIFACT,
                    byte_offset=abs_start + hit.first_offset,
                    byte_length=None,
                    archive_member=None,
                    matched_value=f"tensor={desc.name!r} dtype={dtype_str} {hit.kind}_count={hit.count}",
                    context_snippet=f"Tensor '{desc.name}' ({dtype_str}) contains {hit.count} {hit.kind.upper()} values",
                ),
                description=(
                    f"GGUF tensor '{desc.name}' ({dtype_str}) contains {hit.count} {hit.kind.upper()} "
                    f"value(s) at byte offset {abs_start + hit.first_offset}."
                ),
                remediation=_RULE_REMEDIATION[rule_id],
            ))

    return findings


__all__ = ["check_gguf_integrity", "_FLOAT_GGML_TYPES"]
