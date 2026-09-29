"""Weight integrity analysis for SafeTensors tensor data."""
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
from model_scanner.analyzers.safetensors.tensors import TensorDescriptor
from model_scanner.core.models import (
    Artifact, CoordinateSpace, Evidence, Finding, ModelFormat,
)

# Local aliases pointing to shared constants (Finding 7)
_RULE_SEVERITY = WI_RULE_SEVERITY
_RULE_CONFIDENCE = WI_RULE_CONFIDENCE
_RULE_NAMES = WI_RULE_NAMES
_RULE_CATEGORY = WI_RULE_CATEGORY
_RULE_REMEDIATION = WI_RULE_REMEDIATION
_KIND_TO_RULE = WI_KIND_TO_RULE


def check_safetensors_integrity(
    data: bytes,
    descriptors: list[TensorDescriptor],
    artifact: Artifact,
    header_len: int,
    counter_start: int = 0,
) -> list[Finding]:
    """Check SafeTensors tensor data for NaN/Inf/extreme-finite values."""
    findings: list[Finding] = []
    counter = [counter_start]

    def _next_id(rule_id: str) -> str:
        counter[0] += 1
        return f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}"

    data_section_start = 8 + header_len

    for desc in descriptors:
        if desc.dtype not in ("F32", "F16", "BF16", "F64"):
            continue
        abs_start = data_section_start + desc.data_start
        abs_end = data_section_start + desc.data_end
        if abs_end > len(data) or abs_start >= abs_end:
            continue

        tensor_bytes = data[abs_start:abs_end]
        hits = check_tensor(tensor_bytes, desc.dtype)

        for hit in hits:
            rule_id = _KIND_TO_RULE[hit.kind]
            findings.append(Finding(
                finding_id=_next_id(rule_id),
                rule_id=rule_id,
                rule_name=_RULE_NAMES[rule_id],
                severity=_RULE_SEVERITY[rule_id],
                confidence=_RULE_CONFIDENCE[rule_id],
                category=_RULE_CATEGORY[rule_id],
                format=ModelFormat.SAFETENSORS,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARTIFACT,
                    byte_offset=abs_start + hit.first_offset,
                    byte_length=None,
                    archive_member=None,
                    matched_value=f"tensor={desc.name!r} dtype={desc.dtype} {hit.kind}_count={hit.count}",
                    context_snippet=f"Tensor '{desc.name}' ({desc.dtype}) contains {hit.count} {hit.kind.upper()} values",
                ),
                description=(
                    f"Tensor '{desc.name}' ({desc.dtype}) contains {hit.count} {hit.kind.upper()} "
                    f"value(s) at byte offset {abs_start + hit.first_offset}."
                ),
                remediation=_RULE_REMEDIATION[rule_id],
            ))

    return findings


__all__ = ["check_safetensors_integrity"]
