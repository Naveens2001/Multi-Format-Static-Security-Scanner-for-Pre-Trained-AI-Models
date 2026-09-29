"""Shared stdlib weight integrity checking — NaN/Inf/extreme detection from raw bytes."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Literal

from model_scanner.core.models import Confidence, Severity

# ---------------------------------------------------------------------------
# Shared WI101/WI102/WI103 rule metadata — single source of truth
# Used by keras/analyzer.py, gguf/integrity.py, and safetensors/integrity.py
# ---------------------------------------------------------------------------

WI_RULE_SEVERITY: dict[str, Severity] = {
    "WI101": Severity.HIGH, "WI102": Severity.HIGH, "WI103": Severity.MEDIUM,
}
WI_RULE_CONFIDENCE: dict[str, Confidence] = {
    "WI101": Confidence.HIGH, "WI102": Confidence.HIGH, "WI103": Confidence.MEDIUM,
}
WI_RULE_NAMES: dict[str, str] = {
    "WI101": "NaN Values Detected in Tensor Weights",
    "WI102": "Infinite Values Detected in Tensor Weights",
    "WI103": "Extreme Finite Values Detected in Tensor Weights",
}
WI_RULE_CATEGORY: dict[str, str] = {
    "WI101": "weight_integrity", "WI102": "weight_integrity", "WI103": "weight_integrity",
}
WI_RULE_REMEDIATION: dict[str, str] = {
    "WI101": "Re-acquire the model from a trusted source. NaN weights indicate corruption or tampering.",
    "WI102": "Re-acquire the model from a trusted source. Infinite weights indicate numerical corruption or tampering.",
    "WI103": (
        "Investigate the origin of extreme-magnitude weights. Values at the maximum finite "
        "representable magnitude suggest numerical instability, overflow during training, or tampering."
    ),
}
WI_KIND_TO_RULE: dict[str, str] = {"nan": "WI101", "inf": "WI102", "extreme": "WI103"}

_MAX_ELEMENTS: int = 2_000_000  # guard against scanning huge tensors

@dataclass(frozen=True)
class IntegrityHit:
    kind: Literal["nan", "inf", "extreme"]
    first_offset: int   # byte offset of first anomalous value in raw_bytes
    count: int          # total anomalous values found

def check_float32(raw_bytes: bytes) -> list[IntegrityHit]:
    """Check raw F32 bytes for NaN/Inf/extreme using IEEE 754 bit patterns.

    Exponent field is bits 30:23 (8 bits).
    - exp == 0xFF: NaN (mantissa != 0) or Inf (mantissa == 0)
    - exp == 0xFE: max finite exponent — |value| in [2^127, 2^128) ≈ [1.7e38, 3.4e38]
    """
    n = min(len(raw_bytes) // 4, _MAX_ELEMENTS)
    nan_count = inf_count = extreme_count = 0
    first_nan = first_inf = first_extreme = -1
    for i in range(n):
        (v,) = struct.unpack_from('<I', raw_bytes, i * 4)
        exp8 = (v >> 23) & 0xFF
        if exp8 == 0xFF:
            man = v & 0x7FFFFF
            if man != 0:
                if nan_count == 0: first_nan = i * 4
                nan_count += 1
            else:
                if inf_count == 0: first_inf = i * 4
                inf_count += 1
        elif exp8 == 0xFE:
            if extreme_count == 0: first_extreme = i * 4
            extreme_count += 1
    hits: list[IntegrityHit] = []
    if nan_count: hits.append(IntegrityHit("nan", first_nan, nan_count))
    if inf_count: hits.append(IntegrityHit("inf", first_inf, inf_count))
    if extreme_count: hits.append(IntegrityHit("extreme", first_extreme, extreme_count))
    return hits

def check_float16(raw_bytes: bytes) -> list[IntegrityHit]:
    """Check raw F16 bytes for NaN/Inf/extreme.

    Exponent field is bits 14:10 (5 bits).
    - exp5 == 0x1F: NaN or Inf
    - exp5 == 0x1E: max finite exponent — |value| in [2^15, 2^16) = [32768, 65536]
    """
    n = min(len(raw_bytes) // 2, _MAX_ELEMENTS)
    nan_count = inf_count = extreme_count = 0
    first_nan = first_inf = first_extreme = -1
    for i in range(n):
        (v,) = struct.unpack_from('<H', raw_bytes, i * 2)
        exp5 = (v >> 10) & 0x1F
        if exp5 == 0x1F:
            man = v & 0x03FF
            if man != 0:
                if nan_count == 0: first_nan = i * 2
                nan_count += 1
            else:
                if inf_count == 0: first_inf = i * 2
                inf_count += 1
        elif exp5 == 0x1E:
            if extreme_count == 0: first_extreme = i * 2
            extreme_count += 1
    hits: list[IntegrityHit] = []
    if nan_count: hits.append(IntegrityHit("nan", first_nan, nan_count))
    if inf_count: hits.append(IntegrityHit("inf", first_inf, inf_count))
    if extreme_count: hits.append(IntegrityHit("extreme", first_extreme, extreme_count))
    return hits

def check_bfloat16(raw_bytes: bytes) -> list[IntegrityHit]:
    """Check raw BF16 bytes for NaN/Inf/extreme.

    BF16 shares the 8-bit exponent of F32. Bits 14:7 (8 bits).
    - exp8 == 0xFF: NaN or Inf
    - exp8 == 0xFE: max finite exponent — |value| in [2^127, 2^128) ≈ [1.7e38, 3.4e38]
    """
    n = min(len(raw_bytes) // 2, _MAX_ELEMENTS)
    nan_count = inf_count = extreme_count = 0
    first_nan = first_inf = first_extreme = -1
    for i in range(n):
        (v,) = struct.unpack_from('<H', raw_bytes, i * 2)
        exp8 = (v >> 7) & 0xFF
        if exp8 == 0xFF:
            man = v & 0x7F
            if man != 0:
                if nan_count == 0: first_nan = i * 2
                nan_count += 1
            else:
                if inf_count == 0: first_inf = i * 2
                inf_count += 1
        elif exp8 == 0xFE:
            if extreme_count == 0: first_extreme = i * 2
            extreme_count += 1
    hits: list[IntegrityHit] = []
    if nan_count: hits.append(IntegrityHit("nan", first_nan, nan_count))
    if inf_count: hits.append(IntegrityHit("inf", first_inf, inf_count))
    if extreme_count: hits.append(IntegrityHit("extreme", first_extreme, extreme_count))
    return hits

def check_float64(raw_bytes: bytes) -> list[IntegrityHit]:
    """Check raw F64 bytes for NaN/Inf/extreme.

    Exponent field is bits 62:52 (11 bits).
    - exp11 == 0x7FF: NaN or Inf
    - exp11 == 0x7FE: max finite exponent — |value| in [2^1023, 2^1024) ≈ [8.9e307, 1.8e308]
    """
    n = min(len(raw_bytes) // 8, _MAX_ELEMENTS)
    nan_count = inf_count = extreme_count = 0
    first_nan = first_inf = first_extreme = -1
    for i in range(n):
        (v,) = struct.unpack_from('<Q', raw_bytes, i * 8)
        exp11 = (v >> 52) & 0x7FF
        if exp11 == 0x7FF:
            man = v & 0x000FFFFFFFFFFFFF
            if man != 0:
                if nan_count == 0: first_nan = i * 8
                nan_count += 1
            else:
                if inf_count == 0: first_inf = i * 8
                inf_count += 1
        elif exp11 == 0x7FE:
            if extreme_count == 0: first_extreme = i * 8
            extreme_count += 1
    hits: list[IntegrityHit] = []
    if nan_count: hits.append(IntegrityHit("nan", first_nan, nan_count))
    if inf_count: hits.append(IntegrityHit("inf", first_inf, inf_count))
    if extreme_count: hits.append(IntegrityHit("extreme", first_extreme, extreme_count))
    return hits

def check_tensor(raw_bytes: bytes, dtype: str) -> list[IntegrityHit]:
    """Dispatch to dtype-specific checker. Returns [] for non-float dtypes."""
    if dtype in ("F32",):
        return check_float32(raw_bytes)
    if dtype in ("F16",):
        return check_float16(raw_bytes)
    if dtype in ("BF16",):
        return check_bfloat16(raw_bytes)
    if dtype in ("F64",):
        return check_float64(raw_bytes)
    return []

__all__ = [
    "IntegrityHit", "check_tensor", "check_float32", "check_float16", "check_bfloat16", "check_float64",
    "WI_RULE_SEVERITY", "WI_RULE_CONFIDENCE", "WI_RULE_NAMES", "WI_RULE_CATEGORY",
    "WI_RULE_REMEDIATION", "WI_KIND_TO_RULE",
]
