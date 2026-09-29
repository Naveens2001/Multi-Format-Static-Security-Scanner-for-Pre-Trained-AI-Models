"""
model_scanner.analyzers.gguf.tensors
====================================

Tensor descriptor parser and structural integrity validator for GGUF model files.

Verifications Performed
-----------------------
- Tensor name uniqueness, format, and encoding (GU205)
- Tensor dimension count and non-zero / non-overflow dimension sizes (GU206)
- GGML tensor type validation against canonical types (GU207)
- Offset non-negativity and alignment checks (GU208)
- Tensor data range within file bounds (GU209)
- Overlapping tensor byte range detection (GU210)
- Truncated / malformed descriptor streams (GU203)

SECURITY GUARANTEE
------------------
- Pure standard library struct/byte parsing only.
- Computes tensor byte sizes algebraically from dimensions and block sizes;
  never allocates memory for tensor weights.
- Never calls eval(), exec(), pickle.loads(), or torch.load().

Design Reference
----------------
Section 17 of docs/implementation_design.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

# ---------------------------------------------------------------------------
# GGML Tensor Types & Block Dimensions
# ---------------------------------------------------------------------------


class GGMLType(IntEnum):
    """Canonical GGML quantization and floating-point types."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q4_2 = 4  # obsolete
    Q4_3 = 5  # obsolete
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    Q4_0_4_4 = 31
    Q4_0_4_8 = 32
    Q4_0_8_8 = 33
    TQ1_0 = 34
    TQ2_0 = 35


# Mapping of GGMLType -> (block_size_in_elements, type_size_in_bytes)
_GGML_TYPE_INFO: dict[int, tuple[int, int]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q4_2: (32, 20),
    GGMLType.Q4_3: (32, 28),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q8_1: (32, 36),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
    GGMLType.Q8_K: (256, 292),
    GGMLType.IQ2_XXS: (256, 66),
    GGMLType.IQ2_XS: (256, 74),
    GGMLType.IQ3_XXS: (256, 98),
    GGMLType.IQ1_S: (256, 50),
    GGMLType.IQ4_NL: (32, 18),
    GGMLType.IQ3_S: (256, 110),
    GGMLType.IQ2_S: (256, 82),
    GGMLType.IQ4_XS: (256, 136),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (1, 2),
    GGMLType.I32: (1, 4),
    GGMLType.I64: (1, 8),
    GGMLType.F64: (1, 8),
    GGMLType.IQ1_M: (256, 56),
    GGMLType.BF16: (1, 2),
    GGMLType.Q4_0_4_4: (32, 18),
    GGMLType.Q4_0_4_8: (32, 18),
    GGMLType.Q4_0_8_8: (32, 18),
    GGMLType.TQ1_0: (256, 36),
    GGMLType.TQ2_0: (256, 68),
}

MAX_TENSOR_DIMENSIONS: int = 16
DEFAULT_GGUF_ALIGNMENT: int = 32


@dataclass(frozen=True)
class GGUFTensorDescriptor:
    """Parsed tensor metadata descriptor."""

    name: str
    n_dimensions: int
    dimensions: tuple[int, ...]
    tensor_type: int
    offset: int
    size_bytes: int
    byte_offset: int  # File offset where descriptor begins


@dataclass(frozen=True)
class GGUFTensorIssue:
    """A structural anomaly discovered in tensor metadata."""

    rule_id: str
    byte_offset: int
    tensor_name: str
    detail: str


def compute_tensor_byte_size(tensor_type: int, dimensions: tuple[int, ...]) -> int:
    """Calculate the byte size of a tensor from its type and shape.

    Returns
    -------
    int
        Calculated size in bytes (or 0 for empty/scalar with 0 elements).

    Raises
    -------
    ValueError
        If tensor type is unknown, dimensions are negative, or arithmetic overflows.
    """
    if tensor_type not in _GGML_TYPE_INFO:
        raise ValueError(f"Unknown GGML tensor type: {tensor_type}")

    if not dimensions:
        # Scalar tensor (0 dims) -> 1 element
        num_elements = 1
    else:
        num_elements = 1
        for d in dimensions:
            if d < 0:
                raise ValueError(f"Negative dimension size: {d}")
            if d == 0:
                return 0
            num_elements *= d
            if num_elements > 10**18:
                raise OverflowError(f"Tensor element count overflow: {num_elements}")

    block_size, type_size = _GGML_TYPE_INFO[tensor_type]
    # Total blocks = ceil(num_elements / block_size)
    num_blocks = (num_elements + block_size - 1) // block_size
    return num_blocks * type_size


def parse_and_validate_tensors(
    data: bytes,
    start_offset: int,
    tensor_count: int,
    version: int,
    alignment: int = DEFAULT_GGUF_ALIGNMENT,
) -> tuple[list[GGUFTensorDescriptor], list[GGUFTensorIssue], int]:
    """Parse tensor descriptors and validate structural integrity.

    Parameters
    ----------
    data:
        Full raw file bytes.
    start_offset:
        Offset in data where tensor descriptors begin.
    tensor_count:
        Declared number of tensor descriptors.
    version:
        GGUF format version (1, 2, or 3).
    alignment:
        Declared alignment boundary in bytes (e.g. from general.alignment).

    Returns
    -------
    tuple[list[GGUFTensorDescriptor], list[GGUFTensorIssue], int]
        (descriptors, issues, data_section_start_offset)
    """
    total_size = len(data)
    descriptors: list[GGUFTensorDescriptor] = []
    issues: list[GGUFTensorIssue] = []
    seen_names: set[str] = set()

    # Validate alignment
    if alignment <= 0 or (alignment & (alignment - 1)) != 0 or alignment > 65_536:
        issues.append(
            GGUFTensorIssue(
                rule_id="GU208",
                byte_offset=start_offset,
                tensor_name="",
                detail=(
                    f"Invalid GGUF alignment value: {alignment} (must be a positive "
                    f"power of 2 up to 64 KiB)"
                ),
            )
        )
        alignment = DEFAULT_GGUF_ALIGNMENT

    # Check for excessive tensor count
    if tensor_count > 100_000:
        issues.append(
            GGUFTensorIssue(
                rule_id="GU201",
                byte_offset=start_offset,
                tensor_name="",
                detail=f"GGUF tensor count ({tensor_count}) exceeds maximum limit (100,000)",
            )
        )
        return descriptors, issues, start_offset

    offset = start_offset
    len_size = 4 if version == 1 else 8
    len_fmt = "<I" if version == 1 else "<Q"
    dim_fmt = "<I" if version == 1 else "<Q"
    dim_size = 4 if version == 1 else 8

    for i in range(tensor_count):
        desc_start = offset

        # 1. Read tensor name
        if offset + len_size > total_size:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    tensor_name=f"tensor_{i}",
                    detail=f"File truncated before tensor {i} name length",
                )
            )
            break

        (name_len,) = struct.unpack_from(len_fmt, data, offset)
        offset += len_size

        if offset + name_len > total_size:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    tensor_name=f"tensor_{i}",
                    detail=(
                        f"Tensor {i} name length ({name_len} bytes) extends beyond end of file"
                    ),
                )
            )
            break

        raw_name = data[offset : offset + name_len]
        offset += name_len
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            name = raw_name.decode("utf-8", errors="replace")
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU205",
                    byte_offset=desc_start,
                    tensor_name=name,
                    detail=f"Tensor {i} name is not valid UTF-8",
                )
            )

        if not name:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU205",
                    byte_offset=desc_start,
                    tensor_name="",
                    detail=f"Tensor {i} has empty name",
                )
            )

        if name in seen_names:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU205",
                    byte_offset=desc_start,
                    tensor_name=name,
                    detail=f"Duplicate tensor name detected: {name!r}",
                )
            )
        seen_names.add(name)

        # 2. Read n_dimensions
        if offset + 4 > total_size:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    tensor_name=name,
                    detail=f"File truncated before tensor {name!r} n_dimensions",
                )
            )
            break

        (n_dims,) = struct.unpack_from("<I", data, offset)
        offset += 4

        if n_dims > MAX_TENSOR_DIMENSIONS:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU206",
                    byte_offset=offset - 4,
                    tensor_name=name,
                    detail=(
                        f"Tensor {name!r} dimension count ({n_dims}) exceeds limit "
                        f"({MAX_TENSOR_DIMENSIONS})"
                    ),
                )
            )
            # Cannot safely unpack dimensions
            break

        # 3. Read dimensions
        if offset + n_dims * dim_size > total_size:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    tensor_name=name,
                    detail=f"File truncated before tensor {name!r} dimensions",
                )
            )
            break

        dims_list: list[int] = []
        invalid_dim = False
        for _ in range(n_dims):
            (d,) = struct.unpack_from(dim_fmt, data, offset)
            offset += dim_size
            dims_list.append(d)
            if d < 0:
                invalid_dim = True

        if invalid_dim:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU206",
                    byte_offset=offset - n_dims * dim_size,
                    tensor_name=name,
                    detail=f"Tensor {name!r} contains negative dimension in {dims_list}",
                )
            )

        dims = tuple(dims_list)

        # 4. Read tensor type
        if offset + 4 > total_size:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    tensor_name=name,
                    detail=f"File truncated before tensor {name!r} type",
                )
            )
            break

        (t_type,) = struct.unpack_from("<I", data, offset)
        offset += 4

        if t_type not in _GGML_TYPE_INFO:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU207",
                    byte_offset=offset - 4,
                    tensor_name=name,
                    detail=f"Tensor {name!r} has invalid GGML tensor type: {t_type}",
                )
            )

        # 5. Read tensor offset
        if offset + 8 > total_size:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU203",
                    byte_offset=offset,
                    tensor_name=name,
                    detail=f"File truncated before tensor {name!r} offset",
                )
            )
            break

        (t_offset,) = struct.unpack_from("<Q", data, offset)
        offset += 8

        if t_offset < 0:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU208",
                    byte_offset=offset - 8,
                    tensor_name=name,
                    detail=f"Tensor {name!r} has negative offset: {t_offset}",
                )
            )

        # Calculate byte size
        try:
            size_bytes = compute_tensor_byte_size(t_type, dims) if t_type in _GGML_TYPE_INFO else 0
        except (ValueError, OverflowError) as exc:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU206",
                    byte_offset=desc_start,
                    tensor_name=name,
                    detail=f"Invalid shape/size for tensor {name!r}: {exc}",
                )
            )
            size_bytes = 0

        desc = GGUFTensorDescriptor(
            name=name,
            n_dimensions=n_dims,
            dimensions=dims,
            tensor_type=t_type,
            offset=t_offset,
            size_bytes=size_bytes,
            byte_offset=desc_start,
        )
        descriptors.append(desc)

    # ----------------------------------------------------------------------
    # Compute Data Section Alignment & Bounds
    # ----------------------------------------------------------------------
    pad = (alignment - (offset % alignment)) % alignment
    data_section_start = offset + pad

    # Check bounds and overlaps
    _validate_tensor_spans_and_overlaps(
        descriptors, data_section_start, total_size, issues
    )

    return descriptors, issues, data_section_start


def _validate_tensor_spans_and_overlaps(
    descriptors: list[GGUFTensorDescriptor],
    data_section_start: int,
    total_size: int,
    issues: list[GGUFTensorIssue],
) -> None:
    """Check tensor file bounds (GU209) and interval overlaps (GU210)."""
    # 1. Bounds check
    for desc in descriptors:
        if desc.size_bytes == 0:
            continue
        tensor_end = data_section_start + desc.offset + desc.size_bytes
        if tensor_end > total_size:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU209",
                    byte_offset=desc.byte_offset,
                    tensor_name=desc.name,
                    detail=(
                        f"Tensor {desc.name!r} range [{data_section_start + desc.offset}.."
                        f"{tensor_end}] extends beyond end of file ({total_size} bytes)"
                    ),
                )
            )

    # 2. Overlap check
    # Filter non-empty tensors and sort by offset
    valid_tensors = [d for d in descriptors if d.size_bytes > 0]
    valid_tensors.sort(key=lambda d: d.offset)

    for i in range(len(valid_tensors) - 1):
        curr = valid_tensors[i]
        nxt = valid_tensors[i + 1]
        curr_end = curr.offset + curr.size_bytes
        if curr_end > nxt.offset:
            issues.append(
                GGUFTensorIssue(
                    rule_id="GU210",
                    byte_offset=nxt.byte_offset,
                    tensor_name=f"{curr.name} & {nxt.name}",
                    detail=(
                        f"Tensor range overlap detected: {curr.name!r} "
                        f"[offset {curr.offset}..{curr_end}] overlaps {nxt.name!r} "
                        f"[offset {nxt.offset}..{nxt.offset + nxt.size_bytes}]"
                    ),
                )
            )


__all__ = [
    "DEFAULT_GGUF_ALIGNMENT",
    "MAX_TENSOR_DIMENSIONS",
    "GGMLType",
    "GGUFTensorDescriptor",
    "GGUFTensorIssue",
    "compute_tensor_byte_size",
    "parse_and_validate_tensors",
]
