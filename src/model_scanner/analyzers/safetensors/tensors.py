"""
model_scanner.analyzers.safetensors.tensors
============================================

Tensor descriptor validation for SafeTensors files.

After the JSON header has been parsed, this module inspects each tensor
descriptor dict to verify:

- All required fields are present (``dtype``, ``shape``, ``data_offsets``).
- ``dtype`` is in the canonical SafeTensors type set.
- ``shape`` contains only non-negative integers.
- ``data_offsets`` is a two-element list ``[start, end]`` with
  ``0 <= start <= end`` (zero-length spans are permitted for scalars
  where start == end).
- No tensor region extends beyond the declared data section boundary.
- No two tensor regions overlap.

SECURITY GUARANTEE
------------------
- No tensor data is loaded or deserialized.
- Arithmetic operations use Python int exclusively (no numpy, struct, etc.).
- Never calls pickle.loads(), torch.load(), eval(), or exec().
- Does not import torch, safetensors, or any ML framework.

Design Reference
----------------
Section 16.3 of docs/implementation_design.md
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical SafeTensors data-type strings (format spec v1.0)
SAFETENSORS_VALID_DTYPES: frozenset[str] = frozenset(
    [
        "F64",   # 64-bit float
        "F32",   # 32-bit float
        "F16",   # 16-bit float (half)
        "BF16",  # Brain float 16
        "F8_E5M2",  # 8-bit float (E5M2, spec >=1.0.0)
        "F8_E4M3",  # 8-bit float (E4M3, spec >=1.0.0)
        "I64",   # 64-bit signed int
        "I32",   # 32-bit signed int
        "I16",   # 16-bit signed int
        "I8",    # 8-bit signed int
        "U8",    # 8-bit unsigned int
        "BOOL",  # Boolean
    ]
)

# Hard sentinel: prevent iteration over pathologically large tensor counts
_MAX_TENSOR_COUNT: int = 50_000


@dataclass(frozen=True)
class TensorDescriptor:
    """Parsed tensor metadata from the SafeTensors JSON header.

    Attributes
    ----------
    name : str
        Tensor name (the JSON key).
    dtype : str
        Data type string, e.g. ``"F32"``.
    shape : list[int]
        Dimension list, e.g. ``[768, 3072]``.  Empty list ``[]`` = scalar.
    data_start : int
        Start byte offset within the data section (after header).
    data_end : int
        End byte offset within the data section (exclusive).
    """

    name: str
    dtype: str
    shape: list[int]
    data_start: int
    data_end: int


@dataclass(frozen=True)
class TensorValidationIssue:
    """A single tensor validation problem found during inspection.

    Attributes
    ----------
    rule_id : str
        One of ``ST203``, ``ST204``, ``ST205``, ``ST206``, ``ST208``.
    tensor_name : str
        The affected tensor name.
    detail : str
        Human-readable description of the problem.
    byte_offset : int | None
        Byte offset in the header (relative to start of JSON) where the
        offending data_offsets field begins, if determinable.
    """

    rule_id: str
    tensor_name: str
    detail: str
    byte_offset: int | None = None


def validate_tensors(
    header_dict: dict[str, object],
    data_section_size: int,
) -> tuple[list[TensorDescriptor], list[TensorValidationIssue]]:
    """Validate all tensor descriptors in the parsed JSON header.

    Parameters
    ----------
    header_dict:
        The parsed JSON dict (output of ``json.loads()``).
        The ``__metadata__`` key is skipped automatically.
    data_section_size:
        Total byte size of the data section that follows the header.
        Computed as ``file_size - 8 - header_len``.

    Returns
    -------
    tuple[list[TensorDescriptor], list[TensorValidationIssue]]
        A tuple of (valid descriptors, issues).  Issues are emitted for
        every problem found; validation continues through all tensors.
    """
    descriptors: list[TensorDescriptor] = []
    issues: list[TensorValidationIssue] = []

    tensor_count = 0

    for name, value in header_dict.items():
        # Skip the reserved metadata key
        if name == "__metadata__":
            continue

        tensor_count += 1
        if tensor_count > _MAX_TENSOR_COUNT:
            issues.append(
                TensorValidationIssue(
                    rule_id="ST201",
                    tensor_name=name,
                    detail=(
                        f"Tensor count exceeds sentinel limit of {_MAX_TENSOR_COUNT:,}. "
                        "Further tensor descriptors are not validated."
                    ),
                )
            )
            break

        if not isinstance(value, dict):
            issues.append(
                TensorValidationIssue(
                    rule_id="ST206",
                    tensor_name=name,
                    detail=(
                        f"Tensor descriptor for {name!r} is not a dict "
                        f"(got {type(value).__name__})"
                    ),
                )
            )
            continue

        tensor_dict: dict[str, object] = value  # isinstance check narrows above

        # ------------------------------------------------------------------
        # ST206: Required field presence
        # ------------------------------------------------------------------
        missing = [f for f in ("dtype", "shape", "data_offsets") if f not in tensor_dict]
        if missing:
            issues.append(
                TensorValidationIssue(
                    rule_id="ST206",
                    tensor_name=name,
                    detail=f"Tensor {name!r} missing required fields: {missing}",
                )
            )
            continue  # Cannot proceed without basic structure

        dtype = tensor_dict["dtype"]
        shape = tensor_dict["shape"]
        data_offsets = tensor_dict["data_offsets"]

        # ------------------------------------------------------------------
        # ST204: dtype validation
        # ------------------------------------------------------------------
        if not isinstance(dtype, str):
            issues.append(
                TensorValidationIssue(
                    rule_id="ST204",
                    tensor_name=name,
                    detail=f"Tensor {name!r} dtype is not a string: {dtype!r}",
                )
            )
        elif dtype not in SAFETENSORS_VALID_DTYPES:
            issues.append(
                TensorValidationIssue(
                    rule_id="ST204",
                    tensor_name=name,
                    detail=(
                        f"Tensor {name!r} has unknown/invalid dtype {dtype!r}. "
                        f"Valid types: {sorted(SAFETENSORS_VALID_DTYPES)}"
                    ),
                )
            )

        # ------------------------------------------------------------------
        # ST205: shape validation
        # ------------------------------------------------------------------
        shape_ok = True
        if not isinstance(shape, list):
            issues.append(
                TensorValidationIssue(
                    rule_id="ST205",
                    tensor_name=name,
                    detail=f"Tensor {name!r} shape is not a list: {shape!r}",
                )
            )
            shape_ok = False
        else:
            for i, dim in enumerate(shape):
                if not isinstance(dim, int):
                    issues.append(
                        TensorValidationIssue(
                            rule_id="ST205",
                            tensor_name=name,
                            detail=(
                                f"Tensor {name!r} shape[{i}]={dim!r} is not an integer"
                            ),
                        )
                    )
                    shape_ok = False
                elif dim < 0:
                    issues.append(
                        TensorValidationIssue(
                            rule_id="ST205",
                            tensor_name=name,
                            detail=(
                                f"Tensor {name!r} shape[{i}]={dim} is negative"
                            ),
                        )
                    )
                    shape_ok = False

        # ------------------------------------------------------------------
        # ST203 / ST208: data_offsets validation
        # ------------------------------------------------------------------
        offsets_ok = True
        if not isinstance(data_offsets, list) or len(data_offsets) != 2:
            issues.append(
                TensorValidationIssue(
                    rule_id="ST203",
                    tensor_name=name,
                    detail=(
                        f"Tensor {name!r} data_offsets must be a 2-element list, "
                        f"got {data_offsets!r}"
                    ),
                )
            )
            offsets_ok = False
        else:
            start_raw, end_raw = data_offsets[0], data_offsets[1]

            # Type check
            if not isinstance(start_raw, int) or not isinstance(end_raw, int):
                issues.append(
                    TensorValidationIssue(
                        rule_id="ST203",
                        tensor_name=name,
                        detail=(
                            f"Tensor {name!r} data_offsets values must be integers, "
                            f"got [{type(start_raw).__name__}, {type(end_raw).__name__}]"
                        ),
                    )
                )
                offsets_ok = False
            else:
                start: int = start_raw
                end: int = end_raw

                # Negative check
                if start < 0 or end < 0:
                    issues.append(
                        TensorValidationIssue(
                            rule_id="ST203",
                            tensor_name=name,
                            detail=(
                                f"Tensor {name!r} data_offsets [{start}, {end}] "
                                "contains negative value"
                            ),
                        )
                    )
                    offsets_ok = False

                # start > end check (start == end is valid for zero-element tensors)
                elif start > end:
                    issues.append(
                        TensorValidationIssue(
                            rule_id="ST203",
                            tensor_name=name,
                            detail=(
                                f"Tensor {name!r} data_offsets [{start}, {end}]: "
                                "start > end is invalid"
                            ),
                        )
                    )
                    offsets_ok = False

                # Bounds check against data section size
                elif end > data_section_size:
                    issues.append(
                        TensorValidationIssue(
                            rule_id="ST208",
                            tensor_name=name,
                            detail=(
                                f"Tensor {name!r} data_offsets [{start}, {end}] "
                                f"extends beyond data section size {data_section_size}"
                            ),
                        )
                    )
                    offsets_ok = False

        # ------------------------------------------------------------------
        # Accumulate valid descriptor for overlap detection
        # ------------------------------------------------------------------
        if (
            offsets_ok
            and isinstance(dtype, str)
            and shape_ok
            and isinstance(shape, list)
        ):
            descriptors.append(
                TensorDescriptor(
                    name=name,
                    dtype=dtype if isinstance(dtype, str) else "",
                    shape=[d for d in shape if isinstance(d, int)],
                    data_start=data_offsets[0],  # type: ignore[index]
                    data_end=data_offsets[1],   # type: ignore[index]
                )
            )

    # ------------------------------------------------------------------
    # ST202: Overlap detection — O(N log N) sort-then-scan
    # ------------------------------------------------------------------
    overlap_issues = _detect_overlaps(descriptors)
    issues.extend(overlap_issues)

    return descriptors, issues


def _detect_overlaps(
    descriptors: list[TensorDescriptor],
) -> list[TensorValidationIssue]:
    """Detect pairwise-overlapping tensor byte ranges.

    Uses an O(N log N) sweep-line algorithm:
    1. Sort descriptors by start offset.
    2. Walk the sorted list; for each entry, check whether it overlaps with
       the previous one.  Zero-length regions (start == end) do not trigger
       overlap with adjacent regions at the same boundary.

    Parameters
    ----------
    descriptors:
        Fully valid tensor descriptors (no ST203/ST208 issues).

    Returns
    -------
    list[TensorValidationIssue]
        Zero or more ST202 overlap findings.
    """
    if len(descriptors) < 2:
        return []

    # Sort by start offset; secondary key: name for determinism
    sorted_descs = sorted(descriptors, key=lambda d: (d.data_start, d.name))
    overlap_issues: list[TensorValidationIssue] = []
    prev = sorted_descs[0]

    for curr in sorted_descs[1:]:
        # Overlap: curr starts before prev ends AND at least one has non-zero span
        prev_span = prev.data_end - prev.data_start
        if curr.data_start < prev.data_end and (prev_span > 0 or curr.data_end > curr.data_start):
            overlap_issues.append(
                TensorValidationIssue(
                    rule_id="ST202",
                    tensor_name=curr.name,
                    detail=(
                        f"Tensor {curr.name!r} [{curr.data_start}, {curr.data_end}] "
                        f"overlaps with {prev.name!r} [{prev.data_start}, {prev.data_end}]"
                    ),
                    byte_offset=curr.data_start,
                )
            )
        # Advance the high-water mark
        if curr.data_end > prev.data_end:
            prev = curr
        elif curr.data_end == prev.data_end:
            # Choose next "prev" as the one with larger end for continued sweep
            prev = curr if curr.data_start > prev.data_start else prev

    return overlap_issues


__all__ = [
    "SAFETENSORS_VALID_DTYPES",
    "TensorDescriptor",
    "TensorValidationIssue",
    "validate_tensors",
]
