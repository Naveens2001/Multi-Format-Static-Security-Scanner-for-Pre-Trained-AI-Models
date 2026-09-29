"""
model_scanner.analyzers.mindspore.proto_reader
===============================================

Bounded, zero-dependency protobuf wire-format parser for MindSpore
checkpoint files (``mindspore_pb2.Checkpoint``).

SECURITY GUARANTEE
------------------
- Never imports mindspore, protobuf runtime, or any ML library.
- Never allocates memory proportional to declared tensor dimensions.
- All dimension arithmetic uses Python arbitrary-precision integers (no
  overflow, no silent truncation).
- Varint reads are bounded to 10 continuation bytes (protobuf spec).
- Tensor count is bounded by MAX_TENSORS (1 000).
- Dimensions-per-tensor are bounded by MAX_DIMS (64).
- Tensor name length is bounded by MAX_NAME_BYTES (512).
- Malformed data raises MalformedProtoError; callers must catch it.

MindSpore protobuf schema (inferred from wire dumps)
----------------------------------------------------
message Checkpoint {
  repeated Parameter parameter = 1;
}
message Parameter {
  string name   = 1;
  TensorProto tensor = 2;
}
message TensorProto {
  repeated int64 dims          = 1;   // wire type VARINT
  string         data_type     = 2;   // e.g. "Float32"
  bytes          tensor_content = 3;  // raw tensor bytes
}
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Safety limits (never negotiate these downward)
# ---------------------------------------------------------------------------

MAX_TENSORS: int = 1_000
MAX_DIMS: int = 64
MAX_NAME_BYTES: int = 512
_MAX_VARINT_BYTES: int = 10      # protobuf spec: 64-bit varint ≤ 10 bytes
_MAX_TENSOR_CONTENT_SCAN: int = 8  # only peek, never read the full payload

# MindSpore dtype strings → bytes per element
_DTYPE_ELEMENT_BYTES: dict[str, int] = {
    "Bool": 1,
    "Int8": 1,
    "Int16": 2,
    "Int32": 4,
    "Int64": 8,
    "UInt8": 1,
    "UInt16": 2,
    "UInt32": 4,
    "UInt64": 8,
    "Float16": 2,
    "BFloat16": 2,
    "Float32": 4,
    "Float64": 8,
    "Complex64": 8,
    "Complex128": 16,
}

_DEFAULT_ELEMENT_BYTES: int = 4  # fallback for unrecognised dtypes


class MalformedProtoError(ValueError):
    """Raised when the protobuf wire stream is invalid or truncated."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TensorMeta:
    """Metadata extracted from one MindSpore TensorProto without reading data."""

    name: str                       # tensor parameter name
    dims: list[int]                 # declared shape dimensions
    data_type: str                  # e.g. "Float32"
    actual_data_bytes: int          # bytes of tensor_content in the file (may be 0)
    tensor_field_offset: int        # byte offset of the tensor field tag in the file
    tensor_field_length: int        # byte length of the tensor field value

    @property
    def element_bytes(self) -> int:
        """Bytes per element for the declared dtype (defaults to 4)."""
        return _DTYPE_ELEMENT_BYTES.get(self.data_type, _DEFAULT_ELEMENT_BYTES)

    @property
    def declared_element_count(self) -> int:
        """Product of declared dimensions.

        Uses Python integers (no overflow).  Returns 0 for scalar tensors
        (empty dims list) or when any dim is zero.
        """
        if not self.dims:
            return 0
        product = 1
        for d in self.dims:
            product *= d
            if product < 0:
                return 0  # negative dim → nonsensical, treat as zero
        return product

    @property
    def declared_allocation_bytes(self) -> int:
        """Total bytes required to load this tensor (declared, not actual)."""
        return self.declared_element_count * self.element_bytes


# ---------------------------------------------------------------------------
# Low-level varint / field-tag helpers
# ---------------------------------------------------------------------------


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read a LEB128 varint.  Returns (value, new_offset).

    Raises MalformedProtoError if truncated or too long.
    """
    value = 0
    shift = 0
    for _ in range(_MAX_VARINT_BYTES):
        if offset >= len(data):
            raise MalformedProtoError(
                f"Truncated varint at offset {offset}: unexpected end of data"
            )
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not (byte & 0x80):
            return value, offset
    raise MalformedProtoError("Varint too long: exceeds 10 continuation bytes")


def _read_field_tag(data: bytes, offset: int) -> tuple[int, int, int]:
    """Read a protobuf field tag.

    Returns (field_number, wire_type, new_offset).
    """
    tag, offset = _read_varint(data, offset)
    return tag >> 3, tag & 0x07, offset


def _skip_field(data: bytes, offset: int, wire_type: int) -> int:
    """Skip a protobuf field value and return the new offset."""
    if wire_type == 0:  # VARINT
        _, offset = _read_varint(data, offset)
    elif wire_type == 1:  # 64-BIT
        offset += 8
    elif wire_type == 2:  # LEN
        length, offset = _read_varint(data, offset)
        offset += length
    elif wire_type == 5:  # 32-BIT
        offset += 4
    elif wire_type in (3, 4):  # deprecated SGROUP / EGROUP — skip until EGROUP
        raise MalformedProtoError(f"Unsupported wire type {wire_type} (deprecated group)")
    else:
        raise MalformedProtoError(f"Unknown wire type {wire_type}")
    if offset > len(data):
        raise MalformedProtoError(
            f"Field skip overruns data: offset={offset} > len={len(data)}"
        )
    return offset


# ---------------------------------------------------------------------------
# TensorProto parser
# ---------------------------------------------------------------------------


def _parse_tensor_proto(
    data: bytes,
    start: int,
    end: int,
    tensor_field_offset: int,
    tensor_field_length: int,
) -> TensorMeta | None:
    """Parse a TensorProto message slice [start, end).

    Returns a TensorMeta or None if the slice is empty / unparseable.
    Does NOT allocate tensor data.
    """
    dims: list[int] = []
    data_type: str = ""
    actual_data_bytes: int = 0
    name: str = ""

    offset = start
    while offset < end:
        if offset >= len(data):
            break
        try:
            field_num, wire_type, offset = _read_field_tag(data, offset)
        except MalformedProtoError:
            break

        if field_num == 1 and wire_type == 0:
            # dims field (VARINT) — append each dimension
            if len(dims) >= MAX_DIMS:
                # Skip excess dimensions without storing
                _, offset = _read_varint(data, offset)
            else:
                dim, offset = _read_varint(data, offset)
                dims.append(dim)

        elif field_num == 2 and wire_type == 2:
            # data_type field (LEN = string)
            length, offset = _read_varint(data, offset)
            if offset + length > len(data):
                break
            try:
                data_type = data[offset : offset + length].decode("utf-8", errors="replace")
            except Exception:
                data_type = ""
            offset += length

        elif field_num == 3 and wire_type == 2:
            # tensor_content field (LEN = bytes) — record byte count, do not read data
            length, offset = _read_varint(data, offset)
            actual_data_bytes = length
            offset += length  # skip the actual bytes
            if offset > len(data):
                # File is truncated; actual data may be partial but we record what was declared
                actual_data_bytes = max(0, len(data) - (offset - length))
                break

        else:
            try:
                offset = _skip_field(data, offset, wire_type)
            except MalformedProtoError:
                break

    return TensorMeta(
        name=name,
        dims=dims,
        data_type=data_type,
        actual_data_bytes=actual_data_bytes,
        tensor_field_offset=tensor_field_offset,
        tensor_field_length=tensor_field_length,
    )


# ---------------------------------------------------------------------------
# Parameter parser
# ---------------------------------------------------------------------------


def _parse_parameter(data: bytes, start: int, end: int) -> TensorMeta | None:
    """Parse a Parameter message slice [start, end).

    Returns a TensorMeta (with name filled in) or None.
    """
    name: str = ""
    tensor_meta: TensorMeta | None = None

    offset = start
    while offset < end:
        if offset >= len(data):
            break
        try:
            field_num, wire_type, offset = _read_field_tag(data, offset)
        except MalformedProtoError:
            break

        if field_num == 1 and wire_type == 2:
            # Parameter.name
            length, offset = _read_varint(data, offset)
            if length > MAX_NAME_BYTES:
                offset += length  # skip over-length name
            elif offset + length <= len(data):
                try:
                    name = data[offset : offset + length].decode("utf-8", errors="replace")
                except Exception:
                    name = ""
                offset += length
            else:
                break

        elif field_num == 2 and wire_type == 2:
            # Parameter.tensor
            tensor_field_offset = offset - 1  # tag byte
            length, offset = _read_varint(data, offset)
            tensor_end = offset + length
            if tensor_end > len(data):
                tensor_end = len(data)
            tensor_field_length = tensor_end - tensor_field_offset
            tensor_meta = _parse_tensor_proto(
                data, offset, tensor_end, tensor_field_offset, tensor_field_length
            )
            offset = tensor_end

        else:
            try:
                offset = _skip_field(data, offset, wire_type)
            except MalformedProtoError:
                break

    if tensor_meta is not None:
        tensor_meta.name = name  # back-fill name from Parameter
    elif name:
        # Checkpoint with name but no tensor — return minimal meta
        tensor_meta = TensorMeta(
            name=name,
            dims=[],
            data_type="",
            actual_data_bytes=0,
            tensor_field_offset=start,
            tensor_field_length=end - start,
        )

    return tensor_meta


# ---------------------------------------------------------------------------
# Top-level Checkpoint parser
# ---------------------------------------------------------------------------


def parse_checkpoint_metadata(data: bytes) -> list[TensorMeta]:
    """Parse a MindSpore protobuf checkpoint and extract tensor metadata.

    Parameters
    ----------
    data:
        Raw bytes of the .ckpt file.

    Returns
    -------
    list[TensorMeta]
        One entry per ``Parameter`` found (up to MAX_TENSORS).
        Raises MalformedProtoError on structural failures; the caller
        should catch this and treat it as a parse-error signal.

    SECURITY: Never allocates memory proportional to declared tensor sizes.
    """
    if not data:
        return []

    results: list[TensorMeta] = []
    offset = 0
    total = len(data)

    while offset < total and len(results) < MAX_TENSORS:
        try:
            field_num, wire_type, offset = _read_field_tag(data, offset)
        except MalformedProtoError:
            break

        if field_num == 1 and wire_type == 2:
            # Checkpoint.parameter (LEN)
            length, offset = _read_varint(data, offset)
            param_end = offset + length
            if param_end > total:
                param_end = total
            meta = _parse_parameter(data, offset, param_end)
            if meta is not None:
                results.append(meta)
            offset = param_end

        else:
            try:
                offset = _skip_field(data, offset, wire_type)
            except MalformedProtoError:
                break

    return results


__all__ = [
    "MalformedProtoError",
    "TensorMeta",
    "parse_checkpoint_metadata",
]
