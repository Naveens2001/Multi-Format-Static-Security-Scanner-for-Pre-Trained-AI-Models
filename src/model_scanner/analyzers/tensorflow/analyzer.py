"""Static security analyzer for TensorFlow SavedModel files (saved_model.pb)."""
from __future__ import annotations
import re
import urllib.parse
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.core.models import (
    Artifact, Confidence, CoordinateSpace, Evidence, Finding, ModelFormat, ScanConfig, Severity,
)

# Heuristic set of known standard TF ops (non-exhaustive; covers most common ones)
# We detect anything NOT in this set as potentially external
_STANDARD_TF_OPS: frozenset[str] = frozenset([
    # Math ops
    "Add", "AddN", "AddV2", "Sub", "Mul", "Div", "RealDiv", "FloorDiv",
    "Mod", "FloorMod", "Pow", "Abs", "Neg", "Sign", "Ceil", "Floor", "Round",
    "Sqrt", "Rsqrt", "Square", "Exp", "Log", "Log1p", "Sin", "Cos", "Tan",
    "Asin", "Acos", "Atan", "Atan2", "Sinh", "Cosh", "Tanh", "Asinh", "Acosh",
    "Atanh", "Maximum", "Minimum", "Cumsum", "Cumprod",
    # Matrix / linear algebra
    "MatMul", "BatchMatMul", "BatchMatMulV2", "Einsum",
    "Transpose", "Reshape", "Squeeze", "ExpandDims", "Tile",
    # Reduction
    "Sum", "Mean", "Prod", "Max", "Min", "All", "Any",
    "ReduceSum", "ReduceMean", "ReduceProd", "ReduceMax", "ReduceMin",
    "ReduceAll", "ReduceAny",
    # Activation functions
    "Relu", "Relu6", "LeakyRelu", "Elu", "Selu", "Softmax", "LogSoftmax",
    "Sigmoid", "Tanh", "Softplus", "Softsign", "Gelu",
    # Normalization
    "BatchNorm", "FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3",
    "LayerNorm",
    # Convolution
    "Conv2D", "Conv3D", "DepthwiseConv2dNative", "Conv2DBackpropInput",
    "Conv2DBackpropFilter", "AvgPool", "MaxPool", "AvgPool3D", "MaxPool3D",
    "BiasAdd",
    # Shape / slicing / indexing
    "Shape", "Size", "Rank", "Slice", "StridedSlice", "GatherNd", "Gather",
    "ScatterNd", "OneHot", "Concat", "ConcatV2", "Split", "SplitV",
    "Pack", "Unpack", "Stack", "Unstack", "Pad", "PadV2",
    # Conditional ops (Select family)
    "Select", "SelectV2", "SelectV2Scalar", "Where",
    # Type ops
    "Cast", "Bitcast",
    # Variable ops
    "Variable", "VarHandleOp", "ReadVariableOp", "AssignVariableOp",
    "ResourceGather", "ResourceScatterAdd",
    "DisableCopyOnRead",  # variable read-mode flag, not executable
    # Control flow
    "Switch", "Merge", "Enter", "Exit", "NextIteration", "LoopCond",
    "IfOp", "WhileOp", "StatefulIf", "StatefulWhile",
    # Random
    "RandomUniform", "RandomStandardNormal", "TruncatedNormal",
    "RandomShuffle", "RandomCrop",
    # String ops
    "StringJoin", "StringLength", "StringSplit", "StringFormat",
    "StaticRegexFullMatch", "RegexReplace", "StringToHashBucketFast",
    # Embedding
    "EmbeddingLookup", "EmbeddingLookupSparse",
    # Checkpoint infrastructure (present in any model with a checkpoint graph;
    # not a meaningful standalone signal — ReadFile/WriteFile are the real indicators)
    "ShardedFilename", "ShardedFilespec",
    # Misc
    "Assert", "Print", "Identity", "IdentityN", "NoOp", "Const",
    "Placeholder", "PlaceholderV2", "PlaceholderWithDefault",
    "FIFOQueue", "QueueDequeue", "QueueEnqueue",
    "_Arg", "_Retval", "_ArrayToList", "_ListToArray",
    "StatefulPartitionedCall", "PartitionedCall",
    # IO
    "DecodeJpeg", "DecodePng", "DecodeBmp", "EncodeJpeg", "EncodePng",
    "ResizeBilinear", "ResizeNearestNeighbor",
    # Lookup table
    "LookupTableFind", "InitializeTableFromTextFile",
    # Text
    "LookupTableFindV2",
    # Training ops
    "ApplyAdam", "ApplyGradientDescent", "ApplyRMSProp", "ApplyMomentum",
    # Functional API
    "FuncOp",
])

_RULE_SEVERITY = {
    "TF101": Severity.MEDIUM, "TF102": Severity.MEDIUM, "TF103": Severity.CRITICAL,
}
_RULE_CONFIDENCE = {
    "TF101": Confidence.MEDIUM, "TF102": Confidence.HIGH, "TF103": Confidence.HIGH,
}
_RULE_NAMES = {
    "TF101": "External or Custom Op in TensorFlow Graph",
    "TF102": "Malformed TensorFlow SavedModel",
    "TF103": "Path Traversal Indicator in SavedModel Asset",
}
_RULE_CATEGORY = {
    "TF101": "structural_anomaly", "TF102": "structural_anomaly", "TF103": "tampering",
}
_RULE_REMEDIATION = {
    "TF101": (
        "Audit the custom op implementation for malicious behavior. "
        "Prefer models using only standard TensorFlow ops."
    ),
    "TF102": (
        "Re-export the model from TensorFlow. "
        "Verify the file is not truncated and re-acquire from a trusted source."
    ),
    "TF103": (
        "Do not load this SavedModel. Re-acquire from a trusted source and verify that "
        "all asset filenames are plain relative names without traversal sequences."
    ),
}

# Known-dangerous TF ops that are always reported regardless of case or length.
# These are legitimate TF ops but read/write the filesystem or execute Python and
# are therefore high-signal indicators when present in an untrusted model graph.
_DANGEROUS_TF_OPS: frozenset[str] = frozenset([
    "ReadFile", "WriteFile", "PyFunc", "PyFuncStateless", "EagerPyFunc",
    "ShellExecute", "SubprocessOp",
    "DeleteSessionTensor", "GetSessionTensor", "SetSessionTensor",
    "MergeV2Checkpoints", "SaveV2", "RestoreV2",
])

# Protobuf field names and generic metadata strings that the raw wire-format scan
# routinely surfaces. They are not op names and would otherwise generate dozens of
# TF101 false positives per file.
_PROTO_FIELD_BLOCKLIST: frozenset[str] = frozenset([
    # Common protobuf field names
    "type", "shape", "dtype", "value", "values", "resource", "config",
    "input", "inputs", "output", "outputs", "name", "names", "key", "keys",
    "func", "args", "kwargs", "attr", "attrs", "op", "node", "nodes",
    "string", "int", "float", "bool", "bytes", "uint32", "uint64", "int64",
    "int32", "float32", "float64", "double",
    # Common parameter names
    "axis", "shape_and_slices", "shared_name", "container",
    "dtypes", "executor_type", "config_proto", "debug_name", "pattern",
    "separator", "num_shards", "source", "destination", "prefix", "shard",
    "condition", "tensor", "tensors", "tensor_names",
    "checkpoint_prefixes", "file_prefix", "destination_prefix",
    "basename", "filename", "saver_filename", "output_0",
    "validate_shape", "delete_old_dirs", "allowed_devices",
    "annotations", "defaults", "varargs", "varkw", "kwonlyargs",
    "kwonlydefaults", "unknown", "identity", "trace_0",
    "signature_map", "signatures", "serve", "serving_default",
    # Ordinary known values
    "CPU", "GPU", "kEagerRuntime", "part",
    # Protobuf attribute-type keys — appear as attribute names/keys in every
    # TF graph node, not as op names (e.g. "Tout", "Tin", "Tdim").
    "Tout", "Tin", "Tdim", "Tindices", "Tparams", "Tresult", "Tshape",
    "Tkeys", "Tvalues", "Toutput", "Tinput",
    # Python function-signature metadata stored in ConcreteFunction serialisation.
    "FullArgSpec",
])

# Version-string / numeric patterns to reject (e.g. "2.21.0", "1.0", "3.14").
_VERSION_STRING_RE: re.Pattern[str] = re.compile(r"^\d+(\.\d+)+$")

# Trailing numeric instance suffix appended to ordinary op names by the graph
# exporter (e.g. "Identity_1", "Const_2", "StatefulPartitionedCall_3").
_INSTANCE_SUFFIX_RE: re.Pattern[str] = re.compile(r"_\d+$")

# URL-encoded / literal path traversal indicators for asset filename inspection.
_TRAVERSAL_TOKENS: tuple[str, ...] = ("%2e%2e", "%2f", "%5c", "..%2f", "..%5c")


class TensorFlowAnalyzer(BaseFormatAnalyzer):
    """Static security analyzer for TensorFlow SavedModel (saved_model.pb).

    Requires 'protobuf' optional dependency for full graph analysis.
    Falls back to heuristic string scanning if protobuf is unavailable.
    SECURITY: Never imports tensorflow, never executes model code.
    """

    def analyze(self, artifact: Artifact, config: ScanConfig | None = None) -> Sequence[Finding]:
        """Perform static analysis on a TensorFlow SavedModel artifact."""
        data = self._read_artifact_bytes(artifact)
        if not data:
            return []

        findings: list[Finding] = []
        counter = [0]

        def _next_id(rule_id: str) -> str:
            counter[0] += 1
            return f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}"

        def _make_finding(rule_id: str, matched_value: str, description: str) -> Finding:
            return Finding(
                finding_id=_next_id(rule_id),
                rule_id=rule_id,
                rule_name=_RULE_NAMES[rule_id],
                severity=_RULE_SEVERITY[rule_id],
                confidence=_RULE_CONFIDENCE[rule_id],
                category=_RULE_CATEGORY[rule_id],
                format=ModelFormat.TF_SAVEDMODEL,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARTIFACT,
                    byte_offset=0, byte_length=None, archive_member=None,
                    matched_value=matched_value[:512],
                ),
                description=description,
                remediation=_RULE_REMEDIATION[rule_id],
            )

        # Try protobuf-based parsing first
        try:
            findings.extend(self._analyze_with_protobuf(data, artifact, _next_id, _make_finding))
        except ImportError:
            # Fallback: heuristic analysis when protobuf library is unavailable
            findings.extend(self._analyze_heuristic(data, artifact, _next_id, _make_finding))

        # TF103: inspect sibling assets/ directory for path-traversal filenames.
        # Static filename inspection only — asset files are NEVER opened or read.
        findings.extend(self._check_asset_paths(artifact, _make_finding))

        return findings

    def _analyze_with_protobuf(
        self,
        data: bytes,
        artifact: Artifact,
        _next_id: object,
        _make_finding: object,
    ) -> list[Finding]:
        """Parse saved_model.pb using raw protobuf wire format scanning."""
        try:
            import google.protobuf  # noqa: F401  -- just checking if available
        except ImportError:
            raise  # will trigger heuristic fallback

        findings: list[Finding] = []
        make_finding = _make_finding  # type: ignore[assignment]

        try:
            op_names = _extract_ops_from_pb(data, depth=0)
        except Exception as exc:
            findings.append(make_finding(
                "TF102", "PARSE_ERROR",
                f"saved_model.pb parse error: {exc}",
            ))
            return findings

        custom_ops = {op for op in op_names if _is_suspicious_op(op)}
        seen: set[str] = set()
        for op in sorted(custom_ops):
            if op not in seen:
                seen.add(op)
                findings.append(make_finding(
                    "TF101", f"op={op!r}",
                    f"TensorFlow graph contains non-standard op '{op}'. "
                    f"Custom ops may execute native code at inference time.",
                ))
        return findings

    def _analyze_heuristic(
        self,
        data: bytes,
        artifact: Artifact,
        _next_id: object,
        _make_finding: object,
    ) -> list[Finding]:
        """Heuristic analysis when protobuf library is unavailable."""
        # Just verify the file looks like a valid protobuf (non-empty)
        if len(data) < 2:
            make_finding = _make_finding  # type: ignore[assignment]
            return [make_finding("TF102", "EMPTY_FILE", "saved_model.pb is empty or too small.")]
        return []

    def _check_asset_paths(
        self,
        artifact: Artifact,
        _make_finding: object,
    ) -> list[Finding]:
        """Inspect the sibling ``assets/`` directory for path-traversal filenames.

        SECURITY: This performs FILENAME-STRING inspection only. Asset files are
        never opened, read, resolved, or followed. Only the raw directory entry
        names are examined for traversal indicators.
        """
        findings: list[Finding] = []
        make_finding = _make_finding  # type: ignore[assignment]

        if artifact.local_path is None:
            return findings

        try:
            pb_path = Path(artifact.local_path)
            assets_dir = pb_path.parent / "assets"
            if not assets_dir.is_dir():
                return findings
            entries = list(assets_dir.iterdir())  # one level only, non-recursive
        except OSError:
            return findings

        for entry in entries:
            raw_filename = entry.name
            decoded, indicator = _traversal_indicator(raw_filename)
            if indicator is None:
                continue
            findings.append(make_finding(
                "TF103",
                f"asset_filename={raw_filename!r} decoded={decoded!r} indicator={indicator}",
                (
                    f"SavedModel asset directory contains path traversal indicator: "
                    f"'{raw_filename}' -> decoded: '{decoded}'. "
                    f"Asset file was not accessed; detection is based on filename only."
                ),
            ))

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


def _is_suspicious_op(op: str) -> bool:
    """Return True if an extracted string should be reported as a TF101 custom op.

    Two-tier logic:
      Tier 1 — always report known-dangerous ops (filesystem / Python execution),
               regardless of length or case.
      Tier 2 — report an unknown token only when it looks like a genuine op name
               and is not a protobuf field name, version string, or numeric noise.

    This replaces the prior "report every 2-64 char alnum string" heuristic, which
    produced ~100 findings per file.
    """
    if not op:
        return False

    # Tier 1: always-report dangerous ops.
    if op in _DANGEROUS_TF_OPS:
        return True

    # Normalise a trailing "_<digits>" instance suffix (e.g. "Identity_1",
    # "Const_2", "StatefulPartitionedCall_3") before the standard-op check; the
    # graph exporter appends these numeric disambiguators to ordinary ops.
    base = _INSTANCE_SUFFIX_RE.sub("", op)

    # Never report standard ops or protobuf internals.
    if op in _STANDARD_TF_OPS or base in _STANDARD_TF_OPS:
        return False
    if op.startswith("_"):
        return False
    if op in _PROTO_FIELD_BLOCKLIST or base in _PROTO_FIELD_BLOCKLIST:
        return False

    # Reject noise: too short, leading dot/digit, purely numeric, version strings.
    if len(op) < 4:
        return False
    if op[0] == "." or op[0].isdigit():
        return False
    if _VERSION_STRING_RE.match(op):
        return False

    # Tier 2: require the token to START with an uppercase ASCII letter.
    # Genuine TF op names are CamelCase and always begin with uppercase
    # (e.g. ReadFile, MatMul, StatefulPartitionedCall). Tokens that have an
    # uppercase letter somewhere but START lowercase are wire-format fragments
    # produced by mid-string slicing (e.g. "ableOp" from "ReadVariableOp",
    # "storeV2" from "RestoreV2", "adFile" from "ReadFile"). Those fragments
    # pass the alphanumeric filter but are not real op names.
    if not op[0].isupper():
        # Still allow lowercase tokens that embed a clearly suspicious
        # substring or that look like an obfuscated filesystem-path variable
        # name (e.g. "serving_default_cmd", the "____<path>___" pattern).
        lowered = op.lower()
        if any(sub in lowered for sub in (
            "cmd", "shell", "exec", "system", "subprocess", "eval",
            "payload", "evil", "passwd", "backdoor",
        )):
            return True
        if "____" in lowered and any(sub in lowered for sub in ("tmp", "etc", "var", "usr", "home")):
            return True
        return False

    return True


def _traversal_indicator(raw_filename: str) -> tuple[str, str | None]:
    """Return (decoded_name, indicator_or_None) for an asset filename.

    Detects, on the raw filename string ONLY:
      - URL-encoded traversal tokens (%2e%2e, %2f, %5c)
      - '..' path components (after URL-decoding)
      - absolute path components (POSIX '/' or Windows drive / leading backslash)

    The indicator string names the class of traversal found, or None if the
    filename is benign.
    """
    lowered_raw = raw_filename.lower()
    for token in _TRAVERSAL_TOKENS:
        if token in lowered_raw:
            decoded = urllib.parse.unquote(raw_filename)
            return decoded, f"url_encoded_traversal:{token}"

    decoded = urllib.parse.unquote(raw_filename)

    # Check for '..' components in either path separator convention.
    posix_parts = PurePosixPath(decoded).parts
    win_parts = PureWindowsPath(decoded).parts
    if ".." in posix_parts or ".." in win_parts:
        return decoded, "dotdot_component"

    # Absolute POSIX path.
    if decoded.startswith("/"):
        return decoded, "absolute_posix"

    # Absolute Windows path (drive letter or leading backslash / UNC).
    if decoded.startswith("\\"):
        return decoded, "absolute_windows"
    if len(decoded) >= 2 and decoded[1] == ":" and decoded[0].isalpha():
        return decoded, "absolute_windows_drive"

    return decoded, None


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read a protobuf varint from data at pos. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift >= 64:
            raise ValueError("Varint too large")
    raise ValueError("Unexpected end of data reading varint")


def _extract_ops_from_pb(data: bytes, depth: int = 0) -> list[str]:
    """Extract op name strings from a TF SavedModel protobuf using raw parsing.

    This is a best-effort parser that understands enough of the protobuf wire
    format to find string fields that look like op names. We limit recursion
    depth to avoid stack overflow on deeply nested or adversarial inputs.
    """
    # Recursion depth limit to prevent stack overflow on adversarial input
    if depth > 8:
        return []

    ops: list[str] = []
    pos = 0
    length = len(data)

    # Limit parsing to avoid spending too long on huge files
    MAX_PARSE_BYTES = 50 * 1024 * 1024  # 50MB
    if length > MAX_PARSE_BYTES:
        data = data[:MAX_PARSE_BYTES]
        length = MAX_PARSE_BYTES

    # Walk the protobuf scanning for string-length-delimited fields
    while pos < length:
        try:
            tag_varint, pos = _read_varint(data, pos)
        except (ValueError, IndexError):
            break

        wire_type = tag_varint & 0x7

        if wire_type == 0:  # varint
            try:
                _, pos = _read_varint(data, pos)
            except (ValueError, IndexError):
                break
        elif wire_type == 1:  # 64-bit
            pos += 8
        elif wire_type == 2:  # length-delimited
            try:
                length_val, pos = _read_varint(data, pos)
            except (ValueError, IndexError):
                break
            if length_val < 0 or length_val > 10 * 1024 * 1024:
                break
            payload = data[pos:pos + length_val]
            pos += length_val

            # Try to interpret as UTF-8 string if field looks like an op name
            # Op names are typically 3-64 chars, ASCII only
            if 2 <= length_val <= 128:
                try:
                    s = payload.decode("utf-8")
                    # Simple heuristic: op names are CamelCase or UPPER_SNAKE
                    # and don't contain spaces, newlines, slashes
                    if (
                        s and
                        all(c.isalnum() or c in "_-." for c in s) and
                        not s.startswith("/") and
                        len(s) <= 64
                    ):
                        ops.append(s)
                except UnicodeDecodeError:
                    pass

            # Recursively try to parse embedded messages
            # Only if payload is large enough to contain meaningful content
            if length_val > 4 and depth < 8:
                try:
                    sub_ops = _extract_ops_from_pb(payload, depth=depth + 1)
                    ops.extend(sub_ops)
                except Exception:
                    pass
        elif wire_type == 5:  # 32-bit
            pos += 4
        else:
            break  # Unknown wire type - stop parsing

    return ops


__all__ = ["TensorFlowAnalyzer"]
