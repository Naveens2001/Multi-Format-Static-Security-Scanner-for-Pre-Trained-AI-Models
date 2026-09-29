"""Static security analyzer for Keras model files (.h5, .hdf5, .keras ZIP)."""
from __future__ import annotations
import base64
import io
import json
import zipfile
from collections.abc import Sequence
from model_scanner.analyzers.base import BaseFormatAnalyzer
from model_scanner.analyzers._weight_utils import (
    check_tensor,
    WI_RULE_SEVERITY,
    WI_RULE_CONFIDENCE,
    WI_RULE_NAMES,
    WI_RULE_CATEGORY,
    WI_RULE_REMEDIATION,
    WI_KIND_TO_RULE,
)
from model_scanner.core.models import (
    Artifact, Confidence, CoordinateSpace, Evidence, Finding, ModelFormat, ScanConfig, Severity,
)

# Standard Keras built-in class names that must NOT trigger KR101.
#
# This allowlist covers every class that the Keras serializer can legitimately
# embed as a "class_name" value in a model config JSON.  KR101 fires only when
# a class name is absent from this set — i.e. it is a genuinely non-standard /
# registered / custom class that requires human review.
#
# Categories included (all sourced from the official keras.io documentation):
#   - Core / container layers
#   - Convolutional, pooling, recurrent, normalisation, embedding layers
#   - Attention, merge, noise, reshape, wrappers, RNN cell layers
#   - Preprocessing layers
#   - Initializers  ← primary R4 fix; absence caused 8 false positives
#   - Activations (used as class_name in functional-API activation layers)
#   - Regularizers and constraints
#   - Optimizers (appear in compiled-model configs)
#   - Losses and metrics (appear in compiled-model configs)
#   - Keras 3 / TF2 internal serialization helpers
#
# IMPORTANT: adding a class here does NOT suppress detection of genuinely
# custom classes.  Any class_name that is not in this set — including
# attacker-defined names like "MALICIOUS_LAMBDA" — still fires KR101.
_KERAS_BUILTIN_CLASSES: frozenset[str] = frozenset([
    # ------------------------------------------------------------------ Layers
    # Core
    "Dense", "Activation", "Dropout", "Flatten", "Reshape", "Permute",
    "RepeatVector", "Lambda", "ActivityRegularization", "Masking",
    "SpatialDropout1D", "SpatialDropout2D", "SpatialDropout3D",
    # Convolutional
    "Conv1D", "Conv2D", "Conv3D", "SeparableConv1D", "SeparableConv2D",
    "DepthwiseConv2D", "DepthwiseConv1D",
    "Conv2DTranspose", "Conv3DTranspose", "Conv1DTranspose",
    "LocallyConnected1D", "LocallyConnected2D",
    # Pooling
    "MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
    "AveragePooling1D", "AveragePooling2D", "AveragePooling3D",
    "GlobalMaxPooling1D", "GlobalMaxPooling2D", "GlobalMaxPooling3D",
    "GlobalAveragePooling1D", "GlobalAveragePooling2D", "GlobalAveragePooling3D",
    # Recurrent
    "LSTM", "GRU", "SimpleRNN", "TimeDistributed", "Bidirectional",
    "ConvLSTM1D", "ConvLSTM2D", "ConvLSTM3D",
    # RNN cells
    "GRUCell", "LSTMCell", "SimpleRNNCell", "StackedRNNCells",
    # Normalization
    "BatchNormalization", "LayerNormalization", "GroupNormalization",
    "UnitNormalization", "SpectralNormalization",
    # Embedding
    "Embedding",
    # Merging
    "Add", "Subtract", "Multiply", "Average", "Maximum", "Minimum",
    "Concatenate", "Dot",
    # Attention
    "Attention", "AdditiveAttention", "MultiHeadAttention",
    # Noise
    "GaussianNoise", "GaussianDropout", "AlphaDropout",
    # Container / wrappers
    "Sequential", "Functional", "Model", "InputLayer",
    "Wrapper", "TimeDistributed", "Bidirectional",
    # Preprocessing
    "Rescaling", "Resizing", "RandomFlip", "RandomRotation", "RandomZoom",
    "RandomCrop", "RandomTranslation", "CenterCrop", "Normalization",
    "StringLookup", "IntegerLookup", "TextVectorization", "Hashing",
    "RandomContrast", "RandomHeight", "RandomWidth", "RandomBrightness",
    "RandomShear",
    # Reshape / padding / cropping / upsampling
    "ZeroPadding1D", "ZeroPadding2D", "ZeroPadding3D",
    "Cropping1D", "Cropping2D", "Cropping3D",
    "UpSampling1D", "UpSampling2D", "UpSampling3D",
    # TF-backend specific
    "TFSMLayer",

    # ------------------------------------------------------------- Initializers
    # keras.initializers — confirmed false-positive source (R4 / H3 in validation).
    # All names below are documented built-ins; none are user-defined.
    "Zeros", "Ones", "Constant", "Identity",
    "RandomNormal", "RandomUniform", "TruncatedNormal",
    "VarianceScaling",
    "Orthogonal",
    "GlorotNormal", "GlorotUniform",
    "HeNormal", "HeUniform",
    "LecunNormal", "LecunUniform",

    # ------------------------------------------------------- Activation classes
    # These appear as class_name in functional-API activation layers.
    "ReLU", "Softmax", "Sigmoid", "Tanh", "ELU",
    "LeakyReLU", "PReLU", "ThresholdedReLU",
    "Softplus", "Softsign", "SELU", "Swish", "Gelu",
    "Exponential", "Linear", "Mish",

    # ------------------------------------------------------------ Regularizers
    # keras.regularizers
    "L1L2", "L1", "L2",
    "OrthogonalRegularizer",

    # -------------------------------------------------------------- Constraints
    # keras.constraints
    "MaxNorm", "NonNeg", "UnitNorm", "MinMaxNorm", "RadialConstraint",

    # --------------------------------------------------------------- Optimizers
    # Appear in compiled-model configs (keras.optimizers).
    "Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Nadam",
    "Ftrl", "AdamW", "Lion", "Adan",

    # --------------------------------------------------------------- Losses
    # keras.losses — appear in compiled-model configs.
    "MeanSquaredError", "MeanAbsoluteError",
    "BinaryCrossentropy", "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy",
    "KLDivergence", "Huber", "LogCosh",
    "Hinge", "SquaredHinge", "CategoricalHinge",
    "MeanAbsolutePercentageError", "MeanSquaredLogarithmicError",
    "CosineSimilarity", "Poisson",
    "BinaryFocalCrossentropy", "CategoricalFocalCrossentropy",

    # --------------------------------------------------------------- Metrics
    # keras.metrics — appear in compiled-model configs.
    "Accuracy", "BinaryAccuracy", "CategoricalAccuracy",
    "SparseCategoricalAccuracy", "TopKCategoricalAccuracy",
    "SparseTopKCategoricalAccuracy",
    "MeanSquaredError", "RootMeanSquaredError",
    "MeanAbsoluteError", "MeanAbsolutePercentageError",
    "MeanSquaredLogarithmicError", "CosineSimilarity", "LogCoshError",
    "AUC", "Precision", "Recall", "F1Score", "FBetaScore",
    "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives",
    "PrecisionAtRecall", "RecallAtPrecision",
    "SensitivityAtSpecificity", "SpecificityAtSensitivity",
    "MeanIoU", "OneHotIoU", "OneHotMeanIoU",
    "Sum", "Mean",

    # --------------------------------- Keras 3 internal serialisation helpers
    # Not user-facing; appear in Keras 3 / TF2 serialised configs.
    "DTypePolicy", "__keras_tensor__",
])

_RULE_SEVERITY: dict[str, Severity] = {
    "KR101": Severity.MEDIUM, "KR102": Severity.MEDIUM, "KR103": Severity.HIGH,
    "KR104": Severity.HIGH,   # base; escalated to CRITICAL for absolute/traversal paths
    "KR105": Severity.CRITICAL,  # base; may be HIGH when only indirect indicators found
    **WI_RULE_SEVERITY,  # WI101/WI102/WI103 — shared from _weight_utils (Finding 7)
}
_RULE_CONFIDENCE: dict[str, Confidence] = {
    "KR101": Confidence.MEDIUM, "KR102": Confidence.HIGH, "KR103": Confidence.HIGH,
    "KR104": Confidence.HIGH,
    "KR105": Confidence.HIGH,
    **WI_RULE_CONFIDENCE,  # WI101/WI102/WI103 — shared from _weight_utils (Finding 7)
}
_RULE_NAMES: dict[str, str] = {
    "KR101": "External or Custom Class in Keras Config",
    "KR102": "Malformed Keras Model Config or Structure",
    "KR103": "Duplicate Archive Member in Keras ZIP",
    "KR104": "HDF5 External / Virtual Dataset Reference",
    "KR105": "Keras Lambda Bytecode Payload",
    **WI_RULE_NAMES,  # WI101/WI102/WI103 — shared from _weight_utils (Finding 7)
}
_RULE_CATEGORY: dict[str, str] = {
    "KR101": "structural_anomaly", "KR102": "structural_anomaly",
    "KR103": "structural_anomaly",
    "KR104": "code_execution",
    "KR105": "code_execution",
    **WI_RULE_CATEGORY,  # WI101/WI102/WI103 — shared from _weight_utils (Finding 7)
}
_RULE_REMEDIATION: dict[str, str] = {
    "KR101": (
        "Audit the custom class implementation for malicious behavior. "
        "Do not load models with custom classes without inspection."
    ),
    "KR102": (
        "Re-export the model using tf.keras or standalone Keras. "
        "Verify the file is not truncated."
    ),
    "KR103": (
        "Reject this archive. A well-formed .keras file has exactly one entry per name. "
        "Re-export the model with a trusted serializer and re-acquire from a trusted source."
    ),
    "KR104": (
        "Remove all HDF5 ExternalLink and Virtual Dataset Source (VDS) references from this "
        "model. Re-serialize using a trusted Keras exporter with no external dependencies. "
        "Do not load this model — external references enable arbitrary file read on the host "
        "system (CVE-2026-1669)."
    ),
    "KR105": (
        "Do not load this model. A Keras Lambda layer's serialized bytecode contains "
        "indicators of dangerous execution primitives (OS command execution, dynamic import, "
        "or eval/exec). Reject the artifact and re-acquire from a trusted source. "
        "If the Lambda is required, audit the source function and re-serialize from "
        "verified, reviewed code only."
    ),
    **WI_RULE_REMEDIATION,  # WI101/WI102/WI103 — shared from _weight_utils (Finding 7)
}

_KIND_TO_RULE = WI_KIND_TO_RULE  # shared from _weight_utils (Finding 7)


# ---------------------------------------------------------------------------
# KR104 path-analysis helpers (static, pure-string — never opens external files)
# ---------------------------------------------------------------------------

def _kr104_is_absolute(ext_path: str) -> bool:
    """Return True if the external reference path is absolute."""
    if ext_path.startswith("/") or ext_path.startswith("\\"):
        return True
    # Windows drive letter: C:\ or C:/
    if len(ext_path) >= 3 and ext_path[1] == ":" and ext_path[2] in ("/", "\\"):
        return True
    return False


def _kr104_has_traversal(ext_path: str) -> bool:
    """Return True if the external reference path contains directory traversal."""
    normalized = ext_path.replace("\\", "/")
    return ".." in normalized.split("/")


def _kr104_severity(ext_path: str) -> Severity:
    """Return the KR104 severity for the given external file reference path.

    CRITICAL — path is absolute or contains '..' (clearly targeting host FS).
    HIGH     — relative path without traversal (suspicious but less definitive).
    """
    if _kr104_is_absolute(ext_path) or _kr104_has_traversal(ext_path):
        return Severity.CRITICAL
    return Severity.HIGH

# ---------------------------------------------------------------------------
# KR105 — Lambda bytecode payload static analysis
# ---------------------------------------------------------------------------
#
# SAFETY: all functions below are pure static analysis.
# - No marshal.loads(), exec(), eval(), or any code-execution primitive is called.
# - base64.b64decode() is data-only decoding: it produces raw bytes, never executes them.
# - Pattern matching is plain bytes.find() with a word-boundary check.
# - The recovered bytes are never passed to any interpreter or JIT.
#
# Detection strategy:
# Keras serializes Lambda functions as base64(marshal.dumps(func.__code__)).
# The marshal stream stores co_names and co_consts as length-prefixed ASCII/UTF-8
# strings. Dangerous identifiers (os, system, subprocess, eval, exec, …) appear
# verbatim in those strings and can be found by substring search.
# The word-boundary check (byte before and after pattern is not an identifier char)
# prevents partial matches like "eval" inside "evaluate_model".

# CRITICAL: confirmed OS command / process launch primitives
_KR105_CRITICAL_PATTERNS: tuple[tuple[bytes, str], ...] = (
    (b"subprocess", "subprocess module reference"),
    (b"Popen",      "subprocess.Popen class reference"),
    (b"popen",      "popen function reference"),
)

# CRITICAL when BOTH bytes of a pair appear as word-boundary matches in the same payload
_KR105_CRITICAL_COMBOS: tuple[tuple[tuple[bytes, bytes], str], ...] = (
    ((b"os",    b"system"), "os.system (OS command execution)"),
    ((b"nt",    b"system"), "nt.system (Windows command execution)"),
    ((b"posix", b"system"), "posix.system (POSIX command execution)"),
)

# HIGH: dynamic execution primitives — dangerous but require contextual confirmation
_KR105_HIGH_PATTERNS: tuple[tuple[bytes, str], ...] = (
    (b"__import__", "__import__ dynamic module import"),
    (b"eval",       "eval builtin"),
    (b"exec",       "exec builtin"),
)


def _kr105_is_id_byte(b: int) -> bool:
    """True if b is an ASCII identifier character (a-z A-Z 0-9 _)."""
    return (
        0x61 <= b <= 0x7A  # a-z
        or 0x41 <= b <= 0x5A  # A-Z
        or 0x30 <= b <= 0x39  # 0-9
        or b == 0x5F           # _
    )


def _kr105_word_in_bytes(data: bytes, pattern: bytes) -> bool:
    """Return True if pattern appears with word boundaries in data.

    'Word boundary' means the byte immediately before and after the match
    are both non-identifier bytes (or the buffer edge).  This prevents
    short patterns like b'eval' matching inside b'evaluate_model'.
    """
    plen = len(pattern)
    start = 0
    while True:
        pos = data.find(pattern, start)
        if pos == -1:
            return False
        before_ok = pos == 0 or not _kr105_is_id_byte(data[pos - 1])
        after_end = pos + plen
        after_ok = after_end >= len(data) or not _kr105_is_id_byte(data[after_end])
        if before_ok and after_ok:
            return True
        start = pos + 1


def _kr105_decode_code_field(code_value: object) -> bytes | None:
    """Decode a Keras Lambda 'code' field to raw bytes for static scanning.

    Keras 2 / TF2 stores the code field as base64(marshal.dumps(func.__code__)).
    This function decodes the base64 envelope only — it never calls marshal.loads()
    or any execution primitive.  The returned bytes are treated as opaque data.

    Plain-string Lambda source forms (identified by the presence of space characters,
    which are never valid in base64) are encoded directly as UTF-8 for pattern matching.
    Returns None for non-string/non-bytes types.
    """
    if isinstance(code_value, bytes):
        return code_value  # Already raw bytes (rare HDF5 attribute encoding)
    if not isinstance(code_value, str):
        return None
    # Valid base64 strings never contain space characters.  A string with spaces
    # must be plain text (Python source code or a human-readable Lambda representation)
    # and should be scanned as literal UTF-8 bytes.
    if " " in code_value:
        try:
            return code_value.encode("utf-8", errors="replace")
        except Exception:
            return None
    # Try base64 decode first (standard Keras 2/TF2 format)
    try:
        decoded = base64.b64decode(code_value, validate=False)
        if len(decoded) >= 4:
            return decoded
    except Exception:
        pass
    # Fall back: treat as literal text (source code or non-standard serialization)
    try:
        return code_value.encode("utf-8", errors="replace")
    except Exception:
        return None


def _kr105_scan_raw_bytes(raw: bytes) -> list[tuple[str, str]]:
    """Scan raw decoded bytes for dangerous execution indicator patterns.

    Returns a list of (description, severity_string) pairs — one entry per
    distinct pattern class matched.  Severity_string is 'CRITICAL' or 'HIGH'.

    SAFETY: pure static analysis — no marshal.loads(), exec(), eval(), or
    any code-execution primitive is called.  Only bytes.find() + boundary checks.
    """
    hits: list[tuple[str, str]] = []

    # CRITICAL individual patterns
    for pattern, desc in _KR105_CRITICAL_PATTERNS:
        if _kr105_word_in_bytes(raw, pattern):
            hits.append((desc, "CRITICAL"))

    # CRITICAL combos — require BOTH bytes to appear as standalone words
    for (p1, p2), desc in _KR105_CRITICAL_COMBOS:
        if _kr105_word_in_bytes(raw, p1) and _kr105_word_in_bytes(raw, p2):
            hits.append((desc, "CRITICAL"))

    # HIGH patterns — only if no CRITICAL finding was already emitted.
    # A CRITICAL finding subsumes all HIGH findings; emitting both is redundant
    # and inflates the finding count for mixed-pattern payloads. (Finding 4)
    has_critical = bool(hits)  # hits only contains CRITICAL entries at this point
    for pattern, desc in _KR105_HIGH_PATTERNS:
        if _kr105_word_in_bytes(raw, pattern) and not has_critical:
            hits.append((desc, "HIGH"))

    return hits


def _extract_lambda_code_fields(
    config: object, path: str = ""
) -> list[tuple[str, object]]:
    """Recursively find all Keras Lambda code fields in a config tree.

    Returns a list of (json_path, code_value) pairs for:
      - Every '__lambda__' node whose config has a 'code' sub-field
        (standard Keras 2 / TF2 base64-marshal format).
      - Every 'Lambda' layer node whose config.function is a plain string
        (old-style or simplified serialization, e.g. ModelAudit test artifacts).
    """
    results: list[tuple[str, object]] = []
    if isinstance(config, dict):
        cn = config.get("class_name")
        if cn == "__lambda__":
            # Standard Keras 2 / TF2: base64(marshal(code_object)) stored in 'code'
            inner = config.get("config")
            if isinstance(inner, dict) and "code" in inner:
                results.append((path or "<root>", inner["code"]))
            # Recurse into all keys EXCEPT "config" (inner already handled above)
            # and "class_name" (it's a scalar). This prevents double-visiting inner
            # via the generic descent and emitting duplicate (path, code) entries.
            # The bug (Finding 5): the original code did the explicit append above AND
            # then let the generic descent below also visit inner["code"], producing
            # two (path, code) entries for a single __lambda__ node.
            for k, v in config.items():
                if k in ("class_name", "config"):
                    continue
                child_path = f"{path}.{k}" if path else k
                results.extend(_extract_lambda_code_fields(v, child_path))
        elif cn == "Lambda":
            # Plain-string function form: Lambda layer stores function as source text.
            # The string itself is treated as literal bytes for pattern matching.
            inner = config.get("config")
            if isinstance(inner, dict):
                fn = inner.get("function")
                if isinstance(fn, str) and fn:
                    fn_path = f"{path}.config.function" if path else "config.function"
                    results.append((fn_path, fn))
            # Generic descent for all keys — this visits inner["function"] which may be
            # a __lambda__ dict and will be processed by the __lambda__ branch above.
            # We do NOT skip "config" here because there's no double-visit risk:
            # the explicit string extraction above only fires when fn is a str,
            # and the recursive descent only processes dicts/lists.
            for k, v in config.items():
                child_path = f"{path}.{k}" if path else k
                results.extend(_extract_lambda_code_fields(v, child_path))
        else:
            # Generic descent for all other dict types
            for k, v in config.items():
                child_path = f"{path}.{k}" if path else k
                results.extend(_extract_lambda_code_fields(v, child_path))
    elif isinstance(config, list):
        for i, item in enumerate(config):
            results.extend(_extract_lambda_code_fields(item, f"{path}[{i}]"))
    return results


# Keras config list keys whose entries may be bare class/function name strings.
# A bare string inside these lists is a class reference even without a wrapping
# {"class_name": ...} dict, so it must be checked against the built-in allowlist.
_KERAS_CLASS_LIST_KEYS: frozenset[str] = frozenset([
    "layers", "objects", "nodes",
])

# Path components that indicate directory traversal in ZIP member names.
# Python's zipfile.ZipFile.read() does not write to disk, so there is no
# immediate filesystem traversal risk; however, member names with these patterns
# are a recognized indicator of maliciously crafted archives that exploit
# vulnerable ZIP processing libraries (e.g. path traversal during extraction).
_KERAS_ZIP_TRAVERSAL_TOKENS: tuple[str, ...] = ("%2e%2e", "%2f", "%5c")


def _extract_class_names(config: object) -> list[str]:
    """Recursively extract class name references from Keras config JSON.

    Collects:
      - ``class_name`` string values (standard wrapped references).
      - ``registered_name`` string values (custom-object registrations).
      - Bare strings appearing directly inside known class-list keys
        (``layers``/``objects``/``nodes``). A malicious config can put a bare
        class name in such a list to bypass scanners that only look at
        ``class_name`` dicts.
    """
    names: list[str] = []
    if isinstance(config, dict):
        if "class_name" in config:
            cn = config["class_name"]
            if isinstance(cn, str):
                names.append(cn)
        if "registered_name" in config:
            rn = config["registered_name"]
            if isinstance(rn, str) and rn:
                names.append(rn)
        for k, v in config.items():
            if k in _KERAS_CLASS_LIST_KEYS and isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item:
                        names.append(item)  # bare class name string
            names.extend(_extract_class_names(v))
    elif isinstance(config, list):
        for item in config:
            names.extend(_extract_class_names(item))
    return names


class KerasAnalyzer(BaseFormatAnalyzer):
    """Static security analyzer for Keras .h5 and .keras ZIP model files.

    For .h5/.hdf5: requires h5py optional dep. Degrades gracefully if absent.
    For .keras ZIP: uses stdlib zipfile, no external deps required.
    SECURITY: Never calls keras.models.load_model() or any execution primitive.
    """

    model_structure: dict[str, object] | None = None

    def analyze(self, artifact: Artifact, config: ScanConfig | None = None) -> Sequence[Finding]:
        """Perform static analysis on a Keras artifact."""
        data = self._read_artifact_bytes(artifact)
        if not data:
            return []

        # Determine sub-format: ZIP (.keras) or HDF5 (.h5/.hdf5)
        is_zip = data[:4] == b"PK\x03\x04"

        if is_zip:
            return self._analyze_keras_zip(data, artifact)
        else:
            return self._analyze_keras_h5(data, artifact)

    def _analyze_keras_zip(self, data: bytes, artifact: Artifact) -> list[Finding]:
        """Analyze .keras ZIP format (Keras 3 / tf.keras >= 2.12)."""
        findings: list[Finding] = []
        counter = [0]
        model_format = ModelFormat.KERAS

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
                format=model_format,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                    byte_offset=0,
                    byte_length=None,
                    archive_member="config.json",
                    matched_value=matched_value[:512],
                ),
                description=description,
                remediation=_RULE_REMEDIATION[rule_id],
            )

        try:
            zf = zipfile.ZipFile(io.BytesIO(data), 'r')
        except zipfile.BadZipFile as exc:
            findings.append(Finding(
                finding_id=_next_id("KR102"),
                rule_id="KR102",
                rule_name=_RULE_NAMES["KR102"],
                severity=_RULE_SEVERITY["KR102"],
                confidence=_RULE_CONFIDENCE["KR102"],
                category=_RULE_CATEGORY["KR102"],
                format=model_format,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARTIFACT,
                    byte_offset=0, byte_length=None, archive_member=None,
                    matched_value="BAD_ZIP",
                ),
                description=f"Keras .keras ZIP file is invalid: {exc}",
                remediation=_RULE_REMEDIATION["KR102"],
            ))
            return findings

        with zf:
            namelist = zf.namelist()

            # KR103: duplicate archive member detection.
            # zipfile.read(name) resolves to the LAST matching entry while some
            # scanners inspect the FIRST — a known bypass. namelist() preserves all
            # entries including duplicates, so a plain count exposes the technique.
            seen_names: dict[str, int] = {}
            for n in namelist:
                seen_names[n] = seen_names.get(n, 0) + 1
            for dup_name, count in seen_names.items():
                if count <= 1:
                    continue
                counter[0] += 1
                findings.append(Finding(
                    finding_id=f"FINDING-KR103-{artifact.sha256[:8]}-{counter[0]:03d}",
                    rule_id="KR103",
                    rule_name=_RULE_NAMES["KR103"],
                    severity=_RULE_SEVERITY["KR103"],
                    confidence=_RULE_CONFIDENCE["KR103"],
                    category=_RULE_CATEGORY["KR103"],
                    format=model_format,
                    artifact_sha256=artifact.sha256,
                    evidence=Evidence(
                        coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                        byte_offset=0,
                        byte_length=None,
                        archive_member=dup_name,
                        matched_value=f"duplicate_member='{dup_name}' count={count}",
                    ),
                    description=(
                        f"Keras .keras ZIP contains {count} entries named '{dup_name}'. "
                        f"Python's zipfile resolves to the last entry while some scanners "
                        f"inspect the first. This asymmetry is a known scanner bypass technique."
                    ),
                    remediation=_RULE_REMEDIATION["KR103"],
                ))

            # KR102: check for suspicious (path-traversal) archive member names.
            # Python's zipfile.ZipFile.read() does not resolve paths to the
            # filesystem, so there is no scanner-side path traversal. However,
            # member names containing '..' or absolute paths indicate a
            # maliciously crafted archive intended to exploit vulnerable
            # extraction code in downstream consumers.
            for member_name in namelist:
                normalized = member_name.replace("\\", "/")
                parts = [p for p in normalized.split("/") if p]
                is_traversal = (
                    ".." in parts
                    or normalized.startswith("/")
                    or (
                        len(normalized) >= 3
                        and normalized[1] == ":"
                        and normalized[2] in ("/", "\\")
                    )
                    or any(t in member_name.lower() for t in _KERAS_ZIP_TRAVERSAL_TOKENS)
                )
                if is_traversal:
                    counter[0] += 1
                    findings.append(Finding(
                        finding_id=f"FINDING-KR102-{artifact.sha256[:8]}-{counter[0]:03d}",
                        rule_id="KR102",
                        rule_name=_RULE_NAMES["KR102"],
                        severity=_RULE_SEVERITY["KR102"],
                        confidence=_RULE_CONFIDENCE["KR102"],
                        category=_RULE_CATEGORY["KR102"],
                        format=model_format,
                        artifact_sha256=artifact.sha256,
                        evidence=Evidence(
                            coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                            byte_offset=0,
                            byte_length=None,
                            archive_member=member_name,
                            matched_value=f"suspicious_member={member_name!r}",
                        ),
                        description=(
                            f"Keras .keras ZIP contains a member with a "
                            f"path-traversal indicator: '{member_name}'. "
                            f"Archive members with '..' components or absolute paths "
                            f"can cause path traversal during extraction by vulnerable "
                            f"ZIP processing libraries."
                        ),
                        remediation=_RULE_REMEDIATION["KR102"],
                    ))

            # Look for config.json
            if "config.json" not in namelist:
                findings.append(_make_finding(
                    "KR102", "MISSING_CONFIG_JSON",
                    "Keras .keras ZIP does not contain config.json",
                ))
                return findings

            try:
                config_bytes = zf.read("config.json")
                config_obj = json.loads(config_bytes)
            except Exception as exc:
                findings.append(_make_finding(
                    "KR102", "INVALID_CONFIG_JSON",
                    f"Keras config.json is not valid JSON: {exc}",
                ))
                return findings

            # KR101: check for custom class names
            class_names = _extract_class_names(config_obj)
            custom_classes = [cn for cn in class_names if cn not in _KERAS_BUILTIN_CLASSES]
            seen_custom: set[str] = set()
            for cn in custom_classes:
                if cn not in seen_custom:
                    seen_custom.add(cn)
                    findings.append(_make_finding(
                        "KR101", f"class_name={cn!r}",
                        f"Keras config references non-standard class '{cn}'. "
                        f"Custom classes may execute arbitrary Python at load time.",
                    ))

            # KR105: scan Lambda serialized bytecode for dangerous execution indicators.
            # _kr105_decode_code_field() base64-decodes to raw bytes only — no marshal,
            # no exec, no eval.  _kr105_scan_raw_bytes() does pure bytes pattern matching.
            for lambda_path, code_val in _extract_lambda_code_fields(config_obj):
                raw = _kr105_decode_code_field(code_val)
                if raw is None:
                    continue
                payload_hits = _kr105_scan_raw_bytes(raw)
                if not payload_hits:
                    continue
                is_critical = any(sev == "CRITICAL" for _, sev in payload_hits)
                hit_sev = Severity.CRITICAL if is_critical else Severity.HIGH
                indicators = "; ".join(desc for desc, _ in payload_hits)
                counter[0] += 1
                findings.append(Finding(
                    finding_id=f"FINDING-KR105-{artifact.sha256[:8]}-{counter[0]:03d}",
                    rule_id="KR105",
                    rule_name=_RULE_NAMES["KR105"],
                    severity=hit_sev,
                    confidence=Confidence.HIGH,
                    category=_RULE_CATEGORY["KR105"],
                    format=model_format,
                    artifact_sha256=artifact.sha256,
                    evidence=Evidence(
                        coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                        byte_offset=0,
                        byte_length=None,
                        archive_member="config.json",
                        matched_value=(
                            f"lambda_path={lambda_path!r} indicators=[{indicators}]"
                        )[:512],
                    ),
                    description=(
                        f"Keras Lambda serialized at '{lambda_path}' contains dangerous "
                        f"execution indicator(s) in its bytecode: {indicators}. "
                        f"This pattern is consistent with embedded OS command execution "
                        f"or dynamic code evaluation (CVE-2024-3660 pattern)."
                    ),
                    remediation=_RULE_REMEDIATION["KR105"],
                    cve_ids=("CVE-2024-3660",),
                    cwe_ids=("CWE-94", "CWE-502"),
                ))

            # WI101/WI102/WI103: check weight data in zip members.
            # check_tensor is imported at module level (Finding 7 / 6).
            #
            # Handles two layouts used by Keras 3:
            #   1. .npy files directly in the ZIP  (Keras 3 standard — Finding 3)
            #   2. .npz members (zip-of-npy) — kept for any repos using that format
            #
            # model.weights.h5 members are silently skipped: h5py is optional and
            # there is no safe h5py-free fallback for embedded HDF5 streams.

            zip_tensor_names: list[str] = []
            for member_name in namelist:
                # --- Case 1: direct .npy member in the top-level .keras ZIP (Finding 3)
                if member_name.endswith(".npy"):
                    try:
                        npy_data = zf.read(member_name)
                        zip_tensor_names.append(member_name)
                        # .npy format: 128-byte magic + header, then raw C-contiguous array data.
                        # We skip full dtype/shape parsing and use F32 as a heuristic; the
                        # IEEE 754 bit-pattern checks are still valid for any 4-byte-aligned
                        # float32 data and will produce no false positives for other dtypes.
                        if len(npy_data) > 128:
                            hits = check_tensor(npy_data[128:], "F32")
                            for hit in hits:
                                rule_id = _KIND_TO_RULE[hit.kind]
                                counter[0] += 1
                                findings.append(Finding(
                                    finding_id=f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}",
                                    rule_id=rule_id,
                                    rule_name=_RULE_NAMES[rule_id],
                                    severity=_RULE_SEVERITY[rule_id],
                                    confidence=_RULE_CONFIDENCE[rule_id],
                                    category=_RULE_CATEGORY[rule_id],
                                    format=model_format,
                                    artifact_sha256=artifact.sha256,
                                    evidence=Evidence(
                                        coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                                        byte_offset=hit.first_offset,
                                        byte_length=None,
                                        archive_member=member_name,
                                        matched_value=f"array={member_name!r} {hit.kind}_count={hit.count}",
                                    ),
                                    description=f"Keras weight array '{member_name}' contains {hit.count} {hit.kind.upper()} values.",
                                    remediation=_RULE_REMEDIATION[rule_id],
                                ))
                    except Exception:
                        pass
                    continue

                # --- Case 2: .npz member (zip-of-npy) — kept for legacy/alternate repos
                if member_name.endswith(".npz"):
                    try:
                        member_data = zf.read(member_name)
                        try:
                            inner_zip = zipfile.ZipFile(io.BytesIO(member_data), 'r')
                            with inner_zip as iz:
                                for npy_name in iz.namelist():
                                    zip_tensor_names.append(f"{member_name}/{npy_name}")
                                    npy_data = iz.read(npy_name)
                                    if len(npy_data) > 128:
                                        hits = check_tensor(npy_data[128:], "F32")
                                        for hit in hits:
                                            rule_id = _KIND_TO_RULE[hit.kind]
                                            counter[0] += 1
                                            findings.append(Finding(
                                                finding_id=f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}",
                                                rule_id=rule_id,
                                                rule_name=_RULE_NAMES[rule_id],
                                                severity=_RULE_SEVERITY[rule_id],
                                                confidence=_RULE_CONFIDENCE[rule_id],
                                                category=_RULE_CATEGORY[rule_id],
                                                format=model_format,
                                                artifact_sha256=artifact.sha256,
                                                evidence=Evidence(
                                                    coordinate_space=CoordinateSpace.ARCHIVE_MEMBER,
                                                    byte_offset=hit.first_offset,
                                                    byte_length=None,
                                                    archive_member=f"{member_name}/{npy_name}",
                                                    matched_value=f"array={npy_name!r} {hit.kind}_count={hit.count}",
                                                ),
                                                description=f"Keras weight array '{npy_name}' contains {hit.count} {hit.kind.upper()} values.",
                                                remediation=_RULE_REMEDIATION[rule_id],
                                            ))
                        except Exception:
                            pass
                    except Exception:
                        pass

                # model.weights.h5 members are silently skipped (h5py optional, no safe fallback)

            self.model_structure = {
                "tensor_count": len(zip_tensor_names),
                "tensors": [{"name": n, "dtype": "F32", "shape": None} for n in zip_tensor_names[:1000]],
                "format_notes": "Keras ZIP (.keras): shapes not available without numpy; dtype assumed F32.",
            }

        return findings

    def _analyze_keras_h5(self, data: bytes, artifact: Artifact) -> list[Finding]:
        """Analyze Keras HDF5 format (.h5/.hdf5)."""
        try:
            import h5py  # optional dep
        except ImportError as exc:
            raise ImportError(
                "h5py is required to analyze HDF5/Keras (.h5) model files. "
                "Install it with: pip install h5py"
            ) from exc

        findings: list[Finding] = []
        counter = [0]
        model_format = ModelFormat.HDF5

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
                format=model_format,
                artifact_sha256=artifact.sha256,
                evidence=Evidence(
                    coordinate_space=CoordinateSpace.ARTIFACT,
                    byte_offset=0, byte_length=None, archive_member=None,
                    matched_value=matched_value[:512],
                ),
                description=description,
                remediation=_RULE_REMEDIATION[rule_id],
            )

        try:
            f = h5py.File(io.BytesIO(data), 'r')
        except Exception as exc:
            findings.append(_make_finding(
                "KR102", "HDF5_PARSE_ERROR",
                f"Keras HDF5 file could not be opened: {exc}",
            ))
            return findings

        with f:
            # Try to read model_config attribute
            model_config_raw = None
            for attr_name in ("model_config", "keras_version"):
                if attr_name in f.attrs:
                    model_config_raw = f.attrs.get("model_config")
                    break

            if model_config_raw is None:
                findings.append(_make_finding(
                    "KR102", "MISSING_MODEL_CONFIG",
                    "Keras HDF5 file has no model_config attribute. May not be a Keras model.",
                ))
            else:
                try:
                    if isinstance(model_config_raw, bytes):
                        config_str = model_config_raw.decode("utf-8")
                    else:
                        config_str = str(model_config_raw)
                    config_obj = json.loads(config_str)

                    class_names = _extract_class_names(config_obj)
                    custom_classes = [cn for cn in class_names if cn not in _KERAS_BUILTIN_CLASSES]
                    seen_custom: set[str] = set()
                    for cn in custom_classes:
                        if cn not in seen_custom:
                            seen_custom.add(cn)
                            findings.append(_make_finding(
                                "KR101", f"class_name={cn!r}",
                                f"Keras HDF5 model_config references non-standard class '{cn}'. "
                                f"Custom classes may execute arbitrary Python at load time.",
                            ))

                    # KR105: scan Lambda serialized bytecode (HDF5 path).
                    # Same static-only approach as the ZIP path above.
                    for lambda_path, code_val in _extract_lambda_code_fields(config_obj):
                        raw = _kr105_decode_code_field(code_val)
                        if raw is None:
                            continue
                        payload_hits = _kr105_scan_raw_bytes(raw)
                        if not payload_hits:
                            continue
                        is_critical = any(sev == "CRITICAL" for _, sev in payload_hits)
                        hit_sev = Severity.CRITICAL if is_critical else Severity.HIGH
                        indicators = "; ".join(desc for desc, _ in payload_hits)
                        counter[0] += 1
                        findings.append(Finding(
                            finding_id=f"FINDING-KR105-{artifact.sha256[:8]}-{counter[0]:03d}",
                            rule_id="KR105",
                            rule_name=_RULE_NAMES["KR105"],
                            severity=hit_sev,
                            confidence=Confidence.HIGH,
                            category=_RULE_CATEGORY["KR105"],
                            format=model_format,
                            artifact_sha256=artifact.sha256,
                            evidence=Evidence(
                                coordinate_space=CoordinateSpace.ARTIFACT,
                                byte_offset=0,
                                byte_length=None,
                                archive_member=None,
                                matched_value=(
                                    f"lambda_path={lambda_path!r} indicators=[{indicators}]"
                                )[:512],
                            ),
                            description=(
                                f"Keras Lambda serialized at '{lambda_path}' contains dangerous "
                                f"execution indicator(s) in its bytecode: {indicators}. "
                                f"This pattern is consistent with embedded OS command execution "
                                f"or dynamic code evaluation (CVE-2024-3660 pattern)."
                            ),
                            remediation=_RULE_REMEDIATION["KR105"],
                            cve_ids=("CVE-2024-3660",),
                            cwe_ids=("CWE-94", "CWE-502"),
                        ))
                except Exception as exc:
                    findings.append(_make_finding(
                        "KR102", "INVALID_MODEL_CONFIG",
                        f"Keras model_config attribute is not valid JSON: {exc}",
                    ))

            # WI101/WI102/WI103: scan weight datasets
            # check_tensor is imported at module level (Finding 7)

            dtype_map: dict[str, str] = {
                "float32": "F32", "float16": "F16", "bfloat16": "BF16", "float64": "F64",
                "<f4": "F32", "<f2": "F16", "<f8": "F64",
                ">f4": "F32", ">f2": "F16", ">f8": "F64",
            }

            h5_tensor_inventory: list[dict[str, object]] = []

            def _check_group(group: object, h5_path: str = "/", depth: int = 0) -> None:
                if depth > 10:
                    return
                try:
                    keys = list(group.keys())  # type: ignore[union-attr]
                except Exception:
                    return
                for key in keys:
                    item_path = ("/" + key) if h5_path == "/" else (h5_path + "/" + key)

                    # KR104 — ExternalLink detection.
                    # group.get(key, getlink=True) returns the raw link object WITHOUT
                    # following it. This is the only safe way to detect external links
                    # statically: calling group[key] directly would open the external file.
                    try:
                        link = group.get(key, getlink=True)  # type: ignore[union-attr]
                    except Exception:
                        link = None

                    if isinstance(link, h5py.ExternalLink):
                        ext_file = str(link.filename)
                        ext_path = str(link.path)
                        sev = _kr104_severity(ext_file)
                        flags = (
                            (" [ABSOLUTE_PATH]" if _kr104_is_absolute(ext_file) else "")
                            + (" [PATH_TRAVERSAL]" if _kr104_has_traversal(ext_file) else "")
                        )
                        counter[0] += 1
                        findings.append(Finding(
                            finding_id=f"FINDING-KR104-{artifact.sha256[:8]}-{counter[0]:03d}",
                            rule_id="KR104",
                            rule_name=_RULE_NAMES["KR104"],
                            severity=sev,
                            confidence=Confidence.HIGH,
                            category=_RULE_CATEGORY["KR104"],
                            format=model_format,
                            artifact_sha256=artifact.sha256,
                            evidence=Evidence(
                                coordinate_space=CoordinateSpace.ARTIFACT,
                                byte_offset=0,
                                byte_length=None,
                                archive_member=None,
                                matched_value=(
                                    f"external_link h5_path={item_path!r} "
                                    f"file={ext_file!r} ext_path={ext_path!r}{flags}"
                                )[:512],
                            ),
                            description=(
                                f"HDF5 ExternalLink at '{item_path}' references external file "
                                f"'{ext_file}' (HDF5 path: '{ext_path}'). Loading this model "
                                f"will attempt to open the external file, enabling arbitrary "
                                f"file read on the host system."
                            ),
                            remediation=_RULE_REMEDIATION["KR104"],
                            cve_ids=("CVE-2026-1669",),
                            cwe_ids=("CWE-73", "CWE-200"),
                        ))
                        continue  # Never follow ExternalLink

                    try:
                        item = group[key]  # type: ignore[index]
                    except Exception:
                        continue

                    if hasattr(item, 'shape') and hasattr(item, 'dtype'):
                        # KR104 — VDS (Virtual Dataset Source) detection.
                        # item.is_virtual and item.virtual_sources() read HDF5 link
                        # metadata only; they do NOT open or read from external files.
                        # Only item[()] would follow the VDS mapping — we never call it.
                        if getattr(item, 'is_virtual', False):
                            try:
                                for vs in item.virtual_sources():
                                    vds_file = str(vs.file_name)
                                    vds_dset = str(vs.dset_name)
                                    if vds_file == ".":
                                        continue  # same-file VDS is not an external ref
                                    sev = _kr104_severity(vds_file)
                                    flags = (
                                        (" [ABSOLUTE_PATH]" if _kr104_is_absolute(vds_file) else "")
                                        + (" [PATH_TRAVERSAL]" if _kr104_has_traversal(vds_file) else "")
                                    )
                                    counter[0] += 1
                                    findings.append(Finding(
                                        finding_id=f"FINDING-KR104-{artifact.sha256[:8]}-{counter[0]:03d}",
                                        rule_id="KR104",
                                        rule_name=_RULE_NAMES["KR104"],
                                        severity=sev,
                                        confidence=Confidence.HIGH,
                                        category=_RULE_CATEGORY["KR104"],
                                        format=model_format,
                                        artifact_sha256=artifact.sha256,
                                        evidence=Evidence(
                                            coordinate_space=CoordinateSpace.ARTIFACT,
                                            byte_offset=0,
                                            byte_length=None,
                                            archive_member=None,
                                            matched_value=(
                                                f"vds_source h5_path={item_path!r} "
                                                f"file={vds_file!r} dataset={vds_dset!r}{flags}"
                                            )[:512],
                                        ),
                                        description=(
                                            f"HDF5 Virtual Dataset at '{item_path}' maps data from "
                                            f"external source '{vds_file}:{vds_dset}'. Loading this "
                                            f"model will attempt to read data from the external file "
                                            f"(CVE-2026-1669, CWE-73)."
                                        ),
                                        remediation=_RULE_REMEDIATION["KR104"],
                                        cve_ids=("CVE-2026-1669",),
                                        cwe_ids=("CWE-73", "CWE-200"),
                                    ))
                            except Exception:
                                pass
                            continue  # Never read VDS data via item[()]

                        # It's a regular (non-VDS) dataset
                        dtype_name = str(item.dtype.name) if hasattr(item.dtype, 'name') else str(item.dtype)
                        dtype_str = dtype_map.get(dtype_name) or dtype_map.get(str(item.dtype))
                        shape = list(item.shape) if hasattr(item, 'shape') else None
                        if len(h5_tensor_inventory) < 1000:
                            h5_tensor_inventory.append({
                                "name": key,
                                "dtype": dtype_str or dtype_name,
                                "shape": shape,
                            })
                        if dtype_str is None:
                            continue
                        try:
                            # Read raw bytes WITHOUT creating numpy arrays — use tobytes()
                            raw = item[()].tobytes()
                            hits = check_tensor(raw, dtype_str)
                        except Exception:
                            continue
                        for hit in hits:
                            rule_id = _KIND_TO_RULE[hit.kind]
                            counter[0] += 1
                            findings.append(Finding(
                                finding_id=f"FINDING-{rule_id}-{artifact.sha256[:8]}-{counter[0]:03d}",
                                rule_id=rule_id,
                                rule_name=_RULE_NAMES[rule_id],
                                severity=_RULE_SEVERITY[rule_id],
                                confidence=_RULE_CONFIDENCE[rule_id],
                                category=_RULE_CATEGORY[rule_id],
                                format=model_format,
                                artifact_sha256=artifact.sha256,
                                evidence=Evidence(
                                    coordinate_space=CoordinateSpace.ARTIFACT,
                                    byte_offset=hit.first_offset,
                                    byte_length=None,
                                    archive_member=None,
                                    matched_value=f"dataset={key!r} dtype={dtype_str} {hit.kind}_count={hit.count}",
                                ),
                                description=f"Keras HDF5 weight dataset '{key}' ({dtype_str}) contains {hit.count} {hit.kind.upper()} values.",
                                remediation=_RULE_REMEDIATION[rule_id],
                            ))
                    elif hasattr(item, 'keys'):
                        # It's a group — recurse, passing the accumulated HDF5 path
                        _check_group(item, item_path, depth + 1)

            _check_group(f)

            self.model_structure = {
                "tensor_count": len(h5_tensor_inventory),
                "tensors": h5_tensor_inventory,
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


__all__ = ["KerasAnalyzer"]
