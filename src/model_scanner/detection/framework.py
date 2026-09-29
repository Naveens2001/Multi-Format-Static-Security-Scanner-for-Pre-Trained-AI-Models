"""
model_scanner.detection.framework
===================================

Static framework identification for pre-trained model artifacts (A1).

Infers the originating ML framework from evidence already extracted during
format detection and analysis — never from file extension alone.  Returns a
``FrameworkIdentification`` with an explicit confidence level and the
evidence list that drove the conclusion.

SECURITY GUARANTEE
------------------
This module NEVER:
  - Imports torch, transformers, tensorflow, onnx, flax, or similar.
  - Calls eval() / exec() / pickle.loads() / torch.load().
  - Makes network calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Framework(StrEnum):
    """Known ML frameworks whose artifacts this scanner may encounter."""

    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"
    KERAS = "Keras"
    JAX_FLAX = "JAX/Flax"
    SKLEARN = "scikit-learn"
    XGBOOST = "XGBoost"
    LIGHTGBM = "LightGBM"
    ONNX_RUNTIME = "ONNX Runtime"
    NUMPY = "NumPy"
    UNKNOWN = "UNKNOWN"


class FrameworkConfidence(StrEnum):
    """Confidence level of a framework identification."""

    HIGH = "HIGH"      # Structural / opcode evidence is unambiguous
    MEDIUM = "MEDIUM"  # Multiple consistent signals but no single definitive one
    LOW = "LOW"        # Single weak signal or heuristic extension guess


@dataclass(frozen=True)
class FrameworkIdentification:
    """Result of static framework inference.

    Attributes
    ----------
    framework:
        Inferred framework, or ``Framework.UNKNOWN`` when evidence is
        insufficient.
    confidence:
        How certain the inference is.
    evidence:
        Human-readable list of evidence strings that drove the conclusion.
    """

    framework: Framework
    confidence: FrameworkConfidence
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def unknown() -> FrameworkIdentification:
        """Return the canonical UNKNOWN result (no evidence, no guess)."""
        return FrameworkIdentification(
            framework=Framework.UNKNOWN,
            confidence=FrameworkConfidence.LOW,
            evidence=(),
        )


# ---------------------------------------------------------------------------
# Module-name sets for pickle GLOBAL / STACK_GLOBAL opcode matching
# ---------------------------------------------------------------------------

_PYTORCH_MODULES: frozenset[str] = frozenset(
    [
        "torch",
        "torch._utils",
        "torch.storage",
        "torch.nn",
        "torch.nn.modules",
        "torch.nn.parameter",
        "torch._tensor",
        "torch.serialization",
    ]
)

_SKLEARN_MODULES: frozenset[str] = frozenset(
    [
        "sklearn",
        "sklearn.linear_model",
        "sklearn.ensemble",
        "sklearn.svm",
        "sklearn.tree",
        "sklearn.pipeline",
        "sklearn.preprocessing",
        "sklearn.utils",
        "sklearn.externals",
    ]
)

_NUMPY_MODULES: frozenset[str] = frozenset(
    [
        "numpy",
        "numpy.core",
        "numpy.core.multiarray",
        "numpy.core.numeric",
        "numpy.ma",
    ]
)

_XGBOOST_MODULES: frozenset[str] = frozenset(["xgboost", "xgboost.core", "xgboost.sklearn"])

_LIGHTGBM_MODULES: frozenset[str] = frozenset(["lightgbm", "lightgbm.basic", "lightgbm.sklearn"])


def _starts_with_any(module: str, prefixes: frozenset[str]) -> bool:
    """Return True if ``module`` matches or starts with any prefix in ``prefixes``."""
    if module in prefixes:
        return True
    return any(module.startswith(p + ".") for p in prefixes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def infer_from_pickle_globals(
    global_modules: list[str],
) -> FrameworkIdentification:
    """Infer framework from module names seen in GLOBAL/STACK_GLOBAL opcodes.

    This is used for RAW_PICKLE and PYTORCH_ZIP artifacts where the
    disassembler has already extracted the module names.

    Parameters
    ----------
    global_modules:
        List of module name strings from GLOBAL/STACK_GLOBAL opcodes.

    Returns
    -------
    FrameworkIdentification
    """
    if not global_modules:
        return FrameworkIdentification.unknown()

    evidence: list[str] = []
    scores: dict[Framework, int] = {}

    for mod in global_modules:
        if _starts_with_any(mod, _PYTORCH_MODULES):
            scores[Framework.PYTORCH] = scores.get(Framework.PYTORCH, 0) + 1
            evidence.append(f"pickle GLOBAL: {mod!r}")
        elif _starts_with_any(mod, _SKLEARN_MODULES):
            scores[Framework.SKLEARN] = scores.get(Framework.SKLEARN, 0) + 1
            evidence.append(f"pickle GLOBAL: {mod!r}")
        elif _starts_with_any(mod, _XGBOOST_MODULES):
            scores[Framework.XGBOOST] = scores.get(Framework.XGBOOST, 0) + 1
            evidence.append(f"pickle GLOBAL: {mod!r}")
        elif _starts_with_any(mod, _LIGHTGBM_MODULES):
            scores[Framework.LIGHTGBM] = scores.get(Framework.LIGHTGBM, 0) + 1
            evidence.append(f"pickle GLOBAL: {mod!r}")
        elif _starts_with_any(mod, _NUMPY_MODULES):
            scores[Framework.NUMPY] = scores.get(Framework.NUMPY, 0) + 1
            evidence.append(f"pickle GLOBAL (numpy): {mod!r}")

    if not scores:
        return FrameworkIdentification.unknown()

    best_framework = max(scores, key=lambda f: scores[f])
    total_hits = scores[best_framework]
    confidence = FrameworkConfidence.HIGH if total_hits >= 2 else FrameworkConfidence.MEDIUM

    return FrameworkIdentification(
        framework=best_framework,
        confidence=confidence,
        evidence=tuple(evidence),
    )


def infer_from_pytorch_zip(
    member_names: list[str],
    pickle_global_modules: list[str],
) -> FrameworkIdentification:
    """Infer framework from a PyTorch ZIP archive's member structure.

    Parameters
    ----------
    member_names:
        Names of files within the ZIP container.
    pickle_global_modules:
        Module names from GLOBAL opcodes inside ``archive/data.pkl``.

    Returns
    -------
    FrameworkIdentification
    """
    evidence: list[str] = []

    # Canonical PyTorch ZIP structure: archive/data.pkl + archive/version
    has_data_pkl = any("data.pkl" in n for n in member_names)
    has_version = any("version" in n for n in member_names)

    if has_data_pkl:
        evidence.append("ZIP contains archive/data.pkl (canonical PyTorch structure)")
    if has_version:
        evidence.append("ZIP contains archive/version (canonical PyTorch structure)")

    # PyTorch-specific rebuild function in globals
    rebuild_hit = any(
        "_rebuild_tensor" in m or "_rebuild_parameter" in m
        for m in pickle_global_modules
    )
    if rebuild_hit:
        evidence.append(
            "pickle GLOBAL: torch._utils._rebuild_tensor_v2 or similar"
        )

    if has_data_pkl and (has_version or rebuild_hit):
        return FrameworkIdentification(
            framework=Framework.PYTORCH,
            confidence=FrameworkConfidence.HIGH,
            evidence=tuple(evidence),
        )

    if evidence:
        return FrameworkIdentification(
            framework=Framework.PYTORCH,
            confidence=FrameworkConfidence.MEDIUM,
            evidence=tuple(evidence),
        )

    # Fall through to global-module heuristic
    from_globals = infer_from_pickle_globals(pickle_global_modules)
    if from_globals.framework != Framework.UNKNOWN:
        return from_globals

    return FrameworkIdentification.unknown()


def infer_from_safetensors_metadata(
    metadata: dict[str, str],
    dtype_set: set[str],
) -> FrameworkIdentification:
    """Infer framework from SafeTensors ``__metadata__`` key/value pairs.

    Parameters
    ----------
    metadata:
        The ``__metadata__`` dictionary from the SafeTensors header.
    dtype_set:
        Set of dtype strings seen in tensor descriptors.

    Returns
    -------
    FrameworkIdentification
    """
    evidence: list[str] = []

    fmt = metadata.get("format", "").lower()
    if fmt == "pt":
        evidence.append("__metadata__[\"format\"] = \"pt\" → PyTorch")
        return FrameworkIdentification(
            framework=Framework.PYTORCH,
            confidence=FrameworkConfidence.HIGH,
            evidence=tuple(evidence),
        )
    if fmt == "tf":
        evidence.append("__metadata__[\"format\"] = \"tf\" → TensorFlow")
        return FrameworkIdentification(
            framework=Framework.TENSORFLOW,
            confidence=FrameworkConfidence.HIGH,
            evidence=tuple(evidence),
        )
    if fmt == "flax":
        evidence.append("__metadata__[\"format\"] = \"flax\" → JAX/Flax")
        return FrameworkIdentification(
            framework=Framework.JAX_FLAX,
            confidence=FrameworkConfidence.HIGH,
            evidence=tuple(evidence),
        )
    if fmt == "np":
        evidence.append("__metadata__[\"format\"] = \"np\" → NumPy")
        return FrameworkIdentification(
            framework=Framework.NUMPY,
            confidence=FrameworkConfidence.HIGH,
            evidence=tuple(evidence),
        )

    # BF16 dtype is predominantly PyTorch / JAX
    if "BF16" in dtype_set:
        evidence.append("dtype BF16 present (predominantly PyTorch/JAX/Flax)")
        return FrameworkIdentification(
            framework=Framework.PYTORCH,
            confidence=FrameworkConfidence.MEDIUM,
            evidence=tuple(evidence),
        )

    return FrameworkIdentification.unknown()


def infer_from_gguf_metadata(
    kv_pairs: dict[str, object],
) -> FrameworkIdentification:
    """Infer framework from GGUF metadata key/value pairs.

    Parameters
    ----------
    kv_pairs:
        Dictionary of GGUF metadata KV pairs (string keys, arbitrary values).

    Returns
    -------
    FrameworkIdentification
    """
    evidence: list[str] = []

    architecture = kv_pairs.get("general.architecture", "")
    if isinstance(architecture, str) and architecture:
        evidence.append(f"GGUF general.architecture = {architecture!r}")

    file_type = kv_pairs.get("general.file_type")
    if file_type is not None:
        evidence.append(f"GGUF general.file_type = {file_type!r}")

    quant_keys = [k for k in kv_pairs if "quantization" in k.lower()]
    if quant_keys:
        evidence.append(f"GGUF quantization KVs: {quant_keys[:3]}")

    if evidence:
        # GGUF is universally derived from llama.cpp / ggml tooling
        return FrameworkIdentification(
            framework=Framework.PYTORCH,
            confidence=FrameworkConfidence.MEDIUM,
            evidence=(*tuple(evidence), "GGUF is typically converted from PyTorch weights"),
        )

    return FrameworkIdentification.unknown()


__all__ = [
    "Framework",
    "FrameworkConfidence",
    "FrameworkIdentification",
    "infer_from_gguf_metadata",
    "infer_from_pickle_globals",
    "infer_from_pytorch_zip",
    "infer_from_safetensors_metadata",
]
