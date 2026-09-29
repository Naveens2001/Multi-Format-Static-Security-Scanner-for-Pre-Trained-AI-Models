"""
model_scanner.analyzers.pickle.catalog
======================================

Static security catalog of dangerous, suspicious, and benign global imports.

SECURITY GUARANTEE
------------------
Contains static string mappings only. NEVER dynamically imports or resolves
modules/attributes to evaluate danger.

Design Reference
----------------
Section 34.1, 35.1 of docs/implementation_design.md
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum


class GlobalCategory(StrEnum):
    """Categorization of a static global identifier."""

    DANGEROUS = "DANGEROUS"      # Arbitrary code execution, shell injection, exfiltration
    SUSPICIOUS = "SUSPICIOUS"    # Dynamic loaders, code construct mutation, reflection
    SAFE = "SAFE"                # Standard PyTorch/NumPy tensor rebuilders & data types
    UNKNOWN = "UNKNOWN"          # Unclassified third-party or custom class


# ---------------------------------------------------------------------------
# Catalog Mappings (module, name)
# ---------------------------------------------------------------------------

_DANGEROUS_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    [
        # Shell and process execution
        ("os", "system"),
        ("os", "popen"),
        ("os", "popen2"),
        ("os", "popen3"),
        ("os", "popen4"),
        ("os", "execl"),
        ("os", "execle"),
        ("os", "execlp"),
        ("os", "execlpe"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "execvpe"),
        ("os", "spawnl"),
        ("os", "spawnle"),
        ("os", "spawnlp"),
        ("os", "spawnlpe"),
        ("os", "spawnv"),
        ("os", "spawnve"),
        ("os", "spawnvp"),
        ("os", "spawnvpe"),
        ("os", "startfile"),
        ("os", "kill"),
        ("os", "fork"),
        ("os", "remove"),
        ("os", "unlink"),
        ("os", "rmdir"),
        ("os", "removedirs"),
        ("posix", "system"),
        ("posix", "popen"),
        ("posix", "execv"),
        ("posix", "execve"),
        ("nt", "system"),
        ("nt", "popen"),
        ("nt", "spawnv"),
        ("nt", "spawnve"),
        ("nt", "_execv"),
        ("nt", "_execve"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "check_call"),
        ("subprocess", "check_output"),
        ("subprocess", "run"),
        ("subprocess", "getoutput"),
        ("subprocess", "getstatusoutput"),
        ("pty", "spawn"),
        ("commands", "getoutput"),
        ("commands", "getstatusoutput"),
        # Code evaluation / compilation
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("builtins", "__import__"),
        ("builtins", "open"),
        ("__builtin__", "eval"),
        ("__builtin__", "exec"),
        ("__builtin__", "compile"),
        ("__builtin__", "__import__"),
        ("__builtin__", "open"),
        ("__builtin__", "file"),
        ("__builtin__", "execfile"),
        # Network & socket exfiltration
        ("socket", "socket"),
        ("socket", "create_connection"),
        ("urllib.request", "urlopen"),
        ("urllib.request", "urlretrieve"),
        ("urllib2", "urlopen"),
        ("requests", "get"),
        ("requests", "post"),
        ("requests", "request"),
        ("http.client", "HTTPConnection"),
        ("http.client", "HTTPSConnection"),
        ("httplib", "HTTPConnection"),
        ("httplib", "HTTPSConnection"),
    ]
)

_SUSPICIOUS_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    [
        # Dynamic import & module inspection
        ("importlib", "import_module"),
        ("importlib", "__import__"),
        ("sys", "modules"),
        ("sys", "exit"),
        ("sys", "setrecursionlimit"),
        # Reflection & object modification
        ("builtins", "getattr"),
        ("builtins", "setattr"),
        ("builtins", "delattr"),
        ("builtins", "globals"),
        ("builtins", "locals"),
        ("__builtin__", "getattr"),
        ("__builtin__", "setattr"),
        ("__builtin__", "delattr"),
        ("__builtin__", "globals"),
        ("__builtin__", "locals"),
        ("types", "FunctionType"),
        ("types", "CodeType"),
        ("types", "MethodType"),
        ("types", "LambdaType"),
        # FFI & Native bindings
        ("ctypes", "CDLL"),
        ("ctypes", "cdll"),
        ("ctypes", "windll"),
        ("ctypes", "oledll"),
        ("ctypes", "PyDLL"),
        ("ctypes", "pydll"),
        # File destruction / archiving
        ("shutil", "rmtree"),
        ("shutil", "copy"),
        ("shutil", "move"),
        ("tempfile", "NamedTemporaryFile"),
        ("webbrowser", "open"),
        # Specific PyTorch dynamic loaders that MUST NOT be caught by the
        # broad torch.* safe-namespace wildcard. These override the wildcard
        # because SUSPICIOUS catalog is checked before the torch.* fallback.
        ("torch.jit", "load"),
        ("torch.hub", "load"),
        ("torch.serialization", "load"),
        # Dynamic execution primitives
        ("runpy", "_run_code"),
        ("runpy", "run_module"),
        ("runpy", "run_path"),
        ("runpy", "_run_module_as_main"),
    ]
)

_SAFE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    [
        # PyTorch Tensor reconstruction
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_parameter_with_state"),
        ("torch._utils", "_rebuild_qtensor"),
        ("torch._utils", "_rebuild_sparse_tensor"),
        ("torch", "FloatStorage"),
        ("torch", "DoubleStorage"),
        ("torch", "HalfStorage"),
        ("torch", "BFloat16Storage"),
        ("torch", "IntStorage"),
        ("torch", "LongStorage"),
        ("torch", "ShortStorage"),
        ("torch", "CharStorage"),
        ("torch", "ByteStorage"),
        ("torch", "BoolStorage"),
        ("torch", "UntypedStorage"),
        ("torch.storage", "_load_from_bytes"),
        ("torch.nn.modules.container", "Sequential"),
        ("torch.nn.modules.container", "ModuleList"),
        ("torch.nn.modules.container", "ModuleDict"),
        # Collections & primitives
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "Counter"),
        ("collections", "deque"),
        ("builtins", "set"),
        ("builtins", "list"),
        ("builtins", "dict"),
        ("builtins", "tuple"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "bool"),
        ("builtins", "complex"),
        ("builtins", "slice"),
        ("__builtin__", "set"),
        ("__builtin__", "list"),
        ("__builtin__", "dict"),
        ("__builtin__", "tuple"),
        ("__builtin__", "int"),
        ("__builtin__", "float"),
        ("__builtin__", "str"),
        ("__builtin__", "bool"),
        # NumPy data structures
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy", "core"),
    ]
)


def classify_global(
    module: str,
    name: str,
    allowlist: Sequence[str] = (),
    blocklist: Sequence[str] = (),
) -> tuple[GlobalCategory, str]:
    """Classify a global import symbol (module, name) into a GlobalCategory.

    Precedence:
    1. Custom blocklist (configured by user) -> DANGEROUS
    2. Built-in DANGEROUS catalog -> DANGEROUS
    3. Built-in SUSPICIOUS catalog -> SUSPICIOUS
    4. Custom allowlist (configured by user) -> SAFE
    5. Built-in SAFE catalog -> SAFE
    6. Default / unknown -> UNKNOWN

    Parameters
    ----------
    module:
        Target module name e.g. "os", "torch._utils".
    name:
        Target attribute/class name e.g. "system", "_rebuild_tensor".
    allowlist:
        Optional sequence of user-allowed module prefixes.
    blocklist:
        Optional sequence of user-blocked module prefixes.

    Returns
    -------
    tuple[GlobalCategory, str]
        Category enum and human-readable classification reason.
    """
    full_name = f"{module}.{name}" if module and name else (module or name)

    # 1. Custom blocklist check
    for blocked in blocklist:
        if module == blocked or full_name.startswith(blocked):
            return (
                GlobalCategory.DANGEROUS,
                f"Matches user-configured blocklist: {blocked!r}",
            )

    # 2. Built-in dangerous catalog
    if (module, name) in _DANGEROUS_GLOBALS:
        return (
            GlobalCategory.DANGEROUS,
            f"Matched known dangerous code execution primitive: {full_name!r}",
        )

    # 3. Built-in suspicious catalog
    if (module, name) in _SUSPICIOUS_GLOBALS:
        return (
            GlobalCategory.SUSPICIOUS,
            f"Matched suspicious reflection or dynamic loading primitive: {full_name!r}",
        )

    # 4. Custom allowlist check
    for allowed in allowlist:
        if module == allowed or full_name.startswith(allowed):
            return (
                GlobalCategory.SAFE,
                f"Matches user-configured allowlist: {allowed!r}",
            )

    # 5. Built-in safe catalog
    if (module, name) in _SAFE_GLOBALS:
        return (
            GlobalCategory.SAFE,
            f"Matched standard safe serialization global: {full_name!r}",
        )

    # Safe PyTorch namespace check
    is_safe_torch = module.startswith("torch.") and not (
        module.startswith("torch.distributed") or module.startswith("torch.testing")
    )
    if is_safe_torch:
        return GlobalCategory.SAFE, f"Standard PyTorch model namespace: {full_name!r}"

    return GlobalCategory.UNKNOWN, f"Unclassified third-party global: {full_name!r}"


# ---------------------------------------------------------------------------
# BUILD state-key classification (P106)
# ---------------------------------------------------------------------------
#
# A BUILD opcode applies a state dict to an object. Its keys are ordinary
# attribute names - "out_features", "return_indices", "training" - on every
# legitimate torch.nn.Module. Treating any non-SAFE key as suspicious made
# P106 fire on every standard checkpoint and buried real findings under
# dozens of false positives.
#
# P106's documented detection condition is "dangerous keys or __setstate__",
# so only those actually fire now.

#: Dunder / state keys that genuinely subvert deserialization.
DANGEROUS_STATE_KEYS: frozenset[str] = frozenset(
    [
        "__setstate__",
        "__reduce__",
        "__reduce_ex__",
        "__class__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__func__",
        "__self__",
        "__closure__",
        "__init__",
        "__new__",
        "__getattr__",
        "__getattribute__",
        "__dict__",
        "__module__",
        "__import__",
        "__loader__",
        "__spec__",
        "func_globals",
        "func_code",
        "gi_frame",
        "cr_frame",
    ]
)


def is_dangerous_state_key(target: str) -> bool:
    """True when a BUILD target names a state key capable of subverting load.

    Parameters
    ----------
    target:
        The BUILD target repr, e.g. "out_features", "__setstate__",
        "os.system", or "84".

    Returns
    -------
    bool
        True only for dunder/state keys that can redirect execution.
    """
    if not target:
        return False
    leaf = target.rsplit(".", 1)[-1]
    return leaf in DANGEROUS_STATE_KEYS


def is_plain_state_key(target: str) -> bool:
    """True when a BUILD target is an ordinary attribute name, not a symbol.

    A bare identifier with no module path ("out_features") is a state-dict key.
    A numeric literal ("84") is a memo operand that never resolved to a name.
    Neither is a global import, so neither should be classified as one.

    Parameters
    ----------
    target:
        The BUILD target repr.

    Returns
    -------
    bool
        True for bare identifiers and numeric operands.
    """
    if not target:
        return False
    if "." in target:
        return False
    if is_dangerous_state_key(target):
        return False
    # Numeric operand - the disassembler could not resolve a name here.
    if target.lstrip("-").isdigit():
        return True
    return target.isidentifier()


__all__ = [
    "DANGEROUS_STATE_KEYS",
    "GlobalCategory",
    "classify_global",
    "is_dangerous_state_key",
    "is_plain_state_key",
]
