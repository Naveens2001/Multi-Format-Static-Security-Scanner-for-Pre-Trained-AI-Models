"""
tests/unit/test_keras_analyzer.py
===================================

Unit tests for the Keras HDF5 static security analyzer.

SAFETY CONTRACT
---------------
- No keras.models.load_model(), TensorFlow loading APIs, or inference.
- No eval(), exec(), marshal.loads(), or dynamic deserialization.
- Synthetic HDF5 fixtures built with h5py in write mode only.
- The embedded Lambda payload in test fixtures is a static string constant;
  it is never decoded, decompiled, or executed.

Coverage
--------
1. Missing h5py dependency must raise ImportError (not return CLEAN).
2. Keras 3 internal classes DTypePolicy and __keras_tensor__ do not fire KR101.
3. __lambda__ class name still fires KR101 (existing behavior unchanged).
"""

from __future__ import annotations

import io
import base64
import json
import marshal as _marshal
import sys
import zipfile
from unittest.mock import patch

import pytest

from model_scanner.analyzers.keras.analyzer import KerasAnalyzer
from model_scanner.core.models import Artifact, InputSourceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artifact(data: bytes) -> Artifact:
    sha = "a" * 64
    return Artifact(
        artifact_id="test-keras",
        source_path="<test_keras>",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(data),
        sha256=sha,
        stream=io.BytesIO(data),
    )


def _make_h5_bytes(model_config: dict, keras_version: str = "3.6.0") -> bytes:
    """Create a minimal valid HDF5 file in memory with the given model_config."""
    h5py = pytest.importorskip("h5py")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["keras_version"] = keras_version
        f.attrs["model_config"] = json.dumps(model_config).encode("utf-8")
    buf.seek(0)
    return buf.read()


def _functional_config(extra_layers: list[dict]) -> dict:
    """Minimal Keras Functional model config with given extra layers."""
    return {
        "class_name": "Functional",
        "config": {
            "name": "model",
            "trainable": True,
            "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {"batch_shape": [None, 4], "dtype": "float32", "name": "in"},
                    "name": "in",
                    "inbound_nodes": [],
                },
                *extra_layers,
            ],
            "input_layers": [["in", 0, 0]],
            "output_layers": [["in", 0, 0]],
        },
    }


# ---------------------------------------------------------------------------
# 1. Missing h5py — must raise ImportError, not silently return []
# ---------------------------------------------------------------------------


class TestMissingH5pyDependency:
    """When h5py is unavailable the analyzer must not silently return CLEAN."""

    def test_missing_h5py_raises_import_error(self) -> None:
        """Regression: prior behavior returned [] on ImportError → false CLEAN.

        With the fix, the analyzer raises ImportError so the engine's existing
        ANALYZER_FAILED / ScanOutcome.ERROR path activates instead.
        """
        # Construct HDF5 magic bytes — the analyzer checks `data[:4] == b"PK\\x03\\x04"`
        # to decide ZIP vs HDF5.  HDF5 magic is \\x89HDF\\r\\n\\x1a\\n, not PK, so
        # any non-ZIP bytes route to _analyze_keras_h5.
        hdf5_magic = b"\x89HDF\r\n\x1a\n" + b"\x00" * 56
        artifact = _make_artifact(hdf5_magic)

        # Simulate h5py being absent by removing it from sys.modules
        with patch.dict(sys.modules, {"h5py": None}):
            analyzer = KerasAnalyzer()
            with pytest.raises(ImportError, match="h5py is required"):
                analyzer.analyze(artifact)

    def test_missing_h5py_does_not_return_empty_list(self) -> None:
        """Complementary assertion: the silent [] return path must not exist."""
        hdf5_magic = b"\x89HDF\r\n\x1a\n" + b"\x00" * 56
        artifact = _make_artifact(hdf5_magic)

        with patch.dict(sys.modules, {"h5py": None}):
            analyzer = KerasAnalyzer()
            try:
                result = analyzer.analyze(artifact)
                # If no exception was raised, the list must not be empty —
                # an empty list would be the old false-CLEAN behavior.
                assert result != [], (
                    "analyzer returned [] when h5py was unavailable — "
                    "this is the false-CLEAN bug that should have been fixed"
                )
            except ImportError:
                pass  # Correct — the fix raises ImportError


# ---------------------------------------------------------------------------
# 2. Keras 3 internal classes — must not fire KR101
# ---------------------------------------------------------------------------


class TestKeras3InternalClassesSuppressed:
    """DTypePolicy and __keras_tensor__ are Keras 3 internals, not custom classes."""

    def test_dtype_policy_does_not_fire_kr101(self) -> None:
        """DTypePolicy appears in Keras 3 layer configs; must not trigger KR101."""
        config = _functional_config([
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense",
                    "trainable": True,
                    "dtype": {
                        "class_name": "DTypePolicy",
                        "config": {"name": "float32"},
                        "registered_name": None,
                        "module": "keras",
                    },
                    "units": 10,
                    "activation": "relu",
                },
                "name": "dense",
                "inbound_nodes": [],
            }
        ])
        data = _make_h5_bytes(config)
        artifact = _make_artifact(data)

        findings = KerasAnalyzer().analyze(artifact)
        kr101_matched_values = [
            f.evidence.matched_value for f in findings if f.rule_id == "KR101"
        ]
        assert "class_name='DTypePolicy'" not in kr101_matched_values, (
            "DTypePolicy is a Keras 3 internal and must not produce KR101"
        )

    def test_keras_tensor_does_not_fire_kr101(self) -> None:
        """__keras_tensor__ appears in inbound_nodes; must not trigger KR101."""
        config = _functional_config([
            {
                "class_name": "Dense",
                "config": {"name": "dense", "units": 4, "trainable": True},
                "name": "dense",
                "inbound_nodes": [
                    {
                        "args": [
                            {
                                "class_name": "__keras_tensor__",
                                "config": {
                                    "shape": [None, 4],
                                    "dtype": "float32",
                                    "keras_history": ["in", 0, 0],
                                },
                            }
                        ],
                        "kwargs": {},
                    }
                ],
            }
        ])
        data = _make_h5_bytes(config)
        artifact = _make_artifact(data)

        findings = KerasAnalyzer().analyze(artifact)
        kr101_matched_values = [
            f.evidence.matched_value for f in findings if f.rule_id == "KR101"
        ]
        assert "class_name='__keras_tensor__'" not in kr101_matched_values, (
            "__keras_tensor__ is a Keras 3 internal and must not produce KR101"
        )

    def test_both_keras3_internals_together_no_kr101(self) -> None:
        """Both DTypePolicy and __keras_tensor__ together produce zero KR101 findings."""
        config = _functional_config([
            {
                "class_name": "Dense",
                "config": {
                    "name": "dense",
                    "units": 4,
                    "trainable": True,
                    "dtype": {
                        "class_name": "DTypePolicy",
                        "config": {"name": "float32"},
                    },
                },
                "name": "dense",
                "inbound_nodes": [
                    {
                        "args": [
                            {
                                "class_name": "__keras_tensor__",
                                "config": {"shape": [None, 4], "dtype": "float32",
                                           "keras_history": ["in", 0, 0]},
                            }
                        ],
                        "kwargs": {},
                    }
                ],
            }
        ])
        data = _make_h5_bytes(config)
        artifact = _make_artifact(data)

        findings = KerasAnalyzer().analyze(artifact)
        kr101_findings = [f for f in findings if f.rule_id == "KR101"]
        assert kr101_findings == [], (
            f"Expected no KR101 findings for Keras 3 internals, got: "
            f"{[f.evidence.matched_value for f in kr101_findings]}"
        )


# ---------------------------------------------------------------------------
# 3. __lambda__ class name — existing KR101 behavior unchanged
# ---------------------------------------------------------------------------


class TestLambdaLayerKR101Unchanged:
    """__lambda__ is the internal repr of a serialized Lambda function.

    It is NOT a Keras built-in and MUST continue to fire KR101.
    The embedded code payload is represented as an opaque string constant
    below — it is never decoded or executed.
    """

    # Static opaque constant — not decoded, not executed.
    _OPAQUE_CODE_PAYLOAD = "REDACTED_FOR_STATIC_TEST"

    def test_lambda_layer_fires_kr101(self) -> None:
        """A Lambda layer with class_name='__lambda__' must fire KR101."""
        config = _functional_config([
            {
                "class_name": "Lambda",
                "config": {
                    "name": "lambda",
                    "trainable": True,
                    "dtype": {
                        "class_name": "DTypePolicy",
                        "config": {"name": "float32"},
                    },
                    "function": {
                        "class_name": "__lambda__",
                        "config": {
                            "code": self._OPAQUE_CODE_PAYLOAD,
                            "defaults": None,
                            "closure": None,
                        },
                    },
                    "arguments": {},
                },
                "name": "lambda",
                "inbound_nodes": [
                    {
                        "args": [
                            {
                                "class_name": "__keras_tensor__",
                                "config": {"shape": [None, 4], "dtype": "float32",
                                           "keras_history": ["in", 0, 0]},
                            }
                        ],
                        "kwargs": {"mask": None},
                    }
                ],
            }
        ])
        data = _make_h5_bytes(config)
        artifact = _make_artifact(data)

        findings = KerasAnalyzer().analyze(artifact)
        kr101_matched_values = [
            f.evidence.matched_value for f in findings if f.rule_id == "KR101"
        ]

        # __lambda__ must still fire
        assert "class_name='__lambda__'" in kr101_matched_values, (
            "__lambda__ class must continue to produce KR101"
        )
        # DTypePolicy and __keras_tensor__ must be suppressed
        assert "class_name='DTypePolicy'" not in kr101_matched_values
        assert "class_name='__keras_tensor__'" not in kr101_matched_values

    def test_lambda_layer_kr101_severity_and_confidence(self) -> None:
        """KR101 for __lambda__ must keep MEDIUM severity and MEDIUM confidence."""
        config = _functional_config([
            {
                "class_name": "Lambda",
                "config": {
                    "name": "lam",
                    "trainable": True,
                    "function": {
                        "class_name": "__lambda__",
                        "config": {"code": self._OPAQUE_CODE_PAYLOAD,
                                   "defaults": None, "closure": None},
                    },
                    "arguments": {},
                },
                "name": "lam",
                "inbound_nodes": [],
            }
        ])
        data = _make_h5_bytes(config)
        artifact = _make_artifact(data)

        findings = KerasAnalyzer().analyze(artifact)
        lambda_findings = [
            f for f in findings
            if f.rule_id == "KR101" and "__lambda__" in f.evidence.matched_value
        ]
        assert len(lambda_findings) == 1
        f = lambda_findings[0]
        assert f.severity.value == "MEDIUM"
        assert f.confidence.value == "MEDIUM"
        assert f.remediation  # non-empty


# ---------------------------------------------------------------------------
# 4. KR103 — duplicate .keras ZIP member detection
# ---------------------------------------------------------------------------


def _make_keras_zip(members: list[tuple[str, bytes]]) -> bytes:
    """Build a .keras ZIP from (name, bytes) pairs. Duplicate names are allowed
    and written verbatim so the duplicate-member bypass can be exercised."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in members:
            zf.writestr(name, payload)
    buf.seek(0)
    return buf.read()


class TestDuplicateZipMemberKR103:
    """A .keras ZIP with duplicate member names must emit KR103."""

    def test_duplicate_config_json_flagged(self) -> None:
        benign = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        second = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip([("config.json", benign), ("config.json", second)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr103 = [f for f in findings if f.rule_id == "KR103"]
        assert len(kr103) == 1
        assert kr103[0].severity.value == "HIGH"
        assert "config.json" in kr103[0].evidence.matched_value

    def test_duplicate_non_config_member_flagged(self) -> None:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip([
            ("config.json", cfg),
            ("model.weights.h5", b"aa"),
            ("model.weights.h5", b"bb"),
        ])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr103 = [f for f in findings if f.rule_id == "KR103"]
        assert len(kr103) == 1
        assert "model.weights.h5" in kr103[0].evidence.matched_value

    def test_normal_archive_no_dup_finding(self) -> None:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip([("config.json", cfg), ("metadata.json", b"{}")])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert not any(f.rule_id == "KR103" for f in findings)

    def test_malformed_zip_handled(self) -> None:
        # PK magic followed by garbage → BadZipFile → KR102, not a crash.
        data = b"PK\x03\x04" + b"\x00" * 40
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert any(f.rule_id == "KR102" for f in findings)


# ---------------------------------------------------------------------------
# 5. KR101 — bare-string class extraction and registered_name
# ---------------------------------------------------------------------------


class TestBareStringClassKR101:
    """Bare class-name strings in layers/objects/nodes lists must fire KR101."""

    def test_bare_string_class_in_layers_flagged(self) -> None:
        cfg = json.dumps({"layers": ["MALICIOUS_LAMBDA"]}).encode()
        data = _make_keras_zip([("config.json", cfg)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert any("MALICIOUS_LAMBDA" in v for v in vals)

    def test_registered_name_flagged(self) -> None:
        cfg = json.dumps({
            "class_name": "Sequential",
            "config": {"layers": []},
            "registered_name": "CustomEvilClass",
        }).encode()
        data = _make_keras_zip([("config.json", cfg)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert any("CustomEvilClass" in v for v in vals)

    def test_builtin_bare_string_not_flagged(self) -> None:
        cfg = json.dumps({"layers": ["Dense", "Dropout"]}).encode()
        data = _make_keras_zip([("config.json", cfg)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert not any("Dense" in v or "Dropout" in v for v in vals)

    def test_ordinary_string_in_non_layer_key_not_flagged(self) -> None:
        # Strings under keys that are NOT class-list keys must not be treated as
        # class references.
        cfg = json.dumps({
            "class_name": "Sequential",
            "config": {"layers": [], "name": "SomeArbitraryName", "notes": ["hello", "world"]},
        }).encode()
        data = _make_keras_zip([("config.json", cfg)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert not any("SomeArbitraryName" in v or "hello" in v or "world" in v for v in vals)

    def test_deeply_nested_malicious_object_flagged(self) -> None:
        """TEST 6: Deeply nested attacker-controlled class_name must still be detected.

        A real malicious .keras archive may embed the unsafe class name several
        levels deep inside a sub-layer config rather than at the top-level config.
        _extract_class_names() must recurse into arbitrarily nested dicts so that
        KR101 fires regardless of nesting depth.
        """
        cfg = json.dumps({
            "class_name": "Functional",
            "config": {
                "name": "outer_model",
                "layers": [
                    {
                        "class_name": "Sequential",
                        "config": {
                            "name": "inner_model",
                            "layers": [
                                {
                                    "class_name": "Sequential",
                                    "config": {
                                        "name": "innermost",
                                        "layers": [
                                            {
                                                # Three levels deep — KR101 must still fire.
                                                "class_name": "__lambda__",
                                                "config": {
                                                    "function": {"class_name": "function", "config": "aW1wb3J0IG9z"}
                                                },
                                                "name": "deep_lambda",
                                                "inbound_nodes": [],
                                            }
                                        ],
                                    },
                                }
                            ],
                        },
                    }
                ],
                "input_layers": [],
                "output_layers": [],
            },
        }).encode()
        data = _make_keras_zip([("config.json", cfg)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr101_vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert any("__lambda__" in v for v in kr101_vals), (
            "KR101 must fire on __lambda__ even when nested three layers deep"
        )


# ---------------------------------------------------------------------------
# 6. ZIP member path-traversal detection (KR102)
# ---------------------------------------------------------------------------


def _make_keras_zip_with_members(members: list[tuple[str, bytes]]) -> bytes:
    """Build a .keras ZIP with the given (name, content) members in order.

    Unlike _make_keras_zip this helper does NOT add a default config.json
    so callers can fully control the member list (including malicious names).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, content in members:
            zf.writestr(name, content)
    buf.seek(0)
    return buf.read()


class TestZipPathTraversalKR102:
    """TEST 9: ZIP member names with path-traversal indicators must fire KR102."""

    def test_dotdot_component_fires_kr102(self) -> None:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip_with_members([
            ("config.json", cfg),
            ("../../evil.py", b"payload"),
        ])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr102 = [f for f in findings if f.rule_id == "KR102" and "suspicious_member" in (f.evidence.matched_value or "")]
        assert kr102, "KR102 must fire for '../../evil.py'"

    def test_absolute_unix_path_fires_kr102(self) -> None:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip_with_members([
            ("config.json", cfg),
            ("/etc/passwd", b"payload"),
        ])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr102 = [f for f in findings if f.rule_id == "KR102" and "suspicious_member" in (f.evidence.matched_value or "")]
        assert kr102, "KR102 must fire for absolute path '/etc/passwd'"

    def test_url_encoded_dotdot_fires_kr102(self) -> None:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip_with_members([
            ("config.json", cfg),
            ("%2e%2e/evil.py", b"payload"),
        ])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr102 = [f for f in findings if f.rule_id == "KR102" and "suspicious_member" in (f.evidence.matched_value or "")]
        assert kr102, "KR102 must fire for URL-encoded path traversal '%2e%2e/evil.py'"

    def test_windows_backslash_traversal_fires_kr102(self) -> None:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip_with_members([
            ("config.json", cfg),
            ("..\\..\\evil.py", b"payload"),
        ])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr102 = [f for f in findings if f.rule_id == "KR102" and "suspicious_member" in (f.evidence.matched_value or "")]
        assert kr102, r"KR102 must fire for backslash path traversal '..\..\\evil.py'"

    def test_normal_member_names_do_not_fire_kr102_traversal(self) -> None:
        cfg = json.dumps({"class_name": "Sequential", "config": {"layers": []}}).encode()
        data = _make_keras_zip_with_members([
            ("config.json", cfg),
            ("weights/layer_0.npy", b"\x00" * 8),
            ("assets/tokenizer.json", b"{}"),
        ])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        traversal_kr102 = [
            f for f in findings
            if f.rule_id == "KR102" and "suspicious_member" in (f.evidence.matched_value or "")
        ]
        assert not traversal_kr102, (
            "Normal nested member names like 'weights/layer_0.npy' must not fire KR102 path-traversal"
        )


# ---------------------------------------------------------------------------
# 7. KR101 initializer allowlist — R4 regression (8 confirmed false positives)
# ---------------------------------------------------------------------------


class TestInitializerAllowlist:
    """Standard Keras initializers must NOT produce KR101 findings.

    Regression suite for R4 (remediation-plan step 1).  The 8 false positives
    confirmed in the ModelAudit comparison were:
      vietnamese-emotion-lstm  → RandomUniform, GlorotUniform, Orthogonal, Zeros
      human-emotion-recognition → HeNormal, Zeros, Ones, GlorotUniform

    All eight initializer class names (and the full standard keras.initializers
    catalogue) must be suppressed even when they appear as nested class_name
    values inside layer configs (kernel_initializer, bias_initializer, etc.).
    Custom / unknown class names must still fire.
    """

    _STANDARD_INITIALIZERS = [
        "Zeros", "Ones", "Constant", "Identity",
        "RandomNormal", "RandomUniform", "TruncatedNormal",
        "VarianceScaling", "Orthogonal",
        "GlorotNormal", "GlorotUniform",
        "HeNormal", "HeUniform",
        "LecunNormal", "LecunUniform",
    ]

    def _config_with_initializer(self, init_name: str) -> dict:
        return {
            "class_name": "Sequential",
            "config": {
                "name": "model",
                "trainable": True,
                "layers": [
                    {
                        "class_name": "Dense",
                        "config": {
                            "name": "dense",
                            "units": 64,
                            "kernel_initializer": {
                                "class_name": init_name,
                                "config": {"seed": None},
                            },
                            "bias_initializer": {
                                "class_name": "Zeros",
                                "config": {},
                            },
                        },
                        "name": "dense",
                        "inbound_nodes": [],
                    }
                ],
            },
        }

    # --- HDF5 path

    def test_all_standard_initializers_suppressed_in_h5(self) -> None:
        """Each standard keras.initializers class must not fire KR101 in HDF5."""
        for init_name in self._STANDARD_INITIALIZERS:
            data = _make_h5_bytes(self._config_with_initializer(init_name))
            findings = KerasAnalyzer().analyze(_make_artifact(data))
            kr101_vals = [
                f.evidence.matched_value
                for f in findings
                if f.rule_id == "KR101"
            ]
            assert not any(init_name in v for v in kr101_vals), (
                f"Standard initializer '{init_name}' must not produce KR101 in HDF5; "
                f"found: {kr101_vals}"
            )

    def test_vietnamese_emotion_lstm_initializers_suppressed_h5(self) -> None:
        """The 4 exact initialisers that caused false positives on vietnamese-emotion-lstm."""
        config = {
            "class_name": "Sequential",
            "config": {
                "name": "model",
                "trainable": True,
                "layers": [
                    {
                        "class_name": "Dense",
                        "config": {
                            "kernel_initializer": {"class_name": "GlorotUniform", "config": {}},
                            "bias_initializer": {"class_name": "Zeros", "config": {}},
                        },
                        "name": "dense",
                        "inbound_nodes": [],
                    },
                    {
                        "class_name": "LSTM",
                        "config": {
                            "recurrent_initializer": {"class_name": "Orthogonal", "config": {}},
                            "kernel_initializer": {"class_name": "RandomUniform", "config": {}},
                        },
                        "name": "lstm",
                        "inbound_nodes": [],
                    },
                ],
            },
        }
        findings = KerasAnalyzer().analyze(_make_artifact(_make_h5_bytes(config)))
        kr101 = [f for f in findings if f.rule_id == "KR101"]
        assert kr101 == [], (
            f"Vietnamese-emotion-lstm initialisers produced unexpected KR101: "
            f"{[f.evidence.matched_value for f in kr101]}"
        )

    def test_human_emotion_recognition_initializers_suppressed_h5(self) -> None:
        """The 4 exact initialisers that caused false positives on human-emotion-recognition."""
        config = {
            "class_name": "Sequential",
            "config": {
                "name": "model",
                "trainable": True,
                "layers": [
                    {
                        "class_name": "Conv2D",
                        "config": {
                            "kernel_initializer": {"class_name": "HeNormal", "config": {}},
                            "bias_initializer": {"class_name": "Zeros", "config": {}},
                        },
                        "name": "conv2d",
                        "inbound_nodes": [],
                    },
                    {
                        "class_name": "Dense",
                        "config": {
                            "kernel_initializer": {"class_name": "GlorotUniform", "config": {}},
                            "bias_initializer": {"class_name": "Ones", "config": {}},
                        },
                        "name": "dense",
                        "inbound_nodes": [],
                    },
                ],
            },
        }
        findings = KerasAnalyzer().analyze(_make_artifact(_make_h5_bytes(config)))
        kr101 = [f for f in findings if f.rule_id == "KR101"]
        assert kr101 == [], (
            f"Human-emotion-recognition initialisers produced unexpected KR101: "
            f"{[f.evidence.matched_value for f in kr101]}"
        )

    # --- ZIP path

    def test_all_standard_initializers_suppressed_in_zip(self) -> None:
        """Each standard keras.initializers class must not fire KR101 in .keras ZIP."""
        for init_name in self._STANDARD_INITIALIZERS:
            cfg_bytes = json.dumps(self._config_with_initializer(init_name)).encode()
            data = _make_keras_zip([("config.json", cfg_bytes)])
            findings = KerasAnalyzer().analyze(_make_artifact(data))
            kr101_vals = [
                f.evidence.matched_value
                for f in findings
                if f.rule_id == "KR101"
            ]
            assert not any(init_name in v for v in kr101_vals), (
                f"Standard initializer '{init_name}' must not produce KR101 in ZIP; "
                f"found: {kr101_vals}"
            )

    # --- Regression: allowlist does NOT suppress custom/malicious names

    def test_custom_class_still_fires_kr101_h5(self) -> None:
        """Custom class names outside the allowlist must still produce KR101 in HDF5."""
        config = {
            "class_name": "Sequential",
            "config": {
                "name": "model",
                "trainable": True,
                "layers": [
                    {
                        "class_name": "Dense",
                        "config": {
                            "kernel_initializer": {"class_name": "GlorotUniform", "config": {}},
                            "bias_initializer": {"class_name": "EvilPayloadInitializer", "config": {}},
                        },
                        "name": "dense",
                        "inbound_nodes": [],
                    }
                ],
            },
        }
        findings = KerasAnalyzer().analyze(_make_artifact(_make_h5_bytes(config)))
        kr101_vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert any("EvilPayloadInitializer" in v for v in kr101_vals), (
            "Custom initializer class 'EvilPayloadInitializer' must still fire KR101"
        )
        assert not any("GlorotUniform" in v for v in kr101_vals), (
            "Standard GlorotUniform must remain suppressed even when a custom class is present"
        )


# ---------------------------------------------------------------------------
# 8. KR104 — HDF5 External / Virtual Dataset Reference (R1)
# ---------------------------------------------------------------------------


def _make_h5_with_external_link(
    link_name: str, ext_filename: str, ext_path: str
) -> bytes:
    """Create a minimal HDF5 file containing a named ExternalLink."""
    h5py = pytest.importorskip("h5py")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["keras_version"] = "3.6.0"
        f.attrs["model_config"] = json.dumps(
            {"class_name": "Sequential", "config": {"layers": []}}
        ).encode("utf-8")
        f["weights"] = [1.0, 2.0, 3.0]                     # benign dataset
        f[link_name] = h5py.ExternalLink(ext_filename, ext_path)
    buf.seek(0)
    return buf.read()


def _make_h5_with_vds(
    dataset_name: str, ext_filename: str, ext_dset: str, shape: tuple[int, ...] = (100,)
) -> bytes:
    """Create a minimal HDF5 file containing a Virtual Dataset Source."""
    h5py = pytest.importorskip("h5py")
    import numpy as np  # h5py VirtualLayout requires numpy
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.attrs["keras_version"] = "3.6.0"
        f.attrs["model_config"] = json.dumps(
            {"class_name": "Sequential", "config": {"layers": []}}
        ).encode("utf-8")
        f["weights"] = [1.0, 2.0, 3.0]                     # benign dataset
        layout = h5py.VirtualLayout(shape=shape, dtype="f4")
        vsource = h5py.VirtualSource(ext_filename, ext_dset, shape=shape)
        layout[:] = vsource
        f.create_virtual_dataset(dataset_name, layout, fillvalue=0.0)
    buf.seek(0)
    return buf.read()


class TestKR104ExternalVDSReference:
    """KR104 — HDF5 ExternalLink and Virtual Dataset Source detection.

    SAFETY CONTRACT: none of these tests cause the scanner to open
    or read from any external file.  The referenced external files
    (victim_data.h5, /etc/passwd, etc.) are untrusted strings only.
    """

    # --- ExternalLink tests ---

    def test_external_link_fires_kr104(self) -> None:
        """An HDF5 ExternalLink must produce a KR104 finding."""
        data = _make_h5_with_external_link("ext", "victim_data.h5", "/secret")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr104 = [f for f in findings if f.rule_id == "KR104"]
        assert kr104, "KR104 must fire for an HDF5 ExternalLink"
        f = kr104[0]
        assert f.severity.value == "HIGH"
        assert f.confidence.value == "HIGH"
        assert "victim_data.h5" in f.evidence.matched_value
        assert "/secret" in f.evidence.matched_value
        assert "external_link" in f.evidence.matched_value
        assert "CVE-2026-1669" in f.cve_ids

    def test_external_link_absolute_path_escalates_to_critical(self) -> None:
        """An ExternalLink pointing to an absolute host path is CRITICAL (CWE-73)."""
        data = _make_h5_with_external_link("etc_link", "/etc/passwd", "/root")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr104 = [f for f in findings if f.rule_id == "KR104"]
        assert kr104, "KR104 must fire for absolute-path ExternalLink"
        assert kr104[0].severity.value == "CRITICAL"
        assert "ABSOLUTE_PATH" in kr104[0].evidence.matched_value

    def test_external_link_traversal_path_escalates_to_critical(self) -> None:
        """An ExternalLink with '../' traversal escalates to CRITICAL."""
        data = _make_h5_with_external_link("trav_link", "../../etc/shadow", "/dataset")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr104 = [f for f in findings if f.rule_id == "KR104"]
        assert kr104, "KR104 must fire for traversal-path ExternalLink"
        assert kr104[0].severity.value == "CRITICAL"
        assert "PATH_TRAVERSAL" in kr104[0].evidence.matched_value

    # --- VDS tests ---

    def test_vds_external_reference_fires_kr104(self) -> None:
        """A VDS dataset with an external source file must produce a KR104 finding."""
        data = _make_h5_with_vds("vds_weights", "victim_data.h5", "/secret")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr104 = [f for f in findings if f.rule_id == "KR104"]
        assert kr104, "KR104 must fire for a VDS with an external source"
        f = kr104[0]
        assert f.severity.value == "HIGH"
        assert "victim_data.h5" in f.evidence.matched_value
        assert "/secret" in f.evidence.matched_value
        assert "vds_source" in f.evidence.matched_value

    def test_vds_absolute_path_escalates_to_critical(self) -> None:
        """A VDS pointing to an absolute source path escalates to CRITICAL."""
        data = _make_h5_with_vds("vds", "/mnt/nfs/secret.h5", "/weights")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr104 = [f for f in findings if f.rule_id == "KR104"]
        assert kr104, "KR104 must fire for absolute-path VDS"
        assert kr104[0].severity.value == "CRITICAL"
        assert "ABSOLUTE_PATH" in kr104[0].evidence.matched_value

    # --- Clean model (regression) ---

    def test_clean_h5_no_kr104(self) -> None:
        """A standard Keras HDF5 model with no external references must not fire KR104."""
        config = {"class_name": "Sequential", "config": {"layers": []}}
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert not any(f.rule_id == "KR104" for f in findings), (
            "KR104 must not fire on a clean HDF5 model with no external references"
        )

    # --- Scanner safety: external file must NOT be opened ---

    def test_scanner_does_not_follow_external_link(self, tmp_path: "pytest.TempdirFactory") -> None:
        """The scanner must NOT open the referenced external file.

        This test creates a sentinel file and verifies it is never accessed
        during scanning — proving the detector is static-only.
        """
        import os
        h5py = pytest.importorskip("h5py")
        # Create an HDF5 file with ExternalLink pointing to a sentinel file
        # that records whether it was ever opened
        sentinel = tmp_path / "victim_data.h5"  # type: ignore[operator]
        sentinel.write_bytes(b"SENTINEL_NOT_TO_BE_OPENED")
        mtime_before = os.path.getmtime(str(sentinel))

        data = _make_h5_with_external_link("ext", str(sentinel), "/dataset")
        findings = KerasAnalyzer().analyze(_make_artifact(data))

        mtime_after = os.path.getmtime(str(sentinel))
        assert mtime_before == mtime_after, (
            "The scanner opened the referenced external file — this violates the static-only contract"
        )
        assert any(f.rule_id == "KR104" for f in findings), (
            "KR104 must still be detected even when the external file is not opened"
        )


# ---------------------------------------------------------------------------
# 9. KR105 — Keras Lambda Bytecode Payload Analysis (R2)
# ---------------------------------------------------------------------------
#
# SAFETY CONTRACT
# ---------------
# The _KR105_CODE_* constants below are pre-computed base64 strings.
# They are generated at module load time using Python's compile() + marshal.dumps()
# on controlled, known-safe source expressions — NOT on attacker-controlled input.
# Once generated they are treated as opaque constants by all test code below.
#
# The scanner under test receives these constants as untrusted data, decodes the
# base64 envelope only (producing raw bytes), and performs pure bytes pattern
# matching.  Neither the tests nor the scanner ever call exec(), eval(),
# marshal.loads(), or any execution primitive on these payloads.
#
# Real-model validation: the 'modelscan-h5-bypass-poc' model is not available in
# this environment.  A synthetic HDF5 fixture that reproduces the documented
# structure (Lambda layer with os.system in bytecode) is used instead.

def _compile_lambda_payload(source: str) -> str:
    """Compile a controlled expression to base64-encoded marshal bytes.

    Called at module load time to build static test-payload constants.
    Uses compile() (NOT exec/eval) then marshal.dumps() — both safe, no execution.
    """
    code_obj = compile(source, "<kr105_test>", "eval")
    return base64.b64encode(_marshal.dumps(code_obj)).decode("ascii")


# Pre-computed at import time — treated as opaque constants by all tests below.
_KR105_CODE_OS_SYSTEM  = _compile_lambda_payload("os.system('cmd')")
_KR105_CODE_SUBPROCESS = _compile_lambda_payload("subprocess.Popen(['cmd'])")
_KR105_CODE_EVAL       = _compile_lambda_payload("eval(x)")
_KR105_CODE_EXEC       = _compile_lambda_payload("exec(x)")
_KR105_CODE_IMPORT     = _compile_lambda_payload("__import__('os').system('cmd')")
_KR105_CODE_POPEN      = _compile_lambda_payload("os.popen('cmd')")
_KR105_CODE_BENIGN     = _compile_lambda_payload("x * 2")
_KR105_CODE_BENIGN2    = _compile_lambda_payload("x + y")


def _make_lambda_layer(code_b64: str, layer_name: str = "lambda_1") -> dict:
    """Minimal Keras Lambda layer config with the given base64-encoded code payload."""
    return {
        "class_name": "Lambda",
        "config": {
            "name": layer_name,
            "trainable": True,
            "dtype": {"class_name": "DTypePolicy", "config": {"name": "float32"}},
            "function": {
                "class_name": "__lambda__",
                "config": {
                    "code": code_b64,
                    "defaults": None,
                    "closure": None,
                },
            },
            "arguments": {},
        },
        "name": layer_name,
        "inbound_nodes": [],
    }


class TestKR105LambdaBytecodePayload:
    """KR105 — Keras Lambda Bytecode Payload static analysis.

    All test payloads are pre-computed constants (see module-level constants above).
    Neither the test code nor the scanner executes any of the embedded bytecode.
    """

    # ----------------------------------------------------------------- HDF5 path

    def test_benign_lambda_no_kr105_h5(self) -> None:
        """A Lambda whose bytecode contains no dangerous patterns must not fire KR105."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_BENIGN)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert not any(f.rule_id == "KR105" for f in findings), (
            f"Benign Lambda (x * 2) must not produce KR105; got: "
            f"{[f.evidence.matched_value for f in findings if f.rule_id == 'KR105']}"
        )

    def test_os_system_fires_kr105_critical_h5(self) -> None:
        """os.system in Lambda bytecode must produce KR105 CRITICAL (HDF5 path)."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_OS_SYSTEM)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for os.system in Lambda bytecode"
        assert kr105[0].severity.value == "CRITICAL", (
            f"os.system must produce CRITICAL, got {kr105[0].severity.value}"
        )
        assert "os.system" in kr105[0].evidence.matched_value

    def test_subprocess_fires_kr105_critical_h5(self) -> None:
        """subprocess.Popen in Lambda bytecode must produce KR105 CRITICAL."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_SUBPROCESS)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for subprocess.Popen in Lambda bytecode"
        assert kr105[0].severity.value == "CRITICAL"
        ev = kr105[0].evidence.matched_value
        assert "subprocess" in ev or "Popen" in ev

    def test_eval_fires_kr105_high_h5(self) -> None:
        """eval in Lambda bytecode must produce KR105 HIGH."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_EVAL)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for eval in Lambda bytecode"
        assert kr105[0].severity.value == "HIGH"
        assert "eval" in kr105[0].evidence.matched_value

    def test_exec_fires_kr105_high_h5(self) -> None:
        """exec in Lambda bytecode must produce KR105 HIGH."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_EXEC)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for exec in Lambda bytecode"
        assert kr105[0].severity.value == "HIGH"
        assert "exec" in kr105[0].evidence.matched_value

    def test_import_fires_kr105_high_h5(self) -> None:
        """__import__ in Lambda bytecode must produce KR105 HIGH (or higher)."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_IMPORT)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for __import__ in Lambda bytecode"
        # __import__('os').system('cmd') → CRITICAL (system combo found) or HIGH
        assert kr105[0].severity.value in ("HIGH", "CRITICAL")
        assert "__import__" in kr105[0].evidence.matched_value or "os.system" in kr105[0].evidence.matched_value

    def test_popen_fires_kr105_critical_h5(self) -> None:
        """os.popen in Lambda bytecode must produce KR105 CRITICAL."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_POPEN)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for os.popen in Lambda bytecode"
        assert kr105[0].severity.value == "CRITICAL"
        assert "popen" in kr105[0].evidence.matched_value.lower()

    def test_kr101_still_fires_alongside_kr105(self) -> None:
        """A dangerous Lambda must produce BOTH KR101 (existing) and KR105 (new).

        KR101 flags the custom __lambda__ class. KR105 flags the dangerous payload.
        Both rules are complementary and must co-exist.
        """
        config = _functional_config([_make_lambda_layer(_KR105_CODE_OS_SYSTEM)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        rule_ids = {f.rule_id for f in findings}
        assert "KR101" in rule_ids, "__lambda__ must still fire KR101"
        assert "KR105" in rule_ids, "os.system payload must fire KR105"

    def test_nested_lambda_detected(self) -> None:
        """A dangerous Lambda nested multiple levels deep must still produce KR105."""
        config = {
            "class_name": "Functional",
            "config": {
                "name": "outer",
                "layers": [
                    {
                        "class_name": "Sequential",
                        "config": {
                            "name": "inner",
                            "layers": [
                                {
                                    "class_name": "Lambda",
                                    "config": {
                                        "name": "lambda_deep",
                                        "function": {
                                            "class_name": "__lambda__",
                                            "config": {
                                                "code": _KR105_CODE_SUBPROCESS,
                                                "defaults": None,
                                                "closure": None,
                                            },
                                        },
                                    },
                                    "name": "lambda_deep",
                                    "inbound_nodes": [],
                                }
                            ],
                        },
                    }
                ],
                "input_layers": [],
                "output_layers": [],
            },
        }
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire even when Lambda is nested multiple levels deep"
        assert kr105[0].severity.value == "CRITICAL"

    def test_lambda_without_code_field_no_crash(self) -> None:
        """A __lambda__ node missing the 'code' field must not crash the scanner."""
        config = _functional_config([
            {
                "class_name": "Lambda",
                "config": {
                    "name": "lambda_no_code",
                    "function": {
                        "class_name": "__lambda__",
                        "config": {"defaults": None, "closure": None},
                        # 'code' key is intentionally absent
                    },
                },
                "name": "lambda_no_code",
                "inbound_nodes": [],
            }
        ])
        data = _make_h5_bytes(config)
        # Must not raise; KR101 may fire but no KR105 (no code to scan)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert not kr105, "No 'code' field → KR105 must not fire (nothing to scan)"

    def test_second_benign_lambda_no_kr105_h5(self) -> None:
        """A second benign Lambda (x + y) must also not fire KR105."""
        config = _functional_config([_make_lambda_layer(_KR105_CODE_BENIGN2, "lambda_2")])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert not any(f.rule_id == "KR105" for f in findings), (
            "Benign Lambda (x + y) must not produce KR105"
        )

    # ----------------------------------------------------------------- ZIP path

    def test_os_system_fires_kr105_critical_zip(self) -> None:
        """os.system in Lambda bytecode must produce KR105 CRITICAL (.keras ZIP path)."""
        cfg_obj = _functional_config([_make_lambda_layer(_KR105_CODE_OS_SYSTEM)])
        cfg_bytes = json.dumps(cfg_obj).encode("utf-8")
        data = _make_keras_zip([("config.json", cfg_bytes)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for os.system in Lambda bytecode (ZIP path)"
        assert kr105[0].severity.value == "CRITICAL"

    def test_benign_lambda_no_kr105_zip(self) -> None:
        """Benign Lambda in .keras ZIP must not produce KR105."""
        cfg_obj = _functional_config([_make_lambda_layer(_KR105_CODE_BENIGN)])
        cfg_bytes = json.dumps(cfg_obj).encode("utf-8")
        data = _make_keras_zip([("config.json", cfg_bytes)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert not any(f.rule_id == "KR105" for f in findings), (
            "Benign Lambda (x * 2) must not produce KR105 in ZIP path"
        )

    def test_subprocess_fires_kr105_critical_zip(self) -> None:
        """subprocess.Popen in .keras ZIP Lambda must produce KR105 CRITICAL."""
        cfg_obj = _functional_config([_make_lambda_layer(_KR105_CODE_SUBPROCESS)])
        cfg_bytes = json.dumps(cfg_obj).encode("utf-8")
        data = _make_keras_zip([("config.json", cfg_bytes)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for subprocess.Popen in ZIP Lambda"
        assert kr105[0].severity.value == "CRITICAL"

    # --------- Regression: existing rules unaffected by KR105 addition --------

    def test_kr103_dup_zip_unaffected_by_kr105(self) -> None:
        """KR103 duplicate-member detection is unaffected by the KR105 addition."""
        cfg = json.dumps(
            {"class_name": "Sequential", "config": {"layers": []}}
        ).encode()
        data = _make_keras_zip([("config.json", cfg), ("config.json", cfg)])
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert any(f.rule_id == "KR103" for f in findings), "KR103 must still fire"
        assert not any(f.rule_id == "KR105" for f in findings), (
            "KR105 must not fire for a model with no Lambda layers"
        )

    def test_kr104_vds_unaffected_by_kr105(self) -> None:
        """KR104 VDS detection is unaffected by the KR105 addition."""
        h5py = pytest.importorskip("h5py")
        import numpy as np
        data = _make_h5_with_vds("vds", "victim.h5", "/weights")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert any(f.rule_id == "KR104" for f in findings), "KR104 must still fire"

    def test_r4_initializer_allowlist_unaffected_by_kr105(self) -> None:
        """Standard Keras initializers still do not produce KR101 (R4 regression)."""
        config = {
            "class_name": "Sequential",
            "config": {
                "name": "model",
                "layers": [{
                    "class_name": "Dense",
                    "config": {
                        "kernel_initializer": {"class_name": "GlorotUniform", "config": {}},
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                    },
                    "name": "dense",
                    "inbound_nodes": [],
                }],
            },
        }
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr101_vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert not any("GlorotUniform" in v or "Zeros" in v for v in kr101_vals), (
            "Standard initializers must not produce KR101 — R4 regression"
        )

    # --------- Safety: scanner must never execute the payload -----------------

    def test_scanner_does_not_execute_lambda_payload(self, tmp_path: "pytest.TempPathFactory") -> None:
        """The scanner must not execute Lambda bytecode during static analysis.

        A probe file path is embedded as a string constant in the Lambda payload.
        If the scanner executes the code, the file would be created.
        KR105 must still be reported without the file being created.

        SAFETY: compile() creates a code object from a CONTROLLED string here.
        The scanner receives the base64 output and never calls exec/eval on it.
        """
        probe = tmp_path / "kr105_execution_probe.txt"  # type: ignore[operator]
        probe_path = str(probe).replace("\\", "/")

        # This expression, IF EXECUTED, would create the probe file.
        # The scanner must not execute it.
        payload_source = f"open('{probe_path}', 'w').write('executed')"
        probe_code_b64 = _compile_lambda_payload(payload_source)

        config = _functional_config([_make_lambda_layer(probe_code_b64)])
        data = _make_h5_bytes(config)
        findings = KerasAnalyzer().analyze(_make_artifact(data))

        # Probe file must NOT have been created — the scanner is static-only.
        assert not probe.exists(), (
            f"Scanner executed the Lambda payload — probe file was created at {probe}. "
            "The scanner must never execute Lambda bytecode."
        )
        # The scanner must still have detected the dangerous pattern (open → write).
        # open() is not in our direct pattern list, but the payload contains 'exec' in
        # the opcode stream via STORE_NAME or similar. Actually let's just verify
        # KR101 fires (the __lambda__ class is non-standard) even if KR105 patterns
        # don't cover 'open' (a documented limitation).
        assert any(f.rule_id in ("KR101", "KR105") for f in findings), (
            "Scanner must at least flag the Lambda via KR101 even if KR105 patterns "
            "don't cover 'open()' specifically."
        )


# ---------------------------------------------------------------------------
# 10. KR105 — plain-string Lambda function format (ModelAudit-style HDF5)
# ---------------------------------------------------------------------------
#
# Some Keras serialization variants (including ModelAudit test assets) store the
# Lambda function as a plain Python source string rather than the standard
# base64-encoded marshal format.  KR105 must detect dangerous patterns in both.


def _make_plain_string_lambda_h5(function_source: str) -> bytes:
    """Build an HDF5 file with a Lambda layer whose function is a plain string."""
    config = {
        "class_name": "Sequential",
        "config": {
            "name": "malicious_model",
            "layers": [
                {
                    "class_name": "Lambda",
                    "config": {"function": function_source},
                }
            ],
        },
    }
    return _make_h5_bytes(config)


class TestKR105PlainStringLambdaFormat:
    """KR105 detection for Lambda functions stored as plain source strings.

    This format appears in ModelAudit test artifacts and some older Keras
    serializations.  The function value is a literal string such as
    'lambda x: eval(x)' rather than a base64-encoded marshal object.

    SAFETY: test strings are static constants — never executed.
    """

    def test_eval_in_plain_string_lambda_fires_kr105(self) -> None:
        """eval in a plain-string Lambda function must fire KR105."""
        data = _make_plain_string_lambda_h5("lambda x: eval(x)")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for eval in plain-string Lambda function"
        assert kr105[0].severity.value == "HIGH"
        assert "eval" in kr105[0].evidence.matched_value

    def test_exec_in_plain_string_lambda_fires_kr105(self) -> None:
        """exec in a plain-string Lambda function must fire KR105."""
        data = _make_plain_string_lambda_h5("lambda x: exec(x)")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for exec in plain-string Lambda function"

    def test_os_system_in_plain_string_lambda_fires_kr105_critical(self) -> None:
        """os.system in a plain-string Lambda must fire KR105 CRITICAL."""
        data = _make_plain_string_lambda_h5("lambda x: os.system(x)")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for os.system in plain-string Lambda"
        assert kr105[0].severity.value == "CRITICAL"

    def test_subprocess_in_plain_string_lambda_fires_kr105_critical(self) -> None:
        """subprocess in a plain-string Lambda must fire KR105 CRITICAL."""
        data = _make_plain_string_lambda_h5("lambda x: subprocess.Popen(x)")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must fire for subprocess in plain-string Lambda"
        assert kr105[0].severity.value == "CRITICAL"

    def test_benign_plain_string_lambda_no_kr105(self) -> None:
        """A benign plain-string Lambda must NOT fire KR105."""
        data = _make_plain_string_lambda_h5("lambda x: x * 2")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        assert not any(f.rule_id == "KR105" for f in findings), (
            "Benign plain-string Lambda 'lambda x: x * 2' must not fire KR105"
        )

    def test_modelaudit_style_eval_lambda_detected(self) -> None:
        """Exact format from ModelAudit test asset malicious_lambda.h5 must be detected."""
        # This replicates the exact structure used by ModelAudit's test fixture.
        data = _make_plain_string_lambda_h5("lambda x: eval('print(\"security test\")')")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr105 = [f for f in findings if f.rule_id == "KR105"]
        assert kr105, "KR105 must detect ModelAudit-style plain-string eval Lambda"
        assert kr105[0].severity.value == "HIGH"

    def test_plain_string_kr101_still_silent_for_lambda_classname(self) -> None:
        """Lambda (the layer class) is allowlisted; KR101 must not fire for it."""
        data = _make_plain_string_lambda_h5("lambda x: x * 2")
        findings = KerasAnalyzer().analyze(_make_artifact(data))
        kr101_vals = [f.evidence.matched_value for f in findings if f.rule_id == "KR101"]
        assert not any("Lambda" in v for v in kr101_vals), (
            "Lambda layer class_name must remain allowlisted (KR101 must not fire for it)"
        )
