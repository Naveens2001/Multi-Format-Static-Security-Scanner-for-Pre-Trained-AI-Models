"""
tests/unit/test_tensorflow_analyzer.py
======================================

Unit tests for the TensorFlow SavedModel static security analyzer.

SAFETY CONTRACT
---------------
- No tensorflow import, no SavedModel loading, no graph execution.
- No eval(), exec(), or dynamic deserialization.
- Synthetic saved_model.pb fixtures are hand-built protobuf byte strings
  containing op-name string fields. They are never executed or loaded by TF.
- Asset filenames used for path-traversal tests are inert strings; the asset
  files are never opened or followed.

Coverage
--------
1. TF101 fires for genuinely dangerous ops (ReadFile / WriteFile).
2. Version strings and protobuf field names do NOT produce TF101 (false-positive
   reduction — the core of TC4).
3. Malformed / empty protobuf is handled without crashing.
4. Suspicious custom ops from the adversarial payloads are still detected.
5. TF103 fires for path-traversal filenames in the sibling assets/ directory,
   and does not fire for benign asset names or when there is no assets dir.
6. Wire-format fragments (lowercase-start, internal-uppercase tokens) do NOT
   produce TF101 (precision improvement over TC4 baseline).
7. Protobuf attribute-type keys (Tout, Tdim, FullArgSpec) do NOT fire TF101.
8. Standard ops added in this pass (Select, StaticRegexFullMatch,
   DisableCopyOnRead, ShardedFilename) do NOT fire TF101.
9. TC4 genuine signals (ReadFile, WriteFile, MergeV2Checkpoints, SaveV2,
   RestoreV2, evil variable-name patterns) still fire TF101.
"""

from __future__ import annotations

import io
from pathlib import Path

from model_scanner.analyzers.tensorflow.analyzer import (
    TensorFlowAnalyzer,
    _is_suspicious_op,
    _traversal_indicator,
)
from model_scanner.core.models import Artifact, InputSourceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _pb_string_field(field_number: int, text: str) -> bytes:
    """Encode a length-delimited (wire type 2) string protobuf field."""
    tag = (field_number << 3) | 2
    payload = text.encode("utf-8")
    return _encode_varint(tag) + _encode_varint(len(payload)) + payload


def _build_pb(op_names: list[str]) -> bytes:
    """Build a minimal saved_model.pb-like protobuf containing the given strings."""
    body = b"".join(_pb_string_field(1, name) for name in op_names)
    # Prefix a couple of ordinary metadata strings the way a real pb would carry.
    body = _pb_string_field(1, "2.21.0") + _pb_string_field(1, "serving_default") + body
    return body


def _make_artifact_stream(data: bytes) -> Artifact:
    return Artifact(
        artifact_id="test-tf",
        source_path="saved_model.pb",
        input_source=InputSourceType.IN_MEMORY,
        size_bytes=len(data),
        sha256="c" * 64,
        stream=io.BytesIO(data),
    )


def _make_artifact_local(path: Path) -> Artifact:
    data = path.read_bytes()
    return Artifact(
        artifact_id="test-tf-local",
        source_path=str(path),
        input_source=InputSourceType.LOCAL_FILE,
        size_bytes=len(data),
        sha256="d" * 64,
        local_path=path,
    )


def _tf101_ops(findings) -> list[str]:
    return [f.evidence.matched_value for f in findings if f.rule_id == "TF101"]


# ---------------------------------------------------------------------------
# 1. Dangerous ops detected
# ---------------------------------------------------------------------------

def test_readfile_op_detected() -> None:
    data = _build_pb(["ReadFile"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(data))
    assert any("ReadFile" in v for v in _tf101_ops(findings))


def test_writefile_op_detected() -> None:
    data = _build_pb(["WriteFile"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(data))
    assert any("WriteFile" in v for v in _tf101_ops(findings))


# ---------------------------------------------------------------------------
# 2. False-positive reduction
# ---------------------------------------------------------------------------

def test_version_strings_not_flagged() -> None:
    for s in ("2.21.0", ".0", ".part", "1.0"):
        assert _is_suspicious_op(s) is False, f"{s} should not be flagged"


def test_proto_field_names_not_flagged() -> None:
    for s in ("type", "dtype", "shape", "resource", "config", "value", "output"):
        assert _is_suspicious_op(s) is False, f"{s} should not be flagged"


def test_standard_ops_not_flagged() -> None:
    for s in ("Conv2D", "MatMul", "Identity", "Const"):
        assert _is_suspicious_op(s) is False, f"{s} is a standard op"


def test_false_positive_count_drops() -> None:
    """A pb full of version/field-name noise plus one dangerous op yields a small
    finding set — not the ~100 findings of the prior heuristic."""
    noise = ["2.21.0", "type", "dtype", "shape", "resource", "config", "value",
             "values", "output", "input", "name", "CPU", "GPU", "serving_default",
             "float32", "int64", "bool"]
    data = _build_pb([*noise, "ReadFile"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(data))
    tf101 = _tf101_ops(findings)
    assert any("ReadFile" in v for v in tf101)
    assert len(tf101) <= 5, f"Expected few findings, got {len(tf101)}: {tf101}"


# ---------------------------------------------------------------------------
# 3. Malformed input handled
# ---------------------------------------------------------------------------

def test_empty_pb_handled() -> None:
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(b""))
    assert findings == []


def test_tiny_pb_handled() -> None:
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(b"\x01"))
    # Either TF102 (too small under heuristic) or empty — must not raise.
    assert all(f.rule_id in ("TF102",) for f in findings)


def test_garbage_pb_does_not_crash() -> None:
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(b"\xff\xff\xff\xff garbage"))
    assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# 4. Adversarial payload ops still detected
# ---------------------------------------------------------------------------

def test_payload1_dangerous_checkpoint_ops_detected() -> None:
    """TC4 payload 1 regression: checkpoint filesystem ops still fire TF101.

    ShardedFilename and DisableCopyOnRead are now standard-op suppressions;
    the genuine signals for TC4 payload 1 are MergeV2Checkpoints / SaveV2 /
    RestoreV2 which are in _DANGEROUS_TF_OPS (Tier 1, always reported).
    """
    data = _build_pb(["MergeV2Checkpoints", "SaveV2", "RestoreV2"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(data))
    ops = " ".join(_tf101_ops(findings))
    assert "MergeV2Checkpoints" in ops
    assert "SaveV2" in ops
    assert "RestoreV2" in ops


def test_payload3_cmd_context_detected() -> None:
    data = _build_pb(["serving_default_cmd", "WriteFile"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(data))
    ops = " ".join(_tf101_ops(findings))
    assert "WriteFile" in ops


# ---------------------------------------------------------------------------
# 6. Wire-format fragment suppression (new precision tests)
# ---------------------------------------------------------------------------

def test_wire_fragment_ableop_not_flagged() -> None:
    """'ableOp' is a mid-string slice of 'ReadVariableOp' — must not fire."""
    assert _is_suspicious_op("ableOp") is False


def test_wire_fragment_advariableop_not_flagged() -> None:
    assert _is_suspicious_op("adVariableOp") is False


def test_wire_fragment_storev2_not_flagged() -> None:
    assert _is_suspicious_op("storeV2") is False


def test_wire_fragment_adfile_not_flagged() -> None:
    assert _is_suspicious_op("adFile") is False


def test_wire_fragment_itionedcall_not_flagged() -> None:
    assert _is_suspicious_op("itionedCall") is False


# ---------------------------------------------------------------------------
# 7. Protobuf attribute-key and metadata suppression
# ---------------------------------------------------------------------------

def test_tout_not_flagged() -> None:
    """Tout is a protobuf attribute key (output type), not an op name."""
    assert _is_suspicious_op("Tout") is False


def test_tdim_not_flagged() -> None:
    assert _is_suspicious_op("Tdim") is False


def test_fullargspec_not_flagged() -> None:
    """FullArgSpec is Python function-signature metadata in saved functions."""
    assert _is_suspicious_op("FullArgSpec") is False


# ---------------------------------------------------------------------------
# 8. Newly whitelisted standard ops do not fire TF101
# ---------------------------------------------------------------------------

def test_select_not_flagged() -> None:
    assert _is_suspicious_op("Select") is False


def test_selectv2_not_flagged() -> None:
    assert _is_suspicious_op("SelectV2") is False


def test_staticregexfullmatch_not_flagged() -> None:
    assert _is_suspicious_op("StaticRegexFullMatch") is False


def test_disablecopyonread_not_flagged() -> None:
    assert _is_suspicious_op("DisableCopyOnRead") is False


def test_shardedfilename_not_flagged() -> None:
    """ShardedFilename is a standard checkpoint op present in any TF checkpoint graph."""
    assert _is_suspicious_op("ShardedFilename") is False


# ---------------------------------------------------------------------------
# 9. TC4 genuine signals preserved
# ---------------------------------------------------------------------------

def test_tc4_readfile_still_fires() -> None:
    assert _is_suspicious_op("ReadFile") is True


def test_tc4_writefile_still_fires() -> None:
    assert _is_suspicious_op("WriteFile") is True


def test_tc4_mergev2checkpoints_still_fires() -> None:
    assert _is_suspicious_op("MergeV2Checkpoints") is True


def test_tc4_savev2_still_fires() -> None:
    assert _is_suspicious_op("SaveV2") is True


def test_tc4_restorev2_still_fires() -> None:
    assert _is_suspicious_op("RestoreV2") is True


def test_tc4_evil_variable_name_fires() -> None:
    """TC4 payload 4: variable names with '____' + filesystem path component."""
    assert _is_suspicious_op("ad_disablecopyonread__________tmp_evil_weight") is True
    assert _is_suspicious_op("assignvariableop__________tmp_evil_weight") is True
    assert _is_suspicious_op("read_disablecopyonread__________tmp_evil_weight") is True


def test_tc4_serving_default_cmd_fires() -> None:
    """TC4 payload 3: lowercase node name containing 'cmd'."""
    assert _is_suspicious_op("serving_default_cmd") is True


def test_tc4_precision_after_fix() -> None:
    """A pb with TC4 noise ops + one genuine signal yields exactly the genuine findings.

    This models what the TC4 retest looks like after Phase 4 fixes: the
    12-16 findings per payload should drop to the handful of real signals.
    """
    # These are the common noise ops across all four TC4 payloads:
    noise = [
        "ShardedFilename", "DisableCopyOnRead", "Select", "StaticRegexFullMatch",
        "FullArgSpec", "Tout", "ableOp", "adVariableOp", "storeV2",
    ]
    genuine = ["ReadFile", "MergeV2Checkpoints", "SaveV2", "RestoreV2"]
    data = _build_pb(noise + genuine)
    findings = TensorFlowAnalyzer().analyze(_make_artifact_stream(data))
    tf101 = _tf101_ops(findings)
    # All four genuine ops must still fire.
    found = " ".join(tf101)
    for op in genuine:
        assert op in found, f"{op} not found in TF101 findings: {tf101}"
    # Noise ops must NOT fire. Use exact op='<name>' format to avoid substring
    # collisions (e.g. "storeV2" is a substring of "RestoreV2").
    for op in noise:
        exact = f"op='{op}'"
        assert not any(exact in v for v in tf101), f"Noise op {op!r} should not fire: {tf101}"


# ---------------------------------------------------------------------------
# 5. TF103 asset path traversal
# ---------------------------------------------------------------------------

def _write_saved_model(tmp_path: Path, asset_names: list[str]) -> Path:
    pb = tmp_path / "saved_model.pb"
    pb.write_bytes(_build_pb([]))
    if asset_names:
        assets = tmp_path / "assets"
        assets.mkdir()
        for name in asset_names:
            (assets / name).write_bytes(b"x")
    return pb


def test_no_assets_dir_no_finding(tmp_path: Path) -> None:
    pb = _write_saved_model(tmp_path, [])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_local(pb))
    assert not any(f.rule_id == "TF103" for f in findings)


def test_benign_asset_no_finding(tmp_path: Path) -> None:
    pb = _write_saved_model(tmp_path, ["vocab.txt", "labels.txt"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_local(pb))
    assert not any(f.rule_id == "TF103" for f in findings)


def test_url_encoded_traversal_asset_flagged(tmp_path: Path) -> None:
    pb = _write_saved_model(tmp_path, ["..%2f..%2fpwned.txt"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_local(pb))
    tf103 = [f for f in findings if f.rule_id == "TF103"]
    assert len(tf103) == 1
    assert tf103[0].severity.value == "CRITICAL"


def test_dotdot_asset_flagged(tmp_path: Path) -> None:
    # A literal '..' path component encoded in a single filename entry.
    pb = _write_saved_model(tmp_path, ["..%2fetc%2fpasswd"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_local(pb))
    assert any(f.rule_id == "TF103" for f in findings)


def test_traversal_indicator_helper() -> None:
    assert _traversal_indicator("normal.txt")[1] is None
    assert _traversal_indicator("../evil")[1] == "dotdot_component"
    assert _traversal_indicator("/etc/passwd")[1] == "absolute_posix"
    assert _traversal_indicator("..%2fx")[1].startswith("url_encoded_traversal")


def test_windows_backslash_traversal_asset_flagged(tmp_path: Path) -> None:
    """Test 6 — Windows backslash traversal in asset filename fires TF103."""
    # On Windows, the OS rejects '..' in filenames so we use URL-encoded form.
    pb = _write_saved_model(tmp_path, ["..%5c..%5cevil.py"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_local(pb))
    tf103 = [f for f in findings if f.rule_id == "TF103"]
    assert tf103, "TF103 must fire for %5c (backslash) path traversal"


def test_absolute_path_asset_flagged(tmp_path: Path) -> None:
    """Test 7 — Absolute POSIX path in asset filename fires TF103."""
    # Use URL-encoded form so the filesystem can represent the name.
    pb = _write_saved_model(tmp_path, ["%2fetc%2fpasswd"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_local(pb))
    tf103 = [f for f in findings if f.rule_id == "TF103"]
    assert tf103, "TF103 must fire for URL-encoded absolute path"


def test_normal_asset_no_tf103_finding(tmp_path: Path) -> None:
    """Test 8 — Normal asset filename does not fire TF103."""
    pb = _write_saved_model(tmp_path, ["tokenizer.json", "vocab.txt"])
    findings = TensorFlowAnalyzer().analyze(_make_artifact_local(pb))
    assert not any(f.rule_id == "TF103" for f in findings)


def test_traversal_indicator_windows_drive() -> None:
    """Windows absolute drive paths are detected."""
    assert _traversal_indicator("C:\\evil.py")[1] == "absolute_windows_drive"


def test_traversal_indicator_leading_backslash() -> None:
    assert _traversal_indicator("\\evil.py")[1] == "absolute_windows"
