"""
benchmarks.corpus.generator
===========================

Deterministic synthetic corpus generator for benchmark evaluation (Phase 11).

SAFETY GUARANTEE
----------------
- Zero model execution: never imports ML frameworks or executes model code.
- Purely synthetic binary byte blobs generated in-memory and saved to disk.
- Ground-truth metadata recorded in canonical manifest.json.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

from model_scanner.analyzers.gguf.reader import GGUF_MAGIC, GGUFValueType


def _make_safetensors(
    header_dict: dict[str, Any],
    data_section: bytes = b"",
) -> bytes:
    """Build a SafeTensors byte blob with length prefix."""
    json_bytes = json.dumps(header_dict).encode("utf-8")
    prefix = struct.pack("<Q", len(json_bytes))
    return prefix + json_bytes + data_section


def _pack_gguf_str(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _build_gguf_blob(
    version: int = 3,
    kv_pairs: list[tuple[str, int, Any]] | None = None,
    tensors: list[dict[str, Any]] | None = None,
    alignment: int = 32,
    data_section: bytes = b"",
) -> bytes:
    """Build synthetic GGUF binary blob."""
    kv_list = kv_pairs or []
    t_list = tensors or []

    magic = GGUF_MAGIC
    ver = struct.pack("<I", version)
    counts = struct.pack("<QQ", len(t_list), len(kv_list))
    buf = magic + ver + counts

    for key, vtype, val in kv_list:
        buf += _pack_gguf_str(key)
        buf += struct.pack("<I", vtype)
        if vtype == GGUFValueType.STRING:
            buf += _pack_gguf_str(str(val))
        elif vtype == GGUFValueType.UINT32:
            buf += struct.pack("<I", int(val))
        elif vtype == GGUFValueType.INT32:
            buf += struct.pack("<i", int(val))
        elif vtype == GGUFValueType.UINT64:
            buf += struct.pack("<Q", int(val))
        elif vtype == GGUFValueType.BOOL:
            buf += struct.pack("<B", 1 if val else 0)
        else:
            buf += _pack_gguf_str(str(val))

    for t in t_list:
        buf += _pack_gguf_str(t["name"])
        dims = t["dimensions"]
        buf += struct.pack("<I", len(dims))
        for d in dims:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<I", t["type"])
        buf += struct.pack("<Q", t["offset"])

    pad_len = (alignment - (len(buf) % alignment)) % alignment
    buf += bytes(pad_len)
    return buf + data_section


def _create_pytorch_zip(
    pickle_data: bytes,
    member_name: str = "archive/data.pkl",
) -> bytes:
    """Build in-memory PyTorch ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member_name, pickle_data)
    return buf.getvalue()


def generate_corpus_data() -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    """Generate all synthetic model samples and their ground-truth manifest records.

    Returns
    -------
    tuple[dict[str, bytes], list[dict[str, Any]]]
        Mapping of filename -> bytes and list of manifest entries.
    """
    files: dict[str, bytes] = {}
    manifest: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 1. CLEAN / BENIGN MODELS (Expected: BENIGN)
    # -----------------------------------------------------------------------
    # 1.1 Clean Pickle
    clean_pkl = b"\x80\x02}q\x00."
    files["clean_pickle.pkl"] = clean_pkl
    manifest.append({
        "case_id": "CLEAN-PKL-001",
        "filename": "clean_pickle.pkl",
        "format": "RAW_PICKLE",
        "category": "clean",
        "expected_classification": "BENIGN",
        "expected_outcome": "COMPLETE",
        "expected_rules": [],
        "description": "Standard benign Pickle empty dict payload (PROTO 2).",
    })

    # 1.2 Clean PyTorch ZIP
    clean_pt = _create_pytorch_zip(b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00.")
    files["clean_pytorch_zip.pt"] = clean_pt
    manifest.append({
        "case_id": "CLEAN-PT-001",
        "filename": "clean_pytorch_zip.pt",
        "format": "PYTORCH_ZIP",
        "category": "clean",
        "expected_classification": "BENIGN",
        "expected_outcome": "COMPLETE",
        "expected_rules": ["P209"],  # P209 is informational container evidence
        "description": "Benign modern PyTorch ZIP archive with torch tensor rebuild.",
    })

    # 1.3 Clean SafeTensors
    clean_st = _make_safetensors(
        {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F32", "shape": [2, 4], "data_offsets": [0, 32]},
        },
        data_section=bytes(32),
    )
    files["clean_safetensors.safetensors"] = clean_st
    manifest.append({
        "case_id": "CLEAN-ST-001",
        "filename": "clean_safetensors.safetensors",
        "format": "SAFETENSORS",
        "category": "clean",
        "expected_classification": "BENIGN",
        "expected_outcome": "COMPLETE",
        "expected_rules": [],
        "description": "Standard benign SafeTensors model with valid tensor data.",
    })

    # 1.4 Clean GGUF
    clean_gguf = _build_gguf_blob(
        version=3,
        kv_pairs=[
            ("general.architecture", GGUFValueType.STRING, "llama"),
            ("general.alignment", GGUFValueType.UINT32, 32),
        ],
        tensors=[
            {"name": "tok_embeddings.weight", "dimensions": [64, 64], "type": 0, "offset": 0}
        ],
        data_section=bytes(64 * 64 * 4),
    )
    files["clean_gguf.gguf"] = clean_gguf
    manifest.append({
        "case_id": "CLEAN-GU-001",
        "filename": "clean_gguf.gguf",
        "format": "GGUF",
        "category": "clean",
        "expected_classification": "BENIGN",
        "expected_outcome": "COMPLETE",
        "expected_rules": [],
        "description": "Standard benign GGUF v3 model file with valid metadata.",
    })

    # -----------------------------------------------------------------------
    # 2. MALICIOUS / SECURITY VIOLATIONS (Expected: MALICIOUS)
    # -----------------------------------------------------------------------
    # 2.1 Pickle P101 (os.system)
    files["malicious_pickle_p101_system.pkl"] = b"cos\nsystem\n(S'id'\ntR."
    manifest.append({
        "case_id": "MAL-PKL-P101-001",
        "filename": "malicious_pickle_p101_system.pkl",
        "format": "RAW_PICKLE",
        "category": "code_execution",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["P101"],
        "description": "Pickle payload importing os.system for arbitrary command execution.",
    })

    # 2.2 Pickle P101 (subprocess.Popen)
    files["malicious_pickle_p101_subprocess.pkl"] = b"csubprocess\nPopen\n(S'whoami'\ntR."
    manifest.append({
        "case_id": "MAL-PKL-P101-002",
        "filename": "malicious_pickle_p101_subprocess.pkl",
        "format": "RAW_PICKLE",
        "category": "code_execution",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["P101"],
        "description": "Pickle payload importing subprocess.Popen.",
    })

    # 2.3 Pickle P106 (builtins.eval)
    files["malicious_pickle_p106_eval.pkl"] = (
        b"cbuiltins\neval\n(S'__import__(\"os\").system(\"id\")'\ntR."
    )
    manifest.append({
        "case_id": "MAL-PKL-P106-001",
        "filename": "malicious_pickle_p106_eval.pkl",
        "format": "RAW_PICKLE",
        "category": "code_execution",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["P106"],
        "description": "Pickle payload invoking builtins.eval for dynamic code execution.",
    })

    # 2.4 PyTorch ZIP P101
    files["malicious_pytorch_zip_p101.pt"] = _create_pytorch_zip(
        b"cos\nsystem\n(S'rm -rf /'\ntR."
    )
    manifest.append({
        "case_id": "MAL-PT-P101-001",
        "filename": "malicious_pytorch_zip_p101.pt",
        "format": "PYTORCH_ZIP",
        "category": "code_execution",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["P101", "P209"],
        "description": "PyTorch ZIP container embedding malicious os.system Pickle stream.",
    })

    # 2.5 SafeTensors ST105 (Code Injection in __metadata__)
    files["malicious_safetensors_st105_eval.safetensors"] = _make_safetensors(
        {
            "__metadata__": {"exec": "import os; os.system('id')"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        data_section=bytes(4),
    )
    manifest.append({
        "case_id": "MAL-ST-ST105-001",
        "filename": "malicious_safetensors_st105_eval.safetensors",
        "format": "SAFETENSORS",
        "category": "metadata_injection",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["ST105"],
        "description": "SafeTensors header containing code injection payload in __metadata__.",
    })

    # 2.6 SafeTensors ST102 (Path Traversal)
    files["malicious_safetensors_st102_traversal.safetensors"] = _make_safetensors(
        {
            "__metadata__": {"template_path": "../../../etc/passwd"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        data_section=bytes(4),
    )
    manifest.append({
        "case_id": "MAL-ST-ST102-001",
        "filename": "malicious_safetensors_st102_traversal.safetensors",
        "format": "SAFETENSORS",
        "category": "metadata_injection",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["ST102"],
        "description": "SafeTensors header containing path traversal pattern in __metadata__.",
    })

    # 2.7 SafeTensors ST104 (Credential Exposure)
    files["malicious_safetensors_st104_secret.safetensors"] = _make_safetensors(
        {
            "__metadata__": {"api_token": "sk-proj-1234567890abcdef1234567890abcdef"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        data_section=bytes(4),
    )
    manifest.append({
        "case_id": "MAL-ST-ST104-001",
        "filename": "malicious_safetensors_st104_secret.safetensors",
        "format": "SAFETENSORS",
        "category": "credential_leak",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["ST104"],
        "description": "SafeTensors header containing exposed API key in __metadata__.",
    })

    # 2.8 GGUF GU101 (Metadata Code Injection)
    files["malicious_gguf_gu101_injection.gguf"] = _build_gguf_blob(
        version=3,
        kv_pairs=[
            ("general.architecture", GGUFValueType.STRING, "llama"),
            ("tokenizer.chat_template", GGUFValueType.STRING, "import os; os.system('calc')"),
        ],
        tensors=[],
    )
    manifest.append({
        "case_id": "MAL-GU-GU101-001",
        "filename": "malicious_gguf_gu101_injection.gguf",
        "format": "GGUF",
        "category": "metadata_injection",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["GU101"],
        "description": "GGUF metadata field containing executable Python injection payload.",
    })

    # 2.9 GGUF GU102 (Path Traversal in metadata)
    files["malicious_gguf_gu102_traversal.gguf"] = _build_gguf_blob(
        version=3,
        kv_pairs=[
            ("general.architecture", GGUFValueType.STRING, "llama"),
            ("tokenizer.model_path", GGUFValueType.STRING, "../../etc/shadow"),
        ],
        tensors=[],
    )
    manifest.append({
        "case_id": "MAL-GU-GU102-001",
        "filename": "malicious_gguf_gu102_traversal.gguf",
        "format": "GGUF",
        "category": "metadata_injection",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["GU102"],
        "description": "GGUF metadata field containing directory traversal sequence.",
    })

    # 2.10 GGUF GU103 (Credential Exposure)
    files["malicious_gguf_gu103_credential.gguf"] = _build_gguf_blob(
        version=3,
        kv_pairs=[
            ("general.architecture", GGUFValueType.STRING, "llama"),
            ("author.hf_token", GGUFValueType.STRING, "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"),
        ],
        tensors=[],
    )
    manifest.append({
        "case_id": "MAL-GU-GU103-001",
        "filename": "malicious_gguf_gu103_credential.gguf",
        "format": "GGUF",
        "category": "credential_leak",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["GU103"],
        "description": "GGUF metadata containing hardcoded Hugging Face access token.",
    })

    # 2.11 Format Spoofing G001 + P101
    files["malicious_spoofed_safetensors_g001.safetensors"] = (
        b"cos\nsystem\n(S'id'\ntR."
    )
    manifest.append({
        "case_id": "MAL-SPOOF-G001-001",
        "filename": "malicious_spoofed_safetensors_g001.safetensors",
        "format": "RAW_PICKLE",
        "category": "format_spoofing",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["G001", "P101"],
        "description": "File with .safetensors extension disguising malicious Pickle payload.",
    })

    # -----------------------------------------------------------------------
    # 3. STRUCTURAL ANOMALIES & INTEGRITY FINDINGS (Expected: MALICIOUS / ANOMALY)
    # -----------------------------------------------------------------------
    # 3.1 Pickle P206 (Concatenated Streams)
    files["anomaly_pickle_p206_concat.pkl"] = b"\x80\x02}q\x00.\x80\x02}q\x00."
    manifest.append({
        "case_id": "ANO-PKL-P206-001",
        "filename": "anomaly_pickle_p206_concat.pkl",
        "format": "RAW_PICKLE",
        "category": "structural_anomaly",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["P206"],
        "description": "Multiple concatenated Pickle streams delimited by STOP opcodes.",
    })

    # 3.2 Pickle P207 (Trailing unparsed bytes)
    files["anomaly_pickle_p207_trailing.pkl"] = (
        b"\x80\x02}q\x00.EXTRA_UNPARSED_BYTES_AFTER_STOP"
    )
    manifest.append({
        "case_id": "ANO-PKL-P207-001",
        "filename": "anomaly_pickle_p207_trailing.pkl",
        "format": "RAW_PICKLE",
        "category": "structural_anomaly",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["P207"],
        "description": "Pickle stream with trailing unparsed payload data.",
    })

    # 3.3 SafeTensors ST206 (Tensor offset out of file bounds)
    files["anomaly_safetensors_st206_outofbounds.safetensors"] = _make_safetensors(
        {
            "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 1000]},
        },
        data_section=bytes(8),
    )
    manifest.append({
        "case_id": "ANO-ST-ST206-001",
        "filename": "anomaly_safetensors_st206_outofbounds.safetensors",
        "format": "SAFETENSORS",
        "category": "structural_anomaly",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["ST206"],
        "description": "SafeTensors with tensor offsets extending beyond physical file bounds.",
    })

    # 3.4 SafeTensors ST207 (Overlapping tensor offsets)
    files["anomaly_safetensors_st207_overlap.safetensors"] = _make_safetensors(
        {
            "t1": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "t2": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        },
        data_section=bytes(24),
    )
    manifest.append({
        "case_id": "ANO-ST-ST207-001",
        "filename": "anomaly_safetensors_st207_overlap.safetensors",
        "format": "SAFETENSORS",
        "category": "structural_anomaly",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["ST207"],
        "description": "SafeTensors with overlapping data offset regions.",
    })

    # 3.5 GGUF GU209 (Tensor out of bounds)
    files["anomaly_gguf_gu209_outofbounds.gguf"] = _build_gguf_blob(
        version=3,
        kv_pairs=[("general.architecture", GGUFValueType.STRING, "llama")],
        tensors=[
            {"name": "tensor1", "dimensions": [16], "type": 0, "offset": 5000}
        ],
        data_section=bytes(64),
    )
    manifest.append({
        "case_id": "ANO-GU-GU209-001",
        "filename": "anomaly_gguf_gu209_outofbounds.gguf",
        "format": "GGUF",
        "category": "structural_anomaly",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["GU209"],
        "description": "GGUF model with tensor offset exceeding total file boundary.",
    })

    # 3.6 GGUF GU210 (Overlapping tensor ranges)
    files["anomaly_gguf_gu210_overlap.gguf"] = _build_gguf_blob(
        version=3,
        kv_pairs=[("general.architecture", GGUFValueType.STRING, "llama")],
        tensors=[
            {"name": "t1", "dimensions": [8], "type": 0, "offset": 0},
            {"name": "t2", "dimensions": [8], "type": 0, "offset": 16},
        ],
        data_section=bytes(64),
    )
    manifest.append({
        "case_id": "ANO-GU-GU210-001",
        "filename": "anomaly_gguf_gu210_overlap.gguf",
        "format": "GGUF",
        "category": "structural_anomaly",
        "expected_classification": "MALICIOUS",
        "expected_outcome": "WARNING",
        "expected_rules": ["GU210"],
        "description": "GGUF model with overlapping tensor memory ranges.",
    })

    # -----------------------------------------------------------------------
    # 4. UNSUPPORTED FORMATS (Expected: UNSUPPORTED)
    # -----------------------------------------------------------------------
    # 4.1 Legacy Non-ZIP PyTorch checkpoint
    files["unsupported_legacy_pytorch.bin"] = (
        b"\x80\x02}q\x00(X\x06\x00\x00\x00weightq\x01u."
        b"\x00\x00\x80?\x00\x00\x00@\xcd\xcc\x8c?RAW_FLOATS"
    )
    manifest.append({
        "case_id": "UNSUP-PT-LEGACY-001",
        "filename": "unsupported_legacy_pytorch.bin",
        "format": "UNSUPPORTED",
        "category": "unsupported",
        "expected_classification": "UNSUPPORTED",
        "expected_outcome": "UNSUPPORTED",
        "expected_rules": [],
        "description": "Legacy non-ZIP PyTorch serialization checkpoint.",
    })

    # 4.2 Unknown arbitrary format
    files["unsupported_unknown_format.xyz"] = b"RANDOM_UNSUPPORTED_DATA_BYTES_UNKNOWN_FORMAT"
    manifest.append({
        "case_id": "UNSUP-UNKNOWN-001",
        "filename": "unsupported_unknown_format.xyz",
        "format": "UNSUPPORTED",
        "category": "unsupported",
        "expected_classification": "UNSUPPORTED",
        "expected_outcome": "UNSUPPORTED",
        "expected_rules": [],
        "description": "Unrecognized arbitrary binary/text file format.",
    })

    # -----------------------------------------------------------------------
    # 5. MALFORMED / FATAL CORRUPTION (Expected: ERROR)
    # -----------------------------------------------------------------------
    # 5.1 Malformed Pickle (Truncated stream without STOP / P201)
    files["malformed_pickle_p201_truncated.pkl"] = b"\x80\x02(X\x04\x00\x00\x00data"
    manifest.append({
        "case_id": "MALF-PKL-P201-001",
        "filename": "malformed_pickle_p201_truncated.pkl",
        "format": "RAW_PICKLE",
        "category": "malformed",
        "expected_classification": "ERROR",
        "expected_outcome": "ERROR",
        "expected_rules": ["P201"],
        "description": "Truncated Pickle stream missing terminating STOP opcode.",
    })

    # 5.2 Malformed PyTorch ZIP container
    files["malformed_pytorch_corrupted_zip.pt"] = b"PK\x03\x04CORRUPTED_ZIP_STREAM_PAYLOAD"
    manifest.append({
        "case_id": "MALF-PT-ZIP-001",
        "filename": "malformed_pytorch_corrupted_zip.pt",
        "format": "PYTORCH_ZIP",
        "category": "malformed",
        "expected_classification": "ERROR",
        "expected_outcome": "ERROR",
        "expected_rules": ["P201"],
        "description": "Corrupted PyTorch ZIP container failing ZIP structure validation.",
    })

    # 5.3 Malformed SafeTensors (Truncated header / ST209)
    files["malformed_safetensors_st209_truncated.safetensors"] = b"\x00\x00\x00\x00"
    manifest.append({
        "case_id": "MALF-ST-ST209-001",
        "filename": "malformed_safetensors_st209_truncated.safetensors",
        "format": "SAFETENSORS",
        "category": "malformed",
        "expected_classification": "ERROR",
        "expected_outcome": "ERROR",
        "expected_rules": ["ST209"],
        "description": "Truncated SafeTensors file smaller than 8-byte length prefix.",
    })

    # 5.4 Malformed GGUF (Corrupted header magic / GU202)
    files["malformed_gguf_gu202_corrupted.gguf"] = b"GGUFCORRUPTED_TRUNCATED_HEADER"
    manifest.append({
        "case_id": "MALF-GU-GU202-001",
        "filename": "malformed_gguf_gu202_corrupted.gguf",
        "format": "GGUF",
        "category": "malformed",
        "expected_classification": "ERROR",
        "expected_outcome": "ERROR",
        "expected_rules": ["GU202"],
        "description": "GGUF file with corrupted header and invalid version.",
    })

    return files, manifest


def write_corpus(output_dir: Path) -> Path:
    """Generate and write all synthetic corpus files and manifest.json to disk.

    Parameters
    ----------
    output_dir:
        Directory where corpus files and manifest.json will be saved.

    Returns
    -------
    Path
        Path to the generated manifest.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files, manifest = generate_corpus_data()

    for filename, data in files.items():
        file_path = output_dir / filename
        file_path.write_bytes(data)

    manifest_path = output_dir / "manifest.json"
    manifest_doc = {
        "schema_version": "1.0.0",
        "description": "Deterministic synthetic model security evaluation corpus",
        "total_samples": len(manifest),
        "samples": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_doc, indent=2), encoding="utf-8")
    return manifest_path


__all__ = [
    "generate_corpus_data",
    "write_corpus",
]
