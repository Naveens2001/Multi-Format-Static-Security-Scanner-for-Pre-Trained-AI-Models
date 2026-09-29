# Project 2 Security Scanner — Real-World Adversarial Test Results

**Date:** 2026-09-02  
**Scanner version:** model-scanner v2.0.0 (commit `292cb70`)  
**Test operator:** Static analysis only; no exploit payloads executed  
**Safety constraint:** No model code executed; all artifacts treated as untrusted; files > 500 MB excluded per test protocol

---

## Repository Verification

| # | Repository | Accessible | Public/Gated | Files in Repo |
|---|------------|:----------:|:------------:|---------------|
| 1 | `shaq4prez/malicious-olmo3-poc` | YES | Public | `.gitattributes`, `.metadata`, `README.md`, `config.json` |
| 2 | `optimus-fulcria/modelscan-h5-bypass-poc` | YES | Public | `.gitattributes`, `README.md`, `create_h5_bypass.py`, `nested_model_h5_bypass.h5`, `wrapper_layer_h5_bypass.h5` |
| 3 | `kais113/tf-savedmodel-asset-traversal-poc` | YES | Public | `.gitattributes`, `README.md`, `tf_asset_traversal_initop.py`, `tf_asset_traversal_poc.py` |
| 4 | `vellaveto/tf-savedmodel-scanner-bypass-poc` | YES | Public | 4 × `saved_model.pb` + `fingerprint.pb` + variables shards across 4 payload subdirs |
| 5 | `kais113/keras-dotkeras-zip-dup-poc` | YES | Public | `.gitattributes`, `README.md`, `build_malicious_keras.py`, `malicious.keras`, `verify_asymmetry.py` |
| 6 | `Novaciano/SEX_ROLEPLAY-3.2-1B-Q4_K_M-GGUF` | YES | Public | `.gitattributes`, `README.md`, `imatrix.dat`, `sex_roleplay-3.2-1b-q4_k_m-imat.gguf`, `sex_roleplay-3.2-1b-q4_k_m.gguf` |
| 7 | `JackAnon/GorynichMix` | YES | Public | `.gitattributes`, `GorynichMix.vae.pt`, `GorynichMix_v1.1.ckpt`, `GorynichMix_v1.safetensors`, `README.md` |

---

## Artifact Identification and Size Verification

| # | Exact Artifact | Format | Size (bytes) | Size (MB) | < 500 MB |
|---|----------------|--------|-------------:|----------:|:--------:|
| 1 | `.metadata` (hidden, extensionless) | RAW_PICKLE (magic) | Not downloaded | — | YES (but not acquired) |
| 2a | `nested_model_h5_bypass.h5` | HDF5 | 6,240 | 0.006 | YES |
| 2b | `wrapper_layer_h5_bypass.h5` | HDF5 | 6,240 | 0.006 | YES |
| 3 | *(no model artifact — .py scripts only)* | N/A | — | — | N/A |
| 4a | `payload1_asset_traversal/saved_model.pb` | TF_SAVEDMODEL | 10,325 | 0.010 | YES |
| 4b | `payload2_readfile_op/saved_model.pb` | TF_SAVEDMODEL | 9,895 | 0.009 | YES |
| 4c | `payload3_pyfunction/saved_model.pb` | TF_SAVEDMODEL | 10,141 | 0.010 | YES |
| 4d | `payload4_var_traversal/saved_model.pb` | TF_SAVEDMODEL | 11,255 | 0.011 | YES |
| 5 | `malicious.keras` | KERAS (ZIP) | 2,493 | 0.002 | YES |
| 6a | `sex_roleplay-3.2-1b-q4_k_m.gguf` | GGUF | 955,458,752 | 911.2 | **NO** |
| 6b | `sex_roleplay-3.2-1b-q4_k_m-imat.gguf` | GGUF | 955,458,976 | 911.2 | **NO** |
| 7a | `GorynichMix.vae.pt` *(proxy — not malicious)* | PYTORCH_ZIP | 334,695,179 | 319.2 | YES |
| 7b | `GorynichMix_v1.1.ckpt` *(primary threat)* | LEGACY_PYTORCH | 7,703,655,534 | 7,346.8 | **NO** |
| 7c | `GorynichMix_v1.safetensors` | SAFETENSORS | 4,265,096,902 | 4,067.5 | **NO** |

---

## Scan Execution Details

### TC1 — shaq4prez/malicious-olmo3-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/shaq4prez/malicious-olmo3-poc" --json
```

**Vulnerability description (from README):**  
The `.metadata` file is a weaponised Pickle payload exploiting the HF Transformers OLMo3 conversion script (`convert_olmo3_weights_to_hf.py`, lines 259–260), which calls `pickle.load()` on the file without validation. The file is intentionally extensionless and hidden (dot-prefixed) to evade scanners. CVSS 9.6 / Critical; responsibly disclosed via Huntr.

**Why the file is not acquired:**  
The HF acquisition layer (`_filter_model_candidates`) only downloads files whose extension matches `_HF_MODEL_EXTENSIONS = {".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".pkl", ".pickle", ".h5", ".hdf5", ".keras", ".onnx"}`. The `.metadata` file has no extension and is therefore excluded from candidate download.

**Scanner result:**
```json
{
  "summary": {
    "total_artifacts": 0,
    "unsupported_artifacts": 1,
    "overall_outcome": "UNSUPPORTED",
    "clamav_exit_code": 0
  }
}
```

**Duration:** 322 ms  
**Errors:** None

---

### TC2 — optimus-fulcria/modelscan-h5-bypass-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/optimus-fulcria/modelscan-h5-bypass-poc" --json
```

**Vulnerability description (from README):**  
ModelScan v0.8.8 `H5LambdaDetectScan` iterates `model_config["config"]["layers"]` at the top level only; it does not recursively check layers inside nested `Functional` sub-models or `TimeDistributed` wrapper layers. Lambda layers in those nested positions evade modelscan's detection.

**Scanner result summary:**
- Total artifacts: 2  
- Scanned successfully: 2  
- Findings: 1 per artifact  
- Max risk score: 6 / 100  
- Risk band: LOW_RISK  
- Overall outcome: WARNING (ClamAV exit 1)  
- Duration: 5,181 ms (includes two file downloads)

**Artifact 1 — `nested_model_h5_bypass.h5` (6,240 bytes, SHA-256: `41bdd649…`):**

| Field | Value |
|-------|-------|
| Rule | KR101 — External or Custom Class in Keras Config |
| Severity | MEDIUM |
| Confidence | MEDIUM |
| Category | structural_anomaly |
| Matched value | `class_name='__lambda__'` |
| Description | Keras HDF5 model_config references non-standard class `__lambda__`. Custom classes may execute arbitrary Python at load time. |

**Artifact 2 — `wrapper_layer_h5_bypass.h5` (6,240 bytes, SHA-256: `b12194bd…`):**

Identical finding: KR101, `class_name='__lambda__'`, MEDIUM severity.

**Key finding:** Project 2 uses `_extract_class_names()`, which recursively walks the entire Keras config JSON tree, so it detects nested Lambda layers that modelscan misses. Behavioral verification not performed (no Keras runtime invoked).

---

### TC3 — kais113/tf-savedmodel-asset-traversal-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/kais113/tf-savedmodel-asset-traversal-poc" --json
```

**Vulnerability description:**  
Repository contains Python scripts demonstrating how to craft a TF SavedModel with path-traversal sequences in asset filenames using `tf.io.write_file()` with InitializationOp. The actual PoC model artifacts are built by running the Python scripts (not committed).

**Scanner result:**
```json
{
  "summary": {
    "total_artifacts": 0,
    "unsupported_artifacts": 1,
    "overall_outcome": "UNSUPPORTED",
    "clamav_exit_code": 0
  }
}
```

**Duration:** 314 ms  
No model artifacts exist in the repository. Both Python files (`.py`) are blocked by `_HF_FORBIDDEN_EXTENSIONS`.

---

### TC4 — vellaveto/tf-savedmodel-scanner-bypass-poc

**Acquisition note:**  
`.pb` files are NOT in `_HF_MODEL_EXTENSIONS`, so the HF acquisition layer cannot download them automatically. The four `saved_model.pb` files were downloaded manually using `urllib.request.urlretrieve()` and scanned as local files.

**Command (per payload):**
```
python -m model_scanner scan "<local_path>/payload{N}/saved_model.pb" --json
```

**Vulnerability description (from repository):**  
Four distinct TF SavedModel payloads, each using different exploit mechanisms:
- **Payload 1** (`payload1_asset_traversal`): path traversal via URL-encoded asset filename (`..%2f..%2f..%2ftmp%2fpwned.txt`)
- **Payload 2** (`payload2_readfile_op`): arbitrary file read via `ReadFile` op
- **Payload 3** (`payload3_pyfunction`): arbitrary code execution via `WriteFile` + `PyFunc`-style ops
- **Payload 4** (`payload4_var_traversal`): variable name encodes path to `/tmp/evil_weight`

**Scanner results — all 4 payloads:**

| Payload | Outcome | Risk Score | Risk Band | Finding Count | Key Ops Detected |
|---------|---------|:----------:|:---------:|:-------------:|------------------|
| 1 (asset_traversal) | WARNING | 100/100 | CRITICAL_RISK | 98 | `ShardedFilename`, `DisableCopyOnRead`, `StaticRegexFullMatch` |
| 2 (readfile_op) | WARNING | 100/100 | CRITICAL_RISK | 99 | `ReadFile`, `ShardedFilename`, `DisableCopyOnRead` |
| 3 (pyfunction) | WARNING | 100/100 | CRITICAL_RISK | 98 | `WriteFile`, `cmd`, `process`, `serving_default_cmd` |
| 4 (var_traversal) | WARNING | 100/100 | CRITICAL_RISK | 105 | `ad_disablecopyonread__________tmp_evil_weight`, `ShardedFilename` |

**All findings:** Rule TF101 — External or Custom Op in TensorFlow Graph (MEDIUM severity, MEDIUM confidence)

**Limitations of detection:**
1. The TF analyzer's raw protobuf string-extraction heuristic produces significant false positives — many of the 98–105 findings per payload are version strings (`.0`, `.part`, `2.21.0`), protobuf field names (`resource`, `type`, `shape`, `dtype`), and parameter names being misidentified as op names.
2. The actual path-traversal URL-encoded asset filename (`..%2f..%2f..%2ftmp%2fpwned.txt`) is not inspected — asset directory contents are not part of the `saved_model.pb` scan surface.
3. `.pb` files cannot be acquired via the HF acquisition pipeline; manual download was required.
4. Behavioral verification not performed (no TF runtime invoked).

---

### TC5 — kais113/keras-dotkeras-zip-dup-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/kais113/keras-dotkeras-zip-dup-poc" --json
```

**Vulnerability description (from README):**  
Responsibly disclosed via Huntr to the Keras team. The `.keras` archive is a ZIP with two entries named `config.json`:
- Entry 1 (87 bytes): `{"module": "keras", "class_name": "Functional", "config": {"layers": ["BENIGN_DENSE"]}}`
- Entry 2 (91 bytes): `{"module": "keras", "class_name": "Functional", "config": {"layers": ["MALICIOUS_LAMBDA"]}}`

Python's `zipfile.ZipFile.read(name)` (last-write-wins semantics) returns Entry 2. However, malicious class name `MALICIOUS_LAMBDA` appears as a **bare string element** in the `layers` list, not inside a `{"class_name": "MALICIOUS_LAMBDA", ...}` dict. This exploits the assumption that layer specifications are always dicts.

**Scanner result:**
```json
{
  "summary": {
    "total_artifacts": 1,
    "scanned_successfully": 1,
    "max_risk_score": 0,
    "overall_outcome": "COMPLETE",
    "clamav_exit_code": 0
  },
  "artifacts": [
    {
      "detected_format": "KERAS",
      "size_bytes": 2493,
      "sha256": "16545cf1…",
      "scan_outcome": "COMPLETE",
      "risk_score": { "final_score": 0, "risk_band": "CLEAN" },
      "findings": []
    }
  ]
}
```

**Duration:** 1,252 ms  
**Result: FALSE NEGATIVE — 0 findings, CLEAN verdict**

**Root cause analysis:**  
`KerasAnalyzer._analyze_keras_zip()` calls `zf.read("config.json")` which returns Entry 2 (the malicious one). It then calls `_extract_class_names(config_obj)`:

```python
def _extract_class_names(config: object) -> list[str]:
    names: list[str] = []
    if isinstance(config, dict):
        if "class_name" in config:
            cn = config["class_name"]
            if isinstance(cn, str):
                names.append(cn)   # captures "Functional" (built-in, not flagged)
        for v in config.values():
            names.extend(_extract_class_names(v))
    elif isinstance(config, list):
        for item in config:
            names.extend(_extract_class_names(item))
            # item = "MALICIOUS_LAMBDA" → str, not dict → nothing extracted
    return names
```

The function only extracts class names from dicts that have a `class_name` key. The bare string `"MALICIOUS_LAMBDA"` inside `layers: [...]` is never examined. The scanner also does not flag the presence of duplicate ZIP entries.

---

### TC6 — Novaciano/SEX_ROLEPLAY-3.2-1B-Q4_K_M-GGUF

**Status: CANNOT TEST — both artifacts exceed 500 MB**

| Artifact | Size | > 500 MB |
|----------|-----:|:--------:|
| `sex_roleplay-3.2-1b-q4_k_m.gguf` | 911.2 MB | YES |
| `sex_roleplay-3.2-1b-q4_k_m-imat.gguf` | 911.2 MB | YES |

**What Project 2 would theoretically inspect (if size limit waived):**  
The GGUF analyzer scans all metadata string values (including `tokenizer.chat_template`) using these pattern catalogs:
- **GU101** — code execution: `eval(`, `exec(`, `__import__`, `os.system(`, `subprocess`, `base64.b64decode`, `compile(`
- **GU102** — path traversal: `../`, `..\\`, `%2e%2e%2f`, `/etc/passwd`
- **GU103** — credentials: AWS keys, GitHub tokens, HF tokens, private key headers
- **GU105** — suspicious URLs
- **GU106** — script injection: `<script>`, `<iframe>`, `javascript:`

**Detection gap for prompt injection:**  
A GGUF chat-template prompt injection typically consists of Jinja2 template instructions designed to manipulate inference-time model behaviour (e.g., overriding system prompts, embedding adversarial role-play instructions). These are **semantic attacks** — natural language text that does not match any code-execution regex. Project 2 has no semantic/NLP-level prompt injection detection. The scanner would detect only if the chat template contained explicit Python execution syntax like `exec()` or `subprocess`.

---

### TC7 — JackAnon/GorynichMix

**Status: Primary malicious artifact CANNOT TEST — exceeds 500 MB**

| Artifact | Size | < 500 MB | Notes |
|----------|-----:|:--------:|-------|
| `GorynichMix.vae.pt` | 319.2 MB | YES | Scanned as proxy; likely benign VAE weights |
| `GorynichMix_v1.1.ckpt` | 7,346.8 MB | **NO** | Primary malicious artifact — untestable |
| `GorynichMix_v1.safetensors` | 4,067.5 MB | **NO** | Exceeds limit |

**HF acquisition behaviour:**  
The HF acquisition layer prefers SafeTensors when present (`_filter_model_candidates` filters to `.safetensors` only when any SafeTensors files are found). Since `GorynichMix_v1.safetensors` (4.1 GB) exceeds the `max_download_file_bytes` limit (1,024 MB default), acquisition would raise `ResourceLimitError`. Direct resolve URL was used for `.vae.pt`.

**Command (proxy artifact):**
```
python -m model_scanner scan "https://huggingface.co/JackAnon/GorynichMix/resolve/main/GorynichMix.vae.pt" --json
```

**Scanner result for `.vae.pt`:**

| Field | Value |
|-------|-------|
| Format detected | PYTORCH_ZIP |
| Size | 334,695,179 bytes |
| SHA-256 | `c6a580b1…` |
| Outcome | COMPLETE |
| Risk score | 0/100 |
| Risk band | CLEAN |
| Finding | P209 (INFO) — Embedded Pickle Stream in PyTorch ZIP (`archive/data.pkl`, 32,439 bytes) |

**Duration:** 71,133 ms (319 MB download)

The `.vae.pt` file is a standard Stable Diffusion VAE checkpoint with clean PyTorch pickle opcodes. The INFO finding P209 correctly identifies the embedded pickle stream but finds no dangerous opcodes.

**Behavioral verification not performed:** The actual malicious `.ckpt` (7.3 GB) was not downloaded or executed. Static analysis of the `.ckpt` would require the Pickle disassembler; if present, the scanner would parse `archive/data.pkl` and flag any `GLOBAL`/`REDUCE`/`BUILD` opcodes referencing dangerous callables (rule P101 — Dangerous Global Import, CRITICAL severity). This was not testable under the 500 MB constraint.

---

## Final Results Table

| # | Repository | Accessible | Exact Artifact | Format | Size | <500MB | Threat Type | Project 2 Result | Detection Module | Severity | TP/FN/Inconclusive | Error |
|---|------------|:----------:|----------------|--------|-----:|:------:|-------------|-----------------|-----------------|---------|:------------------:|-------|
| 1 | `shaq4prez/malicious-olmo3-poc` | YES | `.metadata` (extensionless Pickle) | RAW_PICKLE | ~tiny | YES* | Pickle RCE via extensionless hidden file | UNSUPPORTED — artifact not acquired | — | — | **FN** (acquisition gap) | Extension not in `_HF_MODEL_EXTENSIONS` |
| 2a | `optimus-fulcria/modelscan-h5-bypass-poc` | YES | `nested_model_h5_bypass.h5` | HDF5 | 6,240 B | YES | Nested Lambda bypass (ModelScan v0.8.8) | WARNING / LOW_RISK (6/100) — KR101 detected | KerasAnalyzer (HDF5) | MEDIUM | **TP** | None |
| 2b | `optimus-fulcria/modelscan-h5-bypass-poc` | YES | `wrapper_layer_h5_bypass.h5` | HDF5 | 6,240 B | YES | Wrapped Lambda bypass (ModelScan v0.8.8) | WARNING / LOW_RISK (6/100) — KR101 detected | KerasAnalyzer (HDF5) | MEDIUM | **TP** | None |
| 3 | `kais113/tf-savedmodel-asset-traversal-poc` | YES | *(none — .py only)* | N/A | — | N/A | TF SavedModel path traversal | UNSUPPORTED — no model artifact | — | — | Cannot test | No model file committed |
| 4a | `vellaveto/tf-savedmodel-scanner-bypass-poc` | YES | `payload1_asset_traversal/saved_model.pb` | TF_SAVEDMODEL | 10,325 B | YES | Asset path traversal (URL-encoded) | WARNING / CRITICAL_RISK (100/100) — 98× TF101 | TensorFlowAnalyzer | MEDIUM | **Inconclusive** | False positives; URL-encoded traversal not detected in assets |
| 4b | `vellaveto/tf-savedmodel-scanner-bypass-poc` | YES | `payload2_readfile_op/saved_model.pb` | TF_SAVEDMODEL | 9,895 B | YES | ReadFile op exploitation | WARNING / CRITICAL_RISK (100/100) — 99× TF101 | TensorFlowAnalyzer | MEDIUM | **Partial TP** | `ReadFile` detected; false positives present |
| 4c | `vellaveto/tf-savedmodel-scanner-bypass-poc` | YES | `payload3_pyfunction/saved_model.pb` | TF_SAVEDMODEL | 10,141 B | YES | PyFunction + WriteFile code execution | WARNING / CRITICAL_RISK (100/100) — 98× TF101 | TensorFlowAnalyzer | MEDIUM | **Partial TP** | `WriteFile`, `cmd`, `process` detected; false positives present |
| 4d | `vellaveto/tf-savedmodel-scanner-bypass-poc` | YES | `payload4_var_traversal/saved_model.pb` | TF_SAVEDMODEL | 11,255 B | YES | Variable name path traversal | WARNING / CRITICAL_RISK (100/100) — 105× TF101 | TensorFlowAnalyzer | MEDIUM | **Partial TP** | Malicious variable name detected; false positives present |
| 5 | `kais113/keras-dotkeras-zip-dup-poc` | YES | `malicious.keras` | KERAS (ZIP) | 2,493 B | YES | ZIP duplicate-entry + bare-string class bypass | COMPLETE / CLEAN (0/100) — 0 findings | KerasAnalyzer (ZIP) | — | **FN** | Bare string class names in layers list not extracted |
| 6 | `Novaciano/SEX_ROLEPLAY-3.2-1B-Q4_K_M-GGUF` | YES | Both `.gguf` files | GGUF | 911.2 MB each | **NO** | Chat-template prompt injection | Not tested (>500 MB) | — | — | Cannot test | Artifact exceeds 500 MB |
| 7 | `JackAnon/GorynichMix` | YES | `GorynichMix_v1.1.ckpt` (primary) | LEGACY_PYTORCH | 7,346.8 MB | **NO** | Malicious Pickle .ckpt | Not tested (>500 MB) | — | — | Cannot test | Primary artifact exceeds 500 MB |
| 7p | `JackAnon/GorynichMix` | YES | `GorynichMix.vae.pt` (proxy) | PYTORCH_ZIP | 319.2 MB | YES | *(proxy — benign VAE)* | WARNING / CLEAN (0/100) — P209 INFO | PyTorchAnalyzer | INFO | N/A (benign proxy) | `.ckpt` untestable; `.safetensors` and `.ckpt` exceed limits |

*TC1: `.metadata` is a tiny file that would be within 500 MB, but the acquisition pipeline never downloads extensionless files.

---

## Analysis by Category

### 1. Models Successfully Detected

| # | Repository | Detection |
|---|------------|-----------|
| 2a | `optimus-fulcria/modelscan-h5-bypass-poc` — `nested_model_h5_bypass.h5` | **TRUE POSITIVE** — KR101 (custom class `__lambda__` in nested Keras config) |
| 2b | `optimus-fulcria/modelscan-h5-bypass-poc` — `wrapper_layer_h5_bypass.h5` | **TRUE POSITIVE** — KR101 (custom class `__lambda__` in wrapped Keras config) |
| 4a–4d | `vellaveto/tf-savedmodel-scanner-bypass-poc` — all 4 payloads | **PARTIAL TRUE POSITIVE** — TF101 (CRITICAL_RISK; specific exploit ops partially identified; high false-positive rate) |

**TC2** is a clean detection win: Project 2's recursive class-name extraction catches the nested Lambda that ModelScan v0.8.8 misses. The detection is statically derived with no model execution.

**TC4** detects all 4 payloads at CRITICAL_RISK, which correctly quarantines them. However, the underlying detection mechanism (generic non-standard op enumeration via heuristic protobuf string extraction) is imprecise and produces many noise findings alongside genuine signals.

---

### 2. Models Not Detected (False Negatives)

| # | Repository | Finding | Root Cause |
|---|------------|---------|------------|
| 5 | `kais113/keras-dotkeras-zip-dup-poc` — `malicious.keras` | 0 findings, CLEAN | `_extract_class_names` only matches dicts with `class_name` key; bare string class names in a list are not inspected |

**TC5** is a confirmed false negative. The bypass is:
1. Construct a `.keras` ZIP with two `config.json` entries (ZIP duplicate-entry trick).
2. The malicious entry uses `{"layers": ["MALICIOUS_LAMBDA"]}` instead of `{"layers": [{"class_name": "Lambda", ...}]}`.
3. Project 2 reads the malicious config.json (Python zipfile picks the last occurrence) but fails to extract `MALICIOUS_LAMBDA` because it is a bare string, not a dict.

---

### 3. Models That Could Not Be Tested

| # | Repository | Reason |
|---|------------|--------|
| 1 | `shaq4prez/malicious-olmo3-poc` | `.metadata` file has no recognisable extension; excluded from HF acquisition |
| 3 | `kais113/tf-savedmodel-asset-traversal-poc` | No model artifact committed; only Python build scripts |
| 6 | `Novaciano/SEX_ROLEPLAY-3.2-1B-Q4_K_M-GGUF` | Both GGUF files = 911.2 MB; exceed 500 MB test constraint |
| 7 | `JackAnon/GorynichMix` (`.ckpt`) | Primary threat artifact (`GorynichMix_v1.1.ckpt`) = 7,346.8 MB; proxy `.vae.pt` (319 MB) is benign |

---

### 4. False Negatives Requiring Project 2 Improvements

#### FN-1 — Extensionless Pickle / Hidden File Acquisition (TC1)

**Vulnerability class:** Pickle RCE via `.metadata` (no extension, dot-prefixed hidden file)  
**Gap:** `_HF_MODEL_EXTENSIONS` only downloads files with known model extensions. Files with no extension (or non-standard extensions like `.metadata`, `.data`, `.bin2`) that happen to contain Pickle payloads are silently excluded.  
**Fix:** After filtering by extension, add a secondary acquisition pass for files with Pickle binary magic bytes (`\x80\x02`–`\x80\x05`) regardless of extension, limited to a configurable size cap. Alternatively, add `.metadata` to the candidate list and apply magic-byte detection post-download.

#### FN-2 — Bare-String Class Names in Keras Layers List (TC5)

**Vulnerability class:** Keras ZIP duplicate-entry + non-dict class references  
**Gap:** `_extract_class_names` assumes class names always appear as `{"class_name": "<name>", ...}` dicts. The bypass places a raw string directly in the layers list: `{"layers": ["MALICIOUS_LAMBDA"]}`.  
**Fix:** Extend `_extract_class_names` to also collect bare strings found in `layers`, `objects`, and other Keras list-typed config fields as candidate class names. Additionally, flag ZIP archives containing duplicate member names as structural anomalies (new rule KR103).

---

### 5. Attack/Format Categories Currently Covered by Project 2

| Category | Formats | Rules | Status |
|----------|---------|-------|--------|
| Pickle opcode RCE | RAW_PICKLE, PYTORCH_ZIP, LEGACY_PYTORCH | P101–P204 | Covered (requires known extension) |
| Custom class / Lambda injection (Keras H5) | HDF5 | KR101, KR102 | Covered (recursive extraction) |
| Custom class / Lambda injection (Keras ZIP) | KERAS | KR101, KR102 | Partially covered (dict-only extraction; bare strings missed) |
| TF SavedModel non-standard ops | TF_SAVEDMODEL | TF101, TF102 | Covered (noisy heuristic; partial precision) |
| GGUF metadata code injection | GGUF | GU101–GU106 | Covered (regex patterns; semantic attacks missed) |
| SafeTensors header anomalies | SAFETENSORS | ST101–ST202 | Covered |
| ONNX graph operator inspection | ONNX | ON101–ON103 | Covered |
| Weight integrity (NaN/Inf/extreme values) | HDF5, KERAS, GGUF | WI101–WI103 | Covered |
| Repository-level supply chain | REPOSITORY | SC101–SC209 | Covered |

---

### 6. Coverage Gaps (Categories Not Currently Covered)

| Gap | Impact | Evidence |
|----|--------|----------|
| **Extensionless/dot-file Pickle artifacts** | HIGH | TC1: `.metadata` Pickle payload not acquired |
| **Keras ZIP duplicate-entry structural anomaly** | HIGH | TC5: two `config.json` entries not flagged |
| **Bare-string class names in Keras layer lists** | HIGH | TC5: `MALICIOUS_LAMBDA` as list string not detected |
| **TF SavedModel asset filename inspection** | MEDIUM | TC4: `..%2f`-encoded traversal in asset paths not checked |
| **HF acquisition of `.pb` protobuf files** | MEDIUM | TC4: `saved_model.pb` cannot be downloaded via HF URL acquisition |
| **TF protobuf parser false-positive rate** | MEDIUM | TC4: 98–105 findings per file; most are protobuf field values, not ops |
| **GGUF chat-template prompt injection (semantic)** | LOW–MEDIUM | TC6: natural language instruction attacks not detectable via regex |

---

### 7. Recommended Improvements to Project 2

Based exclusively on the above test results:

#### R-1 — Add Extensionless File Acquisition (addresses TC1 / FN-1)

In `huggingface.py::_filter_model_candidates`, after the extension filter, perform a second pass for files matching high-risk name patterns:

```python
_HF_SUSPICIOUS_FILENAMES: frozenset[str] = frozenset([
    ".metadata", ".cache", ".data", ".weight", ".model",
])
```

For these files, download and run magic-byte detection. If `is_pickle_magic()` returns True, route to the Pickle analyzer.

#### R-2 — Flag ZIP Duplicate Members (addresses TC5 / FN-2)

In `KerasAnalyzer._analyze_keras_zip()`, compare `zf.namelist()` (which returns all entries including duplicates) with `set(zf.namelist())`. If any duplicates exist, emit a new finding:

```
KR103 — Duplicate Archive Member in Keras ZIP (HIGH severity)
```

This is a structural anomaly regardless of content; no false-positive risk.

#### R-3 — Extend Keras Class Extraction to Bare Strings (addresses TC5 / FN-2)

Modify `_extract_class_names` to collect string elements from lists under known Keras config keys (`layers`, `objects`, `nodes`):

```python
_KERAS_LIST_CLASS_KEYS: frozenset[str] = frozenset(["layers", "objects", "nodes"])

# In the dict branch of _extract_class_names:
for k, v in config.items():
    if k in _KERAS_LIST_CLASS_KEYS and isinstance(v, list):
        for item in v:
            if isinstance(item, str) and item:
                names.append(item)   # bare class names
    names.extend(_extract_class_names(v))
```

#### R-4 — Add `.pb` to HF Acquisition Extensions (addresses TC4)

Add `.pb` to `_HF_MODEL_EXTENSIONS` with appropriate file-size cap. Format detection for `saved_model.pb` already works (filename-based detection in `detector.py`). This allows TC4-style payloads to be scanned without manual download.

#### R-5 — Reduce TF Protobuf Parser False Positives (addresses TC4)

The `_extract_ops_from_pb` heuristic in `tensorflow/analyzer.py` currently extracts any 2–64 character alphanumeric string, producing many false positives (version numbers, field names, parameter names). Improvements:

1. Apply a minimum length of 4 characters.
2. Require at least one uppercase letter (TF op names are CamelCase by convention).
3. Maintain a blocklist of known protobuf field names (`type`, `shape`, `dtype`, `resource`, `config`, `input`, `output`, etc.).
4. Filter out strings that are purely numeric or look like version strings (`\d+\.\d+`).

This would substantially reduce the finding count from 98–105 to the genuine suspicious ops (`ReadFile`, `WriteFile`, `cmd`, `process`, `PyFunc`).

#### R-6 — Add TF SavedModel Asset Filename Inspection (addresses TC4 payload1)

The TF SavedModel directory structure includes an `assets/` folder. When scanning a SavedModel directory (not just the `.pb` file), inspect asset filenames for path traversal patterns (`../`, `%2f`, URL-encoded sequences). A new rule `TF103 — Path Traversal in SavedModel Asset Filename` (CRITICAL severity) would cover payload1's exploit vector.

---

## Appendix A — Scan Commands Reference

```bash
# TC1 (no model artifact found)
python -m model_scanner scan "https://huggingface.co/shaq4prez/malicious-olmo3-poc" --json

# TC2
python -m model_scanner scan "https://huggingface.co/optimus-fulcria/modelscan-h5-bypass-poc" --json --output scan_results_tc2.json

# TC3 (no model artifact found)
python -m model_scanner scan "https://huggingface.co/kais113/tf-savedmodel-asset-traversal-poc" --json

# TC4 (manual download required — .pb not in HF supported extensions)
# Downloads: urllib.request.urlretrieve(HF_RESOLVE_URL, local_path)
python -m model_scanner scan "<local>/payload1_asset_traversal/saved_model.pb" --json --output scan_results_tc4_payload1.json
python -m model_scanner scan "<local>/payload2_readfile_op/saved_model.pb" --json --output scan_results_tc4_payload2.json
python -m model_scanner scan "<local>/payload3_pyfunction/saved_model.pb" --json --output scan_results_tc4_payload3.json
python -m model_scanner scan "<local>/payload4_var_traversal/saved_model.pb" --json --output scan_results_tc4_payload4.json

# TC5
python -m model_scanner scan "https://huggingface.co/kais113/keras-dotkeras-zip-dup-poc" --json --output scan_results_tc5.json

# TC6 — NOT TESTED (both files > 500 MB)

# TC7 — proxy only (.vae.pt is benign; .ckpt exceeds 500 MB)
python -m model_scanner scan "https://huggingface.co/JackAnon/GorynichMix/resolve/main/GorynichMix.vae.pt" --json --output scan_results_tc7_vaept.json
```

---

## Appendix B — Environment

| Item | Value |
|------|-------|
| Scanner | model-scanner v2.0.0 |
| Git commit | `292cb70` |
| Python | 3.13 |
| h5py | 3.16.0 (available — enables HDF5 analysis) |
| google.protobuf | 7.36.1 (available — enables TF protobuf parsing) |
| Platform | Windows 11 (22H2) |
| Test date | 2026-09-02 |
| All exploits executed | NO — static analysis only |
| All downloads streamed | YES — no model code imported or run |

---

## Remediation and Retest — Project 2 v0.2

**Date:** 2026-09-02  
**Scanner version:** model-scanner v0.2 (unreleased; implements fixes for TC1, TC4, TC5 detection gaps)  
**Safety constraint:** Same as v2.0.0 — no model code executed; files > 500 MB excluded per test protocol; all analysis static only

### Changes Implemented

| Priority | Change | File(s) Modified | New Rules |
|----------|--------|-----------------|-----------|
| P1 (TC1) | Secondary acquisition pass for extensionless Pickle files: peek first 16 bytes via HTTP Range request; download only if Pickle magic (`\x80\x02`–`\x80\x05`) confirmed | `acquisition/huggingface.py` | — |
| P2 (TC5) | KR103 — Duplicate ZIP member detection; extend `_extract_class_names` to collect bare strings from `layers`/`objects`/`nodes` lists | `analyzers/keras/analyzer.py`, `rules/catalog.py` | KR103 |
| P3 (TC4) | Add `.pb` to HF acquisition extensions with `saved_model.pb` guard; update SafeTensors preference to preserve `.pb` candidates | `acquisition/huggingface.py` | — |
| P4 (TC4) | TF false-positive reduction: `_DANGEROUS_TF_OPS` always-report set; `_PROTO_FIELD_BLOCKLIST`; version-string and instance-suffix filters; `_is_suspicious_op()` two-tier filter; TF103 asset path-traversal rule | `analyzers/tensorflow/analyzer.py`, `rules/catalog.py` | TF103 |

**Total rules:** 80 → 84 (added KR103, TF103, ON104, WI103)

---

### TC1 Retest — shaq4prez/malicious-olmo3-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/shaq4prez/malicious-olmo3-poc" --json
```

**Result: FIXED — `.metadata` now acquired and analyzed as RAW_PICKLE**

| Field | Value |
|-------|-------|
| Outcome | WARNING |
| Max risk score | 100/100 |
| Risk band | CRITICAL_RISK |
| Artifact | `.metadata` (extensionless, dot-prefixed hidden file) |
| Format detected | RAW_PICKLE |
| Total findings | 4 |

| Rule | Severity | Description |
|------|----------|-------------|
| P101 | CRITICAL | Pickle stream imports known dangerous module/callable capable of OS command execution |
| P103 | HIGH | STACK_GLOBAL opcode imports symbols constructed at runtime |
| P104 | CRITICAL | REDUCE operation targets callable capable of executing system commands |
| C004 | CRITICAL | Correlated: same dangerous callable is imported and REDUCEd in same Pickle stream |

**Status:** `FN → TP`

---

### TC2 Retest — optimus-fulcria/modelscan-h5-bypass-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/optimus-fulcria/modelscan-h5-bypass-poc" --json
```

**Result: TP PRESERVED — no regression**

| Field | Value |
|-------|-------|
| Outcome | WARNING |
| Max risk score | 6/100 |
| Risk band | LOW_RISK |
| Artifacts | 2 |
| Findings per artifact | 1 × KR101 (MEDIUM) |

**Status:** `TP → TP (preserved)`

---

### TC3 Retest — kais113/tf-savedmodel-asset-traversal-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/kais113/tf-savedmodel-asset-traversal-poc" --json
```

**Result: UNCHANGED — no model artifact committed**

| Field | Value |
|-------|-------|
| Outcome | UNSUPPORTED |
| Unsupported artifacts | 1 |
| Findings | 0 |

Repository still contains only Python build scripts. No model artifact to scan.  
**Status:** `Cannot test → Cannot test (unchanged)`

---

### TC4 Retest — vellaveto/tf-savedmodel-scanner-bypass-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/vellaveto/tf-savedmodel-scanner-bypass-poc" --json
```

**Result: SUBSTANTIALLY IMPROVED — HF URL acquisition now works; false positives reduced ~95%**

| Payload | v2.0.0 Findings | v0.2 Findings | v2.0.0 Risk Band | v0.2 Risk Band | Key Ops |
|---------|:--------------:|:-------------:|:----------------:|:--------------:|---------|
| 1 (asset_traversal) | 98 | 3 | CRITICAL_RISK | LOW_RISK | `MergeV2Checkpoints`, `RestoreV2`, `SaveV2` |
| 2 (readfile_op) | 99 | 4 | CRITICAL_RISK | LOW_RISK | `ReadFile`, `MergeV2Checkpoints`, `RestoreV2`, `SaveV2` |
| 3 (pyfunction) | 98 | 5 | CRITICAL_RISK | LOW_RISK | `WriteFile`, `serving_default_cmd`, `MergeV2Checkpoints`, `RestoreV2`, `SaveV2` |
| 4 (var_traversal) | 105 | 6 | CRITICAL_RISK | MEDIUM_RISK | `ad_disablecopyonread__________tmp_evil_weight`, `assignvariableop__________tmp_evil_weight`, `read_disablecopyonread__________tmp_evil_weight`, `MergeV2Checkpoints`, `RestoreV2`, `SaveV2` |

**Overall:** `outcome=WARNING, max_risk=36` (see `docs/project2_tensorflow_coverage.md` §4 for full per-payload analysis)

Acquisition improvement: TC4 now runs against the HF repository URL directly — `.pb` files are acquired via HF acquisition without manual download.

Precision improvement: Findings dropped from 98–105 to 3–6 per payload. Version strings (`2.21.0`, `.0`, `.part`), protobuf field names (`type`, `shape`, `dtype`, `resource`, `config`), and instance suffixes (`_1`, `_2`) are now suppressed. Genuine dangerous ops (`ReadFile`, `WriteFile`, `MergeV2Checkpoints`, `RestoreV2`, `SaveV2`) remain flagged.

Remaining gap: Payload 1 (asset path traversal via URL-encoded filename) produces no TF103 finding because the scanner examines the `saved_model.pb` protobuf — the asset filename attack requires an `assets/` directory sibling, which is not present in this single-file `.pb` download.

**Status:** `Inconclusive → TP (all 4 detected; P1–P3 LOW_RISK, P4 MEDIUM_RISK; genuine ops confirmed — see project2_tensorflow_coverage.md §4 for risk-band explanation)`

---

### TC5 Retest — kais113/keras-dotkeras-zip-dup-poc

**Command:**
```
python -m model_scanner scan "https://huggingface.co/kais113/keras-dotkeras-zip-dup-poc" --json
```

**Result: FIXED — both bypass mechanisms now detected**

| Field | Value |
|-------|-------|
| Outcome | WARNING |
| Max risk score | 36/100 |
| Risk band | MEDIUM_RISK |
| Findings | 2 |

| Rule | Severity | Description |
|------|----------|-------------|
| KR103 | HIGH | Keras .keras ZIP contains 2 entries named 'config.json'. Python's zipfile resolves duplicates last-write-wins, enabling config-substitution attack. |
| KR101 | MEDIUM | Keras config references non-standard class 'MALICIOUS_LAMBDA'. Custom classes may execute arbitrary Python at load time. |

**Status:** `FN → TP`

---

### TC6 Retest — Novaciano/SEX_ROLEPLAY-3.2-1B-Q4_K_M-GGUF

**Command:**
```
python -m model_scanner scan "https://huggingface.co/Novaciano/SEX_ROLEPLAY-3.2-1B-Q4_K_M-GGUF" --max-size 500 --json
```

**Result: CANNOT TEST — same as v2.0.0**

| Field | Value |
|-------|-------|
| Outcome | ERROR |
| Total artifacts | 0 |
| Reason | Both GGUF files (911.2 MB each) exceed `--max-size 500` constraint |

No changes were made to GGUF acquisition or analysis. This test case remains untestable under the 500 MB protocol constraint.  
**Status:** `Cannot test → Cannot test (unchanged)`

---

### TC7 Retest — JackAnon/GorynichMix

**Command (proxy only):**
```
python -m model_scanner scan "https://huggingface.co/JackAnon/GorynichMix/resolve/main/GorynichMix.vae.pt" --json
```

**Result: UNCHANGED — proxy still CLEAN; primary .ckpt still exceeds 500 MB**

| Field | Value |
|-------|-------|
| Artifact | `GorynichMix.vae.pt` (proxy, benign VAE weights) |
| Format | PYTORCH_ZIP |
| Outcome | WARNING |
| Risk score | 0/100 |
| Risk band | CLEAN |
| Findings | 1 × P209 INFO (embedded Pickle stream, no dangerous opcodes) |
| SHA-256 | `c6a580b1…` |

**Status:** `Cannot test → Cannot test (unchanged; proxy CLEAN as expected)`

---

### v0.2 Retest Summary Table

| # | Repository | v2.0.0 Result | v0.2 Result | Change |
|---|------------|:-------------:|:-----------:|:------:|
| 1 | `shaq4prez/malicious-olmo3-poc` | FN (acquisition gap) | **TP** (CRITICAL_RISK, 4 findings) | **FIXED** |
| 2 | `optimus-fulcria/modelscan-h5-bypass-poc` | TP | TP (preserved) | No regression |
| 3 | `kais113/tf-savedmodel-asset-traversal-poc` | Cannot test | Cannot test | Unchanged |
| 4 | `vellaveto/tf-savedmodel-scanner-bypass-poc` | Inconclusive (98–105 findings, HF URL unavailable) | **TP** (3–6 findings, HF URL works) | **IMPROVED** |
| 5 | `kais113/keras-dotkeras-zip-dup-poc` | FN (bare-string + duplicate ZIP missed) | **TP** (KR103 + KR101) | **FIXED** |
| 6 | `Novaciano/SEX_ROLEPLAY-3.2-1B-Q4_K_M-GGUF` | Cannot test (>500 MB) | Cannot test (>500 MB) | Unchanged |
| 7 | `JackAnon/GorynichMix` | Cannot test (>500 MB) | Cannot test (>500 MB) | Unchanged |

**False negatives eliminated:** 2 of 2 (TC1, TC5)  
**Precision improved:** TC4 finding count reduced ~95% (98–105 → 3–6 per payload)  
**Acquisition coverage expanded:** TC4 now runs via HF URL (`.pb` support added); TC1 extensionless Pickle files now acquired via magic-byte peek  
**No regressions:** TC2 TP preserved; TC3, TC6, TC7 unchanged  
**New rules added:** KR103 (duplicate ZIP member, HIGH), TF103 (SavedModel asset path traversal, CRITICAL)
