# Project 2 — Keras Security Coverage

**Scanner version:** 2.0.0  
**Evaluation date:** 2026-09-02  
**Methodology:** Static acquisition and analysis only — no model inference, no `keras.models.load_model()`, no `eval()`/`exec()`, no deserialization of attacker-controlled objects.

---

## 1. Supported Formats

| Format | Extension | Analysis path |
|--------|-----------|---------------|
| Keras 3 archive | `.keras` | ZIP → `config.json` parse |
| Legacy Keras HDF5 | `.h5`, `.hdf5` | HDF5 → `model_config` attribute parse |

---

## 2. Implemented Rules

### KR101 — External or Custom Class in Keras Config

**Severity:** MEDIUM  **Confidence:** MEDIUM  **CWE:** CWE-94 (Code Injection)

Detects class names in the model config that are not standard built-in Keras classes. When a vulnerable `keras.models.load_model()` call deserializes the config, non-built-in class names are looked up in the custom-object registry and instantiated, which can execute attacker-controlled code.

**Detection logic:**
- Recursively extracts all `class_name` and `registered_name` values from the model config JSON.
- Also extracts bare strings from within list values under the keys `layers`, `objects`, and `nodes` (context-aware bare-string extraction).
- Any extracted name that is **not** in the built-in class allowlist (`_KERAS_BUILTIN_CLASSES`) fires KR101.

**Covered attack vectors:**
- `"class_name": "__lambda__"` — Lambda layer (arbitrary Python expression)
- `"class_name": "CustomEvilClass"` — Arbitrary attacker class via `custom_objects`
- `"registered_name": "AttackerClass"` — Registered custom class override
- Bare string `"MALICIOUS_LAMBDA"` inside a `layers` list
- Deeply nested custom class at any depth (recursive extraction)

**Not covered (by design):**
- Behavioral backdoors embedded entirely within standard operator weights
- Obfuscated class names that appear to be built-in (allowlist bypass via lookalike strings)

---

### KR102 — Missing or Malformed Config / ZIP Path Traversal

**Severity:** MEDIUM  **Confidence:** MEDIUM

Covers two detection branches:

**Branch A — Missing `config.json`:** A `.keras` ZIP that lacks a `config.json` member cannot be safely parsed and is flagged as structurally anomalous.

**Branch B — ZIP member path traversal:** Member names that contain path-traversal indicators are flagged. Python's `zipfile.ZipFile.read()` does not perform filesystem extraction, so there is no scanner-side path traversal risk; however, such names are a recognized indicator of archives crafted to exploit vulnerable extraction code in downstream consumers.

Traversal indicators checked:
- `..` components (after normalizing `\` → `/`)
- Absolute paths starting with `/`
- Windows-style absolute paths (`C:\...`)
- URL-encoded sequences: `%2e%2e` (dot-dot), `%2f` (slash), `%5c` (backslash)

---

### KR103 — Duplicate ZIP Member

**Severity:** HIGH  **Confidence:** HIGH  **CWE:** CWE-694 (Use of Multiple Resources with Duplicate Identifier), CWE-693 (Protection Mechanism Failure)

Python's `zipfile.ZipFile.read(name)` resolves to the **last** matching member when duplicate names exist. A `.keras` archive with two `config.json` entries allows an attacker to present a benign config during static inspection while ensuring the runtime sees a malicious one.

**Detection logic:** Counts member-name occurrences in `namelist()`; flags any name appearing more than once.

**TC5 status:** TRUE POSITIVE — `kais113/keras-dotkeras-zip-dup-poc` on HuggingFace fires both KR103 (`duplicate_member='config.json' count=2`) and KR101 (`class_name='MALICIOUS_LAMBDA'`).

---

## 3. Coverage Matrix

| Attack vector | Rule | Status |
|---------------|------|--------|
| Lambda layer via `class_name: "__lambda__"` | KR101 | **Covered** |
| Arbitrary custom class via `class_name` | KR101 | **Covered** |
| Custom class via `registered_name` | KR101 | **Covered** |
| Bare class string in `layers`/`objects`/`nodes` list | KR101 | **Covered** |
| Deeply nested custom class (3+ levels) | KR101 | **Covered** |
| Keras 3 internal classes (DTypePolicy, `__keras_tensor__`) | KR101 allowlist | **Not flagged (correct)** |
| Duplicate ZIP member hiding malicious config | KR103 | **Covered** |
| ZIP member with `..` path traversal | KR102 | **Covered** |
| ZIP member with absolute path | KR102 | **Covered** |
| ZIP member with URL-encoded traversal | KR102 | **Covered** |
| Missing `config.json` in `.keras` ZIP | KR102 | **Covered** |
| Semantic/behavioral backdoor in weights | — | **Not detectable (static scope)** |
| Obfuscated allowlist-bypass class name | — | **Not covered (heuristic risk)** |

---

## 4. Test Cases (TC)

### TC5 — Duplicate ZIP Member Hiding Lambda Layer

**Artifact:** `kais113/keras-dotkeras-zip-dup-poc` (HuggingFace)  
**Format:** `.keras` (ZIP)  
**Attack:** ZIP has two `config.json` entries; first is benign, second contains `class_name: "MALICIOUS_LAMBDA"`. Python's `zipfile` resolves to the second (malicious) entry at runtime.

| Finding | Rule | Evidence |
|---------|------|----------|
| KR103 | Duplicate ZIP member | `duplicate_member='config.json' count=2` |
| KR101 | Unsafe custom class | `class_name='MALICIOUS_LAMBDA'` |

**Verdict: TRUE POSITIVE**

---

## 5. Test Suite

All tests run without executing model code or loading Keras/TensorFlow.

| Test class | Tests | Coverage |
|------------|-------|----------|
| `TestMissingH5pyDependency` | 2 | h5py ImportError → must not return CLEAN |
| `TestKeras3InternalClassesSuppressed` | 3 | DTypePolicy, __keras_tensor__ not flagged |
| `TestLambdaLayerKR101Unchanged` | 3 | __lambda__ still fires KR101 (regression) |
| `TestDuplicateZipMemberKR103` | 4 | Duplicate name detection |
| `TestBareStringClassKR101` | 5 | Bare string + nested malicious object (TEST 6) |
| `TestZipPathTraversalKR102` | 5 | TEST 9: path traversal member names |
| **Total** | **22** | |

Full regression suite: **1213 passed, 4 skipped, 0 failed**

---

## 6. Known Limitations

1. **Behavioral backdoors:** Weight-space backdoors (e.g., ShadowLogic-style) encoded entirely via standard layers and numeric weights are undetectable by static structural analysis. This is a fundamental scope limitation, not a scanner defect.

2. **Allowlist completeness:** `_KERAS_BUILTIN_CLASSES` is based on the Keras 3.x public API surface. A very new or very old Keras release may have different built-in class names, causing false positives or false negatives.

3. **HDF5 scanning requires h5py:** The `.h5`/`.hdf5` analysis path depends on the optional `h5py` dependency. If `h5py` is unavailable, the analyzer raises `ImportError` (fail-secure: not a false CLEAN).

4. **Obfuscated class names:** An attacker who knows the built-in allowlist could craft a class name that appears to be a Keras built-in (e.g., using look-alike Unicode characters). This is not currently detected.
