# Project 2 — Extended ONNX Security Evaluation

**Initial evaluation date:** 2026-09-02  
**Initial scanner commit:** `93368c0` — Fix adversarial model detection gaps  
**ON104 remediation commit:** see Section 10  
**Baseline commit:** `292cb70`  
**Static-only methodology; no inference executed**

---

## 1. Test Environment

| Item | Value |
|------|-------|
| Scanner | model-scanner v2.0.0 |
| Git commit | `93368c0` |
| OS | Windows 11 Home (10.0.26200) |
| Python | 3.13 |
| h5py | 3.16.0 |
| google.protobuf | 7.36.1 |
| onnx | available (used for static protobuf inspection only) |
| Evaluation size limit | 500 MB (--max-size 500) |
| Methodology | Static acquisition + static analysis only. No model inference, no ONNX Runtime, no OpenVINO, no TensorRT. |
| Safety | Every artifact treated as untrusted. No exploit payloads executed. |

---

## 2. Repository Verification

| TC | Repository | Type | Artifact | Size | Under 500 MB | Vulnerability | Accessible | Testable |
|----|------------|:----:|----------|-----:|:------------:|--------------|:----------:|:--------:|
| TC8 | `rez0/onnx-ort-path-traversal-poc` | Model | `model.onnx` | 159 B | YES | ONNX external-data path traversal / arbitrary file read (ONNX Runtime 1.23.2) | YES | YES |
| TC9 | `ScangGengy/shadow-cnn-onnx-poc` | Model | `shadow_cnn.onnx` | Unknown | — | Model-level backdoor via standard ONNX operators (ShadowLogic) | YES (API) | **NO — GatedRepo** |
| TC10 | `rez0/onnx-lrn-sigsegv-poc` | Model | `lrn_crash_poc.onnx` | 140 B | YES | Malformed LRN operator `size=200000001` causes SIGSEGV in ONNX Runtime | YES | YES |
| TC11 | `0xiviel/poc-openvino-onnx-uint64-overflow` | Dataset | `malicious_onnx_overflow.onnx` | 186 B | YES | uint64 integer overflow in external_data offset bypasses OpenVINO bounds check | YES | YES (via direct URL) |

**TC9 note:** The HF API lists `shadow_cnn.onnx` in the repository siblings. However, all direct resolve requests return HTTP 401 with `X-Error-Code: GatedRepo`. The model requires authenticated HF access and is not available for public download. The scanner confirms this with `overall_outcome: ERROR` (clamav exit code 2).

**TC11 note:** Repository is a HuggingFace dataset (not a model repository). The scanner's HF model-repository acquisition path does not handle dataset repos. The artifact was acquired via its direct resolve URL: `https://huggingface.co/datasets/0xiviel/poc-openvino-onnx-uint64-overflow/resolve/main/malicious_onnx_overflow.onnx`.

**Vulnerability descriptions verified against repository READMEs (not solely from task prompt):**
- TC8 README: "onnxruntime 1.23.2 loads ONNX models with path traversal in `external_data` location fields without validation. Impact: Arbitrary file read."
- TC9 README: "ONNX computational graphs can embed backdoor logic using only standard ONNX operators (no custom ops, no external code). 47 nodes, opset 18, all standard ONNX operators. Trigger: 2×2 red patch in top-left corner."
- TC10 README: "LRN `size` attribute set to 200000001. Validates `size > 0` and `size is odd`, but does not check reasonableness. Impact: SIGSEGV/SIGFPE (DoS) on inference."
- TC11 README: "ONNX external data `offset=0xFFFFFFFFFFFFFFF0`, `length=16`. Sum wraps to 0 mod 2^64, bypassing `m_offset + m_data_length > file_size` check. Confirmed on OpenVINO 2025.4.1."

---

## 3. Scan Results

| TC | Attack Class | Artifact | SHA-256 (first 8) | Scanner Result | Relevant Rule | Severity | Risk Score | Risk Band | Classification |
|----|-------------|----------|:-----------------:|----------------|:-------------:|:--------:|:----------:|:---------:|:--------------:|
| TC8 | ONNX external-data path traversal | `model.onnx` (159 B) | `54706759` | WARNING — 1 finding | ON103 | HIGH | 30/100 | LOW_RISK | **TP** |
| TC9 | Standard-operator backdoor | `shadow_cnn.onnx` | — (not acquired) | ERROR — acquisition failed | — | — | — | — | **CANNOT_TEST** |
| TC10 | LRN SIGSEGV (runtime DoS) | `lrn_crash_poc.onnx` (140 B) | `4aa4a234` | COMPLETE — 0 findings | — | — | 0/100 | CLEAN | **INCONCLUSIVE** |
| TC11 | uint64 overflow in external_data offset | `malicious_onnx_overflow.onnx` (186 B) | `3834d6e1` | COMPLETE — 0 findings *(v0.2)* → WARNING — 1 finding *(ON104)* | ON104 | HIGH | 30/100 | LOW_RISK | **FN → TP** |

---

## 4. Detailed Results

### TC8 — ONNX Runtime Path Traversal (`rez0/onnx-ort-path-traversal-poc`)

**Command:**
```
python -m model_scanner scan "https://huggingface.co/rez0/onnx-ort-path-traversal-poc" --max-size 500 --json --output scan_tc8.json
```

**PoC purpose:**  
Demonstrates that ONNX Runtime 1.23.2 loads models containing path traversal sequences in `external_data` location fields without validation. The `onnx.checker` would block this, but `ort.InferenceSession()` never calls the checker. Loading the model reads `/etc/hosts` (or any attacker-controlled path) and returns its bytes as float32 weights. Responsibly disclosed via Huntr.

**Static structure observed:**  
- Format: ONNX (opset 17, IR version 8)
- Single `Add` node using standard domain
- Initializer `W`: `data_location=EXTERNAL`, `location='../../../../etc/hosts'`, `offset=0`, `length=4`
- SHA-256: `54706759afd3eb55d04c323154740745c1cc1f9f6743ed2c17269e3d94ea6a18` — matches the exact PoC bytes embedded in `tests/unit/test_onnx_analyzer.py` (`_POC_MODEL_BYTES`)

**Scanner behavior:**  
ON103 fires immediately upon inspecting the initializer's `external_data[location]` field. The traversal check splits the location on `/`, identifies `..` components, and emits one HIGH finding.

**Finding detail:**

| Field | Value |
|-------|-------|
| Rule | ON103 — Path Traversal in ONNX External Data Location |
| Severity | HIGH |
| Confidence | HIGH |
| Matched value | `initializer='W' location='../../../../etc/hosts'` |
| Description | ONNX initializer 'W' references external tensor data at path '../../../../etc/hosts', which contains a path-traversal sequence. Loading this with ONNX Runtime would open an attacker-controlled file path. |
| CWE | CWE-22 |
| OWASP | ML06:2023 AI Supply Chain Attacks |
| Risk score | 30/100 (LOW_RISK) |
| Duration | 204 ms |

**Risk score note:** The LOW_RISK band (30/100) is correct — ON103 is a HIGH severity finding with HIGH confidence, and the format multiplier for ONNX is 1.0. The finding genuinely detects the path traversal that makes this PoC dangerous. The low band reflects the ONNX format's base risk multiplier, not the severity of the finding itself.

**Classification: TRUE POSITIVE**  
The static indicator of the attack (path traversal in `external_data[location]`) is directly detected by ON103 exactly as designed.

---

### TC9 — Standard-Operator ONNX Backdoor (`ScangGengy/shadow-cnn-onnx-poc`)

**Command attempted:**
```
python -m model_scanner scan "https://huggingface.co/ScangGengy/shadow-cnn-onnx-poc" --max-size 500 --json --output scan_tc9.json
```

**Acquisition result:**  
HTTP 401 — `X-Error-Code: GatedRepo`. The repository requires authenticated HuggingFace access. Scanner returned `overall_outcome: ERROR`, 0 artifacts.

**PoC purpose (from README, independently verified):**  
Demonstrates the "ShadowLogic" backdoor concept: ONNX computational graphs can encode inference-time backdoor behavior using only standard ONNX operators. The backdoor is triggered by a specific pixel pattern (2×2 red patch in the top-left corner of the input image), which forces class 0 prediction regardless of actual content. Architecture: 47 nodes, opset 18, all standard ONNX operators. No custom operator domain.

**Static structure observed:**  
Cannot be inspected — file not accessible. From the README, the model uses standard operators only and has no custom domains or external data references.

**Scanner behavior:**  
Acquisition failure — no scan performed.

**Analysis (reasoning from README description):**  
Even if the file were accessible, the classification would be **INCONCLUSIVE**, not FN. Reasoning:

1. **No custom operators:** ON101 fires for non-standard domains. With only standard ONNX ops, ON101 would not fire.
2. **No external data path traversal:** ON103 would not fire.
3. **No graph malformation:** ON102 (parse failure) would not fire.
4. **The backdoor is semantic:** The trigger condition is encoded in the mathematical weights and graph topology — specifically in the `Reduce`, `Gather`, or `Compare`-type operators that examine pixel values. Detecting this requires comparing model behavior against expected behavior, which is a semantic/adversarial ML problem, not a static structural security scan.
5. **Project 2's stated scope:** Static structural security analysis (malicious opcodes, dangerous operators, file-system attacks, supply chain indicators). Semantic backdoors are explicitly outside this scope.

A CLEAN result on TC9 would not be a false negative against Project 2's stated detection objectives. It would be an **expected limitation**: the scanner correctly analyzes ONNX structure but does not perform model semantics analysis or behavioral equivalence testing.

**Classification: CANNOT_TEST**  
Primary reason: GatedRepo (no authentication available). Secondary finding: Even if accessible, the attack is semantic/behavioral and cannot be detected by static structural analysis. A clean result would be INCONCLUSIVE, not FN.

---

### TC10 — ONNX LRN SIGSEGV PoC (`rez0/onnx-lrn-sigsegv-poc`)

**Command:**
```
python -m model_scanner scan "https://huggingface.co/rez0/onnx-lrn-sigsegv-poc" --max-size 500 --json --output scan_tc10.json
```

**PoC purpose:**  
Demonstrates that ONNX Runtime crashes with SIGSEGV when the LRN (Local Response Normalization) operator's `size` attribute is set to 200,000,001. The ONNX spec validates `size > 0` and `size is odd`, but does not impose an upper bound. ONNX Runtime allocates internal buffers proportional to `size` without checking system constraints, leading to a memory access violation and process termination (DoS).

**Static structure observed (via `onnx.load_model_from_string`, no inference):**
- Format: ONNX (opset 18, IR version 11)
- Single `LRN` node, domain `""` (standard)
- Operator attributes:
  - `alpha`: FLOAT = 0.0001 (standard default)
  - `beta`: FLOAT = 0.75 (standard default)
  - `bias`: FLOAT = 1.0 (standard default)
  - `size`: **INT = 200,000,001** (malformed — astronomically large for any practical input)
- 0 initializers, 0 external data references
- No path traversal, no custom operators, no external code

**Scanner behavior:**  
0 findings, CLEAN verdict. The LRN node is a standard ONNX operator in the default domain — ON101 does not fire. The graph parses correctly — ON102 does not fire. There is no external data — ON103 does not fire.

**Analysis:**  
The attack surface is the `size` attribute value `200,000,001`. This value is:
- Syntactically valid: the protobuf parses without error.
- Semantically valid per ONNX spec: `size` must be odd and positive — 200,000,001 satisfies both conditions.
- Harmful only at runtime: the SIGSEGV occurs during ONNX Runtime's kernel execution, when internal buffer allocation proportional to `size` exhausts memory.

Project 2 does not claim to detect malformed operator attribute values in standard operators that are only harmful at runtime. Detecting this statically would require heuristic rules such as "flag LRN nodes with `size > N`" — a rule that would need careful calibration to avoid false positives on legitimate large LRN configurations (which are rare but theoretically valid). No such rule exists in Project 2.

**Classification: INCONCLUSIVE**  
The malicious attribute (`size=200000001`) is statically visible, but the damage occurs exclusively during inference. Project 2's stated detection scope is static structural analysis of malicious model artifacts. A rule detecting extreme LRN `size` values would be a new capability (not a missed detection under any existing claim). This is an expected limitation rather than a false negative.

---

### TC11 — OpenVINO ONNX uint64 Overflow (`0xiviel/poc-openvino-onnx-uint64-overflow`)

**Command:**
```
python -m model_scanner scan "https://huggingface.co/datasets/0xiviel/poc-openvino-onnx-uint64-overflow/resolve/main/malicious_onnx_overflow.onnx" --max-size 500 --json --output scan_tc11.json
```

**PoC purpose:**  
Demonstrates an integer overflow in OpenVINO's ONNX external data bounds check. The validation in `tensor_external_data.cpp:47` is:
```cpp
if (file_size <= 0 || m_offset + m_data_length > static_cast<uint64_t>(file_size)) {
    throw error::invalid_external_data{*this};
}
```
With `offset = 0xFFFFFFFFFFFFFFF0` (= 18,446,744,073,709,551,600) and `length = 16`, the sum wraps to `0 (mod 2^64)`, passing the check. The subsequent pointer arithmetic `mapped_memory->data() + offset` produces a wild pointer to unmapped memory. Confirmed on OpenVINO 2025.4.1.

**Static structure observed (via `onnx.load_model_from_string`, no inference):**
- Format: ONNX (opset 13, IR version 7)
- Single `Add` node, standard domain
- Initializer `weights`: `data_location=EXTERNAL`
  - `location = 'overflow_data.bin'` (relative path, no traversal)
  - `offset = 18,446,744,073,709,551,600` (= `0xFFFFFFFFFFFFFFF0`)
  - `length = 16`
  - Arithmetic check: `offset + length = 0 (mod 2^64)` — the overflow

**Scanner behavior:**  
0 findings, CLEAN verdict. ON103 inspects `external_data[location]` for path traversal (`..` components, absolute paths, URL-encoded sequences). The location `'overflow_data.bin'` is a plain relative filename with no traversal indicators — ON103 does not fire. The overflow is in the `offset` field (a numeric value), which ON103 does not examine.

**Analysis:**  
The malicious condition is statically visible in the ONNX protobuf: the `offset` field contains `0xFFFFFFFFFFFFFFF0` — a value that is near the maximum of uint64 and will overflow when added to `length=16`. This value is anomalous: no legitimate ONNX model would reference tensor data starting at byte offset ~18.4 exabytes into a file.

Project 2 does not have a rule detecting anomalous numeric values in external_data `offset` or `length` fields. ON103 was designed for the path-traversal attack vector (the `location` string). The integer overflow attack is a distinct, statically detectable gap: a scanner could emit a finding when `offset + length > 2^64 - 1` or when `offset > some_reasonable_threshold` (e.g., 1 GB).

This is a genuine **detection gap** against a statically visible malicious condition. The location string itself is benign; only the offset value is malicious. The scanner's existing ON103 rule covers path traversal but not integer overflow in numeric metadata fields.

**Classification: FALSE NEGATIVE** *(v0.2 initial evaluation — `93368c0`)*  
The static indicator (extreme `offset` value causing uint64 wraparound) is present in the file and visible without runtime execution. Project 2 inspects external_data but only checks the `location` string, missing the `offset`/`length` overflow. A new rule (ON104 — External Data Offset/Length Overflow) was implemented to close this gap. See Section 10 for the remediation retest.

---

## 5. ONNX Coverage Summary

### Combined Results (Existing + New)

| ID | Repository | Attack Class | v0.2 Initial | After ON104 | Detected Rule |
|----|------------|-------------|:------------:|:-----------:|:-------------:|
| *Existing* | `0xiviel/poc-onnxruntime-sparse-oob` | Sparse tensor out-of-bounds | *(from prior eval)* | *(from prior eval)* | — |
| *Existing* | `0xiviel/poc-openvino-onnx-path-traversal` | OpenVINO path traversal | *(from prior eval)* | *(from prior eval)* | — |
| TC8 | `rez0/onnx-ort-path-traversal-poc` | ONNX RT external-data path traversal | **TP** | **TP** | ON103 |
| TC9 | `ScangGengy/shadow-cnn-onnx-poc` | Standard-operator semantic backdoor | **CANNOT_TEST** | **CANNOT_TEST** | — |
| TC10 | `rez0/onnx-lrn-sigsegv-poc` | LRN operator runtime DoS (SIGSEGV) | **INCONCLUSIVE** | **INCONCLUSIVE** | — |
| TC11 | `0xiviel/poc-openvino-onnx-uint64-overflow` | uint64 overflow in external_data offset | **FN** → | **TP** | ON104 |

### Detection Coverage Map

| Attack Vector | Project 2 Coverage | Notes |
|---------------|:-----------------:|-------|
| External data path traversal (`location` string) | **Covered** | ON103 detects `..`, absolute paths, URL-encoded sequences |
| External data integer overflow (`offset`/`length`) | **Covered** | ON104 detects offset+length >= 2^64 (see Section 10) |
| Standard-operator semantic backdoor | **Not covered** | Out of static-analysis scope; requires behavioral analysis |
| Runtime DoS via extreme operator attribute values | **Not covered** | TC10 — outside stated scope; no attribute-value heuristics |
| Sparse tensor out-of-bounds | *(evaluated separately)* | — |
| Custom/non-standard operator domains | **Covered** | ON101 |
| Malformed/unparseable ONNX graph | **Covered** | ON102 |
| NaN/Inf/extreme weight values | **Covered** | WI101, WI102, WI103 |

---

## 6. Findings and Limitations

### What Project 2 Correctly Detects

1. **Path traversal in external_data `location` fields (ON103):** TC8 confirms that ON103 correctly identifies the `../../../../etc/hosts` traversal pattern. The SHA-256 of the live PoC (`54706759...`) matches the bytes embedded in `tests/unit/test_onnx_analyzer.py`, confirming the test suite is regression-testing this exact real-world threat.

### Confirmed Gaps Found

2. **Integer overflow in external_data `offset` / `length` fields (TC11 — FN):** ON103 inspects only the `location` string for path traversal. The `offset` field `0xFFFFFFFFFFFFFFF0` is a statically detectable anomaly (no legitimate model has 18.4 exabyte offsets) but no rule checks it. Gap: add ON104 — Anomalous External Data Offset/Length.

### Expected Limitations (Not Gaps)

3. **Standard-operator semantic backdoors (TC9 — CANNOT_TEST/INCONCLUSIVE):** Project 2 never claimed to detect behavioral backdoors encoded in model weights. The ShadowLogic backdoor is invisible to structural analysis. This is not a false negative — detecting it would require inference-time behavioral testing or dedicated backdoor detection algorithms, which are out of scope for a static security scanner.

4. **Runtime DoS from extreme operator attribute values (TC10 — INCONCLUSIVE):** `size=200,000,001` in LRN is syntactically valid per ONNX spec. The harm manifests only during runtime buffer allocation. Project 2 does not claim to validate operator attribute semantics. Detecting this would require adding heuristic bounds on specific operator attributes — a narrowly-scoped addition that would need careful false-positive analysis.

5. **Dataset repository acquisition (TC11):** The scanner's HF acquisition path does not handle dataset-type repositories (the API path differs: `/api/datasets/` vs `/api/models/`). TC11 required a direct resolve URL. Dataset repositories are not a declared acquisition target.

---

## 7. Capstone Interpretation

These four additional ONNX test cases reveal a precise picture of Project 2's static analysis depth. The scanner correctly detects **structural attacks** — specifically the injection of path-traversal sequences into ONNX external data location fields (ON103, confirmed TP against the live rez0 PoC) — which represents the highest-confidence, lowest-ambiguity attack vector in ONNX security: a malicious string is present in the file and directly readable by the scanner. The confirmed false negative (TC11, integer overflow in `offset`) demonstrates that ON103's coverage of external data is incomplete: it inspects the `location` string but not the numeric `offset`/`length` values, which can encode a uint64 wraparound condition visible entirely in the static protobuf fields. This gap is addressable by a new rule without any runtime execution. In contrast, TC10 (runtime DoS) and TC9 (semantic backdoor) are not false negatives under Project 2's stated scope: they require inference-time behavior — respectively, buffer allocation proportional to an attribute value, and activation-pattern comparison against a trigger — that is fundamentally inaccessible to static analysis. These results confirm that Project 2 occupies a well-defined position in the AI security toolchain: it reliably detects exploitable static artifacts (structural anomalies, file-system attack vectors, dangerous operators, supply-chain indicators) while correctly making no claim to detect semantic or runtime vulnerabilities. The primary actionable finding from this extended evaluation is a single rule addition (ON104) to cover the numeric external-data overflow pattern demonstrated by TC11.

---

## Appendix A — Scan Commands Reference

```bash
# TC8 — ONNX ORT Path Traversal
python -m model_scanner scan "https://huggingface.co/rez0/onnx-ort-path-traversal-poc" \
    --max-size 500 --json --output scan_tc8.json

# TC9 — Shadow CNN Backdoor (GATED — requires HF authentication)
python -m model_scanner scan "https://huggingface.co/ScangGengy/shadow-cnn-onnx-poc" \
    --max-size 500 --json --output scan_tc9.json
# Returns: overall_outcome=ERROR (HTTP 401 GatedRepo)

# TC10 — ONNX LRN SIGSEGV
python -m model_scanner scan "https://huggingface.co/rez0/onnx-lrn-sigsegv-poc" \
    --max-size 500 --json --output scan_tc10.json

# TC11 — OpenVINO uint64 overflow (dataset repo — direct URL required)
python -m model_scanner scan \
    "https://huggingface.co/datasets/0xiviel/poc-openvino-onnx-uint64-overflow/resolve/main/malicious_onnx_overflow.onnx" \
    --max-size 500 --json --output scan_tc11.json
```

## Appendix B — Static Analysis Notes

### TC10 LRN Attribute Detail
```
Node: LRN (domain='', opset 18)
  alpha:  FLOAT = 1e-4     (standard default)
  beta:   FLOAT = 0.75     (standard default)
  bias:   FLOAT = 1.0      (standard default)
  size:   INT   = 200000001  <-- malicious: 200M exceeds any practical tensor dimension
```
The `size` value passes ONNX spec validation (`size > 0`, `size is odd`). No static rule can reject it without a heuristic bound.

### TC11 External Data Overflow Detail
```
Initializer: weights
  data_location: EXTERNAL
  location: 'overflow_data.bin'   <-- benign relative path (ON103 does not fire)
  offset:   18446744073709551600   = 0xFFFFFFFFFFFFFFF0
  length:   16
  overflow: 0xFFFFFFFFFFFFFFF0 + 16 = 0x10000000000000000 ≡ 0 (mod 2^64)
```
The overflow is statically computable from the protobuf fields. ON103 does not inspect `offset`/`length`.

---

## 10. ON104 Remediation / Retest

### Gap Description (v0.2 initial evaluation, commit `93368c0`)

TC11's vulnerability is encoded entirely in numeric fields of the ONNX protobuf:
- `location = 'overflow_data.bin'` — benign relative path (no traversal indicator)
- `offset = 18446744073709551600` (= `0xFFFFFFFFFFFFFFF0`)
- `length = 16`
- `offset + length = 18446744073709551616` = exactly `2^64`

ON103 inspected `external_data[location]` for path traversal sequences. It never parsed `external_data[offset]` or `external_data[length]`. Therefore the uint64 wraparound condition was statically visible but undetected. Result: **FALSE NEGATIVE**.

### Rule Implemented

**ON104 — External Data Offset/Length Overflow**

| Field | Value |
|-------|-------|
| Rule ID | ON104 |
| Severity | HIGH |
| Confidence | HIGH |
| Category | structural_anomaly |
| CWE | CWE-190 (Integer Overflow or Wraparound) |
| OWASP | ML06:2023 AI Supply Chain Attacks |
| MITRE | T1190: Exploit Public-Facing Application |

**Detection condition:** `external_data[offset] + external_data[length] >= 2^64`

The scanner parses offset and length as Python arbitrary-precision integers (no silent overflow in the scanner). Any sum that reaches or exceeds `2^64 = 18446744073709551616` fires ON104. The evidence string includes the parsed offset, length, their sum, and the wrapped value (`sum mod 2^64`).

**False-positive analysis:** The condition `offset + length >= 2^64` has zero false positives against any legitimate model. No real model file has tensor data starting at a byte offset ≥ 18.4 exabytes (2^64 - length). The ONNX spec stores these values as strings in the protobuf; the only way a legitimate value would trigger ON104 is if someone writes an offset > 18.4 exabytes, which exceeds current total global data storage capacity.

**ON103 unchanged:** The existing path-traversal check is untouched. ON103 and ON104 run as separate passes in the same `graph.initializer` loop. Both can fire simultaneously if a single initializer has both a traversal location and an overflowing offset.

### TC11 Retest Result

**Command:**
```
python -m model_scanner scan \
  "https://huggingface.co/datasets/0xiviel/poc-openvino-onnx-uint64-overflow/resolve/main/malicious_onnx_overflow.onnx" \
  --max-size 500 --json --output scan_tc11_retest.json
```

**Scanner result:**

| Field | Value |
|-------|-------|
| overall_outcome | WARNING |
| finding_count | 1 |
| finding rule | ON104 |
| severity | HIGH |
| confidence | HIGH |
| risk_score | 30/100 |
| risk_band | LOW_RISK |
| artifact size | 186 bytes |
| SHA-256 | `3834d6e1...` |

**Finding evidence (verbatim):**
```
initializer='weights' location='overflow_data.bin' offset=18446744073709551600
length=16 offset+length=18446744073709551616 wrapped_sum=0
```

**Finding description (verbatim):**
> ONNX initializer 'weights' has external_data offset (18446744073709551600) + length (16) = 18446744073709551616, which reaches or exceeds 2^64 (18446744073709551616). A runtime performing this addition in uint64 computes wrapped_sum=0, bypassing file-bounds validation and enabling out-of-bounds memory access.

**Classification: TC11 FN → TP**

The same artifact that returned CLEAN (0 findings) at commit `93368c0` now returns WARNING with 1 ON104 finding. The vulnerability is correctly identified without executing the model.

### Tests Added

Six new tests in `tests/unit/test_onnx_analyzer.py` (class `TestON104ExternalDataOverflow`):

| Test | Parameters | Expected | Result |
|------|-----------|---------|--------|
| TC11 parameters | offset=0xFFFFFFFFFFFFFFF0, length=16, sum=2^64 | ON104 fires | PASS |
| Strict overflow | offset=2^64-16, length=32, sum=2^64+16 | ON104 fires | PASS |
| Sub-boundary | offset=2^64-16, length=15, sum=2^64-1 | ON104 does NOT fire | PASS |
| Normal data | offset=0, length=1024 | ON104 does NOT fire | PASS |
| Large valid offset | offset=100GB, length=4096 | ON104 does NOT fire | PASS |
| ON103 regression | location='../../../../etc/hosts', offset=0, length=4 | ON103 fires, ON104 does NOT | PASS |

The old `93368c0` implementation would fail the TC11 parameters test (no ON104 rule existed). All 6 tests pass after ON104 implementation.

### Full Test Suite After ON104

- `tests/unit/test_onnx_analyzer.py`: **20 passed** (14 existing + 6 new ON104 tests)
- Full suite: **1207 passed, 4 skipped, 0 failed**
- No regression in ON101, ON102, ON103, WI101, WI102, WI103, KR103, TF103, or any other rule.
