# Project 2 — TensorFlow / SavedModel Security Coverage

**Scanner version:** 2.0.0  
**Evaluation date:** 2026-09-02  
**Methodology:** Static acquisition and analysis only — no TensorFlow import, no SavedModel loading, no graph execution, no model inference, no `eval()`/`exec()`.

---

## 1. Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| TensorFlow SavedModel graph | `saved_model.pb` | Protobuf wire-format parsing (no TF runtime) |
| SavedModel assets directory | `assets/` | Filename-only inspection; asset files never opened |
| `.pb` via HuggingFace | `saved_model.pb` | Restricted to the known SavedModel filename only |

---

## 2. Acquisition Support

### `.pb` on HuggingFace

`.pb` is included in the HuggingFace model extension filter (`_HF_MODEL_EXTENSIONS`). To prevent scanning arbitrary protobuf files (config protos, etc.) that are not TensorFlow model graphs, the filter restricts `.pb` downloads to files named exactly `saved_model.pb` (`_HF_SAVEDMODEL_PB_NAMES`). Any other `.pb` filename in a HuggingFace repo is skipped.

This means the scanner can acquire `saved_model.pb` directly from HuggingFace repository listings, eliminating the previous need for manual download.

**Limitation:** Stream-based artifacts acquired without a `local_path` cannot have their sibling `assets/` directory inspected for path-traversal filenames. TF103 fires only when `artifact.local_path` is set.

---

## 3. Rules

### TF101 — External or Custom Op in TensorFlow Graph

**Severity:** MEDIUM  **Confidence:** MEDIUM  **CWE:** CWE-94 (Code Injection)

Detects op names in the SavedModel protobuf that are not in the known-standard op set. Uses a two-tier detection model:

**Tier 1 (always report, regardless of other filters):**
`ReadFile`, `WriteFile`, `PyFunc`, `PyFuncStateless`, `EagerPyFunc`, `ShellExecute`, `SubprocessOp`, `MergeV2Checkpoints`, `SaveV2`, `RestoreV2`, `DeleteSessionTensor`, `GetSessionTensor`, `SetSessionTensor`

These ops perform filesystem I/O or execute Python code and are always reported for any untrusted model.

**Tier 2 (unknown op heuristic — uppercase-first CamelCase requirement):**
A token that is not in the standard-op allowlist and not in the protobuf-field blocklist is reported if it:
1. Starts with an uppercase ASCII letter (genuine TF op names are CamelCase), AND
2. Is not a version string, not purely numeric, and at least 4 characters.

Tokens starting with lowercase but having an internal uppercase letter are excluded from Tier 2 — these are wire-format fragments (e.g. `"ableOp"` sliced from `"ReadVariableOp"`) not genuine op names.

Lowercase-start tokens that contain a clearly suspicious substring (`cmd`, `shell`, `exec`, `eval`, `payload`, `evil`, `passwd`, `backdoor`) or the `____<path-token>` obfuscation pattern are still reported as suspicious.

---

### TF102 — Malformed TensorFlow SavedModel

**Severity:** MEDIUM  **Confidence:** HIGH

Fires when the `saved_model.pb` cannot be parsed as a valid protobuf (wire-format error, empty file, truncation). Indicates potential tampering or corrupt acquisition.

---

### TF103 — Path Traversal Indicator in SavedModel Asset

**Severity:** CRITICAL  **Confidence:** HIGH  **CWE:** CWE-22 (Path Traversal)

Inspects filenames in the sibling `assets/` directory for path-traversal indicators. Detection is based purely on the filename string — asset files are never opened, read, or resolved.

Detected patterns:
- `..` path components (POSIX and Windows separators)
- Absolute POSIX paths (leading `/`)
- Absolute Windows paths (leading `\` or `C:\`)
- URL-encoded traversal tokens: `%2e%2e`, `%2f`, `%5c`, `..%2f`, `..%5c`

---

## 4. TC4 Before/After Results

### TC4 Adversarial Artifacts

Four malicious SavedModel payloads were tested, acquired from a private HuggingFace repository. Each payload embeds a standard TF checkpoint graph (SaveV2, RestoreV2, MergeV2Checkpoints) with additional malicious content specific to the payload.

### Before (commit 292cb70 — original baseline)

| Payload | Findings | Risk Band |
|---------|----------|-----------|
| Payload 1 (asset traversal) | 98 TF101 | CRITICAL_RISK |
| Payload 2 (ReadFile) | 99 TF101 | CRITICAL_RISK |
| Payload 3 (WriteFile) | 98 TF101 | CRITICAL_RISK |
| Payload 4 (evil variable names) | 105 TF101 | CRITICAL_RISK |

The vast majority of these 98-105 findings were false positives: version strings (`.0`, `2.21.0`), protobuf field names (`type`, `dtype`, `shape`, `bool`, `config`), standard ops (`Identity_1`, `Const_2`), and metadata values (`CPU`, `GPU`, `kEagerRuntime`).

### After first improvement (commit 93368c0 — `_PROTO_FIELD_BLOCKLIST` + `_STANDARD_TF_OPS` added)

| Payload | Findings | Risk Band | True Positives |
|---------|----------|-----------|----------------|
| Payload 1 | 12 TF101 | HIGH_RISK | 3 genuine |
| Payload 2 | 15 TF101 | CRITICAL_RISK | 4 genuine (incl. ReadFile) |
| Payload 3 | 14 TF101 | HIGH_RISK | 5 genuine (incl. WriteFile + cmd) |
| Payload 4 | 16 TF101 | CRITICAL_RISK | 6 genuine (incl. evil var names) |

Remaining false positives: `DisableCopyOnRead`, `FullArgSpec`, `Select`, `ShardedFilename`, `StaticRegexFullMatch`, `Tout`, `Tdim` (protobuf attribute keys); and wire-format fragments `ableOp`, `adVariableOp`, `storeV2`, `adFile`, `itionedCall` (mid-string slices that start lowercase but have internal uppercase).

### After Phase 4 improvement — LIVE RETEST (commit 1d0fa5d, scanned 2026-09-02)

All four original artifacts were re-acquired directly from HuggingFace using the current scanner at commit `1d0fa5d`. SHA-256 hashes were verified against the original baseline scan to confirm bit-for-bit artifact identity.

| Payload | SHA-256 (first 16) | Size | Findings | Risk Band | Score |
|---------|--------------------|------|----------|-----------|-------|
| P1 `payload1_asset_traversal` | `145c760e2c4fe09d` | 10325 B | 3 TF101 | LOW_RISK | 18 |
| P2 `payload2_readfile_op` | `bab30ab708cfd838` | 9895 B | 4 TF101 | LOW_RISK | 24 |
| P3 `payload3_pyfunction` | `c32065e4994d7ac7` | 10141 B | 5 TF101 | LOW_RISK | 30 |
| P4 `payload4_var_traversal` | `4fbf7d5edb87eb34` | 11255 B | 6 TF101 | MEDIUM_RISK | 36 |

Live scan JSON files: `scan_tc4_p1_after.json`, `scan_tc4_p2_after.json`, `scan_tc4_p3_after.json`, `scan_tc4_p4_after.json`.

#### Per-payload findings with classification

**Payload 1 — `payload1_asset_traversal`**

| Finding | Signal classification |
|---------|-----------------------|
| `op='MergeV2Checkpoints'` | **Potentially legitimate TF checkpoint op** — present in any model with checkpoint saving |
| `op='RestoreV2'` | **Potentially legitimate TF checkpoint op** |
| `op='SaveV2'` | **Potentially legitimate TF checkpoint op** |
| TF103 | **NOT FIRED** — artifact acquired as stream; no `local_path` → `assets/` directory not accessible |

The asset traversal attack in this payload manifests via the SavedModel `assets/` directory. When the `.pb` is downloaded as a standalone stream-only artifact (no accompanying directory), TF103 cannot fire. This is the stream-only coverage gap documented in the Known Limitations section.

**Payload 2 — `payload2_readfile_op`**

| Finding | Signal classification |
|---------|-----------------------|
| `op='ReadFile'` | **Confirmed malicious indicator** — TF op that reads arbitrary filesystem paths |
| `op='MergeV2Checkpoints'` | **Potentially legitimate TF checkpoint op** |
| `op='RestoreV2'` | **Potentially legitimate TF checkpoint op** |
| `op='SaveV2'` | **Potentially legitimate TF checkpoint op** |

`ReadFile` is unambiguously detected. This is a Tier 1 always-report op.

**Payload 3 — `payload3_pyfunction`**

| Finding | Signal classification |
|---------|-----------------------|
| `op='WriteFile'` | **Confirmed malicious indicator** — TF op that writes arbitrary filesystem paths |
| `op='serving_default_cmd'` | **Confirmed malicious indicator** — Tier 2 lowercase pattern with "cmd" substring |
| `op='MergeV2Checkpoints'` | **Potentially legitimate TF checkpoint op** |
| `op='RestoreV2'` | **Potentially legitimate TF checkpoint op** |
| `op='SaveV2'` | **Potentially legitimate TF checkpoint op** |

**Note:** The payload directory is named `payload3_pyfunction` but the actual graph contains `WriteFile` rather than `PyFunc`/`EagerPyFunc`. No PyFunc op was detected — consistent with the original baseline scan which also showed WriteFile in this slot. The directory name is misleading with respect to the attack vector; `WriteFile` + `serving_default_cmd` are the actual indicators.

**Payload 4 — `payload4_var_traversal`**

| Finding | Signal classification |
|---------|-----------------------|
| `op='ad_disablecopyonread__________tmp_evil_weight'` | **Confirmed malicious indicator** — `____` obfuscation + `/tmp` path component |
| `op='assignvariableop__________tmp_evil_weight'` | **Confirmed malicious indicator** — same pattern |
| `op='read_disablecopyonread__________tmp_evil_weight'` | **Confirmed malicious indicator** — same pattern |
| `op='MergeV2Checkpoints'` | **Potentially legitimate TF checkpoint op** |
| `op='RestoreV2'` | **Potentially legitimate TF checkpoint op** |
| `op='SaveV2'` | **Potentially legitimate TF checkpoint op** |

### False-Positive Precision Improvement Summary

| Stage | Avg. findings | Confirmed malicious signals | Ambiguous (checkpoint ops) | Clear false positives |
|-------|--------------|----------------------------|----------------------------|-----------------------|
| 292cb70 (baseline) | ~100 | 0–3 | 3 | ~95–97 |
| 93368c0 (intermediate) | ~14 | 0–3 | 3 | ~9–11 |
| 1d0fa5d (live retest) | ~4.5 | 0–3 | 3 | 0 |

### Risk Band Observations (actual vs previously documented)

The live results differ from what was projected in the initial 1d0fa5d documentation:

| Payload | Projected risk band | Actual risk band |
|---------|---------------------|-----------------|
| P1 (checkpoint ops only) | HIGH_RISK | LOW_RISK |
| P2 (+ ReadFile) | CRITICAL_RISK | LOW_RISK |
| P3 (+ WriteFile + cmd) | HIGH_RISK | LOW_RISK |
| P4 (+ evil variable names) | CRITICAL_RISK | MEDIUM_RISK |

**Root cause:** TF101 is fixed at MEDIUM severity for all findings (design constraint from project specification: "TF101 must stay at MEDIUM severity — they are potentially suspicious, not automatically malicious"). With 3–6 MEDIUM-severity findings, the cumulative risk score (6–8 points × finding count × 1.5 obfuscation multiplier) stays below the HIGH_RISK threshold.

**This is a known design limitation, not an implementation defect.** The confirmed malicious indicators ARE detected; they are accurately labelled MEDIUM-severity because static graph analysis cannot confirm that the op will actually execute at inference time. The LOW_RISK band is a consequence of the design constraint.

**Implication for analysts:** A LOW_RISK or MEDIUM_RISK TF101 result containing `ReadFile`, `WriteFile`, `PyFunc`, or `____tmp____` variable patterns should be escalated for manual review regardless of the overall risk band. The findings table, not the summary band, is the primary signal for TensorFlow SavedModel artifacts.

---

## 5. False-Positive Analysis

### Root causes of prior false positives

| Category | Examples | Fix |
|----------|----------|-----|
| Version strings | `2.21.0`, `.0`, `21.0` | `_VERSION_STRING_RE` regex rejection |
| Protobuf field names | `type`, `dtype`, `shape`, `config`, `value` | `_PROTO_FIELD_BLOCKLIST` |
| Standard TF ops | `Identity`, `Const`, `MatMul` | `_STANDARD_TF_OPS` allowlist |
| Instance-suffixed ops | `Identity_1`, `Const_2` | `_INSTANCE_SUFFIX_RE` strip + allowlist |
| Protobuf attribute keys | `Tout`, `Tdim`, `FullArgSpec` | Added to `_PROTO_FIELD_BLOCKLIST` |
| Standard ops (missing) | `Select`, `StaticRegexFullMatch`, `DisableCopyOnRead`, `ShardedFilename` | Added to `_STANDARD_TF_OPS` |
| Wire-format fragments | `ableOp`, `adVariableOp`, `storeV2`, `adFile`, `itionedCall` | Require `op[0].isupper()` for Tier 2 |

### Wire-format fragment detection

The raw protobuf wire-format parser extracts string fields that match the alphanumeric filter. When a multi-segment protobuf message is parsed recursively, it may extract substrings of longer strings at sub-field boundaries. These appear as mid-word fragments with a lowercase start letter and internal uppercase (e.g. `"ableOp"` from the tail of `"ReadVariableOp"`).

The fix: Tier 2 detection now requires `op[0].isupper()`. Legitimate TF op names are always CamelCase-starting-uppercase. Fragments that start lowercase but have internal uppercase are no longer reported unless they additionally match a Tier 1 dangerous-substring pattern.

---

## 6. Coverage Matrix

| Attack / Security Area | Format | Status | Rule | Evidence |
|------------------------|--------|--------|------|----------|
| Filesystem read op (ReadFile) | SavedModel | **Covered** | TF101 Tier 1 | Regression test |
| Filesystem write op (WriteFile) | SavedModel | **Covered** | TF101 Tier 1 | Regression test |
| Python execution (PyFunc) | SavedModel | **Covered** | TF101 Tier 1 | `_DANGEROUS_TF_OPS` |
| Checkpoint ops (SaveV2, RestoreV2) | SavedModel | **Covered** | TF101 Tier 1 | Regression test |
| Suspicious custom op (CamelCase, unknown) | SavedModel | **Covered** | TF101 Tier 2 | Regression test |
| Suspicious variable name patterns | SavedModel | **Covered** | TF101 Tier 2 | Regression test |
| SavedModel asset path traversal (filesystem) | SavedModel | **Covered** | TF103 | Regression test |
| SavedModel asset path traversal (URL-encoded) | SavedModel | **Covered** | TF103 | Regression test |
| Malformed / truncated protobuf | SavedModel | **Covered** | TF102 | Regression test |
| `.pb` acquisition via HuggingFace | `.pb` | **Covered** | Acquisition | `_HF_SAVEDMODEL_PB_NAMES` |
| Normal TF graph (no false positives) | SavedModel | **Covered** | TF101 | Negative regression tests |
| Wire-format fragment suppression | SavedModel | **Covered** | TF101 | Negative regression tests |
| Asset traversal in protobuf bytes (no local_path) | SavedModel stream | **Partial** | TF103 | Filesystem check only |
| Semantic/behavioral backdoor in weights | SavedModel | **Out of Scope** | — | No execution |
| Runtime code execution | SavedModel | **Out of Scope** | — | No execution |
| Model inference | SavedModel | **Out of Scope** | — | No inference |
| Runtime memory DoS (attribute bounds) | SavedModel | **Out of Scope** | — | No execution |
| Obfuscated op names (Unicode lookalikes) | SavedModel | **Untested** | — | Not implemented |

---

## 7. Test Coverage

All tests are in `tests/unit/test_tensorflow_analyzer.py`. No TensorFlow is imported; all fixtures are hand-built protobuf byte strings.

| Test | Covers |
|------|--------|
| `test_readfile_op_detected` | TF101 Tier 1 — ReadFile |
| `test_writefile_op_detected` | TF101 Tier 1 — WriteFile |
| `test_version_strings_not_flagged` | FP: version string rejection |
| `test_proto_field_names_not_flagged` | FP: protobuf field blocklist |
| `test_standard_ops_not_flagged` | FP: standard op allowlist |
| `test_false_positive_count_drops` | FP: combined noise + one genuine signal ≤ 5 findings |
| `test_empty_pb_handled` | TF102: empty file |
| `test_tiny_pb_handled` | TF102: truncated file |
| `test_garbage_pb_does_not_crash` | Robustness |
| `test_payload1_dangerous_checkpoint_ops_detected` | TC4 regression: Tier 1 checkpoint ops |
| `test_payload3_cmd_context_detected` | TC4 regression: WriteFile + cmd pattern |
| `test_wire_fragment_*_not_flagged` (5 tests) | FP: lowercase-start wire fragments |
| `test_tout_not_flagged` | FP: protobuf attribute key Tout |
| `test_tdim_not_flagged` | FP: protobuf attribute key Tdim |
| `test_fullargspec_not_flagged` | FP: Python metadata FullArgSpec |
| `test_select_not_flagged` | FP: Select now standard op |
| `test_selectv2_not_flagged` | FP: SelectV2 now standard op |
| `test_staticregexfullmatch_not_flagged` | FP: StaticRegexFullMatch now standard op |
| `test_disablecopyonread_not_flagged` | FP: DisableCopyOnRead now standard op |
| `test_shardedfilename_not_flagged` | FP: ShardedFilename now standard op |
| `test_tc4_*_still_fires` (6 tests) | TC4 genuine signals preserved |
| `test_tc4_precision_after_fix` | Combined precision test |
| `test_no_assets_dir_no_finding` | TF103: no false positive without assets dir |
| `test_benign_asset_no_finding` | TF103: normal asset filenames |
| `test_url_encoded_traversal_asset_flagged` | TF103 Test 5: URL-encoded traversal |
| `test_dotdot_asset_flagged` | TF103 Test 5: dotdot component |
| `test_windows_backslash_traversal_asset_flagged` | TF103 Test 6: backslash traversal |
| `test_absolute_path_asset_flagged` | TF103 Test 7: absolute path |
| `test_normal_asset_no_tf103_finding` | TF103 Test 8: negative case |
| `test_traversal_indicator_helper` | `_traversal_indicator` unit test |
| `test_traversal_indicator_windows_drive` | Windows drive absolute path |
| `test_traversal_indicator_leading_backslash` | Leading backslash |

Total: **42 tests**, 0 failures.

Full suite baseline after this commit: **1239 passed, 4 skipped, 0 failed**.

---

## 8. Known Limitations

1. **Stream-only artifacts:** TF103 (asset path traversal) inspects the filesystem `assets/` directory. For artifacts acquired via stream (no `local_path`), this check is skipped. Asset traversal paths embedded in the `saved_model.pb` protobuf bytes themselves are not currently extracted and checked. This is a known partial-coverage gap.

2. **Raw wire-format parsing:** `_extract_ops_from_pb` is a best-effort raw protobuf parser, not a full proto schema decoder. It may miss op names in deeply nested message structures beyond the recursion depth limit (8 levels), or in message structures that differ significantly from the standard SavedModel layout.

3. **Tier 2 uppercase-first heuristic:** An attacker who knows the Tier 2 heuristic could craft a CamelCase-starting custom op name that happens not to be in any blocklist. No static heuristic can enumerate all possible attacker-controlled op names. The Tier 1 always-report list covers the highest-risk ops.

4. **Behavioral backdoors:** Weight-space backdoors using only standard TF ops and numeric weights (e.g. ShadowLogic-style) are undetectable by static structural analysis. This is a fundamental scope boundary.

5. **`MergeV2Checkpoints`/`SaveV2`/`RestoreV2`:** These ops appear in legitimate training graphs exported as SavedModels and will produce TF101 findings on non-adversarial checkpoints. They are kept in the Tier 1 always-report list because they represent genuine filesystem-access capability that is unusual in a pure inference SavedModel.
