# Project 2 — TC1: Extensionless Artifact Acquisition Gap

**Scanner version:** 2.0.0  
**Fix commit:** 93368c0  
**Live retest commit:** a38bbbb  
**Evaluation date:** 2026-09-02  
**Methodology:** Static acquisition and analysis only — no deserialization, no code execution.

---

## 1. Original Acquisition Gap

### TC1 Artifact

- **Repository:** `https://huggingface.co/shaq4prez/malicious-olmo3-poc`
- **Filename:** `.metadata` (hidden, extensionless)
- **Payload:** Raw Pickle protocol 5 stream (`\x80\x05` magic), weaponized with `posix.system`

### Why Acquisition Failed

The HuggingFace acquisition layer filtered repository files by extension against `_HF_MODEL_EXTENSIONS`. The file `.metadata` has no recognized extension, so `_filter_model_candidates()` excluded it. The Pickle analyzer never received the artifact.

The file was not in `_HF_FORBIDDEN_EXTENSIONS` (it's not `.py`/`.sh`/etc.), not a known non-model file (it's not `README`/`LICENSE`/etc.), and contained a complete malicious Pickle payload — but the extension-based filter blocked acquisition before any content inspection could occur.

**Result (before fix):** 0 artifacts acquired, 0 findings, CLEAN.

---

## 2. Root Cause

Extension-only filtering cannot distinguish:
- A harmless config file with no extension
- A malicious Pickle payload hidden in an extensionless filename

The fix requires a two-stage approach: (1) identify extensionless filenames that are plausible model/payload candidates, then (2) peek the first bytes to confirm Pickle magic before committing to a full download.

---

## 3. Acquisition Design Change

### `_is_suspicious_extensionless(rfilename: str) -> bool`

A classifier that returns `True` for repository filenames that:
- Have **no recognized extension at all** (suffix == "")
- Are NOT under a `.git` path (`.git/`, `.github/`, etc.)
- Are NOT a known non-model bare name (`README`, `LICENSE`, `Makefile`, etc.)
- Do NOT have a recognized non-model extension (`.json`, `.txt`, `.md`, ...)
- Do NOT have a known model extension (already handled by the primary pass)

For hidden files (leading dot), the function strips the leading dot and checks for a secondary extension. `.metadata` → stem "metadata", no second extension → qualifies. `.hidden.json` → has `.json` extension → excluded.

### Secondary Pass in `_acquire_repository`

After the primary pass downloads known-extension candidates, a secondary pass:
1. Identifies all files in the repo that pass `_is_suspicious_extensionless()`.
2. For each, checks the API-declared size against the per-file limit (skip, do not raise, if oversized).
3. Fetches only the **first 16 bytes** via an HTTP `Range: bytes=0-15` request.
4. Passes those bytes to `is_pickle_magic()` (checks `\x80\x02`–`\x80\x05` and legacy `(` opcode).
5. Only if magic matches: downloads the full file within the size limit and routes it through the normal artifact pipeline.

This design ensures:
- **No arbitrary binary downloads.** A file without Pickle magic is discarded after 16 bytes.
- **No deserialization.** The magic check reads bytes; no Pickle parsing occurs at this stage.
- **Size limit preserved.** The per-file byte limit applies to both peek and full download.
- **No SSRF expansion.** Only files already in the HF repo listing are targeted; no user-supplied URLs.

---

## 4. Safety Constraints

| Constraint | Status |
|------------|--------|
| Size limit preserved (500 MB max) | Enforced — oversized files skipped at peek stage |
| No deserialization / `pickle.loads()` | Not performed — magic bytes read only |
| No arbitrary extensionless downloads | Enforced — Pickle magic required before full fetch |
| No code execution or model loading | Not performed |
| No SSRF / new external URLs | Not introduced — only files in HF repo listing |
| Strict mode behavior unchanged | Confirmed — `strict_mode=True` still acquires Pickle files |
| Primary-pass model extensions unaffected | Confirmed — secondary pass is additive |
| `.git`/`.github` paths excluded | Enforced by filename check |
| Known non-model names excluded | Enforced — README, LICENSE, etc. never peeked |

---

## 5. Live Retest Result

**Command:**
```
python -m model_scanner scan "https://huggingface.co/shaq4prez/malicious-olmo3-poc" \
  --max-size 500 --json --output scan_tc1_after.json
```

**Result:**

| Field | Value |
|-------|-------|
| Artifact acquired | `.metadata` |
| Source URL | `.../resolve/main/.metadata` |
| SHA-256 | `4de9561840e43aa6d1e25c01c9ad78290caf36227d58e0a1aa76f624420c3334` |
| Size | 234 bytes |
| Detected format | `RAW_PICKLE` |
| Risk band | CRITICAL_RISK |
| Risk score | 100 |

**Findings:**

| Rule ID | Severity | Confidence | Evidence |
|---------|----------|------------|----------|
| P101 — Dangerous Global Import | CRITICAL | HIGH | `posix.system` (STACK_GLOBAL at byte 28) |
| P103 — STACK_GLOBAL Dynamic Import | HIGH | HIGH | `posix.system` resolved via STACK_GLOBAL |
| P104 — Dangerous REDUCE Invocation | CRITICAL | HIGH | `posix.system` called with shell command |
| C004 — Imported and Invoked Dangerous Callable | CRITICAL | HIGH | P101 + P104 correlated |

**P104 context snippet:**
```
Args: (echo "VULNERABILITY EXPLOITED - Arbitrary code executed via pickle
deserialization" > /tmp/huntr_poc_exploited.txt && ...)
```

No model code was executed. The finding is based purely on static Pickle opcode analysis.

---

## 6. Before vs After

| Metric | Before (292cb70) | After (93368c0 → current) |
|--------|-----------------|--------------------------|
| Artifacts acquired | 0 | 1 (`.metadata`) |
| Findings | 0 | 4 (P101, P103, P104, C004) |
| Risk band | CLEAN | CRITICAL_RISK |
| Acquisition method | Extension filter only | Extension filter + secondary extensionless pass |
| Pickle magic check | N/A | 16-byte peek before full download |

---

## 7. Test Coverage

All tests in `tests/unit/test_acquisition.py`, class `TestSuspiciousExtensionlessAcquisition`.

| Test | Covers |
|------|--------|
| `test_is_suspicious_helper_classification` | `.metadata`, `checkpoint`, `model` included; config/git/docs excluded |
| `test_extensionless_pickle_candidate_included` | Pickle magic → full download occurs |
| `test_extensionless_harmless_excluded` | Non-Pickle magic → download skipped |
| `test_readme_not_included_as_hidden_candidate` | README/LICENSE never peeked |
| `test_oversized_extensionless_excluded` | Size gate before peek |
| `test_is_suspicious_edge_cases` | `.git/config`, `.github/workflows/evil`, nested paths, `.hidden.json` |
| `test_tc1_metadata_acquisition_chain` | Full mock chain: `.metadata` + safetensors in same repo |
| `test_non_pickle_binary_extensionless_not_downloaded` | Non-Pickle binary → acquire not called |
| `test_strict_mode_extensionless_still_acquired` | `strict_mode=True` does not suppress secondary pass |

---

## 8. Known Limitations

1. **Only Pickle magic is checked.** The secondary pass gates on Pickle magic bytes (`\x80\x02`–`\x80\x05` or `(`). A malicious payload in another format (e.g., GGUF, ONNX) hidden in an extensionless file would not be acquired by this path.

2. **HuggingFace repository acquisition only.** The extensionless secondary pass applies to HF repository scans (`HuggingFaceAcquisition`). Local directory scans (`LocalDirectoryAcquisition`) and direct URL scans are unaffected — they already accept any file the user explicitly provides.

3. **No recursive subdirectory descent.** The secondary pass only checks filenames listed in the HF API `siblings` response (the repo root listing). Nested subdirectory files not surfaced in the API response are not discovered.

4. **Size declared in API metadata.** The size gate uses the `size` field from the HF API response. If the CDN returns a different size than declared, the peek step's fallback 64 KB cap limits exposure.
