# Capstone Changelog

## Phase 7 — Coverage Quality Gate — 2026-08-21

Raised overall test coverage from 89% to 90.7% and activated the `fail_under = 90`
quality gate in `pyproject.toml`.

### Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` | Set `fail_under = 90` (was `0`) |
| `tests/unit/test_scan_orchestrator.py` | +9 tests for engine.py uncovered paths |
| `tests/unit/test_pickle_disassembler.py` | +7 tests for disassembler opcode paths |
| `tests/unit/test_pytorch_zip.py` | +3 tests for zip_reader.py resource limits |

### Coverage Paths Covered

**engine.py** (was 79.3%):
- Lines 228-229: in-memory bytes target label
- Line 231: in-memory stream target label
- Lines 237/239/262-263: acquisition ScannerException → early ERROR result
- Lines 297-315: CrossArtifactCorrelator findings injection into artifact results
- Lines 380-390: FormatDispatcher exception → FORMAT_DETECTION_FAILED error
- Lines 462-486: no registered analyzer → NO_ANALYZER warning + UNSUPPORTED outcome
- Lines 491-502: analyzer.analyze() unhandled exception → ANALYZER_FAILED fail-closed
- Line 593: _scan_repository() returns None for non-directory path
- Lines 610-628: RepositoryAnalyzer exception → REPO_ANALYZER_ERROR result
- Lines 698-700: _acquire_artifacts NotImplementedError for IN_MEMORY source type

**disassembler.py** (was 82.8%):
- Lines 326-332: LIST opcode with items between MARK markers
- Lines 338-342: DICT opcode with pairs between MARK markers
- Lines 348-349: POP opcode discards item from non-empty stack
- Lines 352-355: POP_MARK opcode (`b'1'`=0x31) clears to MARK boundary
- Line 269: MEMOIZE opcode memo overflow (when memo table is full)
- Line 282: GET/BINGET opcode on populated memo slot (cache hit path)
- Lines 414-415: _push_stack shift when max_stack_depth exceeded

**zip_reader.py** (was 75.2%):
- Line 108: list_members raises ResourceLimitError when member count > max_members
- Line 120: list_members skips directory entries (`info.is_dir()`)
- Lines 179-187: extract_member raises ResourceLimitError when member size > max_member_bytes

### Quality Gate Results

- Tests: 971 collected, 967 passed, 4 skipped (+19 new tests vs previous 952)
- Ruff: All checks passed
- Mypy: No issues found in 64 source files
- Coverage: **90.7%** (threshold: 90%)

---

## Project 1 Capstone Gap Completion — 2026-08-21

Addressed all remaining capstone gaps identified in the final completion pass.

### Summary of Changes

**Phase 1 — Correlation Output Pipeline**
- Fixed `normalizer.py::normalize_finding` to pass `correlated_finding_ids` through to the
  normalized Finding. Previously this field was silently dropped, breaking the pipeline for
  C001/C002/C003 correlated findings.
- Fixed `json_export.py::_serialize_finding` to include `correlated_finding_ids` as a list in
  the canonical JSON output. Previously the field was absent from all JSON reports.
- Added tests in `test_normalization.py` proving that correlated IDs survive normalization,
  that non-correlated findings emit an empty tuple, and that the batch pipeline preserves them.
- Added tests in `test_json_export.py` proving correlated and non-correlated findings serialize
  the `correlated_finding_ids` field correctly.

**Phase 2 — Rule Catalog Documentation**
- `scripts/generate_rule_catalog.py` already existed; ran it to regenerate `docs/rule_catalog.md`
  with the current 70-rule catalog (was previously stale after adding C001–C003).
- Added `test_rule_catalog_doc_matches_generator` to `test_rule_catalog_integrity.py` to catch
  future drift between the live catalog and the committed Markdown document.

**Phase 3 — Dashboard Completion**
- Updated `index.html`: replaced hardcoded "44 Rules" strings with `<span>` elements that are
  populated dynamically from the `/api/rules` response.
- Added `Input Source` field to the Artifact Forensics panel (shows LOCAL_FILE / PUBLIC_URL etc).
- Added `Correlated Finding IDs` section to the finding detail drawer (hidden for non-correlated
  findings, populated for C001/C002/C003 findings).
- Updated `app.js` to wire up the new input_source and correlated_finding_ids display elements
  and to update the rule count spans dynamically from the API response total.

### Files Changed

| File | Change |
|------|--------|
| `src/model_scanner/core/normalizer.py` | Preserve `correlated_finding_ids` in `normalize_finding` |
| `src/model_scanner/presentation/json_export.py` | Serialize `correlated_finding_ids` in `_serialize_finding` |
| `src/model_scanner/presentation/web/static/index.html` | Update "44 Rules" → dynamic span; add input_source field; add correlated IDs section |
| `src/model_scanner/presentation/web/static/js/app.js` | Wire up input_source, correlated_finding_ids, dynamic rule count |
| `docs/rule_catalog.md` | Regenerated from 70-rule catalog |
| `tests/unit/test_normalization.py` | +3 tests for correlated_finding_ids normalization |
| `tests/unit/test_json_export.py` | +2 tests for correlated_finding_ids JSON serialization |
| `tests/unit/test_rule_catalog_integrity.py` | +1 test for catalog doc freshness |

### Quality Gate Results

- Tests: 948 passed, 4 skipped (baseline was 942 passed, 4 skipped; +6 new tests)
- Ruff: All checks passed
- Mypy: No issues found in 64 source files
- Coverage: 89%

---

## P0 Quality Gate Restoration — 2026-08-21

Restored all quality gates (pytest / ruff / mypy) to green after adding
C001–C003 correlated cross-signal rules.

### Files Changed

| File | Change |
|------|--------|
| `src/model_scanner/core/correlation.py` | Fix 1: Add `ModelFormat` import; fix import ordering (I001); remove trailing/blank-line whitespace (W291/W293); fix long line (E501); remove unnecessary UTF-8 encoding arg (UP012) |
| `src/model_scanner/core/normalizer.py` | Fix 2: Extend `_RULE_ID_REGEX` to accept `C0xx` prefix |
| `src/model_scanner/core/engine.py` | Fix ruff I001: reorder local imports inside `scan()` |
| `src/model_scanner/rules/catalog.py` | Fix 3: Change C003 category from `"obfuscation"` → `"structural_anomaly"`; update docstring rule count 67 → 70 |
| `src/model_scanner/rules/models.py` | Fix ruff W605: make SecurityRule docstring a raw string |
| `tests/unit/test_rules.py` | Fix 4: Update `_RULE_ID_PATTERN` to include `C0`; count 67 → 70; last-rule assertion SC204 → C003; add C001/C002/C003 to expected ID set |
| `tests/unit/test_rule_catalog_integrity.py` | Fix 4: Update `_RULE_ID_RE` to include `C0` |
| `tests/unit/test_web_server.py` | Fix 4: Update rule count assertion 67 → 70 |
| `tests/unit/test_normalization.py` | Fix 2: Add `test_valid_rule_ids_accepted` proving C001/C002/C003 valid and invalid IDs rejected |
| `tests/unit/test_correlation.py` | Fix ruff I001/W293: sort imports, remove blank-line whitespace |

### Root Causes

1. **FR-COR-001 / Missing import** — `correlation.py` referenced `ModelFormat` without importing it.
2. **FR-VAL-001 / Regex gap** — `normalizer._RULE_ID_REGEX` did not include the `C0` prefix, so C-series findings would fail `validate_rule_id`.
3. **SC-CAT-001 / Taxonomy violation** — C003 used category `"obfuscation"` which does not exist in `CANONICAL_CATEGORIES`. The closest canonical category is `"structural_anomaly"` (the same category used by the primary C003 trigger rule P210).
4. **NFR-TEST-001 / Stale count assertions** — Six test assertions still expected 67 rules and excluded C001–C003 from the expected ID set.

### Fixes

1. Added `ModelFormat` to `correlation.py` imports; reformatted import block to multi-line sorted form per ruff isort rules; eliminated all trailing whitespace.
2. Extended `_RULE_ID_REGEX` in `normalizer.py` from `(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2)` to `(P1|P2|ST1|ST2|GU1|GU2|G0|SC1|SC2|C0)`. Added explicit `test_valid_rule_ids_accepted` to prove the seven canonical prefixes are all accepted and invalid IDs rejected.
3. Changed C003 `category` from `"obfuscation"` to `"structural_anomaly"`. Rationale: the project's canonical taxonomy has no obfuscation category; the structural anomaly (nested archive P210) is the unique detection signal for C003.
4. Updated stale expectations in test_rules.py, test_rule_catalog_integrity.py, and test_web_server.py: count 67 → 70, pattern includes C0, last-rule SC204 → C003, expected ID set includes C001/C002/C003.
