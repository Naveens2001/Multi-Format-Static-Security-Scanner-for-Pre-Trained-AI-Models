# Risk Score Formula

This document describes the deterministic composite risk scoring algorithm
used by the static AI model security scanner (`model_scanner.core.scoring`).

_Last verified against `src/model_scanner/core/scoring.py` on 2026-09-07
(post the 2026-09-06 scoring/catalog revision — R5 severity floor, obfuscation
multiplier fix)._

---

## Overview

The scanner produces a **final risk score in [0, 100]** for each artifact,
derived from the security findings detected during static analysis.

The scoring is **purely mathematical and deterministic**: given identical
findings and format, the score is always the same.

---

## Formula

### Step 1 — Deduplicate Findings

Duplicate findings (same rule ID + byte offset + archive member + matched value)
are removed before scoring, so repeated detection of the same issue is counted
only once.

Correlation findings (rule IDs starting with `C`, e.g. `C001`–`C004`) are
excluded from the scoring input entirely before deduplication. They are
derived from primary findings that are already scored individually, so
including them as well would double-count the underlying signal. They still
appear in the findings table and CLI/HTML output — they just don't add to
`WeightedSum`.

### Step 2 — Weighted Sum

```
WeightedSum = SUM_{f in UniqueFindings}(
    SeverityWeight(f.severity) × ConfidenceFactor(f.confidence)
)
```

#### Severity Weights

| Severity | Weight |
|----------|--------|
| CRITICAL | 40.0 |
| HIGH | 20.0 |
| MEDIUM | 5.0 |
| LOW | 1.0 |
| INFO | 0.0 |

#### Confidence Factors

| Confidence | Factor |
|------------|--------|
| HIGH | 1.0 |
| MEDIUM | 0.8 |
| LOW | 0.5 |

### Step 3 — Format Multiplier

The format multiplier reflects the inherent risk surface of the artifact format:

| Format | Multiplier |
|--------|------------|
| RAW_PICKLE | 1.5 |
| LEGACY_PYTORCH | 1.5 |
| PYTORCH_ZIP | 1.3 |
| GGUF | 1.1 |
| GGML | 1.1 |
| SAFETENSORS | 1.0 |
| ONNX | 1.0 |
| HDF5 | 1.0 |
| TF_SAVEDMODEL | 1.0 |
| KERAS | 1.0 |
| UNSUPPORTED | 1.0 |
| (any other format not listed) | 1.0 |

> **Correction (2026-09-07):** an earlier version of this table omitted
> `LEGACY_PYTORCH` and implied it fell under the 1.0 "all others" default.
> It does not — `FORMAT_MULTIPLIERS` in `scoring.py` hardcodes it to **1.5**,
> the same weight as `RAW_PICKLE`, since legacy (non-ZIP) PyTorch archives
> carry the same pickle-deserialization risk surface.

### Step 4 — Obfuscation Multiplier

The obfuscation multiplier is elevated when the artifact exhibits signs of
intentional obfuscation or structural tampering:

| Condition | Multiplier |
|-----------|------------|
| Structural anomaly **and** unresolvable symbol | 1.75 |
| Structural anomaly only | 1.50 |
| Unresolvable symbol only | 1.25 |
| Neither | 1.00 |

**Structural anomaly** is flagged by findings in the P2xx, ST2xx, GU2xx rule
families or the `structural_anomaly` / `tampering` category.

**Unresolvable symbol** is flagged by rules P103/P104/P105 or the
`obfuscation` / `unresolvable` category.

**INFO-severity findings never contribute to either flag**, regardless of
rule family. This specifically matters for `P209` ("Embedded Pickle Stream in
PyTorch ZIP"): it shares the `P2xx` prefix but fires on the
`archive/data.pkl` member present in *every* legitimate PyTorch checkpoint —
it describes expected structure, not an anomaly. Before this exclusion was
added, `P209` was silently setting the structural-anomaly flag on every
PyTorch scan and inflating the score by 1.5×; it no longer does.

### Step 5 — Raw Score

```
RawScore = WeightedSum × FormatMultiplier × ObfuscationMultiplier
```

### Step 6 — Final Score (numeric)

```
FinalScore = min(100, max(0, floor(RawScore + 0.5)))
```

The score is clamped to [0, 100] and rounded to the nearest integer using
half-up rounding. This score maps to a risk band via the table in
"Risk Bands" below.

### Step 7 — Severity Floor (R5)

**Added 2026-09-06.** The purely arithmetic band from Step 6 is then checked
against a floor derived from the single worst finding's severity, and raised
if the floor is stricter:

| Worst finding severity present | Minimum risk band |
|---|---|
| CRITICAL | CRITICAL_RISK |
| HIGH | HIGH_RISK |
| MEDIUM | MEDIUM_RISK |
| LOW | LOW_RISK |
| INFO only | (no floor — arithmetic band stands, i.e. CLEAN) |

```
FinalBand = max(NumericBand(FinalScore), SeverityFloor(worst_finding.severity))
```

This floor only ever **raises** the band — it never lowers one the arithmetic
formula already put higher. It exists because a lone HIGH or CRITICAL finding
on a low-multiplier format can arithmetically score low enough to land in
LOW_RISK or MEDIUM_RISK (e.g. a single HIGH finding on SAFETENSORS with no
obfuscation multiplier scores 20 → numerically LOW_RISK), which would
under-communicate a genuinely serious individual finding. The floor guarantees
the report never undersells the single worst thing found, regardless of how
the weighted sum nets out.

---

## Risk Bands

| Final Score | Risk Band |
|-------------|-----------|
| 0 | CLEAN |
| 1 – 30 | LOW_RISK |
| 31 – 60 | MEDIUM_RISK |
| 61 – 89 | HIGH_RISK |
| 90 – 100 | CRITICAL_RISK |

A separate `UNANALYZED` band exists outside this numeric scale — it is
returned when no analyzer ran at all (e.g. an `INCONCLUSIVE` or
`UNSUPPORTED` scan outcome), signaling that the absence of findings proves
nothing, as opposed to `CLEAN`, which means an analyzer ran and found
nothing.

---

## How to Interpret the Risk Score

The risk score is a **deterministic aggregate of static suspicion signals**. It is
deliberately conservative (fail-loud), and it must be read alongside the individual
findings — not in isolation.

1. **The score measures aggregate static suspicion, not exploitability.**
   A high score means the artifact accumulated many (or high-severity) suspicious
   static signals. It does **not** assert that any code path is reachable, that an
   exploit was proven, or that the model was executed. All analysis is static; nothing
   is deserialized or run.

2. **Many structural findings can push the score into a high/critical band on their
   own.** Because the weighted sum accumulates across *every* distinct finding, a large,
   complex, but entirely legitimate artifact can accrue enough MEDIUM/HIGH structural
   findings (e.g. `P205`, `P208`, `ST2xx`, `GU2xx`) to reach HIGH_RISK or CRITICAL_RISK
   even with **zero** code-execution findings. The 250 MB Matcha-TTS PyTorch Lightning
   checkpoint is the canonical example: it scores CRITICAL_RISK from hundreds of
   structural signals while containing **no** `P101`/`P104` dangerous-callable findings.

3. **A CRITICAL_RISK band driven only by structural rules is not a confirmation of
   malicious code execution.** Before quarantining or attributing intent, verify whether
   the score is backed by genuine code-execution signals — the primary ones are `P101`
   (Dangerous Global Import), `P104` (Dangerous REDUCE), `ST101`/`GU101` (metadata code
   execution), `SC104` (dangerous system call in bundled code), and the correlated
   `C001`–`C004` chains. Their **absence** is itself a meaningful negative result.

4. **Always inspect the individual findings and correlation results.** The findings
   table and any `C0xx` correlated findings are the authoritative signal. The numeric
   band is a triage aid, not a verdict. Two artifacts with the same band can carry very
   different real-world risk.

5. **`TF101` is MEDIUM severity by design.** TensorFlow SavedModel op findings are fixed
   at MEDIUM because static graph analysis cannot confirm an op will execute at inference
   time. As a result, a SavedModel containing genuinely dangerous ops such as `ReadFile`,
   `WriteFile`, or `PyFunc` can still produce an overall band as low as **MEDIUM_RISK**
   (the R5 floor guarantees at least MEDIUM_RISK once any MEDIUM finding is present, but
   nothing higher without additional signals).
   For TensorFlow artifacts, the presence of these ops in the findings table — not the
   summary band — is the signal that must be acted on.

6. **A lone HIGH/CRITICAL finding always reaches at least its own band (R5).** Before
   2026-09-06, a single HIGH finding on a 1.0-multiplier format with no obfuscation
   could arithmetically land in LOW_RISK (score 20). It cannot anymore: the severity
   floor guarantees at least HIGH_RISK whenever any HIGH finding is present, and at
   least CRITICAL_RISK whenever any CRITICAL finding is present — see Step 7.

---

## Examples

### Example 1 — Clean artifact

- No findings → WeightedSum = 0 → FinalScore = **0** (CLEAN)

### Example 2 — Single CRITICAL finding (HIGH confidence)

- WeightedSum = 40.0 × 1.0 = 40.0
- Format: RAW_PICKLE → FormatMultiplier = 1.5
- No obfuscation → ObfuscationMultiplier = 1.0
- RawScore = 40.0 × 1.5 × 1.0 = 60.0
- Numeric band: **60** → MEDIUM_RISK
- Severity floor: CRITICAL finding present → floor = CRITICAL_RISK
- **FinalBand = CRITICAL_RISK** (floor overrides the numeric MEDIUM_RISK)

### Example 3 — CRITICAL + structural anomaly + obfuscated globals

- WeightedSum = 40.0 (CRITICAL) + 20.0 (HIGH) + 5.0 (MEDIUM) = 65.0
- Format: RAW_PICKLE → FormatMultiplier = 1.5
- Structural + unresolvable → ObfuscationMultiplier = 1.75
- RawScore = 65.0 × 1.5 × 1.75 = 170.625
- FinalScore = min(100, floor(170.625 + 0.5)) = **100** (CRITICAL_RISK; floor agrees)

### Example 4 — Single HIGH finding, safe format, no obfuscation (R5 floor in effect)

- WeightedSum = 20.0 × 1.0 = 20.0
- Format: SAFETENSORS → FormatMultiplier = 1.0
- No obfuscation → ObfuscationMultiplier = 1.0
- RawScore = 20.0 × 1.0 × 1.0 = 20.0
- Numeric band: **20** → LOW_RISK
- Severity floor: HIGH finding present → floor = HIGH_RISK
- **FinalBand = HIGH_RISK** (without the R5 floor this would incorrectly read LOW_RISK)

---

## Implementation Reference

- **Source:** `src/model_scanner/core/scoring.py` — `RiskEngine.calculate()`
- **Calibration constants:** `SEVERITY_WEIGHTS`, `CONFIDENCE_FACTORS`,
  `FORMAT_MULTIPLIERS`, `OBFUSCATION_*`, `_SEVERITY_BAND_FLOOR` in the same module.
- **Risk band mapping:** `model_scanner.core.models.risk_band_for_score()`
- **Note:** `scoring.py`'s own module docstring still labels the calibration
  constants "PROPOSED / INITIAL — Subject to Empirical Calibration." That
  language predates this document's last verification pass and should be
  revisited/removed once the weights are treated as final for submission.

_The formula is deterministic and has zero mandatory runtime dependencies
(stdlib only). No ML frameworks, network calls, or dynamic code execution._
