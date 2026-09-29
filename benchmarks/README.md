# Model Scanner — Benchmark Evaluation Harness (Phase 11)

Reproducible, deterministic evaluation framework for measuring static security analysis detection accuracy, latency, and throughput across pre-trained AI model formats (Pickle, PyTorch ZIP, SafeTensors, GGUF).

---

## 1. Quick Start

Run the complete evaluation suite using the standard entry point:

```bash
python -m benchmarks.harness
```

Or specify custom output paths:

```bash
python -m benchmarks.harness --output-json benchmarks/results.json --output-md benchmarks/RESULTS.md
```

---

## 2. Core Measurement Dimensions

### 2.1 Detection Metrics
- **True Positives (TP):** Malicious / security-violating models correctly detected with security findings.
- **True Negatives (TN):** Clean / benign models correctly passed without security findings.
- **False Positives (FP):** Clean models incorrectly flagged with security findings.
- **False Negatives (FN):** Malicious models that evaded detection.
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall (TPR):** $\frac{TP}{TP + FN}$
- **F1 Score:** $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
- **False Positive Rate (FPR):** $\frac{FP}{FP + TN}$
- **False Negative Rate (FNR):** $\frac{FN}{FN + TP}$

### 2.2 Performance Metrics
- **Detection Latency:** High-resolution monotonic timing per sample (mean, median, min, max in milliseconds).
- **Throughput:** Total file bytes scanned per second ($\text{MB/s}$).
- **Peak Memory:** Peak heap allocation traced via Python `tracemalloc`.

---

## 3. Evaluation Corpus

The benchmark corpus consists of 27 deterministic synthetic samples covering:
1. **Clean / Benign:** Safe Pickle, PyTorch ZIP with tensor rebuild, SafeTensors, and GGUF v3.
2. **Code Execution:** Dangerous globals (`os.system`, `subprocess.Popen`), dynamic `eval`, and PyTorch ZIP embedded exploits.
3. **Metadata Injection:** SafeTensors and GGUF header code injection, path traversal, and credential leaks.
4. **Structural Anomalies:** Concatenated Pickle streams, trailing unparsed bytes, out-of-bounds tensor offsets, and overlapping tensor ranges.
5. **Format Spoofing:** SafeTensors extension disguising Pickle payloads (`G001`).
6. **Unsupported Formats:** Legacy non-ZIP PyTorch binary checkpoints and unrecognized formats.
7. **Malformed / Corrupted Containers:** Truncated Pickles, corrupted ZIP streams, truncated SafeTensors headers, and invalid GGUF magic bytes.

---

## 4. Security Guarantee

- **Zero Model Execution:** Scanning is 100% static. No weights are deserialized or executed.
- **No Heavy ML Runtimes:** The harness strictly uses standard library modules and the public `model_scanner.api.scan()` interface.
