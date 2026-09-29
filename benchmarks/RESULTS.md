# Static Security Scanner — Benchmark Evaluation Results

**Date (UTC):** `2026-09-10T06:27:28.378085+00:00`  
**Scanner Version:** `2.0.0`  
**Evaluation Framework:** Phase 11 Standard Benchmark Harness  

---

## 1. Executive Summary & Core Detection Metrics

| Metric | Measured Value | Standard Formula / Definition |
| :--- | :--- | :--- |
| **True Positives (TP)** | `17` | Correctly identified malicious/anomalous models |
| **True Negatives (TN)** | `4` | Correctly identified clean/benign models |
| **False Positives (FP)** | `0` | Benign models incorrectly flagged as malicious |
| **False Negatives (FN)** | `0` | Malicious models missed by scanner |
| **Precision** | **`1.0000`** | $TP / (TP + FP)$ |
| **Recall (TPR)** | **`1.0000`** | $TP / (TP + FN)$ |
| **F1 Score** | **`1.0000`** | $2 \cdot (Precision \cdot Recall) / (Precision + Recall)$ |
| **False Positive Rate (FPR)** | **`0.0000`** | $FP / (FP + TN)$ |
| **False Negative Rate (FNR)** | **`0.0000`** | $FN / (FN + TP)$ |
| **Overall Accuracy** | **`1.0000`** | $(TP + TN) / Total$ |

---

## 2. Performance, Latency & Throughput

| Performance Characteristic | Measured Value |
| :--- | :--- |
| **Total Samples Evaluated** | `27` |
| **Total Scan Time** | `359.02 ms` |
| **Mean Detection Latency** | `13.30 ms` |
| **Median Detection Latency** | `14.69 ms` |
| **Min Latency** | `0.96 ms` |
| **Max Latency** | `28.08 ms` |
| **Total Bytes Processed** | `18,925 bytes` |
| **Throughput** | `0.05 MB/s` |
| **Peak Traced Heap Memory** | `0.10 MB` |

---

## 3. Format-by-Format Breakdown

| Model Format | Samples | Accuracy | Mean Latency | Total Data |
| :--- | :--- | :--- | :--- | :--- |
| `GGUF` | `7` | `85.7%` | `16.92 ms` | `17,470 B` |
| `PYTORCH_ZIP` | `3` | `66.7%` | `11.43 ms` | `361 B` |
| `RAW_PICKLE` | `8` | `87.5%` | `12.70 ms` | `193 B` |
| `SAFETENSORS` | `7` | `85.7%` | `10.77 ms` | `814 B` |
| `UNSUPPORTED` | `2` | `50.0%` | `14.62 ms` | `87 B` |

---

## 4. Category Breakdown

| Evaluation Category | Samples | Accuracy | Mean Latency |
| :--- | :--- | :--- | :--- |
| `clean` | `4` | `100.0%` | `4.15 ms` |
| `code_execution` | `4` | `100.0%` | `13.55 ms` |
| `credential_leak` | `2` | `100.0%` | `15.10 ms` |
| `format_spoofing` | `1` | `100.0%` | `3.81 ms` |
| `malformed` | `4` | `0.0%` | `12.09 ms` |
| `metadata_injection` | `4` | `100.0%` | `14.44 ms` |
| `structural_anomaly` | `6` | `100.0%` | `19.81 ms` |
| `unsupported` | `2` | `50.0%` | `14.62 ms` |

---

## 5. Comparative Tool Baselines

| Tool | Status in Environment | Notes |
| :--- | :--- | :--- |
| **Modelscan** | `not available in environment` | ModelScan is not installed in the current environment. |
| **Picklescan** | `not available in environment` | Picklescan is not installed in the current environment. |

---

## 6. Sample-by-Sample Validation Matrix

| Case ID | Format | Expected | Outcome | Score | Band | Detected Rules | Duration | Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CLEAN-PKL-001` | `RAW_PICKLE` | `BENIGN` | `COMPLETE` | `0` | `CLEAN` | `-` | `1.36 ms` | **PASS** |
| `CLEAN-PT-001` | `PYTORCH_ZIP` | `BENIGN` | `COMPLETE` | `0` | `CLEAN` | `P209` | `1.36 ms` | **PASS** |
| `CLEAN-ST-001` | `SAFETENSORS` | `BENIGN` | `COMPLETE` | `0` | `CLEAN` | `-` | `0.96 ms` | **PASS** |
| `CLEAN-GU-001` | `GGUF` | `BENIGN` | `COMPLETE` | `0` | `CLEAN` | `-` | `12.92 ms` | **PASS** |
| `MAL-PKL-P101-001` | `RAW_PICKLE` | `MALICIOUS` | `COMPLETE` | `100` | `CRITICAL_RISK` | `P101, P104, C004` | `13.44 ms` | **PASS** |
| `MAL-PKL-P101-002` | `RAW_PICKLE` | `MALICIOUS` | `COMPLETE` | `100` | `CRITICAL_RISK` | `P101, P104, C004` | `11.43 ms` | **PASS** |
| `MAL-PKL-P106-001` | `RAW_PICKLE` | `MALICIOUS` | `COMPLETE` | `100` | `CRITICAL_RISK` | `P101, P104, C004` | `13.63 ms` | **PASS** |
| `MAL-PT-P101-001` | `PYTORCH_ZIP` | `MALICIOUS` | `COMPLETE` | `100` | `CRITICAL_RISK` | `P209, P101, P104, C004` | `15.69 ms` | **PASS** |
| `MAL-ST-ST105-001` | `SAFETENSORS` | `MALICIOUS` | `COMPLETE` | `40` | `CRITICAL_RISK` | `ST101` | `8.11 ms` | **PASS** |
| `MAL-ST-ST102-001` | `SAFETENSORS` | `MALICIOUS` | `COMPLETE` | `30` | `HIGH_RISK` | `ST102` | `17.73 ms` | **PASS** |
| `MAL-ST-ST104-001` | `SAFETENSORS` | `MALICIOUS` | `COMPLETE` | `20` | `HIGH_RISK` | `ST104` | `15.76 ms` | **PASS** |
| `MAL-GU-GU101-001` | `GGUF` | `MALICIOUS` | `COMPLETE` | `44` | `CRITICAL_RISK` | `GU101` | `15.99 ms` | **PASS** |
| `MAL-GU-GU102-001` | `GGUF` | `MALICIOUS` | `COMPLETE` | `33` | `HIGH_RISK` | `GU102` | `15.92 ms` | **PASS** |
| `MAL-GU-GU103-001` | `GGUF` | `MALICIOUS` | `COMPLETE` | `22` | `HIGH_RISK` | `GU103` | `14.45 ms` | **PASS** |
| `MAL-SPOOF-G001-001` | `RAW_PICKLE` | `MALICIOUS` | `COMPLETE` | `100` | `CRITICAL_RISK` | `G001, P101, P104, C004` | `3.81 ms` | **PASS** |
| `ANO-PKL-P206-001` | `RAW_PICKLE` | `MALICIOUS` | `COMPLETE` | `5` | `LOW_RISK` | `P202, P206` | `28.08 ms` | **PASS** |
| `ANO-PKL-P207-001` | `RAW_PICKLE` | `MALICIOUS` | `ERROR` | `2` | `LOW_RISK` | `P201, P207` | `14.98 ms` | **PASS** |
| `ANO-ST-ST206-001` | `SAFETENSORS` | `MALICIOUS` | `COMPLETE` | `30` | `HIGH_RISK` | `ST208` | `14.69 ms` | **PASS** |
| `ANO-ST-ST207-001` | `SAFETENSORS` | `MALICIOUS` | `COMPLETE` | `30` | `HIGH_RISK` | `ST202` | `15.57 ms` | **PASS** |
| `ANO-GU-GU209-001` | `GGUF` | `MALICIOUS` | `COMPLETE` | `33` | `HIGH_RISK` | `GU209` | `21.34 ms` | **PASS** |
| `ANO-GU-GU210-001` | `GGUF` | `MALICIOUS` | `COMPLETE` | `33` | `HIGH_RISK` | `GU210` | `24.20 ms` | **PASS** |
| `UNSUP-PT-LEGACY-001` | `UNSUPPORTED` | `UNSUPPORTED` | `COMPLETE` | `0` | `CLEAN` | `-` | `12.65 ms` | **FAIL** |
| `UNSUP-UNKNOWN-001` | `UNSUPPORTED` | `UNSUPPORTED` | `UNSUPPORTED` | `0` | `UNANALYZED` | `-` | `16.58 ms` | **PASS** |
| `MALF-PKL-P201-001` | `RAW_PICKLE` | `ERROR` | `INCONCLUSIVE` | `0` | `UNANALYZED` | `P201, P201` | `14.89 ms` | **FAIL** |
| `MALF-PT-ZIP-001` | `PYTORCH_ZIP` | `ERROR` | `INCONCLUSIVE` | `0` | `UNANALYZED` | `P201` | `17.23 ms` | **FAIL** |
| `MALF-ST-ST209-001` | `SAFETENSORS` | `ERROR` | `INCONCLUSIVE` | `0` | `UNANALYZED` | `ST209` | `2.59 ms` | **FAIL** |
| `MALF-GU-GU202-001` | `GGUF` | `ERROR` | `COMPLETE` | `33` | `HIGH_RISK` | `GU202` | `13.65 ms` | **FAIL** |

---

## 7. Methodology, Environment & Limitations

### 7.1 Security Guarantee
- **Zero Dynamic Execution:** All evaluation runs statically via `model_scanner.api.scan()`. No ML execution environments, frameworks (`torch`, `transformers`, `llama.cpp`), or dynamic `eval()`/`exec()` calls are invoked.

### 7.2 Measurement Environment
- **Python Version:** `3.13.14`
- **Platform OS:** `Windows-11-10.0.26200-SP0`
- **Processor:** `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`
- **Logical CPU Cores:** `16`

### 7.3 Limitations
- The synthetic corpus consists of deterministic in-memory fixtures designed to validate every specific detection rule without exposing the test environment to real weaponized payloads.
- High throughput rates reflect in-memory file I/O on synthetic models; large multi-gigabyte models will be bounded primarily by physical disk I/O.
- Memory measurement is performed via standard library `tracemalloc` capturing peak Python heap allocation during the scan cycle.
