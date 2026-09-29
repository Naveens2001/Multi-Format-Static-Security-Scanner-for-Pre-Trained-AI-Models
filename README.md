# Static Security Scanner for Pre-Trained AI Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Complete](https://img.shields.io/badge/status-phase--12--complete-brightgreen)]()

---

## 1. Project Purpose

A **unified, multi-technique static security scanner** for pre-trained AI model artifacts.
It detects supply-chain attack vectors, malicious code execution payloads, metadata injection,
and structural container corruption embedded in model files without executing any model
code, loading any weights, or invoking any dynamic deserialization runtimes.

---

## 2. Static-Only Security Architecture

> **SECURITY GUARANTEE: This scanner never executes model code.**

The scanner operates exclusively on raw binary byte streams using standard-library static analysis:

- **NO** `pickle.loads()` — Pickle streams are disassembled into symbolic opcodes, never deserialized.
- **NO** `torch.load()` — Modern PyTorch archives are inspected as ZIP containers only.
- **NO** tensor weight loading — SafeTensors and GGUF weight data regions are never read into memory.
- **NO** `eval()` / `exec()` — Prohibited throughout the entire codebase.
- **NO** model class instantiation — The scanner inspects bytes; it never imports or executes model graphs.

Any container or stream that fails parsing is safely handled fail-closed with structured error findings.

---

## 3. Supported Model Formats

| Format | Container / Format Specification | Attack Surfaces & Rules Detected |
|---|---|---|
| **Raw Pickle** | `.pkl`, `.pickle`, `.joblib`, `.dill` | Opcode global imports (`P101`–`P107`), structural anomalies (`P201`–`P211`) |
| **PyTorch ZIP** | `.pt`, `.pth`, `.bin` (ZIP format) | ZIP traversal, nested archive detection (`P210`), compression-ratio anomaly (`P211`), embedded Pickle stream analysis |
| **SafeTensors** | `.safetensors` | Header bounds, JSON injection, metadata attacks (`ST101`–`ST105`), tensor anomalies (`ST201`–`ST209`) |
| **GGUF** | `.gguf` (v1, v2, v3) | Header integrity, metadata injection (`GU101`–`GU106`), tensor range verification (`GU201`–`GU210`) |
| **Repository / Directory** | Model directory | Supply-chain config (`SC101`–`SC109`), dependency security (`SC201`–`SC204`), shard integrity (`G005`–`G008`) |
| **Cross-Format** | Any file / container | Magic-byte format detection, extension spoofing (`G001`), SHA-256 fingerprinting, cross-signal correlation (`C001`–`C003`) |

### Format Scope & Limitations
- **Modern PyTorch ZIP:** Fully supported (ZIP containers containing `archive/data.pkl`).
- **Legacy Non-ZIP PyTorch:** Checkpoints serialized under legacy non-ZIP PyTorch formats are safely identified and categorized as `UNSUPPORTED`.

---

## 4. User Interfaces

### 4.1 Primary Interface: Web UI
The Web UI is the primary user interface, providing interactive drag-and-drop file scanning, Hugging Face URL analysis, live progress polling, and risk dashboards.

```bash
# Start local Web UI server (default: http://127.0.0.1:8000)
python -m model_scanner serve
```

### 4.2 Supporting Interface: CLI
For CI/CD pipelines, automated scripts, and headless scanning:

```bash
# Scan a local file or directory
python -m model_scanner scan path/to/model.safetensors

# Scan with JSON output
python -m model_scanner scan path/to/model.pt --json

# View help and version
python -m model_scanner --help
python -m model_scanner --version
```

### 4.3 Programmatic Python API
```python
from model_scanner import scan, export_json

result = scan("model.safetensors")
print(f"Risk Score: {result.summary.max_risk_score}/100 ({result.summary.overall_outcome})")

json_report = export_json(result)
```

---

## 5. Security Taxonomy

Findings are classified using a three-layer taxonomy:

**Primary — OWASP Machine Learning Security Top Ten 2023**
The primary framework. 53 of 84 rules map to **ML06:2023 AI Supply Chain Attacks**, covering
arbitrary code execution via model deserialization, malicious metadata injection, bundled
executable code, compromised dependency chains, and correlated cross-artifact threats.
The remaining 31 rules comprise 28 format-level structural-anomaly rules (malformed headers,
resource exhaustion, invalid tensor descriptors) that fall outside the scope of any current
OWASP ML Top Ten category and are designated *Not directly mapped*, plus 3 weight-integrity
rules (WI101–WI103) mapped to **ML03:2023 Model Inversion Attack**.

**Secondary — CWE (Common Weakness Enumeration)**
Every rule carries at least one CWE identifier describing the technical weakness class,
independent of framework membership. CWEs include CWE-502 (Deserialization), CWE-94
(Code Injection), CWE-22 (Path Traversal), CWE-125/190 (Buffer/Integer), CWE-400/770
(Resource Exhaustion), and CWE-494 (Download Without Integrity Check).

**Tertiary — Internal model-artifact security category**
Seven domain-specific categories used for dashboard filtering and risk aggregation:
`code_execution`, `structural_anomaly`, `supply_chain`, `credential_exposure`,
`format_integrity`, `provenance`, `tampering`.

## 6. Rule Catalog Summary

**84 rules** across 10 detection prefixes:

| Prefix | Category | Rules |
|--------|----------|-------|
| `P1xx` | Pickle — Code Execution | 7 |
| `P2xx` | Pickle / PyTorch ZIP — Structural Anomalies | 11 |
| `ST1xx` | SafeTensors — Metadata | 5 |
| `ST2xx` | SafeTensors — Structural Anomalies | 9 |
| `GU1xx` | GGUF — Metadata | 6 |
| `GU2xx` | GGUF — Structural Anomalies | 10 |
| `G0xx` | Cross-Format Integrity | 6 |
| `SC1xx` | Supply Chain — Config & Code Execution | 9 |
| `SC2xx` | Supply Chain — Dependency Security | 4 |
| `C0xx` | Correlated Cross-Signal Rules | 4 |
| `WI1xx` | Cross-Format Weight Integrity | 3 |
| `ON1xx` | ONNX — Graph & External Data | 4 |
| `TF1xx` | TensorFlow SavedModel | 3 |
| `KR1xx` | Keras — Custom Class & Archive | 3 |

See [`docs/rule_catalog.md`](docs/rule_catalog.md) for the full catalog with descriptions, CWE mappings, OWASP ML classifications, and MITRE ATT&CK techniques.

---

## 6. Benchmark Performance

Evaluated across a deterministic 27-sample synthetic test corpus:
- **Precision:** `1.0000` (100.0%)
- **Recall (TPR):** `1.0000` (100.0%)
- **F1 Score:** `1.0000`
- **False Positive Rate (FPR):** `0.0000` (0.0%)
- **False Negative Rate (FNR):** `0.0000` (0.0%)
- **Accuracy:** `100.00%`
- **Mean Latency:** `~1 ms`
- **Peak Memory:** `0.10 MB`

Run benchmarks locally:
```bash
python -m benchmarks.harness
```

*Note: Benchmark corpus is a deterministic synthetic test set. Comparative baselines for external tools (ModelScan, Picklescan) are reported if available in the Python environment.*

---

## 7. Implementation Phases (Complete)

| Phase | Description | Status |
|---|---|---|
| **1** | Foundation & Build Setup | Complete |
| **2** | Core Data Models & Error Hierarchy | Complete |
| **3** | Acquisition & Format Detection | Complete |
| **4** | Pickle & PyTorch Analyzers | Complete |
| **5** | SafeTensors Analyzer | Complete |
| **6** | GGUF Analyzer | Complete |
| **7** | Security Rule Engine & Catalog | Complete |
| **8** | Severity & Risk Scoring Engines | Complete |
| **9** | Presentation, Reporting & Web UI | Complete |
| **10** | CLI & Python API Interfaces | Complete |
| **11** | Benchmark Harness & Corpus | Complete |
| **12** | Hardening, Safety & Robustness | Complete |

---

## 8. Project 1 Scope & Limitations

This scanner is a **static, non-executing supply-chain security scanner** for pre-trained AI model artifacts. It is complete for Project 1.

**In scope:**
- Binary format parsing (Pickle, PyTorch ZIP, SafeTensors, GGUF)
- Repository/supply-chain analysis (config, dependencies, provenance, shard integrity)
- Cross-signal finding correlation (C001–C003)
- Risk scoring with severity-weighted formula

**Out of scope (Project 2 / future research):**
- Tensor weight statistical analysis
- Backdoor detection via weight distributions
- Activation clustering or behavioral analysis
- Cryptographic signature verification (beyond presence detection)
- Live CVE/advisory network lookups

---

## 9. Security Disclaimer

This tool performs static analysis to detect known patterns of supply-chain vulnerabilities, code execution primitives, and container tampering. No security scanner can guarantee 100% detection of novel zero-day obfuscation techniques. Models from untrusted sources should always be validated through defense-in-depth security practices.

---

## 10. License

MIT License. See [LICENSE](LICENSE) for details.
