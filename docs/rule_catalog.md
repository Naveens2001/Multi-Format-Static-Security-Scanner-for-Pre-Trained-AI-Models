# Static Security Rule Catalog

**Total rules:** 87  
**Scanner version:** 2.0.0  
**Format:** `RULE_ID — Name — Severity — Category`

---

## Table of Contents

- [Pickle — Code Execution (P1xx)](#pickle-—-code-execution-p1xx) (7 rules)
- [Pickle / PyTorch ZIP — Structural Anomalies (P2xx)](#pickle-pytorch-zip-—-structural-anomalies-p2xx) (11 rules)
- [SafeTensors — Metadata (ST1xx)](#safetensors-—-metadata-st1xx) (5 rules)
- [SafeTensors — Structural Anomalies (ST2xx)](#safetensors-—-structural-anomalies-st2xx) (9 rules)
- [GGUF — Metadata (GU1xx)](#gguf-—-metadata-gu1xx) (6 rules)
- [GGUF — Structural Anomalies (GU2xx)](#gguf-—-structural-anomalies-gu2xx) (10 rules)
- [Cross-Format Integrity (G0xx)](#cross-format-integrity-g0xx) (6 rules)
- [Supply Chain — Config & Code Execution (SC1xx)](#supply-chain-—-config-code-execution-sc1xx) (9 rules)
- [Supply Chain — Dependency Security (SC2xx)](#supply-chain-—-dependency-security-sc2xx) (4 rules)
- [Correlated Cross-Signal Rules (C0xx)](#correlated-cross-signal-rules-c0xx) (4 rules)
- [Cross-Format Weight Integrity (WI1xx)](#cross-format-weight-integrity-wi1xx) (3 rules)
- [ONNX Security Rules (ON1xx)](#onnx-security-rules-on1xx) (4 rules)
- [TensorFlow SavedModel Security Rules (TF1xx)](#tensorflow-savedmodel-security-rules-tf1xx) (3 rules)
- [Keras Security Rules (KR1xx)](#keras-security-rules-kr1xx) (5 rules)

---

## Pickle — Code Execution (P1xx)

### P101 — Dangerous Global Import

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204.002: User Execution: Malicious File` |

**Description:** The Pickle stream imports a known dangerous module or callable capable of arbitrary code execution (e.g. os.system, subprocess.Popen).

**Detection Condition:** GLOBAL or STACK_GLOBAL opcode targets a blocklisted callable.

**Remediation:** Do not deserialize this model. Re-serialize model weights using SafeTensors.

---

### P102 — Suspicious Global Import

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |

**Description:** The Pickle stream imports a dynamic loader, reflection, or native FFI binding symbol.

**Detection Condition:** GLOBAL opcode references dynamic loader or reflection modules.

**Remediation:** Audit referenced global symbols and verify if necessary for model inference.

---

### P103 — STACK_GLOBAL Dynamic Import

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502`, `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** The Pickle stream uses STACK_GLOBAL opcode to import symbols constructed dynamically.

**Detection Condition:** STACK_GLOBAL opcode resolves to a symbol that is not on the SAFE allowlist. HIGH when the symbol is blocklisted, MEDIUM otherwise; the normalizer escalates MEDIUM to HIGH in strict mode.

**Remediation:** Audit stack-constructed symbols and convert model weights to SafeTensors.

---

### P104 — Dangerous REDUCE Invocation

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502`, `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204.002: User Execution: Malicious File` |

**Description:** A REDUCE operation targets a callable capable of executing system commands.

**Detection Condition:** REDUCE opcode invokes a blocklisted callable, or a runpy dynamic-execution primitive that is escalated from SUSPICIOUS because it is actually called.

**Remediation:** Reject this model immediately. It contains an arbitrary code execution payload.

---

### P105 — Suspicious REDUCE Invocation

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |

**Description:** A REDUCE operation targets an unverified or dynamic callable.

**Detection Condition:** REDUCE opcode invokes a callable on the SUSPICIOUS list (dynamic loader, reflection, or FFI binding). An UNKNOWN callable does not fire this rule.

**Remediation:** Inspect the callable target and replace with standard state dictionary weights.

---

### P106 — Suspicious BUILD State Mutation

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |

**Description:** A BUILD operation mutates object state with suspicious parameters.

**Detection Condition:** BUILD opcode modifies a deserialization-critical state key (__setstate__, __class__, __globals__, ...) or targets a non-SAFE global. Ordinary module attribute names are not reported.

**Remediation:** Verify object state dictionary and ensure only tensor weights are reconstructed.

---

### P107 — Suspicious NEWOBJ Instantiation

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204: User Execution` |

**Description:** A NEWOBJ/NEWOBJ_EX opcode instantiates an unverified class.

**Detection Condition:** NEWOBJ or NEWOBJ_EX instantiates a class that is not on the SAFE globals allowlist. HIGH when the class is blocklisted, MEDIUM otherwise.

**Remediation:** Ensure only trusted tensor container classes are instantiated.

---

## Pickle / PyTorch ZIP — Structural Anomalies (P2xx)

### P201 — Malformed or Truncated Opcode Stream

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** The Pickle opcode stream is corrupted, truncated, or contains malformed arguments.

**Detection Condition:** Opcode argument parsing reaches premature EOF or invalid byte syntax.

**Remediation:** Inspect file integrity or re-download the model file.

---

### P202 — Duplicate or Misplaced PROTO Header

| Field | Value |
|-------|-------|
| **Severity** | 🔵 `LOW` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** A PROTO opcode appears more than once or at an unexpected position in the stream.

**Detection Condition:** PROTO opcode encountered at non-zero offset or multiple times.

**Remediation:** Clean up corrupted stream headers using standard serialization tools.

---

### P203 — Unsupported Protocol Version

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** The stream declares an invalid or unsupported Pickle protocol version (> 5).

**Detection Condition:** PROTO opcode specifies protocol version > 5 or < 0.

**Remediation:** Re-serialize the model using standard Pickle protocol 2, 4, or 5.

---

### P204 — FRAME Boundary Desynchronization

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-125` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1203: Exploitation for Client Execution` |

**Description:** An opcode crosses beyond declared FRAME boundary in Protocol 4/5 framing.

**Detection Condition:** Opcode span exceeds the current FRAME segment boundary.

**Remediation:** Re-serialize with standard pickle tools to fix framing desynchronization.

---

### P205 — Invalid Memo Slot Reference

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** A GET or BINGET opcode references an unassigned memo slot index.

**Detection Condition:** GET/BINGET opcode references a memo index that was never stored.

**Remediation:** Verify that the model serialization was not tampered with or truncated.

---

### P206 — Concatenated Pickle Streams

| Field | Value |
|-------|-------|
| **Severity** | 🔵 `LOW` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-436` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** Multiple valid Pickle streams delimited by STOP opcodes are concatenated in one file.

**Detection Condition:** Additional valid opcodes detected after an initial STOP opcode.

**Remediation:** Ensure each model file contains exactly one discrete serialized payload.

---

### P207 — Trailing Unparsed Bytes

| Field | Value |
|-------|-------|
| **Severity** | 🔵 `LOW` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-436` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** Unparsed trailing data detected after the final STOP opcode in the stream.

**Detection Condition:** Non-opcode byte sequence exists after the terminating STOP opcode.

**Remediation:** Strip unparsed payload trailing bytes to prevent payload smuggling.

---

### P208 — Excessive Opcode or Memo Resource Usage

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-400` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** Pickle stream exceeded maximum opcode count or memo entry ceiling (DoS guard).

**Detection Condition:** Opcode count > max_opcodes or memo size > max_memo_slots.

**Remediation:** Check model size or increase scanner resource ceilings if legitimate.

---

### P209 — Embedded Pickle Stream in PyTorch ZIP Container

| Field | Value |
|-------|-------|
| **Severity** | ⚪ `INFO` |
| **Category** | `format_integrity` |
| **Affected Format** | `PYTORCH_ZIP` |
| **CWE** | `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204.002: User Execution: Malicious File` |

**Description:** A PyTorch ZIP archive contains an embedded Pickle stream member. This is expected for standard PyTorch checkpoints but is flagged for awareness; the stream is analysed for malicious opcodes separately.

**Detection Condition:** ZIP member has a Pickle magic-byte prefix (\x80\x02 or \x80\x04/\x05) or passes the Protocol-0 heuristic.

**Remediation:** Inspect embedded Pickle opcodes for dangerous calls. Prefer migrating model weights to the SafeTensors format.

---

### P210 — Nested Archive in PyTorch ZIP Container

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `PYTORCH_ZIP` |
| **CWE** | `CWE-434`, `CWE-693` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** A PyTorch ZIP archive contains a nested ZIP archive member. Legitimate model checkpoints do not embed nested archives; this pattern is associated with evasion or payload staging.

**Detection Condition:** A non-Pickle, non-tensor ZIP member starts with PK\x03\x04 magic bytes (nested archive, depth >= 2).

**Remediation:** Inspect the nested archive contents. Reject artifacts from untrusted sources that contain unexpected nested archives.

---

### P211 — Extreme Compression Ratio (Zip-Bomb Indicator)

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `PYTORCH_ZIP` |
| **CWE** | `CWE-400`, `CWE-770` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** A ZIP member's declared uncompressed size is more than 1000x its compressed size, a hallmark of zip-bomb payloads. Decompressing this artifact could exhaust disk or memory resources.

**Detection Condition:** compress_size / file_size < 0.001 for any non-empty ZIP member.

**Remediation:** Reject or quarantine this archive. Verify artifact provenance before expanding.

---

## SafeTensors — Metadata (ST1xx)

### ST101 — Metadata Code Execution String

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |

**Description:** SafeTensors metadata contains executable code or eval/exec command primitives.

**Detection Condition:** __metadata__ value matches known code execution regex pattern.

**Remediation:** Remove executable code strings from the __metadata__ header dictionary.

---

### ST102 — Path Traversal Sequence in Metadata

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `tampering` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-22` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1083: File and Directory Discovery` |

**Description:** SafeTensors metadata contains directory traversal patterns such as '../' or absolute filesystem paths.

**Detection Condition:** __metadata__ values match a path traversal regex pattern (e.g. '../', '..\\', '%2e%2e%2f', '/etc/passwd'); OR any top-level tensor name (JSON key) contains the same patterns. Legitimate tensor names never contain filesystem traversal sequences.

**Remediation:** Remove filesystem path references from metadata and tensor names. SafeTensors __metadata__ must contain only plain informational strings; tensor names must not contain path-traversal sequences.

---

### ST103 — Suspicious URL in Metadata

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `credential_exposure` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-200` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1071.001: Application Layer Protocol: Web Protocols` |

**Description:** SafeTensors metadata contains unverified external HTTP/HTTPS URLs or C2 endpoints.

**Detection Condition:** __metadata__ value contains an external URL string.

**Remediation:** Verify referenced URLs and ensure no command-and-control links exist in metadata.

---

### ST104 — Credential or Secret Exposure in Metadata

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `credential_exposure` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-798`, `CWE-200` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1552.001: Unsecured Credentials: Credentials In Files` |

**Description:** SafeTensors metadata contains hardcoded API keys, private tokens, or credentials.

**Detection Condition:** __metadata__ value matches credential or API key regex pattern.

**Remediation:** Purge exposed credentials immediately and revoke any compromised tokens.

---

### ST105 — Script Injection Tag in Metadata

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `code_execution` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-79` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059.007: Command and Scripting Interpreter: JavaScript` |

**Description:** SafeTensors metadata contains HTML script tags or XSS injection strings.

**Detection Condition:** __metadata__ value contains <script>, <iframe>, or javascript: tags.

**Remediation:** Sanitize metadata values to remove HTML and client-side script injection tags.

---

## SafeTensors — Structural Anomalies (ST2xx)

### ST201 — Header Size or JSON Token DoS

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-400` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** The SafeTensors JSON header exceeds the configured size budget, or the header exhausts the scanner's JSON token, nesting, or tensor-count ceilings while being parsed. Both are denial-of-service guards.

**Detection Condition:** Header length prefix > max_safetensors_header_mb (default 16 MB, hard maximum 128 MB), JSON token or nesting budget exhausted, or the number of tensor descriptors exceeds the sentinel limit.

**Remediation:** Re-export the model with a header within the configured budget (default 16 MB). Raise max_safetensors_header_mb only for an artifact whose provenance you trust.

---

### ST202 — Tensor Data Range Overlap

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-125` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** Two or more tensor descriptors declare overlapping data offset byte spans.

**Detection Condition:** Tensor span [start_i, end_i) overlaps with [start_j, end_j).

**Remediation:** Re-serialize tensors to ensure each tensor has a distinct byte span.

---

### ST203 — Invalid Tensor Data Offsets

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-125` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1190: Exploit Public-Facing Application` |

**Description:** A tensor descriptor declares a structurally invalid data_offsets field: not a two-element list, not integers, negative, or with a start offset greater than its end offset.

**Detection Condition:** data_offsets is not a 2-element list, contains a non-integer, contains a negative value, or data_offsets[0] > data_offsets[1].

**Remediation:** Ensure every tensor declares data_offsets as a forward, non-negative [start, end] integer pair. Re-serialize with a trusted serializer.

---

### ST204 — Invalid Tensor dtype

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** A tensor descriptor's dtype is not a string, or is not one of the dtypes defined by the SafeTensors specification.

**Detection Condition:** dtype is not a string, or dtype is not in the standard SafeTensors type table.

**Remediation:** Use standard SafeTensors data types (e.g. F32, F16, BF16, I64, I32, BOOL).

---

### ST205 — Invalid Tensor shape

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-190` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** A tensor's shape is not a list, or contains an entry that is not an integer or that is negative.

**Detection Condition:** shape is not a list, or any shape[i] is not an integer or is negative.

**Remediation:** Ensure every tensor shape is a list of non-negative integers. Re-serialize the model with a trusted serializer.

---

### ST206 — Missing Required Tensor Fields

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** A tensor entry in the header is not a JSON object, or is missing one of the required descriptor fields (dtype, shape, data_offsets).

**Detection Condition:** Tensor descriptor is not a JSON object, or the keys 'dtype', 'shape', or 'data_offsets' are absent from it.

**Remediation:** Ensure all tensor entries adhere to the SafeTensors JSON schema specification.

---

### ST207 — Duplicate Tensor Name

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** SafeTensors header contains duplicate tensor dictionary keys.

**Detection Condition:** Duplicate tensor name detected during streaming token evaluation.

**Remediation:** Ensure tensor names are strictly unique within the header JSON object.

---

### ST208 — Tensor Range Outside File Bounds

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-125` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** A tensor's declared data_offsets end lies beyond the end of the SafeTensors data section, so the descriptor claims bytes the file does not contain.

**Detection Condition:** data_offsets[1] > the size of the data section (file size - 8 - header length).

**Remediation:** Re-acquire the file from a trusted source and verify its checksum. A tensor range running past the end of the file indicates truncation or a hand-edited header.

---

### ST209 — Truncated or Unreadable Header

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `SAFETENSORS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** The SafeTensors file is truncated or its header is unreadable. The file may be corrupted, only partially downloaded, or deliberately malformed.

**Detection Condition:** Header length prefix missing, JSON parse error, or file shorter than 8 + header_len bytes.

**Remediation:** Re-acquire the file from a trusted source. Verify checksum after download.

---

## GGUF — Metadata (GU1xx)

### GU101 — Metadata Code Injection

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |

**Description:** GGUF metadata KV entry contains executable script or shell command strings.

**Detection Condition:** GGUF KV string value matches command injection regex.

**Remediation:** Remove executable code primitives from model metadata KV pairs.

---

### GU102 — Path Traversal Sequence in Metadata

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `tampering` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-22` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1083: File and Directory Discovery` |

**Description:** GGUF metadata contains directory traversal patterns (e.g. ../ or /etc/).

**Detection Condition:** GGUF KV value contains '../' or absolute filesystem paths.

**Remediation:** Sanitize metadata paths to prevent arbitrary file overwrite or traversal.

---

### GU103 — Metadata Credential Exposure

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `credential_exposure` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-798` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1552.001: Unsecured Credentials: Credentials In Files` |

**Description:** GGUF metadata KV entries contain exposed private keys or API credentials.

**Detection Condition:** GGUF KV string value matches credential or private key regex.

**Remediation:** Revoke exposed tokens immediately and strip secrets from GGUF metadata.

---

### GU104 — Oversized Metadata Staging Buffer

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-400` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** A GGUF metadata string or array exceeds safe parsing length budget (1 MB).

**Detection Condition:** Metadata value byte length > max_gguf_value_bytes.

**Remediation:** Reduce metadata string lengths to conform to standard GGUF limits.

---

### GU105 — Suspicious URL in Metadata

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `credential_exposure` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-200` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1071.001: Application Layer Protocol: Web Protocols` |

**Description:** GGUF metadata contains unverified external network URLs.

**Detection Condition:** GGUF KV value contains an external HTTP/HTTPS URL.

**Remediation:** Verify all metadata URL references to ensure no C2 links are present.

---

### GU106 — Script Injection Tag in Metadata

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `code_execution` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-79` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059.007: Command and Scripting Interpreter: JavaScript` |

**Description:** GGUF metadata contains embedded HTML/JavaScript injection tags.

**Detection Condition:** GGUF KV value contains <script> or javascript: tags.

**Remediation:** Sanitize metadata values to remove HTML/script tags.

---

## GGUF — Structural Anomalies (GU2xx)

### GU201 — GGUF Resource Limit Exceeded

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-400` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** GGUF KV count or tensor count exceeds maximum safe parsing limits.

**Detection Condition:** KV count > max_gguf_kv_count or tensor count > 100,000.

**Remediation:** Ensure model KV counts and tensor counts are within specification budgets.

---

### GU202 — Invalid Magic or Unsupported GGUF Version

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `format_integrity` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** GGUF file has invalid magic bytes or declares an unsupported format version.

**Detection Condition:** Magic != 'GGUF' or version not in (1, 2, 3).

**Remediation:** Convert model to standard GGUF v2/v3 format using official llama.cpp tools.

---

### GU203 — Truncated or Malformed GGUF Stream

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** GGUF binary stream is truncated or descriptor arguments reach unexpected EOF.

**Detection Condition:** Stream read fails due to unexpected EOF during header unpacking.

**Remediation:** Re-download model file or verify disk storage integrity.

---

### GU204 — Invalid Metadata Value Type

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** GGUF metadata KV specifies an undefined or unrecognized GGUFValueType enum.

**Detection Condition:** Value type ID not in standard GGUFValueType range [0..12].

**Remediation:** Ensure all metadata KV types adhere to the canonical GGUF specification.

---

### GU205 — Invalid or Duplicate Tensor Name

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** GGUF tensor name is empty, duplicate, or contains invalid non-UTF-8 bytes.

**Detection Condition:** Tensor name is duplicate or cannot be decoded as valid UTF-8.

**Remediation:** Ensure tensor names are strictly unique and valid UTF-8 strings.

---

### GU206 — Invalid Tensor Dimensions

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-190` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1499: Endpoint Denial of Service` |

**Description:** A GGUF tensor declares a dimension count above the scanner's rank limit (16), a negative dimension, or a shape whose element count overflows while its byte size is computed.

**Detection Condition:** Tensor rank > 16, any declared dimension is negative, or tensor byte-size computation raises an overflow.

**Remediation:** Ensure tensor dimension counts and ranks match standard GGML specifications.

---

### GU207 — Invalid Tensor Type

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** GGUF tensor specifies an unrecognized or invalid GGMLType enum value.

**Detection Condition:** GGML tensor type enum ID not in recognized type table.

**Remediation:** Use standard GGML quantization or floating-point tensor types.

---

### GU208 — Invalid Tensor Offset or Alignment

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-125` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** GGUF tensor data offset is negative or violates alignment boundary.

**Detection Condition:** Offset < 0 or alignment not a power of two.

**Remediation:** Align tensor data offsets to standard 32-byte boundaries.

---

### GU209 — Tensor Range Outside File Bounds

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-125` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1190: Exploit Public-Facing Application` |

**Description:** GGUF tensor data span exceeds physical file size bounds.

**Detection Condition:** Tensor data section start + byte size > total file length.

**Remediation:** Verify file integrity and ensure the binary was not truncated during transfer.

---

### GU210 — Overlapping Tensor Data Ranges

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `GGUF, GGML` |
| **CWE** | `CWE-125` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** Two or more GGUF tensor descriptors declare overlapping byte ranges.

**Detection Condition:** Tensor byte range [offset_i, offset_i + size_i) overlaps with another.

**Remediation:** Re-serialize model tensors to ensure all data spans are disjoint.

---

## Cross-Format Integrity (G0xx)

### G001 — Format Mismatch / Extension Spoofing

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `format_integrity` |
| **Affected Format** | `ALL` |
| **CWE** | `CWE-436`, `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1036.007: Masquerading: Double File Extension` |

**Description:** File extension indicates one model format (e.g. .safetensors), but internal magic byte sniffing detects a different, potentially executable format (e.g. Pickle).

**Detection Condition:** Header magic format conflicts with declared file extension format.

**Remediation:** Ensure file extension accurately reflects underlying binary format.

---

### G002 — Format Identified — No Analyzer Available

| Field | Value |
|-------|-------|
| **Severity** | ⚪ `INFO` |
| **Category** | `format_integrity` |
| **Affected Format** | `ALL` |
| **CWE** | `CWE-693` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development Tools` |

**Description:** The artifact was identified by magic bytes as a known ML model format (e.g. TFLite, NumPy NPY/NPZ, Flax MessagePack) but this scanner has no registered analyzer for that format.  The artifact was not inspected; security coverage is incomplete. Note: ONNX, HDF5, and Keras formats DO have registered analyzers and will not fire this rule.

**Detection Condition:** Magic-byte sniffing identifies a recognized but unanalyzed format (NumPy NPY/NPZ, TFLite, Flax MessagePack). No analyzer is registered, so no rules ran against the artifact.

**Remediation:** Verify the artifact using a format-specific tool.  Do not assume a clean result from this scanner alone.

---

### G005 — Shard Declared in Index but Missing on Disk

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** The model index (model.safetensors.index.json, pytorch_model.bin.index.json, etc.) declares a shard file that is not present on disk.

**Detection Condition:** A shard filename listed in the index JSON weight_map is absent from the directory.

**Remediation:** Re-download all declared shards from the original source. A missing shard indicates an incomplete or tampered download.

---

### G006 — Undeclared Shard File Present

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** A shard file with a weight-file extension exists in the model directory but is not declared in any index file.

**Detection Condition:** A weight-extension file found on disk is not listed in the index JSON weight_map.

**Remediation:** Verify whether the undeclared shard file is legitimate. Undeclared shards may indicate unauthorized additions.

---

### G007 — Shard Size Disagrees with Index

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** The on-disk byte size of a shard file does not match the size recorded in the model index file.

**Detection Condition:** os.path.getsize(shard) != index_metadata['size_bytes'] for a declared shard.

**Remediation:** Re-download the affected shard from the original source. A size mismatch may indicate truncation or tampering.

---

### G008 — Shard Count Mismatch

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** The number of shards on disk does not match the number declared in the model index.

**Detection Condition:** len(on_disk_shards) != len(index_declared_shards) for a multi-shard model.

**Remediation:** Verify the shard count against the original repository. A count mismatch may indicate missing or extra shards.

---

## Supply Chain — Config & Code Execution (SC1xx)

### SC101 — Remote Code Execution via auto_map / Custom Class Reference

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-502`, `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |

**Description:** The model config.json contains an auto_map entry that references a Python class or module.  When loaded with trust_remote_code=True, HuggingFace Transformers will execute that code automatically.

**Detection Condition:** auto_map key present in config.json with a non-empty value.

**Remediation:** Inspect the referenced Python module before loading the model. Avoid loading models with custom code unless you trust the source.

---

### SC102 — trust_remote_code Enabled

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204.002: User Execution: Malicious File` |

**Description:** The model repository config declares trust_remote_code=True, instructing frameworks to execute bundled Python files without further confirmation.

**Detection Condition:** trust_remote_code field is True in config.json or tokenizer_config.json.

**Remediation:** Do not set trust_remote_code=True without auditing all bundled Python files. Prefer models without custom code requirements.

---

### SC103 — Bundled Executable Python Code

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204.002: User Execution: Malicious File` |

**Description:** One or more Python source files (.py) are bundled alongside the model weights. These files may be executed when the model is loaded via HuggingFace Transformers.

**Detection Condition:** A .py file with a name matching common model-code patterns (modeling_*.py, tokenization_*.py, etc.) exists in the repository root.

**Remediation:** Audit all bundled Python files before loading the model. Run a static AST scan and code review on modeling_*.py and similar files.

---

### SC104 — Dangerous System Call in Bundled Code

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-78`, `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |

**Description:** Static AST analysis of a bundled Python file found a call to a dangerous system API (os.system, subprocess.Popen, eval, exec, etc.).

**Detection Condition:** AST visitor finds a Call node whose function resolves to a blocklisted system/exec call in a bundled .py file.

**Remediation:** Do not load this model without auditing the bundled Python code. The dangerous call may execute arbitrary system commands.

---

### SC105 — Network Access in Bundled Code

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-918` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1071: Application Layer Protocol` |

**Description:** Static AST analysis of a bundled Python file found an import or call to a network-access module (urllib, requests, socket, httpx, etc.).

**Detection Condition:** AST visitor finds an import of or call to a blocklisted network module in a bundled .py file.

**Remediation:** Audit the bundled Python code for unintended network access. Network calls in model-loading code may exfiltrate data or download payloads.

---

### SC106 — Unsigned Artifact — No Signature File Found

| Field | Value |
|-------|-------|
| **Severity** | ⚪ `INFO` |
| **Category** | `provenance` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1553: Subvert Trust Controls` |

**Description:** No cryptographic signature file (.sig, .asc, sha256sum.txt, checksums.txt) was found alongside the model weight artifacts.

**Detection Condition:** None of the standard signature/checksum filenames exist in the repository root.

**Remediation:** Consider verifying artifact provenance via a signed release or Sigstore. An unsigned artifact does not guarantee authenticity.

---

### SC107 — Hash Self-Inconsistency (Possible Tampering)

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `provenance` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494`, `CWE-345` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1565: Data Manipulation` |

**Description:** The SHA-256 hash of an artifact on disk does not match the hash declared in the repository's index or metadata file.

**Detection Condition:** sha256(artifact_bytes) != declared_hash from model index or metadata.

**Remediation:** Do not use this artifact — the SHA-256 hash does not match the index declaration. Re-download from a trusted source and verify again.

---

### SC108 — LFS Pointer Stub — Content Not Fetched

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `provenance` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** A model weight file contains a Git LFS pointer stub rather than actual weight data.  The real content was never fetched from LFS storage.

**Detection Condition:** File begins with 'version https://git-lfs.github.com/spec/v1' and size < 512 bytes — indicates an unfetched LFS pointer.

**Remediation:** Run `git lfs pull` to fetch actual weight content before scanning or deploying. An LFS stub contains only metadata, not real model weights.

---

### SC109 — Config Declared Format Disagrees with Detected Format

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `provenance` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-436` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1036: Masquerading` |

**Description:** The model_type or library_name declared in config.json does not match the format detected from the model weight file magic bytes.

**Detection Condition:** library_name in config.json implies a different weight format than the format detected from on-disk weight file headers.

**Remediation:** Verify that the model format matches the declared library. Inconsistency may indicate a mislabelled or repackaged artifact.

---

## Supply Chain — Dependency Security (SC2xx)

### SC201 — Unpinned Dependency

| Field | Value |
|-------|-------|
| **Severity** | 🔵 `LOW` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-1104` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development Tools` |

**Description:** A dependency declared in requirements.txt, setup.cfg, or pyproject.toml uses a range specifier (>=, ~=, *, <, !=) rather than an exact pin (==).

**Detection Condition:** Dependency specifier does not contain '==' or has only a bare package name.

**Remediation:** Pin all dependencies to exact versions (==) in deployment artifacts. Unpinned dependencies allow silent vulnerability introduction.

---

### SC202 — Dependency with Known Advisory

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-1104` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development Tools` |

**Description:** A declared dependency matches a package version range that has a known security advisory in the bundled advisory snapshot.

**Detection Condition:** Package name and version specifier overlap with an entry in the local advisory snapshot under data/advisories/.

**Remediation:** Upgrade the affected dependency to a version that does not have a known advisory. Check the advisory for affected version ranges and patches.

---

### SC203 — Direct VCS/URL Dependency

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-494` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development Tools` |

**Description:** A dependency is specified as a direct VCS URL (git+https://, svn+, hg+) or a direct HTTP/HTTPS URL rather than a versioned PyPI package.

**Detection Condition:** Requirement line starts with git+, svn+, hg+, bzr+, or contains @ http(s)://.

**Remediation:** Replace direct VCS/URL dependencies with pinned PyPI versions. Direct VCS deps bypass the PyPI registry security model.

---

### SC204 — Possible Typosquatting (Edit Distance 1)

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-1104` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development Tools` |

**Description:** A declared dependency name has an edit distance of 1 from a well-known popular package, which may indicate a typosquatting attack.

**Detection Condition:** Levenshtein distance between package name and a popular package name is 1.

**Remediation:** Verify the package name is correct and not a typosquatting package. Compare carefully against the intended popular package name.

---

## Correlated Cross-Signal Rules (C0xx)

### C001 — Active Supply Chain Execution Intent

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `supply_chain` |
| **Affected Format** | `REPOSITORY` |
| **CWE** | `CWE-94`, `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204.002: User Execution: Malicious File` |

**Description:** The repository contains bundled executable code AND the config explicitly instructs the ML framework to execute it (trust_remote_code=True). This elevates bundled code from a latent risk to an active threat vector.

**Detection Condition:** SC102 (trust_remote_code) is present AND either SC101 (auto_map) or SC103 (Bundled Code) is present in the repository.

**Remediation:** Do not load this model unless you fully trust the publisher. Review the bundled code for malicious behavior before instantiation.

---

### C002 — Networked Executable Payload

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `ALL` |
| **CWE** | `CWE-78`, `CWE-918` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1105: Ingress Tool Transfer` |

**Description:** The model exhibits both arbitrary execution capabilities and network access primitives. This strongly indicates a dropper, reverse shell, or C2 beacon.

**Detection Condition:** (P101 (Dangerous Import) or SC104 (System Call)) AND SC105 (Network Access).

**Remediation:** Reject this model immediately. It contains an active network-connected code execution payload.

---

### C003 — Obfuscated Payload Staging

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `PYTORCH_ZIP` |
| **CWE** | `CWE-434`, `CWE-693` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** Nested archives combined with dynamic imports or suspicious reductions strongly indicate deliberate evasion and payload staging.

**Detection Condition:** P210 (Nested Archive) AND (P103 (STACK_GLOBAL) OR P105 (Suspicious REDUCE)).

**Remediation:** Avoid unpacking nested archives from unverified sources. Convert model to SafeTensors format if legitimate.

---

### C004 — Imported and Invoked Dangerous Callable

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `RAW_PICKLE, PYTORCH_ZIP` |
| **CWE** | `CWE-502`, `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1204.002: User Execution: Malicious File` |

**Description:** The same dangerous callable is both imported by a GLOBAL/STACK_GLOBAL opcode and invoked by a REDUCE opcode within one Pickle stream. Import alone can be incidental; import followed by invocation is a completed code-execution chain.

**Detection Condition:** A P101/P102 finding and a P104/P105 finding share the same matched_value and archive member within a single artifact.

**Remediation:** Reject this model. The Pickle stream does not merely reference a dangerous callable, it calls it during deserialization.

---

## Cross-Format Weight Integrity (WI1xx)

### WI101 — NaN Values Detected in Tensor Weights

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `weight_integrity` |
| **Affected Format** | `SAFETENSORS, GGUF, ONNX, HDF5, KERAS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML03:2023 Model Inversion Attack |
| **MITRE ATT&CK** | `T1565: Data Manipulation` |

**Description:** One or more tensor weight values are NaN (Not a Number). NaN weights indicate corrupt training, deliberate tampering, or a poisoned model that may produce unpredictable outputs.

**Detection Condition:** Float tensor bytes contain IEEE 754 NaN bit patterns.

**Remediation:** Re-acquire the model from a trusted source and verify its checksum. NaN weights may indicate model corruption or tampering.

---

### WI102 — Infinite Values Detected in Tensor Weights

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `weight_integrity` |
| **Affected Format** | `SAFETENSORS, GGUF, ONNX, HDF5, KERAS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML03:2023 Model Inversion Attack |
| **MITRE ATT&CK** | `T1565: Data Manipulation` |

**Description:** One or more tensor weight values are positive or negative infinity. Infinite weights indicate corrupt training, numerical overflow, or deliberate tampering that degrades model behavior.

**Detection Condition:** Float tensor bytes contain IEEE 754 Inf bit patterns.

**Remediation:** Re-acquire the model from a trusted source and verify its checksum. Infinite weights indicate numerical corruption or tampering.

---

### WI103 — Extreme Finite Values Detected in Tensor Weights

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `weight_integrity` |
| **Affected Format** | `SAFETENSORS, GGUF, ONNX, HDF5, KERAS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML03:2023 Model Inversion Attack |
| **MITRE ATT&CK** | `T1565: Data Manipulation` |

**Description:** One or more tensor weight values are at the maximum finite representable magnitude for their dtype (e.g., F32 exponent == 0xFE, meaning |value| ≥ 1.7e38). Values this close to overflow suggest numerical instability, gradient explosion during training, or deliberate weight manipulation.

**Detection Condition:** Float tensor bytes have exponent bits equal to the maximum finite exponent for the dtype (F32/BF16: 0xFE; F16: 0x1E; F64: 0x7FE).

**Remediation:** Investigate the origin of extreme-magnitude weights. Values at the maximum finite representable magnitude suggest numerical instability, overflow during training, or tampering. Re-acquire from a trusted source and verify with the model provider.

---

## ONNX Security Rules (ON1xx)

### ON101 — External or Custom Operator in ONNX Graph

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `ONNX` |
| **CWE** | `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** The ONNX graph references an operator from a non-standard domain (not 'ai.onnx' or ''). External and custom operators may execute arbitrary native code at inference time and should be carefully audited before deployment.

**Detection Condition:** NodeDef.domain is not '' or 'ai.onnx'.

**Remediation:** Audit the custom operator implementation for malicious behavior. Prefer models using only standard ONNX operators. Do not deploy models with unaudited custom operators in production.

---

### ON102 — Malformed ONNX Graph Structure

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `ONNX` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** The ONNX model graph is structurally invalid: it may contain missing graph definitions, empty node lists in a non-trivial model, or protobuf fields indicating structural corruption.

**Detection Condition:** ONNX graph proto is missing or structurally invalid.

**Remediation:** Re-export the model from a trusted ONNX-compatible framework. Use onnx.checker.check_model() to validate before deployment.

---

### ON103 — Path Traversal in ONNX External Data Location

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `ONNX` |
| **CWE** | `CWE-22` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1083: File and Directory Discovery` |

**Description:** An ONNX initializer references external tensor data at a location containing path-traversal sequences (e.g. '../'). When ONNX Runtime loads such a model, it opens the specified path relative to the model file's directory, allowing an attacker to read arbitrary files on the host filesystem (e.g. /etc/hosts, /etc/passwd, Windows credential stores).

**Detection Condition:** TensorProto.external_data[location] contains '..' component or is an absolute path (starts with '/' or a drive letter).

**Remediation:** Do not load this model with ONNX Runtime. Re-acquire the model from a trusted source and verify that all initializer external_data locations are relative paths without traversal sequences.

---

### ON104 — External Data Offset/Length Overflow

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `ONNX` |
| **CWE** | `CWE-190` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1190: Exploit Public-Facing Application` |

**Description:** An ONNX initializer's external_data segment declares an offset and length whose sum reaches or exceeds 2^64, producing a 64-bit integer wraparound when the addition is performed in uint64 arithmetic. Runtimes that compute bounds checks as 'm_offset + m_data_length > file_size' in uint64 will evaluate the wrapped (near-zero) sum against file_size, silently passing the check and subsequently dereferencing a wild pointer at the attacker-controlled wrapped address. Confirmed exploitable in OpenVINO 2025.4.1.

**Detection Condition:** TensorProto.external_data 'offset' (integer) + 'length' (integer) >= 2^64. Detection uses Python arbitrary-precision integer arithmetic — there is no silent overflow in the scanner itself. Malformed or non-integer values are skipped safely.

**Remediation:** Do not load this model with any ONNX runtime. Re-export the model from a trusted ONNX-compatible framework with correctly bounded external_data offset and length values. Verify that no initializer offset + length reaches or exceeds 2^64.

---

## TensorFlow SavedModel Security Rules (TF1xx)

### TF101 — External or Custom Op in TensorFlow Graph

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `TF_SAVEDMODEL` |
| **CWE** | `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** The TensorFlow SavedModel graph references an operation that is not in the standard TensorFlow op set. Custom ops may execute arbitrary native code at inference time and require careful auditing.

**Detection Condition:** NodeDef.op not in standard TF op set.

**Remediation:** Audit the custom op implementation for malicious behavior. Prefer models using only standard TensorFlow ops. Do not deploy models with unaudited custom ops in production.

---

### TF102 — Malformed TensorFlow SavedModel

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `TF_SAVEDMODEL` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** The saved_model.pb file is malformed, truncated, or cannot be parsed as a valid TensorFlow SavedModel protobuf. This may indicate corruption or deliberate tampering.

**Detection Condition:** saved_model.pb fails protobuf parsing or has invalid structure.

**Remediation:** Re-export the model from TensorFlow. Verify the file is not truncated and re-acquire from a trusted source.

---

### TF103 — Path Traversal Indicator in SavedModel Asset

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `tampering` |
| **Affected Format** | `TF_SAVEDMODEL` |
| **CWE** | `CWE-22` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1083: File and Directory Discovery` |

**Description:** The SavedModel asset directory contains a filename with a path-traversal indicator such as '../' or a URL-encoded traversal sequence (%2e%2e, %2f, %5c), or an absolute path. When TensorFlow restores such an asset it may write to or read from a location outside the model directory, enabling arbitrary file overwrite or disclosure on the host filesystem.

**Detection Condition:** An entry in the sibling assets/ directory has a filename containing a '..' component, a URL-encoded traversal token, or an absolute path. Detection is filename-string only; the asset file is never accessed.

**Remediation:** Do not load this SavedModel. Re-acquire from a trusted source and verify that all asset filenames are plain relative names without traversal sequences.

---

## Keras Security Rules (KR1xx)

### KR101 — External or Custom Class in Keras Config

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `HDF5, KERAS` |
| **CWE** | `CWE-94` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1195: Supply Chain Compromise` |

**Description:** The Keras model configuration references a layer or class that is not in the standard Keras built-in class set. Custom classes may execute arbitrary Python code when the model is loaded and should be audited before deployment.

**Detection Condition:** Layer class_name not in standard keras built-in class set.

**Remediation:** Audit the custom class implementation for malicious behavior. Do not load models with custom classes using trust settings that execute arbitrary code. Inspect the class before deployment.

---

### KR102 — Malformed Keras Model Config or Structure

| Field | Value |
|-------|-------|
| **Severity** | 🟡 `MEDIUM` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `HDF5, KERAS` |
| **CWE** | `CWE-20` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** The Keras model configuration JSON is malformed, missing, or contains invalid structure. This may indicate corruption, truncation, or a tampered model file.

**Detection Condition:** Keras config JSON is missing, invalid, or structurally malformed.

**Remediation:** Re-export the model using tf.keras or standalone Keras. Verify the file is not truncated and re-acquire from a trusted source.

---

### KR103 — Duplicate Archive Member in Keras ZIP

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `structural_anomaly` |
| **Affected Format** | `KERAS` |
| **CWE** | `CWE-694`, `CWE-693` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1027: Obfuscated Files or Information` |

**Description:** A Keras .keras ZIP archive contains more than one member with the same name. Python's zipfile resolves a name to the LAST matching entry, while some scanners inspect the FIRST. This asymmetry is a known scanner-bypass technique: a benign first config.json can hide a malicious duplicate that the loader actually uses.

**Detection Condition:** zipfile.namelist() contains a duplicated member name (count > 1).

**Remediation:** Reject this archive. A well-formed .keras file has exactly one entry per name. Re-export the model with a trusted serializer and re-acquire from a trusted source.

---

### KR104 — HDF5 External / Virtual Dataset Reference

| Field | Value |
|-------|-------|
| **Severity** | 🟠 `HIGH` |
| **Category** | `code_execution` |
| **Affected Format** | `HDF5, KERAS` |
| **CWE** | `CWE-73`, `CWE-200` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1083: File and Directory Discovery` |
| **CVE** | `CVE-2026-1669` |

**Description:** The HDF5 file contains an ExternalLink or a Virtual Dataset (VDS) mapping that references a file path outside the model artifact. When a consumer opens the model, HDF5 will automatically follow the link and read data from the referenced external file. An attacker can exploit this to exfiltrate host data, trigger SSRF, or load a malicious payload from a controlled path — without the consumer's explicit consent.

**Detection Condition:** h5py.Group.get(key, getlink=True) returns h5py.ExternalLink, or dataset.is_virtual is True and virtual_sources() contains a source whose file_name is not '.' (i.e. references an external file).

**Remediation:** Do not load this model file. HDF5 external links and virtual datasets that reference arbitrary host paths are a path-traversal and data exfiltration vector. Reject the artifact and re-acquire from a trusted source that does not embed external references. If external datasets are required by design, restrict the HDF5 library's allowed paths using the h5py SWMR/external-link guard settings and validate all referenced paths against an allowlist before opening.

---

### KR105 — Keras Lambda Bytecode Payload

| Field | Value |
|-------|-------|
| **Severity** | 🔴 `CRITICAL` |
| **Category** | `code_execution` |
| **Affected Format** | `HDF5, KERAS` |
| **CWE** | `CWE-94`, `CWE-502` |
| **OWASP ML** | ML06:2023 AI Supply Chain Attacks |
| **MITRE ATT&CK** | `T1059: Command and Scripting Interpreter` |
| **CVE** | `CVE-2024-3660` |

**Description:** A Keras Lambda layer's serialized bytecode contains indicators of dangerous execution primitives: OS command execution (os.system, subprocess, popen), dynamic code evaluation (eval, exec), or dynamic module import (__import__). Keras serializes Lambda functions as base64-encoded Python marshal bytecode. When the model is loaded, Keras deserializes and executes this code, making a malicious Lambda an arbitrary-code-execution vector.

**Detection Condition:** The 'code' field inside a '__lambda__' config node, after base64 decoding, contains one or more of: subprocess, Popen, popen (CRITICAL); or the combination os+system, nt+system, posix+system as standalone identifiers (CRITICAL); or __import__, eval, exec as standalone identifiers (HIGH). All matches use word-boundary checks to prevent partial-name false positives. No marshal.loads(), exec(), or eval() is called during detection.

**Remediation:** Do not load this model. A Keras Lambda layer's serialized bytecode contains indicators of dangerous execution primitives (OS command execution, dynamic import, or eval/exec). Reject the artifact and re-acquire from a trusted source. If the Lambda is required, audit the source function and re-serialize from verified, reviewed code only.

---

_Generated automatically by `scripts/generate_rule_catalog.py`._
_Do not edit manually — update `src/model_scanner/rules/catalog.py` and regenerate._