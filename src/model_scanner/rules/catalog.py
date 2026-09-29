"""
model_scanner.rules.catalog
===========================

Centralized, immutable security rule catalog for pre-trained AI model formats
(Phase 8). Registers all 86 canonical static security analysis rules.

Relationship to the analyzers
-----------------------------
Analyzers do **not** read this catalog at scan time — each one carries its own
``_RULE_METADATA`` / ``_RULE_SEVERITY`` / ``_RULE_NAMES`` table and emits
findings from that. This catalog is the documentation surface: it backs the
Rule Catalog view in the Web UI, the ``docs/rule_catalog.md`` reference, and the
Impact / Why-this-matters / CWE / OWASP / MITRE fields of every generated
report.

Because there are two tables, they can drift, and a catalog that contradicts
the findings on screen is worse than no catalog. ``severity``, ``name`` and
``category`` here therefore mirror exactly what the analyzers emit, and
``tests/unit/test_catalog_analyzer_sync.py`` fails the build if they ever
diverge again. ``severity`` is the *default* an analyzer assigns; several rules
escalate it per finding (e.g. P103/P107 go HIGH on a blocklisted symbol) and
strict mode escalates further in ``core/normalizer.py``.

``cve_ids`` is populated only for rules that map to a verified, published CVE.
OWASP identifiers follow the OWASP Machine Learning Security Top Ten **2023**
edition.
"""

from __future__ import annotations

from model_scanner.core.models import Severity
from model_scanner.rules.models import SecurityRule

_CANONICAL_RULES: tuple[SecurityRule, ...] = (
    # ---------------------------------------------------------------------------
    # P1xx: Pickle Opcode & Execution Rules (7 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="P101",
        name="Dangerous Global Import",
        severity=Severity.CRITICAL,
        category="code_execution",
        description=(
            "The Pickle stream imports a known dangerous module or callable capable of "
            "arbitrary code execution (e.g. os.system, subprocess.Popen)."
        ),
        detection_condition=(
            "GLOBAL or STACK_GLOBAL opcode targets a blocklisted callable."
        ),
        remediation=(
            "Do not deserialize this model. Re-serialize model weights using SafeTensors."
        ),
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204.002: User Execution: Malicious File",
    ),
    SecurityRule(
        rule_id="P102",
        name="Suspicious Global Import",
        severity=Severity.HIGH,
        category="code_execution",
        description=(
            "The Pickle stream imports a dynamic loader, reflection, or native FFI binding symbol."
        ),
        detection_condition=(
            "GLOBAL opcode references dynamic loader or reflection modules."
        ),
        remediation=(
            "Audit referenced global symbols and verify if necessary for model inference."
        ),
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="P103",
        name="STACK_GLOBAL Dynamic Import",
        severity=Severity.MEDIUM,
        category="code_execution",
        description=(
            "The Pickle stream uses STACK_GLOBAL opcode to import symbols "
            "constructed dynamically."
        ),
        detection_condition=(
            "STACK_GLOBAL opcode resolves to a symbol that is not on the SAFE allowlist. HIGH when "
            "the symbol is blocklisted, MEDIUM otherwise; the normalizer escalates MEDIUM to HIGH "
            "in strict mode."
        ),
        remediation=(
            "Audit stack-constructed symbols and convert model weights to SafeTensors."
        ),
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502", "CWE-94"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="P104",
        name="Dangerous REDUCE Invocation",
        severity=Severity.CRITICAL,
        category="code_execution",
        description=(
            "A REDUCE operation targets a callable capable of executing system commands."
        ),
        detection_condition=(
            "REDUCE opcode invokes a blocklisted callable, or a runpy dynamic-execution primitive "
            "that is escalated from SUSPICIOUS because it is actually called."
        ),
        remediation=(
            "Reject this model immediately. It contains an arbitrary code execution payload."
        ),
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502", "CWE-94"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204.002: User Execution: Malicious File",
    ),
    SecurityRule(
        rule_id="P105",
        name="Suspicious REDUCE Invocation",
        severity=Severity.HIGH,
        category="code_execution",
        description="A REDUCE operation targets an unverified or dynamic callable.",
        detection_condition=(
            "REDUCE opcode invokes a callable on the SUSPICIOUS list (dynamic loader, reflection, "
            "or FFI binding). An UNKNOWN callable does not fire this rule."
        ),
        remediation=(
            "Inspect the callable target and replace with standard state dictionary weights."
        ),
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="P106",
        name="Suspicious BUILD State Mutation",
        severity=Severity.MEDIUM,
        category="code_execution",
        description="A BUILD operation mutates object state with suspicious parameters.",
        detection_condition=(
            "BUILD opcode modifies a deserialization-critical state key "
            "(__setstate__, __class__, __globals__, ...) or targets a non-SAFE global. "
            "Ordinary module attribute names are not reported."
        ),
        remediation=(
            "Verify object state dictionary and ensure only tensor weights are reconstructed."
        ),
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="P107",
        name="Suspicious NEWOBJ Instantiation",
        severity=Severity.MEDIUM,
        category="code_execution",
        description="A NEWOBJ/NEWOBJ_EX opcode instantiates an unverified class.",
        detection_condition=(
            "NEWOBJ or NEWOBJ_EX instantiates a class that is not on the SAFE globals allowlist. "
            "HIGH when the class is blocklisted, MEDIUM otherwise."
        ),
        remediation="Ensure only trusted tensor container classes are instantiated.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204: User Execution",
    ),
    # ---------------------------------------------------------------------------
    # P2xx: Pickle / PyTorch ZIP Structural Integrity Rules (11 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="P201",
        name="Malformed or Truncated Opcode Stream",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "The Pickle opcode stream is corrupted, truncated, or contains "
            "malformed arguments."
        ),
        detection_condition=(
            "Opcode argument parsing reaches premature EOF or invalid byte syntax."
        ),
        remediation="Inspect file integrity or re-download the model file.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="P202",
        name="Duplicate or Misplaced PROTO Header",
        severity=Severity.LOW,
        category="structural_anomaly",
        description=(
            "A PROTO opcode appears more than once or at an unexpected position in the stream."
        ),
        detection_condition="PROTO opcode encountered at non-zero offset or multiple times.",
        remediation="Clean up corrupted stream headers using standard serialization tools.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="P203",
        name="Unsupported Protocol Version",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The stream declares an invalid or unsupported Pickle protocol version (> 5)."
        ),
        detection_condition="PROTO opcode specifies protocol version > 5 or < 0.",
        remediation="Re-serialize the model using standard Pickle protocol 2, 4, or 5.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="P204",
        name="FRAME Boundary Desynchronization",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "An opcode crosses beyond declared FRAME boundary in Protocol 4/5 framing."
        ),
        detection_condition="Opcode span exceeds the current FRAME segment boundary.",
        remediation="Re-serialize with standard pickle tools to fix framing desynchronization.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-125",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1203: Exploitation for Client Execution",
    ),
    SecurityRule(
        rule_id="P205",
        name="Invalid Memo Slot Reference",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description="A GET or BINGET opcode references an unassigned memo slot index.",
        detection_condition="GET/BINGET opcode references a memo index that was never stored.",
        remediation="Verify that the model serialization was not tampered with or truncated.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="P206",
        name="Concatenated Pickle Streams",
        severity=Severity.LOW,
        category="structural_anomaly",
        description=(
            "Multiple valid Pickle streams delimited by STOP opcodes are "
            "concatenated in one file."
        ),
        detection_condition="Additional valid opcodes detected after an initial STOP opcode.",
        remediation="Ensure each model file contains exactly one discrete serialized payload.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-436",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="P207",
        name="Trailing Unparsed Bytes",
        severity=Severity.LOW,
        category="structural_anomaly",
        description="Unparsed trailing data detected after the final STOP opcode in the stream.",
        detection_condition=(
            "Non-opcode byte sequence exists after the terminating STOP opcode."
        ),
        remediation="Strip unparsed payload trailing bytes to prevent payload smuggling.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-436",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="P208",
        name="Excessive Opcode or Memo Resource Usage",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "Pickle stream exceeded maximum opcode count or memo entry ceiling (DoS guard)."
        ),
        detection_condition="Opcode count > max_opcodes or memo size > max_memo_slots.",
        remediation="Check model size or increase scanner resource ceilings if legitimate.",
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-400",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="P209",
        name="Embedded Pickle Stream in PyTorch ZIP Container",
        severity=Severity.INFO,
        category="format_integrity",
        description=(
            "A PyTorch ZIP archive contains an embedded Pickle stream member. "
            "This is expected for standard PyTorch checkpoints but is flagged for "
            "awareness; the stream is analysed for malicious opcodes separately."
        ),
        detection_condition=(
            "ZIP member has a Pickle magic-byte prefix (\\x80\\x02 or \\x80\\x04/\\x05) "
            "or passes the Protocol-0 heuristic."
        ),
        remediation=(
            "Inspect embedded Pickle opcodes for dangerous calls. "
            "Prefer migrating model weights to the SafeTensors format."
        ),
        affected_format="PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204.002: User Execution: Malicious File",
    ),
    SecurityRule(
        rule_id="P210",
        name="Nested Archive in PyTorch ZIP Container",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "A PyTorch ZIP archive contains a nested ZIP archive member. "
            "Legitimate model checkpoints do not embed nested archives; this pattern "
            "is associated with evasion or payload staging."
        ),
        detection_condition=(
            "A non-Pickle, non-tensor ZIP member starts with PK\\x03\\x04 magic bytes "
            "(nested archive, depth >= 2)."
        ),
        remediation=(
            "Inspect the nested archive contents. Reject artifacts from untrusted sources "
            "that contain unexpected nested archives."
        ),
        affected_format="PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-434", "CWE-693"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="P211",
        name="Extreme Compression Ratio (Zip-Bomb Indicator)",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "A ZIP member's declared uncompressed size is more than 1000x its compressed "
            "size, a hallmark of zip-bomb payloads. Decompressing this artifact could "
            "exhaust disk or memory resources."
        ),
        detection_condition=(
            "compress_size / file_size < 0.001 for any non-empty ZIP member."
        ),
        remediation=(
            "Reject or quarantine this archive. Verify artifact provenance before "
            "expanding."
        ),
        affected_format="PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-400", "CWE-770"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    # ---------------------------------------------------------------------------
    # ST1xx: SafeTensors Metadata Security Rules (5 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="ST101",
        name="Metadata Code Execution String",
        severity=Severity.CRITICAL,
        category="code_execution",
        description=(
            "SafeTensors metadata contains executable code or eval/exec command primitives."
        ),
        detection_condition="__metadata__ value matches known code execution regex pattern.",
        remediation="Remove executable code strings from the __metadata__ header dictionary.",
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-94",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="ST102",
        name="Path Traversal Sequence in Metadata",
        severity=Severity.HIGH,
        category="tampering",
        description=(
            "SafeTensors metadata contains directory traversal patterns such as '../' "
            "or absolute filesystem paths."
        ),
        detection_condition=(
            "__metadata__ values match a path traversal regex pattern "
            "(e.g. '../', '..\\\\', '%2e%2e%2f', '/etc/passwd'); "
            "OR any top-level tensor name (JSON key) contains the same patterns. "
            "Legitimate tensor names never contain filesystem traversal sequences."
        ),
        remediation=(
            "Remove filesystem path references from metadata and tensor names. "
            "SafeTensors __metadata__ must contain only plain informational strings; "
            "tensor names must not contain path-traversal sequences."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-22",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1083: File and Directory Discovery",
    ),
    SecurityRule(
        rule_id="ST103",
        name="Suspicious URL in Metadata",
        severity=Severity.MEDIUM,
        category="credential_exposure",
        description=(
            "SafeTensors metadata contains unverified external HTTP/HTTPS URLs or C2 endpoints."
        ),
        detection_condition="__metadata__ value contains an external URL string.",
        remediation=(
            "Verify referenced URLs and ensure no command-and-control links exist in metadata."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-200",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1071.001: Application Layer Protocol: Web Protocols",
    ),
    SecurityRule(
        rule_id="ST104",
        name="Credential or Secret Exposure in Metadata",
        severity=Severity.HIGH,
        category="credential_exposure",
        description=(
            "SafeTensors metadata contains hardcoded API keys, private tokens, or credentials."
        ),
        detection_condition="__metadata__ value matches credential or API key regex pattern.",
        remediation="Purge exposed credentials immediately and revoke any compromised tokens.",
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-798", "CWE-200"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1552.001: Unsecured Credentials: Credentials In Files",
    ),
    SecurityRule(
        rule_id="ST105",
        name="Script Injection Tag in Metadata",
        severity=Severity.HIGH,
        category="code_execution",
        description="SafeTensors metadata contains HTML script tags or XSS injection strings.",
        detection_condition="__metadata__ value contains <script>, <iframe>, or javascript: tags.",
        remediation=(
            "Sanitize metadata values to remove HTML and client-side script injection tags."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-79",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059.007: Command and Scripting Interpreter: JavaScript",
    ),
    # ---------------------------------------------------------------------------
    # ST2xx: SafeTensors Structural Integrity Rules (9 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="ST201",
        name="Header Size or JSON Token DoS",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "The SafeTensors JSON header exceeds the configured size budget, or the header "
            "exhausts the scanner's JSON token, nesting, or tensor-count ceilings while being "
            "parsed. Both are denial-of-service guards."
        ),
        detection_condition=(
            "Header length prefix > max_safetensors_header_mb (default 16 MB, hard maximum 128 "
            "MB), JSON token or nesting budget exhausted, or the number of tensor descriptors "
            "exceeds the sentinel limit."
        ),
        remediation=(
            "Re-export the model with a header within the configured budget (default 16 MB). Raise "
            "max_safetensors_header_mb only for an artifact whose provenance you trust."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-400",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="ST202",
        name="Tensor Data Range Overlap",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "Two or more tensor descriptors declare overlapping data offset byte spans."
        ),
        detection_condition="Tensor span [start_i, end_i) overlaps with [start_j, end_j).",
        remediation="Re-serialize tensors to ensure each tensor has a distinct byte span.",
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-125",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="ST203",
        name="Invalid Tensor Data Offsets",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "A tensor descriptor declares a structurally invalid data_offsets field: not a "
            "two-element list, not integers, negative, or with a start offset greater than its end "
            "offset."
        ),
        detection_condition=(
            "data_offsets is not a 2-element list, contains a non-integer, contains a negative "
            "value, or data_offsets[0] > data_offsets[1]."
        ),
        remediation=(
            "Ensure every tensor declares data_offsets as a forward, non-negative [start, end] "
            "integer pair. Re-serialize with a trusted serializer."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-125",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1190: Exploit Public-Facing Application",
    ),
    SecurityRule(
        rule_id="ST204",
        name="Invalid Tensor dtype",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "A tensor descriptor's dtype is not a string, or is not one of the dtypes defined by "
            "the SafeTensors specification."
        ),
        detection_condition=(
            "dtype is not a string, or dtype is not in the standard SafeTensors type table."
        ),
        remediation="Use standard SafeTensors data types (e.g. F32, F16, BF16, I64, I32, BOOL).",
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="ST205",
        name="Invalid Tensor shape",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "A tensor's shape is not a list, or contains an entry that is not an integer or that "
            "is negative."
        ),
        detection_condition=(
            "shape is not a list, or any shape[i] is not an integer or is negative."
        ),
        remediation=(
            "Ensure every tensor shape is a list of non-negative integers. Re-serialize the model "
            "with a trusted serializer."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-190",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="ST206",
        name="Missing Required Tensor Fields",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "A tensor entry in the header is not a JSON object, or is missing one of the required "
            "descriptor fields (dtype, shape, data_offsets)."
        ),
        detection_condition=(
            "Tensor descriptor is not a JSON object, or the keys 'dtype', 'shape', or "
            "'data_offsets' are absent from it."
        ),
        remediation=(
            "Ensure all tensor entries adhere to the SafeTensors JSON schema specification."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="ST207",
        name="Duplicate Tensor Name",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description="SafeTensors header contains duplicate tensor dictionary keys.",
        detection_condition="Duplicate tensor name detected during streaming token evaluation.",
        remediation="Ensure tensor names are strictly unique within the header JSON object.",
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="ST208",
        name="Tensor Range Outside File Bounds",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "A tensor's declared data_offsets end lies beyond the end of the SafeTensors data "
            "section, so the descriptor claims bytes the file does not contain."
        ),
        detection_condition=(
            "data_offsets[1] > the size of the data section (file size - 8 - header length)."
        ),
        remediation=(
            "Re-acquire the file from a trusted source and verify its checksum. A tensor range "
            "running past the end of the file indicates truncation or a hand-edited header."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-125",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="ST209",
        name="Truncated or Unreadable Header",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "The SafeTensors file is truncated or its header is unreadable. "
            "The file may be corrupted, only partially downloaded, or deliberately malformed."
        ),
        detection_condition=(
            "Header length prefix missing, JSON parse error, or file shorter than "
            "8 + header_len bytes."
        ),
        remediation=(
            "Re-acquire the file from a trusted source. Verify checksum after download."
        ),
        affected_format="SAFETENSORS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    # ---------------------------------------------------------------------------
    # GU1xx: GGUF Metadata Security Rules (6 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="GU101",
        name="Metadata Code Injection",
        severity=Severity.CRITICAL,
        category="code_execution",
        description="GGUF metadata KV entry contains executable script or shell command strings.",
        detection_condition="GGUF KV string value matches command injection regex.",
        remediation="Remove executable code primitives from model metadata KV pairs.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-94",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="GU102",
        name="Path Traversal Sequence in Metadata",
        severity=Severity.HIGH,
        category="tampering",
        description="GGUF metadata contains directory traversal patterns (e.g. ../ or /etc/).",
        detection_condition="GGUF KV value contains '../' or absolute filesystem paths.",
        remediation="Sanitize metadata paths to prevent arbitrary file overwrite or traversal.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-22",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1083: File and Directory Discovery",
    ),
    SecurityRule(
        rule_id="GU103",
        name="Metadata Credential Exposure",
        severity=Severity.HIGH,
        category="credential_exposure",
        description="GGUF metadata KV entries contain exposed private keys or API credentials.",
        detection_condition="GGUF KV string value matches credential or private key regex.",
        remediation="Revoke exposed tokens immediately and strip secrets from GGUF metadata.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-798",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1552.001: Unsecured Credentials: Credentials In Files",
    ),
    SecurityRule(
        rule_id="GU104",
        name="Oversized Metadata Staging Buffer",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description="A GGUF metadata string or array exceeds safe parsing length budget (1 MB).",
        detection_condition="Metadata value byte length > max_gguf_value_bytes.",
        remediation="Reduce metadata string lengths to conform to standard GGUF limits.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-400",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="GU105",
        name="Suspicious URL in Metadata",
        severity=Severity.MEDIUM,
        category="credential_exposure",
        description="GGUF metadata contains unverified external network URLs.",
        detection_condition="GGUF KV value contains an external HTTP/HTTPS URL.",
        remediation="Verify all metadata URL references to ensure no C2 links are present.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-200",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1071.001: Application Layer Protocol: Web Protocols",
    ),
    SecurityRule(
        rule_id="GU106",
        name="Script Injection Tag in Metadata",
        severity=Severity.HIGH,
        category="code_execution",
        description="GGUF metadata contains embedded HTML/JavaScript injection tags.",
        detection_condition="GGUF KV value contains <script> or javascript: tags.",
        remediation="Sanitize metadata values to remove HTML/script tags.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-79",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059.007: Command and Scripting Interpreter: JavaScript",
    ),
    # ---------------------------------------------------------------------------
    # GU2xx: GGUF Structural Integrity Rules (10 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="GU201",
        name="GGUF Resource Limit Exceeded",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description="GGUF KV count or tensor count exceeds maximum safe parsing limits.",
        detection_condition="KV count > max_gguf_kv_count or tensor count > 100,000.",
        remediation="Ensure model KV counts and tensor counts are within specification budgets.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-400",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="GU202",
        name="Invalid Magic or Unsupported GGUF Version",
        severity=Severity.HIGH,
        category="format_integrity",
        description="GGUF file has invalid magic bytes or declares an unsupported format version.",
        detection_condition="Magic != 'GGUF' or version not in (1, 2, 3).",
        remediation="Convert model to standard GGUF v2/v3 format using official llama.cpp tools.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="GU203",
        name="Truncated or Malformed GGUF Stream",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "GGUF binary stream is truncated or descriptor arguments reach unexpected EOF."
        ),
        detection_condition="Stream read fails due to unexpected EOF during header unpacking.",
        remediation="Re-download model file or verify disk storage integrity.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="GU204",
        name="Invalid Metadata Value Type",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "GGUF metadata KV specifies an undefined or unrecognized GGUFValueType enum."
        ),
        detection_condition="Value type ID not in standard GGUFValueType range [0..12].",
        remediation="Ensure all metadata KV types adhere to the canonical GGUF specification.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="GU205",
        name="Invalid or Duplicate Tensor Name",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description="GGUF tensor name is empty, duplicate, or contains invalid non-UTF-8 bytes.",
        detection_condition="Tensor name is duplicate or cannot be decoded as valid UTF-8.",
        remediation="Ensure tensor names are strictly unique and valid UTF-8 strings.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="GU206",
        name="Invalid Tensor Dimensions",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "A GGUF tensor declares a dimension count above the scanner's rank limit (16), a "
            "negative dimension, or a shape whose element count overflows while its byte size is "
            "computed."
        ),
        detection_condition=(
            "Tensor rank > 16, any declared dimension is negative, or tensor byte-size computation "
            "raises an overflow."
        ),
        remediation="Ensure tensor dimension counts and ranks match standard GGML specifications.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-190",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    SecurityRule(
        rule_id="GU207",
        name="Invalid Tensor Type",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description="GGUF tensor specifies an unrecognized or invalid GGMLType enum value.",
        detection_condition="GGML tensor type enum ID not in recognized type table.",
        remediation="Use standard GGML quantization or floating-point tensor types.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="GU208",
        name="Invalid Tensor Offset or Alignment",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description="GGUF tensor data offset is negative or violates alignment boundary.",
        detection_condition="Offset < 0 or alignment not a power of two.",
        remediation="Align tensor data offsets to standard 32-byte boundaries.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-125",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="GU209",
        name="Tensor Range Outside File Bounds",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description="GGUF tensor data span exceeds physical file size bounds.",
        detection_condition="Tensor data section start + byte size > total file length.",
        remediation=(
            "Verify file integrity and ensure the binary was not truncated during transfer."
        ),
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-125",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1190: Exploit Public-Facing Application",
    ),
    SecurityRule(
        rule_id="GU210",
        name="Overlapping Tensor Data Ranges",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description="Two or more GGUF tensor descriptors declare overlapping byte ranges.",
        detection_condition=(
            "Tensor byte range [offset_i, offset_i + size_i) overlaps with another."
        ),
        remediation="Re-serialize model tensors to ensure all data spans are disjoint.",
        affected_format="GGUF, GGML",
        cve_ids=(),
        cwe_ids=("CWE-125",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    # ---------------------------------------------------------------------------
    # G0xx: Cross-Format & General Security Rules (2 rules: G001-G002)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="G001",
        name="Format Mismatch / Extension Spoofing",
        severity=Severity.HIGH,
        category="format_integrity",
        description=(
            "File extension indicates one model format (e.g. .safetensors), but internal "
            "magic byte sniffing detects a different, potentially executable format (e.g. Pickle)."
        ),
        detection_condition="Header magic format conflicts with declared file extension format.",
        remediation="Ensure file extension accurately reflects underlying binary format.",
        affected_format="ALL",
        cve_ids=(),
        cwe_ids=("CWE-436", "CWE-20"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1036.007: Masquerading: Double File Extension",
    ),
    SecurityRule(
        rule_id="G002",
        name="Format Identified — No Analyzer Available",
        severity=Severity.INFO,
        category="format_integrity",
        description=(
            "The artifact was identified by magic bytes as a known ML model format "
            "(e.g. TFLite, NumPy NPY/NPZ, Flax MessagePack) but this scanner has no registered "
            "analyzer for that format.  The artifact was not inspected; security coverage "
            "is incomplete. Note: ONNX, HDF5, and Keras formats DO have registered analyzers "
            "and will not fire this rule."
        ),
        detection_condition=(
            "Magic-byte sniffing identifies a recognized but unanalyzed format "
            "(NumPy NPY/NPZ, TFLite, Flax MessagePack). No analyzer is registered, "
            "so no rules ran against the artifact."
        ),
        remediation=(
            "Verify the artifact using a format-specific tool.  "
            "Do not assume a clean result from this scanner alone."
        ),
        affected_format="ALL",
        cve_ids=(),
        cwe_ids=("CWE-693",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development Tools",
    ),
    # ---------------------------------------------------------------------------
    # G00x: Cross-Format Shard Integrity Rules (4 rules: G005-G008)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="G005",
        name="Shard Declared in Index but Missing on Disk",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "The model index (model.safetensors.index.json, pytorch_model.bin.index.json, etc.) "
            "declares a shard file that is not present on disk."
        ),
        detection_condition=(
            "A shard filename listed in the index JSON weight_map is absent from the directory."
        ),
        remediation=(
            "Re-download all declared shards from the original source. "
            "A missing shard indicates an incomplete or tampered download."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    SecurityRule(
        rule_id="G006",
        name="Undeclared Shard File Present",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "A shard file with a weight-file extension exists in the model directory "
            "but is not declared in any index file."
        ),
        detection_condition=(
            "A weight-extension file found on disk is not listed in the index JSON weight_map."
        ),
        remediation=(
            "Verify whether the undeclared shard file is legitimate. "
            "Undeclared shards may indicate unauthorized additions."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    SecurityRule(
        rule_id="G007",
        name="Shard Size Disagrees with Index",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "The on-disk byte size of a shard file does not match the size recorded "
            "in the model index file."
        ),
        detection_condition=(
            "os.path.getsize(shard) != index_metadata['size_bytes'] for a declared shard."
        ),
        remediation=(
            "Re-download the affected shard from the original source. "
            "A size mismatch may indicate truncation or tampering."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    SecurityRule(
        rule_id="G008",
        name="Shard Count Mismatch",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The number of shards on disk does not match the number declared in the model index."
        ),
        detection_condition=(
            "len(on_disk_shards) != len(index_declared_shards) for a multi-shard model."
        ),
        remediation=(
            "Verify the shard count against the original repository. "
            "A count mismatch may indicate missing or extra shards."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    # ---------------------------------------------------------------------------
    # SC1xx: Supply Chain — Config & Code Execution Rules (9 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="SC101",
        name="Remote Code Execution via auto_map / Custom Class Reference",
        severity=Severity.HIGH,
        category="supply_chain",
        description=(
            "The model config.json contains an auto_map entry that references a Python "
            "class or module.  When loaded with trust_remote_code=True, HuggingFace "
            "Transformers will execute that code automatically."
        ),
        detection_condition=(
            "auto_map key present in config.json with a non-empty value."
        ),
        remediation=(
            "Inspect the referenced Python module before loading the model. "
            "Avoid loading models with custom code unless you trust the source."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-502", "CWE-94"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="SC102",
        name="trust_remote_code Enabled",
        severity=Severity.HIGH,
        category="supply_chain",
        description=(
            "The model repository config declares trust_remote_code=True, instructing "
            "frameworks to execute bundled Python files without further confirmation."
        ),
        detection_condition=(
            "trust_remote_code field is True in config.json or tokenizer_config.json."
        ),
        remediation=(
            "Do not set trust_remote_code=True without auditing all bundled Python files. "
            "Prefer models without custom code requirements."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-94",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204.002: User Execution: Malicious File",
    ),
    SecurityRule(
        rule_id="SC103",
        name="Bundled Executable Python Code",
        severity=Severity.MEDIUM,
        category="supply_chain",
        description=(
            "One or more Python source files (.py) are bundled alongside the model weights. "
            "These files may be executed when the model is loaded via HuggingFace Transformers."
        ),
        detection_condition=(
            "A .py file with a name matching common model-code patterns "
            "(modeling_*.py, tokenization_*.py, etc.) exists in the repository root."
        ),
        remediation=(
            "Audit all bundled Python files before loading the model. "
            "Run a static AST scan and code review on modeling_*.py and similar files."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-94",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204.002: User Execution: Malicious File",
    ),
    SecurityRule(
        rule_id="SC104",
        name="Dangerous System Call in Bundled Code",
        severity=Severity.CRITICAL,
        category="code_execution",
        description=(
            "Static AST analysis of a bundled Python file found a call to a "
            "dangerous system API (os.system, subprocess.Popen, eval, exec, etc.)."
        ),
        detection_condition=(
            "AST visitor finds a Call node whose function resolves to a "
            "blocklisted system/exec call in a bundled .py file."
        ),
        remediation=(
            "Do not load this model without auditing the bundled Python code. "
            "The dangerous call may execute arbitrary system commands."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-78", "CWE-94"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="SC105",
        name="Network Access in Bundled Code",
        severity=Severity.MEDIUM,
        category="supply_chain",
        description=(
            "Static AST analysis of a bundled Python file found an import or call "
            "to a network-access module (urllib, requests, socket, httpx, etc.)."
        ),
        detection_condition=(
            "AST visitor finds an import of or call to a blocklisted network module "
            "in a bundled .py file."
        ),
        remediation=(
            "Audit the bundled Python code for unintended network access. "
            "Network calls in model-loading code may exfiltrate data or download payloads."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-918",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1071: Application Layer Protocol",
    ),
    SecurityRule(
        rule_id="SC106",
        name="Unsigned Artifact — No Signature File Found",
        severity=Severity.INFO,
        category="provenance",
        description=(
            "No cryptographic signature file (.sig, .asc, sha256sum.txt, checksums.txt) "
            "was found alongside the model weight artifacts."
        ),
        detection_condition=(
            "None of the standard signature/checksum filenames exist in the repository root."
        ),
        remediation=(
            "Consider verifying artifact provenance via a signed release or Sigstore. "
            "An unsigned artifact does not guarantee authenticity."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1553: Subvert Trust Controls",
    ),
    SecurityRule(
        rule_id="SC107",
        name="Hash Self-Inconsistency (Possible Tampering)",
        severity=Severity.HIGH,
        category="provenance",
        description=(
            "The SHA-256 hash of an artifact on disk does not match the hash declared "
            "in the repository's index or metadata file."
        ),
        detection_condition=(
            "sha256(artifact_bytes) != declared_hash from model index or metadata."
        ),
        remediation=(
            "Do not use this artifact — the SHA-256 hash does not match the index declaration. "
            "Re-download from a trusted source and verify again."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494", "CWE-345"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1565: Data Manipulation",
    ),
    SecurityRule(
        rule_id="SC108",
        name="LFS Pointer Stub — Content Not Fetched",
        severity=Severity.MEDIUM,
        category="provenance",
        description=(
            "A model weight file contains a Git LFS pointer stub rather than actual "
            "weight data.  The real content was never fetched from LFS storage."
        ),
        detection_condition=(
            "File begins with 'version https://git-lfs.github.com/spec/v1' and "
            "size < 512 bytes — indicates an unfetched LFS pointer."
        ),
        remediation=(
            "Run `git lfs pull` to fetch actual weight content before scanning or deploying. "
            "An LFS stub contains only metadata, not real model weights."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    SecurityRule(
        rule_id="SC109",
        name="Config Declared Format Disagrees with Detected Format",
        severity=Severity.MEDIUM,
        category="provenance",
        description=(
            "The model_type or library_name declared in config.json does not match "
            "the format detected from the model weight file magic bytes."
        ),
        detection_condition=(
            "library_name in config.json implies a different weight format than "
            "the format detected from on-disk weight file headers."
        ),
        remediation=(
            "Verify that the model format matches the declared library. "
            "Inconsistency may indicate a mislabelled or repackaged artifact."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-436",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1036: Masquerading",
    ),
    # ---------------------------------------------------------------------------
    # SC2xx: Supply Chain — Dependency Security Rules (4 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="SC201",
        name="Unpinned Dependency",
        severity=Severity.LOW,
        category="supply_chain",
        description=(
            "A dependency declared in requirements.txt, setup.cfg, or pyproject.toml "
            "uses a range specifier (>=, ~=, *, <, !=) rather than an exact pin (==)."
        ),
        detection_condition=(
            "Dependency specifier does not contain '==' or has only a bare package name."
        ),
        remediation=(
            "Pin all dependencies to exact versions (==) in deployment artifacts. "
            "Unpinned dependencies allow silent vulnerability introduction."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-1104",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique=(
            "T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development "
            "Tools"
        ),
    ),
    SecurityRule(
        rule_id="SC202",
        name="Dependency with Known Advisory",
        severity=Severity.HIGH,
        category="supply_chain",
        description=(
            "A declared dependency matches a package version range that has a "
            "known security advisory in the bundled advisory snapshot."
        ),
        detection_condition=(
            "Package name and version specifier overlap with an entry in the "
            "local advisory snapshot under data/advisories/."
        ),
        remediation=(
            "Upgrade the affected dependency to a version that does not have a known advisory. "
            "Check the advisory for affected version ranges and patches."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-1104",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique=(
            "T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development "
            "Tools"
        ),
    ),
    SecurityRule(
        rule_id="SC203",
        name="Direct VCS/URL Dependency",
        severity=Severity.MEDIUM,
        category="supply_chain",
        description=(
            "A dependency is specified as a direct VCS URL (git+https://, svn+, hg+) "
            "or a direct HTTP/HTTPS URL rather than a versioned PyPI package."
        ),
        detection_condition=(
            "Requirement line starts with git+, svn+, hg+, bzr+, or contains @ http(s)://."
        ),
        remediation=(
            "Replace direct VCS/URL dependencies with pinned PyPI versions. "
            "Direct VCS deps bypass the PyPI registry security model."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-494",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique=(
            "T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development "
            "Tools"
        ),
    ),
    SecurityRule(
        rule_id="SC204",
        name="Possible Typosquatting (Edit Distance 1)",
        severity=Severity.HIGH,
        category="supply_chain",
        description=(
            "A declared dependency name has an edit distance of 1 from a well-known "
            "popular package, which may indicate a typosquatting attack."
        ),
        detection_condition=(
            "Levenshtein distance between package name and a popular package name is 1."
        ),
        remediation=(
            "Verify the package name is correct and not a typosquatting package. "
            "Compare carefully against the intended popular package name."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-1104",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique=(
            "T1195.001: Supply Chain Compromise: Compromise Software Dependencies and Development "
            "Tools"
        ),
    ),
    # ---------------------------------------------------------------------------
    # C0xx: Correlated Cross-Signal Security Rules (4 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="C001",
        name="Active Supply Chain Execution Intent",
        severity=Severity.CRITICAL,
        category="supply_chain",
        description=(
            "The repository contains bundled executable code AND the config explicitly "
            "instructs the ML framework to execute it (trust_remote_code=True). "
            "This elevates bundled code from a latent risk to an active threat vector."
        ),
        detection_condition=(
            "SC102 (trust_remote_code) is present AND either SC101 (auto_map) or "
            "SC103 (Bundled Code) is present in the repository."
        ),
        remediation=(
            "Do not load this model unless you fully trust the publisher. "
            "Review the bundled code for malicious behavior before instantiation."
        ),
        affected_format="REPOSITORY",
        cve_ids=(),
        cwe_ids=("CWE-94", "CWE-502"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204.002: User Execution: Malicious File",
    ),
    SecurityRule(
        rule_id="C002",
        name="Networked Executable Payload",
        severity=Severity.CRITICAL,
        category="code_execution",
        description=(
            "The model exhibits both arbitrary execution capabilities and network access "
            "primitives. This strongly indicates a dropper, reverse shell, or C2 beacon."
        ),
        detection_condition=(
            "(P101 (Dangerous Import) or SC104 (System Call)) AND SC105 (Network Access)."
        ),
        remediation=(
            "Reject this model immediately. It contains an active network-connected "
            "code execution payload."
        ),
        affected_format="ALL",
        cve_ids=(),
        cwe_ids=("CWE-78", "CWE-918"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1105: Ingress Tool Transfer",
    ),
    SecurityRule(
        rule_id="C003",
        name="Obfuscated Payload Staging",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "Nested archives combined with dynamic imports or suspicious reductions "
            "strongly indicate deliberate evasion and payload staging."
        ),
        detection_condition=(
            "P210 (Nested Archive) AND (P103 (STACK_GLOBAL) OR P105 (Suspicious REDUCE))."
        ),
        remediation=(
            "Avoid unpacking nested archives from unverified sources. "
            "Convert model to SafeTensors format if legitimate."
        ),
        affected_format="PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-434", "CWE-693"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="C004",
        name="Imported and Invoked Dangerous Callable",
        severity=Severity.CRITICAL,
        category="code_execution",
        description=(
            "The same dangerous callable is both imported by a GLOBAL/STACK_GLOBAL "
            "opcode and invoked by a REDUCE opcode within one Pickle stream. Import "
            "alone can be incidental; import followed by invocation is a completed "
            "code-execution chain."
        ),
        detection_condition=(
            "A P101/P102 finding and a P104/P105 finding share the same "
            "matched_value and archive member within a single artifact."
        ),
        remediation=(
            "Reject this model. The Pickle stream does not merely reference a "
            "dangerous callable, it calls it during deserialization."
        ),
        affected_format="RAW_PICKLE, PYTORCH_ZIP",
        cve_ids=(),
        cwe_ids=("CWE-502", "CWE-94"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1204.002: User Execution: Malicious File",
    ),
    # ---------------------------------------------------------------------------
    # MS1xx: MindSpore Checkpoint Security Rules (1 rule)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="MS101",
        name="Excessive Memory Allocation in MindSpore Checkpoint",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "A MindSpore checkpoint parameter declares tensor dimensions whose "
            "product — multiplied by the element byte width — would require "
            "gigabytes or terabytes of memory to load, but the file contains "
            "only a trivial number of actual tensor data bytes. "
            "This is a CWE-789 (Uncontrolled Memory Allocation) allocation bomb: "
            "calling mindspore.load_checkpoint() on such a file will attempt "
            "a massive memory allocation, crashing the process or exhausting "
            "system resources."
        ),
        detection_condition=(
            "Declared allocation (product(dims) * bytes_per_element) >= 10 GB "
            "AND actual tensor_content bytes / declared_bytes < 1e-6; "
            "OR declared allocation >= 100 GB regardless of actual data ratio. "
            "Arithmetic uses Python arbitrary-precision integers — no overflow, "
            "no tensor allocation in the scanner."
        ),
        remediation=(
            "Do not load this checkpoint with mindspore.load_checkpoint(). "
            "Re-acquire model weights from a trusted source and verify the "
            "declared tensor shapes match the actual data present in the file. "
            "Convert to SafeTensors format for format-verified supply chains."
        ),
        affected_format="MINDSPORE_CKPT",
        cve_ids=(),
        cwe_ids=("CWE-789",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1499: Endpoint Denial of Service",
    ),
    # ---------------------------------------------------------------------------
    # WI1xx: Cross-Format Weight Integrity Rules (2 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="WI101",
        name="NaN Values Detected in Tensor Weights",
        severity=Severity.HIGH,
        category="weight_integrity",
        description=(
            "One or more tensor weight values are NaN (Not a Number). "
            "NaN weights indicate corrupt training, deliberate tampering, or "
            "a poisoned model that may produce unpredictable outputs."
        ),
        detection_condition="Float tensor bytes contain IEEE 754 NaN bit patterns.",
        remediation=(
            "Re-acquire the model from a trusted source and verify its checksum. "
            "NaN weights may indicate model corruption or tampering."
        ),
        affected_format="SAFETENSORS, GGUF, ONNX, HDF5, KERAS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML03:2023 Model Inversion Attack",
        mitre_technique="T1565: Data Manipulation",
    ),
    SecurityRule(
        rule_id="WI102",
        name="Infinite Values Detected in Tensor Weights",
        severity=Severity.HIGH,
        category="weight_integrity",
        description=(
            "One or more tensor weight values are positive or negative infinity. "
            "Infinite weights indicate corrupt training, numerical overflow, or "
            "deliberate tampering that degrades model behavior."
        ),
        detection_condition="Float tensor bytes contain IEEE 754 Inf bit patterns.",
        remediation=(
            "Re-acquire the model from a trusted source and verify its checksum. "
            "Infinite weights indicate numerical corruption or tampering."
        ),
        affected_format="SAFETENSORS, GGUF, ONNX, HDF5, KERAS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML03:2023 Model Inversion Attack",
        mitre_technique="T1565: Data Manipulation",
    ),
    # ---------------------------------------------------------------------------
    # ON1xx: ONNX-Specific Security Rules (2 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="ON101",
        name="External or Custom Operator in ONNX Graph",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The ONNX graph references an operator from a non-standard domain "
            "(not 'ai.onnx' or ''). External and custom operators may execute "
            "arbitrary native code at inference time and should be carefully "
            "audited before deployment."
        ),
        detection_condition="NodeDef.domain is not '' or 'ai.onnx'.",
        remediation=(
            "Audit the custom operator implementation for malicious behavior. "
            "Prefer models using only standard ONNX operators. "
            "Do not deploy models with unaudited custom operators in production."
        ),
        affected_format="ONNX",
        cve_ids=(),
        cwe_ids=("CWE-94",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    SecurityRule(
        rule_id="ON102",
        name="Malformed ONNX Graph Structure",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The ONNX model graph is structurally invalid: it may contain missing "
            "graph definitions, empty node lists in a non-trivial model, or "
            "protobuf fields indicating structural corruption."
        ),
        detection_condition="ONNX graph proto is missing or structurally invalid.",
        remediation=(
            "Re-export the model from a trusted ONNX-compatible framework. "
            "Use onnx.checker.check_model() to validate before deployment."
        ),
        affected_format="ONNX",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="ON103",
        name="Path Traversal in ONNX External Data Location",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "An ONNX initializer references external tensor data at a location "
            "containing path-traversal sequences (e.g. '../'). When ONNX Runtime "
            "loads such a model, it opens the specified path relative to the model "
            "file's directory, allowing an attacker to read arbitrary files on the "
            "host filesystem (e.g. /etc/hosts, /etc/passwd, Windows credential stores)."
        ),
        detection_condition=(
            "TensorProto.external_data[location] contains '..' component or "
            "is an absolute path (starts with '/' or a drive letter)."
        ),
        remediation=(
            "Do not load this model with ONNX Runtime. Re-acquire the model from "
            "a trusted source and verify that all initializer external_data locations "
            "are relative paths without traversal sequences."
        ),
        affected_format="ONNX",
        cve_ids=(),
        cwe_ids=("CWE-22",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1083: File and Directory Discovery",
    ),
    SecurityRule(
        rule_id="ON104",
        name="External Data Offset/Length Overflow",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "An ONNX initializer's external_data segment declares an offset and length "
            "whose sum reaches or exceeds 2^64, producing a 64-bit integer wraparound "
            "when the addition is performed in uint64 arithmetic. Runtimes that compute "
            "bounds checks as 'm_offset + m_data_length > file_size' in uint64 will "
            "evaluate the wrapped (near-zero) sum against file_size, silently passing "
            "the check and subsequently dereferencing a wild pointer at the attacker-"
            "controlled wrapped address. Confirmed exploitable in OpenVINO 2025.4.1."
        ),
        detection_condition=(
            "TensorProto.external_data 'offset' (integer) + 'length' (integer) >= 2^64. "
            "Detection uses Python arbitrary-precision integer arithmetic — there is no "
            "silent overflow in the scanner itself. Malformed or non-integer values are "
            "skipped safely."
        ),
        remediation=(
            "Do not load this model with any ONNX runtime. Re-export the model from a "
            "trusted ONNX-compatible framework with correctly bounded external_data "
            "offset and length values. Verify that no initializer offset + length "
            "reaches or exceeds 2^64."
        ),
        affected_format="ONNX",
        cve_ids=(),
        cwe_ids=("CWE-190",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1190: Exploit Public-Facing Application",
    ),
    # ---------------------------------------------------------------------------
    # TF1xx: TensorFlow SavedModel Security Rules (2 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="TF101",
        name="External or Custom Op in TensorFlow Graph",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The TensorFlow SavedModel graph references an operation that is "
            "not in the standard TensorFlow op set. Custom ops may execute "
            "arbitrary native code at inference time and require careful auditing."
        ),
        detection_condition="NodeDef.op not in standard TF op set.",
        remediation=(
            "Audit the custom op implementation for malicious behavior. "
            "Prefer models using only standard TensorFlow ops. "
            "Do not deploy models with unaudited custom ops in production."
        ),
        affected_format="TF_SAVEDMODEL",
        cve_ids=(),
        cwe_ids=("CWE-94",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    SecurityRule(
        rule_id="TF102",
        name="Malformed TensorFlow SavedModel",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The saved_model.pb file is malformed, truncated, or cannot be parsed "
            "as a valid TensorFlow SavedModel protobuf. This may indicate "
            "corruption or deliberate tampering."
        ),
        detection_condition="saved_model.pb fails protobuf parsing or has invalid structure.",
        remediation=(
            "Re-export the model from TensorFlow. "
            "Verify the file is not truncated and re-acquire from a trusted source."
        ),
        affected_format="TF_SAVEDMODEL",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="TF103",
        name="Path Traversal Indicator in SavedModel Asset",
        severity=Severity.CRITICAL,
        category="tampering",
        description=(
            "The SavedModel asset directory contains a filename with a path-traversal "
            "indicator such as '../' or a URL-encoded traversal sequence (%2e%2e, %2f, "
            "%5c), or an absolute path. When TensorFlow restores such an asset it may "
            "write to or read from a location outside the model directory, enabling "
            "arbitrary file overwrite or disclosure on the host filesystem."
        ),
        detection_condition=(
            "An entry in the sibling assets/ directory has a filename containing a "
            "'..' component, a URL-encoded traversal token, or an absolute path. "
            "Detection is filename-string only; the asset file is never accessed."
        ),
        remediation=(
            "Do not load this SavedModel. Re-acquire from a trusted source and verify "
            "that all asset filenames are plain relative names without traversal sequences."
        ),
        affected_format="TF_SAVEDMODEL",
        cve_ids=(),
        cwe_ids=("CWE-22",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1083: File and Directory Discovery",
    ),
    # ---------------------------------------------------------------------------
    # KR1xx: Keras-Specific Security Rules (3 rules)
    # ---------------------------------------------------------------------------
    SecurityRule(
        rule_id="KR101",
        name="External or Custom Class in Keras Config",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The Keras model configuration references a layer or class that is "
            "not in the standard Keras built-in class set. Custom classes may "
            "execute arbitrary Python code when the model is loaded and "
            "should be audited before deployment."
        ),
        detection_condition="Layer class_name not in standard keras built-in class set.",
        remediation=(
            "Audit the custom class implementation for malicious behavior. "
            "Do not load models with custom classes using trust settings that "
            "execute arbitrary code. Inspect the class before deployment."
        ),
        affected_format="HDF5, KERAS",
        cve_ids=(),
        cwe_ids=("CWE-94",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1195: Supply Chain Compromise",
    ),
    SecurityRule(
        rule_id="KR102",
        name="Malformed Keras Model Config or Structure",
        severity=Severity.MEDIUM,
        category="structural_anomaly",
        description=(
            "The Keras model configuration JSON is malformed, missing, or contains "
            "invalid structure. This may indicate corruption, truncation, or "
            "a tampered model file."
        ),
        detection_condition="Keras config JSON is missing, invalid, or structurally malformed.",
        remediation=(
            "Re-export the model using tf.keras or standalone Keras. "
            "Verify the file is not truncated and re-acquire from a trusted source."
        ),
        affected_format="HDF5, KERAS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="KR103",
        name="Duplicate Archive Member in Keras ZIP",
        severity=Severity.HIGH,
        category="structural_anomaly",
        description=(
            "A Keras .keras ZIP archive contains more than one member with the same "
            "name. Python's zipfile resolves a name to the LAST matching entry, while "
            "some scanners inspect the FIRST. This asymmetry is a known scanner-bypass "
            "technique: a benign first config.json can hide a malicious duplicate that "
            "the loader actually uses."
        ),
        detection_condition=(
            "zipfile.namelist() contains a duplicated member name (count > 1)."
        ),
        remediation=(
            "Reject this archive. A well-formed .keras file has exactly one entry per "
            "name. Re-export the model with a trusted serializer and re-acquire from a "
            "trusted source."
        ),
        affected_format="KERAS",
        cve_ids=(),
        cwe_ids=("CWE-694", "CWE-693"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1027: Obfuscated Files or Information",
    ),
    SecurityRule(
        rule_id="KR104",
        name="HDF5 External / Virtual Dataset Reference",
        severity=Severity.HIGH,
        category="code_execution",
        description=(
            "The HDF5 file contains an ExternalLink or a Virtual Dataset (VDS) "
            "mapping that references a file path outside the model artifact. "
            "When a consumer opens the model, HDF5 will automatically follow the "
            "link and read data from the referenced external file. An attacker can "
            "exploit this to exfiltrate host data, trigger SSRF, or load a malicious "
            "payload from a controlled path — without the consumer's explicit consent."
        ),
        detection_condition=(
            "h5py.Group.get(key, getlink=True) returns h5py.ExternalLink, or "
            "dataset.is_virtual is True and virtual_sources() contains a source "
            "whose file_name is not '.' (i.e. references an external file)."
        ),
        remediation=(
            "Do not load this model file. HDF5 external links and virtual datasets "
            "that reference arbitrary host paths are a path-traversal and data "
            "exfiltration vector. Reject the artifact and re-acquire from a trusted "
            "source that does not embed external references. If external datasets are "
            "required by design, restrict the HDF5 library's allowed paths using the "
            "h5py SWMR/external-link guard settings and validate all referenced paths "
            "against an allowlist before opening."
        ),
        affected_format="HDF5, KERAS",
        cve_ids=("CVE-2026-1669",),
        cwe_ids=("CWE-73", "CWE-200"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1083: File and Directory Discovery",
    ),
    SecurityRule(
        rule_id="KR105",
        name="Keras Lambda Bytecode Payload",
        severity=Severity.CRITICAL,
        category="code_execution",
        description=(
            "A Keras Lambda layer's serialized bytecode contains indicators of dangerous "
            "execution primitives: OS command execution (os.system, subprocess, popen), "
            "dynamic code evaluation (eval, exec), or dynamic module import (__import__). "
            "Keras serializes Lambda functions as base64-encoded Python marshal bytecode. "
            "When the model is loaded, Keras deserializes and executes this code, making "
            "a malicious Lambda an arbitrary-code-execution vector."
        ),
        detection_condition=(
            "The 'code' field inside a '__lambda__' config node, after base64 decoding, "
            "contains one or more of: subprocess, Popen, popen (CRITICAL); or the "
            "combination os+system, nt+system, posix+system as standalone identifiers "
            "(CRITICAL); or __import__, eval, exec as standalone identifiers (HIGH). "
            "All matches use word-boundary checks to prevent partial-name false positives. "
            "No marshal.loads(), exec(), or eval() is called during detection."
        ),
        remediation=(
            "Do not load this model. A Keras Lambda layer's serialized bytecode contains "
            "indicators of dangerous execution primitives (OS command execution, dynamic "
            "import, or eval/exec). Reject the artifact and re-acquire from a trusted "
            "source. If the Lambda is required, audit the source function and re-serialize "
            "from verified, reviewed code only."
        ),
        affected_format="HDF5, KERAS",
        cve_ids=("CVE-2024-3660",),
        cwe_ids=("CWE-94", "CWE-502"),
        owasp_category="ML06:2023 AI Supply Chain Attacks",
        mitre_technique="T1059: Command and Scripting Interpreter",
    ),
    SecurityRule(
        rule_id="WI103",
        name="Extreme Finite Values Detected in Tensor Weights",
        severity=Severity.MEDIUM,
        category="weight_integrity",
        description=(
            "One or more tensor weight values are at the maximum finite representable "
            "magnitude for their dtype (e.g., F32 exponent == 0xFE, meaning |value| ≥ 1.7e38). "
            "Values this close to overflow suggest numerical instability, gradient explosion "
            "during training, or deliberate weight manipulation."
        ),
        detection_condition=(
            "Float tensor bytes have exponent bits equal to the maximum finite exponent "
            "for the dtype (F32/BF16: 0xFE; F16: 0x1E; F64: 0x7FE)."
        ),
        remediation=(
            "Investigate the origin of extreme-magnitude weights. "
            "Values at the maximum finite representable magnitude suggest numerical "
            "instability, overflow during training, or tampering. "
            "Re-acquire from a trusted source and verify with the model provider."
        ),
        affected_format="SAFETENSORS, GGUF, ONNX, HDF5, KERAS",
        cve_ids=(),
        cwe_ids=("CWE-20",),
        owasp_category="ML03:2023 Model Inversion Attack",
        mitre_technique="T1565: Data Manipulation",
    ),
)

def _build_rule_map(rules: tuple[SecurityRule, ...]) -> dict[str, SecurityRule]:
    """Index rules by ID, refusing to silently drop a duplicated rule_id.

    A plain dict comprehension would let a copy-pasted rule shadow the original
    and shrink the catalog without any error, so the collision is raised here at
    import time instead.
    """
    mapping: dict[str, SecurityRule] = {}
    for rule in rules:
        if rule.rule_id in mapping:
            raise ValueError(
                f"Duplicate rule_id {rule.rule_id!r} registered in _CANONICAL_RULES"
            )
        mapping[rule.rule_id] = rule
    return mapping


_RULE_MAP: dict[str, SecurityRule] = _build_rule_map(_CANONICAL_RULES)


def _format_tokens(affected_format: str) -> frozenset[str]:
    """Split an ``affected_format`` declaration into its individual format tokens."""
    return frozenset(t.strip().upper() for t in affected_format.split(",") if t.strip())


class RuleCatalog:
    """Centralized, queryable registry of all static security analysis rules."""

    @classmethod
    def get_rule(cls, rule_id: str) -> SecurityRule:
        """Look up a SecurityRule by rule_id.

        Parameters
        ----------
        rule_id:
            The canonical rule ID (e.g. 'P101', 'ST101', 'GU101', 'G001').

        Returns
        -------
        SecurityRule
            The registered rule definition.

        Raises
        ------
        KeyError
            If rule_id is not registered in the catalog.
        """
        if rule_id not in _RULE_MAP:
            raise KeyError(f"Rule ID {rule_id!r} not found in RuleCatalog")
        return _RULE_MAP[rule_id]

    @classmethod
    def has_rule(cls, rule_id: str) -> bool:
        """Check if a rule ID is registered in the catalog."""
        return rule_id in _RULE_MAP

    @classmethod
    def list_all_rules(cls) -> tuple[SecurityRule, ...]:
        """Return every registered rule in deterministic registration order."""
        return _CANONICAL_RULES

    @classmethod
    def get_rules_by_category(cls, category: str) -> tuple[SecurityRule, ...]:
        """Filter registered rules by security category."""
        return tuple(r for r in _CANONICAL_RULES if r.category == category)

    @classmethod
    def get_rules_by_format(cls, model_format: str) -> tuple[SecurityRule, ...]:
        """Filter registered rules by affected format.

        ``affected_format`` is a comma-separated list of format tokens, so the
        match is made token by token. A substring test would report every
        ``RAW_PICKLE`` rule for the non-existent format ``"PICKLE"`` and every
        ``GGUF, GGML`` rule for ``"GG"``.

        Rules declaring ``ALL`` are always included.
        """
        target = model_format.strip().upper()
        return tuple(
            r
            for r in _CANONICAL_RULES
            if r.affected_format == "ALL" or target in _format_tokens(r.affected_format)
        )

    @classmethod
    def get_rules_by_severity(cls, severity: Severity) -> tuple[SecurityRule, ...]:
        """Filter registered rules by severity level."""
        return tuple(r for r in _CANONICAL_RULES if r.severity == severity)


__all__ = [
    "RuleCatalog",
]
