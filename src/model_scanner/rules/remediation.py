"""
model_scanner.rules.remediation
===============================

Actionable remediation guidance and safe model format migration engine (Phase 8).
"""

from __future__ import annotations

from model_scanner.rules.catalog import RuleCatalog

_UNANALYZED_FORMAT_GUIDE: str = (
    "FORMAT NOT COVERED BY THIS SCANNER:\n"
    "1. This format was identified but has no registered analyzer, so no rule "
    "ran against the artifact. A clean report is an absence of evidence, not "
    "evidence of absence - do not read it as an all-clear.\n"
    "2. Run a format-specific checker before use (e.g. onnx.checker for ONNX), "
    "and inspect any custom operators, external-data references, or embedded "
    "Python the container permits.\n"
    "3. Where the publisher also ships SafeTensors weights, prefer those and "
    "scan them instead."
)

_MIGRATION_GUIDES: dict[str, str] = {
    "RAW_PICKLE": (
        "MIGRATION TO SAFETENSORS (RECOMMENDED):\n"
        "1. Never call pickle.load() / torch.load(weights_only=False) on this file: "
        "unpickling executes whatever the stream tells it to.\n"
        "2. Convert the weights in a sandbox, with weights_only=True so only "
        "tensor storage is reconstructed and no arbitrary global is called:\n"
        "   from safetensors.torch import save_file\n"
        "   state_dict = torch.load('model.pkl', map_location='cpu', weights_only=True)\n"
        "   save_file(state_dict, 'model.safetensors')\n"
        "3. If step 2 raises on a disallowed global, the file relies on code "
        "execution to load - treat it as malicious rather than allowlisting the global.\n"
        "4. Re-scan the converted SafeTensors artifact before use."
    ),
    "PYTORCH_ZIP": (
        "MIGRATION TO SAFETENSORS:\n"
        "1. Extract weights in an isolated environment using 'weights_only=True'.\n"
        "2. Export tensor state dictionaries into standard SafeTensors format (.safetensors).\n"
        "3. Eliminate executable code objects and custom classes from the serialization graph."
    ),
    "SAFETENSORS": (
        "SAFETENSORS REMEDIATION:\n"
        "1. Inspect and sanitize __metadata__ JSON header entries to remove "
        "unverified URLs or tokens.\n"
        "2. Ensure tensor data offsets are non-overlapping and contiguous.\n"
        "3. Keep total header size within the 16 MB safe limit."
    ),
    "GGUF": (
        "GGUF / GGML REMEDIATION:\n"
        "1. Sanitize metadata KV entries to remove any script or path traversal patterns.\n"
        "2. Ensure tensor data offsets are strictly aligned to 32-byte boundaries.\n"
        "3. Re-export using standard llama.cpp conversion scripts."
    ),
    "GGML": (
        "GGML MIGRATION:\n"
        "1. GGML is a legacy format; upgrade model weights to modern GGUF v3 format.\n"
        "2. Run official llama.cpp 'convert-llama-ggml-to-gguf.py' migration tool."
    ),
    "REPOSITORY": (
        "MODEL REPOSITORY REMEDIATION:\n"
        "1. Load with trust_remote_code=False; if the model cannot run without it, "
        "read every bundled .py file before deciding to trust the publisher.\n"
        "2. Pin the exact revision (commit SHA, not a branch or tag) when fetching "
        "the repository, so the code you reviewed is the code you load.\n"
        "3. Pin every declared dependency to an exact version (==) and replace "
        "direct VCS/URL requirements with released PyPI versions.\n"
        "4. Verify shard checksums against the index file before deployment."
    ),
    "ONNX": _UNANALYZED_FORMAT_GUIDE,
    "HDF5": _UNANALYZED_FORMAT_GUIDE,
    "TFLITE": _UNANALYZED_FORMAT_GUIDE,
    "NUMPY_NPY": _UNANALYZED_FORMAT_GUIDE,
    "NUMPY_NPZ": _UNANALYZED_FORMAT_GUIDE,
    "FLAX_MSGPACK": _UNANALYZED_FORMAT_GUIDE,
}

_DEFAULT_MIGRATION_GUIDE: str = (
    "GENERAL MODEL REMEDIATION:\n"
    "1. Replace untrusted binary model files with verified SafeTensors artifacts.\n"
    "2. Verify artifact SHA-256 cryptographic hashes against author release signatures.\n"
    "3. Enforce static pre-deployment CI/CD scanning before mounting weights in production."
)


# Names users and callers reach for that are not ModelFormat members.
_FORMAT_ALIASES: dict[str, str] = {
    "PICKLE": "RAW_PICKLE",
    "PYTORCH": "PYTORCH_ZIP",
    "PT": "PYTORCH_ZIP",
    "TORCH": "PYTORCH_ZIP",
    "KERAS": "HDF5",
    "H5": "HDF5",
    "NPY": "NUMPY_NPY",
    "NPZ": "NUMPY_NPZ",
}


class RemediationRegistry:
    """Queryable registry for actionable vulnerability remediation and migration advice."""

    @classmethod
    def get_guidance(cls, rule_id: str) -> str:
        """Return the actionable remediation guidance for a specific rule ID.

        Parameters
        ----------
        rule_id:
            The canonical rule identifier (e.g. 'P101', 'ST101', 'GU101', 'G001').

        Returns
        -------
        str
            Actionable remediation instructions.

        Raises
        ------
        KeyError
            If rule_id is not registered in the catalog.
        """
        rule = RuleCatalog.get_rule(rule_id)
        return rule.remediation

    @classmethod
    def get_cve_mappings(cls, rule_id: str) -> tuple[str, ...]:
        """Return the CVE identifiers mapped to a specific rule ID.

        Parameters
        ----------
        rule_id:
            The canonical rule identifier.

        Returns
        -------
        tuple[str, ...]
            Tuple of mapped CVE IDs. Empty for every rule in the shipped
            catalog - these rules detect malicious or malformed artifacts
            rather than a specific vulnerable implementation.

        Raises
        ------
        KeyError
            If rule_id is not registered in the catalog.
        """
        rule = RuleCatalog.get_rule(rule_id)
        return rule.cve_ids

    @classmethod
    def get_cwe_mappings(cls, rule_id: str) -> tuple[str, ...]:
        """Return the CWE identifiers mapped to a specific rule ID.

        Parameters
        ----------
        rule_id:
            The canonical rule identifier.

        Returns
        -------
        tuple[str, ...]
            Tuple of mapped CWE IDs (e.g. ('CWE-502',)).

        Raises
        ------
        KeyError
            If rule_id is not registered in the catalog.
        """
        rule = RuleCatalog.get_rule(rule_id)
        return rule.cwe_ids

    @classmethod
    def get_migration_advice(cls, model_format: str) -> str:
        """Return format-specific safe conversion and migration instructions.

        Parameters
        ----------
        model_format:
            The format name, normally a ``ModelFormat`` value (e.g. 'RAW_PICKLE',
            'PYTORCH_ZIP', 'SAFETENSORS', 'GGUF', 'REPOSITORY'). Unknown names
            fall back to the general guidance.

        Returns
        -------
        str
            Step-by-step format migration guidance.
        """
        key = model_format.strip().upper()
        key = _FORMAT_ALIASES.get(key, key)
        return _MIGRATION_GUIDES.get(key, _DEFAULT_MIGRATION_GUIDE)


__all__ = [
    "RemediationRegistry",
]
