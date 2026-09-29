"""
model_scanner.config.loader
===========================

Load ScanConfig from a JSON or TOML configuration file.

Supported formats:
- JSON  (any filename, detected by .json extension)
- TOML  (any filename, detected by .toml extension)

SECURITY GUARANTEE
------------------
This module only reads configuration files using standard-library parsers.
It NEVER executes code, imports ML frameworks, or performs network I/O.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any

from model_scanner.core.models import ScanConfig


def load_config(path: str | Path) -> ScanConfig:
    """Load a ScanConfig from a JSON or TOML file.

    Parameters
    ----------
    path:
        Path to a ``.json`` or ``.toml`` configuration file.

    Returns
    -------
    ScanConfig
        Populated configuration object (unrecognized keys are ignored).

    Raises
    ------
    ValueError
        If the file extension is not ``.json`` or ``.toml``.
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        with open(p, "rb") as fh:
            raw = tomllib.load(fh)
        # Support a [scanner] table at the top level
        if "scanner" in raw and isinstance(raw["scanner"], dict):
            raw = raw["scanner"]
    else:
        raise ValueError(
            f"Unsupported config file format {suffix!r}. "
            f"Expected .json or .toml."
        )

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a JSON object / TOML table at its root.")

    known_fields = {f.name for f in dataclasses.fields(ScanConfig)}

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known_fields:
            continue
        # Coerce list → tuple for tuple-typed fields
        if isinstance(value, list):
            value = tuple(str(v) for v in value)
        kwargs[key] = value

    return ScanConfig(**kwargs)


__all__ = ["load_config"]
