"""
model_scanner.analyzers.safetensors.token_guard
================================================

Lexical pre-scanner for SafeTensors JSON headers.

This module performs a single-pass byte scan of the raw JSON header bytes
BEFORE calling ``json.loads()``.  It enforces three resource ceilings that
protect against adversarial JSON DoS attacks:

1. **Token budget** — total count of structural JSON delimiters
   (``{``, ``}``, ``[``, ``]``, ``:``, ``,``) must not exceed
   ``max_json_tokens``.

2. **Nesting depth** — maximum bracket nesting depth must not exceed
   ``max_json_depth``.

3. **Duplicate top-level key detection** — keys at the root object level
   are collected so that the caller can report ST207 (Duplicate Tensor Name).

SECURITY GUARANTEE
------------------
- No ``json.loads()``, ``eval()``, or ``exec()`` is called in this module.
- The scan is purely byte-level; no Python objects are constructed.
- Memory usage is O(number of top-level keys x average key length), bounded
  by the token budget.

Design Reference
----------------
Section 16.2 of docs/implementation_design.md
"""

from __future__ import annotations

from model_scanner.core.exceptions import ResourceLimitError, SafeTensorsAnalysisError

# ASCII byte values for JSON structural characters
_OPEN_BRACE: int = ord("{")
_CLOSE_BRACE: int = ord("}")
_OPEN_BRACKET: int = ord("[")
_CLOSE_BRACKET: int = ord("]")
_COLON: int = ord(":")
_COMMA: int = ord(",")
_DOUBLE_QUOTE: int = ord('"')
_BACKSLASH: int = ord("\\")


class TokenGuardResult:
    """Result of the :func:`scan_json_tokens` pre-validation pass.

    Attributes
    ----------
    token_count : int
        Total number of structural JSON tokens counted.
    max_depth : int
        Maximum bracket/brace nesting depth observed.
    top_level_keys : list[str]
        Keys found at the root JSON object's top level.
    duplicate_keys : list[str]
        Top-level keys that appeared more than once.
    """

    __slots__ = ("duplicate_keys", "max_depth", "token_count", "top_level_keys")

    def __init__(
        self,
        token_count: int,
        max_depth: int,
        top_level_keys: list[str],
        duplicate_keys: list[str],
    ) -> None:
        self.token_count = token_count
        self.max_depth = max_depth
        self.top_level_keys = top_level_keys
        self.duplicate_keys = duplicate_keys


def scan_json_tokens(
    raw_json: bytes,
    max_tokens: int,
    max_depth: int,
) -> TokenGuardResult:
    """Perform a bounded single-pass lexical scan of raw JSON bytes.

    Parameters
    ----------
    raw_json:
        Raw bytes of the JSON header (not including the 8-byte length prefix).
    max_tokens:
        Maximum permitted number of structural tokens.  Derived from
        ``ScanConfig.max_json_tokens``.
    max_depth:
        Maximum permitted bracket nesting depth.  Derived from
        ``ScanConfig.max_json_depth``.

    Returns
    -------
    TokenGuardResult
        Counts, depth, and top-level keys found in the scan.

    Raises
    ------
    SafeTensorsAnalysisError
        If the JSON bytes are empty.
    ResourceLimitError
        If token count or nesting depth exceeds the configured ceiling.
    """
    if not raw_json:
        raise SafeTensorsAnalysisError(
            message="SafeTensors JSON header is empty",
            error_code="ST_HEADER_EMPTY",
        )

    token_count: int = 0
    current_depth: int = 0
    observed_max_depth: int = 0
    in_string: bool = False
    escape_next: bool = False

    # Top-level key collection — only keys at depth == 1 (root object)
    top_level_keys: list[str] = []
    seen_keys: dict[str, int] = {}   # key -> first occurrence index
    duplicate_keys: list[str] = []

    # Simple string-extraction state for top-level key collection
    # We collect UTF-8 key bytes when we are in a string at depth 1 and
    # the previous non-whitespace token was a '{' or ','  (i.e., key position)
    _collecting_key: bool = False
    _key_buf: bytearray = bytearray()

    # Tracks whether the next string at depth 1 is a key (True) or value
    _expecting_key_at_depth_1: bool = True

    for byte in raw_json:
        # ----------------------------------------------------------------
        # String-state tracking
        # ----------------------------------------------------------------
        if escape_next:
            escape_next = False
            if _collecting_key:
                _key_buf.append(byte)
            continue

        if in_string:
            if byte == _BACKSLASH:
                escape_next = True
                if _collecting_key:
                    _key_buf.append(byte)
            elif byte == _DOUBLE_QUOTE:
                # String ends
                in_string = False
                if _collecting_key:
                    try:
                        key_str = bytes(_key_buf).decode("utf-8", errors="replace")
                    except Exception:
                        key_str = repr(bytes(_key_buf))
                    top_level_keys.append(key_str)
                    if key_str in seen_keys:
                        if key_str not in duplicate_keys:
                            duplicate_keys.append(key_str)
                    else:
                        seen_keys[key_str] = len(top_level_keys) - 1
                    _collecting_key = False
                    _key_buf.clear()
                    # After a key at depth 1 the next string will be a value
                    _expecting_key_at_depth_1 = False
            else:
                if _collecting_key:
                    _key_buf.append(byte)
            continue

        # ----------------------------------------------------------------
        # Not inside a string
        # ----------------------------------------------------------------
        if byte == _DOUBLE_QUOTE:
            in_string = True
            # Begin collecting if we are at depth 1 and expecting a key
            if current_depth == 1 and _expecting_key_at_depth_1:
                _collecting_key = True
                _key_buf.clear()
            continue

        if byte in (_OPEN_BRACE, _OPEN_BRACKET):
            token_count += 1
            if token_count > max_tokens:
                raise ResourceLimitError(
                    message=(
                        f"SafeTensors JSON header exceeds token budget of "
                        f"{max_tokens:,} tokens"
                    ),
                    error_code="ST_TOKEN_BUDGET_EXCEEDED",
                    details=f"token_count>{max_tokens}",
                    limit_name="max_json_tokens",
                    limit_value=max_tokens,
                    actual_value=token_count,
                )
            current_depth += 1
            if current_depth > observed_max_depth:
                observed_max_depth = current_depth
            if current_depth > max_depth:
                raise ResourceLimitError(
                    message=(
                        f"SafeTensors JSON header nesting depth {current_depth} "
                        f"exceeds limit of {max_depth}"
                    ),
                    error_code="ST_DEPTH_EXCEEDED",
                    details=f"depth={current_depth}, limit={max_depth}",
                    limit_name="max_json_depth",
                    limit_value=max_depth,
                    actual_value=current_depth,
                )
            if byte == _OPEN_BRACE and current_depth == 2:
                # Entering a value object at depth 2 means next key resets
                _expecting_key_at_depth_1 = False

        elif byte in (_CLOSE_BRACE, _CLOSE_BRACKET):
            token_count += 1
            if token_count > max_tokens:
                raise ResourceLimitError(
                    message=(
                        f"SafeTensors JSON header exceeds token budget of "
                        f"{max_tokens:,} tokens"
                    ),
                    error_code="ST_TOKEN_BUDGET_EXCEEDED",
                    details=f"token_count>{max_tokens}",
                    limit_name="max_json_tokens",
                    limit_value=max_tokens,
                    actual_value=token_count,
                )
            if current_depth > 0:
                current_depth -= 1
            # After closing depth 1 back to depth 0 or depth 1, next string is a key
            if current_depth == 1:
                _expecting_key_at_depth_1 = True

        elif byte == _COLON:
            token_count += 1
            if token_count > max_tokens:
                raise ResourceLimitError(
                    message=(
                        f"SafeTensors JSON header exceeds token budget of "
                        f"{max_tokens:,} tokens"
                    ),
                    error_code="ST_TOKEN_BUDGET_EXCEEDED",
                    details=f"token_count>{max_tokens}",
                    limit_name="max_json_tokens",
                    limit_value=max_tokens,
                    actual_value=token_count,
                )

        elif byte == _COMMA:
            token_count += 1
            if token_count > max_tokens:
                raise ResourceLimitError(
                    message=(
                        f"SafeTensors JSON header exceeds token budget of "
                        f"{max_tokens:,} tokens"
                    ),
                    error_code="ST_TOKEN_BUDGET_EXCEEDED",
                    details=f"token_count>{max_tokens}",
                    limit_name="max_json_tokens",
                    limit_value=max_tokens,
                    actual_value=token_count,
                )
            # After a comma at depth 1 the next string is a key
            if current_depth == 1:
                _expecting_key_at_depth_1 = True

    return TokenGuardResult(
        token_count=token_count,
        max_depth=observed_max_depth,
        top_level_keys=top_level_keys,
        duplicate_keys=duplicate_keys,
    )


__all__ = ["TokenGuardResult", "scan_json_tokens"]
