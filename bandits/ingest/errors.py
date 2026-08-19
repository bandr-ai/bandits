"""Infer a coarse ``error_kind`` from a tool's error response.

Why this exists: the rebuilt environment has to be able to *reproduce adversity*
(PLAN.md Step 7). To do that, stage 2 needs to group failure responses into named
error modes, and grouping needs a stable label. That label is inferred here, once,
at ingest time, so every later stage sees the same string.

The rule is deliberately small and explicit. We read a label out of the response
where an export actually put one; we never guess from prose, and we never invent a
kind. When nothing is declared, the kind stays ``None`` and the record is still an
honest ``CallStatus.ERROR`` with an unknown mode.
"""

from __future__ import annotations

import re
from typing import Any

from bandits.contracts import JsonValue

#: Keys that, at the top level of a response object, are understood to carry a
#: machine-readable failure label. Ordered: the first one present wins.
ERROR_LABEL_KEYS: tuple[str, ...] = ("error", "error_kind", "error_code", "code", "type")

#: Keys read from a *nested* error object, e.g. ``{"error": {"code": "not_found"}}``.
NESTED_LABEL_KEYS: tuple[str, ...] = ("kind", "code", "type", "name", "reason")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def normalize_error_kind(raw: str) -> str | None:
    """Turn a declared error label into a stable snake_case slug.

    Exports write ``not_found``, ``NOT_FOUND`` and ``NotFound`` for the same thing;
    they must land on one key or stage 2 will report three error modes where there
    is one. Returns ``None`` for a label that slugifies to nothing.
    """
    slug = _SLUG_RE.sub("_", raw.strip().lower()).strip("_")
    return slug or None


def _from_scalar(value: Any) -> str | None:
    """Read a label from a scalar error field, ignoring anything that is not text."""
    if isinstance(value, str):
        return normalize_error_kind(value)
    if isinstance(value, bool):
        # ``{"error": true}`` says *that* it failed, never *how*. Not a kind.
        return None
    if isinstance(value, int):
        return normalize_error_kind(str(value))
    return None


def _from_mapping(value: dict[str, Any]) -> str | None:
    """Read a label from a nested error object, e.g. ``{"code": "not_found"}``."""
    for key in NESTED_LABEL_KEYS:
        if key in value:
            kind = _from_scalar(value[key])
            if kind is not None:
                return kind
    return None


def infer_error_kind(response: JsonValue) -> str | None:
    """Return the coarse error label for an error response, or ``None`` if unknown.

    Our fixture corpus puts the label at ``response["error"]`` as a plain string,
    which is the common shape; we also accept the other keys in
    :data:`ERROR_LABEL_KEYS` and a nested error object. Anything else -- a bare
    string body, a list, a success-shaped payload -- yields ``None`` rather than a
    fabricated kind.
    """
    if not isinstance(response, dict):
        return None
    for key in ERROR_LABEL_KEYS:
        if key not in response:
            continue
        value = response[key]
        if isinstance(value, dict):
            kind = _from_mapping(value)
        else:
            kind = _from_scalar(value)
        if kind is not None:
            return kind
    return None


def looks_like_error(response: JsonValue) -> bool:
    """True when a response object declares a failure on its face.

    Needed by the chat-json adapter: plain chat transcripts carry no span status,
    so the only evidence a call failed is the shape of the response itself. OTLP
    does not use this -- there the span status is authoritative.
    """
    if not isinstance(response, dict):
        return False
    for key in ("error", "error_kind", "error_code"):
        if key in response and response[key] not in (None, False, "", {}):
            return True
    status = response.get("status")
    return isinstance(status, str) and status.strip().lower() == "error"


__all__ = [
    "ERROR_LABEL_KEYS",
    "NESTED_LABEL_KEYS",
    "infer_error_kind",
    "looks_like_error",
    "normalize_error_kind",
]
