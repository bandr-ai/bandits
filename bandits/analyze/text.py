"""Text normalization shared by stages that compare request strings.

Case and separators only. Nothing here hides a value: two requests that differ
in a route, a date or an id stay different after normalization, because a
normalizer that collapsed them would silently merge tasks a verifier has to
tell apart.
"""

from __future__ import annotations

import re

_NON_TOKEN = re.compile(r"[^a-z0-9<>_]+")


def normalize_instruction(instruction: str) -> str:
    """Normalize case and separators without hiding any values."""
    return " ".join(_NON_TOKEN.sub(" ", instruction.lower()).split())
