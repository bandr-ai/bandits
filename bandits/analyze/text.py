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


_PARAMETER_RULES = (
    # Read off the raw instruction rather than the normalized form, which drops
    # the punctuation that makes these recognisable at all.
    re.compile(r"[\"'“‘]([^\"'”’\n]{1,120})[\"'”’]"),
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    re.compile(r"(~?[\w./-]*\.[A-Za-z]{2,5})\b"),
    re.compile(r"(~/[\w./-]+)"),
    re.compile(r"\$?(\d[\d,]*(?:\.\d+)?)"),
)
"""Where a request carries a value that decides what its correct answer is.

Narrower than the embedding miner's original rule set — no bare month names —
because this only feeds ``request_signature``'s paraphrase check, not a
similarity-threshold join. False positives would silently merge two different
requests, while false negatives can let repeat work cross the evaluation split.
"""

_QUOTED_DATE_RANGE = re.compile(
    r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s+(?:to|through|until|-)\s+"
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*$",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    "a an the to for of on in at by from with please could can would kindly "
    "i me my you your this that it is are was were be issue go ahead".split()
)


def request_parameter_values(instruction: str) -> tuple[str, ...]:
    """The values that make one request specific rather than a kind of request.

    A filename, an amount, a date, a quoted string: change one and the
    correct answer changes with it, even though almost every word of the
    instruction is unchanged.

    Deliberately lexical. It recognises a value written down, not a scope
    described in prose — 'my playlists' against 'my song library' names two
    different things and yields no parameter here.
    """
    found: list[tuple[int, int, str]] = []
    for rule in _PARAMETER_RULES:
        for match in rule.finditer(instruction):
            value = match.group(1).strip().lower()
            span = match.span(1)
            # Quotes do not turn an otherwise bare date range into one opaque
            # value. Extract its two dates in the same order as the unquoted
            # spelling, while leaving arbitrary quoted output whole.
            if rule is _PARAMETER_RULES[0] and (
                date_range := _QUOTED_DATE_RANGE.fullmatch(match.group(1))
            ):
                offset = match.start(1)
                for index in (1, 2):
                    date_span = date_range.span(index)
                    found.append(
                        (
                            offset + date_span[0],
                            offset + date_span[1],
                            date_range.group(index).lower(),
                        )
                    )
                continue
            # Higher-priority compound values (quoted text, ISO dates) own
            # their entire span; do not also extract their numeric components.
            if value and not any(span[0] < end and start < span[1] for start, end, _ in found):
                found.append((*span, value))
    return tuple(value for _, _, value in sorted(found))


def request_signature(instruction: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """A cheap, non-fuzzy paraphrase key: same named values, same action words.

    Two requests naming the same order, file or date are not necessarily the
    same request — "refund order 7741" and "cancel order 7741" share a value
    and nothing else. Two requests are treated as one only when they *also*
    share every content word once stopwords and the parameter values
    themselves are removed, which is true of true paraphrases ("refund order
    7741" / "please issue a refund for order 7741") and false whenever the
    verb, object or constraint differs.
    """
    parameters = request_parameter_values(instruction)
    normalized = normalize_instruction(instruction)
    remaining = normalized
    for value in parameters:
        remaining = remaining.replace(value, " ")
    content_words = frozenset(remaining.split()) - _STOPWORDS
    return parameters, content_words
