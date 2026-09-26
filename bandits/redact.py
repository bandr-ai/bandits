"""Detect and redact common secrets and PII before trace bytes are parsed or stored.

Runs on raw source bytes so a secret never reaches the artifact store, even
transiently. Every match is found against the *original* bytes in a single pass
and the output is rebuilt once, so a replacement that shortens the text cannot
shift the reported location of a later one.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bandits.traces import TraceIssue

_REPLACEMENT = b"[REDACTED:%s]"
_NEWLINE = b"\n"


@dataclass(frozen=True)
class Rule:
    """One detector. ``group`` selects which part of the match is replaced.

    Group 0 replaces the whole match. A named-value rule replaces only the value
    group so that the key it was stored under survives and stays readable.
    """

    kind: str
    pattern: re.Pattern[bytes]
    group: int = 0
    accept: Callable[[re.Match[bytes]], bool] | None = None
    """Optional check applied after matching.

    Judgement that would make a pattern ambiguous belongs here instead. Encoding
    it in the regex is how a rule ends up backtracking for a minute and a half
    over a large file before matching nothing at all.
    """


@dataclass(frozen=True)
class RedactionRuleset:
    """A named, versioned set of rules.

    The name is recorded on the corpus. Without it, changing a rule would produce
    a different corpus from the same source bytes with nothing to explain why.
    """

    name: str
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class RedactedSource:
    data: bytes
    source_digest: str
    ruleset: str
    issues: tuple[TraceIssue, ...]


# These intentionally target high-confidence forms. Broad guesses (for example,
# every long number) would destroy useful trace evidence and create silent bias.
_PRIVATE_KEY = Rule(
    "private_key",
    re.compile(
        rb"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
    ),
)
_BEARER_TOKEN = Rule("bearer_token", re.compile(rb"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{12,}"))
_OPENAI_KEY = Rule("openai_api_key", re.compile(rb"\bsk-[A-Za-z0-9_-]{12,}\b"))
_AWS_KEY = Rule("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"))
_ROLE_LOCALS = frozenset({b"git", b"noreply", b"no-reply", b"donotreply", b"do-not-reply"})
_RESERVED_DOMAINS = (b"example.com", b"example.org", b"example.net", b"localhost")


def _is_personal_address(match: re.Match[bytes]) -> bool:
    """Reject role addresses, reserved domains, and anything without a real TLD.

    Measured on a real 20 MB session log: without this, the rule matched 64
    times, of which 4 were actual addresses. The rest were git remotes, doc
    examples, and Python decorators.
    """
    local, _, domain = match.group().partition(b"@")
    if local.lower() in _ROLE_LOCALS or domain.lower().endswith(_RESERVED_DOMAINS):
        return False
    tld = domain.rpartition(b".")[2]
    return 2 <= len(tld) <= 24 and tld.isalpha()


_EMAIL = Rule(
    "email_address",
    # Possessive quantifiers throughout: a dotted run in source code otherwise
    # offers this pattern exponentially many ways to split, and it spends them.
    # Not preceded by a backslash either — inside JSON, "\n@pytest.mark.x" offers
    # "n@pytest.mark.x" as a well-formed address, and consuming that leading "n"
    # orphans the escape and destroys the whole record.
    re.compile(
        rb"(?<![\\A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}+@[A-Za-z0-9-]{1,63}+(?:\.[A-Za-z0-9-]{1,63}+)++"
    ),
    accept=_is_personal_address,
)

# The value runs to its closing quote rather than to the first space: a secret
# containing a space would otherwise be only partly replaced.
_NAMED_VALUE = Rule(
    "named_secret",
    re.compile(
        rb"(?i)([\"']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)"
        rb"[\"']?\s*[:=]\s*[\"'])(?!\[REDACTED:)([^\"'\n]{4,}?)([\"'])"
    ),
    group=2,
)

_SECRET_RULES: tuple[Rule, ...] = (
    _PRIVATE_KEY,
    _BEARER_TOKEN,
    _OPENAI_KEY,
    _AWS_KEY,
    _NAMED_VALUE,
)

DEFAULT_RULESET = RedactionRuleset("default-v1", _SECRET_RULES + (_EMAIL,))

# An email is sometimes the task's own identifier rather than incidental personal
# data — the instruction names an account, and redacting it leaves an instruction
# that identifies nothing. Callers who need those instructions intact can drop
# that one rule and still redact every secret.
SECRETS_ONLY_RULESET = RedactionRuleset("secrets-only-v1", _SECRET_RULES)

_RULESETS = {r.name: r for r in (DEFAULT_RULESET, SECRETS_ONLY_RULESET)}


def ruleset_by_name(name: str) -> RedactionRuleset:
    if name not in _RULESETS:
        raise ValueError(f"unknown redaction ruleset {name!r}; known: {sorted(_RULESETS)}")
    return _RULESETS[name]


def _matches(data: bytes, ruleset: RedactionRuleset) -> list[tuple[int, int, str]]:
    """Every span to replace, resolved against the original bytes and non-overlapping."""
    spans = [
        (*match.span(rule.group), rule.kind)
        for rule in ruleset.rules
        for match in rule.pattern.finditer(data)
        if match.span(rule.group) != (-1, -1)
        and (rule.accept is None or rule.accept(match))
        # A replacement starting one byte after a backslash leaves that backslash
        # attached to the marker. In JSON that is an invalid escape, and the
        # record carrying it is lost entirely rather than merely redacted.
        and not _follows_escape(data, match.span(rule.group)[0])
    ]
    # Longest match wins where two rules overlap, so a key inside a larger
    # credential block is not replaced twice or split in half.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    kept: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, kind in spans:
        if start >= last_end:
            kept.append((start, end, kind))
            last_end = end
    return kept


def _follows_escape(data: bytes, start: int) -> bool:
    return start > 0 and data[start - 1 : start] == b"\\"


def redact_source(path: Path, ruleset: RedactionRuleset = DEFAULT_RULESET) -> RedactedSource:
    """Read *path*, returning safe bytes while hashing the exact original bytes."""
    original = path.read_bytes()
    spans = _matches(original, ruleset)

    chunks: list[bytes] = []
    issues: list[TraceIssue] = []
    cursor = 0
    # Matches arrive in order, so each line number is counted on from the last
    # one; counting from the top every time is quadratic on a large export.
    line, counted_to = 1, 0
    for start, end, kind in spans:
        chunks.append(original[cursor:start])
        chunks.append(_REPLACEMENT % kind.encode())
        cursor = end
        line += original.count(_NEWLINE, counted_to, start)
        counted_to = start
        issues.append(
            TraceIssue(
                kind="redaction",
                detail=f"redacted detected {kind}",
                location=f"{path}:{line}",
            )
        )
    chunks.append(original[cursor:])

    return RedactedSource(
        data=b"".join(chunks),
        source_digest=hashlib.sha256(original).hexdigest(),
        ruleset=ruleset.name,
        issues=tuple(issues),
    )
