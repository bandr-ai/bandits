"""Secret scan over instruction.md, task.toml and environment/.

Flags likely credentials: well-known API-key/token shapes, private keys, bearer
tokens, `password`/`secret`/`token`/`api_key` names (also inside env-style names like
DB_PASSWORD) followed by a literal value, "enter <value> in the password field" prose, and
URLs with embedded credentials. A hit reports the file, line and a redacted preview, never
the secret. Binary files are skipped. Allow a legitimate hit per file with
`BuildResult.secret_allow=["environment/data/x"]` (logged in results.jsonl).
"""

from __future__ import annotations

import re
from pathlib import Path

from .leak import Leak, read_text

KEY_PATTERNS = {
    "aws_access_key": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "api_key": r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}",
    "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_\-]{30,}",
    "jwt": r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",
    "private_key": r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----",
    "bearer_token": r"\b[Bb]earer\s+[A-Za-z0-9._~+/=\-]{16,}",
}
# password: hunter2 / secret = "abc123" / "token": "abcd1234efgh" / DB_PASSWORD=... /
# AWS_SECRET_ACCESS_KEY=...: the keyword may sit inside an env-style name (prefix any word chars,
# suffix only after _ or -, so `tokenizer`/`secretary` are not keywords)
ASSIGN_RE = re.compile(
    r"""(?ix)(?<![A-Za-z0-9])\w*?(?:passw(?:or)?d|pwd|secret|token|api[_-]?key)(?P<suffix>[_-]\w*)?
    ["']?\s*[:=]\s*["']?(?P<val>[^\s"',;)}\]]{6,})"""
)
# prose: "enter X7#kq2 in the Password field" / "type `s3cret!` into the token box"
PROSE_RE = re.compile(
    r"""(?ix)\b(?:enter|type|input)\s+["'`]?(?P<val>[^\s"'`]{6,})["'`]?\s+(?:in|into)\s+
    (?:the\s+)?(?:\w+\s+)?(?:password|passcode|secret|token|api[ _-]?key)\b"""
)
# names like TOKEN_URL, password_min_length, SECRET_FILE describe a setting, not a secret value
SETTING_SUFFIX_RE = re.compile(
    r"(?i)[_-](?:\w*[_-])?(?:url|uri|endpoint|file|path|dir|name|type|len|length|count|limit|"
    r"max|min|header|field|env|var|ttl|expiry|expires|timeout)$"
)
URL_RE = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:(?P<val>[^/\s@]+)@", re.I)
PLACEHOLDER_RE = re.compile(
    r"(?i)^(?:<.*>|\$\{?\w+\}?|\{\{.*\}\}|\*+|x+|\.+|redacted|changeme|example|your[_-].*|null|none|"
    r"true|false|env\(.*|os\.environ.*|.*\bplaceholder\b.*)$"
)


def _preview(m: str) -> str:
    return f"{m[:4]}...[redacted, {len(m)} chars]" if len(m) > 8 else "[redacted]"


CODE_RE = re.compile(r"[(\[]|^[A-Za-z_]\w*\.[A-Za-z_]")  # a call, an index, attribute access


def _literal(val: str) -> bool:
    """A real-looking value: not a placeholder, not a plain lowercase word, not code."""
    return not (PLACEHOLDER_RE.match(val) or re.fullmatch(r"[a-z]+", val) or CODE_RE.search(val))


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """-> [(line number, kind, redacted preview)]"""
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for kind, pat in KEY_PATTERNS.items():
            for m in re.finditer(pat, line):
                hits.append((n, kind, _preview(m.group(0))))
        for m in ASSIGN_RE.finditer(line):
            val = m.group("val")
            # token_count: 123456 / password_min_length: 12345678 are settings, not secrets
            suffix = m.group("suffix")
            if suffix and (re.fullmatch(r"[\d.]+", val) or SETTING_SUFFIX_RE.search(suffix)):
                continue
            if _literal(val):
                hits.append((n, "credential_assignment", _preview(val)))
        for m in PROSE_RE.finditer(line):
            if _literal(m.group("val")):
                hits.append((n, "credential_in_prose", _preview(m.group("val"))))
        for m in URL_RE.finditer(line):
            if _literal(m.group("val")):
                hits.append((n, "url_credentials", _preview(m.group("val"))))
    return hits


def secret_files(task_dir: Path) -> list[Path]:
    out = [task_dir / n for n in ("instruction.md", "task.toml") if (task_dir / n).is_file()]
    env = task_dir / "environment"
    if env.is_dir():
        out += sorted(p for p in env.rglob("*") if p.is_file())
    return out


def scan(task_dir: Path, allow: list[str] | None = None) -> list[Leak]:
    allow = allow or []
    found: list[Leak] = []
    for p in secret_files(task_dir):
        relp = p.relative_to(task_dir).as_posix()
        text = None if relp in allow else read_text(p)
        if text is None:  # allowed, or binary
            continue
        seen = set()
        for line, kind, prev in scan_text(text):
            if (line, kind) not in seen:
                seen.add((line, kind))
                found.append(Leak(relp, kind, f"line {line}: {prev}"))
    return found
