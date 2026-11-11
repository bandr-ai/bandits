"""Leak scan over instruction.md + environment/.

Gold answers live only in tests/ (and solution/). This checks the solver-visible
side does not contain: answer keys / the original final response (`hidden`
strings), files named like answer keys, or byte-identical copies of anything in
tests/ or solution/. Legitimate evidence that merely shares words with the
answer is not flagged: short hidden values only match in labelled context
("answer: 42"), long ones only as whole-string or long verbatim runs, and a label
that is one item of a listed allowed answer set only in labelled context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

NAME_RE = re.compile(
    r"(^|[_\-./])(answers?|expected|solutions?|solve|gold|ground[_-]?truth|hidden|answer[_-]?key|oracle)([_\-./]|$)",
    re.I,
)
LABEL_RE = r"(?:answer|expected|solution|correct|gold|ground[ _-]?truth)"


@dataclass
class Leak:
    where: str
    kind: str
    detail: str

    def to_dict(self) -> dict:
        return {"where": self.where, "kind": self.kind, "detail": self.detail}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _visible_files(task_dir: Path) -> list[Path]:
    out = []
    instr = task_dir / "instruction.md"
    if instr.is_file():
        out.append(instr)
    env = task_dir / "environment"
    if env.is_dir():
        out += sorted(p for p in env.rglob("*") if p.is_file())
    return out


def read_text(p: Path) -> str | None:
    """File text, or None for binary files (undecodable or containing NUL bytes)."""
    data = p.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan(
    task_dir: Path,
    hidden: list[str],
    allow: list[str] | None = None,
    min_len: int = 12,
    run_len: int = 60,
    vocab: list[str] | None = None,
) -> list[Leak]:
    """`vocab`: the allowed answer set (labels), when the check is an allowed set. A hidden value
    that is a member of it and only ever appears as one item of a list (label vocabulary) in a
    file is not a leak by itself there; "answer: X" style mentions are still caught."""
    allow = allow or []
    vocab_n = {norm(str(v)) for v in vocab or [] if str(v).strip()}
    leaks: list[Leak] = []
    hidden = [str(h) for h in hidden if str(h).strip()]
    guarded: dict[str, str] = {}  # sha -> path under tests/ or solution/
    for sub in ("tests", "solution"):
        d = task_dir / sub
        if d.is_dir():
            for p in d.rglob("*"):
                if p.is_file():
                    guarded[hashlib.sha256(p.read_bytes()).hexdigest()] = (
                        f"{sub}/{p.relative_to(d)}"
                    )

    for p in _visible_files(task_dir):
        relp = p.relative_to(task_dir).as_posix()
        if relp in allow:
            continue
        if p.name != "instruction.md" and NAME_RE.search(p.name):
            leaks.append(
                Leak(relp, "named_like_answer", f"file name {p.name!r} looks like an answer key")
            )
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in guarded:
            leaks.append(Leak(relp, "copy_of_hidden_file", f"identical to {guarded[digest]}"))
        text = read_text(p)
        if text is None:
            continue
        ntext = norm(text)
        for h in hidden:
            nh = norm(h)
            if len(nh) >= min_len and not (nh in vocab_n and listed_everywhere(text, nh)):
                if nh in ntext:
                    leaks.append(Leak(relp, "hidden_string", f"contains hidden text {short(h)}"))
                    continue
                if len(nh) >= run_len * 1.5 and _shares_run(nh, ntext, run_len):
                    leaks.append(
                        Leak(
                            relp,
                            "verbatim_run",
                            f"shares a {run_len}+ char verbatim run with {short(h)}",
                        )
                    )
            else:
                pat = (
                    rf"\b{LABEL_RE}\b[\w ]{{0,12}}?[\"']?\s*(?:[:=]|\b(?:is|was|are|equals?)\b)"
                    rf"\s*[\"']?\s*{re.escape(nh)}(?![\w])"
                )
                if re.search(pat, ntext, re.I):
                    leaks.append(
                        Leak(
                            relp,
                            "labelled_answer",
                            f"short hidden value {short(h)} appears after an answer-like label",
                        )
                    )
    return leaks


ITEM_START = r"^\s*(?:[-*+\u2022|]|\d+[.)])\s*[`*\"']*$"
SEP_BEFORE = r"[\w`'\"]\s*(?:[,;|/]|\b(?:or|and)\b)\s*[`'\"]?$"
SEP_AFTER = r"^[`'\"]?\s*(?:[,;|/]|,?\s*\b(?:or|and)\b)\s*[`'\"]?\w"


def listed_everywhere(text: str, label_n: str) -> bool:
    """True if every occurrence of the label is one item of a list: separated from a sibling by
    , ; | / or/and, or opening a bullet/table line when the text has several of those."""
    pat = re.compile(
        rf"(?<!\w){re.escape(label_n).replace(chr(92) + ' ', chr(92) + 's+')}(?!\w)", re.I
    )
    hits = list(pat.finditer(text))
    if not hits:
        return False
    lines = text.splitlines()
    items = sum(1 for ln in lines if re.match(r"\s*(?:[-*+\u2022|]|\d+[.)])\s+\S", ln))
    for m in hits:
        before, after = text[max(0, m.start() - 40) : m.start()], text[m.end() : m.end() + 40]
        line_prefix = text[text.rfind("\n", 0, m.start()) + 1 : m.start()]
        listed = (
            re.search(SEP_BEFORE, before, re.I)
            or re.match(SEP_AFTER, after, re.I)
            or (items >= 2 and re.match(ITEM_START, line_prefix))
        )
        if not listed:
            return False
    return True


def short(s: str) -> str:
    s = s.replace("\n", " ")
    return repr(s if len(s) <= 40 else s[:37] + "...")


def _shares_run(needle: str, hay: str, n: int) -> bool:
    """True if any n-char window of needle (stepping by n//2) occurs in hay."""
    step = max(1, n // 2)
    for i in range(0, len(needle) - n + 1, step):
        if needle[i : i + n] in hay:
            return True
    return False
