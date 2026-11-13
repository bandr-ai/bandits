"""Level-1 content gates for one task package: structure, leak scan, secret scan,
plus non-blocking warnings (giveaways, missing alternatives, dangling /app paths)."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import harbor, leak, secrets
from .workspace import read_json


@dataclass
class Level1:
    structure: list[str]
    leaks: list[leak.Leak]
    secrets: list[leak.Leak]
    warnings: list[str] = field(default_factory=list)


def _vocab(spec: dict, meta: dict) -> list[str]:
    """The label vocabulary: answer.choices when given, else for an allowed set
    expected + alternatives + wrong answers."""
    if spec.get("choices"):
        return [str(c) for c in spec["choices"]]
    if spec.get("check") != "allowed_set" or not isinstance(spec.get("expected"), list):
        return []
    return (
        [str(x) for x in spec["expected"]]
        + meta.get("alternatives", [])
        + meta.get("wrong_answers", [])
    )


def _expected_values(spec: dict) -> list[str]:
    exp = spec.get("expected")
    vals = exp if isinstance(exp, list) else [exp]
    return [str(v) for v in vals if v is not None and str(v).strip()]


def _whole_token(v: str, text: str) -> bool:
    """v as a whole token: not part of a longer word or number (a sentence-final period is ok)."""
    return bool(re.search(rf"(?<![\w.]){re.escape(v.strip())}(?!\w|\.\w)", text))


def warnings_for(task_dir: Path, spec: dict, meta: dict, min_len: int = 12) -> list[str]:
    out: list[str] = []
    check = spec.get("check")
    labelled = check == "allowed_set" or (check == "exact" and spec.get("choices"))
    if labelled and not meta.get("alternatives"):
        out.append(
            "alternatives: allowed-set/label check with no `alternatives` given; "
            "list other valid answers the verifier must accept, or confirm there are none"
        )
    vocab_n = {leak.norm(v) for v in _vocab(spec, meta)}
    env = task_dir / "environment"
    instr = task_dir / "instruction.md"
    files = [instr] if instr.is_file() else []
    if env.is_dir():
        files += sorted(q for q in env.rglob("*") if q.is_file())
    expected = _expected_values(spec)
    for p in files:
        text = leak.read_text(p)
        if text is None:
            continue
        rel = p.relative_to(task_dir).as_posix()
        is_instr = p == instr
        for v in expected:
            nv = leak.norm(v)
            if nv in vocab_n and leak.defines_vocab(text, nv, vocab_n):
                continue  # the file lists/defines the label vocabulary
            if is_instr and leak.listed_everywhere(text, nv):
                continue  # offered as one option of several ("yes or no")
            if is_instr and _whole_token(v, text):
                out.append(f"giveaway: expected answer {leak.short(v)} appears verbatim in {rel}")
        if is_instr:
            # a short hidden value only fails the leak gate in labelled context; any bare
            # occurrence in the instruction is still worth a look
            exp_n = {leak.norm(v) for v in expected}
            for h in dict.fromkeys(str(x) for x in meta.get("hidden", []) if str(x).strip()):
                nh = leak.norm(h)
                if len(nh) >= min_len or nh in exp_n:
                    continue
                if nh in vocab_n and leak.defines_vocab(text, nh, vocab_n):
                    continue
                if leak.listed_everywhere(text, nh):
                    continue
                if _whole_token(h, text):
                    out.append(
                        f"hidden_value: short hidden value {leak.short(h)} appears in {rel}; "
                        "make sure the instruction does not give the answer away"
                    )
    instr = task_dir / "instruction.md"
    if instr.is_file():
        refs = {
            m.rstrip(".,;:)'\"`")
            for m in re.findall(r"/app/[^\s`'\"<>()\[\]{}*?]+", instr.read_text())
        }
        for ref in sorted(refs):
            sub = ref[len("/app/") :]
            if not sub or sub == "answer" or sub.startswith("answer/"):
                continue
            if not any((env / base / sub).exists() for base in ("data", ".")):
                out.append(
                    f"missing_file: instruction.md references {ref} but the package has no such file"
                )
    return out


def scan_task(task_dir: Path, meta: dict, cfg: dict) -> Level1:
    """`meta` is the task's taskmeta (hidden, leak_allow, secret_allow, ...)."""
    spec = read_json(task_dir / "tests" / "expected.json", {})
    lk = leak.scan(
        task_dir,
        meta.get("hidden", []),
        meta.get("leak_allow", []),
        cfg["leak"]["min_len"],
        cfg["leak"]["run_len"],
        _vocab(spec, meta),
    )
    sc = secrets.scan(task_dir, meta.get("secret_allow", []))
    return Level1(
        harbor.structure_check(task_dir),
        lk,
        sc,
        warnings_for(task_dir, spec, meta, cfg["leak"]["min_len"]),
    )


def apply(row: dict, res: Level1) -> None:
    """Record the level-1 outcome on a results row."""
    row["gates"]["structure"] = "fail" if res.structure else "pass"
    row["gates"]["leak"] = "fail" if res.leaks else "pass"
    row["gates"]["secrets"] = "fail" if res.secrets else "pass"
    probs = (
        res.structure
        + [f"leak: {x.kind} in {x.where}: {x.detail}" for x in res.leaks]
        + [f"secret scan: {x.kind} in {x.where}: {x.detail}" for x in res.secrets]
    )
    if probs:
        row["problems"] = probs
    else:
        row.pop("problems", None)
    if res.warnings:
        row["warnings"] = res.warnings
    else:
        row.pop("warnings", None)


def withhold_secrets(task_dir: Path, row: dict) -> None:
    """Do not leave credential-blocked packages in the export directory.

    Keep the redacted gate findings and provenance; fixing the generator and
    regenerating is the only way to restore a removed package.
    """
    if row["gates"].get("secrets") == "fail":
        shutil.rmtree(task_dir)
        row["withheld"] = "secret scan failed; package removed from out/tasks; fix and regenerate"

        # Derived messages can quote instruction fragments (e.g. a missing-file
        # warning). Do not re-export the detected credential through diagnostics.
        def scrub(value):
            if isinstance(value, str):
                return "[redacted: credential-like text]" if secrets.scan_text(value) else value
            if isinstance(value, dict):
                return {key: scrub(item) for key, item in value.items()}
            if isinstance(value, list):
                return [scrub(item) for item in value]
            return value

        cleaned = scrub(row)
        row.clear()
        row.update(cleaned)
    else:
        row.pop("withheld", None)
