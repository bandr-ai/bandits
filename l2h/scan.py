"""Level-1 content gates for one task package: structure, leak scan, secret scan,
plus non-blocking warnings (giveaways, missing alternatives, dangling /app paths)."""

from __future__ import annotations

import re
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
    """The label vocabulary when the check is an allowed set: expected + alternatives + wrong."""
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


def warnings_for(task_dir: Path, spec: dict, meta: dict) -> list[str]:
    out: list[str] = []
    check = spec.get("check")
    if check == "allowed_set" and not meta.get("alternatives"):
        out.append(
            "alternatives: allowed-set/label check with no `alternatives` given; "
            "list other valid answers the verifier must accept, or confirm there are none"
        )
    vocab_n = {leak.norm(v) for v in _vocab(spec, meta)}
    env = task_dir / "environment"
    if check != "forbidden" and env.is_dir():
        for p in sorted(q for q in env.rglob("*") if q.is_file()):
            text = leak.read_text(p)
            if text is None:
                continue
            for v in _expected_values(spec):
                if leak.norm(v) in vocab_n and leak.listed_everywhere(text, leak.norm(v)):
                    continue
                if re.search(rf"(?<![\w.]){re.escape(v.strip())}(?![\w.])", text):
                    rel = p.relative_to(task_dir).as_posix()
                    out.append(
                        f"giveaway: expected answer {leak.short(v)} appears verbatim in {rel}"
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
    return Level1(harbor.structure_check(task_dir), lk, sc, warnings_for(task_dir, spec, meta))


def apply(row: dict, res: Level1) -> None:
    """Record the level-1 outcome on a results row."""
    row["gates"]["structure"] = "fail" if res.structure else "pass"
    row["gates"]["leak"] = "fail" if res.leaks else "pass"
    row["gates"]["secrets"] = "fail" if res.secrets else "pass"
    probs = (
        res.structure
        + [f"leak: {x.kind} in {x.where}: {x.detail}" for x in res.leaks]
        + [f"secret: {x.kind} in {x.where}: {x.detail}" for x in res.secrets]
    )
    if probs:
        row["problems"] = probs
    else:
        row.pop("problems", None)
    if res.warnings:
        row["warnings"] = res.warnings
    else:
        row.pop("warnings", None)
