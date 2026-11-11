"""Level-1 content gates for one task package: structure, leak scan, secret scan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import harbor, leak, secrets


@dataclass
class Level1:
    structure: list[str]
    leaks: list[leak.Leak]
    secrets: list[leak.Leak]
    warnings: list[str] = field(default_factory=list)


def scan_task(task_dir: Path, meta: dict, cfg: dict) -> Level1:
    """`meta` is the task's taskmeta (hidden, leak_allow, secret_allow, ...)."""
    lk = leak.scan(
        task_dir,
        meta.get("hidden", []),
        meta.get("leak_allow", []),
        cfg["leak"]["min_len"],
        cfg["leak"]["run_len"],
    )
    sc = secrets.scan(task_dir, meta.get("secret_allow", []))
    return Level1(harbor.structure_check(task_dir), lk, sc)


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
