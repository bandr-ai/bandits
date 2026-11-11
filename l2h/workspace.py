"""Workspace layout, config, JSONL IO, the read-only raw record reader."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .output import L2HError, UsageError

CONFIG_NAME = "l2h.json"
DEFAULT_CONFIG = {
    "logs": None,
    "seed": 0,
    "sample_size": 3,
    "max_fix_attempts": 3,
    "max_field_chars": 500,
    "leak": {"min_len": 12, "run_len": 60},
    "gate_timeout_sec": 60,
    "harbor_timeout_sec": 1800,
}
TEXT_EXTS = {".txt", ".md", ".log", ".out", ".text"}
RECORD_EXTS = {".jsonl", ".ndjson", ".json", ".csv", ".tsv"} | TEXT_EXTS


def sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, text: str | bytes) -> None:
    """Write via a temp file in the same dir so an interrupt never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".l2h-tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode() if isinstance(text, str) else text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write(path, "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass
class Workspace:
    root: Path

    # layout
    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    @property
    def notes(self) -> Path:
        return self.root / "source_notes.md"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs.json"

    @property
    def parse_py(self) -> Path:
        return self.root / "parse.py"

    @property
    def checks_py(self) -> Path:
        return self.root / "checks.py"

    @property
    def tasktypes(self) -> Path:
        return self.root / "tasktypes"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions.jsonl"

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def internal(self) -> Path:
        return self.out / ".l2h"

    @property
    def trajectories(self) -> Path:
        return self.out / "trajectories.jsonl"

    @property
    def findings(self) -> Path:
        return self.out / "findings.jsonl"

    @property
    def excluded(self) -> Path:
        return self.out / "excluded.jsonl"

    @property
    def queue(self) -> Path:
        return self.out / "queue.jsonl"

    @property
    def tasks(self) -> Path:
        return self.out / "tasks"

    @property
    def results(self) -> Path:
        return self.out / "results.jsonl"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    @property
    def state_path(self) -> Path:
        return self.internal / "state.json"

    def task_dir(self, task_id: str) -> Path:
        for row in self.result_rows():
            if row["task_id"] == task_id:
                return self.tasks / row["type"] / task_id
        raise L2HError(
            f"unknown task {task_id!r}", "run `l2h status` or look in out/results.jsonl for ids"
        )

    # config / state
    def config(self) -> dict:
        cfg = read_json(self.config_path)
        if cfg is None:
            raise UsageError(
                f"{self.root} is not an l2h workspace (no {CONFIG_NAME})",
                "run `l2h init <logs> <workspace>` or pass --workspace",
            )
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        merged.update(cfg)
        return merged

    def logs_root(self) -> Path:
        logs = self.config().get("logs")
        if not logs:
            raise UsageError("workspace has no logs path", "re-run `l2h init <logs> <workspace>`")
        p = Path(logs)
        if not p.exists():
            raise L2HError(f"logs path {p} no longer exists", "restore it or re-init the workspace")
        return p

    def state(self) -> dict:
        return read_json(self.state_path, {})

    def update_state(self, **kw: Any) -> None:
        s = self.state()
        s.update(kw)
        write_json(self.state_path, s)

    # results
    def result_rows(self) -> list[dict]:
        return read_jsonl(self.results)

    def write_results(self, rows: list[dict]) -> None:
        write_jsonl(self.results, sorted(rows, key=lambda r: (r["type"], r["task_id"])))

    def decision_rows(self) -> list[dict]:
        return read_jsonl(self.decisions)


def find_workspace(arg: str | None) -> Workspace:
    """--workspace, then $L2H_WORKSPACE, then the nearest parent of cwd with l2h.json."""
    cand = arg or os.environ.get("L2H_WORKSPACE")
    if cand:
        return Workspace(Path(cand).resolve())
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / CONFIG_NAME).exists():
            return Workspace(p)
    return Workspace(cur)


# ---------------------------------------------------------------- raw records


def list_log_files(logs: Path) -> list[Path]:
    """Sorted, deterministic list of files under logs (a single file is allowed)."""
    if logs.is_file():
        return [logs]
    out = []
    for p in sorted(logs.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(logs).parts):
            out.append(p)
    return out


def rel(logs: Path, p: Path) -> str:
    return p.name if logs.is_file() else p.relative_to(logs).as_posix()


def iter_records(path: Path) -> Iterator[tuple[int, Any]]:
    """Yield (ordinal, record) for one native file, by file type.

    jsonl: one record per non-blank line (bad lines yield {"_unparsed": text}).
    json: array elements, or the single object. csv/tsv: one dict per row.
    other text: the whole file as one record {"text": ...}.
    Binary/unreadable files yield a single {"_binary": size} record.
    """
    ext = path.suffix.lower()
    try:
        if ext in (".jsonl", ".ndjson"):
            n = 0
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        yield n, json.loads(line)
                    except json.JSONDecodeError:
                        yield n, {"_unparsed": line.rstrip("\n")}
                    n += 1
        elif ext == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError as e:
                yield 0, {"_unparsed": f"invalid JSON: {e}"}
                return
            if isinstance(data, list):
                yield from enumerate(data)
            else:
                yield 0, data
        elif ext in (".csv", ".tsv"):
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                yield from enumerate(csv.DictReader(f, delimiter="\t" if ext == ".tsv" else ","))
        else:
            data = path.read_bytes()
            try:
                yield 0, {"text": data.decode("utf-8")}
            except UnicodeDecodeError:
                yield 0, {"_binary": len(data)}
    except OSError as e:
        raise L2HError(f"cannot read {path}: {e}", "check permissions on the logs folder") from e


def record_id(relfile: str, n: int) -> str:
    return f"{relfile}#{n}"


def split_record_id(rid: str) -> tuple[str, int] | None:
    if "#" not in rid:
        return None
    f, _, n = rid.rpartition("#")
    return (f, int(n)) if n.isdigit() else None


def get_raw_record(logs: Path, rid: str) -> Any:
    parts = split_record_id(rid)
    if not parts:
        raise L2HError(f"{rid!r} is not a record id (expected <file>#<n>)", "see `l2h sample`")
    f, n = parts
    p = logs / f if logs.is_dir() else logs
    if not p.exists():
        raise L2HError(f"no such log file {f!r}", "ids look like <relative file>#<ordinal>")
    for i, rec in iter_records(p):
        if i == n:
            return rec
    raise L2HError(f"{f} has no record #{n}", "use `l2h query --count` to see sizes")


def shape_of(rec: Any) -> str:
    if isinstance(rec, dict):
        return ",".join(sorted(map(str, rec.keys())))[:200]
    return type(rec).__name__


def clear_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
