"""Synthetic fixtures only: a tiny made-up corpus and workspace code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from l2h.cli import main
from l2h.gates import LocalRunner

PARSE_PY = """
import json
from l2h import ParseError, Trajectory

def parse(paths):
    for p in paths:
        if p.suffix != '.jsonl':
            continue
        n = 0
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    yield ParseError(p.name, f"bad json on record {n}", n)
                    n += 1
                    continue
                if rec.get("kind") == "qa":
                    if "prompt" not in rec:
                        yield ParseError(p.name, f"record {n} has no prompt", n)
                    else:
                        yield Trajectory.from_atif(
                            {"agent": {"name": "qa-bot", "version": "unknown"},
                             "steps": [
                                 {"source": "user", "message": rec["prompt"]},
                                 {"source": "agent", "message": "reading",
                                  "tool_calls": [{"tool_call_id": "c1", "function_name": "read",
                                                  "arguments": {"path": "nums.txt"}}],
                                  "observation": {"results": [
                                      {"source_call_id": "c1", "content": rec["nums"]}]}},
                                 {"source": "agent", "message": rec.get("answer") or ""},
                             ]},
                            id=rec["sid"],
                            source_refs=[{"file": p.name, "offset": n}],
                            attachments=[{"name": "nums.txt", "text": rec["nums"]}],
                            meta={"nums": rec["nums"], "answer": rec.get("answer")},
                            ended="done" if rec.get("outcome") == "ok" else "error",
                        )
                n += 1
"""

CHECKS_PY = """
from l2h import Check, Finding

def has_answer(t):
    ok = t["meta"].get("answer") not in (None, "")
    return Finding("pass" if ok else "fail", evidence=[t["source_refs"][0]], note="" if ok else "no answer")

def boom(t):
    raise RuntimeError("check bug in record 7")

CHECKS = [Check("has_answer", 1, has_answer), Check("boom", 1, boom)]
"""

JOBS_JSON = """{"jobs": [
  {"id": "main", "scope": "full_job", "description": "add up the numbers the user gave",
   "where": "every trajectory", "repeats": "same numbers", "outcome": "logged answer only"},
  {"id": "sub", "scope": "subtask", "parent": "main", "description": "read the numbers",
   "where": "every trajectory", "repeats": "none", "outcome": "unknown"}
]}
"""

TASKTYPE_PY = """
from l2h import Answer, Blocked, BuildResult

SPEC = {"unit": "episode", "env": "static", "job": "main", "task_scope": "decision", "check": "numeric", "gold": "imitation",
        "hidden": ["answer"], "applies_when": "has an answer"}
VERSION = 1
LEAK = False

def applies(t, findings):
    if findings["has_answer"]["result"] != "pass":
        return (False, "no logged answer to build a task from")
    return True

def group_key(t):
    return t["meta"]["nums"]

def build(group, out):
    t = group[0]
    nums = t["meta"]["nums"]
    if nums == "BLOCK":
        return Blocked("needs a live system", missing=["running service"])
    if nums == "CRASH":
        raise ValueError("generator bug for sample 9")
    instr = "Add up the numbers in /app/nums.txt."
    if LEAK:
        instr += f" (The logged answer was {t['meta']['answer']}.)"
    return BuildResult(
        task_id="t_" + t["id"],
        instruction=instr,
        evidence={"nums.txt": nums},
        answer=Answer(expected=t["meta"]["answer"], check="numeric", tolerance=0.01),
        wrong_answers=[str(float(t["meta"]["answer"]) + 1)],
        alternatives=[t["meta"]["answer"] + ".0"],
        assumptions=["logged answer taken as correct"],
    )
"""

RECORDS = [
    {
        "kind": "qa",
        "sid": "q1",
        "prompt": "sum these",
        "nums": "17\n25",
        "answer": "42",
        "outcome": "ok",
    },
    {
        "kind": "qa",
        "sid": "q2",
        "prompt": "sum these",
        "nums": "3\n4",
        "answer": "7",
        "outcome": "ok",
    },
    {
        "kind": "qa",
        "sid": "q3",
        "prompt": "sum these",
        "nums": "10\n11",
        "answer": "21",
        "outcome": "ok",
    },
    {
        "kind": "qa",
        "sid": "q4",
        "prompt": "sum these",
        "nums": "1\n1",
        "answer": "",
        "outcome": "error",
    },
    {
        "kind": "qa",
        "sid": "q5",
        "nums": "9\n9",
        "answer": "18",
        "outcome": "ok",
    },  # no prompt: parse error
    {
        "kind": "qa",
        "sid": "q6",
        "prompt": "sum these",
        "nums": "BLOCK",
        "answer": "5",
        "outcome": "ok",
    },
    {"kind": "heartbeat", "ts": None},  # lands in no trajectory
]


def write_logs(root: Path) -> Path:
    logs = root / "logs"
    logs.mkdir()
    lines = (
        [json.dumps(r) for r in RECORDS[:4]] + ["{not json"] + [json.dumps(r) for r in RECORDS[4:]]
    )
    (logs / "a.jsonl").write_text("\n".join(lines) + "\n")
    (logs / "notes.txt").write_text("just some free text, one record")
    return logs


class Env:
    def __init__(self, tmp: Path, capsys):
        self.tmp, self.capsys = tmp, capsys
        self.logs = write_logs(tmp)
        self.ws = tmp / "ws"

    def run(self, *args: str, ws: bool = True):
        a = list(args)
        if ws and a[0] != "init":
            a += ["--workspace", str(self.ws)]
        self.capsys.readouterr()
        code = main(a)
        cap = self.capsys.readouterr()
        parsed = None
        if "--json" in a and cap.out.strip().startswith("{"):
            parsed = json.loads(cap.out)
        return code, cap.out, cap.err, parsed

    def init(self):
        return self.run("init", str(self.logs), str(self.ws), "--allow-missing-runtime", "--json")

    def write(self, rel: str, text: str):
        p = self.ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def snapshot(self) -> dict:
        out = {}
        for p in sorted(self.tmp.rglob("*")):
            if p.is_file():
                out[p.relative_to(self.tmp).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    def build_pipeline(self, leak: bool = False, drop_block: bool = False):
        assert self.init()[0] == 0
        self.write("parse.py", PARSE_PY)
        self.write("checks.py", CHECKS_PY)
        self.write("jobs.json", JOBS_JSON)
        tt = TASKTYPE_PY.replace("LEAK = False", f"LEAK = {leak}")
        self.write("tasktypes/addnums.py", tt)
        assert self.run("run", "parse", "--json")[0] == 0
        assert self.run("run", "checks", "--json")[0] == 0
        return self


@pytest.fixture
def env(tmp_path, capsys):
    return Env(tmp_path, capsys)


@dataclass
class _HarborSim(LocalRunner):
    """Stands in for a real Harbor run in tests: host execution, but its outcomes carry
    trial ids the way Harbor's result.json does (the evidence `validated` requires)."""

    name: str = "harbor"

    def run_solution(self, task_dir):
        return replace(super().run_solution(task_dir), trial="sim-trial")

    def run_verifier(self, task_dir, answer):
        return replace(super().run_verifier(task_dir, answer), trial="sim-trial")


@pytest.fixture
def local_runner():
    """Opt out of the Harbor stand-in: use the real host-local runner."""


@pytest.fixture(autouse=True)
def _harbor_grade_runner(request, monkeypatch):
    if "local_runner" not in request.fixturenames:
        from l2h import gates

        monkeypatch.setattr(gates, "LocalRunner", _HarborSim)
    monkeypatch.setattr("l2h.gates.harbor_available", lambda: False)
