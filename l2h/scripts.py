"""Load agent-written workspace code and scaffold new files."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import JOB_SCOPES, TASK_SCOPES, Check, ContractError, safe_id
from .output import L2HError, UsageError
from .workspace import Workspace, sha

PARSER_TEMPLATE = '''"""parse.py: turn the raw logs into trajectories. Contract: `l2h schema`.

Run with `l2h run parse`. Keep unknown values as None; keep unfamiliar fields in meta.
A trajectory is an ATIF document (agent, steps with source/message/tool_calls/observation)
wrapped by Trajectory.from_atif, which adds the l2h block (id, source_refs, ended, meta).
"""
from collections.abc import Iterator
from pathlib import Path

from l2h import ParseError, Trajectory


def parse(paths: list[Path]) -> Iterator[Trajectory | ParseError]:
    # yield Trajectory.from_atif(
    #     {"agent": {"name": "...", "version": "unknown"},
    #      "steps": [{"source": "user", "message": "..."},
    #                {"source": "agent", "message": "...",
    #                 "tool_calls": [{"tool_call_id": "c1", "function_name": "...", "arguments": {}}],
    #                 "observation": {"results": [{"source_call_id": "c1", "content": "..."}]}}]},
    #     id="...", source_refs=[{"file": <path relative to logs>, "offset": <record #>}],
    #     ended="done" | "cut_off" | "error" | "unknown", meta={})
    # Never treat a clean exit as success: record how the run actually ended.
    # Logs that already are ATIF (Harbor agent/trajectory.json): see `l2h docs --topic parsing`.
    raise NotImplementedError("write the parser after reading samples (`l2h sample`)")
    yield  # pragma: no cover
'''

CHECKS_HEADER = '''"""checks.py: observations about each trajectory. Contract: `l2h schema`.

A finding is an observation, not an exclusion. Return pass, fail or unknown with evidence.
"""
from l2h import Check, Finding, Trajectory

CHECKS: list[Check] = []
'''

CHECK_TEMPLATE = """

def check_{name}(t: Trajectory) -> Finding:
    # return Finding("pass" | "fail" | "unknown", evidence=[refs], note="why")
    return Finding("unknown", note="not implemented")


CHECKS.append(Check("{name}", 1, check_{name}))
"""

TASKTYPE_TEMPLATE = '''"""Task type {name}. Contract: `l2h schema`. Generators follow Harbor's adapter pattern."""
from pathlib import Path

from l2h import Answer, Blocked, BuildResult, Candidate, Outcome, Trajectory

SPEC = {{
    "unit": "episode",            # episode | subtask
    "env": "evidence",            # evidence | harness | rebuilt | partial
    "job": "TODO",               # a job id from jobs.json
    "task_scope": "TODO",         # choose: full_job | subtask | decision (`l2h docs --topic tasktypes`)
    "check": "exact",             # exact | allowed_set | numeric | forbidden (negative tasks)
    "gold": "imitation",          # verified | computed | independent | solved | imitation | negative | none
    "hidden": ["answer key", "original final response"],
    "applies_when": "describe which records and findings this type can use",
}}
VERSION = 1


def applies(t: Trajectory, findings: dict) -> bool:
    """findings maps check id -> {{"result", "evidence", "note"}} for this trajectory.
    May also return (False, "reason") so the exclusion reason is recorded."""
    return False


def group_key(t: Trajectory) -> str:
    """Attempts sharing a key become one task."""
    return t["id"]


def build(group: list[Trajectory], out: Path) -> BuildResult | Candidate | Blocked:
    return Candidate("not implemented yet", gaps=["generator not written"])
    # A full job with its own verifier on the end state:
    # return BuildResult(
    #     task_id="...",
    #     instruction="...",                  # the original request; never the recorded answer
    #     files={{"environment/Dockerfile": "...", "tests/test.sh": "...",
    #            "solution/solve.sh": "..."}},  # solve.sh from a passing logged attempt
    #     env_source="rebuilt", oracle_kind="replay", grading_method="behavioral",
    #     gold_type="independent", grading_reason="...",
    #     wrong_answers=[Outcome("bash doing a plausible wrong thing")],
    #     gaps=[{{"what": "...", "affects": "fidelity"}}],
    # )
    # An answer-file task (label, number, short text):
    # return BuildResult(
    #     task_id="...",
    #     instruction="...",              # what to do; never the recorded answer
    #     evidence={{"notes.txt": "..."}},  # lands at /app/notes.txt; values may be bytes (images)
    #     answer=Answer(expected="...", check="exact"),
    #     wrong_answers=["a meaningful wrong answer"],
    #     gold_type="imitation",
    # )
'''


JOBS_TEMPLATE = """{
  "jobs": [
    {
      "id": "whole-run",
      "scope": "full_job",
      "description": "TODO: the job the logged agent was given, start to end (inputs -> output)",
      "where": "TODO: which trajectories contain it (a query expression over r, or prose)",
      "repeats": "TODO: how attempts at the same instance are recognised (or 'none')",
      "outcome": "TODO: how success/failure is known from the logs (or 'unknown')",
      "notes": ""
    },
    {
      "id": "TODO-sub-capability",
      "scope": "subtask",
      "parent": "whole-run",
      "description": "TODO: a sub-agent or step with its own inputs, tools and output",
      "where": "",
      "repeats": "",
      "outcome": "",
      "notes": ""
    }
  ]
}
"""

JOB_FIELDS = ("id", "scope", "description", "where", "repeats", "outcome")
# the scaffold's own placeholder values; any other text (even "build a TODO app") is the user's
JOB_PLACEHOLDERS = {
    v.strip()
    for job in json.loads(JOBS_TEMPLATE)["jobs"]
    for v in job.values()
    if isinstance(v, str) and v.startswith("TODO")
} | {"TODO"}


def load_jobs(ws: Workspace) -> dict[str, dict]:
    """The job map (jobs.json): every job the corpus contains, full jobs and sub-capabilities."""
    if not ws.jobs.exists():
        raise L2HError(
            "no jobs.json: map the jobs in the corpus before building task types",
            "`l2h new jobs`, then list the full job(s), sub-capabilities, repeats and outcomes",
        )
    try:
        data = json.loads(ws.jobs.read_text())
        jobs = data["jobs"]
        assert isinstance(jobs, list) and jobs
    except (ValueError, KeyError, AssertionError, TypeError) as e:
        raise L2HError(f"jobs.json is not valid: {e}", 'expected {"jobs": [ {...}, ... ]}') from e
    out: dict[str, dict] = {}
    for j in jobs:
        missing = [f for f in JOB_FIELDS if not str((j or {}).get(f, "")).strip()]
        if missing:
            raise L2HError(f"jobs.json: job {j.get('id')!r} is missing {missing}", "fill them in")
        left = [f for f, v in j.items() if isinstance(v, str) and v.strip() in JOB_PLACEHOLDERS]
        if left:
            raise L2HError(
                f"jobs.json: job {j['id']!r} still has the scaffold's TODO placeholder in {left}",
                "fill them in",
            )
        if j["scope"] not in JOB_SCOPES:
            raise L2HError(f"jobs.json: job {j['id']!r} scope must be one of {list(JOB_SCOPES)}")
        out[safe_id(str(j["id"]))] = j
    for j in out.values():
        if j.get("parent") and j["parent"] not in out:
            raise L2HError(f"jobs.json: job {j['id']!r} has unknown parent {j['parent']!r}")
    if not any(j["scope"] == "full_job" for j in out.values()):
        raise L2HError("jobs.json lists no full_job", "name the job the logged agent was given")
    return out


def scaffold(ws: Workspace, kind: str, name: str, force: bool = False) -> tuple[Path, str]:
    """Return (path, new_content) without writing; caller decides (dry-run)."""
    if kind == "parser":
        return ws.parse_py, PARSER_TEMPLATE
    if kind == "jobs":
        return ws.jobs, JOBS_TEMPLATE
    try:
        if kind not in ("parser", "jobs"):
            name = safe_id(name or "")
    except ContractError as e:
        raise UsageError(str(e), "use letters, digits, '_', '-', '.'") from e
    if kind == "check":
        name = name.replace("-", "_").replace(".", "_")
        if ws.checks_py.exists():
            cur = ws.checks_py.read_text()
            if f'Check("{name}"' in cur:
                raise L2HError(f"check {name!r} already exists in checks.py", "pick another name")
        else:
            cur = CHECKS_HEADER
        return ws.checks_py, cur + CHECK_TEMPLATE.format(name=name)
    if kind == "tasktype":
        return ws.tasktypes / f"{name}.py", TASKTYPE_TEMPLATE.format(name=name)
    raise UsageError(f"unknown kind {kind!r}", "use parser, jobs, check or tasktype")


@dataclass
class Loaded:
    module: Any
    path: Path
    digest: str


def load_module(path: Path, modname: str) -> Loaded:
    if not path.exists():
        raise L2HError(f"{path.name} not found", "create it with `l2h new` (see `l2h docs`)")
    text = path.read_text()
    spec = importlib.util.spec_from_file_location(f"l2h_ws_{modname}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    old_flag, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # never litter the workspace
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise L2HError(
            f"{path.name} failed to import: {type(e).__name__}: {e}",
            f"fix {path} (run `python {path}` to see the traceback)",
        ) from e
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = old_flag
    return Loaded(mod, path, sha(text))


def load_parser(ws: Workspace) -> Loaded:
    ld = load_module(ws.parse_py, "parse")
    if not callable(getattr(ld.module, "parse", None)):
        raise L2HError("parse.py has no parse(paths) function", "see `l2h schema`")
    return ld


def load_checks(ws: Workspace) -> tuple[Loaded, list[Check]]:
    ld = load_module(ws.checks_py, "checks")
    checks = getattr(ld.module, "CHECKS", None)
    if not isinstance(checks, list):
        raise L2HError(
            "checks.py must define CHECKS = [Check(id, version, fn), ...]", "see `l2h schema`"
        )
    seen = set()
    for c in checks:
        if not isinstance(c, Check) or not callable(c.fn):
            raise ContractError(f"CHECKS entries must be Check(id, version, fn), got {c!r}")
        if c.id in seen:
            raise ContractError(f"duplicate check id {c.id!r}")
        seen.add(c.id)
    return ld, checks


@dataclass
class TaskType:
    name: str
    loaded: Loaded

    @property
    def spec(self) -> dict:
        return getattr(self.loaded.module, "SPEC", {})

    @property
    def version(self) -> int:
        return int(getattr(self.loaded.module, "VERSION", 1))

    @property
    def generator(self) -> str:
        return f"{self.name}@{self.version}"


def load_tasktypes(ws: Workspace, only: str | None = None) -> list[TaskType]:
    if not ws.tasktypes.is_dir():
        raise L2HError("no tasktypes/ directory", "create one with `l2h new tasktype <name>`")
    jobs = load_jobs(ws)
    out = []
    for p in sorted(ws.tasktypes.glob("*.py")):
        if p.name.startswith("_") or (only and p.stem != only):
            continue
        ld = load_module(p, f"tt_{p.stem}")
        for fn in ("applies", "group_key", "build"):
            if not callable(getattr(ld.module, fn, None)):
                raise L2HError(f"tasktypes/{p.name} is missing {fn}()", "see `l2h schema`")
        if not isinstance(getattr(ld.module, "SPEC", None), dict):
            raise L2HError(f"tasktypes/{p.name} is missing SPEC", "see `l2h schema`")
        scope = ld.module.SPEC.get("task_scope")
        if scope not in TASK_SCOPES:
            what = "is still the scaffold's TODO" if scope == "TODO" else f"is {scope!r}"
            raise L2HError(
                f"tasktypes/{p.name}: SPEC['task_scope'] {what}; choose one of "
                f"{' | '.join(TASK_SCOPES)}",
                "full_job = the logged agent's whole job rebuilt; subtask = one sub-capability "
                "rebuilt; decision = evidence given, its decision asked "
                "(`l2h docs --topic tasktypes`)",
            )
        if ld.module.SPEC.get("job") not in jobs:
            raise L2HError(
                f"tasktypes/{p.name}: SPEC['job'] must name a job in jobs.json, "
                f"got {ld.module.SPEC.get('job')!r}",
                f"known jobs: {', '.join(jobs)}",
            )
        out.append(TaskType(p.stem, ld))
    if only and not out:
        raise L2HError(f"no task type {only!r}", "list them with `l2h status`")
    if not out:
        raise L2HError("tasktypes/ has no task types", "create one with `l2h new tasktype <name>`")
    return out
