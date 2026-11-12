"""Contract types shared between mint and agent-written workspace code.

Everything here round-trips through plain JSON: agent code may return the
dataclasses or plain dicts, and mint validates and normalises both.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ATIF_SOURCES = ("system", "user", "agent")
ENDED = ("done", "cut_off", "error", "unknown")
RESULTS = ("pass", "fail", "unknown")
# gold = why the grading is trusted (grading trust); see guides/tasktypes.md
GOLD_TYPES = ("verified", "computed", "independent", "solved", "imitation", "none")
CHECK_KINDS = ("exact", "allowed_set", "numeric")
GRADING_METHODS = (*CHECK_KINDS, "behavioral", "judge")  # how the verifier grades
NORMALIZERS = ("strip", "lower", "collapse_ws", "percent", "thousands", "sorted_list")
STATUSES = ("validated", "generated", "exec_failed", "blocked")
GATE_STATES = ("pass", "fail", "error", "not_run", "no_oracle")
# where solution/solve.sh comes from: a logged attempt replayed, a fresh solve, a known-good
# reference, a script that writes the expected answer (proves grader wiring only), or none
ORACLE_KINDS = ("replay", "fresh", "reference", "answer", "none")
# where the starting environment comes from: mint's default image + evidence files, a supplied
# harness, rebuilt from the traces, or rebuilt with known gaps
ENV_SOURCES = ("evidence", "harness", "rebuilt", "partial")
GAP_AFFECTS = ("execution", "fidelity", "grading", "none")
# the only reasons a job may be Blocked; anything else is a Candidate with gaps
BLOCK_KINDS = ("secret", "leak", "no_instruction")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class ContractError(ValueError):
    """Agent-written code returned something that breaks the contract."""


def safe_id(value: str) -> str:
    """Return value if usable as a directory/task name, else raise."""
    if not isinstance(value, str) or not _ID_RE.match(value) or ".." in value:
        raise ContractError(
            f"id {value!r} must match [A-Za-z0-9][A-Za-z0-9_.-]* (it becomes a directory name)"
        )
    return value


# The ATIF (Agent Trajectory Interchange Format) version mint targets. Harbor's model
# accepts ATIF-v1.0 .. v1.8; documents without a schema_version get this one.
ATIF_VERSION = "ATIF-v1.8"
EXT_KEY = "mint"  # extra["mint"]: what mint needs and ATIF lacks
_EXT_FIELDS = ("id", "source_refs", "ended", "meta", "attachments")


def message_text(msg: Any) -> str:
    """Text of an ATIF message / observation content: a string, or the text parts of a list."""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        return "\n".join(p["text"] for p in msg if isinstance(p, dict) and p.get("text"))
    return ""


@dataclass
class Call:
    """A tool call and the observation result matched to it by tool_call_id."""

    step: dict
    id: str
    name: str
    arguments: dict
    result: dict | None  # the matching observation result, None when never answered

    @property
    def output(self) -> str | None:
        return None if self.result is None else message_text(self.result.get("content"))


class Trajectory(dict):
    """A parsed trajectory: an ATIF trajectory dict plus the mint block in extra["mint"].

    ATIF keys (schema_version, agent, steps, subagent_trajectories, ...) are plain dict keys.
    The mint fields id, source_refs, ended, meta, attachments also read as t["meta"], t.meta.
    """

    def _ext(self) -> dict:
        extra = dict.get(self, "extra")
        ext = extra.get(EXT_KEY) if isinstance(extra, dict) else None
        return ext if isinstance(ext, dict) else {}

    def __missing__(self, key: str) -> Any:
        ext = self._ext()
        if key in _EXT_FIELDS and key in ext:
            return ext[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        # `"meta" in t` agrees with t["meta"]; keys()/iteration stay the stored ATIF keys, so
        # dict(t) / json round-trips do not duplicate the mint block at top level
        return dict.__contains__(self, key) or (key in _EXT_FIELDS and key in self._ext())

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    @classmethod
    def from_atif(
        cls,
        doc: dict,
        *,
        id: str,
        source_refs: list,
        ended: str = "unknown",
        meta: dict | None = None,
        attachments: list | None = None,
    ) -> Trajectory:
        """Wrap an ATIF document (e.g. a Harbor agent/trajectory.json) with the mint block."""
        t = cls(doc)
        t.setdefault("schema_version", ATIF_VERSION)
        for i, s in enumerate(t.get("steps") or [], 1):
            if isinstance(s, dict):
                s.setdefault("step_id", i)
        extra = dict(t.get("extra") or {})
        extra[EXT_KEY] = {
            "id": id,
            "source_refs": source_refs,
            "ended": ended,
            "meta": meta or {},
            "attachments": attachments or [],
        }
        t["extra"] = extra
        return t

    def flat(self) -> dict:
        """ATIF keys plus the mint fields at top level: the view `r` has in query/sample."""
        return {**self, **self._ext()}

    def calls(self) -> list[Call]:
        """Every tool call in step order, each with its matched observation result.

        Results match calls by source_call_id == tool_call_id. Only in a step with no ids at
        all (no call has tool_call_id, no result has source_call_id) and as many results as
        calls do they pair by position; any partial ids make position ambiguous, so unmatched
        calls stay unanswered rather than taking another call's output.
        """
        out = []
        for s in self.get("steps") or []:
            calls = [c for c in s.get("tool_calls") or [] if isinstance(c, dict)]
            res = [
                r for r in (s.get("observation") or {}).get("results") or [] if isinstance(r, dict)
            ]
            by_id = {r["source_call_id"]: r for r in res if r.get("source_call_id")}
            unkeyed = [r for r in res if not r.get("source_call_id")]
            no_ids = not by_id and not any(c.get("tool_call_id") for c in calls)
            by_pos = (
                dict(enumerate(unkeyed)) if no_ids and calls and len(calls) == len(unkeyed) else {}
            )
            for i, c in enumerate(calls):
                cid = c.get("tool_call_id")
                r = by_id.get(cid) if cid in by_id else by_pos.get(i)
                out.append(Call(s, cid, c.get("function_name"), c.get("arguments") or {}, r))
        return out

    @property
    def final_message(self) -> str | None:
        """Text of the last agent step if it made no tool call (the run's final answer)."""
        agent = [s for s in self.get("steps") or [] if s.get("source") == "agent"]
        if not agent or agent[-1].get("tool_calls"):
            return None
        return message_text(agent[-1].get("message")) or None

    @property
    def subagents(self) -> list[Trajectory]:
        """Embedded subagent trajectories (ATIF subagent_trajectories), same accessors."""
        return [Trajectory(s) for s in self.get("subagent_trajectories") or []]


@dataclass
class ParseError:
    """A record the parser could not turn into a trajectory."""

    file: str
    error: str
    offset: int | None = None
    kind: str = "parse_error"

    def to_dict(self) -> dict:
        return {"file": self.file, "offset": self.offset, "error": self.error, "kind": self.kind}


def _atif_model():
    """Harbor's ATIF pydantic model, or None when harbor is not importable."""
    try:
        from harbor.models.trajectories import Trajectory as AtifTrajectory
    except Exception:
        return None
    return AtifTrajectory


def validate_atif(doc: dict) -> None:
    """Raise ContractError unless doc is a valid ATIF trajectory.

    Always checks the basics; additionally runs Harbor's own model when it is importable.
    """
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ContractError("not ATIF: 'steps' must be a non-empty list")
    if not all(isinstance(s, dict) and s.get("source") in ATIF_SOURCES for s in steps):
        raise ContractError(f"not ATIF: every step needs a dict with source in {ATIF_SOURCES}")
    model = _atif_model()
    if model is None:
        return
    try:
        model.model_validate(doc)
    except Exception as e:
        errs = getattr(e, "errors", None)
        if not errs:
            raise ContractError(f"not ATIF: {e}") from e
        first = errs()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise ContractError(f"not ATIF: {loc}: {first['msg']}") from e


def normalize_trajectory(obj: Any) -> Trajectory:
    """Validate a parse() result: a valid ATIF trajectory carrying the mint block."""
    if not isinstance(obj, dict):
        raise ContractError(f"parse() yielded {type(obj).__name__}; expected Trajectory or dict")
    if "events" in obj and "steps" not in obj:
        raise ContractError(
            "parse() yielded the pre-ATIF format (`events`); build each trajectory with "
            "Trajectory.from_atif(doc, id=..., source_refs=..., ended=..., meta=...) where doc has "
            "ATIF `agent` and `steps` (see `mint docs --topic parsing`)"
        )
    t = Trajectory(obj)
    ext = t._ext()
    if not isinstance(ext, dict) or ext.get("id") in (None, ""):
        raise ContractError(
            f"trajectory is missing extra['{EXT_KEY}']['id'] (Trajectory.from_atif)"
        )
    ext["id"] = str(ext["id"])
    tid = ext["id"]
    for key in ("source_refs", "attachments"):
        if not isinstance(ext.setdefault(key, []), list):
            raise ContractError(f"trajectory {tid}: '{key}' must be a list")
    if not isinstance(ext.setdefault("meta", {}), dict):
        raise ContractError(f"trajectory {tid}: 'meta' must be an object")
    if ext.setdefault("ended", "unknown") not in ENDED:
        raise ContractError(f"trajectory {tid}: 'ended' must be one of {ENDED}")
    if not ext["source_refs"]:
        raise ContractError(
            f"trajectory {tid}: 'source_refs' is empty; every trajectory must trace to the raw logs"
        )
    for ref in ext["source_refs"]:
        if not isinstance(ref, dict) or "file" not in ref or not ("offset" in ref or "key" in ref):
            raise ContractError(
                f"trajectory {tid}: each source_ref needs 'file' and 'offset' or 'key', got {ref!r}"
            )
    validate_atif(t)
    return t


@dataclass
class Finding:
    result: str
    evidence: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {"result": self.result, "evidence": list(self.evidence), "note": self.note}


def normalize_finding(obj: Any) -> Finding:
    if isinstance(obj, Finding):
        f = obj
    elif isinstance(obj, dict):
        f = Finding(obj.get("result"), list(obj.get("evidence") or []), str(obj.get("note") or ""))
    else:
        raise ContractError(f"check returned {type(obj).__name__}; expected Finding or dict")
    if f.result not in RESULTS:
        raise ContractError(f"finding result {f.result!r} must be one of {RESULTS}")
    return f


@dataclass
class Check:
    id: str
    version: int
    fn: Callable[[Trajectory], Any]


@dataclass
class Answer:
    """What the verifier compares the solver's answer file against."""

    expected: Any
    check: str = "exact"
    file: str = "answer.txt"
    normalize: list = field(default_factory=lambda: ["strip"])
    tolerance: float = 0.0
    rel_tolerance: float = 0.0
    choices: list = field(default_factory=list)  # label vocabulary shown to the solver, if any

    def validate(self) -> None:
        if self.check not in CHECK_KINDS:
            raise ContractError(f"answer.check {self.check!r} must be one of {CHECK_KINDS}")
        bad = [n for n in self.normalize if n not in NORMALIZERS]
        if bad:
            raise ContractError(f"unknown normalizers {bad}; allowed {NORMALIZERS}")
        if self.check == "allowed_set" and not (isinstance(self.expected, list) and self.expected):
            raise ContractError("allowed_set needs a non-empty list as answer.expected")
        if self.choices:
            if not all(isinstance(c, str) and c.strip() for c in self.choices):
                raise ContractError("answer.choices must be a list of non-empty strings")
            exp = self.expected if isinstance(self.expected, list) else [self.expected]
            missing = [e for e in exp if str(e).strip() not in [c.strip() for c in self.choices]]
            if self.check in ("exact", "allowed_set") and missing:
                raise ContractError(f"answer.expected {missing} not in answer.choices")
        if self.check == "exact" and isinstance(self.expected, list | dict):
            raise ContractError("exact needs a scalar expected answer; use allowed_set for lists")
        if not self.file.strip() or "/" in self.file or self.file.startswith("."):
            raise ContractError("answer.file must be a plain file name")

    def to_dict(self) -> dict:
        return {
            "expected": self.expected,
            "check": self.check,
            "file": self.file,
            "normalize": list(self.normalize),
            "tolerance": self.tolerance,
            "rel_tolerance": self.rel_tolerance,
            "choices": list(self.choices),
        }

    def canonical(self) -> str:
        """The single value an oracle writes."""
        return str(self.expected[0] if isinstance(self.expected, list) else self.expected)


@dataclass
class Outcome:
    """An executable outcome for the exec gates: a bash script run in the task environment
    in place of solution/solve.sh (e.g. start a broken server, build a wrong binary).
    Used as a wrong outcome (must fail the verifier) or an alternative (must pass)."""

    script: str
    note: str = ""
    # free text: where this outcome comes from ("logged attempt X, outcome known from ...",
    # "synthetic shortcut"); shown in `mint report`, never gates anything
    source: str = ""

    def to_dict(self) -> dict:
        return {"script": self.script, "note": self.note, "source": self.source}


def normalize_outcome(x: Any) -> str | dict:
    """A wrong answer / alternative: a plain answer string, or an Outcome as a dict."""
    if isinstance(x, Outcome):
        x = x.to_dict()
    if isinstance(x, dict):
        if not isinstance(x.get("script"), str) or not x["script"].strip():
            raise ContractError("an Outcome needs a non-empty bash `script`")
        return {
            "script": x["script"],
            "note": str(x.get("note") or ""),
            "source": str(x.get("source") or ""),
        }
    if isinstance(x, str | int | float):
        return str(x)
    raise ContractError(
        f"wrong answers / alternatives must be strings or Outcome, got {type(x).__name__}"
    )


def normalize_gaps(gaps: list) -> list[dict]:
    """Gaps as [{"what", "affects"}]; a bare string is assumed to affect fidelity."""
    out = []
    for g in gaps or []:
        if isinstance(g, str):
            g = {"what": g, "affects": "fidelity"}
        if not isinstance(g, dict) or not str(g.get("what") or "").strip():
            raise ContractError(f"a gap needs a non-empty 'what', got {g!r}")
        aff = g.get("affects", "fidelity")
        if aff not in GAP_AFFECTS:
            raise ContractError(f"gap affects {aff!r} must be one of {GAP_AFFECTS}")
        out.append({"what": str(g["what"]), "affects": aff})
    return out


@dataclass
class BuildResult:
    """Returned by a tasktype's build() for one group of trajectories.

    files: extra/override files, keys relative to the task dir. Anything mint can
    generate (task.toml, tests/*, solution/solve.sh, environment/Dockerfile) is
    filled in when absent. `instruction` and `evidence` are conveniences:
    evidence keys land in environment/data/<key> and appear at /app/<key>.

    Grading: either `answer` (mint writes an answer-file verifier) or your own
    files["tests/test.sh"] (custom verifier, e.g. behavioural checks on the end state);
    a custom verifier needs at least one wrong Outcome. Oracle: mint writes solve.sh for
    an `answer`; otherwise supply files["solution/solve.sh"] or set oracle_kind="none".
    """

    task_id: str
    gold_type: str = "imitation"
    assumptions: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    instruction: str | None = None
    evidence: dict = field(default_factory=dict)
    answer: Answer | None = None
    wrong_answers: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    leak_allow: list = field(default_factory=list)
    secret_allow: list = field(default_factory=list)
    difficulty: str = "medium"
    tags: list = field(default_factory=list)
    description: str = ""
    # provenance (see guides/tasktypes.md); None = mint infers it and checks it is consistent
    oracle_kind: str | None = None
    env_source: str | None = None
    grading_method: str | None = None
    grading_reason: str = ""
    gaps: list = field(default_factory=list)
    # which logged attempt / solve the solution came from and how its outcome is known (free text)
    oracle_source: str = ""
    # the verifier's own time budget in seconds (task.toml [verifier] timeout_sec); None = 120.
    # Give it room for its setup and per-call work; it is not the solver's budget.
    verifier_timeout_sec: float | None = None
    # run-time env passed into the task environment (task.toml [environment] env), e.g. a user
    # simulator's model and key as host references: {"USER_SIM_MODEL": "${USER_SIM_MODEL}"}.
    # Values must be ${VAR} references or plain settings, never literal secrets.
    runtime_env: dict = field(default_factory=dict)
    # the solver's time budget (task.toml [agent] timeout_sec); None = 900. Conversations need more.
    agent_timeout_sec: float | None = None


@dataclass
class Blocked:
    """No task may be built: only `kind` secret / leak / no_instruction. A job that is merely
    hard to rebuild is a Candidate with its gaps."""

    reason: str
    missing: list = field(default_factory=list)
    kind: str = "other"


@dataclass
class Candidate:
    """The job was identified but no runnable package could be built yet: what is missing
    and what would close it. Kept visible in the queue and the report, never counted as a task."""

    reason: str
    gaps: list = field(default_factory=list)
    next_action: str = ""
