"""Contract types shared between l2h and agent-written workspace code.

Everything here round-trips through plain JSON: agent code may return the
dataclasses or plain dicts, and l2h validates and normalises both.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ENDED = ("done", "cut_off", "error", "unknown")
RESULTS = ("pass", "fail", "unknown")
GOLD_TYPES = ("verified", "solved", "imitation", "none")
CHECK_KINDS = ("exact", "allowed_set", "numeric")
NORMALIZERS = ("strip", "lower", "collapse_ws", "percent", "thousands", "sorted_list")
STATUSES = ("validated", "generated", "blocked")
GATE_STATES = ("pass", "fail", "error", "not_run")

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


class Trajectory(dict):
    """A parsed trajectory. A dict with attribute access for convenience.

    Keys: id, source_refs, events, attachments, meta, ended.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


@dataclass
class ParseError:
    """A record the parser could not turn into a trajectory."""

    file: str
    error: str
    offset: int | None = None
    kind: str = "parse_error"

    def to_dict(self) -> dict:
        return {"file": self.file, "offset": self.offset, "error": self.error, "kind": self.kind}


def normalize_trajectory(obj: Any) -> Trajectory:
    """Validate a parse() result against the Trajectory contract."""
    if not isinstance(obj, dict):
        raise ContractError(f"parse() yielded {type(obj).__name__}; expected Trajectory or dict")
    t = Trajectory(obj)
    if "id" not in t or t["id"] in (None, ""):
        raise ContractError("trajectory is missing 'id'")
    t["id"] = str(t["id"])
    for key, typ in (("source_refs", list), ("events", list), ("attachments", list)):
        t.setdefault(key, [])
        if not isinstance(t[key], typ):
            raise ContractError(f"trajectory {t['id']}: '{key}' must be a list")
    if not isinstance(t.setdefault("meta", {}), dict):
        raise ContractError(f"trajectory {t['id']}: 'meta' must be an object")
    if t.setdefault("ended", "unknown") not in ENDED:
        raise ContractError(f"trajectory {t['id']}: 'ended' must be one of {ENDED}")
    if not t["source_refs"]:
        raise ContractError(
            f"trajectory {t['id']}: 'source_refs' is empty; every trajectory must trace to the raw logs"
        )
    for ref in t["source_refs"]:
        if not isinstance(ref, dict) or "file" not in ref or not ("offset" in ref or "key" in ref):
            raise ContractError(
                f"trajectory {t['id']}: each source_ref needs 'file' and 'offset' or 'key', got {ref!r}"
            )
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

    def validate(self) -> None:
        if self.check not in CHECK_KINDS:
            raise ContractError(f"answer.check {self.check!r} must be one of {CHECK_KINDS}")
        bad = [n for n in self.normalize if n not in NORMALIZERS]
        if bad:
            raise ContractError(f"unknown normalizers {bad}; allowed {NORMALIZERS}")
        if self.check == "allowed_set" and not (isinstance(self.expected, list) and self.expected):
            raise ContractError("allowed_set needs a non-empty list as answer.expected")
        if self.check == "exact" and isinstance(self.expected, list | dict):
            raise ContractError("exact needs a scalar expected answer; use allowed_set for lists")
        if "/" in self.file or self.file.startswith("."):
            raise ContractError("answer.file must be a plain file name")

    def to_dict(self) -> dict:
        return {
            "expected": self.expected,
            "check": self.check,
            "file": self.file,
            "normalize": list(self.normalize),
            "tolerance": self.tolerance,
            "rel_tolerance": self.rel_tolerance,
        }

    def canonical(self) -> str:
        """The single value an oracle writes."""
        return str(self.expected[0] if isinstance(self.expected, list) else self.expected)


@dataclass
class BuildResult:
    """Returned by a tasktype's build() for one group of trajectories.

    files: extra/override files, keys relative to the task dir. Anything l2h can
    generate (task.toml, tests/*, solution/solve.sh, environment/Dockerfile) is
    filled in when absent. `instruction` and `evidence` are conveniences:
    evidence keys land in environment/data/<key> and appear at /app/<key>.
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
    difficulty: str = "medium"
    tags: list = field(default_factory=list)
    description: str = ""


@dataclass
class Blocked:
    reason: str
    missing: list = field(default_factory=list)
