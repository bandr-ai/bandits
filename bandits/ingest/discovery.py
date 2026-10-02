"""Which recorded field holds a workflow's request and its answer, when nobody said.

A pre-pass over the export: the same read, redaction, decoding and
classification as the loader, and the same invocation candidates, but no
spans, nodes or corpus. Each candidate keeps only the string leaves of its
input and output (two levels deep), hashed, which is enough to run
:func:`~bandits.ingest.workflow.resolve_task` exactly as the loader will.

A field is proposed only when it settles every trace the way the loader will
check it: exactly one candidate per trace resolves it. Options are grouped by
the candidate's identity (declared kind and name). When two different
identities each settle every trace with different fields, nothing is chosen:
which one is the application run is a fact about the application, not about
the record, and the user is shown both with the flag that picks each.
Identities whose fields are the same (one run recorded under two names) are
one option: passing those fields loads the same corpus either way.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandits.ingest.otlp_standard import (
    _INPUT_VALUE_KEYS,
    _OUTPUT_VALUE_KEYS,
    _invocation_candidates,
    _io_value,
    _prepare,
    _read,
)
from bandits.ingest.report import IngestReport
from bandits.ingest.workflow import resolve_task
from bandits.redact import DEFAULT_RULESET, RedactionRuleset
from bandits.traces import TraceIssue

TASK_KEYS = ("query", "question", "input", "prompt", "message", "task", "request", "text")
ANSWER_KEYS = ("answer", "output", "response", "result", "text", "content", "message")
"""Field names that commonly hold a request and its answer, in priority order.
Matched on a path's last component only, at most two levels below the
record's ``input``/``output``."""

MAX_DEPTH = 2


@dataclass(frozen=True)
class Candidate:
    span_id: str
    identity: tuple[str, str]
    """``(declared kind label, name)``: what this span is across traces."""

    record: dict[str, Any]
    """``{"input": ..., "output": ...}`` with every string leaf replaced by a
    hash ("" when blank), so equal text compares equal and nothing else is kept."""


@dataclass
class TraceRequests:
    trace_id: str
    candidates: list[Candidate]


@dataclass
class RequestSummary:
    traces: list[TraceRequests] = field(default_factory=list)


@dataclass(frozen=True)
class Option:
    paths: tuple[str, ...]
    identities: tuple[tuple[str, str], ...]
    """Whose fields these are; empty for an answer path (read on the chosen run)."""

    covered: int
    total: int

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.covered == self.total

    def describe(self) -> str:
        who = f" → {', '.join(map(identity_label, self.identities))}" if self.identities else ""
        return f"{', '.join(self.paths)}{who} {self.covered}/{self.total}"


@dataclass
class Discovery:
    """What discovery found and, when it is unambiguous, chose."""

    task_fields: tuple[str, ...] = ()
    task_options: list[Option] = field(default_factory=list)
    delivered_field: str | None = None
    answer_options: list[Option] = field(default_factory=list)

    def hints(self) -> list[str]:
        """How to pass each task option found, for a trace whose task is missing."""
        return [
            " ".join(f"--task-field {path}" for path in option.paths)
            + f" (selects {' or '.join(map(identity_label, option.identities))}; "
            f"{option.covered}/{option.total})"
            for option in self.task_options
        ]


def identity_label(identity: tuple[str, str]) -> str:
    label, name = identity
    return f"{name}({label.rsplit('=', 1)[-1]})"


def _digest(text: str) -> str:
    return "" if not text.strip() else hashlib.sha256(text.encode()).hexdigest()[:16]


def _hashed(value: object, depth: int = 0) -> object:
    """String leaves hashed, two levels deep; anything else dropped."""
    if isinstance(value, str):
        return _digest(value)
    if isinstance(value, dict) and depth < MAX_DEPTH:
        kept = {k: _hashed(v, depth + 1) for k, v in value.items() if isinstance(k, str)}
        return {k: v for k, v in kept.items() if v is not None}
    return None


def _summarize(path: Path, ruleset: RedactionRuleset, summary: RequestSummary) -> None:
    issues: list[TraceIssue] = []  # the loader reports these; discovery only reads
    read = _read(path, ruleset, issues, IngestReport())
    scratch: Counter[str] = Counter()
    for trace_id, decoded in read.by_trace.items():
        decoded, _ = _prepare(trace_id, decoded, issues)
        candidates = []
        for span_id in _invocation_candidates(decoded):
            span = decoded[span_id]
            record = {
                "input": _hashed(_io_value(span.attributes, _INPUT_VALUE_KEYS, scratch)),
                "output": _hashed(_io_value(span.attributes, _OUTPUT_VALUE_KEYS, scratch)),
            }
            candidates.append(Candidate(span_id, (span.label, span.name), record))
        summary.traces.append(TraceRequests(trace_id, candidates))


def discover_requests(
    path: Path, source: str, ruleset: RedactionRuleset = DEFAULT_RULESET
) -> RequestSummary:
    """Every trace's invocation candidates, summarized; no corpus is built."""
    from bandits.ingest.native import _CONVERTERS, NativeConversion

    summary = RequestSummary()
    if source in _CONVERTERS:
        conversion = NativeConversion(path, source, ruleset)
        for chunk in conversion.chunks():
            _summarize(chunk, ruleset, summary)
    elif source == "otlp-std":
        _summarize(path, ruleset, summary)
    else:
        raise ValueError(f"request discovery reads OTLP and native exports, not {source!r}")
    return summary


def pick_path(record: dict[str, Any], root: str, keys: tuple[str, ...]) -> str | None:
    """The one path on *record* a field search would read: the whole value when
    it is text, else the first key in priority order, shallowest first."""
    value = record.get(root)
    if isinstance(value, str):
        return root if value else None
    found: list[tuple[int, int, str]] = []
    stack: list[tuple[str, object, int]] = [(root, value, 0)]
    while stack:
        prefix, node, depth = stack.pop()
        if not isinstance(node, dict) or depth >= MAX_DEPTH:
            continue
        for key, child in node.items():
            path = f"{prefix}.{key}"
            if isinstance(child, str) and child and key in keys:
                found.append((keys.index(key), depth, path))
            stack.append((path, child, depth + 1))
    return min(found)[2] if found else None


def _selected(trace: TraceRequests, paths: tuple[str, ...]) -> list[Candidate]:
    """Candidates where *paths* resolve, as ``build_request`` will check it."""
    return [c for c in trace.candidates if resolve_task(c.record, paths)[1] != "unresolved"]


def _ordered(paths: set[str], keys: tuple[str, ...]) -> tuple[str, ...]:
    def rank(path: str) -> tuple[int, int, str]:
        if "." not in path:  # the whole value is text
            return (-1, 0, path)
        return (keys.index(path.rsplit(".", 1)[-1]), path.count("."), path)

    return tuple(sorted(paths, key=rank))


def task_options(summary: RequestSummary) -> list[Option]:
    """One option per set of fields, with the identities that use it and how
    many traces it settles."""
    by_identity: dict[tuple[str, str], set[str]] = {}
    for trace in summary.traces:
        for candidate in trace.candidates:
            path = pick_path(candidate.record, "input", TASK_KEYS)
            if path is not None:
                by_identity.setdefault(candidate.identity, set()).add(path)
    by_paths: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for identity, found in by_identity.items():
        by_paths.setdefault(_ordered(found, TASK_KEYS), []).append(identity)
    with_candidates = [t for t in summary.traces if t.candidates]
    options = []
    for paths, identities in by_paths.items():
        covered = sum(len(_selected(t, paths)) == 1 for t in with_candidates)
        options.append(Option(paths, tuple(sorted(identities)), covered, len(with_candidates)))
    return sorted(options, key=lambda o: (-o.covered, o.paths))


def chosen_runs(summary: RequestSummary, task_fields: tuple[str, ...]) -> list[Candidate | None]:
    """The candidate the loader will choose in each trace: the sole candidate,
    or the only one where the task fields resolve; None when it chooses none."""
    runs = []
    for trace in summary.traces:
        if len(trace.candidates) == 1:
            runs.append(trace.candidates[0])
            continue
        resolving = _selected(trace, task_fields) if task_fields else []
        runs.append(resolving[0] if len(resolving) == 1 else None)
    return runs


def answer_options(runs: list[Candidate | None]) -> list[Option]:
    """The answer path each chosen run would be read from, with coverage."""
    chosen = [run for run in runs if run is not None]
    picked = Counter(
        path for run in chosen if (path := pick_path(run.record, "output", ANSWER_KEYS)) is not None
    )
    return [Option((path,), (), count, len(chosen)) for path, count in picked.most_common()]


def discover(
    summary: RequestSummary,
    *,
    task_fields: tuple[str, ...] = (),
    delivered_field: str | None = None,
) -> Discovery:
    """Choose what is unambiguous; declared fields are kept as given."""
    found = Discovery(task_fields=task_fields, delivered_field=delivered_field)
    if not task_fields:
        found.task_options = task_options(summary)
        complete = [option for option in found.task_options if option.complete]
        if len(complete) == 1:
            found.task_fields = complete[0].paths
    if delivered_field is None:
        found.answer_options = answer_options(chosen_runs(summary, found.task_fields))
        complete = [option for option in found.answer_options if option.complete]
        if len(found.answer_options) == 1 and complete:
            found.delivered_field = complete[0].paths[0]
    return found
