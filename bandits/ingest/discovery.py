"""Which recorded field holds a workflow's request and its answer, when nobody said.

A pre-pass over the export: the same read, redaction, decoding and
classification as the loader, and the same invocation candidates, but no
spans, nodes or corpus. Each candidate keeps only the string leaves of its
input and output (two levels deep), hashed, which is enough to run
:func:`~bandits.ingest.workflow.select_invocation` exactly as the loader will.

A field set is chosen by itself when it settles every trace. Otherwise every
field found is declared together, and each trace is settled the loader's way:
candidates holding the same text agree (the task is certain), candidates
holding different text are a conflict, kept with every value and left for a
consumer to resolve. Nothing is guessed and nothing waits on a person.
Options are still grouped by the candidate's identity (declared kind and
name) and shown, with the flag that selects each, for anyone who wants one.
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
    disk_read,
    shape_id,
)
from bandits.ingest.report import IngestReport
from bandits.ingest.workflow import select_invocation
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
    shape_id: str = ""


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
    # Surrounding whitespace aside, as the loader compares task text.
    text = text.strip()
    return "" if not text else hashlib.sha256(text.encode()).hexdigest()[:16]


def _hashed(value: object, depth: int = 0) -> object:
    """String leaves hashed, two levels deep; anything else dropped."""
    if isinstance(value, str):
        return _digest(value)
    if isinstance(value, dict) and depth < MAX_DEPTH:
        kept = {k: _hashed(v, depth + 1) for k, v in value.items() if isinstance(k, str)}
        return {k: v for k, v in kept.items() if v is not None}
    return None


def _summarize(
    path: Path, ruleset: RedactionRuleset, summary: RequestSummary, scratch_dir: Path | None
) -> None:
    issues: list[TraceIssue] = []  # the loader reports these; discovery only reads
    scratch: Counter[str] = Counter()
    with disk_read(path, ruleset, issues, IngestReport(), scratch_dir) as read:
        for trace_id, decoded in read.by_trace.items():
            decoded, _ = _prepare(trace_id, decoded, issues)
            if not decoded:
                continue
            candidates = []
            for span_id in _invocation_candidates(decoded):
                span = decoded[span_id]
                record = {
                    "input": _hashed(_io_value(span.attributes, _INPUT_VALUE_KEYS, scratch)),
                    "output": _hashed(_io_value(span.attributes, _OUTPUT_VALUE_KEYS, scratch)),
                }
                candidates.append(Candidate(span_id, (span.label, span.name), record))
            summary.traces.append(TraceRequests(trace_id, candidates, shape_id(decoded)))


def discover_requests(
    path: Path,
    source: str,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    scratch_dir: Path | None = None,
) -> RequestSummary:
    """Every trace's invocation candidates, summarized; no corpus is built.

    Temporary files go in *scratch_dir* (default: the working directory).
    """
    from bandits.ingest.native import _CONVERTERS, NativeConversion

    summary = RequestSummary()
    if source in _CONVERTERS:
        conversion = NativeConversion(path, source, ruleset, scratch_dir=scratch_dir)
        for chunk in conversion.chunks():
            _summarize(chunk, ruleset, summary, scratch_dir)
    elif source == "otlp-std":
        _summarize(path, ruleset, summary, scratch_dir)
    else:
        raise ValueError(f"request discovery reads OTLP and native exports, not {source!r}")
    return summary


def restricted(summary: RequestSummary, identities: set[tuple[str, str]]) -> RequestSummary:
    """The same traces with only candidates of the given identities."""
    return RequestSummary(
        [
            TraceRequests(
                t.trace_id, [c for c in t.candidates if c.identity in identities], t.shape_id
            )
            for t in summary.traces
        ]
    )


def identity_coverage(summary: RequestSummary, identities: set[tuple[str, str]]) -> tuple[int, int]:
    """``(traces where the identities alone pick exactly one candidate, traces
    with candidates)``: whether a mapping's invocation list can choose by itself."""
    with_candidates = [t for t in summary.traces if t.candidates]
    single = sum(sum(c.identity in identities for c in t.candidates) == 1 for t in with_candidates)
    return single, len(with_candidates)


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


def _select(trace: TraceRequests, paths: tuple[str, ...]) -> tuple[Candidate | None, bool]:
    """The candidate the loader will choose with *paths* declared, by the same
    rule (:func:`select_invocation`), on hashed text; and whether its task
    resolves (a sole candidate is chosen even when it holds no task)."""
    by_id = {c.span_id: c for c in trace.candidates}
    chosen, _, found = select_invocation(
        list(by_id), {span_id: c.record for span_id, c in by_id.items()}, paths
    )
    if chosen is None:
        return None, False
    return by_id[chosen], any(c.span_id == chosen for c in found)


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
        covered = sum(_select(t, paths)[1] for t in with_candidates)
        options.append(Option(paths, tuple(sorted(identities)), covered, len(with_candidates)))
    return sorted(options, key=lambda o: (-o.covered, o.paths))


def chosen_runs(summary: RequestSummary, task_fields: tuple[str, ...]) -> list[Candidate | None]:
    """The candidate the loader will choose in each trace: the sole candidate,
    or the only one where the task fields resolve; None when it chooses none."""
    return [_select(trace, task_fields)[0] for trace in summary.traces]


def answer_runs(summary: RequestSummary, task_fields: tuple[str, ...]) -> list[list[Candidate]]:
    """Per trace, the runs the loader may read the answer from: the chosen run,
    or every run that agrees on the task (the loader takes the one whose output
    holds the delivered field); empty when it chooses none."""
    groups = []
    for trace in summary.traces:
        by_id = {c.span_id: c for c in trace.candidates}
        chosen, basis, found = select_invocation(
            list(by_id), {span_id: c.record for span_id, c in by_id.items()}, task_fields
        )
        if chosen is None:
            groups.append([])
        elif basis.startswith("agreement"):
            groups.append([by_id[s] for s in dict.fromkeys(c.span_id for c in found)])
        else:
            groups.append([by_id[chosen]])
    return groups


def answer_options(runs: list[Candidate | None] | list[list[Candidate]]) -> list[Option]:
    """Each answer path the chosen runs hold, with how many traces hold it.

    A run per trace, or a group of agreeing runs per trace: a trace holds a
    path when any run in its group does.
    """
    groups = [run if isinstance(run, list) else ([] if run is None else [run]) for run in runs]
    groups = [group for group in groups if group]
    picked: Counter[str] = Counter()
    for group in groups:
        picked.update(
            {
                path
                for run in group
                if (path := pick_path(run.record, "output", ANSWER_KEYS)) is not None
            }
        )
    return [Option((path,), (), count, len(groups)) for path, count in picked.most_common()]


def discover(
    summary: RequestSummary,
    *,
    task_fields: tuple[str, ...] = (),
    delivered_field: str | None = None,
) -> Discovery:
    """Choose what is unambiguous; declared fields are kept as given.

    One option that settles every trace is chosen. Otherwise every option's
    fields are declared together: where they hold the same text the task is
    certain, and where they differ the loader keeps every value unresolved,
    so nothing is guessed and nothing waits on a person.
    """
    found = Discovery(task_fields=task_fields, delivered_field=delivered_field)
    if not task_fields:
        found.task_options = task_options(summary)
        complete = [option for option in found.task_options if option.complete]
        if len(complete) == 1:
            found.task_fields = complete[0].paths
        elif found.task_options:
            found.task_fields = tuple(
                dict.fromkeys(path for option in found.task_options for path in option.paths)
            )
    if delivered_field is None:
        found.answer_options = answer_options(answer_runs(summary, found.task_fields))
        complete = [option for option in found.answer_options if option.complete]
        if len(complete) == 1:
            found.delivered_field = complete[0].paths[0]
    return found
