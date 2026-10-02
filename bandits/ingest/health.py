"""What an ingest actually recovered, as problems a user can act on.

``ingest`` runs this on every load. A *fatal* problem means the corpus would
be useless (nothing read, or no model call with any content); nothing is saved.
Everything else is a *warning*: the corpus is saved and the full detail stays
in its issue list.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bandits.traces import NOTICE_ISSUE_KINDS, SpanKind, SpanStatus, Trace, TraceCorpus

_ROLES = ("system", "developer", "user", "assistant", "tool")
_EXAMPLES = 3


@dataclass
class Health:
    traces: int = 0
    model_calls: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    fatal: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def note(self, kind: str, where: str) -> None:
        self.counts[kind] += 1
        found = self.examples.setdefault(kind, [])
        if len(found) < _EXAMPLES:
            found.append(where)


def _messages(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _valid(value: Any) -> bool:
    value = _messages(value)
    return isinstance(value, list) and all(
        isinstance(m, dict) and m.get("role") in _ROLES and isinstance(m.get("parts"), list)
        for m in value
    )


def _has_content(value: Any) -> bool:
    value = _messages(value)
    return isinstance(value, list) and any(
        isinstance(m, dict) and (m.get("parts") or m.get("content")) for m in value
    )


PER_CALL_SOURCES = ("otlp-std", "langfuse", "langsmith", "phoenix")
"""Readers that record each model call's own input and output. The others keep a
conversation at the trace level, so a call without its own input is normal there."""


def check(corpus: TraceCorpus, source: str, hints: list[str] | None = None) -> Health:
    """``hints``: the request fields discovery found, each as the flags that
    select it, for the warning about traces without a task."""
    health = Health()
    for trace in corpus.traces:
        collect(health, trace, source, workflow=corpus.workflow is not None)
    return finish(health, corpus, hints)


def collect(health: Health, trace: Trace, source: str, *, workflow: bool) -> None:
    health.traces += 1
    health.model_calls += sum(s.kind == SpanKind.MODEL for s in trace.spans)
    if workflow and trace.task is None:
        health.note("no_task", f"trace {trace.trace_id}")
    for span in trace.spans:
        if span.kind != SpanKind.MODEL or source not in PER_CALL_SOURCES:
            continue
        a = span.attributes
        where = f"trace {trace.trace_id} span {span.span_id} ({span.name})"
        in_msgs, out_msgs = a.get("gen_ai.input.messages"), a.get("gen_ai.output.messages")
        failed = span.status == SpanStatus.ERROR
        if not _has_content(in_msgs):
            kind = "input_unread" if a.get("input.value") is not None else "input_missing"
            health.note(kind, where)
        if a.get("bandits.output_unusable_reason") and not failed:
            health.note("output_unusable", where)
        elif not _has_content(out_msgs) and span.output is None:
            health.note("failed_no_output" if failed else "output_missing", where)
        if any(not _valid(v) for v in (in_msgs, out_msgs) if v is not None and v != []):
            health.note("bad_messages", where)


def finish(health: Health, corpus: TraceCorpus, hints: list[str] | None = None) -> Health:
    c = health.counts
    other = Counter(i.kind for i in corpus.issues if i.kind not in NOTICE_ISSUE_KINDS)

    if not health.traces:
        health.fatal.append("no traces could be read from this file")
    elif not health.model_calls:
        health.fatal.append(
            "no model calls were found; the file may use span kinds Bandits does not "
            "recognize (see `unrepresented_span` issues with --dry-run)"
        )
    elif c["input_missing"] + c["input_unread"] == health.model_calls:
        health.fatal.append(
            f"none of the {health.model_calls} model calls has a readable input; the export "
            "may have been captured with content recording turned off"
        )

    def warn(kind: str, what: str, why: str) -> None:
        if c[kind]:
            where = "; ".join(health.examples.get(kind, []))
            health.warnings.append(
                f"{c[kind]} {what}\n    why: {why}" + (f"\n    e.g. {where}" if where else "")
            )

    warn(
        "input_missing",
        "model call(s) have no recorded input",
        "the exporter did not record the prompt (content capture off, or dropped upstream)",
    )
    warn(
        "input_unread",
        "model call(s) have an input Bandits could not read as chat messages",
        "the recorded input is in a shape Bandits does not know (often app data, "
        "not a prompt); it is kept raw in the corpus",
    )
    warn(
        "output_missing",
        "successful model call(s) have no recorded output",
        "the exporter did not record the reply; these calls cannot become training rows",
    )
    warn(
        "output_unusable",
        "model call(s) recorded an output that is not an answer",
        "the output field holds something else (e.g. a description of the API response "
        "object); it is kept raw but not used as an answer",
    )
    warn(
        "bad_messages",
        "model call(s) have malformed message lists",
        "a message has no valid role or no parts list",
    )
    warn(
        "no_task",
        "trace(s) have no task",
        "the request field could not be chosen; pass one of: " + " | ".join(hints)
        if hints
        else "no request field was found in the run's input; pass --task-field PATH "
        "to say where it is",
    )
    for kind, count in other.most_common():
        detail = next((i.detail for i in corpus.issues if i.kind == kind), "")
        health.warnings.append(f"{count} `{kind}` issue(s)\n    e.g. {detail[:200]}")
    return health
