"""The readiness assessment itself: corpus in, verdict out.

Six signals, each one a link in the chain from telemetry to a code reward. They
are reported individually and never averaged, for the same reason fidelity is
per-tool: a single composite score tells a customer nothing about what to fix,
and the whole point of this page is that it is actionable in their logging
config.

======================  ============================================================
Signal                  What breaks without it
======================  ============================================================
``invocation_points``   Everything. No action space, no state, no verifier.
``arguments``           No replay, no filler-data patterns, no write semantics.
``responses``           No response shape and no database. Reward becomes a judge.
``identifiers``         No entities. Responses stay opaque blobs, no state to assert.
``state_changes``       Nothing ever changed, so there is nothing to verify.
``error_modes``         The env can never return a failure. The agent never sees adversity.
======================  ============================================================

Two further signals are reported but never gate the verdict: ``instructions``
(a task statement can be written by hand if it is missing) and ``outcome_labels``
(useful for trace filtering, not required to reconstruct a world).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from bandits.contracts import CallStatus, JsonObject, JsonValue, TraceCorpus
from bandits.state.identifiers import find_identifiers, ok_invocations, scalar_fields

#: A signal at or above this share of its population counts as present.
PASS_RATIO = 0.5

#: Below this share of traces carrying invocation points, the export is not a
#: tool-call export at all, whatever the absolute count says.
MIN_TRACE_COVERAGE = 0.2

#: An identifier field must recur on at least this many distinct calls before it
#: is evidence of a row handle rather than a value that happened to repeat.
MIN_IDENTIFIER_MENTIONS = 3


class Verdict(str, Enum):
    """What triage concludes about this export."""

    GO = "GO"
    PARTIAL = "PARTIAL"
    NO_GO = "NO_GO"


@dataclass(frozen=True)
class Signal:
    """One readiness check, with the number behind it.

    ``detail`` is written for the customer's platform engineer, because the fix for
    a failing signal is almost always a change to what their tracer records.
    """

    name: str
    present: bool
    observed: int
    population: int
    detail: str
    blocking: bool = True
    """False for signals that inform the report but never move the verdict."""

    @property
    def ratio(self) -> float:
        return self.observed / self.population if self.population else 0.0


@dataclass(frozen=True)
class ToolReadiness:
    """Per-tool view: which tools can be reconstructed and which cannot."""

    tool: str
    calls: int
    with_arguments: int
    with_object_response: int
    error_calls: int
    identifier_fields: tuple[str, ...] = ()
    """Identifier-shaped fields this tool's responses expose. Evidence of an entity."""

    @property
    def reconstructible(self) -> bool:
        """True when this tool's records carry enough to reimplement it over a store.

        Requires a structured response on the majority of successful calls -- a tool
        that only ever returned prose has no fields to build a table from, and an env
        that invents them is a fabrication with a confident interface.
        """
        successful = self.calls - self.error_calls
        if successful <= 0:
            return False
        return self.with_object_response / successful >= PASS_RATIO

    @property
    def note(self) -> str:
        if self.reconstructible:
            return "ok"
        if self.calls == self.error_calls:
            return "only ever failed"
        if self.with_object_response == 0:
            return "no structured response"
        return "structured response on a minority of calls"


@dataclass(frozen=True)
class TriageReport:
    """The one page. Signals, per-tool readiness, and a position."""

    source: str
    traces: int
    invocations: int
    traces_with_invocations: int
    signals: tuple[Signal, ...] = ()
    tools: tuple[ToolReadiness, ...] = ()
    issue_counts: dict[str, int] = field(default_factory=dict)
    verdict: Verdict = Verdict.NO_GO
    reasons: tuple[str, ...] = ()

    def signal(self, name: str) -> Signal | None:
        for candidate in self.signals:
            if candidate.name == name:
                return candidate
        return None

    @property
    def blocking_failures(self) -> tuple[Signal, ...]:
        return tuple(s for s in self.signals if s.blocking and not s.present)

    def to_json(self) -> JsonObject:
        """Serializable form, for the platform to store alongside the deployment."""
        return {
            "source": self.source,
            "traces": self.traces,
            "invocations": self.invocations,
            "traces_with_invocations": self.traces_with_invocations,
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "signals": [
                {
                    "name": s.name,
                    "present": s.present,
                    "observed": s.observed,
                    "population": s.population,
                    "ratio": round(s.ratio, 4),
                    "blocking": s.blocking,
                    "detail": s.detail,
                }
                for s in self.signals
            ],
            "tools": [
                {
                    "tool": t.tool,
                    "calls": t.calls,
                    "with_arguments": t.with_arguments,
                    "with_object_response": t.with_object_response,
                    "error_calls": t.error_calls,
                    "identifier_fields": list(t.identifier_fields),
                    "reconstructible": t.reconstructible,
                    "note": t.note,
                }
                for t in self.tools
            ],
            "issue_counts": dict(self.issue_counts),
        }


def _identifier_fields_by_tool(corpus: TraceCorpus) -> dict[str, set[str]]:
    """Which identifier-shaped fields each tool exposes in its *responses*.

    Response-side only, deliberately. An id passed *into* a tool proves the agent
    knew a value; an id coming *out* is what links two tools to the same row, and
    that linkage is the evidence stage 3 turns into an entity.
    """
    index = find_identifiers(corpus)
    by_tool: dict[str, set[str]] = {}
    for name, identifier in index.fields.items():
        if identifier.occurrences < MIN_IDENTIFIER_MENTIONS:
            continue
        for tool in identifier.response_tools:
            by_tool.setdefault(tool, set()).add(name)
    return by_tool


def _response_objects(value: JsonValue) -> list[JsonObject]:
    """Every JSON object nested anywhere inside one response body.

    Tools wrap rows inconsistently -- ``{"order": {...}}``, ``{"orders": [...]}``,
    or the row bare at the top level. Walking the whole body means the check does
    not depend on guessing which wrapper this customer's tools happen to use.
    """
    if isinstance(value, dict):
        out = [value]
        for nested in value.values():
            out.extend(_response_objects(nested))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_response_objects(item))
        return out
    return []


def _state_change_evidence(corpus: TraceCorpus) -> tuple[int, int]:
    """``(rows observed changing, rows observed more than once)``.

    Triage cannot classify tools as read or write -- that is stage 2's job, with the
    declared registry and the response-shape evidence it has and we do not. But it
    can ask the question that actually matters for reward: **did anything ever
    change?**

    Group every response object by the identifier value it carries, and look for a
    field that took two different values across those observations. An order seen as
    ``shipped`` and later as ``refunded`` is a state change, recorded, and it is
    exactly the assertion a verifier would make. A corpus where no row is ever seen
    twice with different contents is a read-only corpus, and there is nothing in it
    to verify no matter how many tools it has.

    The denominator is rows seen more than once, not all rows: a row observed a
    single time cannot demonstrate a change, and counting it as a failure would
    penalize a corpus for being short rather than for being read-only.

    Which field names count as keys is **inherited from stage 3**, not decided here.
    :func:`find_identifiers` nominates a field only when the same value appears as an
    argument and as a *top-level* response scalar, so a key that only ever appears
    nested is not nominated and this check reports nothing for it. That is the
    correct behaviour rather than a gap to widen: reconstruction will not build an
    entity for that key either, and a triage GO that the pipeline cannot honour is
    worse than a NO_GO. Rows *are* matched inside envelopes once the key itself has
    been established elsewhere, which is the common mixed-wrapper case.
    """
    index = find_identifiers(corpus)
    key_names = {
        name
        for name, identifier in index.fields.items()
        if identifier.occurrences >= MIN_IDENTIFIER_MENTIONS
    }
    if not key_names:
        return 0, 0

    observations: dict[tuple[str, object], list[dict[str, object]]] = {}
    for inv in ok_invocations(corpus):
        for obj in _response_objects(inv.response):
            fields = scalar_fields(obj)
            for name, value in fields.items():
                if name in key_names:
                    observations.setdefault((name, value), []).append(fields)

    repeated = 0
    changed = 0
    for (key_name, _), seen in observations.items():
        if len(seen) < 2:
            continue
        repeated += 1
        names = {name for fields in seen for name in fields}
        for name in names:
            if name == key_name:
                continue
            values = {fields[name] for fields in seen if name in fields}
            if len(values) > 1:
                changed += 1
                break
    return changed, repeated


def triage_corpus(corpus: TraceCorpus) -> TriageReport:
    """Assess one already-ingested corpus for environment readiness.

    Takes a :class:`~bandits.contracts.TraceCorpus` rather than a path so that the
    check runs on exactly the records ingest recovered -- including the ones it had
    to reconstruct. Triaging the raw file separately would measure a different
    population than the pipeline will actually see.
    """
    traces = corpus.traces
    invocations = [inv for trace in traces for inv in trace.invocations]
    traces_with = sum(1 for trace in traces if trace.invocations)

    identifier_fields = _identifier_fields_by_tool(corpus)
    tools = _tool_readiness(invocations, identifier_fields)

    with_args = sum(1 for inv in invocations if inv.arguments)
    with_response = sum(1 for inv in invocations if isinstance(inv.response, dict) and inv.response)
    errors = sum(1 for inv in invocations if inv.status is CallStatus.ERROR)
    changed_rows, repeated_rows = _state_change_evidence(corpus)
    with_ids = sum(1 for tool in tools if tool.identifier_fields)
    instructions = sum(1 for trace in traces if trace.instruction)
    labelled = sum(1 for trace in traces if trace.outcome is not None)

    signals = (
        Signal(
            name="invocation_points",
            present=bool(traces) and traces_with / len(traces) >= MIN_TRACE_COVERAGE,
            observed=traces_with,
            population=len(traces),
            detail=(
                "traces carrying at least one recorded tool call. Without these there is no "
                "action space and reward can only be an LLM judge"
            ),
        ),
        Signal(
            name="arguments",
            present=bool(invocations) and with_args / len(invocations) >= PASS_RATIO,
            observed=with_args,
            population=len(invocations),
            detail="calls whose arguments were retained. Needed to replay a call and to learn write semantics",
        ),
        Signal(
            name="responses",
            present=bool(invocations) and with_response / len(invocations) >= PASS_RATIO,
            observed=with_response,
            population=len(invocations),
            detail=(
                "calls whose response body was retained as structured data. This is the evidence "
                "the database behind the tools is inferred from"
            ),
        ),
        Signal(
            name="identifiers",
            present=bool(tools) and with_ids > 0,
            observed=with_ids,
            population=len(tools),
            detail=(
                "tools whose responses expose a recurring id. Recurrence across tools is what "
                "turns separate responses into rows of one table"
            ),
        ),
        Signal(
            name="state_changes",
            present=changed_rows > 0,
            observed=changed_rows,
            population=repeated_rows,
            detail=(
                "rows observed taking a new field value, out of rows seen more than once. "
                "This is the change a verifier asserts on; without it the corpus is read-only"
            ),
        ),
        Signal(
            name="error_modes",
            present=errors > 0,
            observed=errors,
            population=len(invocations),
            detail=(
                "recorded failures. An environment that can only succeed trains an agent that has "
                "never seen adversity"
            ),
        ),
        Signal(
            name="instructions",
            present=bool(traces) and instructions / len(traces) >= PASS_RATIO,
            observed=instructions,
            population=len(traces),
            blocking=False,
            detail="traces with a recoverable task statement. Can be written by hand if missing",
        ),
        Signal(
            name="outcome_labels",
            present=bool(traces) and labelled > 0,
            observed=labelled,
            population=len(traces),
            blocking=False,
            detail="traces with a ground-truth success label. Used to filter training traces, not to reconstruct",
        ),
    )

    verdict, reasons = _decide(signals, tools)
    return TriageReport(
        source=corpus.source,
        traces=len(traces),
        invocations=len(invocations),
        traces_with_invocations=traces_with,
        signals=signals,
        tools=tools,
        issue_counts=dict(Counter(issue.kind for issue in corpus.issues)),
        verdict=verdict,
        reasons=reasons,
    )


def _tool_readiness(
    invocations: list,
    identifier_fields: dict[str, set[str]],
) -> tuple[ToolReadiness, ...]:
    """Per-tool counts, ordered by call volume: the tools worth fixing come first."""
    by_tool: dict[str, list] = {}
    for inv in invocations:
        by_tool.setdefault(inv.tool, []).append(inv)

    out = [
        ToolReadiness(
            tool=name,
            calls=len(calls),
            with_arguments=sum(1 for c in calls if c.arguments),
            with_object_response=sum(
                1
                for c in calls
                if c.status is CallStatus.OK and isinstance(c.response, dict) and c.response
            ),
            error_calls=sum(1 for c in calls if c.status is CallStatus.ERROR),
            identifier_fields=tuple(sorted(identifier_fields.get(name, ()))),
        )
        for name, calls in by_tool.items()
    ]
    return tuple(sorted(out, key=lambda t: (-t.calls, t.tool)))


def _decide(signals: tuple[Signal, ...], tools: tuple[ToolReadiness, ...]) -> tuple[Verdict, tuple[str, ...]]:
    """Turn the signals into a position, and say why in the customer's own terms.

    ``invocation_points`` is special-cased: it is not one failing signal among six,
    it is the precondition for the other five meaning anything. When it fails the
    others are noise -- there is nothing to have arguments *of* -- so the verdict is
    NO_GO and the reason names only the real problem.
    """
    reasons: list[str] = []
    points = next(s for s in signals if s.name == "invocation_points")
    if not points.present:
        return Verdict.NO_GO, (
            f"only {points.observed}/{points.population} traces contain a recorded tool call. "
            "This export is a transcript log, not a tool-call log: the action space, the state "
            "and the verifier are all unrecoverable from it. Fix the tracer before rebuilding "
            "anything.",
        )

    failures = [s for s in signals if s.blocking and not s.present and s.name != "invocation_points"]
    for signal in failures:
        reasons.append(
            f"{signal.name}: {signal.observed}/{signal.population} -- {signal.detail}"
        )

    unusable = [t for t in tools if not t.reconstructible]
    if unusable:
        listed = ", ".join(f"{t.tool} ({t.note})" for t in unusable[:5])
        suffix = f", +{len(unusable) - 5} more" if len(unusable) > 5 else ""
        reasons.append(f"{len(unusable)}/{len(tools)} tools cannot be reconstructed: {listed}{suffix}")

    hard = {"responses", "identifiers"}
    if any(s.name in hard for s in failures):
        return Verdict.NO_GO, tuple(reasons)
    if reasons:
        return Verdict.PARTIAL, tuple(reasons)
    return Verdict.GO, (
        f"{len(tools)} tools, {points.population} traces: arguments, responses and recurring "
        "identifiers are all present. Reconstruction can proceed to the fidelity gate, which is "
        "where the real number comes from.",
    )
