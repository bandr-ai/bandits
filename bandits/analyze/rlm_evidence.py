"""Indexed, range-retrievable evidence for the miner, and the two helpers over it.

Why an index rather than a rendering. The trajectory view used to flatten a run
into lines, which lost what a workflow export actually records: system prompts
inside each model call's input, tool calls expressed only as structured output
parts, and the clues ingest kept when no task resolved. Appending every model
input instead grew one twenty-trace batch roughly tenfold, repeated the same
context on every call, and still cut long payloads by length. Neither is a
faithful way to show a source.

So a run is indexed once: every recorded input/output message, tool call,
tool payload and candidate instruction becomes an item with an index-issued
reference, a source reference naming where it was recorded, the origin of the
text (an internal model's system prompt is not a user's request), and the full
content. Repeated content is stored once by digest; each occurrence keeps its
own place in the chronology. The miner navigates with :meth:`inspect_run` and
reads exact character ranges with :meth:`get_evidence`. Nothing here
summarizes, ranks or decides what a run was about — parsing a recorded context
payload locates it, it does not make it the task.

References are identifiers, never paths, expressions or URLs: a reference that
the index did not issue for that run resolves to nothing.

Two evidence policies, because two consumers need different things. An
*account* must describe intent, execution and result, so it reads every recorded
field — an ``expected`` key can hold a requirement and an ``outcome`` key an
observed change — with values the deterministic analysis classified annotated by
that classification, so an evaluator's score is marked as a claim instead of
being hidden or mistaken for an observation. *Grouping* (family formation) keeps
the trajectory view's outcome-key redaction: it should reopen requested work,
not results. Key redaction narrows what grouping sees; it cannot remove outcome
information that model-written milestones or free text carry, and nothing here
claims it does.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from bandits.analyze.rlm_corpus import _OUTCOME_CLAIMS, _redact
from bandits.traces import Span, SpanKind, Trace, WorkflowNode

INDEX_VERSION = 2
"""Bumped whenever references, origins or serialization change. Part of every
account's identity, so an account citing refs from another index version is
never silently reused against this one.

2: evidence policies; overview pages are one bounded row stream; event ref
lists and limitations are retrievable items."""

SERIALIZATION = "json-sorted-v1"
"""How structured values become text, so character offsets into them are stable."""

POLICY_VERSION = 1


class EvidencePolicy(str, Enum):
    ACCOUNT = "account"
    """Every recorded field and span status; nothing withheld by key name.
    Analysis-classified values are annotated, never removed."""

    GROUPING = "grouping"
    """Outcome-bearing keys withheld (``rlm_corpus._WITHHELD_KEYS``) and span
    status omitted, for family formation."""


EVALUATOR_CLAIMS = frozenset({"recorded_score"})
"""Analysis claims that report somebody's judgement of a run, not what happened.

Classified by the deterministic analysis's own rules (``outcomes.py``), not by
key name here. The other outcome claims — exit codes, final state fields, span
errors — are observations of the run and are annotated as such."""

INSPECT_PAGE_CHARS = 6000
"""Default ceiling on one serialized overview page. A starting budget, not a
measured optimum: it bounds what one call prints into the root's history, and
nothing it leaves out is lost — every row is on some page and every ref is
retrievable."""

EVIDENCE_DEFAULT_LIMIT = 4096
EVIDENCE_MAX_LIMIT = 8192
_EXCERPT = 160

_BINARY_PARTS = frozenset({"blob", "file", "uri"})

_LABELED = re.compile(r"\s*([^\n{}\[\]]{1,80}):[ \t]*\r?\n")
"""A short label line ending in a colon, e.g. ``Some context:`` before a JSON body.

Generic on purpose: the label is recorded, never matched against a known list,
so the parser works on any export that prefixes a payload with a heading."""


def serialize(value: Any) -> str:
    """Deterministic JSON text for a structured value. Offsets index into this."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def excerpt(text: str, width: int = _EXCERPT) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


@dataclass(frozen=True)
class LabeledPayload:
    """A recorded text split into an optional label, a JSON value and the rest.

    Offsets are Python character offsets into the original text, so each piece
    stays traceable to exactly where it was recorded. ``status`` is ``parsed``,
    ``no_json`` (the text does not start a JSON value after its label) or
    ``malformed`` (it starts one that does not decode); the latter two keep the
    text whole rather than guessing.
    """

    status: str
    label: str | None = None
    value: Any = None
    json_start: int | None = None
    json_end: int | None = None
    suffix_start: int | None = None
    suffix: str = ""
    error: str = ""


def parse_labeled_json(text: str) -> LabeledPayload:
    """Decode the first JSON value after an optional ``Label:`` line.

    ``JSONDecoder.raw_decode`` rather than ``json.loads``: a payload followed by
    free-text hints is common, and decoding the whole remainder rejects it as
    "extra data" while the object itself is well formed. The trailing text is
    kept separately with its offset, never discarded.
    """
    match = _LABELED.match(text)
    label = match.group(1).strip() if match else None
    position = match.end() if match else 0
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] not in "{[":
        return LabeledPayload(status="no_json", label=label)
    try:
        value, end = json.JSONDecoder().raw_decode(text, position)
    except ValueError as exc:
        return LabeledPayload(status="malformed", label=label, error=str(exc))
    return LabeledPayload(
        status="parsed",
        label=label,
        value=value,
        json_start=position,
        json_end=end,
        suffix_start=end,
        suffix=text[end:],
    )


@dataclass
class EvidenceItem:
    """One indexed piece of recorded evidence."""

    ref: str
    run_id: str
    source_ref: str
    """Where it was recorded: span/field/index path, or a request/clue field."""

    origin: str
    """What produced the text, e.g. ``model_input.system`` (an internal model's
    instruction), ``tool.result`` or ``clue.first_model_prompt``. Never the
    initiating actor: a ``model_input.user`` message is the program's prompt to
    one call, not necessarily a person's request."""

    evidence_type: str
    """``text``, ``json``, ``binary`` or ``unavailable``."""

    representation: str
    digest: str | None = None
    span_id: str | None = None
    available: bool = True
    limitations: tuple[str, ...] = ()
    withheld: tuple[str, ...] = ()
    """Keys the view policy removed from this item's structured value."""

    descriptor: dict[str, Any] = field(default_factory=dict)
    """For binary/unavailable items: what is known about the missing content."""

    analysis_claims: tuple[dict[str, Any], ...] = ()
    """What the deterministic analysis read off this item's span, each labeled
    ``evaluator_assertion`` or ``observation``. Account policy only."""


@dataclass
class EventRow:
    order: int
    span_id: str
    kind: str
    name: str
    parent_span_id: str | None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    payload_refs: list[str] = field(default_factory=list)
    linked_spans: list[str] = field(default_factory=list)
    status: str | None = None
    """The span's recorded status; account policy only."""

    def refs(self) -> list[str]:
        return self.inputs + self.outputs + self.payload_refs


class RunIndex:
    """Every indexed item of one run, in chronological order of its events."""

    def __init__(self, run_id: str, *, corpus_version: str, policy: EvidencePolicy) -> None:
        self.run_id = run_id
        self.corpus_version = corpus_version
        self.policy = policy
        self.items: dict[str, EvidenceItem] = {}
        self.events: list[EventRow] = []
        self.candidates: list[str] = []
        self.system_prompts: dict[str, list[str]] = {}
        """Distinct system prompt ref -> occurrence refs, in order."""

        self.limitations: list[str] = []
        self._content: dict[str, str] = {}
        self._first_occurrence: dict[str, str] = {}

    # --- construction --------------------------------------------------------

    def _store(self, text: str) -> str:
        digest = digest_of(text)
        self._content.setdefault(digest, text)
        return digest

    def add(self, item: EvidenceItem, content: str | None) -> EvidenceItem:
        if item.ref in self.items:  # pragma: no cover - refs are constructed unique
            raise ValueError(f"duplicate evidence ref {item.ref}")
        if content is not None:
            item.digest = self._store(content)
            self._first_occurrence.setdefault(item.digest, item.ref)
        self.items[item.ref] = item
        return item

    # --- reading ---------------------------------------------------------------

    def content(self, ref: str) -> str | None:
        item = self.items.get(ref)
        if item is None or item.digest is None:
            return None
        return self._content[item.digest]

    def first_occurrence(self, ref: str) -> str | None:
        """The earliest ref carrying the same content, when this one repeats it."""
        item = self.items.get(ref)
        if item is None or item.digest is None:
            return None
        first = self._first_occurrence.get(item.digest)
        return first if first != ref else None

    def span_ids(self) -> set[str]:
        return {event.span_id for event in self.events}

    def all_text(self) -> Iterable[str]:
        return self._content.values()


def _structured(value: Any, policy: EvidencePolicy) -> tuple[Any, tuple[str, ...]]:
    """A structured payload under the policy, plus the keys it withheld."""
    if policy is EvidencePolicy.ACCOUNT:
        return value, ()
    removed: set[str] = set()
    kept = _redact(value, removed)
    return kept, tuple(sorted(removed))


def _messages(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, dict)]


def _text_of(part: dict[str, Any]) -> str | None:
    content = part.get("content")
    return content if isinstance(content, str) else None


class _Builder:
    def __init__(
        self,
        trace: Trace,
        index: RunIndex,
        control_markers: Sequence[str],
        claims: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.trace = trace
        self.index = index
        self.policy = index.policy
        self.markers = tuple(control_markers)
        self.claims = claims if index.policy is EvidencePolicy.ACCOUNT else {}
        """span id -> analysis claims read off that span."""
        self.call_refs: dict[str, str] = {}
        """Recorded tool-call id -> the ref of the call part that issued it."""

    def _strip(self, text: str) -> tuple[str, tuple[str, ...]]:
        removed = []
        for marker in self.markers:
            if marker in text:
                removed.append(marker)
                text = text.replace(marker, "")
        return text, tuple(removed)

    def _claims(self, span_id: str | None, origin: str) -> tuple[dict[str, Any], ...]:
        # Attached to what a span returned or carried, not to the prompts sent to it.
        if span_id is None or origin.startswith(("model_input", "clue", "request", "user_turn")):
            return ()
        return tuple(self.claims.get(span_id, ()))

    def text(self, ref: str, text: str, *, source_ref: str, origin: str, span_id: str | None):
        text, removed = self._strip(text)
        return self.index.add(
            EvidenceItem(
                ref=ref,
                run_id=self.index.run_id,
                source_ref=source_ref,
                origin=origin,
                evidence_type="text",
                representation="text",
                span_id=span_id,
                withheld=removed,
                analysis_claims=self._claims(span_id, origin),
            ),
            text,
        )

    def structured(
        self, ref: str, value: Any, *, source_ref: str, origin: str, span_id: str | None
    ):
        kept, withheld = _structured(value, self.policy)
        return self.index.add(
            EvidenceItem(
                ref=ref,
                run_id=self.index.run_id,
                source_ref=source_ref,
                origin=origin,
                evidence_type="json",
                representation=SERIALIZATION,
                span_id=span_id,
                withheld=withheld,
                limitations=(
                    (
                        f"withheld by the {self.policy.value} policy v{POLICY_VERSION}: "
                        f"{', '.join(withheld)}",
                    )
                    if withheld
                    else ()
                ),
                analysis_claims=self._claims(span_id, origin),
            ),
            serialize(kept),
        )

    def unavailable(self, ref: str, *, source_ref: str, origin: str, reason: str, **descriptor):
        return self.index.add(
            EvidenceItem(
                ref=ref,
                run_id=self.index.run_id,
                source_ref=source_ref,
                origin=origin,
                evidence_type="binary" if descriptor.get("part_type") else "unavailable",
                representation="descriptor",
                available=False,
                limitations=(reason,),
                descriptor=descriptor,
            ),
            None,
        )

    # --- candidate instructions ---------------------------------------------

    def candidate_text(
        self, ref: str, text: str, *, source_ref: str, origin: str, span_id: str | None
    ):
        """Index a candidate instruction and, when it carries one, its JSON payload."""
        self.text(ref, text, source_ref=source_ref, origin=origin, span_id=span_id)
        self.index.candidates.append(ref)
        parsed = parse_labeled_json(text)
        if parsed.status == "parsed":
            self.structured(
                f"{ref}.json",
                parsed.value,
                source_ref=f"{source_ref}[{parsed.json_start}:{parsed.json_end}]",
                origin=f"{origin}.json",
                span_id=span_id,
            )
            self.index.candidates.append(f"{ref}.json")
            if parsed.suffix.strip():
                self.text(
                    f"{ref}.suffix",
                    parsed.suffix,
                    source_ref=f"{source_ref}[{parsed.suffix_start}:]",
                    origin=f"{origin}.suffix",
                    span_id=span_id,
                )
                self.index.candidates.append(f"{ref}.suffix")
        elif parsed.status == "malformed":
            self.index.limitations.append(
                f"{ref} begins a JSON value after its label that does not decode "
                f"({parsed.error}); kept as raw text"
            )

    def request(self) -> None:
        request = self.trace.request
        if self.trace.task and request is None:
            self.candidate_text(
                "task", self.trace.task, source_ref="trace.task", origin="trace.task", span_id=None
            )
        for i, turn in enumerate(self.trace.user_turns):
            self.candidate_text(
                f"turn{i}",
                turn.text,
                source_ref=f"trace.user_turns[{i}]",
                origin="user_turn.declared" if turn.origin == "declared" else "user_turn.recorded",
                span_id=turn.after_span_id,
            )
        if request is None:
            return
        known_spans = {span.span_id for span in self.trace.spans}
        known_nodes = {node.span_id for node in self.trace.workflow_nodes}
        source = request.source_span_id
        if source is None:
            self.index.limitations.append(
                f"no invocation span was selected for this run ({request.invocation_basis})"
            )
        elif source not in known_spans and source not in known_nodes:
            self.index.limitations.append(
                f"the request's source span {source} is a structural reference only: it is not "
                "among the exported spans or workflow nodes, so no source record is retrievable"
            )
        if request.task is not None:
            self.candidate_text(
                "request.task",
                request.task,
                source_ref=f"request.task ({request.task_path})",
                origin="request.task.declared",
                span_id=source,
            )
        if request.raw_input in (None, "", {}, []):
            self.unavailable(
                "request.input",
                source_ref="request.raw_input",
                origin="request.raw_input",
                reason="the invocation's input was not recorded",
            )
            # Listed anyway: where the request would have been is part of the
            # answer, and its absence must be visible rather than implied.
            self.index.candidates.append("request.input")
        elif isinstance(request.raw_input, str):
            self.candidate_text(
                "request.input",
                request.raw_input,
                source_ref="request.raw_input",
                origin="request.raw_input",
                span_id=source,
            )
        else:
            self.structured(
                "request.input",
                request.raw_input,
                source_ref="request.raw_input",
                origin="request.raw_input",
                span_id=source,
            )
            self.index.candidates.append("request.input")
        for i, clue in enumerate(request.tentative_tasks):
            self.candidate_text(
                f"clue{i}.{clue.clue}",
                clue.value,
                source_ref=f"request.tentative_tasks[{i}]",
                origin=f"clue.{clue.clue}",
                span_id=clue.span_id,
            )

    # --- chronology --------------------------------------------------------------

    def _message_items(
        self,
        prefix: str,
        messages: list[dict[str, Any]],
        *,
        span_id: str,
        field_name: str,
        row: EventRow,
    ) -> list[str]:
        refs: list[str] = []
        direction = "input" if field_name.endswith("input.messages") else "output"
        for i, message in enumerate(messages):
            role = str(message.get("role") or "unknown")
            for j, part in enumerate(message.get("parts") or []):
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                ref = f"{prefix}{i}" + (f".{j}" if j else "")
                source_ref = f"span:{span_id}/{field_name}[{i}].parts[{j}]"
                origin = f"model_{direction}.{role}"
                if kind == "text" and _text_of(part) is not None:
                    item = self.text(
                        ref, _text_of(part), source_ref=source_ref, origin=origin, span_id=span_id
                    )
                    if role == "system":
                        self.index.system_prompts.setdefault(self._system_ref(item), []).append(ref)
                    refs.append(ref)
                elif kind == "tool_call":
                    payload = {"name": part.get("name"), "arguments": part.get("arguments")}
                    self.structured(
                        ref,
                        payload,
                        source_ref=source_ref,
                        origin=f"{origin}.tool_call",
                        span_id=span_id,
                    )
                    call_id = part.get("id")
                    if direction == "output":
                        row.tool_calls.append(
                            {"ref": ref, "name": part.get("name"), "call_id": call_id}
                        )
                        if isinstance(call_id, str):
                            self.call_refs.setdefault(call_id, ref)
                    refs.append(ref)
                elif kind == "tool_call_response":
                    payload = {k: part.get(k) for k in ("id", "name", "result") if k in part}
                    self.structured(
                        ref,
                        payload,
                        source_ref=source_ref,
                        origin=f"{origin}.tool_result",
                        span_id=span_id,
                    )
                    refs.append(ref)
                elif kind in _BINARY_PARTS:
                    self.unavailable(
                        ref,
                        source_ref=source_ref,
                        origin=origin,
                        reason=f"{kind} part: binary/multimodal content is not retrievable as text",
                        part_type=kind,
                        mime_type=part.get("mime_type") or part.get("mime"),
                        uri=part.get("uri") if isinstance(part.get("uri"), str) else None,
                    )
                    refs.append(ref)
                else:
                    self.structured(
                        ref, part, source_ref=source_ref, origin=f"{origin}.{kind}", span_id=span_id
                    )
                    refs.append(ref)
        return refs

    def _system_ref(self, item: EvidenceItem) -> str:
        return f"sys:{item.digest[:10]}"

    def spans(self) -> None:
        """One event per recorded span or workflow node, in start-time order.

        The order is fixed before any ref is issued, so ``e<n>`` is always the
        ``n``-th event of the chronology the overview pages through.
        """
        timeline = sorted(
            [(span.started_at, 1, i, span) for i, span in enumerate(self.trace.spans)]
            + [(node.started_at, 0, i, node) for i, node in enumerate(self.trace.workflow_nodes)],
            key=lambda entry: entry[:3],
        )
        links: dict[str, list[str]] = {}
        for link in self.trace.evidence:
            if link.kind == "tool_result" and link.target_span_id:
                links.setdefault(link.call_span_id, []).append(link.target_span_id)
        rows_by_span: dict[str, EventRow] = {}
        for order, (_, is_span, _, record) in enumerate(timeline):
            row = self._span_row(order, record) if is_span else self._node_row(order, record)
            if self.policy is EvidencePolicy.ACCOUNT:
                row.status = record.status.value
            rows_by_span[record.span_id] = row
            self.index.events.append(row)
        for span_id, targets in links.items():
            row = rows_by_span.get(span_id)
            if row is None:
                continue
            row.linked_spans.extend(t for t in targets if t not in row.linked_spans)
            for call in row.tool_calls:
                call_id = call.get("call_id")
                match = next(
                    (t for t in targets if isinstance(call_id, str) and call_id in t), None
                )
                if match is not None and match in rows_by_span:
                    call["result_span"] = match
                    call["result_refs"] = rows_by_span[match].payload_refs

    def finish(self) -> None:
        """Make every event's ref list and every limitation retrievable by ref.

        An overview page that must shorten a row to fit keeps a pointer here, so
        nothing a page leaves out stops being discoverable.
        """
        for event in self.index.events:
            self.structured(
                f"e{event.order}.refs",
                event.refs(),
                source_ref=f"index:e{event.order}/refs",
                origin="index.event_refs",
                span_id=None,
            )
        for i, text in enumerate(self.index.limitations):
            self.text(
                f"limitation{i}",
                text,
                source_ref=f"index:limitations[{i}]",
                origin="index.limitation",
                span_id=None,
            )

    def _node_row(self, order: int, node: WorkflowNode) -> EventRow:
        row = EventRow(order, node.span_id, "node", node.name, node.parent_span_id)
        for name, value in (("input", node.input), ("output", node.output)):
            if value is None:
                continue
            self.structured(
                f"e{order}.{name}",
                value,
                source_ref=f"node:{node.span_id}/{name}",
                origin=f"node.{name}",
                span_id=node.span_id,
            )
            row.payload_refs.append(f"e{order}.{name}")
        return row

    def _span_row(self, order: int, span: Span) -> EventRow:
        row = EventRow(order, span.span_id, span.kind.value, span.name, span.parent_span_id)
        prefix = f"e{order}"
        inputs = _messages(span.attributes.get("gen_ai.input.messages"))
        outputs = _messages(span.attributes.get("gen_ai.output.messages"))
        row.inputs = self._message_items(
            f"{prefix}.in",
            inputs,
            span_id=span.span_id,
            field_name="gen_ai.input.messages",
            row=row,
        )
        row.outputs = self._message_items(
            f"{prefix}.out",
            outputs,
            span_id=span.span_id,
            field_name="gen_ai.output.messages",
            row=row,
        )
        role = "tool" if span.kind is SpanKind.TOOL else "model"
        if span.arguments:
            self.structured(
                f"{prefix}.args",
                span.arguments,
                source_ref=f"span:{span.span_id}/arguments",
                origin=f"{role}.arguments",
                span_id=span.span_id,
            )
            row.payload_refs.append(f"{prefix}.args")
        if span.output is not None and not (outputs and span.kind is SpanKind.MODEL):
            origin = f"{role}.result" if role == "tool" else "model.output"
            if isinstance(span.output, str):
                self.text(
                    f"{prefix}.result",
                    span.output,
                    source_ref=f"span:{span.span_id}/output",
                    origin=origin,
                    span_id=span.span_id,
                )
            else:
                self.structured(
                    f"{prefix}.result",
                    span.output,
                    source_ref=f"span:{span.span_id}/output",
                    origin=origin,
                    span_id=span.span_id,
                )
            row.payload_refs.append(f"{prefix}.result")
        elif span.kind is SpanKind.TOOL and span.output is None:
            self.unavailable(
                f"{prefix}.result",
                source_ref=f"span:{span.span_id}/output",
                origin="tool.result",
                reason="the source recorded no result for this tool call",
            )
            row.payload_refs.append(f"{prefix}.result")
        for message in inputs:
            for part in message.get("parts") or []:
                if isinstance(part, dict) and part.get("type") == "tool_call_response":
                    call_ref = self.call_refs.get(part.get("id"))
                    if call_ref:
                        row.linked_spans.append(f"responds-to:{call_ref}")
        return row


class EvidenceCatalog:
    """Indexes for every run the miner may read, plus who read what.

    Built lazily per run and cached, so the 32 MB corpus is not indexed whole
    for a three-run diagnostic. Holds an access log the host reads back to fill
    an account's ``inspected`` coverage: that is the record of what was
    actually retrieved, not what merely appeared in a batch.
    """

    def __init__(
        self,
        traces: dict[str, Trace],
        *,
        corpus_version: str,
        policy: EvidencePolicy = EvidencePolicy.ACCOUNT,
        control_markers: Sequence[str] = (),
        analysis: Any = None,
        page_chars: int = INSPECT_PAGE_CHARS,
    ) -> None:
        if not 1000 <= page_chars <= 50_000:
            raise ValueError("an overview page must allow 1,000 to 50,000 characters")
        self._traces = traces
        self.corpus_version = corpus_version
        self.evidence_policy = policy
        self.page_chars = page_chars
        self._markers = tuple(control_markers)
        self._claims = _claims_by_span(analysis)
        self._indexes: dict[str, RunIndex] = {}
        self._lock = threading.Lock()
        self._scope: tuple[str, ...] | None = None
        self.accesses: list[dict[str, Any]] = []

    @property
    def policy(self) -> str:
        return f"{self.evidence_policy.value}/v{POLICY_VERSION}"

    def index(self, run_id: str) -> RunIndex:
        with self._lock:
            if run_id not in self._indexes:
                trace = self._traces.get(run_id)
                if trace is None:
                    raise KeyError(f"no run {run_id!r} in this corpus")
                index = RunIndex(
                    run_id, corpus_version=self.corpus_version, policy=self.evidence_policy
                )
                builder = _Builder(trace, index, self._markers, self._claims.get(run_id, {}))
                builder.request()
                builder.spans()
                builder.finish()
                self._indexes[run_id] = index
            return self._indexes[run_id]

    # --- scope and access log -------------------------------------------------

    def set_scope(self, run_ids: Sequence[str] | None) -> None:
        """Restrict the helpers to these runs (one invocation's runs), or lift it."""
        self._scope = tuple(run_ids) if run_ids is not None else None

    def _check_run(self, run_id: Any) -> RunIndex:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        if self._scope is not None and run_id not in self._scope:
            raise ValueError(
                f"run {run_id!r} is not in this invocation; readable runs: {list(self._scope)}"
            )
        try:
            return self.index(run_id)
        except KeyError as exc:
            raise ValueError(f"unknown run {run_id!r}") from exc

    def retrieved(self, run_id: str, *, since: int = 0) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Refs, and ``ref[start:end]`` ranges, actually returned for a run.

        Read from the access log from position ``since``, so an invocation counts
        only its own retrievals. A ref that merely appeared in an overview is
        not here: only ``get_evidence`` returning content counts as read.
        """
        refs: dict[str, None] = {}
        ranges: dict[str, None] = {}
        for access in self.accesses[since:]:
            if (
                access.get("run_id") == run_id
                and access["tool"] == "get_evidence"
                and access.get("available")
            ):
                refs.setdefault(access["ref"], None)
                ranges.setdefault(f"{access['ref']}[{access['start']}:{access['end']}]", None)
        return tuple(refs), tuple(ranges)

    def _log(self, entry: dict[str, Any], returned: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.accesses.append(entry)
        # What was actually handed to the miner, so a retrieval can be replayed
        # from the record: refs, ranges and the returned content itself.
        from bandits import ledger

        ledger.record({"event_type": "evidence_access", **entry, "returned": returned})

    # --- the two helpers -----------------------------------------------------------

    def inspect_run(self, run_id: str, cursor: int = 0, limit: int = 0) -> dict:
        """A deterministic, bounded overview of one run. No summary, no judgement.

        One stream of rows, paged by ``cursor``: candidate instructions, distinct
        system prompts and limitations first, then events in chronological
        order. The page fills until ``limit`` rows (0, the default: as many as
        fit) or until the whole serialized page — header, rows and continuation
        — would pass ``page_chars``. The character cap is the bound; there is no
        row ceiling to trip over. A row too large to fit is shortened to its
        refs, and one too large even then to a pointer (``refs_ref``) whose full
        ref list ``get_evidence`` returns; every page holds at least one row.
        """
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer from a previous next_cursor")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer (0: as many rows as fit)")
        index = self._check_run(run_id)
        stream = self._stream(index)
        total = len(stream)
        if cursor > total:
            raise ValueError(f"cursor {cursor} is past the end; this run has {total} rows")

        header: dict[str, Any] = {
            "run_id": run_id,
            "corpus_version": index.corpus_version,
            "index_version": INDEX_VERSION,
            "evidence_policy": self.policy,
            "total_rows": total,
            "total_events": len(index.events),
            "cursor": cursor,
        }
        if cursor == 0:
            header["note"] = (
                "rows of type candidate, system_prompt and limitation come first, then "
                "events in chronological order; read any ref with get_evidence"
            )
        rows: list[dict[str, Any]] = []
        position = cursor
        stop = total if limit == 0 else min(total, cursor + limit)
        while position < stop:
            kind, payload = stream[position]
            # A row that does not fit whole moves to the next page whole. Only a
            # row that cannot fit even alone is shortened, so shortening means
            # "too large for any page", never "the page was nearly full".
            details = ("full",) if rows else ("full", "short", "pointer")
            fitted = None
            for detail in details:
                row = self._render(index, kind, payload, position, detail)
                if self._fits(header, rows + [row], total):
                    fitted = row
                    break
            if fitted is None:
                if rows:
                    break
                fitted = self._render(index, kind, payload, position, "pointer")
            rows.append(fitted)
            position += 1
        page = {**header, "rows": rows, **self._continuation(run_id, position, total)}
        self._log(
            {
                "tool": "inspect_run",
                "run_id": run_id,
                "cursor": cursor,
                "limit": limit,
                "returned_rows": [r["row"] for r in rows],
                "available": True,
            },
            page,
        )
        return page

    @staticmethod
    def _continuation(run_id: str, position: int, total: int) -> dict[str, Any]:
        done = position >= total
        return {
            "done": done,
            "remaining_rows": total - position,
            "next_cursor": None if done else position,
            "next_call": "" if done else f"inspect_run({run_id!r}, cursor={position})",
        }

    def _fits(self, header: dict[str, Any], rows: list[dict[str, Any]], total: int) -> bool:
        # Measured with the longest continuation this run can produce, so adding
        # the real one afterwards cannot push the page over.
        worst = self._continuation(header["run_id"], total - 1 if total else 0, total)
        worst["remaining_rows"] = total
        return len(serialize({**header, "rows": rows, **worst})) <= self.page_chars

    @staticmethod
    def _stream(index: RunIndex) -> list[tuple[str, Any]]:
        return (
            [("candidate", ref) for ref in index.candidates]
            + [("system_prompt", item) for item in index.system_prompts.items()]
            + [("limitation", i) for i in range(len(index.limitations))]
            + [("event", event) for event in index.events]
        )

    def _render(
        self, index: RunIndex, kind: str, payload: Any, row: int, detail: str
    ) -> dict[str, Any]:
        if kind == "candidate":
            entry = {"row": row, "type": "candidate", **self._candidate(index, payload)}
            if detail != "full":
                entry.pop("excerpt", None)
                entry.pop("source_ref", None)
            if detail == "pointer":
                entry.pop("limitations", None)
            return entry
        if kind == "system_prompt":
            shared, occurrences = payload
            text = index.content(occurrences[0]) or ""
            entry = {
                "row": row,
                "type": "system_prompt",
                "ref": occurrences[0],
                "shared_id": shared,
                "occurrences": len(occurrences),
                "length": len(text),
            }
            if detail == "full":
                entry["excerpt"] = excerpt(text, 120)
            return entry
        if kind == "limitation":
            text = index.limitations[payload]
            entry = {"row": row, "type": "limitation", "ref": f"limitation{payload}"}
            if detail == "full":
                entry["excerpt"] = excerpt(text, 300)
            return entry
        event: EventRow = payload
        if detail == "pointer":
            return {
                "row": row,
                "type": "event",
                "order": event.order,
                "span_id": event.span_id[:80],
                "kind": event.kind,
                "refs_ref": f"e{event.order}.refs",
                "note": "shortened to fit the page; get_evidence on refs_ref lists every ref",
            }
        return {"row": row, "type": "event", **self._event(index, event, full=detail == "full")}

    def _candidate(self, index: RunIndex, ref: str) -> dict[str, Any]:
        item = index.items[ref]
        if not item.available:
            return {
                "ref": ref,
                "origin": item.origin,
                "source_ref": item.source_ref,
                "available": False,
                "limitations": list(item.limitations),
            }
        text = index.content(ref) or ""
        return {
            "ref": ref,
            "origin": item.origin,
            "source_ref": item.source_ref,
            "length": len(text),
            "excerpt": excerpt(text),
        }

    def _event(self, index: RunIndex, event: EventRow, *, full: bool) -> dict[str, Any]:
        def describe(ref: str) -> dict[str, Any]:
            item = index.items[ref]
            entry: dict[str, Any] = {"ref": ref, "origin": item.origin}
            if not item.available:
                entry["available"] = False
                return entry
            text = index.content(ref) or ""
            entry["length"] = len(text)
            if item.analysis_claims:
                entry["analysis_claims"] = [
                    c["kind"] + ":" + c["claim"] for c in item.analysis_claims
                ]
            if item.origin.startswith("model_input.system"):
                entry["system_prompt"] = f"sys:{item.digest[:10]}"
                return entry
            repeat = index.first_occurrence(ref)
            if repeat:
                entry["repeat_of"] = repeat
            elif full:
                entry["excerpt"] = excerpt(text, 100)
            return entry

        row: dict[str, Any] = {
            "order": event.order,
            "span_id": event.span_id,
            "kind": event.kind,
            "name": event.name,
            "parent_span_id": event.parent_span_id,
        }
        if event.status is not None:
            row["status"] = event.status
        if event.inputs:
            row["inputs"] = [describe(ref) for ref in event.inputs]
        if event.outputs:
            row["outputs"] = [describe(ref) for ref in event.outputs]
        if event.tool_calls:
            row["tool_calls"] = event.tool_calls
        if event.payload_refs:
            row["payloads"] = [describe(ref) for ref in event.payload_refs]
        if event.linked_spans:
            row["linked"] = event.linked_spans
        return row

    def get_evidence(
        self, run_id: str, ref: str, start: int = 0, limit: int = EVIDENCE_DEFAULT_LIMIT
    ) -> dict:
        """Exact characters ``[start, start+limit)`` of one indexed item.

        ``limit`` above :data:`EVIDENCE_MAX_LIMIT` is capped, not refused: the
        cap bounds one observation, and ``next_start`` continues the rest. Only
        input that cannot mean anything — a negative offset, a non-integer, a
        start past the end — is an error. Unknown and unavailable refs return
        ``available: false`` with ``content: ""`` and an ``unavailable_reason``;
        an available item that is genuinely empty has ``available: true``.
        """
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("start must be a non-negative integer character offset")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer number of characters")
        capped = min(limit, EVIDENCE_MAX_LIMIT)
        index = self._check_run(run_id)
        resolved = ref
        if isinstance(ref, str) and ref.startswith("sys:"):
            resolved = next(
                (occ[0] for shared, occ in index.system_prompts.items() if shared == ref), ref
            )
        item = index.items.get(resolved) if isinstance(resolved, str) else None
        if item is None:
            reason = (
                f"{ref!r} is not a reference this index issued for run {run_id}; "
                "use refs returned by inspect_run"
            )
            response = {
                "ref": ref,
                "source_version": index.corpus_version,
                "evidence_type": "unknown",
                "content": "",
                "available": False,
                "unavailable_reason": reason,
                "limitations": [reason],
            }
            self._log(
                {"tool": "get_evidence", "run_id": run_id, "ref": ref, "available": False},
                response,
            )
            return response
        base = {
            "ref": resolved,
            "source_ref": item.source_ref,
            "source_version": index.corpus_version,
            "origin": item.origin,
            "evidence_type": item.evidence_type,
            "representation": item.representation,
            "limitations": list(item.limitations),
            "analysis_claims": list(item.analysis_claims),
        }
        if not item.available:
            unavailable = {
                **base,
                "content": "",
                "available": False,
                "unavailable_reason": "; ".join(item.limitations) or "not recorded",
                "descriptor": item.descriptor,
            }
            self._log(
                {"tool": "get_evidence", "run_id": run_id, "ref": resolved, "available": False},
                unavailable,
            )
            return unavailable
        text = index.content(resolved) or ""
        total = len(text)
        if start > total:
            raise ValueError(f"start {start} is past the end of {resolved} ({total} characters)")
        end = min(total, start + capped)
        response = {
            **base,
            "digest": item.digest,
            "content": text[start:end],
            "range_start": start,
            "range_end": end,
            "total_length": total,
            "next_start": end if end < total else None,
            "done": end >= total,
            "available": True,
        }
        if capped != limit:
            response["limit_capped_to"] = capped
        self._log(
            {
                "tool": "get_evidence",
                "run_id": run_id,
                "ref": resolved,
                "start": start,
                "end": end,
                "available": True,
            },
            response,
        )
        return response

    def tools(self) -> list[Any]:
        """The two helpers as plain functions the sandbox can register.

        Plain functions with simple annotations: the installed interpreter
        registers a tool's parameters from its signature and passes only the
        simple types through. Every response is made sandbox-safe on the way
        out (see :func:`sandbox_safe`).
        """

        def inspect_run(run_id: str, cursor: int = 0, limit: int = 0) -> dict:
            """Bounded overview page of one run. Rows: type candidate/system_prompt/limitation (one "ref" each), then type event (inputs/outputs/payloads lists of {ref, origin, length}, tool_calls, or refs_ref when shortened). Continue with page["next_call"] until page["done"]. limit=0: as many rows as fit."""
            return sandbox_safe(self.inspect_run(run_id, cursor, limit))

        def get_evidence(
            run_id: str, ref: str, start: int = 0, limit: int = EVIDENCE_DEFAULT_LIMIT
        ) -> dict:
            """Exact text of one ref, characters [start, start+limit) (limit capped at 8192). Returns content, available, total_length, done, and next_start while more remains; when available is false, content is "" and unavailable_reason says why."""
            return sandbox_safe(self.get_evidence(run_id, ref, start, limit))

        return [inspect_run, get_evidence]

    def run_index(self, run_id: str) -> dict[str, Any]:
        """What an account invocation starts with: the run id and its first page.

        The first overview page itself, so no iteration goes to discovering the
        structure, and nothing is described twice. Served through
        :meth:`inspect_run`, so it is logged as read like any other page.
        """
        return sandbox_safe({"run_id": run_id, "first_page": self.inspect_run(run_id)})

    def resolves(self, run_id: str, ref: str) -> bool:
        index = self.index(run_id)
        if ref.startswith("sys:"):
            return ref in index.system_prompts
        return ref in index.items

    def origin_of(self, run_id: str, ref: str) -> str | None:
        index = self.index(run_id)
        if ref.startswith("sys:"):
            return "model_input.system" if ref in index.system_prompts else None
        item = index.items.get(ref)
        return item.origin if item else None


DECLARED_ORIGINS = frozenset(
    {"request.task.declared", "user_turn.recorded", "user_turn.declared", "trace.task"}
)
"""Origins that record a request as such. Anything else — a parsed context
payload, an internal model's prompt, a tentative clue — can support an
*inferred* intent but never a *declared* one."""


def declared_origin(origin: str | None) -> bool:
    if origin is None:
        return False
    return any(origin == o or origin.startswith(o + ".") for o in DECLARED_ORIGINS)


def _claims_by_span(analysis: Any) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Outcome claims the deterministic analysis read off each span, by run.

    The provenance that tells an evaluator's judgement from an observation:
    ``outcomes.py`` decided which spans carry a recorded score, an exit code or
    a final state, by its own rules. The values stay in the evidence; this only
    labels them.
    """
    found: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for evidence in getattr(analysis, "evidence", ()) or ():
        if evidence.claim not in _OUTCOME_CLAIMS or evidence.span_id is None:
            continue
        found.setdefault(evidence.trace_id, {}).setdefault(evidence.span_id, []).append(
            {
                "claim": evidence.claim,
                "kind": "evaluator_assertion"
                if evidence.claim in EVALUATOR_CLAIMS
                else "observation",
                "value": evidence.value,
            }
        )
    return found


def sandbox_safe(value: Any) -> Any:
    """A helper response with no ``None`` anywhere, for the sandbox boundary.

    DSPy 3.3.1's Deno/Pyodide bridge delivers JSON ``null`` as a JavaScript
    ``JsNull`` proxy, for which ``is None`` is false — verified in the real
    interpreter. A ``while page["next_cursor"] is not None`` loop would then
    never end, and ``content[:500]`` on missing evidence fails confusingly. So
    keys whose value is ``None`` are dropped (every response documents what an
    absent key means) and ``None`` inside a list becomes ``""``.
    """
    if isinstance(value, dict):
        return {key: sandbox_safe(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return ["" if item is None else sandbox_safe(item) for item in value]
    return value
