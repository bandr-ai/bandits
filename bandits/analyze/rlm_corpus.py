"""The only way the RLM miner may touch a corpus.

The experiment's whole claim rests on what the model was *not* shown. If the
miner can reach an assistant message, a tool call, a reward, or a known family
label, then a family it discovers may be a family the labels already contained,
and the result stops being evidence about semantic discovery. So access is a
narrow interface rather than a convention: the miner is handed one of these and
never the corpus, and there is no method here that returns anything but user
message text and opaque ids.

This is not semantic preprocessing. Building a view recognizes exactly two
things about a trace — where it starts and ends, and what role the source
attached to each recorded message. It does not summarize, normalize, extract
keywords, or decide what a request is about. A trace whose roles the source
never recorded is marked unreadable rather than reconstructed from some other
field, because inferring user messages from a trace's declared ``task`` would
quietly feed a source-derived summary into an arm whose premise is that only
recorded roles are read.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Any

from bandits.analyze.rlm_models import TraceView, UserMessageView
from bandits.traces import Span, SpanKind, Trace, TraceCorpus

_WITHHELD_KEYS = frozenset(
    {
        "score",
        "scores",
        "reward",
        "rewards",
        "success",
        "successful",
        "passed",
        "failed",
        "correct",
        "label",
        "labels",
        "verdict",
        "grade",
        "rating",
        "evaluation",
        "eval",
        "outcome",
        "is_correct",
        "ground_truth",
        "expected",
        "gold",
        "task_id",
        "family_id",
        "lineage_id",
    }
)
"""Keys stripped from tool payloads before the full-trajectory arm reads them.

Path F widens the input to the whole conversation; it does not widen it to the
answer. A benchmark harness routinely writes its own score into the span that
reports a result, and a miner shown that would group episodes by reward while
appearing to group them by task — the single most damaging outcome this
experiment can produce, because the resulting families look excellent.

A denylist is the wrong shape for a security boundary and the right shape here:
it is applied to a corpus whose outcome fields are already carried as separate
evidence, and the arm's whole purpose is to see tool payloads, so an allowlist
would empty the view it is meant to fill. What it cannot promise is completeness
against a source that names a reward something unforeseen, so
``withheld_fields`` records what it removed and the miner reports it.

Because a warning is not a boundary, the denylist is not the only defence.
:func:`audit_view_leakage` re-reads the built views against the outcome evidence
the deterministic analysis already extracted — the scores, exit codes and
terminal fields that layer found by its own rules — and reports any whose value
still appears in what the miner would be shown. That check does not depend on
guessing field names: it asks whether a known outcome value survived, whatever
it was called. A Path F run is only worth trusting when it comes back empty, and
:func:`ReadOnlyCorpus.leakage_report` is what the miner calls to find out.
"""

_MAX_PAYLOAD_CHARS = 2000
"""How much of one tool payload the trajectory arm may read.

A single tool result can be a whole file. Truncation is by length rather than by
significance, so nothing here decides which half of a payload mattered.
"""


def _strip_control_markers(text: str, markers: Sequence[str], removed: set[str]) -> str:
    """Strip caller-declared literal tokens from one turn's own text.

    Empty by default: this module makes no assumption that any corpus
    contains benchmark scaffolding. A source that does — tau2's simulator
    appends ``###TRANSFER###`` to a user turn's text on most episodes in the
    airline corpus, not only the ones that actually escalate to a human
    agent — is a fact about that source, declared by its caller at the
    ``ReadOnlyCorpus`` boundary, not knowledge this generic view carries for
    every corpus. A miner shown that marker treated "the trace ends in this
    token" as a verifiable outcome and built a family around ending state
    instead of requested task, precisely the confound ``_WITHHELD_KEYS``
    polices for structured fields. Unlike those fields this lives inside the
    message text itself, so it is stripped by substring rather than by key,
    and reported through ``withheld_fields`` the same way.
    """
    for marker in markers:
        if marker in text:
            removed.add(marker)
            text = text.replace(marker, "")
    return text.strip()


def _redact(payload: Any, removed: set[str], depth: int = 0) -> Any:
    """Strip outcome-bearing keys from a payload, recording what was taken.

    Depth-limited because a tool result can nest arbitrarily and this runs over
    every span of every trace; a payload deeper than this is truncated rather
    than walked, which keeps a pathological input from costing the whole view.
    """
    if depth > 6:
        return "[…nested]"
    if isinstance(payload, dict):
        kept = {}
        for key, value in payload.items():
            if isinstance(key, str) and key.strip().lower() in _WITHHELD_KEYS:
                removed.add(key.strip().lower())
                continue
            kept[key] = _redact(value, removed, depth + 1)
        return kept
    if isinstance(payload, (list, tuple)):
        return [_redact(item, removed, depth + 1) for item in payload]
    return payload


def _render_span(span: Span, removed: set[str]) -> str:
    """One span as a line the miner can read, with outcome fields withheld.

    ``status`` is deliberately not rendered. Whether a span errored is a fact
    about how the run went, and an arm that showed it would let "episodes that
    failed" become a family — the exact confound Path F is being tested for.
    """
    role = "assistant" if span.kind is SpanKind.MODEL else "tool"
    body: list[str] = []
    if span.arguments:
        body.append(json.dumps(_redact(span.arguments, removed), sort_keys=True, default=str))
    if span.output is not None:
        body.append(json.dumps(_redact(span.output, removed), sort_keys=True, default=str))
    text = " ".join(body)
    if len(text) > _MAX_PAYLOAD_CHARS:
        text = text[:_MAX_PAYLOAD_CHARS] + "…[truncated]"
    return f"[{role}:{span.name}] {text}".rstrip()


def _trajectory_messages(
    trace: Trace, control_markers: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The whole conversation as ordered lines, plus the fields withheld from it.

    User turns are interleaved by the span they followed, so the request and the
    work answering it read in the order they happened. A turn whose anchor span
    is missing is emitted first rather than dropped: losing a user message would
    quietly turn Path F into a strictly worse Path U.
    """
    removed: set[str] = set()
    by_anchor: dict[str | None, list[str]] = {}
    for turn in trace.user_turns:
        text = _strip_control_markers(turn.text, control_markers, removed)
        if text:
            by_anchor.setdefault(turn.after_span_id, []).append(f"[user] {text}")

    known = {span.span_id for span in trace.spans}
    lines: list[str] = list(by_anchor.pop(None, []))
    for anchor in list(by_anchor):
        if anchor not in known:
            lines.extend(by_anchor.pop(anchor))

    for span in trace.spans:
        lines.append(_render_span(span, removed))
        lines.extend(by_anchor.pop(span.span_id, []))
    return tuple(lines), tuple(sorted(removed))


def build_view(
    trace: Trace, view: TraceView, *, control_markers: Sequence[str] = ()
) -> UserMessageView:
    """Reduce one trace to the user messages the given arm may read.

    ``FIRST_USER_MESSAGE`` keeps only the opening request. The two user-message
    arms exist to measure whether later user turns carry task information or
    agent-dependent noise: a correction may say what the user actually wanted, or
    may say only that this particular agent went wrong, and which it is cannot be
    assumed.

    ``FULL_TRAJECTORY`` adds the assistant turns and tool activity, with rewards
    and evaluator labels stripped on the way through. It is readable whenever the
    trace has spans, because an episode with recorded work and no recorded user
    turn still shows what was done — but it is *unreadable* when the trace has
    neither, since an empty trajectory says nothing at all.

    ``control_markers`` is empty by default: this function makes no assumption
    that any corpus contains benchmark scaffolding. A source that does declares
    its own literal tokens through ``ReadOnlyCorpus``, at the boundary where
    that source-specific fact belongs — never baked in here for every caller.
    """
    if view is TraceView.FULL_TRAJECTORY:
        lines, withheld = _trajectory_messages(trace, control_markers)
        if not lines:
            return UserMessageView(
                trace_id=trace.trace_id,
                readable=False,
                unreadable_reason=(
                    "the source recorded neither user messages nor spans for this trace; "
                    "there is no trajectory to read"
                ),
            )
        return UserMessageView(trace_id=trace.trace_id, messages=lines, withheld_fields=withheld)

    removed: set[str] = set()
    texts = tuple(
        stripped
        for turn in trace.user_turns
        if (stripped := _strip_control_markers(turn.text, control_markers, removed))
    )
    if not texts:
        return UserMessageView(
            trace_id=trace.trace_id,
            readable=False,
            unreadable_reason=(
                "the source recorded no user-role messages for this trace; its request "
                "is not readable without inferring one from a field the miner may not see"
            ),
        )

    if view is TraceView.FIRST_USER_MESSAGE:
        texts = texts[:1]

    return UserMessageView(
        trace_id=trace.trace_id, messages=texts, withheld_fields=tuple(sorted(removed))
    )


class ReadOnlyCorpus:
    """Generic access to user-message views, and nothing else.

    Every method is deliberately shape-agnostic: count, list, fetch, sample. The
    miner writes its own hypotheses and decisions in its own artifacts, and has
    no operation here that could mutate a trace or read an excluded field.

    Views are built once at construction rather than per call, so repeated
    access to the same trace cannot drift, and so a trace that is unreadable is
    unreadable identically everywhere it appears.
    """

    def __init__(
        self,
        corpus: TraceCorpus,
        *,
        view: TraceView = TraceView.USER_MESSAGES,
        control_markers: Sequence[str] = (),
    ) -> None:
        """``control_markers`` declares literal tokens to strip from message
        text before the miner ever reads it — empty unless the caller knows
        its source writes benchmark scaffolding into a turn's own text (tau2's
        ``###TRANSFER###``, for example). That knowledge belongs to whoever
        constructs this corpus for a specific source, not to this class.
        """
        self._view = view
        self._views: dict[str, UserMessageView] = {
            trace.trace_id: build_view(trace, view, control_markers=control_markers)
            for trace in corpus.traces
        }
        # Source order, not sorted: the corpus order is a fact about the export,
        # and any shuffling this miner does is seeded and recorded separately.
        self._ids: tuple[str, ...] = tuple(self._views)

    @property
    def view(self) -> TraceView:
        return self._view

    def count_traces(self) -> int:
        return len(self._ids)

    def list_trace_ids(self, offset: int = 0, limit: int | None = None) -> tuple[str, ...]:
        if offset < 0:
            raise ValueError("offset must not be negative")
        window = self._ids[offset:]
        return window if limit is None else window[:limit]

    def get_user_messages(self, trace_id: str) -> UserMessageView:
        try:
            return self._views[trace_id]
        except KeyError as exc:
            raise KeyError(f"no trace {trace_id!r} in this corpus") from exc

    def get_user_message_batch(self, trace_ids: Sequence[str]) -> tuple[UserMessageView, ...]:
        """Fetch many views at once, in the order asked for.

        Order is preserved rather than normalized because the caller's order is
        how a chunk was composed — an unseen trace beside a suspected
        counterexample — and re-sorting it here would silently undo that.
        """
        return tuple(self.get_user_messages(trace_id) for trace_id in trace_ids)

    def sample_trace_ids(
        self, count: int, seed: int, exclude: Sequence[str] = ()
    ) -> tuple[str, ...]:
        """A reproducible sample, drawn from a local generator.

        Seeded through its own ``random.Random`` rather than the module-level
        functions: a run that reseeded the global generator would change the
        behaviour of anything else in the process that draws from it, and two
        runs of this miner must be comparable without depending on what else ran.
        """
        pool = [trace_id for trace_id in self._ids if trace_id not in set(exclude)]
        if count >= len(pool):
            return tuple(pool)
        return tuple(random.Random(seed).sample(pool, count))

    def withheld_fields(self) -> tuple[str, ...]:
        """Every outcome-bearing key stripped across the corpus, deduplicated.

        Reported by the miner as a limitation. Under the user-message arms this
        is always empty, which is itself the point: those arms never read a
        payload that could have carried a score.
        """
        return tuple(
            sorted({field for view in self._views.values() for field in view.withheld_fields})
        )

    def leakage_report(self, analysis: Any = None) -> tuple[str, ...]:
        """Outcome values that survived redaction into what the miner will read.

        Empty is the only acceptable result for a Path F run. Takes the
        deterministic analysis because that layer already located every outcome
        this corpus records, by rules that never consulted this denylist; asking
        whether those exact values are still visible is a check the denylist
        cannot pass by being lucky with names.
        """
        if analysis is None or not self._view.reads_agent_behavior:
            return ()
        return audit_view_leakage(self._views, analysis)

    def readable_trace_ids(self) -> tuple[str, ...]:
        return tuple(tid for tid, view in self._views.items() if view.readable)

    def unreadable_trace_ids(self) -> tuple[str, ...]:
        """Traces the miner may not classify, kept visible rather than dropped.

        A trace missing from a taxonomy because nothing could be read off it is
        a fact about the corpus. Dropping it silently would inflate coverage.
        """
        return tuple(tid for tid, view in self._views.items() if not view.readable)


_OUTCOME_CLAIMS = frozenset(
    {"recorded_score", "command_exit_code", "final_state_field", "span_error"}
)
"""Claims from the deterministic analysis that carry outcome information.

Named from :mod:`bandits.analyze.outcomes`, which extracted them by its own
rules. Using that layer's findings rather than this module's denylist is the
point: it is an independent opinion about where the outcomes in this corpus are.
"""


def audit_view_leakage(views: dict[str, UserMessageView], analysis: Any) -> tuple[str, ...]:
    """Report outcome values still visible in the views the miner will read.

    Compares against values, not names. A harness that writes its reward under
    an unforeseen key defeats the denylist and does not defeat this, because the
    deterministic analysis already recorded what that reward *was*.

    Short and boolean values are skipped: a ``0``, a ``1`` or a ``True`` appears
    in ordinary tool output constantly, and flagging those would drown a real
    finding in noise rather than surface it.
    """
    findings: list[str] = []
    for evidence in getattr(analysis, "evidence", ()) or ():
        if evidence.claim not in _OUTCOME_CLAIMS:
            continue
        view = views.get(evidence.trace_id)
        if view is None or not view.readable:
            continue
        # ``final_state_field`` wraps its reading in a descriptor dict; the
        # leak is the inner value, and comparing the wrapper's repr would never
        # match anything the miner actually reads.
        raw = evidence.value
        if isinstance(raw, dict) and "value" in raw:
            raw = raw["value"]
        rendered = str(raw)
        if len(rendered) < 4 or rendered.lower() in ("true", "false", "none"):
            continue
        # A value echoing the trace's own id is an identifier the source threads
        # through its payloads — a confirmation number, a request id — not an
        # outcome. ``outcomes.py`` records every terminal field it finds, so
        # without this the check fires on every trace that returns a receipt and
        # the real findings drown in it.
        if evidence.trace_id in rendered:
            continue
        if any(rendered in message for message in view.messages):
            findings.append(
                f"{evidence.trace_id}: the value of {evidence.claim} is still visible "
                "in the trajectory the miner would read"
            )
    return tuple(sorted(set(findings)))
