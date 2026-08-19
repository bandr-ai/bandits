"""Read / write / external classification -- the decision the rest of the build hangs on.

Each bucket is reimplemented differently when the environment is rebuilt
(PLAN.md, Step 7): a READ is answered from the rebuilt store, a WRITE mutates it
and is what the verifier asserts on, an EXTERNAL is faked and its attempt is
written to the effect ledger. Get this wrong and either the reward function
checks nothing, or the environment "sends" an email.

Everything here is derived from the corpus. No LLM, no network, no name-only
guessing. The four rules, in the order they are applied:

WRITE
    Calling the tool changes what a later read returns for the same entity id.
    Concretely: take an identifier value out of the tool's arguments; find some
    *other* tool R that returned a body containing that same id both **before**
    and **after** this call, inside the same trace; if the two bodies differ, the
    call in between is a write. A weaker secondary signal is an *echo*: the
    tool's own response repeats a field that an earlier read reported with a
    different value (``refund_order`` answers ``status: refunded`` where
    ``get_order`` had said ``delivered``).

EXTERNAL
    All three must hold: (a) no id-linked state change is observable anywhere in
    the corpus, (b) every successful response is *acknowledgement-shaped* -- null,
    empty, or a one-or-two-key object of flags/short strings with no identifier
    and no nested body, and (c) the name contains an irreversible verb
    (send/email/notify/charge/pay/ship/post/publish/delete/escalate).
    **The name is never sufficient on its own.** A ``send_report`` that returns a
    real body with an id is classified READ, and the evidence string says why.

READ
    Returns a real body -- ideally one containing an identifier -- and no call to
    it was ever followed by a changed read.

UNKNOWN
    Zero calls (declared-only), or genuinely ambiguous: ack-shaped responses with
    no verb, or nothing but errors. An UNKNOWN tool must never be probed
    (PLAN.md, Step 9: "you do not find out a tool was ``charge_card`` by charging
    a card").

Failure modes, all real
-----------------------
* **Blind writes.** A write whose entity is never read again in the same trace is
  invisible and comes out READ or UNKNOWN. The fixture's writes are all bracketed
  by reads; production traces frequently are not. This is the single biggest
  weakness of the approach and the reason `class_confidence` exists.
* **Cross-trace only evidence.** Detection is deliberately *within-trace*. Two
  traces are two different points in time and possibly two different worlds;
  attributing a diff across them to a tool call would invent causality.
* **Third-party reads.** A read against someone else's API (a shipping carrier
  lookup) is EXTERNAL in reality and READ here. It is safe -- reads are stubbed
  from the store -- but the manifest will understate what the env fakes.
* **Confounded windows.** If two candidate writes sit between the same before and
  after reads, both are credited. Correct here (both really did write) but it
  would over-credit a read sitting between a write and its follow-up read; the
  ``R != T`` clause plus requiring the *same* R on both sides is what keeps the
  fixture's ``get_order`` from being called a write.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from bandits.contracts import CallStatus, InvocationPoint, JsonValue, ToolClass, TraceCorpus

from .profiling import canonical, flatten, json_type, name_looks_like_identifier

IRREVERSIBLE_VERBS = frozenset(
    {
        "send",
        "email",
        "mail",
        "notify",
        "alert",
        "charge",
        "pay",
        "ship",
        "post",
        "publish",
        "delete",
        "escalate",
        "dispatch",
    }
)
"""Verbs whose real-world effect cannot be undone by resetting a database.

Membership here is *necessary but never sufficient* for EXTERNAL -- see the
module docstring and `classify_tool`.
"""

ACK_KEYS = frozenset(
    {
        "ok",
        "sent",
        "success",
        "succeeded",
        "done",
        "delivered",
        "accepted",
        "queued",
        "received",
        "acknowledged",
        "status",
        "result",
        "message",
        "detail",
    }
)
"""Keys that carry no state. A response made only of these is an acknowledgement."""

ACK_MAX_KEYS = 2
ACK_MAX_STRING = 32

CONF_WRITE_STATE_CHANGE = 0.95
CONF_WRITE_ECHO_ONLY = 0.6
CONF_EXTERNAL = 0.75
CONF_READ_WITH_ID = 0.85
CONF_READ_PLAIN = 0.7
CONF_UNKNOWN = 0.0


@dataclass(frozen=True)
class Classification:
    """One tool's verdict, its confidence, and the reasons a human will read."""

    tool_class: ToolClass
    confidence: float
    evidence: tuple[str, ...] = ()


@dataclass
class _ToolEvidence:
    """Raw signals gathered for one tool before any verdict is taken."""

    calls: int = 0
    ok_calls: int = 0
    state_changes: list[str] = field(default_factory=list)
    echoes: list[str] = field(default_factory=list)
    windows_checked: int = 0
    ack_responses: int = 0
    body_responses: int = 0
    body_with_id: int = 0
    verbs: tuple[str, ...] = ()


# ---------------------------------------------------------------- shape tests


def is_acknowledgement(response: JsonValue) -> bool:
    """True when a response carries a receipt and no state.

    ``None``, ``{}``, ``[]``, ``{"sent": true}``, ``{"ok": true, "status": "queued"}``
    are acknowledgements. ``{"policy": "Refunds accepted within..."}`` is not --
    ``policy`` is not an ack key and the value is a real body. ``{"order_id":
    7741, "status": "refunded"}`` is not -- it carries an identifier.

    Failure mode: a backend that answers writes with ``{"status": "ok"}`` and
    reads with ``{"status": "delivered"}`` is indistinguishable at this level;
    both look like acks. That is why an ack alone never decides a class.
    """
    if response is None:
        return True
    if isinstance(response, (list, tuple)):
        return len(response) == 0
    if not isinstance(response, Mapping):
        # A bare scalar body (a policy string, a count) is a body, not a receipt.
        return False
    if not response:
        return True
    if len(response) > ACK_MAX_KEYS:
        return False
    for key, value in response.items():
        if str(key).lower() not in ACK_KEYS:
            return False
        if name_looks_like_identifier(str(key)):
            return False
        if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
            return False
        if isinstance(value, str) and len(value) > ACK_MAX_STRING:
            return False
    return True


def split_name(name: str) -> tuple[str, ...]:
    """Tokenize a tool name: snake_case, kebab-case, dots and camelCase all split."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return tuple(t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t)


def irreversible_verbs(name: str) -> tuple[str, ...]:
    """Irreversible verbs present in a tool name, in name order."""
    return tuple(t for t in split_name(name) if t in IRREVERSIBLE_VERBS)


# ---------------------------------------------------------------- id linking


def identifier_values(payload: JsonValue) -> dict[str, JsonValue]:
    """Scalar identifier-looking values in a payload, keyed by their dotted path.

    Only the *name* rule is used here (``order_id``, ``sku``, ``ids[]``...):
    linking a write to a read has to be conservative, and a value-recurrence
    guess would let a shared status string bridge two unrelated calls.
    """
    found: dict[str, JsonValue] = {}
    for path, value in flatten(payload):
        if json_type(value) not in ("string", "integer"):
            continue
        if name_looks_like_identifier(path):
            found[path] = value
    return found


def contains_value(payload: JsonValue, value: JsonValue) -> bool:
    """True when `value` appears anywhere in `payload`, at any depth, same type."""
    target = (json_type(value), canonical(value))
    for _path, seen in flatten(payload):
        if (json_type(seen), canonical(seen)) == target:
            return True
    return False


def body_diff(before: JsonValue, after: JsonValue) -> list[tuple[str, JsonValue, JsonValue]]:
    """Fields present in both bodies whose value changed, sorted by path."""
    b = dict(flatten(before))
    a = dict(flatten(after))
    return [
        (path, b[path], a[path])
        for path in sorted(set(b) & set(a))
        if canonical(b[path]) != canonical(a[path])
    ]


# ---------------------------------------------------------------- evidence pass


def _gather(corpus: TraceCorpus, tools: Iterable[str]) -> dict[str, _ToolEvidence]:
    evidence = {name: _ToolEvidence(verbs=irreversible_verbs(name)) for name in tools}

    for trace in corpus.traces:
        ordered: list[InvocationPoint] = sorted(trace.invocations, key=lambda i: i.step)
        for inv in ordered:
            ev = evidence.setdefault(inv.tool, _ToolEvidence(verbs=irreversible_verbs(inv.tool)))
            ev.calls += 1
            if inv.status is not CallStatus.OK:
                continue
            ev.ok_calls += 1
            if is_acknowledgement(inv.response):
                ev.ack_responses += 1
            else:
                ev.body_responses += 1
                if identifier_values(inv.response):
                    ev.body_with_id += 1

        for inv in ordered:
            if inv.status is not CallStatus.OK:
                continue
            ev = evidence[inv.tool]
            ids = identifier_values(inv.arguments)
            for arg_path, id_value in sorted(ids.items()):
                _check_state_change(ordered, inv, arg_path, id_value, ev, trace.trace_id)
                _check_echo(ordered, inv, arg_path, id_value, ev, trace.trace_id)
    return evidence


def _readers_for(
    ordered: Sequence[InvocationPoint], id_value: JsonValue, exclude_tool: str
) -> dict[str, list[InvocationPoint]]:
    """Other tools' successful calls whose *response* mentions `id_value`."""
    readers: dict[str, list[InvocationPoint]] = {}
    for other in ordered:
        if other.tool == exclude_tool or other.status is not CallStatus.OK:
            continue
        if not isinstance(other.response, Mapping):
            continue
        if contains_value(other.response, id_value):
            readers.setdefault(other.tool, []).append(other)
    return readers


def _check_state_change(
    ordered: Sequence[InvocationPoint],
    inv: InvocationPoint,
    arg_path: str,
    id_value: JsonValue,
    ev: _ToolEvidence,
    trace_id: str,
) -> None:
    """The primary WRITE test: same reader, same id, different body across the call."""
    for reader, calls in sorted(_readers_for(ordered, id_value, inv.tool).items()):
        before = [c for c in calls if c.step < inv.step]
        after = [c for c in calls if c.step > inv.step]
        if not before or not after:
            continue
        ev.windows_checked += 1
        last_before, first_after = before[-1], after[0]
        diff = body_diff(last_before.response, first_after.response)
        if not diff:
            continue
        changed = "; ".join(
            f"{path}: {canonical(old)} -> {canonical(new)}" for path, old, new in diff[:3]
        )
        ev.state_changes.append(
            f"observed state change: {inv.tool}({arg_path}={canonical(id_value)}) at step "
            f"{inv.step} of trace {trace_id}; {reader} returned a different body for that id "
            f"before (step {last_before.step}) and after (step {first_after.step}) the call "
            f"[{changed}]"
        )


def _check_echo(
    ordered: Sequence[InvocationPoint],
    inv: InvocationPoint,
    arg_path: str,
    id_value: JsonValue,
    ev: _ToolEvidence,
    trace_id: str,
) -> None:
    """Secondary WRITE test: the tool's own response contradicts an earlier read."""
    if not isinstance(inv.response, Mapping):
        return
    for reader, calls in sorted(_readers_for(ordered, id_value, inv.tool).items()):
        before = [c for c in calls if c.step < inv.step]
        if not before:
            continue
        diff = body_diff(before[-1].response, inv.response)
        if not diff:
            continue
        changed = "; ".join(
            f"{path}: {canonical(old)} -> {canonical(new)}" for path, old, new in diff[:3]
        )
        ev.echoes.append(
            f"response echoes a changed field: {inv.tool} answered for "
            f"{arg_path}={canonical(id_value)} with values that differ from what {reader} "
            f"reported earlier in trace {trace_id} [{changed}]"
        )


# ---------------------------------------------------------------- the verdict


def classify_tool(name: str, ev: _ToolEvidence) -> Classification:
    """Apply the four rules to one tool's gathered evidence. See module docstring."""
    verbs = ev.verbs or irreversible_verbs(name)
    verb_note = (
        f"name contains irreversible verb(s) {', '.join(sorted(set(verbs)))}"
        if verbs
        else "name contains no irreversible verb"
    )

    if ev.calls == 0:
        return Classification(
            ToolClass.UNKNOWN,
            CONF_UNKNOWN,
            (
                "declared in the registry but never called in the corpus: zero evidence",
                f"{verb_note}, which is NOT enough to assign a class -- a name is not behaviour",
                "must never be probed to find out (PLAN.md Step 9)",
            ),
        )

    if ev.state_changes:
        reasons = tuple(ev.state_changes[:3])
        if ev.echoes:
            reasons += (ev.echoes[0],)
        return Classification(
            ToolClass.WRITE,
            CONF_WRITE_STATE_CHANGE,
            reasons + (f"{len(ev.state_changes)} such before/after window(s) observed",),
        )

    if ev.echoes:
        return Classification(
            ToolClass.WRITE,
            CONF_WRITE_ECHO_ONLY,
            tuple(ev.echoes[:3])
            + (
                "no read of the same id was available after the call, so the change was "
                "inferred from the tool's own response only -- lower confidence",
            ),
        )

    all_acks = ev.ok_calls > 0 and ev.body_responses == 0

    if all_acks and verbs:
        return Classification(
            ToolClass.EXTERNAL,
            CONF_EXTERNAL,
            (
                f"all {ev.ok_calls} successful response(s) are acknowledgement-shaped "
                "(no identifier, no body -- nothing a rebuilt store could have returned)",
                (
                    f"no id-linked state change was observable in {ev.windows_checked} "
                    "before/after read window(s) across the corpus"
                    if ev.windows_checked
                    else "no id-linked state change is observable anywhere in the corpus "
                    "(no before/after read window was available either)"
                ),
                f"{verb_note}; the verb only decided the class because the "
                "acknowledgement-shaped response was there too -- the name alone is never "
                "sufficient",
                "will be stubbed and its attempt written to the effect ledger, never performed",
            ),
        )

    if all_acks:
        return Classification(
            ToolClass.UNKNOWN,
            CONF_UNKNOWN,
            (
                f"all {ev.ok_calls} successful response(s) are acknowledgement-shaped, so "
                "nothing can be answered from a rebuilt store",
                "no id-linked state change was observed either, so it cannot be called a write",
                f"{verb_note}, so there is no reason to call it external",
                "ambiguous: needs a human or an authorized probe (PLAN.md Step 9)",
            ),
        )

    if ev.body_responses:
        confidence = CONF_READ_WITH_ID if ev.body_with_id else CONF_READ_PLAIN
        reasons = [
            f"{ev.body_responses} of {ev.ok_calls} successful call(s) returned a real body"
            + (
                f", {ev.body_with_id} of them containing an identifier"
                if ev.body_with_id
                else " with no identifier in it (still readable, but weaker evidence)"
            ),
            (
                f"no call was ever followed by a changed read of the same id "
                f"({ev.windows_checked} before/after window(s) checked)"
                if ev.windows_checked
                else "no before/after read window was available to test whether it changes "
                "state, so 'never writes' is unfalsified rather than proven"
            ),
        ]
        if verbs:
            reasons.append(
                f"{verb_note}, but the response is a full body rather than an acknowledgement, "
                "so the name did NOT make this external"
            )
        return Classification(ToolClass.READ, confidence, tuple(reasons))

    return Classification(
        ToolClass.UNKNOWN,
        CONF_UNKNOWN,
        (
            f"called {ev.calls} time(s) but never returned a successful body "
            f"({ev.calls - ev.ok_calls} error(s))",
            "no state change and no readable response: not enough evidence for any class",
            "must never be probed to find out (PLAN.md Step 9)",
        ),
    )


def classify_tools(corpus: TraceCorpus, tools: Iterable[str]) -> dict[str, Classification]:
    """Classify every tool in `tools` plus anything observed in `corpus`."""
    evidence = _gather(corpus, tools)
    return {name: classify_tool(name, ev) for name, ev in sorted(evidence.items())}
