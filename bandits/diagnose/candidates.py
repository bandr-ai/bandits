"""Adapt a model into something the rollout loop can call.

This module owns exactly two things: how a candidate is shown its situation,
and how its reply becomes an action. It owns no state, no scoring and no
retrieval, and it never sees a ``Scenario`` — only the ``CandidateView`` that
scenario chose to expose.

Repeated sampling and pass@k live in ``report.py`` rather than here. A candidate
that knew it was being sampled could behave differently across attempts, and the
independence pass@k assumes would stop holding for a reason nothing recorded.

The controls matter as much as the real adapter. A lab that cannot rank a strong
model above a deliberately broken one has not demonstrated that it ranks
anything, so ``giving_up_candidate`` and ``looping_candidate`` exist to fail in
known ways and are part of the measurement, not test scaffolding.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from bandits.diagnose.models import ActionCall, CandidateView
from bandits.diagnose.rollout import CandidateAction
from bandits.traces import Contract

PROMPT_VERSION = 1

CANDIDATE_INSTRUCTION = """\
You are an agent completing one task for a user.

- Use the tools you were offered. Call one when you need information or need to
  change something; do not describe a call you did not make.
- Say what you are doing when it affects the user, and ask when a decision is
  theirs to make.
- Follow the operating policy you were given. If it forbids an action, refuse
  and explain why rather than doing it anyway.
- Stop when the task is done or cannot be done, and say which.

Reply with either a tool call or a message, not both unless the message
explains the call."""

MAX_HISTORY_CHARS = 6000


class CandidateSpec(Contract):
    """What produced a set of rollouts, pinned so a report can name it.

    A capability number is about one model under one decoding setting reading
    one prompt. Two runs that differ in any of those are two measurements, and
    a report that cannot tell them apart is comparing populations rather than
    candidates.
    """

    candidate_id: str
    model: str = ""
    endpoint: str = ""
    temperature: float = 0.0
    max_tokens: int = 2000
    seed: int = 0
    prompt_version: int = PROMPT_VERSION
    instruction: str = CANDIDATE_INSTRUCTION
    kind: str = "endpoint"
    """``endpoint``, ``scripted``, or the name of a control."""

    notes: str = ""

    @property
    def prompt_digest(self) -> str:
        payload = json.dumps(
            {
                "instruction": self.instruction,
                "model": self.model,
                "endpoint": self.endpoint,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "version": self.prompt_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def render_tools(view: CandidateView) -> str:
    """The toolset as the candidate sees it.

    Which tool to reach for, out of what was on offer, is most of the decision
    being measured — so the schemas are rendered in full rather than as a list
    of names. An empty toolset is stated rather than omitted: a candidate given
    no tools should know that, not infer it from silence.
    """
    if not view.offered_tools:
        return "(no tools are available; you can only reply to the user)"
    lines = []
    for schema in view.offered_tools:
        name = schema.get("name") or (schema.get("function") or {}).get("name") or "?"
        parameters = schema.get("parameters") or (schema.get("function") or {}).get("parameters")
        described = json.dumps(parameters, sort_keys=True) if parameters else "{}"
        description = schema.get("description") or ""
        lines.append(f"- {name}{': ' + description if description else ''}\n    {described}")
    return "\n".join(lines)


def render_history(history: Sequence[dict[str, Any]], limit: int = MAX_HISTORY_CHARS) -> str:
    """The conversation so far, clipped from the front.

    Clipping belongs here, in rendering, for the same reason it does not belong
    in the grounding index: what a prompt can afford is a property of the
    prompt, and the record it draws from stays whole. The tail is kept because
    the most recent observation is what the next action answers.
    """
    rows = []
    for entry in history:
        role = entry.get("role", "?")
        tool = entry.get("tool")
        content = entry.get("content")
        rendered = content if isinstance(content, str) else json.dumps(content, default=str)
        label = f"{role}:{tool}" if tool else role
        rows.append(f"{label}: {rendered}")
    text = "\n".join(rows)
    if len(text) > limit:
        return f"[earlier turns omitted]\n{text[-limit:]}"
    return text or "(the conversation has not started)"


def render_candidate_prompt(
    view: CandidateView,
    history: Sequence[dict[str, Any]],
    *,
    instruction: str = CANDIDATE_INSTRUCTION,
) -> str:
    """Everything the candidate is allowed to know, and nothing else.

    Built only from ``CandidateView``, which is the type that exists so a
    private field cannot reach a candidate by being added to ``Scenario``.
    """
    return "\n\n".join(
        (
            instruction,
            f"OPERATING POLICY:\n{view.system_policy}"
            if view.system_policy
            else "OPERATING POLICY:\n(none was given)",
            f"TASK:\n{view.task}",
            f"TOOLS:\n{render_tools(view)}",
            f"CONVERSATION:\n{render_history(history)}",
        )
    )


def offered_tool_names(view: CandidateView) -> frozenset[str]:
    names = set()
    for schema in view.offered_tools:
        name = schema.get("name") or (schema.get("function") or {}).get("name")
        if name:
            names.add(str(name))
    return frozenset(names)


def parse_action(raw: Any, *, offered: frozenset[str] = frozenset()) -> CandidateAction:
    """Read a model reply into an action.

    A call naming a tool that was never offered is kept as a real action rather
    than dropped. Dropping it would turn a candidate's mistake into silence:
    the environment would see a plain message, the rollout would continue, and
    the report would never record that the candidate reached for something it
    did not have. Kept, it reaches the tool world, finds no grounding, and
    abstains — which is the honest outcome and a countable one.

    ``offered`` is accepted so a caller can mark the action, not so this
    function can censor it.
    """
    if isinstance(raw, CandidateAction):
        return raw

    payload: Any = raw
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    elif isinstance(payload, str):
        text = payload.strip()
        if text[:1] in ("{", "["):
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                return CandidateAction(content=raw)
        else:
            return CandidateAction(content=raw)

    if not isinstance(payload, dict):
        return CandidateAction(content=str(raw))

    calls: list[ActionCall] = []
    raw_calls = payload.get("tool_calls") or payload.get("calls") or []
    if isinstance(raw_calls, dict):
        raw_calls = [raw_calls]
    for index, entry in enumerate(raw_calls if isinstance(raw_calls, list) else []):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = function.get("name") or entry.get("tool")
        if not name:
            continue
        arguments = function.get("arguments", entry.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {"_unparsed": arguments}
        calls.append(
            ActionCall(
                call_id=str(entry.get("id") or f"call-{index}"),
                tool=str(name),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )

    return CandidateAction(
        calls=tuple(calls),
        content=payload.get("content") or payload.get("message"),
        done=bool(payload.get("done") or payload.get("finished")),
    )


def unoffered_calls(action: CandidateAction, offered: frozenset[str]) -> tuple[str, ...]:
    """Tools this action reached for that were never on offer.

    Reported rather than blocked. A candidate inventing a tool is a capability
    finding, and a lab that silently repaired it would be measuring its own
    repair.
    """
    return tuple(call.tool for call in action.calls if call.tool not in offered)


def scripted_candidate(actions: Sequence[CandidateAction]) -> Any:
    """A candidate that replays a fixed list, then finishes.

    Deterministic by construction, which is what makes it usable both in tests
    and as the trivial control in a real comparison.
    """
    remaining = list(actions)

    def candidate(*, view: CandidateView, history: Sequence[dict[str, Any]]) -> CandidateAction:
        if remaining:
            return remaining.pop(0)
        return CandidateAction(done=True)

    return candidate


def giving_up_candidate(message: str = "I cannot help with that.") -> Any:
    """A control that finishes immediately without doing anything.

    Its verdict should be a clean failure on every mutation task and a pass on
    nothing. A lab that scores it above zero on those is reporting something
    other than capability.
    """

    def candidate(*, view: CandidateView, history: Sequence[dict[str, Any]]) -> CandidateAction:
        return CandidateAction(done=True, content=message)

    return candidate


def looping_candidate(call: ActionCall) -> Any:
    """A control that repeats one action forever.

    Terminates as ``ACTION_LOOP``, which is a candidate failure and not an
    abstention: the environment answered every time. If this one leaves the
    denominator, the abstention accounting is wrong.
    """

    def candidate(*, view: CandidateView, history: Sequence[dict[str, Any]]) -> CandidateAction:
        return CandidateAction(calls=(call,))

    return candidate


def build_endpoint_candidate(spec: CandidateSpec, *, api_key: str | None = None) -> Any:
    """A real model behind the candidate protocol.

    The model stack is imported here and nowhere else, so the tests above run
    against injected fakes and CI needs neither a credential nor a sandbox —
    the idiom the rest of this repository already uses for every model call.
    """
    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise RuntimeError(
            "running a model candidate needs the 'diagnose' extra: uv sync --extra diagnose"
        ) from exc

    from bandits.verify.judge import resolve_api_key

    language_model = dspy.LM(
        spec.endpoint or f"fireworks_ai/{spec.model}",
        api_key=api_key or resolve_api_key(),
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        seed=spec.seed,
    )

    class _Act(dspy.Signature):
        situation: str = dspy.InputField(desc="the task, your tools, and the conversation")
        tool_calls: list[dict] = dspy.OutputField(desc="calls to make now, possibly empty")
        content: str = dspy.OutputField(desc="what to say to the user, possibly empty")
        done: bool = dspy.OutputField(desc="true when the task is finished or impossible")

    _Act.__doc__ = spec.instruction

    def candidate(*, view: CandidateView, history: Sequence[dict[str, Any]]) -> CandidateAction:
        prompt = render_candidate_prompt(view, history, instruction=spec.instruction)
        with dspy.context(lm=language_model):
            reply = dspy.Predict(_Act)(situation=prompt)
        return parse_action(reply, offered=offered_tool_names(view))

    return candidate
