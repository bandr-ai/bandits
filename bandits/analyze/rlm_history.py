"""Attribute DSPy's shared call history to the prediction that caused it.

Every RLM stage reaches the provider through DSPy rather than through
``transport.request_with_retry``, so ``ledger.model_call`` never sees those
requests and a stage would otherwise record one event for what is really a
dozen. This module is the seam where those calls become visible: it marks the
history before a prediction, attributes only the entries added after that mark,
and writes each one to the ledger as the physical call it was.

Nothing here knows what a family, a contract or a taxonomy is. It wraps a
callable and reads a language model's history, which is why every stage can
share it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from bandits import ledger


class Predictor(Protocol):
    """Any RLM stage's predictor. Stages narrow this to their own arguments.

    Deliberately permissive in its arguments: mining passes ``chunk`` and
    ``taxonomy``, audit passes five, and a wrapper that named any one stage's
    parameters would raise an unexpected-keyword error for all the others.
    """

    def __call__(self, **inputs: Any) -> Any: ...


class _Spend:
    """The calls the last prediction added to a shared history, and their cost."""

    def __init__(self) -> None:
        self.entries: list[Any] = []

    def __call__(self) -> tuple[int | None, dict[str, int]]:
        return summarize_history(self.entries)


def scoped_to_history(predict: Predictor, language_model: Any) -> Predictor:
    """Wrap a predictor so it reports only the calls *it* made.

    ``lm.history`` is one mutable list the language model appends to for the
    life of the process, shared across every prediction in a run. Reading it
    whole would charge each prediction for all its predecessors, so the length
    is marked before the call and only the entries added after that point are
    attributed to it.

    The slice is copied immediately rather than held as a reference, because
    the list keeps growing: a reference read after the next prediction started
    would describe that prediction's calls too.

    Takes whatever arguments the predictor it wraps takes. It used to name the
    family audit's two, which silently made it unusable by every other RLM
    stage: mining passes ``chunk`` and ``taxonomy``, assignment passes
    ``batch``, and each raised an unexpected-keyword error on its first real
    call. Nothing caught it, because the tests inject plain functions and never
    reach this wrapper -- so the failure only appeared against a live model.
    """
    spend = _Spend()

    def wrapped(**inputs: Any) -> Any:
        before = len(getattr(language_model, "history", ()) or ())
        try:
            return predict(**inputs)
        finally:
            # In `finally` because a failed prediction still spent calls, and
            # those are exactly the ones a rerun that behaved differently needs.
            history = getattr(language_model, "history", None)
            spend.entries = list(history[before:]) if isinstance(history, list) else []
            record_history(spend.entries, language_model=language_model)

    wrapped.spend = spend  # type: ignore[attr-defined]
    return wrapped


_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def record_history(entries: Sequence[Any], *, language_model: Any = None) -> None:
    """Write each RLM subcall to the ledger as the physical call it was.

    Verified against a real run: each entry carries the messages sent, the text
    returned, the code the root model wrote, per-call token usage and the
    provider's own cost figure.

    What is still not visible here: the REPL's stdout appears only inside the
    *next* entry's prompt, as the ``repl_history`` the model is shown, so a
    final iteration's output is unrecoverable from history alone. Retries
    inside litellm are still invisible -- a call that failed and was retried
    below this layer appears as one entry, or as none. A ChatAdapter parse
    failure is not one of those: DSPy's fallback to JSONAdapter issues its own
    real request, which lands here as its own entry, indistinguishable from a
    genuine next RLM iteration without comparing response shapes by hand.
    """
    if not ledger.enabled():
        return
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        outputs = entry.get("outputs") or []
        text = ""
        if outputs:
            first = outputs[0]
            text = first.get("text", "") if isinstance(first, dict) else str(first)
        code = _CODE_BLOCK.findall(text)
        # entry["response"] is the raw litellm response object DSPy kept, not
        # a ledger reconstruction — finish_reason is exactly what DSPy itself
        # reads to decide whether a completion was truncated (lm.py's own
        # ``_check_truncation``), just never previously surfaced past that
        # warning log line.
        finish_reason = None
        response_obj = entry.get("response")
        choices = getattr(response_obj, "choices", None)
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
        # An allowlist, not every non-api_ kwarg: DSPy's kwargs dict is not a
        # contract, and a field added there tomorrow should not silently start
        # appearing in every ledger row until someone notices.
        #
        # temperature and max_tokens are set once on the LM instance
        # (``dspy.LM(..., max_tokens=8192)``), not re-passed on every call, so
        # they never appear in a single call's own kwargs — only in
        # ``language_model.kwargs``, the LM's baked-in defaults. A prior
        # version of this function read only the per-call dict and always
        # logged ``{}`` for a setting that was, in fact, in effect on every
        # request. The per-call dict is read second so a genuine per-call
        # override — an adapter deliberately changing max_tokens for one
        # request — still wins over the instance default.
        call_kwargs = entry.get("kwargs") or {}
        lm_kwargs = getattr(language_model, "kwargs", None) or {}
        request_kwargs = {
            key: value
            for source in (lm_kwargs, call_kwargs)
            for key, value in source.items()
            if key in ("temperature", "max_tokens")
        }
        # ChatAdapter never sets response_format; JSONAdapter always does, on
        # both its direct use and its fallback path after a ChatAdapter parse
        # failure. That fallback issues its own real request, which otherwise
        # lands here indistinguishable from a genuine next RLM iteration.
        adapter = "json" if "response_format" in call_kwargs else "chat"
        ledger.record(
            {
                "event_type": "model_call",
                "provider": "dspy",
                "model": entry.get("model"),
                "iteration": index + 1,
                "adapter": adapter,
                "request": {"messages": entry.get("messages"), "kwargs": request_kwargs},
                "response": {"text": text, "finish_reason": finish_reason},
                # Pulled out of the reply rather than left inside it: the code
                # is what the root model actually did, and grepping a ledger
                # for it should not mean parsing markdown fences back out.
                "generated_code": code,
                "usage": entry.get("usage"),
                "cost_usd": entry.get("cost"),
                "provider_request_id": entry.get("uuid"),
                "timestamp": entry.get("timestamp"),
                "status": "success",
            }
        )


_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
"""What a provider reports. Never derived: a total computed here from a sum
that is missing a call reads as authoritative and is not."""


def summarize_history(entries: Sequence[Any]) -> tuple[int | None, dict[str, int]]:
    """How many DSPy history entries those are, and what they reported.

    One entry is one request DSPy issued, verified against a real run. It is
    not necessarily one *physical* HTTP request: litellm retries below this
    layer, so a call that failed and succeeded on retry appears once, and a
    call that failed permanently may not appear at all. The count is therefore
    a floor on what was spent, which is the honest reading of it.

    Tokens are summed only over the entries that actually reported them. A call
    whose backend said nothing contributes nothing rather than a zero, because
    a zero would silently understate the bill; when no call reported at all the
    result is empty, which reads as unknown.
    """
    totals: dict[str, int] = {}
    for entry in entries:
        usage = entry.get("usage") if isinstance(entry, dict) else getattr(entry, "usage", None)
        if not isinstance(usage, dict):
            continue
        for field in _USAGE_FIELDS:
            value = usage.get(field)
            if isinstance(value, int):
                totals[field] = totals.get(field, 0) + value
    return len(entries), totals
