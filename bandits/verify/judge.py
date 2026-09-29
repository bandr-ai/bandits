"""Call a model to grade or judge something, and flatten a trace to show it.

Shared plumbing for every model-graded step in this package: the next-state
judge (``nextstate.py``), the direct-SFT reviewer (``export/direct_sft.py``),
and the interview interpreter each need to call a model and get text back.
This is the one place that call is made, so retries, the API key lookup, and
the request ledger entry are made once rather than once per caller.
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Callable, Mapping
from typing import Any

from bandits import ledger, providers
from bandits.traces import SpanKind, SpanStatus, Trace
from bandits.transport import request_with_retry

DEFAULT_MODEL = providers.DEFAULT_MODEL

_MAX_OUTPUT_CHARS = 400


class JudgeError(RuntimeError):
    """The judge could not be reached or returned nothing usable."""


def render_transcript(trace: Trace) -> str:
    """Flatten one episode into the text a judge grades.

    Only what the trace recorded. No label, score, or verifier result is
    rendered: a judge shown the answer is measuring nothing.
    """
    lines = [f"Task: {trace.task or '(no instruction recorded)'}"]
    actions = [
        f"  - {span.name}({json.dumps(span.arguments, default=str)[:200]})"
        f" -> {json.dumps(span.output, default=str)[:200]}"
        # A failed call whose result was simply absent renders as "-> null",
        # which reads as an empty success. The status is the whole difference.
        f"{' [TOOL REPORTED AN ERROR]' if span.status is SpanStatus.ERROR else ''}"
        for span in trace.spans
        if span.kind is SpanKind.TOOL
    ]
    lines.append("Agent actions:")
    lines.extend(actions or ["  (none - no tool was called)"])

    final = next(
        (
            span.output
            for span in reversed(trace.spans)
            if span.kind is SpanKind.MODEL and span.output
        ),
        None,
    )
    lines.append(
        f'Final message: "{str(final)[:_MAX_OUTPUT_CHARS]}"' if final else "Final message: (none)"
    )
    return "\n".join(lines)


Completion = Callable[[str, str, float], str]
"""(model, prompt, temperature) -> the model's reply text."""


def resolve_api_key() -> str:
    """The Fireworks key. Kept for callers outside this package: a model's own
    credentials now come from ``providers.credentials``."""
    api_key = providers.env_value("FIREWORKS_API_KEY")
    if not api_key:
        raise JudgeError("FIREWORKS_API_KEY is not set and was not found in .env")
    return api_key


def complete(
    model: str,
    prompt: str,
    temperature: float,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 2000,
    timeout: float = 90,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Call ``model`` on any LiteLLM provider with a prompt and an optional
    higher-priority policy.

    ``max_tokens`` is the whole budget, reasoning included. The next-state
    judge, which is told to think first, ran out of it 29 times in 436 and
    lost the boxed score each time.

    ``timeout`` is per attempt, and a timed-out attempt is retried from
    scratch: under load a full-budget reply queued for minutes, so a 90 s
    timeout abandoned and resent it up to five times. ``extra`` adds request
    fields (a sampling penalty, say) and is recorded with the call.

    Retries are ours, not LiteLLM's (``num_retries`` and the OpenAI client's
    ``max_retries`` are both 0), so each failed attempt still reaches the
    ledger as a ``retry`` event. A field in ``extra`` that LiteLLM does not map
    for this provider is sent as-is in the body rather than dropped: Fireworks
    takes ``repetition_penalty`` and ``reasoning_effort`` that way, and a
    provider that rejects one fails that call visibly instead of silently
    sending a different request than the one recorded.
    """
    try:
        litellm = providers.load_litellm()
        ref = providers.resolve(model)
        reach = providers.credentials(ref)
    except providers.ProviderError as exc:
        raise JudgeError(str(exc)) from exc
    import openai

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    fields = dict(extra or {})
    mapped = set(
        litellm.get_supported_openai_params(model=ref.litellm_id, custom_llm_provider=ref.provider)
        or ()
    )
    named = {key: value for key, value in fields.items() if key in mapped}
    raw = {key: value for key, value in fields.items() if key not in mapped}

    def send() -> object:
        response = litellm.completion(
            model=ref.litellm_id,
            messages=messages,
            temperature=temperature,
            # Reasoning models may spend a substantial part of this budget
            # before emitting their short visible answer. Seven hundred
            # truncated real structured SFT reviews halfway through JSON.
            max_tokens=max_tokens,
            timeout=timeout,
            num_retries=0,
            max_retries=0,
            **reach,
            **named,
            **({"extra_body": raw} if raw else {}),
        )
        return response.model_dump(mode="json")

    with ledger.model_call(
        provider=ref.provider,
        model=model,
        request={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            **fields,
        },
    ) as call:
        try:
            payload = request_with_retry(send)
        except (openai.APIError, urllib.error.URLError, TimeoutError) as exc:
            raise JudgeError(f"judge request failed: {exc}") from exc
        # The whole body, not the text read out of it: `usage` is the only
        # record of what the call cost, and it was discarded one line later.
        call["response"] = payload
    return payload["choices"][0]["message"]["content"] or ""


fireworks_completion = complete
"""The name this had while Fireworks was the only backend. Deprecated."""
