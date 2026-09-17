"""Call a model to grade or judge something, and flatten a trace to show it.

Shared plumbing for every model-graded step in this package: the next-state
judge (``nextstate.py``), the direct-SFT reviewer (``export/direct_sft.py``),
and the interview interpreter each need to call a model and get text back.
This is the one place that call is made, so retries, the API key lookup, and
the request ledger entry are made once rather than once per caller.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from bandits import ledger
from bandits.traces import SpanKind, SpanStatus, Trace
from bandits.transport import request_with_retry

DEFAULT_MODEL = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"

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
    """The Fireworks key, read per call so it is never held in an artifact.

    Shared with the family audit, which reaches the same backend through a
    different client: two lookups would drift and one of them would start
    reporting a missing key that is plainly there.
    """
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        # Keep local dogfooding one command wide without executing arbitrary
        # shell from .env. Only the one key this backend owns is read.
        env_path = Path(".env")
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "FIREWORKS_API_KEY":
                    api_key = value.strip().strip("'\"")
                    break
    if not api_key:
        raise JudgeError("FIREWORKS_API_KEY is not set and was not found in .env")
    return api_key


def fireworks_completion(
    model: str,
    prompt: str,
    temperature: float,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Call Fireworks with a prompt and an optional higher-priority policy.

    ``max_tokens`` is the whole budget, reasoning included. The next-state
    judge, which is told to think first, ran out of it 29 times in 436 and
    lost the boxed score each time.
    """
    api_key = resolve_api_key()

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request = urllib.request.Request(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "temperature": temperature,
                # Reasoning models may spend a substantial part of this budget
                # before emitting their short visible answer. Seven hundred
                # truncated real structured SFT reviews halfway through JSON.
                "max_tokens": max_tokens,
                "messages": messages,
            }
        ).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )

    def send() -> object:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.load(response)

    with ledger.model_call(
        provider="fireworks",
        model=model,
        request={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        },
    ) as call:
        try:
            payload = request_with_retry(send)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JudgeError(f"judge request failed: {exc}") from exc
        # The whole body, not the text read out of it: `usage` is the only
        # record of what the call cost, and it was discarded one line later.
        call["response"] = payload
    return payload["choices"][0]["message"]["content"]
