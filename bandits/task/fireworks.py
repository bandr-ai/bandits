"""Fireworks adapter for reviewed task clarification only.

It can propose a task draft; it cannot change state, invent reward, or score a
rollout. Credentials come only from the process environment and are never saved.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bandits.task.enrich import TaskDraft, TaskEnricher, TaskEnrichmentRequest

DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"
_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
_MAX_MESSAGE_CHARS = 1_200


class FireworksError(RuntimeError):
    """A failed request, with no credentials included in the error."""


def _prompt_evidence(task: TaskEnrichmentRequest) -> dict[str, Any]:
    """Keep task drafting cheap and prevent tool payloads becoming model context.

    Tool results are state evidence for deterministic reconstruction, not task
    wording evidence.  The model sees their presence but never their payload.
    """
    evidence = task.model_dump(mode="json")
    messages: list[dict[str, Any]] = []
    for message in evidence["messages"]:
        compact = dict(message)
        content = compact.get("content")
        if compact.get("role") == "tool":
            compact["content"] = "[tool result omitted from task drafting]"
        elif isinstance(content, str) and len(content) > _MAX_MESSAGE_CHARS:
            compact["content"] = content[:_MAX_MESSAGE_CHARS] + " [truncated]"
        messages.append(compact)
    evidence["messages"] = messages
    return evidence


def fireworks_task_enricher(
    *, api_key: str | None = None, model: str = DEFAULT_MODEL, timeout_s: float = 45.0,
    transport: Callable[[Request, float], bytes] | None = None,
) -> TaskEnricher:
    """Return a Fireworks-backed, schema-constrained task-draft callable."""
    key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise FireworksError("FIREWORKS_API_KEY is required; it is not read from project files")

    def enrich(task: TaskEnrichmentRequest) -> TaskDraft:
        prompt = (
            "Draft a concise reviewer-facing task from this trace evidence. Do not invent "
            "facts. Do not propose state, a tool sequence, a reward, or pass/fail. Put any "
            "uncertainty in ambiguities. Return at most 3 ambiguities and cite at most 5 "
            "supporting message indexes. Keep the instruction under 240 characters and the "
            "rationale under 500 characters.\n\n"
            + json.dumps(_prompt_evidence(task), sort_keys=True)
        )
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON matching the schema."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 700,
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "task_draft", "strict": True, "schema": TaskDraft.model_json_schema(),
            }},
        }
        for attempt in range(2):
            if attempt:
                # A response ending mid-JSON is normally a length cap, not bad task evidence.
                body["max_tokens"] = 1_200
            req = Request(_URL, data=json.dumps(body).encode(), headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            }, method="POST")
            try:
                raw = transport(req, timeout_s) if transport else urlopen(req, timeout=timeout_s).read()
            except HTTPError as exc:
                raise FireworksError(f"Fireworks returned HTTP {exc.code}") from exc
            except URLError as exc:
                raise FireworksError(f"could not reach Fireworks: {exc.reason}") from exc
            try:
                content = json.loads(raw)["choices"][0]["message"]["content"]
                return TaskDraft.model_validate_json(content)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
                if attempt == 0:
                    continue
                raise FireworksError("Fireworks returned an invalid task-draft response after retry") from exc

    return enrich


__all__ = ["DEFAULT_MODEL", "FireworksError", "fireworks_task_enricher"]
