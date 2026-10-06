"""The policy call and judge scoring shared by evaluation and training runs.

``make_chat`` adapts an OpenAI-compatible vLLM server to ``run_rollout``;
with ``keep_tokens`` it asks vLLM for the exact prompt and sampled token ids
and the sampled tokens' log-probs. ``judge_steps`` builds the judge requests
for a finished rollout: the question, the two previous steps, the action and
its real reply. Nothing from ground truth goes into either.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from bandits_jev.rollout import TOOLS, ContextOverflow


def _extra(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None and getattr(obj, "model_extra", None):
        value = obj.model_extra.get(name)
    return value


def make_chat(client: Any, model: str, seed: int, *, keep_tokens: bool = False, max_tokens: int = 1024):
    import openai

    async def chat(messages: list[dict], tool_choice: str) -> dict:
        try:
            reply = await client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice=tool_choice,
                temperature=1.0, max_tokens=max_tokens, seed=seed,
                logprobs=keep_tokens, extra_body={"return_token_ids": True} if keep_tokens else None,
            )
        except openai.BadRequestError as exc:
            if "context length" in str(exc) or "maximum" in str(exc):
                raise ContextOverflow(str(exc)[:200]) from exc
            raise
        choice = reply.choices[0]
        message = choice.message
        out = {
            "content": message.content,
            "tool_calls": [
                {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                for c in message.tool_calls or []
            ],
        }
        if keep_tokens:
            out["prompt_token_ids"] = _extra(reply, "prompt_token_ids")
            out["token_ids"] = _extra(choice, "token_ids")
            content = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
            out["logprobs"] = [token.logprob for token in content]
        return out

    return chat


def judge_requests(question: str, events: list[dict]) -> list[dict]:
    requests, history = [], []
    for event in events:
        requests.append(
            {
                "question": question,
                "previous": [list(step) for step in history[-2:]],
                "tool": event["tool"],
                "action": event["action"],
                "observation": event["observation"],
            }
        )
        history.append((event["tool"], event["action"], event["observation"]))
    return requests


async def judge_rollout(score: Callable[[list[dict]], Awaitable[dict]], question: str, result: dict) -> dict:
    """Attach judge results to the rollout's events; returns the judge's provenance."""
    if not result["events"]:
        return {}
    reply = await score(judge_requests(question, result["events"]))
    for event, scored in zip(result["events"], reply["results"], strict=True):
        event["judge"] = scored
    return reply["judge"]
