"""One policy rollout over the search tools, independent of how the model is served.

``chat`` is any async function that takes the message list (and a tool choice)
and returns the model's next message as {"content": str | None, "tool_calls":
[{"id", "name", "arguments"}]}. One browser action runs per turn, as in the RL
harness: extra calls in a turn are dropped and counted. The last turn forbids
tools so the model must answer. Ground truth (answer, evidence ids) is used
only to score the finished rollout and never enters a prompt.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from bandits_jev.answer_match import answer_matches
from bandits_jev.search_env import Searcher, SearchSession

SYSTEM_PROMPT = (
    "You are a research agent. Answer the question by searching a document collection.\n"
    "Tools: search(query) returns document ids with snippets; open(id) reads a document "
    "whose id appeared in a search result; find(pattern) looks for a regular expression in "
    "the open document.\n"
    "Call one tool at a time. Search several different ways and verify claims in the "
    "documents. When you are confident, reply without a tool call with exactly: "
    "Answer: <short answer>"
)


def _tool(name: str, description: str, argument: str, argument_description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string", "description": argument_description}},
                "required": [argument],
            },
        },
    }


TOOLS = [
    _tool("search", "Search the collection. Returns ids and snippets.", "query", "Search query."),
    _tool("open", "Open a document by an id returned from a search.", "id", "Document id."),
    _tool("find", "Find a regular expression in the open document.", "pattern", "Regular expression."),
]

Chat = Callable[[list[dict[str, Any]], str], Awaitable[dict[str, Any]]]


class ContextOverflow(Exception):
    """Raised by ``chat`` when the conversation no longer fits the model."""


async def run_rollout(task: dict, chat: Chat, searcher: Searcher, *, max_turns: int = 12, top_k: int = 10) -> dict:
    session = SearchSession(searcher, top_k=top_k)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["query"]},
    ]
    events: list[dict[str, Any]] = []
    assistant_texts: list[str] = []
    model_calls: list[dict[str, Any]] = []
    evidence = set(task["evidence_ids"])
    final, ended, error, dropped = None, "max_turns", None, 0
    for turn in range(max_turns):
        last = turn == max_turns - 1
        try:
            message = await chat(messages, "none" if last else "auto")
        except ContextOverflow as exc:
            ended, error = "error", f"context overflow: {exc}"
            break
        calls = message.get("tool_calls") or []
        if message.get("token_ids") is not None:
            model_calls.append(
                {
                    "prompt_token_ids": message["prompt_token_ids"],
                    "token_ids": message["token_ids"],
                    "logprobs": message.get("logprobs"),
                }
            )
        if message.get("content"):
            assistant_texts.append(message["content"])
        if not calls or last:
            final, ended = message.get("content") or "", "answer"
            break
        call, dropped = calls[0], dropped + len(calls) - 1
        try:
            arguments = json.loads(call["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("arguments are not an object")
            seen_before, opened_before = set(session.seen), set(session.opened)
            reply = await asyncio.to_thread(session.execute, "browser." + call["name"], arguments)
        except (json.JSONDecodeError, ValueError) as exc:
            arguments, reply = {"raw": call["arguments"]}, f"Error: invalid arguments ({exc})"
            seen_before = opened_before = set()
        events.append(
            {
                "turn": turn,
                "tool": "browser." + call["name"],
                "action": SearchSession.action_text(arguments),
                "observation": reply,
                # Ground truth about the step, for scoring only: never shown to the policy or judge.
                "new_evidence_seen": len(evidence & (session.seen - seen_before)),
                "evidence_opened": bool(evidence & (session.opened - opened_before)),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": reply})

    def recall(found: set[str]) -> float:
        return len(evidence & found) / len(evidence) if evidence else 0.0

    return {
        "query_id": task["query_id"],
        "ended": ended,
        "error": error,
        "turns": len(events) + (1 if final is not None else 0),
        "tool_calls": len(events),
        "dropped_extra_calls": dropped,
        "final": final,
        "correct": bool(final) and answer_matches(final, task["answer"]),
        "evidence_seen_recall": recall(session.seen),
        "evidence_opened_recall": recall(session.opened),
        "events": events,
        "assistant_texts": assistant_texts,
        # Exact sampled token ids per model call, when the server returned them (training only).
        "model_calls": model_calls,
    }


def summarize(results: list[dict]) -> dict:
    """Headline numbers over finished rollouts whose events carry judge scores."""
    n = len(results)
    events = [e for r in results for e in r["events"]]
    scored = [e for e in events if e.get("judge") and e["judge"]["score"] is not None]
    useful = [e["judge"]["score"] for e in scored if e["new_evidence_seen"] or e["evidence_opened"]]
    other = [e["judge"]["score"] for e in scored if not (e["new_evidence_seen"] or e["evidence_opened"])]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    repeats = sum(
        len({e["action"] for e in r["events"]}) < len(r["events"]) for r in results
    )
    classes = {"positive": 0, "neutral": 0, "negative": 0}
    for e in scored:
        classes[max(e["judge"]["probabilities"], key=e["judge"]["probabilities"].get)] += 1
    return {
        "rollouts": n,
        "answered": sum(r["ended"] == "answer" for r in results) / n,
        "accuracy_proxy": sum(r["correct"] for r in results) / n,
        "mean_tool_calls": sum(r["tool_calls"] for r in results) / n,
        "evidence_seen_recall": sum(r["evidence_seen_recall"] for r in results) / n,
        "evidence_opened_recall": sum(r["evidence_opened_recall"] for r in results) / n,
        "rollouts_with_repeated_action": repeats / n,
        "rollouts_with_dropped_extra_calls": sum(r["dropped_extra_calls"] > 0 for r in results) / n,
        "errors": sum(r["ended"] == "error" for r in results),
        "judged_steps": len(scored),
        "judge_class_share": {k: v / len(scored) for k, v in classes.items()} if scored else None,
        "judge_mean_score_on_evidence_steps": mean(useful),
        "judge_mean_score_on_other_steps": mean(other),
        "evidence_steps": len(useful),
    }
