"""τ²-bench run-log adapter.

τ²-bench publishes one JSON object per evaluated task.  It is close to an
OpenAI chat transcript, but tool results identify the request with ``id`` rather
than ``tool_call_id`` and the canonical outcome is recorded as ``success`` (or
the numeric ``reward``).  This adapter makes that evidence explicit instead of
requiring an export-conversion script before Bandits can inspect a public run.
"""

from __future__ import annotations

from typing import Any

from bandits.contracts import IngestIssue, JsonObject, Trace
from bandits.ingest.chat_json import parse_chat_record


def _outcome(record: JsonObject) -> bool | None:
    """Return τ²'s externally computed task outcome, when it is unambiguous."""
    success = record.get("success")
    if isinstance(success, bool):
        return success
    reward = record.get("reward")
    if isinstance(reward, bool):
        return reward
    if isinstance(reward, (int, float)) and reward in (0, 1):
        return bool(reward)
    return None


def _normalized_messages(raw_messages: list[Any]) -> list[JsonObject]:
    """Translate only the documented τ² naming difference for tool replies."""
    messages: list[JsonObject] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            messages.append(raw)  # Let the chat adapter emit its normal issue.
            continue
        message = dict(raw)
        if message.get("role") == "tool" and "tool_call_id" not in message:
            call_id = message.get("id")
            if isinstance(call_id, str) and call_id:
                message["tool_call_id"] = call_id
        messages.append(message)
    return messages


def parse_tau2_record(
    record: JsonObject,
    *,
    source_digest: str,
    location: str,
    fallback_trace_id: str,
) -> tuple[Trace | None, list[IngestIssue]]:
    """Normalize one τ² evaluation trajectory into Bandits' common trace format."""
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list):
        # Reuse the standard adapter so malformed records get its stable issue.
        normalized: JsonObject = {"messages": raw_messages}
    else:
        task_id = record.get("task_id")
        trace_id = f"tau2:{task_id}" if isinstance(task_id, (str, int)) else fallback_trace_id
        normalized = {
            "conversation_id": trace_id,
            "messages": _normalized_messages(raw_messages),
            "outcome": _outcome(record),
        }

    trace, issues = parse_chat_record(
        normalized,
        source_digest=source_digest,
        location=location,
        fallback_trace_id=fallback_trace_id,
    )
    if trace is None:
        return None, issues
    metadata = dict(trace.metadata)
    metadata.update({
        "tau2_task_id": record.get("task_id"),
        "tau2_reward": record.get("reward"),
        "tau2_success": record.get("success"),
        "tau2_model": record.get("model"),
    })
    return trace.model_copy(update={"source": "tau2", "metadata": metadata}), issues


__all__ = ["parse_tau2_record"]
