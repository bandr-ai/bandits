"""Convert a tau2-bench retail corpus dump into bandits OTLP JSONL, plus a label sidecar.

The dump in ``work/tau2-retail/out/corpus.json`` is a prior prototype's shape:
each trace carries ``messages`` (the interleaved dialogue, with ``tool`` roles
keyed by ``tool_call_id``) and ``invocations`` (the same tool calls, but with
their responses still structured rather than stringified). The messages give
order; the invocations give the payloads worth reading.

Two things are deliberately kept out of the trace. ``tau2_reward`` is written to
a separate labels file rather than into span attributes: bandits reads
``score``/``rating``/``evaluation`` off attributes as ``recorded_score``
evidence, and a verifier drafted from the ground truth would be checking the
answer against itself. And ``lineage_id`` is set to the tau2 task id, so the
four trials of one scenario move across the fit/held-out boundary together
rather than landing on both sides of it.

Usage::

    python scripts/tau2_to_otlp.py <corpus.json> <out.otlp.jsonl> <out.labels.json> [--flatten]
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_BASE = datetime(2026, 3, 1, tzinfo=UTC)
_SCALAR = (str, int, float, bool)


def _stamp(ordinal: int) -> str:
    """Distinct, ordered timestamps. OTLP sorts spans by (started_at, source line)."""
    return (_BASE + timedelta(seconds=ordinal)).isoformat().replace("+00:00", "Z")


def _flatten(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Lift nested scalars to depth-1 keys so ``_scalar_fields`` can see them.

    bandits reads only top-level scalars off a tool result, and tau2 buries the
    interesting facts — what was charged, what was returned — one or two levels
    down. Flattening is a claim about the data, not a fix to bandits, so it is
    opt-in and the two runs are reported separately.
    """
    if depth > 2:
        return {}
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}{key}"
            if isinstance(item, _SCALAR):
                out[name] = item
            else:
                out.update(_flatten(item, f"{name}_", depth + 1))
    elif isinstance(value, list):
        for index, item in enumerate(value[:3]):
            out.update(_flatten(item, f"{prefix}{index}_", depth + 1))
    return out


def _result(response: Any, flatten: bool) -> Any:
    if not flatten or not isinstance(response, (dict, list)):
        return response
    return _flatten(response) or response


def convert(corpus: dict, flatten: bool) -> tuple[list[dict], dict[str, dict]]:
    records: list[dict] = []
    labels: dict[str, dict] = {}
    ordinal = 0
    seen: dict[str, int] = {}

    for trace in corpus["traces"]:
        metadata = trace["metadata"]
        # The dump reuses one trace_id across a task's four trials, which are
        # genuinely distinct episodes (different lengths, different digests).
        # Left alone, the OTLP loader groups by trace_id and would concatenate
        # four episodes into one. The trial index disambiguates them; the task
        # id stays on `session.id`, so lineage still moves the four together.
        base = trace["trace_id"]
        trial = seen[base] = seen.get(base, -1) + 1
        trace_id = f"{base}#{trial}"
        messages = trace["messages"]
        by_call = {i["call_id"]: i for i in trace["invocations"]}

        instruction = next(
            (m["content"] for m in messages if m["role"] == "user" and m.get("content")), None
        )
        if instruction is None:
            continue

        greeting = next(
            (m["content"] for m in messages if m["role"] == "assistant" and m.get("content")), ""
        )

        # One synthetic root per episode. OTLP reads `task` only off the span
        # with no parent, and the tau2 dump declares the instruction nowhere
        # else — it is simply the first thing the user said.
        root_id = f"{trace_id}:s0"
        records.append(
            {
                "trace_id": trace_id,
                "span_id": root_id,
                "parent_span_id": None,
                "name": metadata.get("tau2_model") or "assistant",
                "start_time": _stamp(ordinal),
                "end_time": _stamp(ordinal),
                "attributes": {
                    "gen_ai.operation.name": "chat",
                    "task": instruction,
                    "session.id": f"tau2-task-{metadata['tau2_task_id']}",
                    "gen_ai.completion": greeting,
                },
            }
        )
        ordinal += 1

        emitted = 0
        seen_greeting = False
        root = records[-1]
        # User messages ride on the next span emitted, as the trailing
        # user-role entries of the standard ``gen_ai.input.messages`` — which
        # is where a real exporter would put them, and where bandits reads
        # ``user_turns`` from. Dropping them, as this once did, left every
        # customer reply out of the trace: the one reaction a support agent's
        # message actually gets.
        pending: list[str] = []

        def attach(record: dict, pending: list[str] = pending) -> dict:
            if pending:
                record["attributes"]["gen_ai.input.messages"] = [
                    {"role": "user", "content": text} for text in pending
                ]
                pending.clear()
            return record

        for message in messages:
            role, content = message["role"], message.get("content")

            if role == "user":
                if content:
                    pending.append(content)
                continue

            if role == "assistant":
                if not content:
                    continue  # a bare tool-call carrier; the call is its own span
                if not seen_greeting:
                    seen_greeting = True
                    attach(root)
                    continue  # already the root's completion
                records.append(
                    attach(
                        {
                            "trace_id": trace_id,
                            "span_id": f"{trace_id}:s{emitted + 1}",
                            "parent_span_id": root_id,
                            "name": metadata.get("tau2_model") or "assistant",
                            "start_time": _stamp(ordinal),
                            "end_time": _stamp(ordinal),
                            "attributes": {
                                "gen_ai.operation.name": "chat",
                                "gen_ai.completion": content,
                            },
                        }
                    )
                )
            elif role == "tool":
                invocation = by_call.get(message.get("tool_call_id"))
                if invocation is None:
                    continue
                records.append(
                    attach(
                        {
                            "trace_id": trace_id,
                            "span_id": f"{trace_id}:s{emitted + 1}",
                            "parent_span_id": root_id,
                            "name": invocation["tool"],
                            "start_time": _stamp(ordinal),
                            "end_time": _stamp(ordinal),
                            "attributes": {
                                "gen_ai.operation.name": "execute_tool",
                                "gen_ai.tool.call.arguments": invocation.get("arguments") or {},
                                "gen_ai.tool.call.result": _result(invocation["response"], flatten),
                                "status": "error" if invocation["status"] != "ok" else "ok",
                            },
                        }
                    )
                )
            else:
                continue

            emitted += 1
            ordinal += 1

        labels[trace_id] = {
            "reward": metadata["tau2_reward"],
            "success": metadata["tau2_success"],
            "task_id": metadata["tau2_task_id"],
        }

    return records, labels


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flatten = "--flatten" in sys.argv
    corpus = json.loads(Path(args[0]).read_text())
    records, labels = convert(corpus, flatten)

    Path(args[1]).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    Path(args[2]).write_text(json.dumps(labels, indent=2))
    print(f"wrote {len(records)} spans over {len(labels)} traces -> {args[1]}")
    print(f"wrote {len(labels)} ground-truth labels -> {args[2]}")
