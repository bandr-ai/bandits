"""Banks of real data mined from the seeds: tool observations and user turns."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict

_TOK = re.compile(r"[A-Za-z0-9_]+")


def _tokens(x) -> set[str]:
    return {t.lower() for t in _TOK.findall(json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)}


class ObsBank:
    """Real (tool, arguments) -> result pairs from the seeds."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.by_tool: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            self.by_tool[e["tool"]].append(e)

    @classmethod
    def from_traces(cls, traces: list[dict]) -> ObsBank:
        entries = []
        for t in traces:
            calls = {}
            for m in t["messages"]:
                if m["role"] == "assistant":
                    for c in m.get("tool_calls", []):
                        calls[c["id"]] = c
                elif m["role"] == "tool" and m["tool_call_id"] in calls:
                    c = calls[m["tool_call_id"]]
                    entries.append({"tool": c["name"], "arguments": c["arguments"],
                                    "result": m["content"], "seed_id": t["id"]})
        return cls(entries)

    def retrieve(self, tool: str, arguments: dict | None, k: int, rng: random.Random) -> list[dict]:
        pool = self.by_tool.get(tool, [])
        if not pool or k <= 0:
            return []
        if arguments is None:
            return rng.sample(pool, min(k, len(pool)))
        q = _tokens(arguments)
        scored = sorted(pool, key=lambda e: (-len(q & _tokens(e["arguments"])), rng.random()))
        return scored[:k]

    def examples_for_tools(self, tools: list[dict], k: int, rng: random.Random) -> dict[str, list[dict]]:
        return {t["name"]: self.retrieve(t["name"], None, k, rng) for t in tools if t["name"] in self.by_tool}


class UserBank:
    """Real user messages from the seeds, used as style examples for the user simulator."""

    def __init__(self, entries: list[dict]):
        self.entries = entries

    @classmethod
    def from_traces(cls, traces: list[dict]) -> UserBank:
        return cls([{"content": m["content"], "seed_id": t["id"]}
                    for t in traces for m in t["messages"] if m["role"] == "user" and m["content"].strip()])

    def sample(self, k: int, rng: random.Random, exclude_seed: str | None = None) -> list[str]:
        pool = [e["content"] for e in self.entries if e["seed_id"] != exclude_seed] or \
               [e["content"] for e in self.entries]
        return rng.sample(pool, min(k, len(pool)))


def format_obs_examples(examples: dict[str, list[dict]] | list[dict], max_chars: int = 1500) -> str:
    if isinstance(examples, list):
        examples = {"": examples}
    lines = []
    for exs in examples.values():
        for e in exs:
            lines.append(f"- {e['tool']}({json.dumps(e['arguments'], ensure_ascii=False)}) -> {e['result'][:max_chars]}")
    return "\n".join(lines) or "(none available)"
