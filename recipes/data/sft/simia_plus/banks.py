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

    def __init__(self, entries: list[dict], inputs: list[dict] | None = None):
        self.entries = entries
        self.inputs = inputs or []  # real user turns: where the evidence lives for agents without tools
        self.by_tool: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            self.by_tool[e["tool"]].append(e)

    @classmethod
    def from_traces(cls, traces: list[dict]) -> ObsBank:
        entries = []
        inputs = [{"content": m["content"], "seed_id": t["id"]} for t in traces for m in t["messages"]
                  if m["role"] == "user" and (m.get("content") or "").strip()]
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
        return cls(entries, inputs)

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


    def input_examples(self, k: int, rng: random.Random, exclude_seed: str | None = None) -> list[str]:
        """Real inputs from OTHER seeds: what observations look like for an agent whose evidence arrives in messages."""
        pool = [e["content"] for e in self.inputs if e["seed_id"] != exclude_seed]
        return rng.sample(pool, min(k, len(pool)))


class ResultFormat:
    """The harness's result prefix (cfg.tool_result_prefix, a regex whose group 1, if any, is the 1-based running index
    of tool results), learned from the seeds' own results. It is harness formatting, so code adds it to simulated
    results instead of asking a model to reproduce it. No prefix configured, or none found in the seeds: a no-op."""

    def __init__(self, regex: str | None, traces: list[dict]):
        self.regex, self.template = regex, None
        if not regex:
            return
        for t in traces:
            for m in t.get("messages", []):
                hit = re.match(regex, m.get("content") or "") if m.get("role") == "tool" else None
                if hit:
                    lit = hit.group(0)
                    self.template = (lit[:hit.start(1)], lit[hit.end(1):]) if hit.re.groups else (lit, None)
                    return

    def strip(self, content: str) -> str:
        hit = re.match(self.regex, content) if self.regex and isinstance(content, str) else None
        return content[hit.end():] if hit else content

    def apply(self, content: str, n: int) -> str:
        """content with any prefix it carries replaced by the harness prefix for result number n."""
        if not self.template:
            return content
        head, tail = self.template
        out = (head + str(n) + tail if tail is not None else head) + self.strip(content)
        hit = re.match(self.regex, out)
        if not hit or (hit.re.groups and hit.group(1) != str(n)):
            return content  # the learned template does not round-trip: leave it to the code check
        return out


def format_input_examples(inputs: list[str], max_chars: int = 1500) -> str:
    return "\n".join(f"- real input from another conversation: {x[:max_chars]}" for x in inputs) or "(none available)"


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
