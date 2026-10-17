"""Tool-result rules learned from the seeds, enforced by code (plan step 7: unknown branches stay unknown).

Generic: nothing here names a tool, field or domain. From the seeds we learn, per tool call signature
(tool name + its short enum-like argument values):
  - result keys: every key real calls returned, at the top level and inside list items, per field. A
    generated result may omit keys (optional fields vary) but may not add one the seeds never returned;
  - availability gates: context tokens (strings in list fields of the first user turn) that perfectly
    separate the seeds' available=true calls (token always present) from available=false calls (always
    absent), with enough support on both sides to not be a coincidence.
A generated result with an unseen shape, or available=true without its gate token, is impossible
for the recorded harness and is rejected.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

_OBS = re.compile(r"^\[[A-Za-z_]+:\d+\]\s*")  # a numbered tag some recorders put before a tool result


def _context_tokens(trace: dict) -> set[str] | None:
    """Strings inside list-of-string fields of the first JSON object in the first user turn."""
    user = next((m.get("content") or "" for m in trace.get("messages", []) if m["role"] == "user"), "")
    start = user.find("{")
    if start < 0:
        return None
    try:
        ctx = json.JSONDecoder().raw_decode(user[start:])[0]
    except ValueError:
        return None
    if not isinstance(ctx, dict):
        return None
    return {f"{k}:{x}" for k, v in ctx.items() if isinstance(v, list) for x in v if isinstance(x, str)}


def _calls(trace: dict):
    """(tool name, arguments, parsed result or None) for every answered call."""
    pending = {}
    for m in trace.get("messages", []):
        for c in m.get("tool_calls") or []:
            pending[c["id"]] = c
        if m["role"] == "tool" and m.get("tool_call_id") in pending:
            c = pending[m["tool_call_id"]]
            try:
                result = json.loads(_OBS.sub("", m.get("content") or ""))
            except ValueError:
                result = None
            yield c["name"], c.get("arguments") or {}, result


def _shapes(result) -> list[tuple]:
    """Top-level key set, plus the key set of each dict item in list fields (tagged by field)."""
    if not isinstance(result, dict):
        return []
    out = [("top", tuple(sorted(result)))]
    for k, v in result.items():
        if isinstance(v, list):
            out += [(k, tuple(sorted(x))) for x in v if isinstance(x, dict)]
        elif isinstance(v, dict):
            out.append((k, "dict"))
    return out


class SeedRules:
    def __init__(self, seeds: list[dict], mode_max: int = 5, min_avail: int = 3, min_unavail: int = 2):
        values = defaultdict(set)
        for s in seeds:
            for name, args, _ in _calls(s):
                for k, v in args.items():
                    if isinstance(v, str):
                        values[(name, k)].add(v)
        # a mode argument selects what the tool reads (few distinct short values); free-text args do not
        self.enum_args = {key for key, vs in values.items() if len(vs) <= mode_max and all(len(v) <= 40 for v in vs)}
        self.shapes = defaultdict(set)
        self.availability = defaultdict(set)  # signature -> availability values the seeds returned
        avail, unavail = defaultdict(list), defaultdict(list)
        for s in seeds:
            tokens = _context_tokens(s)
            for name, args, result in _calls(s):
                sig = self.signature(name, args)
                for field, keys in _shapes(result):
                    self.shapes[(sig, field)].update(keys if isinstance(keys, tuple) else (keys,))
                if isinstance(result, dict) and isinstance(result.get("available"), bool):
                    self.availability[sig].add(result["available"])
                    if tokens is not None:
                        for sub in self._subs(sig):
                            (avail if result["available"] else unavail)[sub].append(tokens)
        self.gates = {}
        for sig, seen in avail.items():
            off = unavail.get(sig, [])
            if len(seen) < min_avail or len(off) < min_unavail:
                continue
            gate = {t for t in set.intersection(*seen) if all(t not in u for u in off)}
            if gate:
                self.gates[sig] = gate

    @staticmethod
    def _subs(sig: tuple) -> list[tuple]:
        """The tool alone and the tool with each single mode argument: coarser views learn gates from more seeds."""
        return [sig[:1], *((sig[0], a) for a in sig[1:])]

    def signature(self, name: str, args: dict) -> tuple:
        return (name, *sorted((k, v) for k, v in args.items() if (name, k) in self.enum_args and isinstance(v, str)))

    def errors(self, trace: dict) -> list[str]:
        tokens = _context_tokens(trace)
        out = []
        for name, args, result in _calls(trace):
            sig = self.signature(name, args)
            known = {k: v for k, v in self.shapes.items() if k[0] == sig}
            if known:  # a signature the seeds never called has no learned shapes; other checks own it
                for field, keys in _shapes(result):
                    seen = known.get((sig, field))
                    new = sorted(set(keys if isinstance(keys, tuple) else (keys,)) - (seen or set()))
                    if seen is None or new:
                        out.append(f"{name}{dict(sig[1:])}: result {field} has "
                                   f"{'a field' if seen is None else 'keys ' + str(new)} the seeds never returned")
            returned = isinstance(result, dict) and result.get("available") is True
            if returned and self.availability.get(sig) == {False}:
                out.append(f"{name}{dict(sig[1:])}: returned data, but every seed call of it was unavailable "
                           "(an unknown branch stays unknown)")
            for sub in self._subs(sig):
                gate = self.gates.get(sub)
                # several perfect separators are equivalent witnesses: one present is enough
                if gate and tokens is not None and returned and not gate & tokens:
                    out.append(f"{name}{dict(sub[1:])}: available=true but the context has none of {sorted(gate)}, "
                               "which every seed returning data had and every unavailable one lacked")
                    break
        return out


    def describe(self) -> str:
        """The learned rules as plain text for the planner/generator: prevention uses the same rules code enforces."""
        lines = []
        for sub, gate in sorted(self.gates.items(), key=str):
            lines.append(f"- {sub[0]}{dict(sub[1:]) if len(sub) > 1 else ''} returns data only when the first user input "
                         f"lists one of {sorted(t.split(':', 1)[1] for t in gate)}; otherwise use its unavailable shape.")
        for sig, vals in sorted(self.availability.items(), key=str):
            if vals == {False}:
                lines.append(f"- {sig[0]}{dict(sig[1:])} never returned data in the examples: always use its unavailable shape.")
        fields = defaultdict(set)
        for (sig, field), keys in self.shapes.items():
            fields[(sig[0], field)].update(keys)
        for (tool, field), keys in sorted(fields.items(), key=str):
            lines.append(f"- {tool} result {'top level' if field == 'top' else field + ' items'} uses only keys: {sorted(map(str, keys))}")
        return "\n".join(lines)
