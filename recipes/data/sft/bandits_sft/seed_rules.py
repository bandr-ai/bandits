"""Tool-result rules learned from the seeds, enforced by code (unknown branches stay unknown).

Generic: nothing here names a tool, field or domain. From the seeds we learn, per tool call signature
(tool name + its short enum-like argument values):
  - result keys: every key real calls returned, at the top level and inside list items, per field. A
    generated result may omit keys (optional fields vary) but may not add one the seeds never returned.
    Enforced only where the seeds called that signature often enough to know its keys (min_support);
    with fewer calls an unseen key is as likely an optional field the seeds happened not to show;
  - branch keys: top-level result fields with a few short values that decide which other fields come back
    (for example a flag that is false on an empty result and true on a full one). Learned across all tools.
    The value whose other fields are empty most often is the empty branch; the others are data branches. A signature the seeds only ever saw on
    an empty branch may not return a data branch: that branch is unknown;
  - gates: context tokens (strings in list fields of the first user turn) that perfectly separate the seeds'
    data-branch calls (token always present) from empty-branch calls (always absent), with enough support on
    both sides to not be a coincidence.
A generated result breaking any of these is impossible for the recorded tools and is rejected.
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


def _empty(v) -> bool:
    return v is None or v is False or isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0 \
        or isinstance(v, (str, list, dict)) and not v


def _flag(v) -> bool:
    return isinstance(v, bool) or isinstance(v, str) and 0 < len(v) <= 20


def _emptiness(groups: dict[str, list]) -> dict[str, float]:
    """Per branch value, how empty its results' other fields are (0, "", [], {}, null, false), compared within a tool
    (tools differ in size); a tool showing one value only does not vote. Pooled when no tool shows two values."""
    votes: dict[str, list] = defaultdict(list)
    for tool in {t for g in groups.values() for t, _ in g}:
        mean = {v: [e for t, e in g if t == tool] for v, g in groups.items()}
        mean = {v: sum(e) / len(e) for v, e in mean.items() if e}
        if len(mean) > 1:
            for v, m in mean.items():
                votes[v].append(m)
    if votes:
        return {v: sum(votes[v]) / len(votes[v]) if votes.get(v) else 0.0 for v in groups}
    return {v: sum(e for _, e in g) / len(g) for v, g in groups.items()}


def _witness(rates: dict, order: list[str], empty_at_least: float = 0.8, full_at_most: float = 0.2) -> bool:
    """A real empty branch has a witness field: empty in nearly all of its results (a zero count, an empty list) and
    filled in nearly all results of every other value. A category (values with different but full results) has none."""
    first, rest = rates[order[0]], [rates[v] for v in order[1:]]
    return any(n and e / n >= empty_at_least and all(f in o and o[f][1] and o[f][0] / o[f][1] <= full_at_most for o in rest)
               for f, (e, n) in first.items())


def branch_keys(calls: list[tuple], max_values: int = 3, min_each: int = 2) -> dict[str, list]:
    """Result fields whose value decides which other fields come back, from (tool, arguments, result) calls: few
    short values, each seen at least min_each times, not an echo of an argument, and the emptiest value has a witness
    field (empty under it, filled under the others; see _witness). Values are ordered from the emptiest branch to the
    fullest; every value after the first is a data branch."""
    by_key: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    echoed, present = defaultdict(int), defaultdict(int)
    empties: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    fields: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for tool, args, r in calls:
        for k, v in r.items():
            present[k] += 1
            if _flag(v):
                by_key[k][json.dumps(v)].append((tool, frozenset(r) - {k}))
                empties[k][json.dumps(v)].append((tool, sum(_empty(x) for k2, x in r.items() if k2 != k)))
                for f, x in r.items():
                    if f != k:
                        fields[k][json.dumps(v)][f][0] += _empty(x)
                        fields[k][json.dumps(v)][f][1] += 1
                echoed[k] += v in {a for a in args.values() if isinstance(a, str)}
    out = {}
    for k, groups in by_key.items():
        n = sum(len(g) for g in groups.values())
        if not 2 <= len(groups) <= max_values or any(len(g) < min_each for g in groups.values()):
            continue
        if n < present[k] or echoed[k] > 0.5 * n:
            continue  # takes long or structured values too, or repeats what the call asked for
        score = _emptiness(empties[k])  # higher = emptier
        order = sorted(groups, key=lambda v: (-score[v], v))
        if not _witness(fields[k], order):
            continue  # values differ, but none is an empty result: a category, not a branch
        out[k] = [json.loads(v) for v in order]
    return out


class SeedRules:
    def __init__(self, seeds: list[dict], mode_max: int = 5, min_avail: int = 3, min_unavail: int = 2,
                 min_support: int = 3):
        values = defaultdict(set)
        calls = []
        for s in seeds:
            for name, args, result in _calls(s):
                for k, v in args.items():
                    if isinstance(v, str):
                        values[(name, k)].add(v)
                if isinstance(result, dict):
                    calls.append((name, args, result))
        # a mode argument selects what the tool reads (few distinct short values); free-text args do not
        self.enum_args = {key for key, vs in values.items() if len(vs) <= mode_max and all(len(v) <= 40 for v in vs)}
        self.branch_keys = branch_keys(calls)
        self._order = {k: [json.dumps(v) for v in order] for k, order in self.branch_keys.items()}
        self.min_support = min_support
        self.shapes = defaultdict(set)
        self.support = defaultdict(int)  # (signature, field) -> results that showed that field's shape
        self.branches = defaultdict(set)  # (signature, key) -> branch values the seeds returned
        data, empty = defaultdict(list), defaultdict(list)
        for s in seeds:
            tokens = _context_tokens(s)
            for name, args, result in _calls(s):
                sig = self.signature(name, args)
                for field in {f for f, _ in _shapes(result)}:
                    self.support[(sig, field)] += 1
                for field, keys in _shapes(result):
                    self.shapes[(sig, field)].update(keys if isinstance(keys, tuple) else (keys,))
                for key in self.branch_keys:
                    value = json.dumps(result.get(key)) if isinstance(result, dict) and key in result else None
                    if value in self._order[key]:  # compared as JSON: 1 is not true, 0 is not false
                        self.branches[(sig, key)].add(value)
                        if tokens is not None:
                            side = data if value != self._order[key][0] else empty
                            for sub in self._subs(sig):
                                side[(sub, key)].append(tokens)
        self.gates = {}
        for k, seen in data.items():
            off = empty.get(k, [])
            if len(seen) < min_avail or len(off) < min_unavail:
                continue
            gate = {t for t in set.intersection(*seen) if all(t not in u for u in off)}
            if gate:
                self.gates[k] = gate

    @staticmethod
    def _subs(sig: tuple) -> list[tuple]:
        """The tool alone and the tool with each single mode argument: coarser views learn gates from more seeds."""
        return [sig[:1], *((sig[0], a) for a in sig[1:])]

    def signature(self, name: str, args: dict) -> tuple:
        return (name, *sorted((k, v) for k, v in args.items() if (name, k) in self.enum_args and isinstance(v, str)))

    def empty_only(self) -> list[tuple]:
        """(signature, key) pairs the seeds only ever saw on the empty branch of a branch key."""
        return sorted((k for k, vals in self.branches.items()
                       if vals == {json.dumps(self.branch_keys[k[1]][0])}), key=str)

    def never_returns_data(self, tool: str) -> bool:
        """Every seed call of this tool (any arguments) came back on an empty branch."""
        seen = [(key, vals) for (sig, key), vals in self.branches.items() if sig[0] == tool]
        return bool(seen) and all(vals == {json.dumps(self.branch_keys[key][0])} for key, vals in seen)

    def errors(self, trace: dict) -> list[str]:
        tokens = _context_tokens(trace)
        out = []
        for name, args, result in _calls(trace):
            sig = self.signature(name, args)
            known = {k: v for k, v in self.shapes.items() if k[0] == sig}
            if known and self.support.get((sig, "top"), 0) >= self.min_support:
                # a signature seen too rarely has no reliable key list; other checks own it
                for field, keys in _shapes(result):
                    seen = known.get((sig, field))
                    if seen is not None and self.support[(sig, field)] < self.min_support:
                        continue
                    new = sorted(set(keys if isinstance(keys, tuple) else (keys,)) - (seen or set()))
                    if seen is None or new:
                        out.append(f"{name}{dict(sig[1:])}: result {field} has "
                                   f"{'a field' if seen is None else 'keys ' + str(new)} the seeds never returned")
            for key, order in self.branch_keys.items():
                value = result.get(key) if isinstance(result, dict) else None
                dumped = json.dumps(value) if isinstance(result, dict) and key in result else None
                if dumped not in self._order[key] or dumped == self._order[key][0]:
                    continue
                seen = self.branches.get((sig, key))
                if seen == {self._order[key][0]} or seen is None and self.never_returns_data(name):
                    out.append(f"{name}{dict(sig[1:])}: returned {key}={json.dumps(value)}, but every seed call of it "
                               f"returned {key}={json.dumps(order[0])} (an unknown branch stays unknown)")
                for sub in self._subs(sig):
                    gate = self.gates.get((sub, key))
                    # several perfect separators are equivalent witnesses: one present is enough
                    if gate and tokens is not None and not gate & tokens:
                        out.append(f"{name}{dict(sub[1:])}: {key}={json.dumps(value)} but the context has none of "
                                   f"{sorted(gate)}, which every seed returning data had and every empty one lacked")
                        break
        return out

    def describe(self) -> str:
        """The learned rules as plain text for the planner/generator: prevention uses the same rules code enforces."""
        lines = []
        for key, order in sorted(self.branch_keys.items()):
            lines.append(f"- Result field {key!r} decides what comes back: {json.dumps(order[0])} is the empty result; "
                         f"{', '.join(json.dumps(v) for v in order[1:])} carries data.")
        for (sub, key), gate in sorted(self.gates.items(), key=str):
            empty = json.dumps(self.branch_keys[key][0])
            lines.append(f"- {sub[0]}{dict(sub[1:]) if len(sub) > 1 else ''} returns data only when the first user input "
                         f"lists one of {sorted(t.split(':', 1)[1] for t in gate)}; otherwise it returns {key}={empty}.")
        for sig, key in self.empty_only():
            lines.append(f"- {sig[0]}{dict(sig[1:])} never returned data in the examples: always return "
                         f"{key}={json.dumps(self.branch_keys[key][0])}.")
        fields = defaultdict(set)
        for (sig, field), keys in self.shapes.items():
            if self.support[(sig, field)] >= self.min_support:
                fields[(sig[0], field)].update(keys)
        for (tool, field), keys in sorted(fields.items(), key=str):
            lines.append(f"- {tool} result {'top level' if field == 'top' else field + ' items'} uses only keys: "
                         f"{sorted(map(str, keys))}")
        return "\n".join(lines)
