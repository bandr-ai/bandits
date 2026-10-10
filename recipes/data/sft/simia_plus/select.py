"""Dedup, select down to target_count, and report diversity.

Exact dedup is Simia's (hash of human + assistant text). Near-dedup and balanced selection follow
Datology's curation: generate a bigger pool, remove redundancy, and keep coverage across seeds and
strategies rather than the highest-scoring head (rare-but-useful examples survive).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

_W = re.compile(r"\w+")


def exact_key(trace: dict) -> str:
    parts = [f"{m['role'][0]}:{m['content']}" for m in trace["messages"] if m["role"] in ("user", "assistant")]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def action_signature(trace: dict) -> tuple:
    return tuple(c["name"] for m in trace["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", []))


def _shingles(text: str, n: int = 3) -> set:
    toks = _W.findall(text.lower())
    return {tuple(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}


def first_user(trace: dict) -> str:
    return next((m["content"] for m in trace["messages"] if m["role"] == "user"), "")


def dedup(traces: list[dict], near: bool, threshold: float = 0.8) -> tuple[list[dict], dict]:
    seen, kept, stats = set(), [], Counter()
    buckets: dict[tuple, list[set]] = defaultdict(list)
    for t in traces:
        k = exact_key(t)
        if k in seen:
            stats["exact_dup"] += 1
            continue
        seen.add(k)
        if near:
            sh = _shingles(first_user(t))
            bucket = buckets[action_signature(t)]
            if any(len(sh & o) / max(1, len(sh | o)) >= threshold for o in bucket):
                stats["near_dup"] += 1
                continue
            bucket.append(sh)
        kept.append(t)
    return kept, dict(stats)


def _rank(t: dict) -> tuple:
    j = t["meta"].get("verify", {}).get("judge") or {}
    success = 1 if j.get("task_success", True) else 0
    n_tokens = sum(len(m.get("content") or "") for m in t["messages"])
    return (-success, n_tokens)  # successes first, then the shorter one (Datology "Brevity")


def select(traces: list[dict], target: int) -> list[dict]:
    """Round-robin over (seed, strategy) buckets so every seed and strategy keeps its share."""
    if len(traces) <= target:
        return traces
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for t in traces:
        buckets[(t["meta"].get("seed_id"), t["meta"].get("strategy"))].append(t)
    for b in buckets.values():
        b.sort(key=_rank)
    order = sorted(buckets)
    out, i = [], 0
    while len(out) < target:
        progressed = False
        for key in order:
            if i < len(buckets[key]) and len(out) < target:
                out.append(buckets[key][i])
                progressed = True
        if not progressed:
            break
        i += 1
    return out


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c) if n else 0.0


def diversity_report(traces: list[dict]) -> dict:
    sigs = Counter(action_signature(t) for t in traces)
    firsts = [first_user(t) for t in traces]
    all_sh = [_shingles(f) for f in firsts]
    uniq = set().union(*all_sh) if all_sh else set()
    total = sum(len(s) for s in all_sh)
    return {
        "n": len(traces),
        "distinct_action_signatures": len(sigs),
        "action_signature_entropy_bits": round(entropy(sigs), 3),
        "first_user_distinct_3gram_ratio": round(len(uniq) / total, 3) if total else 0.0,
        "by_strategy": dict(Counter(t["meta"].get("strategy") for t in traces)),
        "by_persona": dict(Counter(t["meta"].get("persona") for t in traces)),
        "by_mode": dict(Counter(t["meta"].get("mode") for t in traces)),
        "with_failure": sum(1 for t in traces if t["meta"].get("failure")),
        "per_seed_min_max": (min(Counter(t["meta"].get("seed_id") for t in traces).values(), default=0),
                             max(Counter(t["meta"].get("seed_id") for t in traces).values(), default=0)),
    }
