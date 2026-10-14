"""Pilots for getting more traces from the same seeds, each written to analyzer/exp_<name>/ and never merged on its own.

reorder : re-sequence the calls and results a trace already has (code only, no model call). Only for seeds whose calls are
          independent (no later call uses an earlier result) and whose system prompt gives no order.
combo   : two variations already judged good on the same seed, touching different tools, applied together.
branch  : one extra tool call the seed never made, with a result shaped like the seeds' own results.
Combination and branch run through the same frozen template and checks as the main pipeline (analyzer.run_arm)."""
from __future__ import annotations

import copy
import itertools
import json
import random
import re
from pathlib import Path

from .analyzer import combine, generate_analyzer, run_arm
from .banks import ObsBank
from .config import Config
from .io import read_jsonl
from .llm import chat_json, get_llm
from .schema import render

ORDER_HINT = re.compile(r"\b(first|before|then|after)\b[^.\n]{0,60}\b[a-z]+_[a-z_]+\b", re.IGNORECASE)


def _calls(trace: dict) -> list[dict]:
    return [c for m in trace["messages"] if m.get("tool_calls") for c in m["tool_calls"]]


def independent_calls(trace: dict) -> bool:
    """>= 2 calls, none of the later calls takes a long string argument that appears in an earlier result, and the system
    prompt does not hint at an order. Crude on purpose: when unsure it answers False."""
    calls = _calls(trace)
    if len(calls) < 2 or ORDER_HINT.search(trace.get("system", "")):
        return False
    results = " ".join(m["content"] for m in trace["messages"] if m["role"] == "tool")
    return not any(str(v) in results for c in calls[1:] for v in c["arguments"].values() if isinstance(v, str) and len(v) > 6)


def reorder_trace(trace: dict, perm: list[int]) -> dict:
    """Same calls and results, in the order given by perm (position p gets original pair perm[p]); numbered result prefixes ("[tag:N]") renumbered."""
    pairs, answered = [], {}
    for m in trace["messages"]:
        if m["role"] == "tool":
            answered[m["tool_call_id"]] = m
    for c in _calls(trace):
        pairs.append((c, answered[c["id"]]))
    assert sorted(perm) == list(range(len(pairs))), "perm must be a permutation of the call positions"
    out, pos = [], 0
    for m in trace["messages"]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            k = len(m["tool_calls"])
            new_calls = [pairs[perm[p]][0] for p in range(pos, pos + k)]
            out.append({**m, "tool_calls": copy.deepcopy(new_calls)})
            pos += k
        elif m["role"] == "tool":
            continue
        else:
            out.append(copy.deepcopy(m))
        if m["role"] == "assistant" and m.get("tool_calls"):
            for p in range(pos - len(m["tool_calls"]), pos):
                _call, res = pairs[perm[p]]
                body = re.sub(r"^\[([a-z]+):\d+\]", lambda m, n=p + 1: f"[{m.group(1)}:{n}]", res["content"])
                out.append({**res, "content": body})
    return {**trace, "messages": out}


def reorder_rows(traces: list[dict], per_trace: int, seed: int = 0) -> list[dict]:
    rng, rows = random.Random(seed), []
    for t in traces:
        if not independent_calls(t):
            continue
        n = len(_calls(t))
        perms = [list(p) for p in itertools.permutations(range(n)) if list(p) != list(range(n))]
        rng.shuffle(perms)
        for k, perm in enumerate(perms[:per_trace]):
            r = reorder_trace(t, perm)
            r["id"] = f"{t['id']}#reorder{k + 1}"
            r["meta"] = {**t.get("meta", {}), "variation": "reorder", "perm": perm, "source_id": t["id"]}
            rows.append(r)
    return rows


def mechanical_ok(trace: dict) -> list[str]:
    ms, issues = trace["messages"], []
    ids = {c["id"] for m in ms for c in m.get("tool_calls") or []}
    ans = [m["tool_call_id"] for m in ms if m["role"] == "tool"]
    if ids != set(ans) or len(ans) != len(set(ans)):
        issues.append("tool call/result pairing")
    seq = [int(re.match(r"\[[a-z]+:(\d+)\]", m["content"]).group(1)) for m in ms if m["role"] == "tool" and re.match(r"\[[a-z]+:\d+\]", m["content"])]
    if seq != list(range(1, len(seq) + 1)):
        issues.append("result numbering")
    return issues


UNAVAILABLE = re.compile(r"unavailable|not captured|was not captured|available[=:]? ?false|no .{0,30}captured", re.IGNORECASE)


def tool_signature(variation: dict, trace: dict) -> tuple[set[str], set[str]]:
    """(tools, kinds) the variation touches, read from its text against this trace's own tool definitions."""
    tools = {t["name"] for t in trace.get("tools", [])}
    kinds = {k for t in trace.get("tools", []) for k in ((t.get("parameters") or {}).get("properties", {}).get("kind") or {}).get("enum", []) or []}
    text = " ".join(str(variation.get(k, "")) for k in ("change", "dependencies")) + " " + " ".join(variation.get("requires_tools") or [])
    return {x for x in tools if x in text}, {k for k in kinds if k in text}


def hard_conflict(a: dict, b: dict, trace: dict) -> bool:
    """Only what is certainly contradictory: two changes on the same tool where either makes that tool's evidence
    unavailable, or a change whose text names no tool/kind at all. Everything else is left to the model screen below."""
    (ta, ka), (tb, kb) = tool_signature(a, trace), tool_signature(b, trace)
    if not (ta | ka) or not (tb | kb):
        return True
    return bool(ta & tb) and (bool(UNAVAILABLE.search(str(a.get("change", "")))) or bool(UNAVAILABLE.search(str(b.get("change", "")))))


SCREEN = """You decide whether two changes to one real trace can both hold at once in ONE consistent version of that trace.

<seed>
{seed}
</seed>
<change_1>{a}</change_1>
<change_2>{b}</change_2>

They are compatible only if a single realistic situation satisfies both, no observation one of them needs contradicts what the other
needs, and the combined situation is still the seed's own situation (same failure, same material). If one change makes a source
unavailable, any other change that reads that source is incompatible. Return JSON only:
{{"compatible": true, "reason": "<one sentence naming the combined situation, or the conflict>"}}"""


def screen_pair(cfg: Config, seed: dict, a: dict, b: dict) -> dict:
    out = chat_json(get_llm(cfg, "judge"), "You are a careful reviewer. Output JSON only.",
                    SCREEN.format(seed=render(seed, max_obs_chars=1500)[:9000], a=json.dumps({k: a.get(k) for k in ("name", "change", "dependencies")}, ensure_ascii=False),
                                  b=json.dumps({k: b.get(k) for k in ("name", "change", "dependencies")}, ensure_ascii=False)))
    return {"compatible": bool(isinstance(out, dict) and out.get("compatible") is True), "reason": str((out or {}).get("reason", ""))[:300] if isinstance(out, dict) else ""}


def combo_jobs(dev: list[dict], good: list[dict], n: int, exclude: set[str], per_seed: int = 2, screen=None,
               max_candidates: int = 6) -> list[dict]:
    """good: [{"seed_id": ..., "variation": spec}] cells already judged good. Candidate pairs on one seed that pass hard_conflict();
    screen(seed, a, b) -> {"compatible": bool, "reason": str} (a model call) decides which are used."""
    by_id = {s["id"]: s for s in dev}
    by_seed: dict[str, list[dict]] = {}
    for g in good:
        if g["seed_id"] in by_id and g["seed_id"][:13] not in exclude:
            # specs saved on traces had applies_to/eligible_seeds stripped before generation: restore neutral defaults
            by_seed.setdefault(g["seed_id"], []).append({"applies_to": ["all"], "requires_tools": [], **g["variation"]})
    jobs, log = [], []
    for sid, vs in sorted(by_seed.items()):
        cands = [(a, b) for a, b in itertools.combinations(vs, 2) if a["name"] != b["name"] and not hard_conflict(a, b, by_id[sid])][:max_candidates]
        taken = 0
        for a, b in cands:
            if taken >= per_seed or len(jobs) >= n:
                break
            verdict = screen(by_id[sid], a, b) if screen else {"compatible": True, "reason": "unscreened"}
            log.append({"seed_id": sid, "a": a["name"], "b": b["name"], **verdict})
            if not verdict["compatible"]:
                continue
            c = combine(a, b)
            c["eligible_seeds"] = [sid]
            jobs.append({"job_id": f"exp_combo/{sid}__{taken}", "seed_id": sid, "strategy": "analyzer", "variation": c,
                         "mode": "analyzer", "version": "exp_combo"})
            taken += 1
    combo_jobs.last_log = log
    return jobs


def unused_reads(seed: dict) -> list[tuple[str, dict]]:
    """(tool, fixed argument) options the seed's tools offer but the seed never called, e.g. an unused enum value of kind."""
    used = {(c["name"], json.dumps({k: v for k, v in c["arguments"].items() if k == "kind"}, sort_keys=True)) for c in _calls(seed)}
    opts = []
    for t in seed.get("tools", []):
        props = (t.get("parameters") or {}).get("properties") or {}
        kinds = (props.get("kind") or {}).get("enum")
        for k in kinds or [None]:
            arg = {"kind": k} if k else {}
            if (t["name"], json.dumps(arg, sort_keys=True)) not in used:
                opts.append((t["name"], arg))
    return opts


def branch_jobs(dev: list[dict], n: int, exclude: set[str], per_seed: int = 1) -> list[dict]:
    jobs = []
    for s in dev:
        if s["id"][:13] in exclude:
            continue
        for k, (tool, arg) in enumerate(unused_reads(s)[:per_seed]):
            label = tool + (f"[{arg['kind']}]" if arg else "")
            v = {"name": f"branch_extra_call_{tool}{'_' + arg['kind'] if arg else ''}", "applies_to": ["all"], "requires_tools": [],
                 "eligible_seeds": [s["id"]], "basis": "seed", "decision_type": "extra evidence", "answer_transition": "unchanged",
                 "teaches": "ask for one more piece of evidence the seed did not ask for and use it correctly",
                 "change": f"After the seed's first tool batch the assistant also calls {label}, which the seed never called. Write that "
                           f"result only in a result shape shown in the observed contracts, consistent with every observation already in the "
                           f"conversation. Change the final answer only if the new evidence warrants it under the system prompt.",
                 "preserve": "Everything else: the failure context, the seed's own calls and results, the answer format.",
                 "dependencies": "The final explanation must account for the new result; no kept observation may contradict it.",
                 "correct_when": "The new result agrees with the kept evidence and the answer follows the system prompt."}
            jobs.append({"job_id": f"exp_branch/{s['id']}__{k}", "seed_id": s["id"], "strategy": "analyzer", "variation": v,
                         "mode": "analyzer", "version": "exp_branch"})
            if len(jobs) >= n:
                return jobs
    return jobs


def run_experiment(cfg: Config, dev: list[dict], name: str, version: int, n: int, good_path: str | None, exclude: set[str]) -> dict:
    root = cfg.out / "analyzer"
    out = root / f"exp_{name}"
    out.mkdir(parents=True, exist_ok=True)
    if name == "reorder":
        base = [s for s in dev if s["id"][:13] not in exclude]
        base += [t for t in read_jsonl(out.parent / "batch" / "verified.jsonl") + read_jsonl(root / f"v{version}" / "verified.jsonl")
                 if t["messages"] and t["meta"]["verify"]["kept"] and t["meta"].get("seed_id", "")[:13] not in exclude]
        if good_path:  # only traces whose (seed, variation) was judged good; the real seeds are always eligible
            ok = {(g["seed_id"], g["variation"]["name"]) for g in json.loads(Path(good_path).read_text())}
            base = [t for t in base if "verify" not in t.get("meta", {}) or (t["meta"].get("seed_id"), t["meta"].get("variation")) in ok]
        rows = reorder_rows(base, per_trace=2)[:n]
        bad = [(r["id"], mechanical_ok(r)) for r in rows if mechanical_ok(r)]
        (out / "reordered.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        rep = {"rows": len(rows), "mechanical_issues": bad, "sources": len(base), "eligible_sources": sum(independent_calls(t) for t in base)}
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        return rep
    template = (root / f"v{version}" / "prompt.txt").read_text()
    contracts = (root / f"v{version}" / "observed_contracts.md").read_text()
    if name == "combo":
        good = json.loads(Path(good_path).read_text()) if good_path else []
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=max(1, cfg.workers))

        def screen(seed, a, b):
            return pool.submit(screen_pair, cfg, seed, a, b).result()
        jobs = combo_jobs(dev, good, n, exclude, screen=screen)
        pool.shutdown()
        (out / "pair_screen.json").write_text(json.dumps(getattr(combo_jobs, "last_log", []), indent=1, ensure_ascii=False))
    elif name == "branch":
        jobs = branch_jobs(dev, n, exclude)
    else:
        raise SystemExit(f"unknown experiment {name!r}")
    if not jobs:
        raise SystemExit(f"no jobs for experiment {name}")
    (out / "jobs.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
    print(f"experiment {name}: {len(jobs)} jobs")
    dev_by = {s["id"]: s for s in dev}
    obs = ObsBank.from_traces(dev)
    for j in jobs:
        j["case_block"] = cfg.case_block
    return run_arm(cfg, out, jobs, lambda j: generate_analyzer(cfg, j, dev_by[j["seed_id"]], template, obs), dev, set(),
                   observed_contracts=contracts)
