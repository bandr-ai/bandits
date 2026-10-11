"""Compare real seeds, earlier Simia runs and analyzer versions on the same code-level metrics.

Every set is re-checked with the current code checks (rule_check, answer_check, provenance), so older runs are
scored by today's rules. Judge results exist only for analyzer runs and are reported separately.
usage (from recipes/data/sft, with the recipe venv): python <this file> <pilot dir> [<old simia runs dir>] > comparison.json
The pilot dir holds answer_schemas.json and out/ (here: fde-work/sft-analyzer-pilot).
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from simia_plus.select import action_signature, exact_key
from simia_plus.verify import answer_check, final_answer, provenance_check, rule_check

PD = Path(sys.argv[1]).resolve()
OLD = Path(sys.argv[2]) if len(sys.argv) > 2 else PD / "old_runs"
SCHEMAS = json.loads((PD / "answer_schemas.json").read_text())
CAUSE = re.compile(r"\bNSEM?_[A-Z_]+\b")


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


seeds = load(PD / "out/seeds.jsonl") + load(PD / "out/heldout.jsonl")
seeds_by_id = {s["id"]: s for s in seeds}
seed_obs = {m["content"] for s in seeds for m in s["messages"] if m["role"] == "tool"}
conclusions = {v for sch in SCHEMAS.values() for v in (sch["properties"].get("conclusion", {}).get("enum") or []) if v}


def grams(text: str, n: int = 3) -> set:
    t = re.findall(r"\w+", text.lower())
    return {tuple(t[i:i + n]) for i in range(max(1, len(t) - n + 1))}


def jacc(a: str, b: str) -> float:
    x, y = grams(a), grams(b)
    return len(x & y) / max(1, len(x | y))


def first_user(t: dict) -> str:
    return next((m["content"] for m in t["messages"] if m["role"] == "user"), "")


def metrics(name: str, traces: list[dict], is_seed: bool = False) -> dict:
    n = len(traces)
    code_ok, schema_ok, bare_json, keys, outs, causes, confs = 0, 0, 0, set(), Counter(), Counter(), []
    calls, obs_len, copied, unavailable, leaks, sim_seed, judged, judge_ok = [], [], 0, 0, 0, [], 0, 0
    n_obs = 0
    for t in traces:
        issues = rule_check(t, 2) + answer_check(t, SCHEMAS) + [f"ungrounded {a}" for a in provenance_check(t, None)]
        code_ok += not issues
        ans = final_answer(t)
        if isinstance(ans, dict):
            bare_json += 1
            schema_ok += not answer_check(t, SCHEMAS)
            outs[str(ans.get("conclusion", "(none)"))] += 1
            causes["null" if ans.get("root_cause") is None else "named"] += 1
            if isinstance(ans.get("confidence"), (int, float)):
                confs.append(ans["confidence"])
        if not issues:
            keys.add(exact_key(t))
        calls.append(sum(len(m.get("tool_calls", [])) for m in t["messages"]))
        for m in t["messages"]:
            if m["role"] != "tool":
                continue
            n_obs += 1
            c = m["content"]
            obs_len.append(len(c))
            copied += (not is_seed) and c in seed_obs
            unavailable += '"available": false' in c or '"available":false' in c
            # a tool naming a catalog cause or an inspector conclusion = the tool hands over the verdict
            leaks += bool(CAUSE.search(c)) or any(re.search(rf"\b{re.escape(k)}\b", c) for k in conclusions if "_" in k)
        seed = seeds_by_id.get(t.get("meta", {}).get("seed_id"))
        if seed and not is_seed:
            sim_seed.append(jacc(first_user(t), first_user(seed)))
        j = (t.get("meta", {}).get("verify") or {}).get("judge")
        if j is not None:
            judged += 1
            judge_ok += t["meta"]["verify"]["kept"]
    sigs = Counter(action_signature(t) for t in traces)
    r = lambda x: round(x, 3)
    return {
        "set": name, "n": n,
        "code_pass": f"{code_ok}/{n}", "unique_code_pass": len(keys),
        "final_is_bare_json": f"{bare_json}/{n}", "answer_schema_ok": f"{schema_ok}/{bare_json}",
        "judge_kept": f"{judge_ok}/{judged}" if judged else "-",
        "root_cause_null_share": r(causes["null"] / max(1, sum(causes.values()))),
        "conclusions": dict(outs.most_common()),
        "confidence_mean": r(statistics.mean(confs)) if confs else None,
        "tool_calls_mean": r(statistics.mean(calls)) if calls else 0, "tool_calls_max": max(calls, default=0),
        "distinct_tool_sequences": len(sigs),
        "tool_result_chars_median": int(statistics.median(obs_len)) if obs_len else 0,
        "tool_results_unavailable_share": r(unavailable / max(1, n_obs)),
        "tool_results_naming_a_verdict_share": r(leaks / max(1, n_obs)),
        "tool_results_copied_from_seeds_share": r(copied / max(1, n_obs)) if not is_seed else None,
        "user_msg_3gram_jaccard_vs_own_seed": r(statistics.mean(sim_seed)) if sim_seed else None,
    }


sets = [("real seeds (17)", seeds, True),
        ("simia plain (old run)", load(OLD / "out_simia2/generated.jsonl"), False),
        ("simia fixed (old run)", load(OLD / "out_simia_fixed2/generated.jsonl"), False)]
for v in ("v1", "v2", "v3"):
    sets.append((f"analyzer {v}", load(PD / f"out/analyzer/{v}/verified.jsonl"), False))
for arm in ("analyzer", "simia_fixed"):
    sets.append((f"held-out {arm}", load(PD / f"out/analyzer/heldout/{arm}/verified.jsonl"), False))
print(json.dumps([metrics(*s) for s in sets if s[1]], indent=1))
