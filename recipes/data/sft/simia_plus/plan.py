"""Turn (seeds, target_count) into generation jobs, and write scenario specs for them.

Every seed is used: jobs are spread round-robin so each seed gets ceil or floor of target*overgen/len(seeds).
Constraints (persona, failure) are attached in code *after* the strategy/spec is chosen, because adding
constraints at generation time cuts diversity (Adaption "Invent": -15-22%).
"""
from __future__ import annotations

import json
import math
import random

from .config import Config
from .io import parallel_map
from .llm import chat_json, get_llm
from .personas import sample_persona
from .prompts import SPEC_BATCH, STRATEGIES
from .schema import tool_calls
from .simia_text import build_sample_text, tools_text

FAILURE_TYPES = ["timeout", "server_error", "not_found", "permission_denied", "invalid_argument"]
SPEC_BATCH_MAX = 8  # specs per LLM call; larger batches drift


def is_conversational(trace: dict) -> bool:
    return sum(1 for m in trace["messages"] if m["role"] == "user") >= 2


def plan_jobs(cfg: Config, seeds: list[dict]) -> list[dict]:
    if not seeds:
        raise ValueError("no seeds")
    rng = random.Random(cfg.random_seed)
    f = cfg.features
    n_jobs = max(len(seeds), math.ceil(cfg.target_count * cfg.overgen))
    by_tools: dict[str, list[str]] = {}
    for s in seeds:
        by_tools.setdefault(json.dumps(sorted(t["name"] for t in s["tools"])), []).append(s["id"])
    strategies = list(cfg.strategy_weights)
    jobs = []
    for j in range(n_jobs):
        seed = seeds[j % len(seeds)]
        strategy = (rng.choices(strategies, weights=[cfg.strategy_weights[s] for s in strategies])[0]
                    if f.strategies else "new_scenario")
        job = {"job_id": f"{seed['id']}__{j // len(seeds)}", "seed_id": seed["id"], "strategy": strategy,
               "mode": (cfg.loop_style if f.loop and rng.random() < cfg.loop_frac else "simia")}
        if strategy == "compose":
            peers = [p for p in by_tools[json.dumps(sorted(t["name"] for t in seed["tools"]))] if p != seed["id"]]
            if peers:
                job["second_seed_id"] = rng.choice(peers)
            else:
                job["strategy"] = "extend"
        if f.persona and (cfg.persona_all_seeds or is_conversational(seed)):
            job["persona"] = sample_persona(rng, cfg.persona_weights)
        if f.failure and rng.random() < cfg.failure_rate:
            n_calls = max(1, len(tool_calls(seed)))
            job["failure"] = {"at_call": rng.randint(1, n_calls), "type": rng.choice(FAILURE_TYPES)}
        jobs.append(job)
    return jobs


def write_specs(cfg: Config, jobs: list[dict], seeds_by_id: dict[str, dict], out_path) -> list[dict]:
    """One batched LLM call per (seed, chunk of jobs); returns jobs with a 'spec' attached."""
    groups: dict[str, list[dict]] = {}
    for job in jobs:
        groups.setdefault(job["seed_id"], []).append(job)
    batches = []
    for seed_id, js in groups.items():
        for i in range(0, len(js), SPEC_BATCH_MAX):
            batches.append({"batch_id": f"{seed_id}#{i // SPEC_BATCH_MAX}", "seed_id": seed_id,
                            "jobs": js[i:i + SPEC_BATCH_MAX]})
    llm = get_llm(cfg, "spec")

    def run(batch: dict) -> list[dict]:
        seed = seeds_by_id[batch["seed_id"]]
        js = batch["jobs"]
        second = next((j["second_seed_id"] for j in js if j.get("second_seed_id")), None)
        second_text = (f"\nSECOND EXAMPLE (for 'compose'):\n{build_sample_text(seeds_by_id[second], include_system=False)}\n"
                       if second else "")
        strategy_list = "\n".join(f"{k + 1}. {j['strategy']}: {STRATEGIES[j['strategy']]}" for k, j in enumerate(js))
        avoid = "a copy of the example task" + ("" if not any(j["strategy"] == "rephrase" for j in js)
                                                 else " (except 'rephrase', which keeps the task)")
        out = chat_json(llm, "You write precise, realistic JSON scenario specs.", SPEC_BATCH.format(
            n=len(js), avoid=avoid, system=seed.get("system", ""), tools=tools_text(seed["tools"]),
            sample_text=build_sample_text(seed, include_system=False), second_example=second_text,
            strategy_list=strategy_list))
        specs = out.get("specs", []) if isinstance(out, dict) else out
        if len(specs) < len(js):
            raise ValueError(f"spec batch returned {len(specs)} specs for {len(js)} jobs")
        rows = []
        for job, spec in zip(js, specs):
            spec = {k: spec.get(k) for k in ("goal", "user_facts", "initial_state", "expected_final_state",
                                              "expected_behavior", "difficulty")}
            spec["initial_state"] = spec["initial_state"] or {}
            spec["expected_final_state"] = spec["expected_final_state"] or {}
            rows.append({**job, "spec": spec, "batch_id": batch["batch_id"]})
        return rows

    rows = parallel_map(run, batches, key=lambda b: b["batch_id"], out_path=out_path, workers=cfg.workers,
                        done_key=lambda r: r["batch_id"], desc="specs")
    return rows
