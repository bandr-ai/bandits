"""simia-plus CLI: ingest -> plan -> specs -> generate -> verify -> select (or `run` for all).

Every stage reads and writes JSONL under out_dir and resumes: re-running skips work already done.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path

from .banks import ObsBank, UserBank
from .config import Config, load_config
from .generate import generate
from .io import parallel_map, read_jsonl, write_jsonl
from .llm import chat_json, configure_call_log, get_llm
from .plan import plan_jobs, write_specs
from .prompts import SEED_CHECK
from .schema import load_trace, render, to_sharegpt
from .select import dedup, diversity_report, select
from .verify import ngrams, user_text, verify


def _p(cfg: Config, name: str) -> Path:
    return cfg.out / name


def cmd_ingest(cfg: Config) -> list[dict]:
    defs = None
    if cfg.tool_defs_path:
        raw_defs = json.loads(Path(cfg.tool_defs_path).read_text())
        items = raw_defs.values() if isinstance(raw_defs, dict) else raw_defs
        defs = {(t.get("function") or t)["name"]: t for t in items}
    seeds = [load_trace(r, i, cfg.seed_format, defs) for i, r in enumerate(read_jsonl(cfg.seeds_path))]
    if not seeds:
        raise SystemExit(f"no seeds in {cfg.seeds_path}")
    counts: dict[str, int] = {}
    for s in seeds:  # several traces can share a session id; keep them all, with distinct ids
        n = counts.get(s["id"], 0)
        counts[s["id"]] = n + 1
        if n:
            s["id"] = f"{s['id']}#{n}"
    eval_ng = _eval_ngrams(cfg)
    if eval_ng:  # decontaminate before synthesis, so generation cannot launder eval items (Datology)
        before = len(seeds)
        seeds = [s for s in seeds if not (ngrams(user_text(s)) & eval_ng)]
        print(f"decontam: dropped {before - len(seeds)} seeds overlapping eval traces")
    if cfg.features.check_seeds:
        llm = get_llm(cfg, "seed_check")

        def check(seed: dict) -> dict:
            r = chat_json(llm, "You check training data quality. Output JSON only.",
                          SEED_CHECK.format(system=seed.get("system", ""), transcript=render(seed)))
            return {"id": seed["id"], **r}

        res = {r["id"]: r for r in parallel_map(check, seeds, key=lambda s: s["id"],
                                                out_path=_p(cfg, "seed_check.jsonl"), workers=cfg.workers, desc="seed check")}
        ok = [s for s in seeds if all(res.get(s["id"], {}).get(k) for k in ("complete", "logical", "well_formatted"))]
        print(f"seed check: kept {len(ok)}/{len(seeds)}")
        seeds = ok
    write_jsonl(_p(cfg, "seeds.jsonl"), seeds)
    obs, users = ObsBank.from_traces(seeds), UserBank.from_traces(seeds)
    write_jsonl(_p(cfg, "obs_bank.jsonl"), obs.entries)
    write_jsonl(_p(cfg, "user_bank.jsonl"), users.entries)
    print(f"ingest: {len(seeds)} seeds, {len(obs.entries)} real tool results, {len(users.entries)} real user turns")
    return seeds


def _eval_ngrams(cfg: Config) -> set:
    ng: set = set()
    for path in cfg.decontam_paths:
        for i, r in enumerate(read_jsonl(path)):
            ng |= ngrams(user_text(load_trace(r, i, "auto")))
    return ng


def _load_seeds(cfg: Config) -> list[dict]:
    seeds = read_jsonl(_p(cfg, "seeds.jsonl"))
    if not seeds:
        raise SystemExit("run `ingest` first")
    return seeds


def cmd_plan(cfg: Config) -> list[dict]:
    path = _p(cfg, "jobs.jsonl")
    jobs = read_jsonl(path)
    if not jobs:
        jobs = plan_jobs(cfg, _load_seeds(cfg))
        write_jsonl(path, jobs)
    print(f"plan: {len(jobs)} jobs for target {cfg.target_count} (overgen {cfg.overgen})")
    return jobs


def cmd_specs(cfg: Config) -> list[dict]:
    jobs = cmd_plan(cfg)
    if not cfg.features.spec:
        return jobs
    seeds = {s["id"]: s for s in _load_seeds(cfg)}
    rows = write_specs(cfg, jobs, seeds, _p(cfg, "specs.jsonl"))
    by_id = {r["job_id"]: r for r in rows}
    return [by_id[j["job_id"]] for j in jobs if j["job_id"] in by_id]


def cmd_generate(cfg: Config) -> list[dict]:
    jobs = cmd_specs(cfg)
    seeds = _load_seeds(cfg)
    seeds_by_id = {s["id"]: s for s in seeds}
    obs, users = ObsBank.from_traces(seeds), UserBank.from_traces(seeds)
    return parallel_map(lambda j: generate(cfg, j, seeds_by_id, obs, users), jobs, key=lambda j: j["job_id"],
                        out_path=_p(cfg, "generated.jsonl"), workers=cfg.workers,
                        done_key=lambda t: t["id"], desc="generate")


def cmd_verify(cfg: Config) -> list[dict]:
    gen = read_jsonl(_p(cfg, "generated.jsonl"))
    eval_ng = _eval_ngrams(cfg)
    return parallel_map(lambda t: verify(cfg, t, eval_ng), gen, key=lambda t: t["id"],
                        out_path=_p(cfg, "verified.jsonl"), workers=cfg.workers, desc="verify")


def cmd_select(cfg: Config) -> dict:
    verified = read_jsonl(_p(cfg, "verified.jsonl"))
    kept = [t for t in verified if t["meta"]["verify"]["kept"]]
    deduped, dup_stats = dedup(kept, near=cfg.features.near_dedup)
    final = select(deduped, cfg.target_count)
    seeds = _load_seeds(cfg)
    write_jsonl(_p(cfg, "final/synthetic.jsonl"), final)
    write_jsonl(_p(cfg, "final/synthetic_sharegpt.jsonl"), (to_sharegpt(t) for t in final))
    write_jsonl(_p(cfg, "final/seeds_plus_synthetic_sharegpt.jsonl"),
                (to_sharegpt(t) for t in [*seeds, *final]))  # accumulate: seeds always stay in the mix
    reasons: dict[str, int] = {}
    for t in verified:
        v = t["meta"]["verify"]
        for r in v["rule_issues"]:
            key = r.split(" ")[0] if r.startswith(("call", "unknown")) else r
            reasons[key] = reasons.get(key, 0) + 1
        if v.get("contaminated"):
            reasons["contaminated"] = reasons.get("contaminated", 0) + 1
        if v.get("ungrounded_args"):
            reasons["ungrounded_args"] = reasons.get("ungrounded_args", 0) + 1
        j = v.get("judge") or {}
        if j and not v["kept"]:
            reasons["judge_rejected"] = reasons.get("judge_rejected", 0) + 1
    report = {"target": cfg.target_count, "generated": len(read_jsonl(_p(cfg, "generated.jsonl"))),
              "verified": len(verified), "kept": len(kept), "after_dedup": len(deduped), "final": len(final),
              "short_of_target": max(0, cfg.target_count - len(final)), "dedup": dup_stats,
              "reject_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
              "features": vars(cfg.features), "diversity": diversity_report(final),
              "llm": _llm_summary(cfg),
              "seed_diversity": diversity_report([{**s, "meta": {"seed_id": s["id"]}} for s in seeds])}
    _p(cfg, "final").mkdir(parents=True, exist_ok=True)
    _p(cfg, "final/report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("target", "generated", "kept", "final", "short_of_target")}))
    return report


def _llm_summary(cfg: Config) -> dict:
    calls = _call_rows(cfg)
    by_role: dict[str, dict] = {}
    for c in calls:
        r = by_role.setdefault(c.get("role") or "?", {"ok": 0, "failed_attempts": 0, "cost_usd": 0.0, "completion_tokens": 0})
        r["ok" if c.get("ok") else "failed_attempts"] += 1
        r["cost_usd"] = round(r["cost_usd"] + (c.get("cost") or 0), 6)
        r["completion_tokens"] += ((c.get("usage") or {}).get("completion_tokens") or 0)
    return {"calls": len(calls), "cost_usd": round(sum(c.get("cost") or 0 for c in calls), 6), "by_role": by_role,
            "providers": dict(Counter(str(c.get("provider")) for c in calls if c.get("ok")))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="simia-plus", description=__doc__)
    ap.add_argument("command", choices=["ingest", "plan", "specs", "generate", "verify", "select", "run"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg.out.mkdir(parents=True, exist_ok=True)
    configure_call_log(_p(cfg, "llm_calls.jsonl"))
    run = {"run_id": datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6],
           "command": args.command, "config_path": str(Path(args.config).resolve()),
           "config": json.loads(Path(args.config).read_text()), "started": _now(), **_provenance()}
    calls_before = _call_rows(cfg)
    status = "ok"
    try:
        if args.command == "run":
            cmd_ingest(cfg)
            cmd_generate(cfg)
            cmd_verify(cfg)
            cmd_select(cfg)
        else:
            {"ingest": cmd_ingest, "plan": cmd_plan, "specs": cmd_specs, "generate": cmd_generate,
             "verify": cmd_verify, "select": cmd_select}[args.command](cfg)
    except BaseException as e:
        status = f"error: {type(e).__name__}: {e}"
        raise
    finally:
        calls = _call_rows(cfg)[len(calls_before):]
        run.update(finished=_now(), status=status, llm_calls=len(calls), llm_failed_attempts=sum(not c.get("ok") for c in calls),
                   cost_usd=round(sum(c.get("cost") or 0 for c in calls), 6),
                   providers=dict(Counter(str(c.get("provider")) for c in calls if c.get("ok"))),
                   files={p.name: sum(1 for _ in p.open()) for p in sorted(cfg.out.glob("*.jsonl"))})
        with _p(cfg, "runs.jsonl").open("a") as f:
            f.write(json.dumps(run, default=str) + "\n")
        print(f"run {run['run_id']}: {status}, {run['llm_calls']} LLM calls, ${run['cost_usd']}")


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _call_rows(cfg: Config) -> list[dict]:
    return read_jsonl(_p(cfg, "llm_calls.jsonl"))


def _provenance() -> dict:
    """Code version for the run manifest (git commit + dirty flag when available)."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=False, cwd=here, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(here)], check=False, cwd=here, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    import openai
    return {"git_commit": commit or None, "git_dirty": dirty, "python": sys.version.split()[0], "openai": openai.__version__}


if __name__ == "__main__":
    main()
