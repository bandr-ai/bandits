"""bandits-sft CLI: ingest -> [analyze -> (heldout)] -> plan -> specs -> generate -> verify -> select (or `run` for all).

Every stage reads and writes JSONL under out_dir and resumes: re-running skips work already done.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
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
from .trace_text import tools_text
from .verify import ngrams, user_text, verify


def _p(cfg: Config, name: str) -> Path:
    return cfg.out / name


AUDIT_PROMPT = """You are checking that raw agent traces were split correctly into a system prompt, tool definitions and a
conversation before they are used as reviewed seeds. Below are, per distinct system prompt, the prompt, the tool
definitions and up to two seeds rendered as messages. Check only what is visible; do not judge the agent's answers. Nothing is truncated for display: text that ends
abruptly really ends there.

{material}

For each prompt report whether: (1) the system prompt is the agent's instructions and contains no conversation, user
data or tool output; (2) the first conversation turn is the user's input, and any material the agent works from (documents,
extracted text, tool-free context) sits where the agent actually receives it; (3) tool calls and tool results pair up
and results are intact (not truncated mid-structure); (4) the answer contract in the system prompt matches the final
reply's form; (5) text shows damage from anonymization or export (names swapped inconsistently, placeholders, cut-off
text). Return JSON only: {{"prompts": [{{"label": "...", "ok": true, "issues": ["<specific, quoted>"]}}], "notes": "..."}}"""


def cmd_audit(cfg: Config) -> None:
    """Advisory LLM check of the ingested seeds' split (system / tools / conversation). Writes ingest_audit.json; never gates."""
    from .analyzer import label_keys, system_labels
    from .verify import system_key
    seeds = read_jsonl(_p(cfg, "seeds.jsonl"))
    if not seeds:
        raise SystemExit("run ingest first")
    parts = []
    for label, keys in system_labels(seeds).items():
        for key in label_keys(keys):
            members = [s for s in seeds if system_key(s.get("system", "")) == key]
            parts.append(f'<prompt label="{label}" seeds="{len(members)}">\n<system>\n{members[0].get("system", "")}\n</system>\n'
                         f'<tools>\n{tools_text(members[0]["tools"]) if members[0]["tools"] else "(none)"}\n</tools>\n'
                         + "".join(f"<seed id=\"{m['id']}\">\n{render(m, max_obs_chars=6000)[:60000]}\n</seed>\n" for m in members[:2])
                         + "</prompt>")
    out = chat_json(get_llm(cfg, "judge"), "You are a careful data auditor. Output JSON only.", AUDIT_PROMPT.format(material="\n".join(parts)))
    _p(cfg, "ingest_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    rows = out.get("prompts", []) if isinstance(out, dict) else []
    print(f"audit: {sum(bool(r.get('ok')) for r in rows)}/{len(rows)} prompts ok; issues: "
          f"{[i for r in rows for i in r.get('issues', [])][:6]}")


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
    seeds = _split(cfg, seeds)
    write_jsonl(_p(cfg, "seeds.jsonl"), seeds)
    obs, users = ObsBank.from_traces(seeds), UserBank.from_traces(seeds)
    write_jsonl(_p(cfg, "obs_bank.jsonl"), obs.entries)
    write_jsonl(_p(cfg, "user_bank.jsonl"), users.entries)
    print(f"ingest: {len(seeds)} seeds, {len(obs.entries)} real tool results, {len(users.entries)} real user turns")
    return seeds


def _split(cfg: Config, seeds: list[dict]) -> list[dict]:
    """Hold out a share of seeds by id (persisted in split.json); returns the dev seeds. Held-out seeds go to
    heldout.jsonl only, so they never reach the banks, the analyzer or generation."""
    if cfg.holdout_frac <= 0:
        return seeds
    path = _p(cfg, "split.json")
    if path.exists():  # a persisted split wins: re-running ingest must not reshuffle
        test = set(json.loads(path.read_text())["heldout"])
    else:
        ids = sorted(s["id"] for s in seeds)
        random.Random(cfg.random_seed).shuffle(ids)
        test = set(ids[:max(1, round(len(ids) * cfg.holdout_frac))])
        path.write_text(json.dumps({"by": "seed id", "random_seed": cfg.random_seed, "holdout_frac": cfg.holdout_frac,
                                    "dev": sorted(set(ids) - test), "heldout": sorted(test)}, indent=1))
    write_jsonl(_p(cfg, "heldout.jsonl"), [s for s in seeds if s["id"] in test])
    dev = [s for s in seeds if s["id"] not in test]
    print(f"split: {len(dev)} dev, {len(seeds) - len(dev)} held out")
    return dev


def cmd_analyze(cfg: Config) -> dict:
    from .analyzer import analyze

    return analyze(cfg, _load_seeds(cfg), read_jsonl(_p(cfg, "heldout.jsonl")), _eval_ngrams(cfg))


def cmd_batch(cfg: Config, version: int, n: int) -> dict:
    from .analyzer import run_batch

    return run_batch(cfg, _load_seeds(cfg), _eval_ngrams(cfg), version, n)


def cmd_experiment(cfg: Config, name: str, version: int, n: int, good: str | None, exclude: str) -> dict:
    from .experiments import run_experiment

    return run_experiment(cfg, _load_seeds(cfg), name, version, n, good, {x for x in exclude.split(",") if x})


def cmd_review(cfg: Config) -> None:
    """Local HTML: every analyzer trace next to its seed, aligned, with changes highlighted and the judge's verdicts."""
    from .review import build_review

    out = build_review(_p(cfg, "analyzer"), _load_seeds(cfg))
    print(f"review: {out}")


def cmd_heldout(cfg: Config) -> dict:
    """Chosen analyzer version vs baseline_prompt=fixed on the held-out seeds. Separate on purpose: looking at held-out
    results turns those seeds into dev data, so this runs only when explicitly asked for."""
    from .analyzer import compare_heldout

    root = _p(cfg, "analyzer")
    if not (root / "chosen.txt").exists():
        raise SystemExit("run `analyze` first")
    heldout = read_jsonl(_p(cfg, "heldout.jsonl"))
    if not heldout:
        raise SystemExit("no held-out seeds (holdout_frac is 0)")
    return compare_heldout(cfg, root, (root / "chosen.txt").read_text(),
                           json.loads((root / "chosen_variations.json").read_text()), heldout, _load_seeds(cfg), _eval_ngrams(cfg))


def cmd_ab(cfg: Config, version: int, jobs: int) -> dict:
    """Generator A/B on identical jobs: analyzer version <version> with the plain output format vs with the case block.
    Paid LLM calls; run only when explicitly asked for."""
    from .analyzer import run_ab

    out = run_ab(cfg, _load_seeds(cfg), _eval_ngrams(cfg), version, jobs)
    print(json.dumps({arm: {k: r[k] for k in ("jobs", "kept", "unique_kept", "cost_usd")} for arm, r in out.items()}))
    return out


def cmd_ab_split(cfg: Config, jobs: int) -> dict:
    """Split-generation A/B on identical jobs (current vs split). Paid LLM calls; run only when explicitly asked for."""
    from .analyzer import run_ab_split

    out = run_ab_split(cfg, _load_seeds(cfg), _eval_ngrams(cfg), jobs)
    print(json.dumps(out, indent=1, default=str))
    return out


def cmd_ab_vary(cfg: Config, jobs: int) -> dict:
    """Variety A/B over the same seeds (current vs vary vs stack). Paid LLM calls; run only when explicitly asked for."""
    from .analyzer import run_ab_vary

    out = run_ab_vary(cfg, _load_seeds(cfg), _eval_ngrams(cfg), jobs)
    print(json.dumps(out, indent=1, default=str))
    return out


def cmd_mixed(cfg: Config, seeds: int) -> dict:
    """Pilot: every variation type per seed, judge keeps the good ones. Paid LLM calls; run only when explicitly asked for."""
    from .analyzer import run_mixed

    out = run_mixed(cfg, _load_seeds(cfg), _eval_ngrams(cfg), seeds)
    print(json.dumps(out, indent=1, default=str))
    return out


def cmd_ab_models(cfg: Config, jobs: int, arms: str) -> dict:
    """Model-settings A/B (e.g. reasoning effort) on identical jobs. Paid LLM calls; run only when explicitly asked for."""
    from .analyzer import run_ab_models

    out = run_ab_models(cfg, _load_seeds(cfg), _eval_ngrams(cfg), jobs, json.loads(Path(arms).read_text()))
    print(json.dumps(out, indent=1, default=str))
    return out


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
    held = [s["id"] for s in seeds if any(s["id"].startswith(q) for q in cfg.quarantine_seeds)]
    if held:
        print(f"quarantined seeds (not used): {held}")
    return [s for s in seeds if s["id"] not in held]


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


def cmd_reparse(cfg: Config) -> list[dict]:
    """Rebuild mode-baseline traces from their saved raw outputs (no LLM calls); clears verify/select outputs."""
    from .generate import trace_from_raw

    seeds = {s["id"]: s for s in _load_seeds(cfg)}
    rows = []
    for t in read_jsonl(_p(cfg, "generated.jsonl")):
        m = t["meta"]
        if m.get("mode") == "baseline" and m.get("raw_output") is not None:
            new = trace_from_raw({"job_id": t["id"]}, seeds[m["seed_id"]], m["raw_output"])
            new["meta"] = {**m, **new["meta"]}
            t = new
        rows.append(t)
    write_jsonl(_p(cfg, "generated.jsonl"), rows)
    for stale in ("verified.jsonl", "verified.errors.jsonl"):
        _p(cfg, stale).unlink(missing_ok=True)
    print(f"reparse: {len(rows)} traces rebuilt from raw output")
    return rows


def cmd_verify(cfg: Config) -> list[dict]:
    gen = read_jsonl(_p(cfg, "generated.jsonl"))
    eval_ng = _eval_ngrams(cfg)
    if cfg.prompt_source == "analyzer":  # seed-expansion traces are judged against their seed and variation
        from .analyzer import expansion_context

        seeds = {s["id"]: s for s in _load_seeds(cfg)}
        observed_contracts = _p(cfg, "analyzer/chosen_observed_contracts.md").read_text()
        return parallel_map(lambda t: verify(cfg, t, eval_ng, expansion_context(t, seeds, observed_contracts)), gen,
                            key=lambda t: t["id"], out_path=_p(cfg, "verified.jsonl"), workers=cfg.workers, desc="verify")
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
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["pipeline"]:  # end-to-end command, own arguments: bandits-sft pipeline --seeds X --out Y --budget 5
        from .pipeline import main as pipeline_main
        sys.exit(pipeline_main(argv[1:]))
    ap = argparse.ArgumentParser(prog="bandits-sft", description=__doc__ + "\nEnd to end: bandits-sft pipeline --help")
    ap.add_argument("command", choices=["ingest", "audit", "analyze", "batch", "experiment", "review", "heldout", "ab", "ab-split", "ab-vary", "mixed", "ab-models", "plan", "specs", "generate", "reparse", "verify",
                                        "select", "run"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--version", type=int, help="ab: analyzer version whose template both arms use")
    ap.add_argument("--jobs", type=int, default=20, help="ab: jobs per arm; mixed: seeds")
    ap.add_argument("--arms", help="ab-models: JSON file {arm: {role: {setting: value}}}")
    ap.add_argument("--name", help="experiment: reorder | combo | branch")
    ap.add_argument("--good", help="experiment combo: JSON list of [{seed_id, variation}] cells already judged good")
    ap.add_argument("--exclude", default="", help="experiment: comma-separated seed id prefixes to skip")
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
        if args.command == "ab-split":
            cmd_ab_split(cfg, args.jobs)
        elif args.command == "ab-vary":
            cmd_ab_vary(cfg, args.jobs)
        elif args.command == "mixed":
            cmd_mixed(cfg, args.jobs)
        elif args.command == "ab-models":
            if not args.arms:
                raise SystemExit("ab-models needs --arms <json file>")
            cmd_ab_models(cfg, args.jobs, args.arms)
        elif args.command == "ab":
            if not args.version:
                raise SystemExit("ab needs --version")
            cmd_ab(cfg, args.version, args.jobs)
        elif args.command == "batch":
            if not args.version:
                raise SystemExit("batch needs --version (the analyzer version whose template to freeze)")
            cmd_batch(cfg, args.version, args.jobs)
        elif args.command == "experiment":
            if not (args.name and args.version):
                raise SystemExit("experiment needs --name and --version (the analyzer version whose template to freeze)")
            cmd_experiment(cfg, args.name, args.version, args.jobs, args.good, args.exclude)
        elif args.command == "run":
            cmd_ingest(cfg)
            cmd_generate(cfg)
            cmd_verify(cfg)
            cmd_select(cfg)
        else:
            {"ingest": cmd_ingest, "audit": cmd_audit, "analyze": cmd_analyze, "review": cmd_review, "heldout": cmd_heldout, "plan": cmd_plan, "specs": cmd_specs, "generate": cmd_generate,
             "reparse": cmd_reparse, "verify": cmd_verify, "select": cmd_select}[args.command](cfg)
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
