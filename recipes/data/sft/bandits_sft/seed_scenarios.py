"""Seeds-only orchestration: existing automatic setup/expansion + new varied scenarios -> dataset.

python -m bandits_sft.seed_scenarios --seeds seeds.jsonl --out work/dataset --target 100 --budget 2

No customer rubric, contracts, labels, model predictions, environment or evaluation files required.
All setup and both generation branches share one metered lifetime budget. No training commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path

from . import scenario_prompts
from .analyzer import basis_errors, seed_json
from .config import load_config
from .io import read_jsonl, write_jsonl
from .llm import Budget, BudgetExhausted, chat_json, configure_call_log, current_item, get_llm, set_budget
from .pipeline import DEFAULT_CONFIG, Pipeline, Stop
from .scenarios import DEFAULTS, append, export, run_job
from .select import dedup, select
from .verify import final_answer, observed_loose_type_args, observed_optional_args


def freeze(out: Path, seeds: Path, config: Path, target: int, share: float, protected: tuple = ()) -> None:
    """Refuse a resume whose task inputs changed (seed content, config). Code fixes, target and lane share are
    operational and may change on resume: each launch's code hashes and settings are appended to code_versions."""
    source = Path(__file__).parent
    code = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in
            ("seed_scenarios.py", "scenario_prompts.py", "scenarios.py", "pipeline.py", "analyzer.py", "verify.py",
             "rubric_judge.py", "consistency.py", "llm.py")}
    settings = {"seed_path": str(seeds.resolve()),
                "hashes": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (seeds, config, *protected)}}
    code = {**code, "target": target, "scenario_share": share}
    path = out / "seed_only_inputs.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved:
        # Older runs also hashed code under "hashes": compare only the input files.
        old = {k: v for k, v in saved.items() if k not in ("code_versions", "target", "scenario_share")}
        old["hashes"] = {k: v for k, v in old.get("hashes", {}).items() if k in settings["hashes"]}
        if old != settings:
            raise ValueError("inputs changed; use a fresh output directory")
    versions = saved.get("code_versions", [])
    if not versions or versions[-1] != code:
        versions.append(code)
    path.write_text(json.dumps({**settings, "code_versions": versions}, indent=2))


def _capability_errors(c, names: set, ids: set, basis_text: str) -> list[str]:
    if not isinstance(c, dict) or not isinstance(c.get("name"), str) or c["name"] in names:
        return ["needs a unique name"]
    if not isinstance(c.get("capability"), str) or not c["capability"].strip():
        return [f"{c['name']}: capability must describe an evidence/decision gap"]
    if not isinstance(c.get("seed_ids"), list) or not c["seed_ids"] or set(c["seed_ids"]) - ids:
        return [f"{c['name']}: seed_ids must be supplied seed ids"]
    if basis_errors(c, basis_text):
        return [(f"{c['name']}: basis must be \"seed\" or words copied verbatim from the contracts' Explicit/Observed "
                 "part or a system prompt (no paraphrase or prefix)")]
    return []


def catalogue(cfg, seeds: list[dict], contracts: str, out_dir: Path) -> list[dict]:
    """LLM coverage catalogue. Each raw answer is saved before checking (resume never pays twice); invalid items are
    dropped one by one after one corrective retry, and kept in coverage_rejected.json."""
    path = out_dir / "coverage.json"
    if path.exists():
        return json.loads(path.read_text())
    current_item.set("scenarios/coverage")
    # Embedded instructions are seed data: a capability may cite them as well as the derived contracts.
    known = re.split(r"(?im)^#+\s*unknown\b", contracts, maxsplit=1)[0]
    basis_text = known + "\n" + "\n".join(sorted({s["system"] for s in seeds if s.get("system")}))
    request = json.dumps({"contracts": contracts, "seeds": [{"id": s["id"], "system": s["system"], "tools": s["tools"],
                                                             "trace": json.loads(seed_json(s))} for s in seeds]},
                         ensure_ascii=False)
    ids = {s["id"] for s in seeds}
    msg = request
    for attempt in range(2):
        raw_path = out_dir / f"coverage_raw_{attempt}.json"
        if raw_path.exists():
            out = json.loads(raw_path.read_text())
        else:
            out = chat_json(get_llm(cfg, "analyzer"), scenario_prompts.COVERAGE, msg)
            raw_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        capabilities = out.get("capabilities") if isinstance(out, dict) else None
        if not isinstance(capabilities, list):
            raise TypeError("coverage catalogue needs capabilities list")
        kept, rejected, names = [], [], set()
        for c in capabilities:
            errs = _capability_errors(c, names, ids, basis_text)
            if errs:
                rejected.append({"item": c, "errors": errs})
            else:
                names.add(c["name"])
                kept.append(c)
        if not rejected:
            break
        msg = (request + "\n\nYour previous catalogue had these problems; return the full corrected catalogue, "
               "fixing or omitting those items:\n" + "\n".join(e for r in rejected for e in r["errors"]))
    (out_dir / "coverage_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=1))
    path.write_text(json.dumps(kept, ensure_ascii=False, indent=2))
    return kept


def stop_text(seeds: list[dict]) -> str:
    """The most common later user turn repeated across seeds (e.g. a harness wrap-up nudge), or ''."""
    counts = Counter(t for s in seeds for t in {(m.get("content") or "").strip() for m in s["messages"][1:]
                                                  if m["role"] == "user"} - {""})
    return counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1] >= 2 else ""


def stop_rate(seeds: list[dict]) -> float:
    """Share of seeds carrying the most common repeated later user turn (e.g. a harness wrap-up nudge). Measured,
    not assumed; 0 when no later user turn repeats across seeds."""
    later = [{(m.get("content") or "").strip() for m in s["messages"][1:] if m["role"] == "user"} - {""} for s in seeds]
    counts = Counter(t for texts in later for t in texts)
    if not counts or counts.most_common(1)[0][1] < 2:
        return 0.0
    text = counts.most_common(1)[0][0]
    return sum(text in texts for texts in later) / len(seeds)


def outcome(seed: dict) -> dict:
    """Short categorical fields of a seed's final JSON answer (labels, not scores or prose); {} for free text."""
    ans = final_answer(seed)
    if not isinstance(ans, dict):
        return {}
    return {k: v for k, v in ans.items() if isinstance(v, bool) or v is None or isinstance(v, str) and len(v) <= 60}


def scenario_jobs(seeds: list[dict], capabilities: list[dict], count: int, rng_seed: int, nudge: float = 0.0) -> list[dict]:
    """Interleaved coverage-gap (treatment) and broad (control) jobs from the same generator and checks.
    Control keeps a uniformly drawn seed's outcome in a new situation, so its label mix matches the seeds'."""
    rng = random.Random(rng_seed)
    by_id = {s["id"]: s for s in seeds}
    jobs = []
    for i in range(count):
        gap = bool(capabilities) and i % 2 == 0
        c = capabilities[(i // 2) % len(capabilities)] if gap else None
        sid = rng.choice(c["seed_ids"] if c else [s["id"] for s in seeds])
        jobs.append({"job_id": f"scenarios/{i:04d}", "cell": "scenario_gap" if gap else "scenario_broad",
                     "seed_id": sid, "dimensions": {**DEFAULTS, "targeting": "coverage"}, "pair": False,
                     "target": {}, "failure_id": None, "stop_nudge": rng.random() < nudge,
                     "allocation": "coverage_gap" if gap else "broad",
                     "capability": c["capability"] if c else None,
                     "outcome": None if gap else outcome(by_id[sid]) or None})
    return jobs


def answer_stats(traces: list[dict]) -> dict:
    """Per allocation: distribution of short string fields and mean of numeric fields in final JSON answers."""
    out = {}
    for t in traces:
        try:
            ans = json.loads(t["messages"][-1].get("content") or "")
        except ValueError:
            continue
        if not isinstance(ans, dict):
            continue
        row = out.setdefault(t["meta"].get("allocation") or "controlled", {"n": 0, "values": {}, "means": {}})
        row["n"] += 1
        for k, v in ans.items():
            if isinstance(v, str) and len(v) <= 60 or v is None:
                row["values"].setdefault(k, Counter())[str(v)] += 1
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                row["means"].setdefault(k, []).append(v)
    for row in out.values():
        row["values"] = {k: dict(c.most_common(15)) for k, c in row["values"].items() if len(c) <= 30}
        row["means"] = {k: round(sum(v) / len(v), 3) for k, v in row["means"].items()}
    return out


def package(p: Pipeline, scenario_out: Path, controlled_count: int, novel_count: int, status: str, budget: float) -> dict:
    cfg = p.cfg
    controlled_pool = p._unique([*p._pilot_unique(), *read_jsonl(p.prod / "verified.jsonl")])
    controlled = select(controlled_pool, controlled_count) if controlled_count else []
    for t in controlled:
        t["meta"].update(lane="controlled", allocation="existing_variations")
    novel = read_jsonl(scenario_out / "synthetic.jsonl")[:novel_count]
    # Shared formats and existing exact/near dedup across both pools; no raw seed rows are appended.
    combined, duplicate_counts = dedup([*controlled, *novel], near=cfg.features.near_dedup)
    fdir = cfg.out / "final"
    write_jsonl(fdir / "synthetic.jsonl", combined)
    lane_counts = dict(Counter(t["meta"]["lane"] for t in combined))
    spent = p.spend.update()
    report = {"status": status, "requested": controlled_count + novel_count, "accepted": len(combined),
              "short_of_target": controlled_count + novel_count - len(combined), "by_lane": lane_counts,
              "controlled_candidates": len(controlled_pool), "scenario": json.loads((scenario_out / "report.json").read_text())
              if (scenario_out / "report.json").exists() else {}, "duplicates_across_pools": duplicate_counts,
              "cost_usd": round(spent, 6), "lifetime_budget_usd": budget,
              "external_eval_decontamination": bool(cfg.decontam_paths), "human_verified": False,
              "answers_by_allocation": answer_stats(combined),
              "stop_nudge_share": round(sum(any(m["role"] == "user" for m in t["messages"][1:]) for t in combined)
                                        / max(1, len(combined)), 3)}
    manifest = {"input": json.loads((cfg.out / "seed_only_inputs.json").read_text()),
                "rubric": cfg.rubric_path, "trace_ids": [t["id"] for t in combined],
                "by_lane": lane_counts, "allocation": dict(Counter(t["meta"].get("allocation", "broad") for t in combined)),
                "cost_usd": round(spent, 6), "synthetic_only": True,
                "limits": ["LLM validated, not human verified", "Coverage gaps are not measured student failures",
                           "No downstream training benefit measured", "No external eval decontamination unless configured"]}
    (fdir / "report.json").write_text(json.dumps(report, indent=2))
    (fdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (fdir / "HANDOFF.md").write_text(
        f"# Synthetic dataset handoff\n\n"
        f"{len(combined)} accepted synthetic traces, from {len(p._seeds()) if (cfg.out / 'seeds.jsonl').exists() else 0} "
        f"normalized reference seeds. Lanes: {lane_counts}. Logged spend: ${spent:.4f}.\n\n"
        "`synthetic.jsonl` rows contain id, system, tools, messages and meta. Messages retain the seed-compatible "
        "user/assistant/tool format; meta records provenance and checks. No original seed rows are appended.\n\n"
        "Setup, rubric, contracts and coverage plans were derived automatically from the supplied seeds. "
        "Controlled examples use the existing generator. New scenarios use separate planning/generation prompts, "
        "then the existing rubric and consistency gates plus dimension checks.\n\n"
        "These are machine-validated examples, not guaranteed-correct labels. Novelty is relative to references. "
        "Coverage targeting does not measure student errors. No training/evaluation improvement is claimed. "
        "See manifest.json and report.json for counts, hashes, partial-run status and limitations.\n")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--scenario-share", type=float, default=0.5, help="1 = new scenarios only (no controlled lane)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="optional model/settings config; not extra task data")
    ap.add_argument("--decontam", nargs="*", default=[], metavar="JSONL",
                    help="eval/test traces never to copy: seeds and outputs overlapping them are dropped (read by code only)")
    ap.add_argument("--dry-run", action="store_true", help="offline ingestion and input checks only; no credentials or API calls")
    a = ap.parse_args(argv)
    if not a.budget > 0 or a.target < 1 or not 0 < a.scenario_share <= 1 or (a.scenario_share < 1 and a.target < 2):
        raise ValueError("budget must be >0, target >=1 (>=2 with both lanes), and 0 < scenario-share <= 1")
    only_scenarios = a.scenario_share == 1
    cfg = load_config(a.config)
    # Supplied labels/rubrics are not needed: derive task setup anew for this output directory.
    if cfg.rubric_path or cfg.answer_schemas_path:
        raise ValueError("seed-only mode derives its rubric/contracts; omit supplied rubric/answer schemas")
    if cfg.features.judge or cfg.features.split_answer or not cfg.features.rubric_judge or not cfg.features.consistency_check:
        raise ValueError("seed-only mode needs rubric+consistency gates, old judge/split_answer off")
    cfg.seeds_path, cfg.out_dir, cfg.prompt_source = a.seeds, a.out, "analyzer"
    cfg.decontam_paths = [*cfg.decontam_paths, *a.decontam]
    novel_count = a.target if only_scenarios else min(a.target - 1, max(1, round(a.target * a.scenario_share)))
    controlled_count = a.target - novel_count
    cfg.out.mkdir(parents=True, exist_ok=True)
    freeze(cfg.out, Path(a.seeds), Path(a.config), a.target, a.scenario_share, tuple(Path(x) for x in cfg.decontam_paths))
    configure_call_log(cfg.out / "llm_calls.jsonl")
    if only_scenarios:
        cfg.per_seed_variations = False  # per-seed proposals feed only the controlled lane
    p = Pipeline(cfg, a.budget, controlled_count)
    if p.state.get("rubric_path"):
        cfg.rubric_path = p.state["rubric_path"]
    scenario_out = cfg.out / "scenarios"
    scenario_out.mkdir(exist_ok=True)
    print(f"QUOTE: ${a.budget:.2f} lifetime maximum for setup, controlled expansion and new scenarios together; "
          f"${p.spend.total:.4f} already logged. Target {controlled_count} controlled + {novel_count} scenarios. "
          "No training or Modal jobs.", flush=True)
    if a.dry_run:
        from .cli import cmd_ingest
        cfg.features.check_seeds = False  # this option promises no LLM calls, even with custom infrastructure settings
        try:
            seeds = cmd_ingest(cfg)
            if not seeds:
                raise ValueError("no usable seeds")
            report = {"status": "offline_inputs_ready", "seeds": len(seeds), "paid_calls": 0,
                      "controlled_requested": controlled_count, "scenario_requested": novel_count,
                      "derived_setup": "will run automatically on live launch"}
            (cfg.out / "dry_run.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report))
            return 0
        finally:
            configure_call_log(None)
    set_budget(Budget(a.budget, p.spend.total))
    status = "complete"
    try:
        p.preflight()
        def contracts():
            from .analyzer import contracts_only
            return contracts_only(cfg, p._seeds())["by"]
        for step in ("ingest", "judge_setup", "contracts" if only_scenarios else "analyzer"):
            p.run_step(step, contracts if step == "contracts" else getattr(p, step))
            if step == "judge_setup":
                p.state["rubric_path"] = cfg.rubric_path
                p.save()
        # Let existing controlled production use half the remaining allowance, saving room for novelty.
        spent = p.spend.update()
        p.budget = spent + (a.budget - spent) * 0.5
        set_budget(Budget(p.budget, spent))
        try:
            for step in () if only_scenarios else ("proposals", "production"):
                p.run_step(step, getattr(p, step))
        except Stop as e:
            if not e.ok:
                raise
        p.budget = a.budget
        set_budget(Budget(a.budget, p.spend.update()))
        seeds = p._seeds()
        # Set here too: production sets them only when it runs in this process (not on resume or scenario-only).
        cfg.tool_optional_args = observed_optional_args(seeds)
        cfg.tool_loose_type_args = observed_loose_type_args(seeds)
        chosen = json.loads((p.root / "chosen.json").read_text())
        contracts = (p.root / f"v{chosen['version']}" / "observed_contracts.md").read_text()
        caps = catalogue(cfg, seeds, contracts, scenario_out)
        jobs_path = scenario_out / "jobs.jsonl"
        if not jobs_path.exists():
            # Attempts are bounded; rejection never triggers unbounded spending to fill a target.
            write_jsonl(jobs_path, scenario_jobs(seeds, caps, math.ceil(novel_count * 2), cfg.random_seed, stop_rate(seeds)))
        protected = read_jsonl(cfg.out / "heldout.jsonl")
        for path in cfg.decontam_paths:
            protected += read_jsonl(path)
        result_path = scenario_out / "results.jsonl"
        rows = read_jsonl(result_path)
        done = {r["id"] for r in rows}
        report = export(scenario_out, rows, {})
        jobs = read_jsonl(jobs_path)
        cells = Counter(j["cell"] for j in jobs)
        # each allocation (gap = treatment, broad = control) fills its own share of the target
        want = {c: math.ceil(novel_count * n / len(jobs)) for c, n in cells.items()}

        nudge_text = stop_text(seeds)

        def one(job):
            current_item.set(job["job_id"])
            return run_job(cfg, job, seeds, contracts, scenario_out, protected, {"stop_text": nudge_text}, {})
        pending = [j for j in jobs if j["job_id"] not in done]
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
            while pending:
                kept = report["kept_by_cell"]
                open_cells = {c for c in want if kept.get(c, 0) < want[c]}
                if not open_cells:
                    break
                wave = [j for j in pending if j["cell"] in open_cells][:max(1, cfg.workers)]
                if not wave:
                    break
                pending = [j for j in pending if j not in wave]
                futures = {ex.submit(copy_context().run, one, j): j for j in wave}
                failed = []
                for f in as_completed(futures):
                    try:
                        row = f.result()
                    except Exception as e:  # noqa: BLE001 - keep the wave's finished rows; paid parts are cached
                        failed.append(e)
                        continue
                    append(result_path, row)
                    rows.append(row)
                    report = export(scenario_out, rows, {})
                    print(f"{row['id']}: {'accepted' if row['kept'] else 'rejected'} ({row['reason'][:80]}); "
                          f"kept {report['kept_by_cell']}", flush=True)
                if failed:  # budget or evaluator outage: stop cleanly, resume re-runs only the unfinished jobs
                    raise next((e for e in failed if isinstance(e, BudgetExhausted)), failed[0])
    except (BudgetExhausted, Stop) as e:
        status = "budget_stop" if isinstance(e, BudgetExhausted) or e.ok else f"stopped: {e}"
    except Exception as e:  # noqa: BLE001 - save the partial dataset and make errors explicit/retryable
        status = f"error: {type(e).__name__}: {e}"
    finally:
        set_budget(None)
        configure_call_log(None)
    report = package(p, scenario_out, controlled_count, novel_count, status, a.budget)
    if status == "complete" and report["short_of_target"]:
        report["status"] = "short_of_target"
        (cfg.out / "final/report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if report["status"] in {"complete", "budget_stop", "short_of_target"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
