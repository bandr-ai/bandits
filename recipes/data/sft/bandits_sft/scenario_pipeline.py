"""Scenario recipe: new situations beyond the seeds, same agent contract and tools.

    bandits-sft dataset --seeds <seeds.jsonl> --out <dir> --budget 5 [--target 100] [--config <cfg.json>]

ingest -> judge_setup -> contracts -> coverage -> scenarios -> report. Everything the checks need is derived from the
seeds (rubric, contracts, coverage catalogue, tool-result rules, label fields); nothing is supplied. Same resume,
budget and report conventions as the controlled pipeline (pipeline.py), which this class extends.

Two arms share one generator and one set of checks: gap jobs exercise a decision or evidence route the seeds
cover rarely (found by the coverage step), broad jobs write a new situation that ends in a real seed's outcome.
A job that fails is retried once, then recorded with category "error"; the run stops only on the circuit breaker
or the budget. A judge outage is never recorded as a reject: the job stays open for the next run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path

from . import scenario_prompts
from .analyzer import basis_errors, seed_json
from .config import Config, load_config
from .io import read_jsonl, write_jsonl
from .llm import Budget, BudgetExhausted, chat_json, configure_call_log, current_item, get_llm, set_budget
from .pipeline import DEFAULT_CONFIG, Pipeline, Stop, _now
from .scenarios import DEFAULTS, EvaluatorError, append, export, label_fields, run_job, seed_rules
from .schema import load_trace, to_sharegpt
from .verify import final_answer, observed_loose_type_args, observed_optional_args

SCENARIO_STEPS = ("ingest", "judge_setup", "contracts", "coverage", "scenarios")
CATEGORIES = ("bad_card", "unsupported", "bad_generation", "seed_rules", "quality_gate", "audit", "control_mismatch",
              "leakage", "duplicate", "error")


# ---- setup derived from the seeds ----

def _wrapups(seeds: list[dict]) -> Counter:
    """Per text, the seeds whose last user turn before the final answer (not the opening turn) says it."""
    texts = Counter()
    for s in seeds:
        m = s["messages"]
        if len(m) >= 3 and m[-1]["role"] == "assistant" and m[-2]["role"] == "user":
            text = (m[-2].get("content") or "").strip()
            if text:
                texts[text] += 1
    return texts


def stop_text(seeds: list[dict]) -> str:
    """A wrap-up user turn the seeds themselves put right before the final answer (a harness nudge), or ''. It must
    precede the final answer in at least max(3, 5%) of the seeds: a chat's repeated 'thanks' does not qualify."""
    top = _wrapups(seeds).most_common(1)
    return top[0][0] if top and top[0][1] >= max(3, math.ceil(0.05 * len(seeds))) else ""


def stop_rate(seeds: list[dict]) -> float:
    """Share of seeds carrying the wrap-up turn, 0 when there is none. Measured from the seeds, never assumed."""
    text = stop_text(seeds)
    return _wrapups(seeds)[text] / len(seeds) if text else 0.0


def outcome(seed: dict, fields: list[str]) -> dict:
    """The seed's values for the label fields ({} for free text or when no label field was learned)."""
    ans = final_answer(seed)
    return {k: ans[k] for k in fields if isinstance(ans, dict) and k in ans}


def scenario_jobs(seeds: list[dict], capabilities: list[dict], count: int, rng_seed: int, nudge: float = 0.0,
                  fields: list[str] | None = None) -> list[dict]:
    """Interleaved coverage-gap and broad jobs from the same generator and checks. A broad job keeps a uniformly drawn
    seed's outcome in a new situation, so its label mix matches the seeds'. Deterministic: the first N jobs of a longer
    plan are the N jobs of a shorter one, so a resume with a bigger target only appends."""
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
                     "outcome": None if gap else outcome(by_id[sid], fields or []) or None})
    return jobs


def answer_stats(traces: list[dict]) -> dict:
    """Per allocation: distribution of short string fields and mean of numeric fields in final JSON answers."""
    out = {}
    for t in traces:
        ans = final_answer(t)
        if not isinstance(ans, dict):
            continue
        row = out.setdefault(t["meta"].get("allocation") or "unlabelled", {"n": 0, "values": {}, "means": {}})
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


def coverage_seeds(seeds: list[dict], max_chars: int) -> list[dict]:
    """The seeds the coverage call gets: one agent (system prompt + tool names) after another in turn, each trace cut to
    head and tail, until max_chars is used. Small seed sets are passed whole."""
    from .scenarios import fit
    groups: dict[str, list[dict]] = {}
    for s in seeds:
        groups.setdefault(json.dumps([s["system"], sorted(t["name"] for t in s["tools"])]), []).append(s)
    cap = max(2000, max_chars // 8)
    out, used = [], 0
    while any(groups.values()):
        for g in groups.values():
            if not g:
                continue
            s = g.pop(0)
            size = len(json.dumps({"system": s["system"], "tools": s["tools"], "trace": fit(s, cap)}, ensure_ascii=False))
            if used + size > max_chars and out:
                return out
            out.append(fit(s, cap))
            used += size
    return out


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


def catalogue(cfg: Config, seeds: list[dict], contracts: str, out_dir: Path) -> list[dict]:
    """LLM coverage catalogue. Each raw answer is saved before checking (resume never pays twice); invalid items are
    dropped one by one after one corrective retry, and kept in coverage_rejected.json. The call sees at most
    cfg.coverage_max_chars of seed text; capabilities may only name seeds it saw."""
    path = out_dir / "coverage.json"
    if path.exists():
        return json.loads(path.read_text())
    current_item.set("scenarios/coverage")
    # Embedded instructions are seed data: a capability may cite them as well as the derived contracts.
    known = re.split(r"(?im)^#+\s*unknown\b", contracts, maxsplit=1)[0]
    basis_text = known + "\n" + "\n".join(sorted({s["system"] for s in seeds if s.get("system")}))
    shown = coverage_seeds(seeds, cfg.coverage_max_chars)
    request = json.dumps({"contracts": contracts, "seeds": [{"id": s["id"], "system": s["system"], "tools": s["tools"],
                                                             "trace": json.loads(seed_json(s))} for s in shown]},
                         ensure_ascii=False)
    ids = {s["id"] for s in shown}
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


def domain_setup(seeds: list[dict], contracts_path: Path, rubric_path: str | None, capabilities: int) -> dict:
    """What the run learned from the seeds. A setup with no label field and no branch key shows up here instead of
    silently switching checks off."""
    rules = seed_rules(seeds)
    fields = label_fields(seeds)
    branch_keys = ["available"] if rules.availability else []
    nudge = stop_text(seeds)
    return {"seeds": len(seeds), "label_fields": fields, "branch_keys": branch_keys, "gates": len(rules.gates),
            "shape_rules": len(rules.shapes), "unavailable_only_signatures": sum(v == {False} for v in rules.availability.values()),
            "stop_text": nudge, "stop_rate": round(stop_rate(seeds), 3), "capabilities": capabilities,
            "contracts_path": str(contracts_path), "rubric_path": rubric_path, "no_op": not fields and not branch_keys}


# ---- the pipeline ----

class ScenarioPipeline(Pipeline):
    STEPS = SCENARIO_STEPS

    def __init__(self, cfg: Config, budget: float, target: int, on_progress: Callable[[int], None] | None = None,
                 progress_every: int = 10):
        super().__init__(cfg, budget, target)
        self.on_progress, self.progress_every = on_progress, max(1, progress_every)
        self.sdir = self.out / "scenarios"
        if self.state.get("rubric_path") and not cfg.rubric_path:
            cfg.rubric_path = self.state["rubric_path"]

    # ---- checks ----

    def preflight(self) -> None:
        cfg, f, errs = self.cfg, self.cfg.features, []
        if cfg.rubric_path and cfg.rubric_path != self.state.get("rubric_path") or cfg.answer_schemas_path:
            errs.append("the scenario recipe derives its rubric and answer rules from the seeds: omit rubric_path and answer_schemas_path")
        if not f.rubric_judge or not f.consistency_check:
            errs.append("features.rubric_judge and features.consistency_check must be on (they are the quality gate)")
        if f.judge or f.split_answer:
            errs.append("features.judge and features.split_answer must be off")
        if cfg.prompt_source != "analyzer":
            errs.append('prompt_source must be "analyzer"')
        if cfg.per_seed_variations:
            errs.append("per_seed_variations must be off (per-seed proposals feed only the controlled recipe)")
        if errs:
            raise Stop("preflight failed:\n  - " + "\n  - ".join(errs))
        super().preflight()

    # ---- steps ----

    def judge_setup(self) -> str:
        note = super().judge_setup()
        self.state["rubric_path"] = self.cfg.rubric_path
        self.save()
        return note

    def contracts(self) -> str:
        from .analyzer import contracts_only
        return contracts_only(self.cfg, self._seeds())["by"]

    def _contracts_path(self) -> Path:
        chosen = json.loads((self.root / "chosen.json").read_text())
        return self.root / f"v{chosen['version']}" / "observed_contracts.md"

    def coverage(self) -> str:
        cfg, seeds = self.cfg, self._seeds()
        self.sdir.mkdir(parents=True, exist_ok=True)
        caps = catalogue(cfg, seeds, self._contracts_path().read_text(), self.sdir)
        setup = domain_setup(seeds, self._contracts_path(), cfg.rubric_path, len(caps))
        (self.out / "setup").mkdir(exist_ok=True)
        (self.out / "setup" / "domain.json").write_text(json.dumps(setup, indent=1))
        jobs = self._planned_jobs(seeds, caps, setup)
        return (f"{len(caps)} capabilities, {len(jobs)} jobs; label fields {setup['label_fields']}, "
                f"branch keys {setup['branch_keys']}, wrap-up turn {'yes' if setup['stop_text'] else 'no'}")

    def _planned_jobs(self, seeds: list[dict], caps: list[dict], setup: dict) -> list[dict]:
        """The frozen job list. A later run with a bigger target appends jobs (the plan is a deterministic prefix)."""
        path = self.sdir / "jobs.jsonl"
        want = scenario_jobs(seeds, caps, math.ceil(self.target * self.cfg.scenario_attempts_per_target),
                             self.cfg.random_seed, setup["stop_rate"], setup["label_fields"])
        have = read_jsonl(path)
        if len(have) < len(want) and want[:len(have)] == have:
            write_jsonl(path, want)
            return want
        return have or want

    def _results(self) -> dict[str, dict]:
        """Latest row per job (a retried error is replaced by its later row)."""
        return {r["id"]: r for r in read_jsonl(self.sdir / "results.jsonl")}

    def scenarios(self) -> str:
        cfg = self.cfg
        seeds = self._seeds()
        cfg.tool_optional_args = observed_optional_args(seeds)  # set here: production is not run, and a resume skips setup
        cfg.tool_loose_type_args = observed_loose_type_args(seeds)
        contracts = self._contracts_path().read_text()
        setup = json.loads((self.out / "setup" / "domain.json").read_text())
        jobs = self._planned_jobs(seeds, json.loads((self.sdir / "coverage.json").read_text()), setup)
        spec = {"stop_text": setup["stop_text"], "label_fields": setup["label_fields"]}
        evals = [load_trace(r, i, "auto") for path in cfg.decontam_paths for i, r in enumerate(read_jsonl(path))]
        protected = [*read_jsonl(self.out / "heldout.jsonl"), *evals]
        result_path = self.sdir / "results.jsonl"
        results = self._results()
        report = export(self.sdir, list(results.values()), {}, setup["label_fields"])
        cells = Counter(j["cell"] for j in jobs)
        want = {c: math.ceil(self.target * n / len(jobs)) for c, n in cells.items()}  # each arm fills its own share
        window = max(1, min(cfg.breaker_window, len(jobs)))
        recent: deque = deque(maxlen=window)
        attempted: set = set()
        outage: set = set()
        last_reported = len(results)

        def attempt(job):
            current_item.set(job["job_id"])
            for n in (1, 2):  # a transient failure is retried once; the paid card and generation are cached
                try:
                    return run_job(cfg, job, seeds, contracts, self.sdir, protected, spec, {})
                except BudgetExhausted:
                    raise
                except EvaluatorError:
                    if n == 2:
                        raise
                except Exception as e:  # noqa: BLE001 - one job never stops the run
                    if n == 2:
                        return {"id": job["job_id"], "job": job, "traces": [], "kept": False, "category": "error",
                                "reason": f"error: {type(e).__name__}: {str(e)[:200]}"}
            raise AssertionError("unreachable")

        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
            while True:
                kept = report["kept_by_cell"]
                open_cells = {c for c in want if kept.get(c, 0) < want[c]}
                wave = [j for j in jobs if j["cell"] in open_cells and j["job_id"] not in attempted
                        and results.get(j["job_id"], {}).get("category", "error") == "error"][:max(1, cfg.workers)]
                if not wave:
                    break
                attempted |= {j["job_id"] for j in wave}
                futures = {ex.submit(copy_context().run, attempt, j): j for j in wave}
                budget_hit = None
                for f in as_completed(futures):
                    try:
                        row = f.result()
                    except BudgetExhausted as e:
                        budget_hit = e
                        continue
                    except EvaluatorError:
                        outage.add(futures[f]["job_id"])
                        recent.append("evaluator error")
                        continue
                    outage.discard(row["id"])
                    append(result_path, row)
                    results[row["id"]] = row
                    recent.append("error" if row.get("category") == "error" else None)
                report = export(self.sdir, list(results.values()), {}, setup["label_fields"])
                bad = [r for r in recent if r]
                print(f"scenarios: {len(results)}/{len(jobs)} jobs, kept {sum(report['kept_by_cell'].values())}, "
                      f"spent ${self.spend.update():.4f}, failures in last {len(recent)}: {len(bad)}")
                if self.on_progress and len(results) // self.progress_every > last_reported // self.progress_every:
                    self.on_progress(len(results))
                last_reported = len(results)
                if budget_hit:
                    raise budget_hit
                if len(recent) >= window and len(bad) > cfg.breaker_max_bad * len(recent):
                    raise Stop(f"circuit breaker: {len(bad)} of the last {len(recent)} jobs failed ({Counter(bad).most_common(3)}); "
                               "something is systemic (provider, key, model). Fix and re-run the same command to resume")
        if outage:
            raise Stop(f"scenarios incomplete: {len(outage)} jobs without a result because a judge or check call failed; "
                       "re-run the same command to retry them")
        return f"{len(results)}/{len(jobs)} jobs, kept {report['kept_by_cell']}, rejected {report['rejected_by_category']}"

    # ---- report ----

    def report(self, stop: Stop | None) -> dict:
        from .cli import _load_seeds
        cfg = self.cfg
        seeds = _load_seeds(cfg) if (self.out / "seeds.jsonl").exists() else []
        setup_path = self.out / "setup" / "domain.json"
        setup = json.loads(setup_path.read_text()) if setup_path.exists() else {}
        rows = list(self._results().values())
        self.sdir.mkdir(parents=True, exist_ok=True)
        scenario = export(self.sdir, rows, {}, setup.get("label_fields", []))
        final = read_jsonl(self.sdir / "synthetic.jsonl")[:self.target]
        fdir = self.out / "final"
        write_jsonl(fdir / "synthetic.jsonl", final)
        write_jsonl(fdir / "synthetic_sharegpt.jsonl", (to_sharegpt(t) for t in final))
        write_jsonl(fdir / "seeds_plus_synthetic_sharegpt.jsonl", (to_sharegpt(t) for t in [*seeds, *final]))
        self.spend.update()
        short = max(0, self.target - len(final))
        status = "complete" if stop is None else "stopped: " + str(stop)
        if stop is None and short:
            status = "short_of_target"
        rep = {"status": status, "target": self.target, "final": len(final), "short_of_target": short,
               "cost_usd": round(self.spend.total, 4), "budget_usd": self.budget,
               "cost_per_final_trace": round(self.spend.total / len(final), 4) if final else None,
               "unpriced_calls": self.spend.unpriced, "by_role": self.role_stats(),
               "steps": self.state["steps"], "fallbacks": self.state["fallbacks"],
               "scenario": {"jobs": len(read_jsonl(self.sdir / "jobs.jsonl")), "jobs_finished": scenario["jobs_finished"],
                            "kept_by_cell": scenario["kept_by_cell"], "rejected_by_category": scenario["rejected_by_category"],
                            "duplicate_or_conflicting_traces": scenario["duplicate_or_conflicting_traces"]},
               "answers_by_allocation": answer_stats(final),
               "setup": {k: v for k, v in setup.items() if k not in ("stop_text", "contracts_path", "rubric_path")}
                        | {"stop_text": bool(setup.get("stop_text"))},
               "rubric": cfg.rubric_path, "external_eval_decontamination": bool(cfg.decontam_paths), "human_verified": False}
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
        (fdir / "report.md").write_text(_markdown(rep))
        return rep


def _markdown(r: dict) -> str:
    s, setup = r["scenario"], r["setup"]
    cost = f"- cost ${r['cost_usd']} of budget ${r['budget_usd']}; per final trace ${r['cost_per_final_trace']}"
    if r["unpriced_calls"]:
        cost += f"; {r['unpriced_calls']} calls without a price (not in the total)"
    learned = (f"- label fields: {setup.get('label_fields') or 'none'}; branch keys: {setup.get('branch_keys') or 'none'}; "
               f"gates {setup.get('gates', 0)}, shape rules {setup.get('shape_rules', 0)}; "
               f"wrap-up turn: {setup.get('stop_text', False)}")
    lines = ["# Scenario report\n", f"**Status:** {r['status']}\n",
             f"- final traces: **{r['final']}** of target {r['target']} (short {r['short_of_target']})",
             f"- jobs finished {s['jobs_finished']} of {s['jobs']} planned; kept by arm: {s['kept_by_cell'] or '-'}",
             cost, "\n## Learned from the seeds\n", learned]
    if setup.get("no_op"):
        lines.append("- **no-op setup**: no label field and no branch key were found, so those checks are off")
    lines += ["\n## Rejected, and why\n", *(f"- {k}: {n}" for k, n in s["rejected_by_category"].items())]
    lines += ["\n## Steps\n", "| step | status | seconds | cost $ | note |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v.get('status')} | {v.get('seconds')} | {v.get('cost_usd')} | {str(v.get('note') or '')[:160]} |"
              for k, v in r["steps"].items()]
    lines += ["\n## Fallbacks fired\n", *([f"- {f['step']}: {f['what']}" for f in r["fallbacks"]] or ["- none"])]
    lines += ["\n## By role\n", "| role | calls | failed attempts | cost $ | mean s | p90 s |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['calls']} | {v['failed_attempts']} | {v['cost_usd']} | {v['mean_latency_s']} | {v['p90_latency_s']} |"
              for k, v in r["by_role"].items()]
    lines += [f"\nRubric: `{r['rubric']}`. LLM validated, not human verified.",
              "Coverage gaps are not measured model failures."]
    return "\n".join(lines) + "\n"


def _freeze(out: Path, files: list[Path]) -> None:
    """Refuse a resume whose seeds or config changed."""
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path = out / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != hashes:
        raise ValueError("inputs changed; use a fresh output directory")
    path.write_text(json.dumps(hashes, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bandits-sft dataset", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True, help="output dir; re-run with the same --out to resume")
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--budget", type=float, required=True, help="USD cap on logged LLM spend for this --out")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="model/settings config; not extra task data")
    ap.add_argument("--decontam", nargs="*", default=[], metavar="JSONL",
                    help="eval/test traces never to copy: seeds and outputs overlapping them are dropped (read by code only)")
    ap.add_argument("--dry-run", action="store_true", help="offline ingestion and input checks only; no credentials or API calls")
    a = ap.parse_args(argv)
    if not a.budget > 0 or a.target < 1:
        raise ValueError("budget must be > 0 and target >= 1")
    sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(a.config)
    cfg.seeds_path, cfg.out_dir, cfg.prompt_source = a.seeds, a.out, "analyzer"
    cfg.per_seed_variations = False
    cfg.decontam_paths = [*cfg.decontam_paths, *a.decontam]
    cfg.out.mkdir(parents=True, exist_ok=True)
    _freeze(cfg.out, [Path(a.seeds), Path(a.config), *map(Path, cfg.decontam_paths)])
    if a.dry_run:
        from .cli import cmd_ingest
        cfg.features.check_seeds = False  # this option promises no LLM calls
        seeds = cmd_ingest(cfg)
        if not seeds:
            raise ValueError("no usable seeds")
        report = {"status": "offline_inputs_ready", "seeds": len(seeds), "paid_calls": 0, "target": a.target, "at": _now()}
        (cfg.out / "dry_run.json").write_text(json.dumps(report))
        print(json.dumps(report))
        return 0
    configure_call_log(cfg.out / "llm_calls.jsonl")
    p = ScenarioPipeline(cfg, a.budget, a.target)
    set_budget(Budget(a.budget, p.spend.total))  # hard cap on every model call; spend so far counts on resume
    stop = None
    try:
        p.preflight()
        for step in SCENARIO_STEPS:
            p.run_step(step, getattr(p, step))
    except Stop as e:
        stop = e
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        stop = Stop("interrupted; re-run the same command to resume")
        print(f"\nSTOPPED: {stop}")
    finally:
        set_budget(None)
        configure_call_log(None)
    rep = p.report(stop)
    print(json.dumps({k: rep[k] for k in ("status", "target", "final", "cost_usd", "budget_usd")}, default=str))
    print(f"report: {cfg.out / 'final' / 'report.md'}")
    return 0 if stop is None or stop.ok else 1


if __name__ == "__main__":
    sys.exit(main())
