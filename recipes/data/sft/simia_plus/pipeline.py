"""One command from seeds to kept traces: preflight -> ingest -> judge setup -> analyzer -> proposals -> production -> report.

    simia-plus pipeline --seeds <seeds.jsonl> --out <dir> --budget 5 [--target 300] [--config <cfg.json>]

Every step reuses the existing stage code and writes under --out; re-running the same command resumes (finished steps
are skipped, unfinished ones pick up from their files). One call log (out/llm_calls.jsonl) feeds the budget.

Fallbacks (each one that fires is listed in the report):
- LLM call fails: retried 10x by llm.py; a production job that still fails is an execution error, retried on the next run.
- judge setup: induction retried once, then the run stops (no judging without domain answer rules). A failed REVISE
  round keeps the induced rubric.
- analyzer: round 2 only when round 1 keeps < analyzer_round2_below; stop when the chosen version keeps < analyzer_stop_below.
- proposals: a seed whose calls fail or whose proposals are all invalid is skipped and logged.
- circuit breaker: stop when more than breaker_max_bad of the last breaker_window production jobs failed (execution
  error, or a judge / consistency call error: those fail closed, so an outage would silently reject everything).
- budget: jobs run in waves; no new wave is started when its estimated cost would pass the budget.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

from .config import Config, load_config
from .io import parallel_map, read_jsonl, write_jsonl
from .llm import Budget, BudgetExhausted, configure_call_log, current_item, get_budget, set_budget

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pipeline_fireworks.json"
ROLES = ("default", "generator", "judge", "analyzer", "diagnose", "rubric_induce", "selftest")
STEPS = ("ingest", "judge_setup", "analyzer", "proposals", "production")
TYPES = ("old", "task", "evidence", "stack")
COST_MARGIN = 1.15  # wave cost estimate = mean cost per finished job x this


class Stop(Exception):
    """Clean stop (budget, breaker, low yield, failed step): the report is still written."""

    def __init__(self, reason: str, ok: bool = False):
        super().__init__(reason)
        self.ok = ok  # True: a normal end (budget reached), exit code 0


class Spend:
    """Running total of logged LLM cost, read incrementally from the call log (rows carry full prompts, so the file is
    never re-read whole)."""

    def __init__(self, path: Path):
        self.path, self.offset, self.total, self.unpriced = path, 0, 0.0, 0
        self.by_item: Counter = Counter()

    def update(self) -> float:
        if self.path.exists():
            with self.path.open("rb") as f:
                f.seek(self.offset)
                data = f.read()
            end = data.rfind(b"\n") + 1  # a row still being written is read next time
            for line in data[:end].splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                cost = row.get("cost")
                if cost is None and row.get("ok"):
                    self.unpriced += 1
                self.total += cost or 0
                self.by_item[str(row.get("item"))] += cost or 0
            self.offset += end
        return self.total


class Pipeline:
    def __init__(self, cfg: Config, budget: float, target: int):
        self.cfg, self.budget, self.target = cfg, budget, target
        self.out = cfg.out
        self.pdir = self.out / "pipeline"
        self.pdir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.pdir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"steps": {}, "fallbacks": []}
        self.spend = Spend(self.out / "llm_calls.jsonl")
        self.spend.update()
        self.root = self.out / "analyzer"
        self.prod = self.root / "prod"

    # ---- bookkeeping ----

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1, default=str))

    def fallback(self, step: str, what: str) -> None:
        row = {"step": step, "what": what}
        if row not in self.state["fallbacks"]:
            self.state["fallbacks"].append(row)
            self.save()
        print(f"[fallback] {step}: {what}")

    def done(self, step: str) -> bool:
        return self.state["steps"].get(step, {}).get("status") == "done"

    def run_step(self, step: str, fn) -> None:
        if self.done(step):
            print(f"== {step}: done earlier, skipped")
            return
        if self.spend.update() >= self.budget:
            raise Stop(f"budget ${self.budget} reached before step {step} (spent ${self.spend.total:.3f})", ok=True)
        print(f"== {step}")
        row = self.state["steps"].setdefault(step, {})
        before, t0 = self.spend.total, time.monotonic()
        row.update(status="running", started=_now())
        self.save()
        try:
            note = fn()
            budget = get_budget()
            if budget is not None and budget.hit:  # a call was refused inside the step: it did not complete
                raise BudgetExhausted("a call was refused")
            row.update(status="done", note=note)
        except BudgetExhausted as e:
            row.update(status="stopped", note=f"budget: {e}"[:300])
            raise Stop(f"budget ${self.budget} reached during step {step} (spent ${self.spend.update():.3f}); finished "
                       "work is kept, re-run with a higher --budget to continue", ok=True) from None
        except Stop as e:
            row.update(status="stopped", note=str(e))
            raise
        except Exception as e:
            row.update(status="failed", note=f"{type(e).__name__}: {e}"[:500])
            raise Stop(f"step {step} failed: {type(e).__name__}: {e}"[:600]) from e
        finally:
            row["seconds"] = round(row.get("seconds", 0) + time.monotonic() - t0, 1)
            row["cost_usd"] = round(row.get("cost_usd", 0) + self.spend.update() - before, 4)
            row["finished"] = _now()
            self.save()

    # ---- steps ----

    def preflight(self) -> None:
        """Before any spend: settings that would make the run pointless or the budget blind."""
        cfg, errs = self.cfg, []
        if not Path(cfg.seeds_path).exists():
            errs.append(f"seeds file not found: {cfg.seeds_path}")
        if not cfg.features.rubric_judge:
            errs.append("features.rubric_judge must be on (the judge is the quality gate)")
        for role in ROLES:
            try:
                m = cfg.model_for(role)
            except KeyError as e:
                errs.append(str(e))
                continue
            if not os.environ.get(m.api_key_env):
                errs.append(f"environment variable {m.api_key_env} is not set (needed by the model config)")
            if m.price_input_per_mtok is None or m.price_output_per_mtok is None:
                errs.append(f"role {role}: price_input_per_mtok / price_output_per_mtok missing (the budget cannot see unpriced calls)")
        if self.budget <= 0:
            errs.append("--budget must be > 0")
        if errs:
            raise Stop("preflight failed:\n  - " + "\n  - ".join(dict.fromkeys(errs)))
        prev = self.state.get("settings")
        now = {"seeds": str(cfg.seeds_path), "target": self.target, "budget": self.budget}
        if prev and prev.get("target") != self.target and (self.prod / "jobs.jsonl").exists():
            print(f"[warn] target changed {prev.get('target')} -> {self.target}, but production jobs are already planned "
                  f"(frozen in {self.prod / 'jobs.jsonl'}); delete analyzer/prod/ to replan")
        if prev and prev.get("seeds") != now["seeds"]:
            raise Stop(f"--out already holds a run on other seeds ({prev.get('seeds')}); use a new --out")
        self.state["settings"] = now
        self.save()
        # rough, from 3 past domains: fixed setup ~$1-1.5, ~0.6-1.2 cents per kept trace
        left = self.budget - self.spend.total
        print(f"preflight ok. spent so far ${self.spend.total:.3f}; budget ${self.budget}; rough estimate: "
              f"~{max(0, int((left - 1.5) / 0.012))}-{max(0, int((left - 1.0) / 0.006))} kept traces (target {self.target})")
        if left < 2:
            print("[warn] less than $2 left: setup alone costs ~$1-1.5")

    def ingest(self) -> str:
        from .cli import _load_seeds, cmd_ingest
        cmd_ingest(self.cfg)
        seeds = _load_seeds(self.cfg)
        if not seeds:
            raise RuntimeError("no usable seeds after ingest and quarantine")
        if len(seeds) < self.cfg.selftest_seeds:
            self.fallback("ingest", f"only {len(seeds)} seeds (self-test wants {self.cfg.selftest_seeds}); results are less reliable")
        return f"{len(seeds)} seeds"

    def judge_setup(self) -> str:
        """Induce the domain slots (retry once, then stop), build planted-mistake cases, one REVISE round."""
        from .rubric_loop import assemble_v2, induce, lint, parse_slots
        from .selftest import build, loop
        cfg, seeds = self.cfg, self._seeds()
        if cfg.rubric_path:  # the config names a rubric: use it as given
            if not Path(cfg.rubric_path).exists():
                raise RuntimeError(f"rubric_path not found: {cfg.rubric_path}")
            return f"rubric from config: {cfg.rubric_path}"
        jdir = self.out / "judge"
        idir, sdir, ldir = jdir / "induce", jdir / "selftest", jdir / "loop"
        for d in (idir, sdir, ldir):
            d.mkdir(parents=True, exist_ok=True)
        if not (idir / "domain.md").exists():
            current_item.set("judge/induce")
            last = None
            for attempt in (1, 2):
                try:
                    text = induce(cfg, seeds, v2=True)
                    slots = parse_slots(text)
                    break
                except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
                    raise
                except Exception as e:  # noqa: BLE001
                    last = e
                    if attempt == 1:
                        self.fallback("judge_setup", f"induction failed ({type(e).__name__}: {str(e)[:120]}); retrying once")
            else:
                raise RuntimeError(f"judge induction failed twice, no judge to gate with: {last}")
            (idir / "domain.md").write_text(text + "\n")
            (idir / "rubric.md").write_text(assemble_v2(slots))
            (idir / "lint.json").write_text(json.dumps(lint(slots, list({s.get("system", "") for s in seeds})), indent=1))
        slots = parse_slots((idir / "domain.md").read_text())
        if not (ldir / "rubric_final.md").exists():
            ids = sorted(s["id"] for s in seeds)
            random.Random(cfg.random_seed).shuffle(ids)
            by_id = {s["id"]: s for s in seeds}
            dev_ids, test_ids = ids[:cfg.selftest_seeds], ids[cfg.selftest_seeds:cfg.selftest_seeds + cfg.selftest_test_seeds]
            for name, sel in (("dev", dev_ids), ("test", test_ids)):
                if sel and not (sdir / f"{name}.jsonl").exists():
                    current_item.set(f"judge/selftest_{name}")
                    write_jsonl(sdir / f"{name}.jsonl", build(cfg, [by_id[i] for i in sel], workers=cfg.workers))
            dev = [c for c in read_jsonl(sdir / "dev.jsonl") if not c.get("skip")]
            test = read_jsonl(sdir / "test.jsonl")
            if not any(c["expected"] == "bad" for c in dev) or not any(c["expected"] == "good" for c in dev):
                self.fallback("judge_setup", "too few usable self-test cases; using the induced rubric without REVISE")
                (ldir / "rubric_final.md").write_text((idir / "rubric.md").read_text())
            else:
                current_item.set("judge/loop")
                try:
                    rep = loop(cfg, slots, dev, test, by_id, ldir, cfg.judge_revise_rounds)
                except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
                    raise
                except Exception as e:  # noqa: BLE001 - REVISE only polishes: keep the induced rubric
                    self.fallback("judge_setup", f"REVISE loop failed ({type(e).__name__}: {str(e)[:120]}); using the induced rubric")
                    (ldir / "rubric_final.md").write_text((idir / "rubric.md").read_text())
                else:
                    r0 = rep["history"][0]["dev"]
                    if r0["errors"] and r0["errors"] >= len(dev):
                        raise RuntimeError("every judge call on the self-test failed: check the judge model and key")
                    if any(h.get("error") for h in rep["history"]):
                        self.fallback("judge_setup", "a REVISE round failed; kept the best rubric so far")
        cfg.rubric_path = str(ldir / "rubric_final.md")
        rep_path = ldir / "report.json"
        hist = json.loads(rep_path.read_text())["history"] if rep_path.exists() else []
        return "dev balanced: " + " -> ".join(str(h["dev"]["balanced"]) for h in hist if "dev" in h) if hist else "induced rubric only"

    def analyzer(self) -> str:
        from .analyzer import analyze
        from .cli import _eval_ngrams
        chosen_path = self.root / "chosen.json"
        if not chosen_path.exists():
            analyze(self.cfg, self._seeds(), [], _eval_ngrams(self.cfg))
        chosen = json.loads(chosen_path.read_text())
        rate = chosen["dev_usable_rates"][f"v{chosen['version']}"]
        if len(chosen["dev_usable_rates"]) > 1:
            self.fallback("analyzer", f"round 1 kept < {self.cfg.analyzer_round2_below:.0%}: ran a revision round")
        if rate < self.cfg.analyzer_stop_below:
            raise Stop(f"analyzer: chosen version keeps {rate:.0%} of pilot jobs (< {self.cfg.analyzer_stop_below:.0%}); "
                       "these seeds may not suit synthesis. See analyzer/v*/report.json")
        return f"chosen v{chosen['version']}, usable {chosen['dev_usable_rates']}"

    def proposals(self) -> str:
        """Variations per seed sized to what the target still needs (and the budget allows), mixed by variation_mix;
        then one frozen job plan, interleaving types so a budget stop keeps the mix."""
        from .analyzer import (
            _stack_pairs,
            applicable,
            coverage_summary,
            label_keys,
            propose_for_seed,
            propose_wide,
            system_labels,
        )
        from .verify import system_key
        cfg, seeds = self.cfg, self._seeds()
        self.prod.mkdir(parents=True, exist_ok=True)
        if (self.prod / "jobs.jsonl").exists():
            return "planned earlier"
        chosen = json.loads((self.root / "chosen.json").read_text())
        vdir = self.root / f"v{chosen['version']}"
        contracts = (vdir / "observed_contracts.md").read_text()
        base_vars = json.loads((vdir / "variations.json").read_text())
        labels = system_labels(seeds)
        by_key = {k: lab for lab, ks in labels.items() for k in label_keys(ks)}
        ledger = read_jsonl(self.root / "ledger.jsonl")
        tried = {(r["seed_id"], r["variation"]) for r in ledger}
        have = len(self._pilot_unique())
        need = max(0, self.target - have)
        if not need:
            write_jsonl(self.prod / "jobs.jsonl", [])
            return f"pilot already kept {have} >= target {self.target}"
        rate = max(chosen["dev_usable_rates"][f"v{chosen['version']}"], cfg.analyzer_stop_below)
        jobs_needed = math.ceil(need / rate)
        per_job = self._pilot_cost_per_job()
        if per_job:
            affordable = int(max(0.0, self.budget - self.spend.update()) / (per_job * COST_MARGIN))
            if affordable < jobs_needed:
                self.fallback("proposals", f"budget affords ~{affordable} of {jobs_needed} jobs the target needs; planned to the budget")
                jobs_needed = affordable
        if jobs_needed <= 0:
            raise Stop(f"budget left (${self.budget - self.spend.total:.3f}) affords no production jobs", ok=True)
        per_seed = math.ceil(jobs_needed / len(seeds)) + 1  # headroom: some proposals are rejected or duplicated
        want = _split_counts(per_seed, cfg.variation_mix)
        ppc = max(1, cfg.proposals_per_call)
        failures: list[dict] = []
        cov = {lab: coverage_summary([s for s in seeds if by_key.get(system_key(s.get("system", ""))) == lab], base_vars,
                                     cfg.outcome_fields) for lab in labels}

        def for_seed(seed: dict) -> list[dict]:
            sid, lab = seed["id"], by_key.get(system_key(seed.get("system", "")), "all")
            out: list[dict] = []
            old = [v for v in applicable(base_vars, seed, labels) if (sid, v["name"]) not in tried]
            wide: list[dict] = []
            try:
                calls = 0
                while len(old) < want["old"] and calls < math.ceil(want["old"] / ppc) + 1:
                    current_item.set(f"prod/propose/old/{sid}")
                    got = propose_for_seed(cfg, seed, lab, labels, seeds, contracts, base_vars + old, ledger,
                                           min(ppc, want["old"] - len(old)))
                    calls += 1
                    if not got:
                        break
                    old += [{**v, "eligible_seeds": [sid]} for v in got]
                n_wide = want["task"] + want["evidence"]
                calls = 0
                while len(wide) < n_wide and calls < math.ceil(n_wide / ppc) + 1:
                    current_item.set(f"prod/propose/wide/{sid}")
                    got = propose_wide(cfg, seed, lab, labels, seeds, contracts, base_vars + old + wide, cov.get(lab, ""),
                                       min(ppc, n_wide - len(wide)))
                    calls += 1
                    if not got:
                        break
                    wide += got
            except BudgetExhausted:  # the run's budget is spent: stop the run, never record it as a result
                raise
            except Exception as e:  # noqa: BLE001 - a failed proposal call only means fewer variations for this seed
                failures.append({"seed": sid, "error": f"{type(e).__name__}: {e}"[:300]})
            out += [{**v, "_type": "old"} for v in old[:want["old"]]]
            out += [{**v, "_type": v["kind"]} for v in wide]
            out += [{**v, "_type": "stack"} for v in _stack_pairs(old, seed, labels)[:want["stack"]]]
            return out

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
            per = dict(zip([s["id"] for s in seeds], ex.map(for_seed, seeds)))
        for sid, vs in per.items():  # names must be unique per seed (ledger and reports key on seed + name)
            names: Counter = Counter()
            for v in vs:
                names[v["name"]] += 1
                if names[v["name"]] > 1:
                    v["name"] = f'{v["name"]}__{sid[-6:]}_{names[v["name"]]}'
        skipped = [sid for sid, vs in per.items() if not vs]
        for sid in skipped:
            self.fallback("proposals", f"seed {sid}: no usable variations, skipped")
        write_jsonl(self.prod / "variations.jsonl", [{"seed_id": sid, **v} for sid, vs in per.items() for v in vs])
        (self.prod / "proposal_failures.json").write_text(json.dumps(
            {"failed_calls": failures, "rejected_old": getattr(propose_for_seed, "rejected", []),
             "rejected_wide": getattr(propose_wide, "rejected", [])}, ensure_ascii=False, indent=1, default=str))
        queues = {sid: _interleave(vs, cfg.variation_mix) for sid, vs in per.items() if vs}
        jobs, k = [], 0
        while len(jobs) < jobs_needed and any(len(q) > k for q in queues.values()):
            for sid, q in queues.items():
                if k < len(q) and len(jobs) < jobs_needed and (sid, q[k]["name"]) not in tried:
                    v = {x: y for x, y in q[k].items() if x != "_type"}
                    jobs.append({"job_id": f"prod/{sid}__{k}", "seed_id": sid, "strategy": "analyzer", "mode": "analyzer",
                                 "version": "prod", "case_block": cfg.case_block, "arm": q[k]["_type"], "variation": v})
            k += 1
        write_jsonl(self.prod / "jobs.jsonl", jobs)
        mix = Counter(j["arm"] for j in jobs)
        return f"pilot kept {have}; {len(jobs)} jobs planned for {need} more ({dict(mix)}); {len(skipped)} seeds skipped"

    def production(self) -> str:
        """Generate + verify in waves; budget checked before each wave, circuit breaker after each."""
        from .analyzer import expansion_context, generate_analyzer, mark_lesson_duplicates
        from .banks import ObsBank
        from .cli import _eval_ngrams
        from .verify import observed_loose_type_args, observed_optional_args, verify
        cfg, seeds = self.cfg, self._seeds()
        jobs = read_jsonl(self.prod / "jobs.jsonl")
        if not jobs:
            return "no jobs"
        chosen = json.loads((self.root / "chosen.json").read_text())
        vdir = self.root / f"v{chosen['version']}"
        template, contracts = (vdir / "prompt.txt").read_text(), (vdir / "observed_contracts.md").read_text()
        by_id, obs, eval_ng = {s["id"]: s for s in seeds}, ObsBank.from_traces(seeds), _eval_ngrams(cfg)
        cfg.tool_optional_args = observed_optional_args(seeds)
        cfg.tool_loose_type_args = observed_loose_type_args(seeds)
        gen_path, ver_path = self.prod / "generated.jsonl", self.prod / "verified.jsonl"
        ver = read_jsonl(ver_path)
        errored = [t for t in ver if _job_failure(t) in ("judge call error", "consistency call error")]
        if errored:  # judged while a call failed (fail closed): judge again on resume, reusing the generation
            write_jsonl(ver_path, [t for t in ver if t not in errored])
            print(f"production: re-judging {len(errored)} traces whose judge/check call failed earlier")
        wave_size = max(4, 2 * cfg.workers)
        recent: deque = deque(maxlen=cfg.breaker_window)
        attempted: set = set()
        stop = None
        while True:
            done = {t["id"] for t in read_jsonl(ver_path)}
            pending = [j for j in jobs if j["job_id"] not in done and j["job_id"] not in attempted]
            if not pending:
                break
            per_job = self._job_cost(done) or self._pilot_cost_per_job() or 0.0
            left = self.budget - self.spend.update()
            n = min(wave_size, len(pending), int(left / (per_job * COST_MARGIN)) if per_job else wave_size)
            if n <= 0:
                stop = Stop(f"budget ${self.budget} reached: {len(pending)} production jobs not started (spent ${self.spend.total:.3f})", ok=True)
                break
            wave = pending[:n]
            attempted |= {j["job_id"] for j in wave}
            gen = parallel_map(lambda j: generate_analyzer(cfg, j, by_id[j["seed_id"]], template, obs), wave,
                               key=lambda j: j["job_id"], out_path=gen_path, workers=cfg.workers, done_key=lambda t: t["id"])
            ids = {j["job_id"] for j in wave}
            new = [t for t in gen if t["id"] in ids and t["id"] not in done]
            ver = parallel_map(lambda t: verify(cfg, t, eval_ng, expansion_context(t, by_id, contracts)), new,
                               key=lambda t: t["id"], out_path=ver_path, workers=cfg.workers)
            ver_by = {t["id"]: t for t in ver}
            for j in wave:
                recent.append(_job_failure(ver_by.get(j["job_id"])))
            bad = [r for r in recent if r]
            print(f"production: {len(done) + len(ver_by.keys() & ids)}/{len(jobs)} jobs, spent ${self.spend.update():.3f}, "
                  f"failures in last {len(recent)}: {len(bad)}")
            if len(recent) >= cfg.breaker_window and len(bad) > cfg.breaker_max_bad * len(recent):
                top = Counter(bad).most_common(3)
                stop = Stop(f"circuit breaker: {len(bad)} of the last {len(recent)} jobs failed ({top}); something is "
                            "systemic (provider, key, model). Fix and re-run the same command to resume")
                break
        ver = read_jsonl(ver_path)
        sig = sorted(t["id"] for t in ver if t["meta"]["verify"].get("kept"))
        if cfg.features.lesson_dedup and sig and self.state.get("lesson_dedup_of") != sig:
            for t in ver:
                t["meta"]["verify"].pop("lesson_dup_of", None)
                t["meta"]["verify"].pop("lesson_dup_error", None)
            current_item.set("prod/lesson_dedup")
            write_jsonl(ver_path, mark_lesson_duplicates(cfg, ver))
            self.state["lesson_dedup_of"] = sig
            self.save()
        if stop:
            raise stop
        return f"{len(ver)}/{len(jobs)} jobs verified, {len(sig)} kept before dedup"

    # ---- report ----

    def report(self, stop: Stop | None) -> dict:
        from .analyzer import pilot_report
        from .cli import _load_seeds
        from .schema import to_sharegpt
        from .select import select
        cfg = self.cfg
        seeds = _load_seeds(cfg) if (self.out / "seeds.jsonl").exists() else []
        pilot = [t for v in sorted(self.root.glob("v*/verified.jsonl")) for t in read_jsonl(v)]
        prod = read_jsonl(self.prod / "verified.jsonl")
        pool = pilot + prod
        unique = self._unique(pool)
        uids = {id(t) for t in unique}
        final = select(unique, self.target)
        fdir = self.out / "final"
        write_jsonl(fdir / "synthetic.jsonl", final)
        write_jsonl(fdir / "synthetic_sharegpt.jsonl", (to_sharegpt(t) for t in final))
        write_jsonl(fdir / "seeds_plus_synthetic_sharegpt.jsonl", (to_sharegpt(t) for t in [*seeds, *final]))
        jobs = sum(len(read_jsonl(p)) for p in self.root.glob("v*/jobs.jsonl")) + len(read_jsonl(self.prod / "jobs.jsonl"))
        started = sum(len(read_jsonl(p)) for p in self.root.glob("v*/jobs.jsonl")) + len(prod)
        why = Counter(_why(t) for t in pool)
        why.pop("kept", None)
        kept_n = sum(bool(t["meta"]["verify"].get("kept")) for t in pool)
        why["duplicate (exact/near/same lesson)"] = kept_n - len(unique)
        self.spend.update()
        calls = read_jsonl(self.out / "llm_calls.jsonl")
        roles: dict[str, dict] = {}
        for c in calls:
            r = roles.setdefault(c.get("role") or "?", {"calls": 0, "failed_attempts": 0, "cost_usd": 0.0, "lat": []})
            r["calls"] += 1
            r["failed_attempts"] += not c.get("ok")
            r["cost_usd"] += c.get("cost") or 0
            if c.get("ok"):
                r["lat"].append(c.get("latency_s") or 0)
        for r in roles.values():
            lat = sorted(r.pop("lat"))
            r["cost_usd"] = round(r["cost_usd"], 4)
            r["mean_latency_s"] = round(sum(lat) / len(lat), 1) if lat else None
            r["p90_latency_s"] = lat[int(0.9 * (len(lat) - 1))] if lat else None
        flags: dict[str, Counter] = {}  # soft flags: how often each fired, and how often the judges then rejected
        for t in pool:
            v = t["meta"]["verify"]
            for f in {_flag_kind(x) for x in v.get("soft_flags") or []}:
                c = flags.setdefault(f, Counter())
                c["fired"] += 1
                c["rejected"] += not v.get("kept")
        by_type = {}
        for typ in TYPES:
            ts = [t for t in prod if t["meta"].get("arm") == typ]
            if ts:
                by_type[typ] = f'{sum(id(t) in uids for t in ts)}/{len(ts)} kept'
        rep = {"status": "complete" if stop is None else ("stopped: " + str(stop)),
               "target": self.target, "final": len(final), "short_of_target": max(0, self.target - len(final)),
               "jobs_planned": jobs, "jobs_started": started, "kept_before_dedup": kept_n, "unique_kept": len(unique),
               "from_pilot": sum(id(t) in uids for t in pilot), "from_production": sum(id(t) in uids for t in prod),
               "production_kept_by_type": by_type,
               "soft_flags": {k: dict(c) for k, c in flags.items()},
               "rejected_why": dict(why.most_common()),
               "cost_usd": round(self.spend.total, 4), "budget_usd": self.budget,
               "cost_per_final_trace": round(self.spend.total / len(final), 4) if final else None,
               "unpriced_calls": self.spend.unpriced,
               "steps": self.state["steps"], "fallbacks": self.state["fallbacks"], "by_role": roles,
               "rubric": cfg.rubric_path,
               "quality_detail": pilot_report(cfg, pool, seeds, attempted=started) if pool and seeds else None}
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
        (fdir / "report.md").write_text(_markdown(rep))
        return rep

    # ---- helpers ----

    def _seeds(self) -> list[dict]:
        from .cli import _load_seeds
        return _load_seeds(self.cfg)

    def _unique(self, pool: list[dict]) -> list[dict]:
        from .select import dedup
        kept = [t for t in pool if t["meta"]["verify"].get("kept") and not t["meta"]["verify"].get("lesson_dup_of")]
        unique, _ = dedup(kept, near=self.cfg.features.near_dedup)
        return unique

    def _pilot_unique(self) -> list[dict]:
        return self._unique([t for v in sorted(self.root.glob("v*/verified.jsonl")) for t in read_jsonl(v)])

    def _pilot_cost_per_job(self) -> float | None:
        jobs = [j["job_id"] for p in self.root.glob("v*/jobs.jsonl") for j in read_jsonl(p)]
        self.spend.update()
        cost = sum(self.spend.by_item.get(j, 0) for j in jobs)
        return cost / len(jobs) if jobs and cost else None

    def _job_cost(self, done: set) -> float | None:
        self.spend.update()
        cost = sum(self.spend.by_item.get(j, 0) for j in done)
        return cost / len(done) if done and cost else None


def _split_counts(n: int, mix: dict) -> dict:
    """n split over TYPES by mix (largest remainder), at least 1 for every type with a positive share when n allows."""
    raw = {t: n * mix.get(t, 0) for t in TYPES}
    out = {t: int(raw[t]) for t in TYPES}
    for t in sorted(TYPES, key=lambda t: raw[t] - out[t], reverse=True)[: n - sum(out.values())]:
        out[t] += 1
    return out


def _interleave(vs: list[dict], mix: dict) -> list[dict]:
    """One seed's variations ordered by smooth weighted round robin over their types, so any prefix keeps the mix."""
    queues = {t: [v for v in vs if v["_type"] == t] for t in TYPES}
    credit = dict.fromkeys(TYPES, 0.0)
    out = []
    while any(queues.values()):
        live = [t for t in TYPES if queues[t]]
        w = {t: mix.get(t, 0) or 1e-6 for t in live}
        for t in live:
            credit[t] += w[t]
        pick = max(live, key=lambda t: credit[t])
        credit[pick] -= sum(w.values())
        out.append(queues[pick].pop(0))
    return out


def _flag_kind(flag: str) -> str:
    """A soft flag without its specifics (message numbers, key lists, error detail), so counts group by kind."""
    return re.sub(r"\(.*", "", re.sub(r"\[.*?\]", "[..]", re.sub(r"\d+", "N", flag))).strip()[:80]


def _job_failure(t: dict | None) -> str | None:
    """Why a production job counts against the circuit breaker, or None. Refusals and judge rejections are normal."""
    if t is None:
        return "execution error"
    v, m = t["meta"]["verify"], t["meta"]
    if m.get("refusal"):
        return None
    if not t["messages"]:
        return "no parseable conversation"
    if (v.get("rubric") or {}).get("label") == "error":
        return "judge call error"
    if (v.get("consistency") or {}).get("label") == "error":
        return "consistency call error"
    return None


def _why(t: dict) -> str:
    v, m = t["meta"]["verify"], t["meta"]
    if v.get("kept"):
        return "kept"
    if m.get("refusal") or v.get("refused"):
        return "generator refused the variation"
    if v.get("rule_issues"):
        return "code check: " + str(v["rule_issues"][0]).split(":")[0][:60]
    if v.get("contaminated"):
        return "overlaps eval traces"
    if v.get("change_issues"):
        return "change gate (nothing varied / copy of seed)"
    r, c = v.get("rubric") or {}, v.get("consistency") or {}
    if r.get("label") == "error" or c.get("label") == "error":
        return "judge/check call error (fail closed)"
    if r.get("label") not in (None, "good"):
        return f"judge: {r.get('label')}"
    if c.get("label") == "bad":
        kinds = [k for k, xs in (c.get("problems") or {}).items() if xs]
        return "consistency: " + ",".join(kinds)
    return "other"


def _markdown(r: dict) -> str:
    lines = ["# Pipeline report\n", f"**Status:** {r['status']}\n",
             f"- final traces: **{r['final']}** of target {r['target']} (short {r['short_of_target']})",
             (f"- unique kept {r['unique_kept']} (pilot {r['from_pilot']}, production {r['from_production']}); "
              f"jobs started {r['jobs_started']} of {r['jobs_planned']} planned"),
             f"- production kept by type: {r['production_kept_by_type'] or '-'}",
             f"- soft flags (fired / then rejected by the judges): {r['soft_flags'] or '-'}",
             f"- cost ${r['cost_usd']} of budget ${r['budget_usd']}; per final trace ${r['cost_per_final_trace']}"
             + (f"; {r['unpriced_calls']} calls without a price (not in the total)" if r["unpriced_calls"] else ""),
             "\n## Rejected, and why\n"] + [f"- {k}: {n}" for k, n in r["rejected_why"].items()] + \
            ["\n## Steps\n", "| step | status | seconds | cost $ | note |", "|---|---|---|---|---|"] + \
            [f"| {k} | {s.get('status')} | {s.get('seconds')} | {s.get('cost_usd')} | {str(s.get('note') or '')[:160]} |"
             for k, s in r["steps"].items()] + \
            ["\n## Fallbacks fired\n"] + ([f"- {f['step']}: {f['what']}" for f in r["fallbacks"]] or ["- none"]) + \
            ["\n## By role\n", "| role | calls | failed attempts | cost $ | mean s | p90 s |", "|---|---|---|---|---|---|"] + \
            [f"| {k} | {v['calls']} | {v['failed_attempts']} | {v['cost_usd']} | {v['mean_latency_s']} | {v['p90_latency_s']} |"
             for k, v in r["by_role"].items()] + [f"\nRubric: `{r['rubric']}`. Detail: report.json (quality_detail)."]
    return "\n".join(lines) + "\n"


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="simia-plus pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", help="seed traces (overrides the config's seeds_path)")
    ap.add_argument("--out", help="output dir (overrides the config's out_dir); re-run with the same --out to resume")
    ap.add_argument("--budget", type=float, required=True, help="USD cap on logged LLM spend for this --out")
    ap.add_argument("--target", type=int, help="traces wanted (default: the config's target_count)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help=f"run config (default: {DEFAULT_CONFIG.name})")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in a log file as it happens
    cfg = load_config(a.config)
    if a.seeds:
        cfg.seeds_path = a.seeds
    if a.out:
        cfg.out_dir = a.out
    cfg.prompt_source = "analyzer"
    cfg.out.mkdir(parents=True, exist_ok=True)
    configure_call_log(cfg.out / "llm_calls.jsonl")
    p = Pipeline(cfg, a.budget, a.target or cfg.target_count)
    set_budget(Budget(a.budget, p.spend.total))  # hard cap on every model call; spend so far counts on resume
    if p.state.get("rubric_path"):
        cfg.rubric_path = p.state["rubric_path"]
    stop = None
    try:
        p.preflight()
        for step in STEPS:
            p.run_step(step, getattr(p, step))
            if step == "judge_setup":
                p.state["rubric_path"] = cfg.rubric_path
                p.save()
    except Stop as e:
        stop = e
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        stop = Stop("interrupted; re-run the same command to resume")
        print(f"\nSTOPPED: {stop}")
    set_budget(None)
    rep = p.report(stop)
    print(json.dumps({k: rep[k] for k in ("status", "target", "final", "unique_kept", "cost_usd", "budget_usd")}, default=str))
    print(f"report: {cfg.out / 'final' / 'report.md'}")
    return 0 if stop is None or stop.ok else 1


if __name__ == "__main__":
    sys.exit(main())
