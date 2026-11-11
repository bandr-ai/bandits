"""`mint validate`: the three validation levels."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field

from . import scan
from .gates import Runner, RunOutcome, derive_wrong, pick_runner, plan_exec_gates
from .output import MINTError, progress
from .results import check_invariants, finalize, is_judged
from .stages import group_errors, package_hash
from .workspace import Workspace, read_json, write_json


def _meta(ws: Workspace, tid: str) -> dict:
    return read_json(
        ws.internal / "taskmeta" / f"{tid}.json",
        {
            "wrong_answers": [],
            "alternatives": [],
            "hidden": [],
            "leak_allow": [],
            "secret_allow": [],
        },
    )


def seeded_sample(items: list, k: int, seed: int, salt: str) -> list:
    rng = random.Random(f"{seed}:{salt}")
    pool = sorted(items)
    return sorted(rng.sample(pool, min(k, len(pool))))


def refresh_evidence(ws: Workspace, row: dict) -> None:
    """Drop gate evidence if the package on disk no longer matches what was gated."""
    d = ws.tasks / row["type"] / row["task_id"]
    if not d.is_dir():  # deleted package: nothing it passed still counts
        row["gates"].update(structure="not_run", leak="not_run", secrets="not_run", exec="not_run")
        for k in ("runner", "trials", "exec_detail"):
            row["gates"].pop(k, None)
        finalize(row)
        return
    now = package_hash(d, _meta(ws, row["task_id"]))
    if row.get("content_hash") and now != row["content_hash"]:
        # structure/leak were judged on the old content too: level 1 must re-run
        row["gates"].update(structure="not_run", leak="not_run", secrets="not_run", exec="not_run")
        for k in ("runner", "trials", "exec_detail"):
            row["gates"].pop(k, None)
        row["changed"] = True
        finalize(row)
    row["content_hash"] = now


def level1(ws: Workspace, row: dict) -> None:
    """Structure + leak for one task row (mutates row)."""
    d = ws.tasks / row["type"] / row["task_id"]
    if not d.is_dir():
        row["gates"]["structure"] = "fail"
        row["problems"] = ["task directory is missing; re-run `mint run generate`"]
        return
    scan.apply(row, scan.scan_task(d, _meta(ws, row["task_id"]), ws.config()))


def _plan_task(
    ws: Workspace, row: dict, runner: Runner
) -> tuple[list[Callable[[], RunOutcome]], Callable[[list[RunOutcome]], dict]]:
    """The independent gate runs of one task and how to assemble their outcomes."""
    d = ws.tasks / row["type"] / row["task_id"]
    m = _meta(ws, row["task_id"])
    spec = read_json(d / "tests" / "expected.json", {})  # absent for a custom verifier
    wrong, auto = m["wrong_answers"], False
    if not wrong and "expected" in spec:
        canon = spec["expected"][0] if isinstance(spec["expected"], list) else spec["expected"]
        wrong, auto = [derive_wrong(str(canon), spec.get("check", "exact"))], True
    answer_file = spec.get("file", "answer.txt") if spec else None
    return plan_exec_gates(runner, d, answer_file, wrong, m["alternatives"], auto)


def level2_task(ws: Workspace, row: dict, runner: Runner) -> dict:
    """All exec gates of one task, run one after another."""
    calls, assemble = _plan_task(ws, row, runner)
    return assemble([c() for c in calls])


@dataclass
class _TaskRun:
    row: dict
    ttype: str
    calls: list[Callable[[], RunOutcome]]
    assemble: Callable[[list[RunOutcome]], dict]
    outs: list = field(default_factory=list)
    remaining: int = 0
    started: float | None = None

    def __post_init__(self):
        self.outs = [None] * len(self.calls)
        self.remaining = len(self.calls)


def _gates_file(ws: Workspace, tid: str):
    return ws.internal / "gates" / f"{tid}.json"


def _saved_result(ws: Workspace, row: dict, runner: Runner) -> dict | None:
    """Saved exec result of a row whose evidence is current (same package, same runner,
    a pass or a fail), or None. Package changes already cleared the evidence in
    refresh_evidence, so a leftover pass/fail here is for exactly this package."""
    g = row["gates"]
    if g["exec"] not in ("pass", "fail", "no_oracle") or g.get("runner") != runner.name:
        return None
    try:
        res = read_json(_gates_file(ws, row["task_id"]))
    except (OSError, ValueError):
        return None
    if not isinstance(res, dict) or res.get("exec") != g["exec"] or "detail" not in res:
        return None
    return res


def _run_tasks(
    runs: list[_TaskRun],
    jobs: int,
    on_done: Callable[[_TaskRun, dict, float], None],
) -> None:
    """Run every gate run of every task on a thread pool; call `on_done` (main thread only,
    so a single writer) as soon as all gate runs of a task have finished. A failure in any run
    cancels the queued ones, lets running ones end and re-raises: finished tasks were already
    handed to `on_done`."""

    def timed(run: _TaskRun, call: Callable[[], RunOutcome]) -> RunOutcome:
        if run.started is None:
            run.started = time.monotonic()
        return call()

    with ThreadPoolExecutor(max_workers=max(1, jobs), thread_name_prefix="mint-gate") as ex:
        futs = {}
        for run in runs:
            for i, call in enumerate(run.calls):
                futs[ex.submit(timed, run, call)] = (run, i)

        def handle(f) -> None:
            run, i = futs[f]
            run.outs[i] = f.result()
            run.remaining -= 1
            if not run.remaining:
                on_done(run, run.assemble(run.outs), time.monotonic() - (run.started or 0))

        try:
            for f in as_completed(futs):
                handle(f)
        except BaseException:
            for f in futs:
                f.cancel()
            ex.shutdown(wait=True)  # running gate runs end; their finished tasks are still saved
            for f in futs:
                run, i = futs[f]
                if f.done() and not f.cancelled() and f.exception() is None and run.outs[i] is None:
                    with suppress(Exception):
                        handle(f)
            raise


def failure_causes(rows_detail: list[tuple[str, dict]]) -> list[dict]:
    items = []
    for tid, res in rows_detail:
        det = res["detail"]
        for gate, label in (
            ("oracle", "oracle"),
            ("noop", "no-op"),
            ("wrong", "wrong answer"),
            ("alternatives", "alternative answer"),
        ):
            g = det[gate]["gate"]
            if g in ("pass", "not_run"):
                continue
            if g == "error":
                runs = det[gate].get("runs") or [det[gate]]
                why = (
                    next((r.get("detail") for r in runs if r.get("state") == "infra_error"), "")
                    or "infrastructure error"
                )
                items.append((f"INFRA {label}: {why}", tid))
            else:
                msg = {
                    "oracle": "oracle did not pass (reward below 1 or solve.sh failed)",
                    "noop": "no-op answer was accepted by the verifier",
                    "wrong": "wrong answer was accepted by the verifier",
                    "alternatives": "valid alternative answer was rejected (verifier too strict)",
                }[gate]
                items.append((msg, tid))
    return group_errors(items)


def run_validate(
    ws: Workspace,
    only_type: str | None,
    sample_size: int | None,
    max_fix: int | None,
    levels: set[int],
    task_ids: list[str],
    dry: bool,
    runner: Runner | None = None,
    jobs: int | None = None,
) -> dict:
    t0 = time.monotonic()
    cfg = ws.config()
    jobs = cfg["validate_jobs"] if jobs is None else jobs
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise MINTError(f"jobs must be an integer >= 1, got {jobs!r}", "e.g. --jobs 4")
    sample_size = cfg["sample_size"] if sample_size is None else sample_size
    max_fix = cfg["max_fix_attempts"] if max_fix is None else max_fix
    rows = ws.result_rows()
    if not rows:
        raise MINTError("no tasks to validate", "run `mint run generate` first")
    runner = runner or pick_runner("auto", cfg)
    targets = [
        r
        for r in rows
        if (not only_type or r["type"] == only_type) and (not task_ids or r["task_id"] in task_ids)
    ]
    if task_ids and len(targets) != len(set(task_ids)):
        missing = sorted(set(task_ids) - {r["task_id"] for r in targets})
        raise MINTError(f"unknown task id(s): {missing}", "ids are in out/results.jsonl")

    state = ws.state().get("validate", {"attempts": {}})
    attempts = state["attempts"]
    summary: dict = {
        "levels": sorted(levels),
        "runner": runner.name,
        "jobs": jobs,
        "types": {},
        "infra_errors": [],
    }

    for r in targets:
        refresh_evidence(ws, r)
    for r in targets:
        g = r["gates"]
        if 1 in levels or "not_run" in (g["structure"], g["leak"], g.get("secrets", "not_run")):
            level1(ws, r)  # also for edited packages, whatever --level says
            finalize(r)
    if 1 in levels:
        bad = [
            r
            for r in targets
            if "fail" in (r["gates"]["structure"], r["gates"]["leak"], r["gates"].get("secrets"))
        ]
        summary["level1"] = {
            "checked": len(targets),
            "failed": len(bad),
            "failures": [{"task_id": r["task_id"], "problems": r.get("problems", [])} for r in bad][
                :20
            ],
        }

    detail_dump: dict[str, dict] = {}
    if 2 in levels:
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in targets:
            by_type[r["type"]].append(r)
        # phase 1 (sequential, cheap): decide what each type needs, run nothing yet
        plans: dict[str, dict] = {}
        for ttype, trs in sorted(by_type.items()):
            ok_rows = [
                r
                for r in trs
                if (not is_judged(r))  # judge gates cost calls: opt-in
                and r["gates"]["structure"] == "pass"
                and r["gates"]["leak"] == "pass"
                and r["gates"].get("secrets") == "pass"
                and not r.get("blocked_reason")
            ]
            if attempts.get(ttype, 0) >= max_fix and not task_ids:
                summary["types"][ttype] = {
                    "state": "exhausted",
                    "sampled": [],
                    "passed": [],
                    "failed": [],
                    "infra_error": [],
                    "attempts": attempts.get(ttype, 0),
                    "max_fix_attempts": max_fix,
                    "causes": [],
                    "message": f"type {ttype!r} already used {attempts.get(ttype, 0)}/{max_fix} fix attempts; "
                    "stop fixing it: `mint decide` why, leave its tasks exec_failed / generated in the report, "
                    "or raise --max-fix-attempts deliberately",
                }
                continue
            if task_ids:
                sample = ok_rows
            else:
                # only passes made by the selected runner count; e.g. earlier
                # host-local passes are re-run when Harbor is selected
                def counts(r):
                    return (
                        r["gates"]["exec"] in ("pass", "no_oracle")
                        and r["gates"].get("runner") == runner.name
                    )

                validated = [r for r in ok_rows if counts(r)]
                pending = [r["task_id"] for r in ok_rows if not counts(r)]
                need = max(0, sample_size - len(validated))
                pick = set(seeded_sample(pending, need, cfg["seed"], ttype))
                pick |= {r["task_id"] for r in ok_rows if r.get("changed") and not counts(r)}
                sample = [r for r in ok_rows if r["task_id"] in pick]
            # A saved current result (same package + runner, pass or fail) is never re-run
            # unless the task was named with --task. Results saved by an interrupted run but
            # not yet counted against the type (`exec_unaccounted`) join this run's outcome.
            reused: dict[str, dict] = {}
            in_sample = {r["task_id"] for r in sample}
            if not task_ids:
                for r in ok_rows:
                    if r["task_id"] in in_sample or r.get("exec_unaccounted"):
                        res = _saved_result(ws, r, runner)
                        if res is not None:
                            reused[r["task_id"]] = res
            involved = [r for r in ok_rows if r["task_id"] in in_sample or r["task_id"] in reused]
            plans[ttype] = {"trs": trs, "ok_rows": ok_rows, "involved": involved, "reused": reused}

        runs: list[_TaskRun] = []
        results: dict[str, dict] = {}
        for ttype, pl in plans.items():
            for r in pl["involved"]:
                tid = r["task_id"]
                if tid in pl["reused"]:
                    results[tid] = pl["reused"][tid]
                    progress(
                        f"validate {ttype}/{tid}: saved result reused ({results[tid]['exec']})"
                    )
                else:
                    calls, assemble = _plan_task(ws, r, runner)
                    runs.append(_TaskRun(r, ttype, calls, assemble))
        if runs and not dry:
            ws.write_results(rows)  # level-1 / refresh outcomes survive an early kill

        # phase 2: all gate runs of all tasks on one pool; each finished task is saved at once
        finished = 0

        def on_done(run: _TaskRun, res: dict, secs: float) -> None:
            nonlocal finished
            r = run.row
            r["gates"]["exec"], r["gates"]["runner"] = res["exec"], res["runner"]
            r["gates"]["trials"] = res.get("trials", [])
            r["gates"]["exec_detail"] = {k: v["gate"] for k, v in res["detail"].items()}
            r.pop("changed", None)
            if res["exec"] == "error":
                r.pop("exec_unaccounted", None)
            else:
                r["exec_unaccounted"] = True  # saved, not yet counted against the type's budget
            finalize(r)
            results[r["task_id"]] = res
            detail_dump[r["task_id"]] = res
            if not dry:  # gates file first: a row never points at evidence that is not on disk
                write_json(_gates_file(ws, r["task_id"]), res)
                ws.write_results(rows)
            finished += 1
            progress(
                f"[{finished}/{len(runs)}] {run.ttype}/{r['task_id']}: exec {res['exec']} ({secs:.0f}s)"
            )

        if runs:
            _run_tasks(runs, jobs, on_done)

        # phase 3 (sequential, in type order): the per-type verdict, identical to a serial run
        for ttype, pl in plans.items():
            trs, ok_rows = pl["trs"], pl["ok_rows"]
            res_here = [(r["task_id"], results[r["task_id"]]) for r in pl["involved"]]
            for r in pl["involved"]:
                r.pop("exec_unaccounted", None)
                if results[r["task_id"]]["exec"] == "error":
                    summary["infra_errors"].append({"task_id": r["task_id"], "type": ttype})
            fails = [t for t, res in res_here if res["exec"] == "fail"]
            errs = [t for t, res in res_here if res["exec"] == "error"]
            passed = [t for t, res in res_here if res["exec"] in ("pass", "no_oracle")]
            type_state = "passed"
            if fails:
                type_state = "failed"
                attempts[ttype] = attempts.get(ttype, 0) + 1
                why = f"type failed sampled validation (attempt {attempts[ttype]}/{max_fix}); fix the generator and re-run `mint run generate`"
                for r in trs:
                    if r["gates"]["exec"] not in ("pass", "no_oracle"):
                        r["blocked_reason"] = why
                        finalize(r)
            elif errs:
                type_state = "infra_error"
            elif any(r.get("blocked_reason") for r in trs):
                # nothing was sampled this run, but rows are still exec_failed from an earlier
                # failure: not "passed", and no attempt is spent or refunded
                type_state = "failed"
            elif passed:
                attempts[ttype] = 0
            elif any(r["gates"]["exec"] in ("pass", "no_oracle") for r in ok_rows):
                pass  # already validated earlier; the budget is only refunded by a fresh pass
            else:
                type_state = "nothing_to_validate"
            summary["types"][ttype] = {
                "state": type_state,
                "sampled": [t for t, _ in res_here],
                "resumed": [t for t in pl["reused"]],
                "passed": passed,
                "failed": fails,
                "infra_error": errs,
                "attempts": attempts.get(ttype, 0),
                "max_fix_attempts": max_fix,
                "causes": failure_causes(res_here),
            }
        summary["types"] = dict(sorted(summary["types"].items()))
    if 3 in levels:
        k = 5
        aud = {}
        for ttype in sorted({r["type"] for r in targets}):
            ids = [
                r["task_id"]
                for r in targets
                if r["type"] == ttype and r["status"] not in ("blocked", "exec_failed")
            ]
            aud[ttype] = seeded_sample(ids, k, cfg["seed"], "audit:" + ttype)
        summary["level3_audit_sample"] = aud

    check_invariants(rows)
    if not dry:
        ws.write_results(rows)
        ws.update_state(validate={"attempts": attempts})
        for tid, res in detail_dump.items():
            write_json(ws.internal / "gates" / f"{tid}.json", res)
    summary["review_for_derivability"] = sorted(
        {t for v in summary["types"].values() for t in v["sampled"]}
    )
    summary["warnings"] = {r["task_id"]: r["warnings"] for r in targets if r.get("warnings")}
    summary["any_failure"] = any(
        v["state"] in ("failed", "exhausted") for v in summary["types"].values()
    ) or bool(summary.get("level1", {}).get("failed"))
    summary["any_infra_error"] = bool(summary["infra_errors"])
    summary["wall_seconds"] = round(time.monotonic() - t0, 1)
    if 2 in levels:
        summary["judge_skipped"] = sorted(r["task_id"] for r in targets if is_judged(r))
    return summary
