"""`l2h validate`: the three validation levels."""

from __future__ import annotations

import random
from collections import defaultdict

from . import scan
from .gates import Runner, derive_wrong, pick_runner, run_exec_gates
from .output import L2HError, progress
from .results import check_invariants, finalize
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
        row["problems"] = ["task directory is missing; re-run `l2h run generate`"]
        return
    scan.apply(row, scan.scan_task(d, _meta(ws, row["task_id"]), ws.config()))


def level2_task(ws: Workspace, row: dict, runner: Runner) -> dict:
    d = ws.tasks / row["type"] / row["task_id"]
    m = _meta(ws, row["task_id"])
    spec = read_json(d / "tests" / "expected.json", {})  # absent for a custom verifier
    wrong, auto = m["wrong_answers"], False
    if spec.get("check") == "forbidden":
        wrong = [str(x) for x in spec["expected"]]  # the known-wrong answers must fail
    elif not wrong and "expected" in spec:
        canon = spec["expected"][0] if isinstance(spec["expected"], list) else spec["expected"]
        wrong, auto = [derive_wrong(str(canon), spec.get("check", "exact"))], True
    answer_file = spec.get("file", "answer.txt") if spec else None
    return run_exec_gates(runner, d, answer_file, wrong, m["alternatives"], auto)


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
) -> dict:
    cfg = ws.config()
    sample_size = cfg["sample_size"] if sample_size is None else sample_size
    max_fix = cfg["max_fix_attempts"] if max_fix is None else max_fix
    rows = ws.result_rows()
    if not rows:
        raise L2HError("no tasks to validate", "run `l2h run generate` first")
    runner = runner or pick_runner("auto", cfg)
    targets = [
        r
        for r in rows
        if (not only_type or r["type"] == only_type) and (not task_ids or r["task_id"] in task_ids)
    ]
    if task_ids and len(targets) != len(set(task_ids)):
        missing = sorted(set(task_ids) - {r["task_id"] for r in targets})
        raise L2HError(f"unknown task id(s): {missing}", "ids are in out/results.jsonl")

    state = ws.state().get("validate", {"attempts": {}})
    attempts = state["attempts"]
    summary: dict = {
        "levels": sorted(levels),
        "runner": runner.name,
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
        for ttype, trs in sorted(by_type.items()):
            ok_rows = [
                r
                for r in trs
                if r["gates"]["structure"] == "pass"
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
                    "stop fixing it: `l2h decide` why, leave its tasks exec_failed / generated in the report, "
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
            res_here: list[tuple[str, dict]] = []
            for r in sample:
                progress(f"validate {ttype}/{r['task_id']}")
                res = level2_task(ws, r, runner)
                r["gates"]["exec"], r["gates"]["runner"] = res["exec"], res["runner"]
                r["gates"]["trials"] = res.get("trials", [])
                r["gates"]["exec_detail"] = {k: v["gate"] for k, v in res["detail"].items()}
                r.pop("changed", None)
                finalize(r)
                res_here.append((r["task_id"], res))
                detail_dump[r["task_id"]] = res
                if res["exec"] == "error":
                    summary["infra_errors"].append({"task_id": r["task_id"], "type": ttype})
            fails = [t for t, res in res_here if res["exec"] == "fail"]
            errs = [t for t, res in res_here if res["exec"] == "error"]
            passed = [t for t, res in res_here if res["exec"] in ("pass", "no_oracle")]
            type_state = "passed"
            if fails:
                type_state = "failed"
                attempts[ttype] = attempts.get(ttype, 0) + 1
                why = f"type failed sampled validation (attempt {attempts[ttype]}/{max_fix}); fix the generator and re-run `l2h run generate`"
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
                "passed": passed,
                "failed": fails,
                "infra_error": errs,
                "attempts": attempts.get(ttype, 0),
                "max_fix_attempts": max_fix,
                "causes": failure_causes(res_here),
            }
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
    return summary
