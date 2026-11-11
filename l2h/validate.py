"""`l2h validate`: the three validation levels."""

from __future__ import annotations

import random
from collections import defaultdict

from . import harbor, leak
from .gates import LocalRunner, Runner, derive_wrong, run_exec_gates
from .output import L2HError, progress
from .results import check_invariants, finalize
from .stages import group_errors, package_hash
from .workspace import Workspace, read_json, write_json


def _meta(ws: Workspace, tid: str) -> dict:
    return read_json(
        ws.internal / "taskmeta" / f"{tid}.json",
        {"wrong_answers": [], "alternatives": [], "hidden": [], "leak_allow": []},
    )


def seeded_sample(items: list, k: int, seed: int, salt: str) -> list:
    rng = random.Random(f"{seed}:{salt}")
    pool = sorted(items)
    return sorted(rng.sample(pool, min(k, len(pool))))


def refresh_evidence(ws: Workspace, row: dict) -> None:
    """Drop exec evidence if the package on disk no longer matches what was gated."""
    d = ws.tasks / row["type"] / row["task_id"]
    if not d.is_dir():
        return
    now = package_hash(d)
    if row.get("content_hash") and now != row["content_hash"]:
        row["gates"]["exec"] = "not_run"
        row["gates"].pop("runner", None)
        row["gates"].pop("exec_detail", None)
        row["changed"] = True
        finalize(row)
    row["content_hash"] = now


def level1(ws: Workspace, row: dict) -> None:
    """Structure + leak for one task row (mutates row)."""
    cfg = ws.config()
    d = ws.tasks / row["type"] / row["task_id"]
    if not d.is_dir():
        row["gates"]["structure"] = "fail"
        row["problems"] = ["task directory is missing; re-run `l2h run generate`"]
        return
    probs = harbor.structure_check(d)
    m = _meta(ws, row["task_id"])
    lk = leak.scan(d, m["hidden"], m["leak_allow"], cfg["leak"]["min_len"], cfg["leak"]["run_len"])
    row["gates"]["structure"] = "fail" if probs else "pass"
    row["gates"]["leak"] = "fail" if lk else "pass"
    row["problems"] = probs + [f"leak: {x.kind} in {x.where}: {x.detail}" for x in lk]
    if not row["problems"]:
        row.pop("problems")


def level2_task(ws: Workspace, row: dict, runner: Runner) -> dict:
    d = ws.tasks / row["type"] / row["task_id"]
    m = _meta(ws, row["task_id"])
    spec = read_json(d / "tests" / "expected.json", {})
    wrong, auto = m["wrong_answers"], False
    if not wrong:
        canon = (
            spec["expected"][0] if isinstance(spec.get("expected"), list) else spec.get("expected")
        )
        wrong, auto = [derive_wrong(str(canon), spec.get("check", "exact"))], True
    return run_exec_gates(runner, d, spec.get("file", "answer.txt"), wrong, m["alternatives"], auto)


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
    runner = runner or LocalRunner(timeout=cfg["gate_timeout_sec"])
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
    if 1 in levels:
        for r in targets:
            level1(ws, r)
            finalize(r)
        bad = [
            r for r in targets if r["gates"]["structure"] == "fail" or r["gates"]["leak"] == "fail"
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
                and not r.get("blocked_reason")
            ]
            if attempts.get(ttype, 0) >= max_fix and not task_ids:
                raise L2HError(
                    f"type {ttype!r} already used {attempts[ttype]}/{max_fix} fix attempts",
                    "stop fixing it: `l2h decide` why, leave it blocked in the report, or raise --max-fix-attempts deliberately",
                )
            if task_ids:
                sample = ok_rows
            else:
                validated = [r for r in ok_rows if r["gates"]["exec"] == "pass"]
                pending = [r["task_id"] for r in ok_rows if r["gates"]["exec"] != "pass"]
                need = max(0, sample_size - len(validated))
                pick = set(seeded_sample(pending, need, cfg["seed"], ttype))
                pick |= {
                    r["task_id"]
                    for r in ok_rows
                    if r.get("changed") and r["gates"]["exec"] != "pass"
                }
                sample = [r for r in ok_rows if r["task_id"] in pick]
            res_here: list[tuple[str, dict]] = []
            for r in sample:
                progress(f"validate {ttype}/{r['task_id']}")
                res = level2_task(ws, r, runner)
                r["gates"]["exec"], r["gates"]["runner"] = res["exec"], res["runner"]
                r["gates"]["exec_detail"] = {k: v["gate"] for k, v in res["detail"].items()}
                r.pop("changed", None)
                finalize(r)
                res_here.append((r["task_id"], res))
                detail_dump[r["task_id"]] = res
                if res["exec"] == "error":
                    summary["infra_errors"].append({"task_id": r["task_id"], "type": ttype})
            fails = [t for t, res in res_here if res["exec"] == "fail"]
            errs = [t for t, res in res_here if res["exec"] == "error"]
            passed = [t for t, res in res_here if res["exec"] == "pass"]
            type_state = "passed"
            if fails:
                type_state = "failed"
                attempts[ttype] = attempts.get(ttype, 0) + 1
                why = f"type failed sampled validation (attempt {attempts[ttype]}/{max_fix}); fix the generator and re-run `l2h run generate`"
                for r in trs:
                    if r["gates"]["exec"] != "pass":
                        r["blocked_reason"] = why
                        finalize(r)
            elif errs:
                type_state = "infra_error"
            elif res_here or any(r["gates"]["exec"] == "pass" for r in ok_rows):
                attempts[ttype] = 0
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
            ids = [r["task_id"] for r in targets if r["type"] == ttype and r["status"] != "blocked"]
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
    summary["any_failure"] = any(v["state"] == "failed" for v in summary["types"].values()) or bool(
        summary.get("level1", {}).get("failed")
    )
    summary["any_infra_error"] = bool(summary["infra_errors"])
    return summary
