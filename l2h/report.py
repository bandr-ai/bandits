"""`l2h report`: the user's single touchpoint. Deterministic (no timestamps)."""

from __future__ import annotations

from collections import Counter, defaultdict

from .stages import source_strings
from .validate import seeded_sample
from .workspace import Workspace, read_json, read_jsonl


def outcomes(ws: Workspace) -> dict:
    trajs = read_jsonl(ws.trajectories)
    used_src = {s for r in ws.result_rows() for s in r["sources"]}
    excluded = {e["trajectory_id"]: e["reason"] for e in read_jsonl(ws.excluded)}
    queued = {i for q in read_jsonl(ws.queue) for i in q["trajectory_ids"]}
    res = Counter()
    unaccounted = []
    for t in trajs:
        if set(source_strings([t])) & used_src:
            res["used"] += 1
        elif t["id"] in queued:
            res["blocked"] += 1
        elif t["id"] in excluded:
            res["excluded"] += 1
        else:
            res["unaccounted"] += 1
            unaccounted.append(t["id"])
    return {"total": len(trajs), **res, "unaccounted_ids": unaccounted[:10]}


def build_report(ws: Workspace) -> str:
    st = ws.state()
    rows = ws.result_rows()
    L: list[str] = ["# Logs to Harbor report", ""]
    p = st.get("parse", {}).get("summary")
    L += ["## Counts per stage", ""]
    if p:
        L += [
            f"- raw files: {p['files']}, raw records: {p['raw_records']}",
            f"- trajectories parsed: {p['trajectories']} (parse errors: {p['parse_errors']})",
            f"- **raw records that landed in no trajectory: {p['records_in_no_trajectory']}**"
            + (f" (e.g. {', '.join(p['orphan_examples'])})" if p["orphan_examples"] else ""),
            "- how runs ended: " + ", ".join(f"{k}={v}" for k, v in sorted(p["ended"].items())),
        ]
        for g in p["error_groups"][:5]:
            L.append(
                f"  - parse error x{g['count']}: {g['cause']} (e.g. {', '.join(g['examples'])})"
            )
    else:
        L.append("- parse: not run")
    oc = outcomes(ws) if ws.trajectories.exists() else None
    if oc:
        L.append(
            f"- outcome per trajectory: used={oc.get('used', 0)}, excluded={oc.get('excluded', 0)}, "
            f"blocked={oc.get('blocked', 0)}, **unaccounted={oc.get('unaccounted', 0)}**"
        )
    L.append("")

    L += ["## Findings and exclusion rates per check", ""]
    c = st.get("checks", {}).get("summary")
    decisions = ws.decision_rows()
    trusted = {
        d.get("check")
        for d in decisions
        if d.get("kind") == "spot_check" and d.get("verdict") == "pass"
    }
    if c:
        L += [
            "| check | version | pass | fail | unknown | fail rate | spot-checked |",
            "|---|---|---|---|---|---|---|",
        ]
        for cid, v in sorted(c["per_check"].items()):
            L.append(
                f"| {cid} | {v['version']} | {v['pass']} | {v['fail']} | {v['unknown']} | {v['fail_rate']:.1%} | {'yes' if cid in trusted else '**NO**'} |"
            )
        for g in c["error_groups"][:5]:
            L.append(f"- check error x{g['count']}: {g['cause']} (e.g. {', '.join(g['examples'])})")
    else:
        L.append("- checks: not run")
    exc = read_jsonl(ws.excluded)
    L += ["", f"Excluded records: {len(exc)}"]
    for reason, n in Counter(e["reason"] for e in exc).most_common(10):
        L.append(f"- {n} x {reason}")
    L.append("")

    L += ["## Tasks per type", ""]
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    if rows:
        L += [
            "| type | tasks | validated | generated | blocked | verified | solved | imitation |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, rs in sorted(by.items()):
            sc, gc = Counter(r["status"] for r in rs), Counter(r["gold"] for r in rs)
            L.append(
                f"| {t} | {len(rs)} | {sc['validated']} | {sc['generated']} | {sc['blocked']} | {gc['verified']} | {gc['solved']} | {gc['imitation']} |"
            )
    else:
        L.append("- no tasks generated")
    L += ["", "## Gate results", ""]
    gc = Counter()
    for r in rows:
        gc[f"structure={r['gates']['structure']}"] += 1
        gc[f"leak={r['gates']['leak']}"] += 1
        gc[f"exec={r['gates']['exec']}"] += 1
    L.append(", ".join(f"{k}: {v}" for k, v in sorted(gc.items())) or "- none")
    runners = {r["gates"].get("runner") for r in rows if r["gates"]["exec"] != "not_run"}
    if runners:
        L.append(f"- exec gates ran with runner(s): {', '.join(sorted(map(str, runners)))}")
    if any(r["gates"]["exec"] == "error" for r in rows):
        L.append(
            "- **infrastructure errors** (not task failures): "
            + ", ".join(r["task_id"] for r in rows if r["gates"]["exec"] == "error")
        )
    auto = [
        r["task_id"]
        for r in rows
        if r["gates"].get("exec_detail")
        and (
            read_json(ws.internal / "gates" / f"{r['task_id']}.json", {})
            .get("detail", {})
            .get("wrong", {})
            .get("auto")
        )
    ]
    if auto:
        L.append(
            f"- wrong-answer gate used an auto-derived wrong answer (weak) for: {', '.join(auto[:10])}"
        )
    L.append("")

    L += ["## Blocked", ""]
    blocked = [r for r in rows if r["status"] == "blocked"]
    for r in blocked[:30]:
        L.append(f"- task {r['task_id']} ({r['type']}): {r.get('reason')}")
    for q in read_jsonl(ws.queue):
        L.append(
            f"- {q['kind']} {q['type']}/{q['group']}: {q['reason']}"
            + (f"; missing: {', '.join(map(str, q['missing']))}" if q["missing"] else "")
        )
    if not blocked and not ws.queue.exists():
        L.append("- none")
    L.append("")

    L += ["## Decisions", ""]
    for d in decisions:
        L.append(f"- [{d.get('kind', 'note')}] {d['what']} — why: {d['why']}")
    if not decisions:
        L.append("- none recorded")
    reviewed = {
        t
        for d in decisions
        if d.get("kind") in ("derivability", "audit")
        for t in d.get("tasks", [])
    }
    sampled = [r["task_id"] for r in rows if r["gates"]["exec"] == "pass"]
    L += [
        "",
        f"Validated tasks without a derivability/audit decision: {len([t for t in sampled if t not in reviewed])} of {len(sampled)}",
        "",
    ]

    L += ["## Sample tasks (spot-check)", ""]
    seed = ws.config()["seed"]
    for t, rs in sorted(by.items()):
        ids = seeded_sample([r["task_id"] for r in rs], 5, seed, "report:" + t)
        L.append(f"- {t}: " + ", ".join(f"out/tasks/{t}/{i}" for i in ids))
    L += ["", "## What to inspect next", ""]
    if oc and oc.get("unaccounted"):
        L.append(
            "- some trajectories have no recorded outcome: " + ", ".join(oc["unaccounted_ids"])
        )
    L.append("- spot-check the sample tasks above with `l2h task show <id>`")
    L.append("")
    return "\n".join(L)
