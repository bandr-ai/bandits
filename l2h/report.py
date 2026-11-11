"""`l2h report`: the user's single touchpoint. Deterministic (no timestamps)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from .output import L2HError
from .scripts import load_jobs, load_tasktypes
from .stages import load_trajectories, source_strings
from .validate import seeded_sample
from .workspace import Workspace, read_json, read_jsonl


def outcomes(ws: Workspace) -> dict:
    trajs = load_trajectories(ws)
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


VALIDATED_NOTE = (
    "`validated` means the gates passed (oracle accepted, no-op and wrong answers rejected, "
    "in a real Harbor run); it does not mean the answer is correct."
)

SCOPE_LABEL = {
    "full_job": "full job",
    "subtask": "sub-capability",
    "decision": "decision-from-evidence",
}
SCOPE_NOTE = (
    "scope: `full job` rebuilds the logged agent's whole job (inputs, tools, output format) and grades "
    "its real output; `sub-capability` rebuilds one sub-agent or step with its own inputs and tools; "
    "`decision-from-evidence` gives the evidence the agent gathered and asks for its "
    "decision, not the full job."
)


def _job_lines(ws: Workspace, by_type: dict) -> list[str]:
    """Jobs from jobs.json and what was built for each: unbuilt jobs stay visible."""
    try:
        jobs = load_jobs(ws)
        tts = load_tasktypes(ws)
    except L2HError as e:
        return ["", "## Jobs", "", f"- job map unavailable: {e.message}"]
    types_of: dict[str, list[str]] = defaultdict(list)
    for t in tts:
        types_of[t.spec.get("job")].append(t.name)
    L = [
        "",
        "## Jobs",
        "",
        "| job | scope | parent | task types | tasks |",
        "|---|---|---|---|---|",
    ]
    for jid, j in jobs.items():
        names = types_of.get(jid, [])
        n = sum(len(by_type.get(t, [])) for t in names)
        L.append(
            f"| {jid} | {j['scope']} | {j.get('parent') or ''} | {', '.join(names) or 'none'} | {n} |"
        )
    unbuilt = [jid for jid in jobs if not types_of.get(jid)]
    if unbuilt:
        L += ["", f"Jobs with no task type yet (say why in a decision): {', '.join(unbuilt)}"]
    return L


def status_gold(rows: list[dict]) -> Counter:
    """Counts like {"validated (imitation)": 2}: status always shown with its gold kind."""
    return Counter(f"{r['status']} ({r['gold']})" for r in rows)


def validated_without_decision(ws: Workspace, rows: list[dict]) -> list[str]:
    """Validated tasks nobody has recorded an audit/derivability decision for."""
    reviewed = {
        t
        for d in ws.decision_rows()
        if d.get("kind") in ("derivability", "audit")
        for t in d.get("tasks", [])
    }
    return sorted(
        r["task_id"] for r in rows if r["status"] == "validated" and r["task_id"] not in reviewed
    )


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
        try:
            tts = load_tasktypes(ws)
        except L2HError:
            tts = []
        scopes = {t.name: t.spec.get("task_scope") for t in tts}
        L += [
            "| type | scope | tasks | validated | generated | blocked | verified | solved | imitation | negative |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t, rs in sorted(by.items()):
            sc, gc = Counter(r["status"] for r in rs), Counter(r["gold"] for r in rs)
            scope = SCOPE_LABEL.get(scopes.get(t), "unknown")
            L.append(
                f"| {t} | {scope} | {len(rs)} | {sc['validated']} | {sc['generated']} | {sc['blocked']} | {gc['verified']} | {gc['solved']} | {gc['imitation']} | {gc['negative']} |"
            )
        L += ["", SCOPE_NOTE, "", VALIDATED_NOTE, ""]
        L += [f"- {k}: {v}" for k, v in sorted(status_gold(rows).items())]
    else:
        L.append("- no tasks generated")
    L += _job_lines(ws, by)
    L += ["", "## Gate results", ""]
    gc = Counter()
    for r in rows:
        gc[f"structure={r['gates']['structure']}"] += 1
        gc[f"leak={r['gates']['leak']}"] += 1
        gc[f"secrets={r['gates'].get('secrets', 'not_run')}"] += 1
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

    L += ["## Warnings (do not block)", ""]
    wk: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        for w in r.get("warnings", []):
            wk[w.split(":", 1)[0]].append(f"{r['task_id']}: {w.split(':', 1)[-1].strip()}")
    for kind, items in sorted(wk.items()):
        L.append(f"- {kind} x{len(items)}")
        L += [f"  - {x}" for x in items[:5]]
    allowed = [r for r in rows if r.get("allow")]
    for r in allowed[:20]:
        L.append(f"- allow-listed in {r['task_id']}: {json.dumps(r['allow'], sort_keys=True)}")
    if not wk and not allowed:
        L.append("- none")
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
    undecided = validated_without_decision(ws, rows)
    n_val = sum(1 for r in rows if r["status"] == "validated")
    L += ["", f"Validated tasks without a derivability/audit decision: {len(undecided)} of {n_val}"]
    if undecided:
        L.append(
            "- **WARNING**: not audited, so unknown whether the answer is derivable from the inputs: "
            + ", ".join(undecided[:20])
            + (f" (+{len(undecided) - 20} more)" if len(undecided) > 20 else "")
        )
    L.append("")

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
