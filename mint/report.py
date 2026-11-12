"""`mint report`: the user's single touchpoint. Deterministic (no timestamps)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from .output import MINTError
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
            res["queued"] += 1
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

# grading trust that is not independent of the logged run: never counted as ready
WEAK_TRUST = ("imitation", "none")
# an `answer` oracle only counts when the expected answer itself was verified or computed
STRONG_ANSWER_TRUST = ("verified", "computed")


def ready(row: dict) -> tuple[bool, list[str]]:
    """Ready = validated, environment not partial, grading trust independent of the logged
    answer, and no gap affecting execution, fidelity or grading. Computed, never declared."""
    why = []
    if row["status"] != "validated":
        why.append(f"status {row['status']}")
    prov = row.get("provenance") or {}
    if prov.get("env_source") == "partial":
        why.append("environment partial")
    if row.get("gold") in WEAK_TRUST:
        why.append(f"grading trust {row.get('gold')}")
    elif prov.get("oracle") == "answer" and row.get("gold") not in STRONG_ANSWER_TRUST:
        # an oracle that writes the expected answer proves wiring only
        why.append(f"answer oracle with grading trust {row.get('gold')}")
    open_gaps = [g for g in prov.get("gaps") or [] if g.get("affects") != "none"]
    if open_gaps:
        why.append(f"{len(open_gaps)} open gap(s)")
    return not why, why


def _exec_problems(r: dict) -> tuple[list[str], list[str]]:
    """(grading failures, execution/infra errors) as gate names, kept apart: a gate that
    could not run says nothing about the task."""
    g = r["gates"]
    det = g.get("exec_detail") or {}
    fails = [k for k, v in det.items() if v == "fail"]
    errs = [k for k, v in det.items() if v == "error"]
    if g["exec"] == "fail" and not fails:
        fails = ["exec"]
    if g["exec"] == "error" and not errs:
        errs = ["exec"]
    return fails, errs


def _outcome_line(label: str, items: list[dict]) -> str:
    srcs = Counter(x.get("source") or "no source given" for x in items)
    shown = "; ".join(f"{x['text']} [{x.get('source') or 'no source given'}]" for x in items[:3])
    kinds = Counter(x["kind"] for x in items)
    return (
        f"  - {label} x{len(items)} ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))}; "
        f"{len(srcs)} source(s)): {shown}{' ...' if len(items) > 3 else ''}"
    )


def _job_lines(ws: Workspace, by_type: dict) -> list[str]:
    """One row per job in jobs.json (each rebuilt end to end): what exists, how ready, what is
    missing. A job with no task and no candidate is flagged."""
    try:
        jobs = load_jobs(ws)
        tts = load_tasktypes(ws)
    except MINTError as e:
        return ["## Jobs", "", f"- job map unavailable: {e.message}", ""]
    full_types = defaultdict(list)
    for t in tts:
        full_types[t.spec.get("job")].append(t.name)
    queue = read_jsonl(ws.queue)
    L = [
        "## Jobs",
        "",
        "| job | task types | tasks | validated | ready | candidates | blocked | errors | "
        "grading failed | infra errors | env | oracle | grading | gaps |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    missing = []
    covered_src: set = set()
    covered_ids: set = set()
    for jid in jobs:
        names = full_types.get(jid, [])
        rs = [r for n in names for r in by_type.get(n, [])]
        cands = [q for q in queue if q["type"] in names and q["kind"] == "candidate"]
        n_blocked = sum(1 for q in queue if q["type"] in names and q["kind"] == "blocked")
        n_err = sum(1 for q in queue if q["type"] in names and q["kind"] == "exception")
        covered_src.update(x for r in rs for x in r["sources"])
        covered_ids.update(
            i
            for q in queue
            if q["type"] in names and q["kind"] in ("candidate", "blocked")
            for i in q["trajectory_ids"]
        )
        provs = [r.get("provenance") or {} for r in rs]
        env = Counter(p.get("env_source") or "evidence" for p in provs)
        orc = Counter(p.get("oracle") or "answer" for p in provs)
        grad = Counter(
            f"{(p.get('grading') or {}).get('method') or '?'}/{r.get('gold')}"
            for p, r in zip(provs, rs, strict=True)
        )
        gaps = Counter(
            f"{g['affects']}: {g['what']}"
            for src in [*provs, *cands]
            for g in src.get("gaps") or []
        )

        def fmt(c):
            return ", ".join(f"{k} x{v}" for k, v in c.most_common(3)) or "-"

        n_ready = sum(1 for r in rs if ready(r)[0])
        n_val = sum(1 for r in rs if r["status"] == "validated")
        probs = [(r["task_id"], *_exec_problems(r)) for r in rs]
        n_fail = sum(1 for _, f, _e in probs if f)
        n_infra = sum(1 for _, _f, e in probs if e)
        L.append(
            f"| {jid} | {', '.join(names) or '**none**'} | {len(rs)} | {n_val} | {n_ready} | "
            f"{len(cands)} | {n_blocked} | {n_err} | {n_fail} | {n_infra} | {fmt(env)} | "
            f"{fmt(orc)} | {fmt(grad)} | {fmt(gaps)} |"
        )
        for tid, f, e in probs:
            if f:
                L.append(f"  - grading failed {tid}: {', '.join(f)}")
            if e:
                L.append(f"  - infra error (not a grading failure) {tid}: {', '.join(e)}")
        for r, p in list(zip(rs, provs, strict=True))[:10]:
            L.append(
                f"  - {r['task_id']}: oracle {p.get('oracle') or 'answer'}, "
                f"source: {p.get('oracle_source') or 'none given'}"
            )
            oc = p.get("outcomes") or {}
            for label in ("wrong", "alternatives"):
                if oc.get(label):
                    L.append("  " + _outcome_line(label, oc[label]))
        if len(rs) > 10:
            L.append(f"  - ... {len(rs) - 10} more tasks (see each task's task.toml)")
        if not rs and not cands:
            missing.append(jid)
        for q in cands[:5]:
            L.append(
                f"  - candidate {q['group']}: {q['reason']}"
                + (f"; next: {q['next_action']}" if q.get("next_action") else "")
            )
    if missing:
        L += [
            "",
            "**Jobs with no task and no candidate (every job needs at least a candidate):** "
            f"{', '.join(missing)}",
        ]
    if ws.trajectories.exists():
        trajs = load_trajectories(ws)
        uncovered = [
            t["id"]
            for t in trajs
            if t["id"] not in covered_ids and not (set(source_strings([t])) & covered_src)
        ]
        L += [
            "",
            f"Trajectories with no task or candidate: **{len(uncovered)} of {len(trajs)}**"
            + (f" (e.g. {', '.join(uncovered[:10])})" if uncovered else "")
            + ". Each logged run was doing some job: map it, build it, or say why in a decision.",
        ]
    L += [
        "",
        "ready = validated, environment not partial, grading trust not "
        f"{'/'.join(WEAK_TRUST)} (an `answer` oracle also needs verified/computed), no open gaps.",
        "",
    ]
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
    L: list[str] = ["# Mint report", ""]
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
            f"queued (candidate/blocked)={oc.get('queued', 0)}, "
            f"**unaccounted={oc.get('unaccounted', 0)}**"
        )
    L.append("")
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    L += _job_lines(ws, by)

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
    if rows:
        L += [
            "| type | tasks | validated | ready | generated | exec_failed | blocked | gold (trust) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, rs in sorted(by.items()):
            sc, gc = Counter(r["status"] for r in rs), Counter(r["gold"] for r in rs)
            n_ready = sum(1 for r in rs if ready(r)[0])
            gold = ", ".join(f"{k} {v}" for k, v in sorted(gc.items()))
            L.append(
                f"| {t} | {len(rs)} | {sc['validated']} | {n_ready} | {sc['generated']} | "
                f"{sc['exec_failed']} | {sc['blocked']} | {gold} |"
            )
        L += ["", VALIDATED_NOTE, ""]
        L += [f"- {k}: {v}" for k, v in sorted(status_gold(rows).items())]
    else:
        L.append("- no tasks generated")
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
    if any(_exec_problems(r)[1] for r in rows):
        L.append(
            "- **infrastructure errors** (not grading failures): "
            + ", ".join(
                f"{r['task_id']} ({', '.join(_exec_problems(r)[1])})"
                for r in rows
                if _exec_problems(r)[1]
            )
        )
    if any(_exec_problems(r)[0] for r in rows):
        L.append(
            "- grading failures: "
            + ", ".join(
                f"{r['task_id']} ({', '.join(_exec_problems(r)[0])})"
                for r in rows
                if _exec_problems(r)[0]
            )
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

    L += ["## Blocked, failed and candidates", ""]
    blocked = [r for r in rows if r["status"] in ("blocked", "exec_failed")]
    for r in blocked[:30]:
        L.append(f"- {r['status']} task {r['task_id']} ({r['type']}): {r.get('reason')}")
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
    L.append("- spot-check the sample tasks above with `mint task show <id>`")
    L.append("")
    return "\n".join(L)
