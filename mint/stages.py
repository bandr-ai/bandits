"""`mint run parse|checks|generate`: run agent-written code over the corpus."""

from __future__ import annotations

import json
import re
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import harbor, scan
from .models import (
    BLOCK_KINDS,
    Blocked,
    BuildResult,
    Candidate,
    ContractError,
    ParseError,
    Trajectory,
    normalize_finding,
    normalize_gaps,
    normalize_trajectory,
)
from .output import MINTError, progress
from .results import check_invariants, finalize, new_row
from .scripts import load_checks, load_parser, load_tasktypes
from .workspace import (
    Workspace,
    atomic_write,
    clear_dir,
    iter_records,
    list_log_files,
    read_json,
    read_jsonl,
    rel,
    sha,
    write_json,
    write_jsonl,
)


def inputs_digest(*paths: Path) -> str:
    """Digest of the files a stage consumed, to detect a stale downstream stage."""
    return sha("|".join(sha(p.read_bytes()) if p.exists() else "-" for p in paths))


def cause_of(msg: str) -> str:
    """Normalise an error message so errors with one root cause group together."""
    s = re.sub(r"0x[0-9a-fA-F]+|\b[0-9a-f]{8,}\b", "<hex>", msg)
    s = re.sub(r"'[^']{0,80}'|\"[^\"]{0,80}\"", "<str>", s)
    s = re.sub(r"\d+", "N", s)
    return s.strip()[:200]


def group_errors(items: list[tuple[str, str]], examples: int = 3) -> list[dict]:
    """items = [(message, example_id)] -> [{cause, count, examples}] biggest first."""
    by: dict[str, list[str]] = defaultdict(list)
    for msg, ex in items:
        by[cause_of(msg)].append(ex)
    out = [{"cause": c, "count": len(v), "examples": v[:examples]} for c, v in by.items()]
    return sorted(out, key=lambda d: (-d["count"], d["cause"]))


def _stored(summary: dict) -> dict:
    """What we persist: the summary minus the diff, so identical reruns write identical state."""
    return {k: v for k, v in summary.items() if k != "changed_since_last_run"}


def _flat_counts(d: dict, prefix: str = "") -> dict[str, int]:
    """Integer counts of a summary, nested count dicts flattened (ended.done, by_status.generated)."""
    out: dict[str, int] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out[prefix + k] = v
        elif isinstance(v, dict) and k != "changed_since_last_run":
            out.update(_flat_counts(v, f"{prefix}{k}."))
    return out


def delta(prev: dict | None, cur: dict, content: dict | None = None) -> dict:
    """What changed vs the previous run summary: count differences (nested counts included)
    plus `content` (e.g. how many outputs changed with unchanged counts)."""
    if not prev:
        return {"first_run": True}
    p, c = _flat_counts(prev), _flat_counts(cur)
    out: dict[str, Any] = {k: c.get(k, 0) - p.get(k, 0) for k in sorted(p.keys() | c.keys())}
    out = {k: v for k, v in out.items() if v}
    out.update({k: v for k, v in (content or {}).items() if v})
    return out or {"none": "same results as last run"}


# ------------------------------------------------------------------- parse


def raw_inventory(ws: Workspace) -> dict[str, int]:
    logs = ws.logs_root()
    return {rel(logs, p): sum(1 for _ in iter_records(p)) for p in list_log_files(logs)}


def run_parse(ws: Workspace, limit: int | None, dry: bool, no_cache: bool) -> dict:
    logs = ws.logs_root()
    ld = load_parser(ws)
    files = list_log_files(logs)
    fp = sha(
        ld.digest
        + "|".join(f"{rel(logs, p)}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files)
    )
    st = ws.state().get("parse", {})
    if (
        not no_cache
        and limit is None
        and st.get("fingerprint") == fp
        and st.get("complete", True)
        and ws.trajectories.exists()
    ):
        return {
            **st["summary"],
            "cached": True,
            "changed_since_last_run": {"none": "cached: inputs unchanged"},
        }

    inv = raw_inventory(ws)
    trajs, errors, seen = [], [], set()
    n_seen = 0
    try:
        for item in ld.module.parse(files):
            n_seen += 1
            if n_seen % 1000 == 0:
                progress(f"parse: {n_seen} items")
            if isinstance(item, ParseError):
                errors.append((item.error, f"{item.file}#{item.offset}"))
                continue
            try:
                t = normalize_trajectory(item)
                if t["id"] in seen:
                    raise ContractError(f"duplicate trajectory id {t['id']!r}")
                for r in t["source_refs"]:
                    f = r["file"]
                    if f not in inv:
                        raise ContractError(f"source_ref file {f!r} is not under the logs folder")
                    off = r.get("offset")
                    if isinstance(off, int) and not 0 <= off < inv[f]:
                        raise ContractError(
                            f"source_ref {f}#{off} is out of range (file has {inv[f]} records)"
                        )
            except ContractError as e:
                errors.append(
                    (str(e), str(Trajectory(item).get("id")) if isinstance(item, dict) else "?")
                )
                continue
            seen.add(t["id"])
            trajs.append(t)
            if limit and len(trajs) >= limit:
                break
    except MINTError:
        raise
    except Exception as e:
        raise MINTError(
            f"parse() crashed after {n_seen} items: {type(e).__name__}: {e}",
            "fix parse.py; the traceback tail is: "
            + traceback.format_exc().strip().splitlines()[-3:][0],
        ) from e

    covered: set[tuple[str, int]] = set()
    key_files: set[str] = set()
    for t in trajs:
        for r in t["source_refs"]:
            off = r.get("offset")
            if isinstance(off, int):
                covered.add((r["file"], off))
            elif isinstance(off, list):
                covered.update((r["file"], o) for o in off if isinstance(o, int))
            else:
                key_files.add(r["file"])
    orphans = [
        f"{f}#{i}"
        for f, n in inv.items()
        if f not in key_files
        for i in range(n)
        if (f, i) not in covered
    ]
    total = sum(inv.values())
    n_calls = n_unanswered = 0
    stack = list(trajs)
    while stack:
        t = stack.pop()
        for c in t.calls():
            n_calls += 1
            n_unanswered += c.result is None
        stack += t.subagents
    summary = {
        "files": len(inv),
        "raw_records": total,
        "trajectories": len(trajs),
        "parse_errors": len(errors),
        "records_in_no_trajectory": len(orphans),
        "orphan_examples": orphans[:5],
        "key_ref_files_not_counted": sorted(key_files),
        "ended": dict(Counter(t["ended"] for t in trajs)),
        "tool_calls": n_calls,
        "tool_calls_unanswered": n_unanswered,
        "error_groups": group_errors(errors),
        "limited": limit is not None,
    }
    content = {}
    if st.get("summary") and ws.trajectories.exists():
        before = {}
        for line in ws.trajectories.read_text().splitlines():
            if line.strip():
                before[str(Trajectory(json.loads(line)).get("id"))] = sha(line.encode())
        content["trajectories_changed"] = sum(
            1
            for t in trajs
            if t["id"] in before
            and before[t["id"]] != sha(json.dumps(t, sort_keys=True, default=str).encode())
        )
    summary["changed_since_last_run"] = delta(st.get("summary"), summary, content)
    if not dry:
        write_jsonl(ws.trajectories, trajs)
        ws.update_state(
            parse={
                "fingerprint": fp,
                "complete": limit is None,
                "script": ld.digest,
                "summary": _stored(summary),
            }
        )
    return summary


# ------------------------------------------------------------------ checks


def load_trajectories(ws: Workspace) -> list[dict]:
    if not ws.trajectories.exists():
        raise MINTError("out/trajectories.jsonl not found", "run `mint run parse` first")
    return [Trajectory(t) for t in read_jsonl(ws.trajectories)]


def run_checks(ws: Workspace, limit: int | None, dry: bool) -> dict:
    trajs = load_trajectories(ws)
    if limit:
        trajs = trajs[:limit]
    ld, checks = load_checks(ws)
    rows, errs = [], []
    for t in trajs:
        for c in checks:
            try:
                f = normalize_finding(c.fn(t))
            except Exception as e:  # a crashing check is an observation gap, not a verdict
                msg = f"{type(e).__name__}: {e}"
                errs.append((f"check {c.id}: {msg}", t["id"]))
                from .models import Finding

                f = Finding("unknown", [], f"check crashed: {msg}")
            rows.append(
                {
                    "trajectory_id": t["id"],
                    "check": c.id,
                    "version": c.version,
                    "result": f.result,
                    "evidence": f.evidence,
                    "note": f.note,
                }
            )
    per: dict[str, Any] = {}
    for c in checks:
        cnt = Counter(r["result"] for r in rows if r["check"] == c.id)
        n = sum(cnt.values())
        per[c.id] = {
            "version": c.version,
            "pass": cnt["pass"],
            "fail": cnt["fail"],
            "unknown": cnt["unknown"],
            "fail_rate": round(cnt["fail"] / n, 4) if n else 0.0,
        }
    prev = ws.state().get("checks", {}).get("summary", {})
    changed = {}
    for cid, v in per.items():
        p = prev.get("per_check", {}).get(cid)
        if p is None:
            changed[cid] = "new check"
        elif p["fail"] != v["fail"] or p["version"] != v["version"]:
            changed[cid] = f"fail {p['fail']} -> {v['fail']}"
    summary = {
        "trajectories": len(trajs),
        "checks": len(checks),
        "findings": len(rows),
        "per_check": per,
        "error_groups": group_errors(errs),
        "changed_since_last_run": changed or {"none": "same results as last run"},
        "limited": limit is not None,
    }
    if not dry:
        write_jsonl(ws.findings, rows)
        ws.update_state(
            checks={
                "script": ld.digest,
                "summary": _stored(summary),
                "inputs": inputs_digest(ws.trajectories),
            }
        )
    return summary


# ---------------------------------------------------------------- generate


def findings_by_traj(ws: Workspace) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(dict)
    for r in read_jsonl(ws.findings):
        out[r["trajectory_id"]][r["check"]] = {k: r[k] for k in ("result", "evidence", "note")}
    return out


def source_strings(group: list[dict]) -> list[str]:
    out = []
    for t in group:
        for r in t["source_refs"]:
            off = r.get("offset", r.get("key"))
            out.append(f"{r['file']}#{off}")
    return sorted(set(out))


def package_hash(task_dir: Path, meta: dict | None = None) -> str:
    """Fingerprint of what the gates judged: the package bytes plus the wrong answers and
    alternatives the exec gates try (they live outside the package)."""
    h = []
    for p in sorted(task_dir.rglob("*")):
        if p.is_file():
            h.append(f"{p.relative_to(task_dir).as_posix()}:{sha(p.read_bytes())}")
    if meta and (meta.get("wrong_answers") or meta.get("alternatives")):
        h.append("wrong:" + json.dumps(meta.get("wrong_answers", [])))
        h.append("alt:" + json.dumps(meta.get("alternatives", [])))
    return sha("\n".join(h))


def run_generate(ws: Workspace, limit: int | None, dry: bool, only_type: str | None) -> dict:
    trajs = load_trajectories(ws)
    # --limit picks records, but a task is always rebuilt from its WHOLE group:
    # groups touching a selected record are built with every member.
    selected = {t["id"] for t in trajs[:limit]} if limit else None
    scope_ids = set(selected) if selected is not None else None
    if not ws.findings.exists():
        progress("note: no findings yet; applies() will see empty findings")
    fmap = findings_by_traj(ws)
    types = load_tasktypes(ws)
    cfg = ws.config()
    old_rows = ws.result_rows()
    old_by_id = {r["task_id"]: r for r in old_rows}
    # captured now: an unchanged row is reused (and its generator updated) in place below
    old_sig = {r["task_id"]: (r.get("content_hash"), r.get("generator")) for r in old_rows}
    built_types = [t for t in types if not only_type or t.name == only_type]
    if only_type and not built_types:
        raise MINTError(f"no task type {only_type!r}", "list task types with `mint status`")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="mint-gen-") if dry else None
    tasks_root = Path(tmp_ctx.name) / "tasks" if dry else ws.tasks
    staging_root = Path(tempfile.mkdtemp(prefix="mint-stage-"))
    new_rows, queue, excluded_reasons = [], [], {}
    applied_any: set[str] = set()
    ids_seen: dict[str, str] = {}
    taskmeta: dict[str, dict] = {}

    try:
        for tt in types:
            reasons: dict[str, str] = {}
            groups: dict[str, list[dict]] = defaultdict(list)
            for t in trajs:
                try:
                    res = tt.loaded.module.applies(t, fmap.get(t["id"], {}))
                    ok, why = res if isinstance(res, tuple) else (bool(res), "")
                except Exception as e:
                    queue.append(
                        _q(
                            tt.name,
                            t["id"],
                            "exception",
                            f"applies() crashed: {type(e).__name__}: {e}",
                            [t["id"]],
                        )
                    )
                    continue
                if ok:
                    applied_any.add(t["id"])
                    try:
                        groups[str(tt.loaded.module.group_key(t))].append(t)
                    except Exception as e:
                        queue.append(
                            _q(
                                tt.name,
                                t["id"],
                                "exception",
                                f"group_key() crashed: {type(e).__name__}: {e}",
                                [t["id"]],
                            )
                        )
                elif why:
                    reasons[t["id"]] = why
            for tid, why in reasons.items():
                excluded_reasons.setdefault(tid, []).append(f"{tt.name}: {why}")
            if only_type and tt.name != only_type:
                continue
            progress(f"generate {tt.name}: {len(groups)} group(s)")
            for gk in sorted(groups):
                group = groups[gk]
                if selected is not None:
                    if not any(t["id"] in selected for t in group):
                        continue
                    scope_ids.update(t["id"] for t in group)
                srcs = source_strings(group)
                stage = staging_root / tt.name / sha(gk)[:12]
                stage.mkdir(parents=True, exist_ok=True)
                try:
                    res = tt.loaded.module.build(group, stage)
                    if isinstance(res, Blocked):
                        if res.kind not in BLOCK_KINDS:
                            raise ContractError(
                                f"a job may be Blocked only for {list(BLOCK_KINDS)} "
                                f"(got kind {res.kind!r}: {res.reason}); return "
                                "Candidate(reason, gaps=[...], next_action=...) or build it "
                                "with its gaps instead"
                            )
                        queue.append(
                            _q(
                                tt.name,
                                gk,
                                "blocked",
                                res.reason,
                                [t["id"] for t in group],
                                res.missing,
                            )
                        )
                        continue
                    if isinstance(res, Candidate):
                        q = _q(
                            tt.name,
                            gk,
                            "candidate",
                            res.reason,
                            [t["id"] for t in group],
                            [g["what"] for g in normalize_gaps(res.gaps)],
                        )
                        q["gaps"] = normalize_gaps(res.gaps)
                        q["next_action"] = res.next_action
                        q["job"] = tt.spec.get("job")
                        queue.append(q)
                        continue
                    if not isinstance(res, BuildResult):
                        raise ContractError(
                            f"build() returned {type(res).__name__}; "
                            "expected BuildResult, Candidate or Blocked"
                        )
                    if res.task_id in ids_seen:
                        raise ContractError(
                            f"task_id {res.task_id!r} already produced by {ids_seen[res.task_id]}"
                        )
                    ids_seen[res.task_id] = tt.name
                    dest = tasks_root / tt.name / res.task_id
                    # generator-written files in the stage dir are carried over
                    extra = {}
                    for p in sorted(stage.rglob("*")):
                        if p.is_file():
                            extra[p.relative_to(stage).as_posix()] = p.read_bytes()
                    res.files = {**extra, **res.files}
                    harbor.materialize(res, dest, tt.name, tt.generator, srcs, spec=tt.spec)
                except Exception as e:
                    detail = f"{type(e).__name__}: {e}"
                    queue.append(
                        _q(
                            tt.name,
                            gk,
                            "exception",
                            f"build() failed: {detail}",
                            [t["id"] for t in group],
                        )
                    )
                    continue
                row = new_row(res.task_id, tt.name, srcs, tt.generator, res.gold_type)
                row["assumptions"] = list(res.assumptions)
                row["provenance"] = {
                    "job": tt.spec.get("job"),
                    "oracle": res.oracle_kind,
                    "env_source": res.env_source,
                    "grading": {
                        "method": res.grading_method,
                        "trust": res.gold_type,
                        "reason": res.grading_reason,
                    },
                    "gaps": res.gaps,
                }
                hidden = [str(h) for h in res.hidden]
                # a custom verifier lists its own hidden values
                if res.answer is not None:
                    hidden.append(res.answer.canonical())
                    if isinstance(res.answer.expected, list):
                        hidden += [str(x) for x in res.answer.expected]
                    hidden += [a for a in res.alternatives if isinstance(a, str)]
                meta = {
                    "wrong_answers": list(res.wrong_answers),
                    "alternatives": list(res.alternatives),
                    "hidden": hidden,
                    "leak_allow": list(res.leak_allow),
                    "secret_allow": list(res.secret_allow),
                }
                row["content_hash"] = package_hash(dest, meta)
                scan.apply(row, scan.scan_task(dest, meta, cfg))
                allow = {
                    k: list(v)
                    for k, v in (("leak", res.leak_allow), ("secrets", res.secret_allow))
                    if v
                }
                if allow:
                    row["allow"] = allow  # explicit allow-lists stay visible in results
                old = old_by_id.get(res.task_id)
                if (
                    old
                    and old.get("content_hash") == row["content_hash"]
                    and not old.get("blocked_reason")
                ):
                    fresh = row
                    row = old  # unchanged: keep exec evidence (idempotent rerun)...
                    row["sources"], row["generator"] = srcs, tt.generator
                    # ...but structure/leak are re-judged now (hidden values may have changed)
                    for g in ("structure", "leak", "secrets"):
                        row["gates"][g] = fresh["gates"][g]
                    for k in ("problems", "warnings", "allow"):
                        if fresh.get(k):
                            row[k] = fresh[k]
                        else:
                            row.pop(k, None)
                elif old:
                    row["changed"] = True  # same id, new content: revalidate
                taskmeta[res.task_id] = meta
                new_rows.append(finalize(row))
    finally:
        clear_dir(staging_root)

    partial = bool(limit)
    scope = trajs if scope_ids is None else [t for t in trajs if t["id"] in scope_ids]
    built_names = {tt.name for tt in built_types}
    new_ids = {r["task_id"] for r in new_rows}
    dropped: list[dict] = []
    if partial:
        # a limited run replaces only its own slice; but a selected group whose rebuild
        # failed or was blocked must not keep its old (possibly validated) row
        scope_src = {x for t in scope for x in source_strings([t])}
        dropped = [
            r
            for r in old_rows
            if r["type"] in built_names
            and r["task_id"] not in new_ids
            and set(r["sources"]) & scope_src
        ]
        gone = new_ids | {r["task_id"] for r in dropped}
        kept = [r for r in old_rows if r["type"] not in built_names or r["task_id"] not in gone]
    else:
        kept = [r for r in old_rows if only_type and r["type"] != only_type]
    clash = sorted(new_ids & {r["task_id"] for r in kept})
    if clash:
        owners = {r["task_id"]: r["type"] for r in kept}
        raise MINTError(
            f"task id(s) {clash[:3]} are produced by this run but already owned by another type "
            f"({owners[clash[0]]!r}); ids must be unique across types",
            "include the type or a stable prefix in the task_id your generator returns",
        )
    all_rows = kept + new_rows
    check_invariants(all_rows)

    used = {s for r in new_rows for s in r["sources"]}
    queued_ids = {i for q in queue for i in q["trajectory_ids"]}
    excluded = []
    if not only_type:
        for t in scope:
            if t["id"] not in applied_any:
                excluded.append(
                    {
                        "trajectory_id": t["id"],
                        "sources": source_strings([t]),
                        "reason": "; ".join(excluded_reasons.get(t["id"], []))
                        or "no task type applies (applies() returned False for every type)",
                    }
                )
    counts = Counter(r["status"] for r in new_rows)
    by_type = Counter(r["type"] for r in new_rows)
    prev = ws.state().get("generate", {}).get("summary", {})
    rebuilt = [(old_sig[r["task_id"]], r) for r in new_rows if r["task_id"] in old_sig]
    summary = {
        "trajectories": len(scope),
        "tasks": len(new_rows),
        "by_status": dict(counts),
        "by_type": dict(by_type),
        "excluded": len(excluded),
        "queued": len(queue),
        "queue_groups": group_errors([(q["reason"], q["group"]) for q in queue]),
        "structure_or_leak_failures": [
            {"task_id": r["task_id"], "problems": r.get("problems", [])}
            for r in new_rows
            if "fail" in (r["gates"]["structure"], r["gates"]["leak"], r["gates"].get("secrets"))
        ][:20],
        "sources_used": len(used),
        "trajectories_in_queue": len(queued_ids),
        "limited": limit is not None or bool(only_type),
        "dropped_stale": len(dropped),
    }
    summary["changed_since_last_run"] = delta(
        prev,
        summary,
        {
            "tasks_content_changed": sum(1 for (h, _), r in rebuilt if h != r["content_hash"]),
            "tasks_generator_changed": sum(1 for (_, g), r in rebuilt if g != r["generator"]),
        },
    )
    if dry:
        tmp_ctx.cleanup()
        return summary

    # persist: swap in new task dirs, drop stale ones for rebuilt types
    for r in dropped:
        clear_dir(ws.tasks / r["type"] / r["task_id"])
    for tt in [] if partial else built_types:
        keep = {r["task_id"] for r in new_rows if r["type"] == tt.name}
        tdir = ws.tasks / tt.name
        if tdir.is_dir():
            for d in tdir.iterdir():
                if d.is_dir() and d.name not in keep:
                    clear_dir(d)
    ws.write_results(all_rows)
    meta_dir = ws.internal / "taskmeta"
    for tid, m in taskmeta.items():
        write_json(meta_dir / f"{tid}.json", m)
    seen_t = {t["id"] for t in scope}
    if partial:
        old_ex = [
            e for e in read_jsonl(ws.excluded) if only_type or e["trajectory_id"] not in seen_t
        ]
        write_jsonl(ws.excluded, old_ex + excluded)
        old_q = [
            q
            for q in read_jsonl(ws.queue)
            if not (set(q["trajectory_ids"]) & seen_t and q["type"] in built_names)
        ]
        write_jsonl(ws.queue, old_q + queue)
    elif not only_type:
        write_jsonl(ws.excluded, excluded)
        write_jsonl(ws.queue, queue)
    else:
        old_q = [q for q in read_jsonl(ws.queue) if q["type"] != only_type]
        write_jsonl(ws.queue, old_q + queue)
    ws.update_state(
        generate={
            "summary": _stored(summary),
            "inputs": inputs_digest(ws.trajectories, ws.findings, *[t.loaded.path for t in types]),
        }
    )
    return summary


def _q(
    ttype: str, group: str, kind: str, reason: str, tids: list[str], missing: list | None = None
) -> dict:
    return {
        "type": ttype,
        "group": group,
        "kind": kind,
        "reason": reason,
        "missing": missing or [],
        "trajectory_ids": tids,
    }


__all__ = [
    "run_parse",
    "run_checks",
    "run_generate",
    "group_errors",
    "cause_of",
    "atomic_write",
    "read_json",
]
