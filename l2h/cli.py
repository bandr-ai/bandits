"""The `l2h` command line. Each command returns a Result; main() renders it."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from importlib import resources
from pathlib import Path

from . import __version__, explore, stages
from . import report as report_mod
from . import validate as validate_mod
from .gates import LocalRunner, pick_runner
from .harbor import ANSWER_DIR
from .models import (
    ATIF_VERSION,
    CHECK_KINDS,
    ENDED,
    GATE_STATES,
    GOLD_TYPES,
    NORMALIZERS,
    STATUSES,
    TASK_SCOPES,
    ContractError,
)
from .output import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    L2HError,
    Result,
    UsageError,
    render,
    render_error,
)
from .results import finalize
from .scan import scan_task
from .scripts import scaffold
from .workspace import (
    CONFIG_NAME,
    DEFAULT_CONFIG,
    Workspace,
    atomic_write,
    find_workspace,
    list_log_files,
    read_json,
    read_jsonl,
    rel,
    sha,
    write_json,
)

NOTES_TEMPLATE = """# Source notes

Fill this in while reading samples (`l2h sample`). Keep it current.

## What is a record / a trajectory?
## Boundaries (sessions, calls, tools, sub-agents, attachments)
## What the requested work was (separate from what the logged agent did)
## What is recorded vs missing
## Samples I read
## Open questions
"""


# ----------------------------------------------------------------- commands


def cmd_init(a, ws: Workspace) -> Result:
    logs = Path(a.logs).expanduser().resolve()
    if not logs.exists():
        raise UsageError(
            f"logs path {logs} does not exist",
            "pass the folder (or file) that holds the agent logs",
        )
    try:
        files = list_log_files(logs)
        for p in files[:50]:
            with open(p, "rb") as f:
                f.read(1)
    except OSError as e:
        raise L2HError(
            f"logs at {logs} are not readable: {e}", "fix permissions (l2h only ever reads them)"
        ) from e
    tools = {"docker": bool(shutil.which("docker")), "harbor": bool(shutil.which("harbor"))}
    missing = [k for k, v in tools.items() if not v]
    hints = {
        "docker": "install Docker Desktop / docker engine and make sure `docker info` works",
        "harbor": "install Harbor: `uv tool install harbor` (see harborframework docs)",
    }
    if missing and not a.allow_missing_runtime:
        raise L2HError(
            f"missing on this machine: {', '.join(missing)}",
            "; ".join(hints[m] for m in missing)
            + "; or pass --allow-missing-runtime to proceed with local verifier checks only",
        )
    cfg_path = ws.config_path
    existing = read_json(cfg_path)
    if existing and existing.get("logs") != str(logs):
        raise L2HError(
            f"{ws.root} is already a workspace for {existing.get('logs')}",
            "use a new workspace folder, or edit l2h.json deliberately",
        )
    inv = {}
    for p in files:
        from .workspace import iter_records

        inv[rel(logs, p)] = {
            "size": p.stat().st_size,
            "ext": p.suffix.lower() or "(none)",
            "records": sum(1 for _ in iter_records(p)),
        }
    by_ext = Counter(v["ext"] for v in inv.values())
    data = {
        "workspace": str(ws.root),
        "logs": str(logs),
        "already_initialized": bool(existing),
        "files": len(inv),
        "records": sum(v["records"] for v in inv.values()),
        "bytes": sum(v["size"] for v in inv.values()),
        "by_extension": dict(by_ext),
        "tools": tools,
        "would_write": []
        if existing
        else [CONFIG_NAME, "source_notes.md", "decisions.jsonl", "out/", "tasktypes/"],
    }
    if not a.dry_run:
        ws.root.mkdir(parents=True, exist_ok=True)
        for d in (ws.out, ws.tasks, ws.internal, ws.tasktypes):
            d.mkdir(parents=True, exist_ok=True)
        if not existing:
            cfg = {**DEFAULT_CONFIG, "logs": str(logs)}
            write_json(cfg_path, cfg)
        if not ws.notes.exists():
            atomic_write(ws.notes, NOTES_TEMPLATE)
        if not ws.decisions.exists():
            atomic_write(ws.decisions, "")
        write_json(ws.internal / "inventory.json", inv)
    text = (
        f"{'already initialized' if existing else 'initialized'}: {ws.root}\n"
        f"logs: {logs} ({data['files']} files, {data['records']} records, {data['bytes']} bytes)\n"
        f"types: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_ext.items()))
        + (f"\nmissing runtime (allowed): {', '.join(missing)}" if missing else "")
        + ("\n(dry-run: nothing written)" if a.dry_run else "")
    )
    return Result(
        data, text, ["l2h sample  # read native samples, then write parse.py with `l2h new parser`"]
    )


def _stage_report(ws: Workspace) -> dict:
    st = ws.state()
    rows = ws.result_rows()
    notes_done = ws.notes.exists() and ws.notes.read_text() != NOTES_TEMPLATE
    parse = st.get("parse")
    stale_parse = bool(
        parse and ws.parse_py.exists() and sha(ws.parse_py.read_text()) != parse["script"]
    )
    chk = st.get("checks")
    stale_checks = bool(
        chk
        and (
            (ws.checks_py.exists() and sha(ws.checks_py.read_text()) != chk["script"])
            or stages.inputs_digest(ws.trajectories) != chk.get("inputs")
        )
    )
    types = (
        sorted(p.stem for p in ws.tasktypes.glob("*.py") if not p.name.startswith("_"))
        if ws.tasktypes.is_dir()
        else []
    )
    gen = st.get("generate")
    stale_gen = False
    if gen:
        paths = [ws.tasktypes / f"{t}.py" for t in types]
        stale_gen = stages.inputs_digest(ws.trajectories, ws.findings, *paths) != gen.get("inputs")
    sc = Counter(r["status"] for r in rows)
    stages_ = {
        "1_understand": {
            "done": bool(notes_done and parse and not stale_parse),
            "notes_written": notes_done,
            "parser_exists": ws.parse_py.exists(),
            "parse_run": bool(parse),
            "stale": stale_parse,
        },
        "2_findings": {
            "done": bool(chk and not stale_checks),
            "checks_exists": ws.checks_py.exists(),
            "run": bool(chk),
            "stale": stale_checks,
        },
        "3_task_types": {"done": bool(types), "types": types},
        "4_generate": {
            "done": bool(gen and not stale_gen),
            "run": bool(gen),
            "stale": stale_gen,
            "tasks": len(rows),
        },
        "5_validate": {
            "done": bool(rows) and any(r["status"] == "validated" for r in rows),
            "local_only": bool(rows)
            and not any(r["status"] == "validated" for r in rows)
            and any(r["gates"]["exec"] == "pass" for r in rows),
            "by_status": dict(sc),
            "by_status_gold": dict(report_mod.status_gold(rows)),
            "validated_without_decision": report_mod.validated_without_decision(ws, rows),
        },
        "6_report": {"done": ws.report.exists()},
    }
    blockers = []
    p = (parse or {}).get("summary", {})
    if p.get("parse_errors"):
        blockers.append(f"{p['parse_errors']} parse errors ({p['error_groups'][0]['cause']!r} ...)")
    if p.get("records_in_no_trajectory"):
        blockers.append(f"{p['records_in_no_trajectory']} raw records landed in no trajectory")
    q = read_jsonl(ws.queue)
    if q:
        blockers.append(f"{len(q)} queued blocked/exception groups (out/queue.jsonl)")
    if sc["blocked"]:
        blockers.append(f"{sc['blocked']} blocked tasks")
    errs = [r["task_id"] for r in rows if r["gates"]["exec"] == "error"]
    if errs:
        blockers.append(f"{len(errs)} tasks with infrastructure errors in exec gates")
    for t, n in st.get("validate", {}).get("attempts", {}).items():
        if n:
            blockers.append(f"type {t}: {n} failed fix attempts")
    return {"stages": stages_, "blockers": blockers}


def _next_for(ws: Workspace, rep: dict) -> list[str]:
    s = rep["stages"]
    if not ws.config_path.exists():
        return ["l2h init <logs> <workspace>"]
    if not s["1_understand"]["notes_written"] and not s["1_understand"]["parser_exists"]:
        return ["l2h sample"]
    if not s["1_understand"]["parser_exists"]:
        return ["l2h new parser  # then fill in parse.py and source_notes.md"]
    if s["1_understand"]["stale"] or not s["1_understand"]["parse_run"]:
        return ["l2h run parse"]
    if not s["1_understand"]["notes_written"]:
        return ["write source_notes.md (what a record is, boundaries, what is missing)"]
    if not s["2_findings"]["checks_exists"]:
        return ["l2h new check <name>"]
    if not s["2_findings"]["done"]:
        return ["l2h run checks"]
    if not s["3_task_types"]["done"]:
        return ["l2h new tasktype <name>"]
    if not s["4_generate"]["done"]:
        return ["l2h run generate --limit 5  # build a few varied instances first"]
    if not s["5_validate"]["done"]:
        if s["5_validate"].get("local_only"):
            return [
                "install Harbor and start Docker, then `l2h validate --runner harbor` "
                "(host-local passes leave tasks `generated`)"
            ]
        return ["l2h validate"]
    if not s["6_report"]["done"]:
        return ["l2h report"]
    return ["l2h report  # refresh report.md, then review it"]


def cmd_status(a, ws: Workspace) -> Result:
    ws.config()
    rep = _stage_report(ws)
    lines = []
    for k, v in rep["stages"].items():
        flags = []
        if v.get("stale"):
            flags.append("STALE: inputs changed, re-run")
        lines.append(
            f"[{'x' if v['done'] else ' '}] {k}" + (f"  ({'; '.join(flags)})" if flags else "")
        )
    v5 = rep["stages"]["5_validate"]
    if v5["by_status"]:
        lines.append(
            "tasks: " + ", ".join(f"{k}={n}" for k, n in sorted(v5["by_status_gold"].items()))
        )
    if v5["by_status"].get("validated"):
        lines.append(f"note: {report_mod.VALIDATED_NOTE}")
        if v5["validated_without_decision"]:
            lines.append(
                f"warning: {len(v5['validated_without_decision'])} validated task(s) have no "
                "audit/derivability decision (`l2h decide ... --kind derivability --task <id>`)"
            )
    if rep["blockers"]:
        lines.append("blockers:")
        lines += [f"  - {b}" for b in rep["blockers"]]
    return Result(rep, "\n".join(lines), _next_for(ws, rep))


def _fmt_item(it: dict, max_chars: int, full: bool) -> str:
    rec = explore.truncate(it["record"], 0 if full else max_chars, f"l2h show {it['id']}")
    head = f"## {it['id']}" + (
        f"  [{it['file_type']}, shape: {it['shape'][:80]}]"
        if "file_type" in it
        else f"  [stratum: {it['stratum']} ({it['stratum_size']})]"
        if "stratum" in it
        else f"  [check: {it['check_result']}]"
    )
    return head + "\n" + json.dumps(rec, indent=1, default=str)


def cmd_sample(a, ws: Workspace) -> Result:
    ws.config()
    if a.matching and a.stratify:
        raise UsageError("--matching and --stratify are exclusive", "use one of them")
    if a.matching:
        data = explore.sample_matching(ws, a.matching, a.n, a.unseen, a.dry_run)
    elif a.stratify:
        data = explore.sample_stratified(ws, a.stratify, a.n, a.unseen, a.dry_run)
    else:
        data = explore.sample_native(ws, a.n, a.unseen, a.dry_run)
    cfg = ws.config()
    if not a.full:
        pass
    text = "\n\n".join(_fmt_item(i, cfg["max_field_chars"], a.full) for i in data["items"])
    meta = (
        f"{len(data['items'])} of {data['total_records']} records, {data['strata']} strata, {data['remaining_unseen']} unseen left"
        if data["mode"] in ("native", "stratified")
        else f"check {a.matching}: population {data['population']}"
    )
    return Result(
        data,
        meta + "\n\n" + text,
        ["l2h show <id>  # whole record", "l2h sample --unseen  # more, skipping these"],
    )


def cmd_show(a, ws: Workspace) -> Result:
    ws.config()
    d = explore.show_record(ws, a.id, a.field, a.attachments, a.full, a.dry_run)
    text = json.dumps(d["record"], indent=1, default=str)
    if "attachments" in d:
        text += "\nattachments:\n" + "\n".join(
            f"  {x['name']}: {x['path']} ({x['source']})" for x in d["attachments"]
        )
    return Result(d, text, ["l2h query '<expr>' --count  # measure how common this is"])


def cmd_query(a, ws: Workspace) -> Result:
    ws.config()
    d = explore.query_corpus(ws, a.expr, a.count, a.group_by, a.grep, a.raw, a.limit)
    lines = [f"scanned {d['scanned']} {d['source']}, matched {d['matched']}"]
    if "groups" in d:
        lines += [f"  {v:>7}  {k}" for k, v in d["groups"].items()]
    if "ids" in d:
        lines += [f"  {i}" for i in d["ids"]]
        if d["cut"]:
            lines.append(
                f"  …[CUT: showed {d['shown']} of {d['matched']}; use --limit {d['matched']} for all]"
            )
    if d.get("expression_errors"):
        lines.append(f"expression errors: {d['expression_errors']}")
    return Result(d, "\n".join(lines), ["l2h show <id>", "l2h sample --unseen"])


def cmd_new(a, ws: Workspace) -> Result:
    ws.config()
    path, content = scaffold(ws, a.kind, a.name or "")
    exists = path.exists()
    if a.kind != "check" and exists and not a.force:
        raise L2HError(f"{path.name} already exists", "edit it, or pass --force to overwrite")
    if not a.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, content)
    nxt = {
        "parser": "fill in parse.py, then `l2h run parse --limit 20`",
        "check": "implement the check, then `l2h run checks`",
        "tasktype": "implement it, then `l2h run generate --limit 5`",
    }[a.kind]
    return Result(
        {"path": str(path), "written": not a.dry_run},
        f"{'would write' if a.dry_run else 'wrote'} {path}",
        [nxt],
    )


def _group_lines(groups: list[dict]) -> list[str]:
    return [
        f"  x{g['count']}  {g['cause']}  (e.g. {', '.join(map(str, g['examples']))})"
        for g in groups
    ]


def cmd_run(a, ws: Workspace) -> Result:
    ws.config()
    if a.stage == "parse":
        s = stages.run_parse(ws, a.limit, a.dry_run, a.no_cache)
        lines = [
            f"{s['trajectories']} trajectories from {s['raw_records']} raw records in {s['files']} files"
            + (" (cached)" if s.get("cached") else ""),
            f"parse errors: {s['parse_errors']}; raw records in no trajectory: {s['records_in_no_trajectory']}",
            "ended: " + ", ".join(f"{k}={v}" for k, v in sorted(s["ended"].items())),
        ]
        lines += _group_lines(s["error_groups"])
        lines.append(f"changed since last run: {s['changed_since_last_run']}")
        nxt = [
            "l2h new check <name>  # then `l2h run checks`",
            "l2h show <trajectory id>  # eyeball the parse",
        ]
        if s["records_in_no_trajectory"]:
            nxt.insert(0, f"investigate dropped records, e.g. `l2h show {s['orphan_examples'][0]}`")
    elif a.stage == "checks":
        s = stages.run_checks(ws, a.limit, a.dry_run)
        lines = [
            f"{s['findings']} findings ({s['checks']} checks x {s['trajectories']} trajectories)"
        ]
        for cid, v in s["per_check"].items():
            lines.append(
                f"  {cid}@{v['version']}: pass={v['pass']} fail={v['fail']} unknown={v['unknown']} ({v['fail_rate']:.1%} fail)"
            )
        lines += _group_lines(s["error_groups"])
        lines.append(f"changed since last run: {s['changed_since_last_run']}")
        nxt = [
            "l2h sample --matching <check>  # spot-check on fresh matches/non-matches",
            "l2h new tasktype <name>",
        ]
    else:
        s = stages.run_generate(ws, a.limit, a.dry_run, a.type)
        lines = [
            f"{s['tasks']} tasks {s['by_status']} by type {s['by_type']}; excluded {s['excluded']}; queued {s['queued']}",
        ]
        lines += _group_lines(s["queue_groups"])
        for f in s["structure_or_leak_failures"][:5]:
            lines.append(f"  FAIL {f['task_id']}: {f['problems'][:2]}")
        lines.append(f"changed since last run: {s['changed_since_last_run']}")
        nxt = ["l2h task show <id>  # see exactly what the solver sees", "l2h validate"]
        if s["queued"]:
            nxt.insert(
                0, "investigate out/queue.jsonl entries (do not invent data to unblock them)"
            )
    if a.dry_run:
        lines.append("(dry-run: nothing written)")
    return Result(s, "\n".join(lines), nxt)


def _task_tree(d: Path, max_chars: int) -> list[dict]:
    out = []
    for p in sorted((d / "environment").rglob("*")):
        if p.is_file():
            raw = p.read_bytes()
            try:
                txt = raw.decode()
                prev = (
                    txt
                    if max_chars <= 0 or len(txt) <= max_chars
                    else txt[:max_chars]
                    + f"…[CUT: showed {max_chars} of {len(txt)} chars; full: {p}]"
                )
            except UnicodeDecodeError:
                prev = f"<binary {len(raw)} bytes>"
            out.append({"path": p.relative_to(d).as_posix(), "size": len(raw), "content": prev})
    return out


def cmd_task_show(a, ws: Workspace) -> Result:
    ws.config()
    d = ws.task_dir(a.id)
    cfg = ws.config()
    mc = 0 if a.full else cfg["max_field_chars"]
    meta = read_json(ws.internal / "taskmeta" / f"{a.id}.json", {})
    res = scan_task(d, meta, cfg)
    leaks = res.leaks + res.secrets
    instr = (d / "instruction.md").read_text()
    files = _task_tree(d, mc)
    data = {
        "task_id": a.id,
        "path": str(d),
        "solver_sees": {"instruction.md": instr, "environment": files},
        "hidden_from_solver": ["tests/", "solution/", "task.toml"],
        "answer_location": f"{ANSWER_DIR}/ (declared as an artifact)",
        "leaks": [x.to_dict() for x in leaks],
    }
    text = (
        f"# {a.id}  ({d})\n\n--- instruction.md ---\n{instr}\n--- environment/ ---\n"
        + "\n".join(f"[{f['path']}] ({f['size']} bytes)\n{f['content']}" for f in files)
    )
    text += "\n\nNOT visible to the solver: tests/, solution/, task.toml\n"
    text += (
        "LEAKS FOUND:\n" + "\n".join(f"  {x.where}: {x.kind}: {x.detail}" for x in leaks)
        if leaks
        else "leak scan: clean"
    )
    return Result(
        data,
        text,
        [f"l2h task try {a.id} --answer <x>", "l2h validate --task " + a.id],
        exit=EXIT_OK,
    )


def cmd_task_try(a, ws: Workspace) -> Result:
    ws.config()
    d = ws.task_dir(a.id)
    spec = read_json(d / "tests" / "expected.json", {})
    fname = a.answer_file or spec.get("file", "answer.txt")
    runner = LocalRunner(timeout=ws.config()["gate_timeout_sec"])
    out = runner.run_verifier(d, {fname: a.answer} if a.answer is not None else {})
    data = {"task_id": a.id, "answer": a.answer, "runner": "local", **out.to_dict()}
    if out.state != "ok":
        raise L2HError(
            f"verifier infrastructure error: {out.detail}",
            f"run `bash {d}/tests/test.sh` with L2H_ANSWER_DIR/L2H_REWARD_DIR set to debug",
        )
    text = f"reward={out.reward} ({'PASS' if out.reward and out.reward >= 1 else 'FAIL'}) for answer {a.answer!r}"
    return Result(
        data, text, [f"l2h task try {a.id} --answer <another>", f"l2h validate --task {a.id}"]
    )


def _warning_lines(warnings: dict[str, list[str]]) -> list[str]:
    """Non-blocking warnings grouped by kind (the text before the first colon)."""
    by: dict[str, list[str]] = {}
    for tid, ws_ in sorted(warnings.items()):
        for w in ws_:
            by.setdefault(w.split(":", 1)[0], []).append(f"{tid}: {w.split(':', 1)[-1].strip()}")
    out = ["warnings (do not block; read them):"]
    for kind, items in sorted(by.items()):
        out.append(f"  {kind} x{len(items)}")
        out += [f"    {x}" for x in items[:3]]
    return out


def cmd_validate(a, ws: Workspace) -> Result:
    ws.config()
    try:
        levels = {int(x) for x in a.level.split(",")}
        assert levels <= {1, 2, 3}
    except (ValueError, AssertionError):
        raise UsageError(
            f"--level {a.level!r} must be a comma list from 1,2,3", "e.g. --level 1,2"
        ) from None
    s = validate_mod.run_validate(
        ws,
        a.type,
        a.sample_size,
        a.max_fix_attempts,
        levels,
        a.task or [],
        a.dry_run,
        runner=pick_runner(a.runner, ws.config()),
    )
    lines = [f"runner: {s['runner']}  levels: {s['levels']}"]
    if s["runner"] == "local":
        lines.append(
            "note: host-local runner (no Harbor+Docker): passes keep tasks `generated`; "
            "`validated` needs a Harbor run (install harbor, start Docker, or --runner harbor)"
        )
    if "level1" in s:
        lines.append(f"level 1: {s['level1']['checked']} checked, {s['level1']['failed']} failed")
        for f in s["level1"]["failures"][:5]:
            lines.append(f"  {f['task_id']}: {f['problems'][:2]}")
    for t, v in s["types"].items():
        lines.append(
            f"level 2 {t}: {v['state']} (sampled {len(v['sampled'])}, passed {len(v['passed'])}, failed {len(v['failed'])}, infra errors {len(v['infra_error'])}; attempts {v['attempts']}/{v['max_fix_attempts']})"
        )
        lines += _group_lines(v["causes"])
        if v.get("message"):
            lines.append(f"  {v['message']}")
    if "level3_audit_sample" in s:
        lines.append(
            "level 3 audit sample (read these; record the verdict with `l2h decide --kind audit`):"
        )
        for t, ids in s["level3_audit_sample"].items():
            lines.append(f"  {t}: {', '.join(ids)}")
    if s["warnings"]:
        lines += _warning_lines(s["warnings"])
    if s["review_for_derivability"]:
        lines.append(
            "also review: can the answer be DERIVED from the inputs? (the oracle only proves wiring) -> "
            + ", ".join(s["review_for_derivability"][:5])
        )
    if a.dry_run:
        lines.append("(dry-run: results not saved)")
    code = EXIT_ERROR if s["any_failure"] or s["any_infra_error"] else EXIT_OK
    nxt = []
    if s["any_failure"]:
        nxt.append(
            "fix the cause above in tasktypes/<type>.py, then `l2h run generate` and `l2h validate`"
        )
    if s["any_infra_error"]:
        nxt.append(
            "infra errors are not task failures: fix the environment (see `l2h task try <id>`) and re-run `l2h validate`"
        )
    nxt += [
        'l2h decide "<verdict>" --why "<evidence>" --kind derivability --task <id>',
        "l2h report",
    ]
    return Result(s, "\n".join(lines), nxt, exit=code)


def cmd_decide(a, ws: Workspace) -> Result:
    ws.config()
    entry = {
        "kind": a.kind,
        "what": a.what,
        "why": a.why,
        "tasks": sorted(a.task or []),
        "type": a.type,
        "check": a.check,
        "verdict": a.verdict,
    }
    entry = {k: v for k, v in entry.items() if v not in (None, [], "")}
    entry["id"] = sha(json.dumps(entry, sort_keys=True))[:10]
    rows = ws.decision_rows()
    dup = any(r.get("id") == entry["id"] for r in rows)
    effects = []
    if a.kind == "audit" and a.verdict == "fail":
        if not a.type:
            raise UsageError(
                "an audit fail needs --type", "e.g. --kind audit --verdict fail --type decide_x"
            )
        effects.append(f"tasks of type {a.type} marked blocked (send the batch back to step 3)")
    if not a.dry_run:
        if not dup:
            with open(ws.decisions, "a") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        if effects:
            rs = ws.result_rows()
            for r in rs:
                if r["type"] == a.type:
                    r["blocked_reason"] = f"batch audit failed: {a.what}"
                    finalize(r)
            ws.write_results(rs)
    return Result(
        {
            "decision": entry,
            "duplicate": dup,
            "effects": effects,
            "written": not a.dry_run and not dup,
        },
        ("already recorded" if dup else "recorded")
        + f" decision {entry['id']}"
        + "".join(f"\n{e}" for e in effects),
        ["l2h report"]
        if not effects
        else [f"fix tasktypes/{a.type}.py, `l2h run generate --type {a.type}`, `l2h validate`"],
    )


def cmd_report(a, ws: Workspace) -> Result:
    ws.config()
    md = report_mod.build_report(ws)
    if not a.dry_run:
        atomic_write(ws.report, md)
    return Result(
        {"path": str(ws.report), "written": not a.dry_run, "chars": len(md)},
        md if a.dry_run else f"wrote {ws.report} ({len(md)} chars)",
        [f"read {ws.report}", "l2h status"],
    )


def schema_data() -> dict:
    return {
        "version": __version__,
        "parse.py": {
            "signature": "def parse(paths: list[Path]) -> Iterator[Trajectory | ParseError]",
            "Trajectory": {
                "format": f"an ATIF trajectory (Harbor's Agent Trajectory Interchange Format, {ATIF_VERSION}; validated against Harbor's model when importable) plus extra['l2h']. Build with Trajectory.from_atif(doc, id=, source_refs=, ended=, meta=, attachments=)",
                "ATIF": {
                    "schema_version": f"default {ATIF_VERSION}",
                    "agent": "{name, version, model_name?, tool_definitions?}; both name and version are required (version 'unknown' if unrecorded)",
                    "steps": "non-empty [{step_id (auto-numbered from 1), source: system|user|agent, message: str | [parts], timestamp?, model_name?, reasoning_content?, tool_calls?: [{tool_call_id, function_name, arguments}], observation?: {results: [{source_call_id, content, subagent_trajectory_ref?}]}, extra?}]; tool_calls, model_name, reasoning_content, metrics only on agent steps",
                    "subagent_trajectories": "optional [full ATIF trajectory with trajectory_id]; referenced by observation results via subagent_trajectory_ref",
                    "extra": "free; l2h keeps its own block under extra['l2h']",
                },
                "extra.l2h": {
                    "id": "str, unique",
                    "source_refs": "[{file: path relative to logs, offset: int (record ordinal, see `l2h sample` ids) | key: str}], non-empty",
                    "ended": list(ENDED),
                    "meta": "object; unknown values stay None, unfamiliar fields kept here",
                    "attachments": "[{name, path | text | data_b64}]",
                },
                "accessors": "t['id'|'source_refs'|'ended'|'meta'|'attachments'] and t.meta etc.; t['steps']; t.calls() -> [Call(step, id, name, arguments, result, .output)]; t.final_message; t.subagents; message_text(msg). In query/sample, r is flat: r.steps, r.meta, r.ended, r.id",
            },
            "ParseError": "ParseError(file, error, offset=None)",
        },
        "checks.py": {
            "CHECKS": "[Check(id, version, fn)]",
            "fn": "def fn(t: Trajectory) -> Finding  # Finding(result: pass|fail|unknown, evidence=[refs], note='')",
            "note": "findings are observations, not exclusions",
        },
        "tasktypes/<type>.py": {
            "SPEC": f"{{unit, env, task_scope in {list(TASK_SCOPES)}, check, gold, hidden, applies_when}}; task_scope is required: full_job = the logged agent's whole job rebuilt, decision = evidence given, its decision asked",
            "VERSION": "int, bump when the generator changes (generator = <type>@<VERSION>)",
            "applies": "def applies(t, findings) -> bool | (bool, reason)",
            "group_key": "def group_key(t) -> str  # attempts sharing a key become one task",
            "build": "def build(group, out: Path) -> BuildResult | Blocked",
            "BuildResult": {
                "task_id": "str [A-Za-z0-9._-]",
                "instruction": "str; never the recorded answer",
                "evidence": "{relative path: str|bytes} -> environment/data/<path>, visible at /app/<path>; binary files are skipped by the leak and secret scans",
                "answer": f"Answer(expected, check in {list(CHECK_KINDS)}, file='answer.txt', normalize subset of {list(NORMALIZERS)}, tolerance, rel_tolerance, pattern); forbidden = negative gold: expected is the known-wrong answers, any other well-formed answer passes (needs alternatives)",
                "wrong_answers": "meaningful wrong answers the verifier must reject (else an auto one is used and flagged)",
                "alternatives": "other valid answers the verifier must accept",
                "leak_allow": "[task-relative path] skipped by the leak scan (recorded in results)",
                "secret_allow": "[task-relative path] skipped by the secret scan (recorded in results)",
                "hidden": "extra strings that must never appear in instruction.md/environment/ (original final response, ...)",
                "files": "optional overrides/extras keyed by task-relative path",
                "gold_type": list(GOLD_TYPES),
                "assumptions": "[str]",
            },
            "Blocked": "Blocked(reason, missing=[...])",
        },
        "l2h.json": DEFAULT_CONFIG,
        "results.jsonl row": {
            "task_id": "str",
            "type": "str",
            "sources": ["file#offset"],
            "generator": "type@version",
            "gold": list(GOLD_TYPES),
            "gates": {
                "structure": list(GATE_STATES),
                "leak": list(GATE_STATES),
                "secrets": list(GATE_STATES),
                "exec": list(GATE_STATES),
                "runner": ["local", "harbor"],
                "trials": "Harbor trial ids; required (with runner=harbor) for `validated`",
            },
            "status": list(STATUSES),
            "rule": "validated only if gates.exec == pass from a real Harbor run (runner=harbor with trial ids); host-local passes stay generated",
        },
        "task package": [
            "task.toml",
            "instruction.md",
            "environment/",
            "tests/test.sh",
            "solution/solve.sh",
        ],
        "exit codes": {
            "0": "ok",
            "1": "error / validation failed",
            "2": "bad args or config",
            "130": "interrupted",
        },
    }


def cmd_schema(a, ws) -> Result:
    d = schema_data()
    return Result(d, json.dumps(d, indent=2), ["l2h docs  # guides"])


def _pkg_text(name: str) -> str:
    return resources.files("l2h").joinpath(name).read_text()


def guide_topics() -> list[str]:
    return sorted(
        p.name[:-3]
        for p in resources.files("l2h").joinpath("guides").iterdir()
        if p.name.endswith(".md")
    )


def cmd_docs(a, ws) -> Result:
    topics = guide_topics()
    if not a.topic:
        return Result(
            {"skill": _pkg_text("SKILL.md"), "topics": topics},
            _pkg_text("SKILL.md") + "\nguides: " + ", ".join(topics),
            ["l2h docs --topic <name>"],
        )
    if a.topic not in topics:
        raise UsageError(f"no guide {a.topic!r}", f"topics: {', '.join(topics)}")
    body = _pkg_text(f"guides/{a.topic}.md")
    return Result({"topic": a.topic, "text": body}, body, ["l2h docs  # other topics"])


# -------------------------------------------------------------------- parser


class Parser(argparse.ArgumentParser):
    def error(self, message: str):
        raise UsageError(message, f"see `{self.prog} --help`")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("global options")
    g.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="one JSON result on stdout (progress goes to stderr)",
    )
    g.add_argument(
        "--full",
        action="store_true",
        default=argparse.SUPPRESS,
        help="show everything, no truncation",
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="write nothing; show what would happen",
    )
    g.add_argument(
        "-w",
        "--workspace",
        default=argparse.SUPPRESS,
        help="workspace folder (default: nearest l2h.json upwards, or $L2H_WORKSPACE)",
    )

    p = Parser(prog="l2h", description="Turn agent logs into Harbor tasks.", parents=[common])
    p.add_argument("--version", action="version", version=f"l2h {__version__}")
    sub = p.add_subparsers(dest="command", required=True, parser_class=Parser)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_, parents=[common])
        sp.set_defaults(fn=fn)
        return sp

    sp = add("init", cmd_init, "create a workspace for a logs folder")
    sp.add_argument("logs")
    sp.add_argument("workspace_dir", nargs="?", help="workspace folder (default: ./l2h-workspace)")
    sp.add_argument(
        "--allow-missing-runtime",
        action="store_true",
        help="proceed without Docker/Harbor (local verifier checks only)",
    )

    add("status", cmd_status, "where the run is and what is next")

    sp = add("sample", cmd_sample, "pick varied native records")
    sp.add_argument("-n", "--n", type=int, default=5, dest="n", help="how many records")
    sp.add_argument("--unseen", action="store_true", help="skip records already shown")
    sp.add_argument("--matching", metavar="CHECK", help="fresh matches and non-matches of a check")
    sp.add_argument(
        "--stratify",
        metavar="EXPR",
        help="group key as a python expression over `r` (e.g. 'r.meta.tool'); picks evenly across groups",
    )

    sp = add("show", cmd_show, "show one whole record or trajectory")
    sp.add_argument("id")
    sp.add_argument("--field", help="dotted path, e.g. events.3.content")
    sp.add_argument("--attachments", action="store_true", help="materialise attachments as files")

    sp = add("query", cmd_query, "count/group/grep across the whole corpus")
    sp.add_argument("expr", help="python expression over `r`, e.g. 'len(r.events) > 10'")
    sp.add_argument("--count", action="store_true")
    sp.add_argument("--group-by", metavar="PATH_OR_EXPR")
    sp.add_argument("--grep", metavar="REGEX")
    sp.add_argument(
        "--raw", action="store_true", help="query raw records even when trajectories exist"
    )
    sp.add_argument("--limit", type=int, default=20)

    sp = add("new", cmd_new, "scaffold parser/check/tasktype with the contract filled in")
    sp.add_argument("kind", choices=["parser", "check", "tasktype"])
    sp.add_argument("name", nargs="?")
    sp.add_argument("--force", action="store_true")

    sp = add("run", cmd_run, "run agent-written code over the corpus")
    sp.add_argument("stage", choices=["parse", "checks", "generate"])
    sp.add_argument("--limit", type=int)
    sp.add_argument("--type", help="generate: only this task type")
    sp.add_argument("--no-cache", action="store_true", help="parse: ignore the cache")

    tp = sub.add_parser("task", help="inspect/try one task", parents=[common])
    tsub = tp.add_subparsers(dest="task_command", required=True, parser_class=Parser)
    t1 = tsub.add_parser("show", help="what the solver will see", parents=[common])
    t1.add_argument("id")
    t1.set_defaults(fn=cmd_task_show)
    t2 = tsub.add_parser("try", help="run the verifier locally (no Docker)", parents=[common])
    t2.add_argument("id")
    t2.add_argument("--answer", help="answer text (omit for a no-op run)")
    t2.add_argument("--answer-file")
    t2.set_defaults(fn=cmd_task_try)

    sp = add("validate", cmd_validate, "run the validation levels")
    sp.add_argument("--type")
    sp.add_argument("--task", action="append", help="validate specific task id(s) (repeatable)")
    sp.add_argument("--sample-size", type=int)
    sp.add_argument("--max-fix-attempts", type=int)
    sp.add_argument("--level", default="1,2", help="comma list of 1,2,3 (default 1,2)")
    sp.add_argument(
        "--runner",
        default="auto",
        choices=["auto", "harbor", "local"],
        help="auto = Harbor if harbor+docker are installed, else host-local (tasks stay `generated`)",
    )

    sp = add("decide", cmd_decide, "log a decision with its evidence")
    sp.add_argument("what")
    sp.add_argument("--why", required=True)
    sp.add_argument(
        "--kind",
        default="note",
        choices=["note", "spot_check", "derivability", "audit", "exclusion"],
    )
    sp.add_argument("--task", action="append")
    sp.add_argument("--type")
    sp.add_argument("--check")
    sp.add_argument("--verdict", choices=["pass", "fail"])

    add("report", cmd_report, "write report.md")
    add("schema", cmd_schema, "print the contracts and config schemas")
    sp = add("docs", cmd_docs, "print the bundled guides")
    sp.add_argument("--topic")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    command = next((x for x in argv if not x.startswith("-")), "l2h")
    try:
        args = build_parser().parse_args(argv)
        as_json = getattr(args, "json", False)
        args.full = getattr(args, "full", False)
        args.dry_run = getattr(args, "dry_run", False)
        command = args.command + (f" {args.task_command}" if args.command == "task" else "")
        if args.command == "init":
            ws = Workspace(
                Path(
                    args.workspace_dir or getattr(args, "workspace", None) or "l2h-workspace"
                ).resolve()
            )
        else:
            ws = find_workspace(getattr(args, "workspace", None))
        res = args.fn(args, ws)
        print(render(command, res, as_json))
        return res.exit
    except L2HError as e:
        if as_json:
            print(render_error(command, e, True))
            print(f"error: {e.message}\nfix: {e.fix}", file=sys.stderr)
        else:
            print(render_error(command, e, False), file=sys.stderr)
        return e.code
    except ContractError as e:
        err = L2HError(str(e), "fix the agent-written code (see `l2h schema`)")
        print(
            render_error(command, err, as_json) if as_json else render_error(command, err, False),
            file=sys.stderr if not as_json else sys.stdout,
        )
        return EXIT_ERROR
    except KeyboardInterrupt:
        print(
            "interrupted; nothing is left half-written (writes are atomic). Re-run to resume: `l2h status`",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED
    except SystemExit as e:  # --help / --version
        return int(e.code or 0)


if __name__ == "__main__":
    sys.exit(main())
