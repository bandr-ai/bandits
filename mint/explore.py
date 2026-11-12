"""sample / show / query: look at the corpus without writing throwaway scripts."""

from __future__ import annotations

import base64
import builtins
import json
import random
import re
from collections import Counter, defaultdict
from typing import Any

from .models import Trajectory, normalize_finding
from .output import MINTError, UsageError
from .scripts import load_checks
from .stages import load_trajectories
from .workspace import (
    Workspace,
    get_raw_record,
    iter_records,
    list_log_files,
    read_json,
    record_id,
    rel,
    shape_of,
    split_record_id,
    write_json,
)

# ---------------------------------------------------------------- truncation


def truncate(obj: Any, max_chars: int, cmd: str, path: str = "") -> Any:
    """Cut long strings/lists, leaving a visible marker that says how to see all."""
    if max_chars <= 0:
        return obj
    if isinstance(obj, str) and len(obj) > max_chars:
        more = f" --field {path}" if path else ""
        return (
            obj[:max_chars]
            + f"…[CUT: showed {max_chars} of {len(obj)} chars; full: {cmd}{more} --full]"
        )
    if isinstance(obj, list):
        keep = [
            truncate(x, max_chars, cmd, f"{path}.{i}".lstrip(".")) for i, x in enumerate(obj[:50])
        ]
        if len(obj) > 50:
            keep.append(
                f"…[CUT: showed 50 of {len(obj)} items; full: {cmd} --field {path or '.'} --full]"
            )
        return keep
    if isinstance(obj, dict):
        return {k: truncate(v, max_chars, cmd, f"{path}.{k}".lstrip(".")) for k, v in obj.items()}
    return obj


def dotted(obj: Any, path: str) -> Any:
    cur = obj
    for part in [p for p in path.split(".") if p]:
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise MINTError(
                f"no field {path!r} (stopped at {part!r})", "run `mint show <id>` to see the fields"
            )
    return cur


# -------------------------------------------------------------------- sample


def _seen(ws: Workspace) -> set[str]:
    return set(read_json(ws.internal / "seen.json", []))


def _mark_seen(ws: Workspace, ids: list[str]) -> None:
    write_json(ws.internal / "seen.json", sorted(_seen(ws) | set(ids)))


def _size_bucket(n: int) -> str:
    return "small" if n < 10_000 else "medium" if n < 1_000_000 else "large"


def sample_native(ws: Workspace, n: int, unseen: bool, dry: bool) -> dict:
    logs = ws.logs_root()
    cfg = ws.config()
    seen = _seen(ws) if unseen else set()
    strata: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    total = 0
    sizes = {}
    for p in list_log_files(logs):
        rf = rel(logs, p)
        size = p.stat().st_size
        sizes[rf] = size
        for i, rec in iter_records(p):
            total += 1
            rid = record_id(rf, i)
            if rid in seen:
                continue
            strata[(p.suffix.lower(), _size_bucket(size), shape_of(rec))].append((rf, i))
    rng = random.Random(f"{cfg['seed']}:sample:{len(seen)}")
    # per stratum: spread over position (first, last, middle, then seeded random)
    queues = []
    for key in sorted(strata):
        items = strata[key]
        order = []
        for idx in (0, len(items) - 1, len(items) // 2):
            if items[idx] not in order:
                order.append(items[idx])
        rest = [x for x in items if x not in order]
        rng.shuffle(rest)
        queues.append((key, order + rest))
    picked = []
    while len(picked) < n and any(q for _, q in queues):
        for key, q in queues:
            if q and len(picked) < n:
                picked.append((key, q.pop(0)))
    wanted = {rid for _, (f, i) in picked for rid in [record_id(f, i)]}
    recs = {}
    for f in {f for _, (f, _) in picked}:
        for i, rec in iter_records(logs / f if logs.is_dir() else logs):
            if record_id(f, i) in wanted:
                recs[record_id(f, i)] = rec
    items = []
    for key, (f, i) in picked:
        rid = record_id(f, i)
        items.append(
            {
                "id": rid,
                "file": f,
                "file_type": key[0] or "(none)",
                "file_size": sizes[f],
                "shape": key[2],
                "stratum_size": len(strata[key]),
                "record": recs[rid],
            }
        )
    if not dry and unseen is not None:
        _mark_seen(ws, [x["id"] for x in items])
    return {
        "mode": "native",
        "total_records": total,
        "strata": len(strata),
        "remaining_unseen": sum(len(v) for v in strata.values()) - len(items),
        "items": items,
    }


def sample_matching(ws: Workspace, check_id: str, n: int, unseen: bool, dry: bool) -> dict:
    trajs = load_trajectories(ws)
    _, checks = load_checks(ws)
    chk = next((c for c in checks if c.id == check_id), None)
    if not chk:
        raise MINTError(
            f"no check {check_id!r}", f"known: {', '.join(c.id for c in checks) or '(none)'}"
        )
    seen = _seen(ws) if unseen else set()
    groups: dict[str, list[dict]] = {"fail": [], "pass": [], "unknown": []}
    for t in trajs:
        if t["id"] in seen:
            continue
        try:
            r = normalize_finding(chk.fn(Trajectory(t))).result
        except Exception:
            r = "unknown"
        groups[r].append(t)
    rng = random.Random(f"{ws.config()['seed']}:matching:{check_id}:{len(seen)}")
    pools = {}
    for label in ("fail", "pass"):
        pools[label] = list(groups[label])
        rng.shuffle(pools[label])
    # n split across fails and passes (fails first), topped up from the other side
    take = {"fail": min(len(pools["fail"]), (n + 1) // 2)}
    take["pass"] = min(len(pools["pass"]), n - take["fail"])
    take["fail"] = min(len(pools["fail"]), n - take["pass"])
    chosen = [(label, t) for label in ("fail", "pass") for t in pools[label][: take[label]]]
    if not dry:
        _mark_seen(ws, [t["id"] for _, t in chosen])
    return {
        "mode": "matching",
        "check": check_id,
        "population": {k: len(v) for k, v in groups.items()},
        "items": [{"id": t["id"], "check_result": label, "record": t} for label, t in chosen],
    }


def sample_stratified(ws: Workspace, expr: str, n: int, unseen: bool, dry: bool) -> dict:
    """Group records by `expr` (over r) and pick evenly across the groups, seeded.
    Uses parsed trajectories when they exist, else raw records."""
    code = _compile(expr)
    if ws.trajectories.exists():
        source, records = "trajectories", [(t["id"], t) for t in load_trajectories(ws)]
    else:
        source, logs = "raw", ws.logs_root()
        records = [
            (record_id(rel(logs, p), i), rec)
            for p in list_log_files(logs)
            for i, rec in iter_records(p)
        ]
    seen = _seen(ws) if unseen else set()
    groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for rid, rec in records:
        if rid in seen:
            continue
        try:
            key = eval(code, {**_scope(), "r": _R(_view(rec)) if isinstance(rec, dict) else rec})
        except Exception as e:
            key = f"<error: {type(e).__name__}>"
        groups[key if isinstance(key, str) else json.dumps(key, default=str)].append((rid, rec))
    rng = random.Random(f"{ws.config()['seed']}:stratify:{expr}:{len(seen)}")
    queues = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
    picked = []
    while len(picked) < n and any(queues.values()):
        for k, q in queues.items():  # one per group per round: even across groups
            if q and len(picked) < n:
                picked.append((k, q.pop(0)))
    if not dry:
        _mark_seen(ws, [rid for _, (rid, _) in picked])
    return {
        "mode": "stratified",
        "source": source,
        "stratify": expr,
        "groups": {k: len(v) for k, v in groups.items()},
        "total_records": len(records),
        "strata": len(groups),
        "remaining_unseen": sum(len(q) for q in queues.values()),
        "items": [
            {"id": rid, "stratum": k, "stratum_size": len(groups[k]), "record": rec}
            for k, (rid, rec) in picked
        ],
    }


# ---------------------------------------------------------------------- show


def _safe_part(s: str) -> str:
    """One path component: sanitised, and never `.`/`..`/empty."""
    s = re.sub(r"[^\w.\-]", "_", s)
    return f"_{s}" if s in ("", ".", "..") else s


def show_record(
    ws: Workspace, rid: str, field: str | None, attachments: bool, full: bool, dry: bool
) -> dict:
    max_chars = 0 if full else ws.config()["max_field_chars"]
    traj = None
    if ws.trajectories.exists():
        traj = next((t for t in load_trajectories(ws) if t["id"] == rid), None)
    if traj is not None:
        kind, rec = "trajectory", traj
    elif split_record_id(rid):
        kind, rec = "raw", get_raw_record(ws.logs_root(), rid)
    else:
        raise MINTError(
            f"no trajectory or raw record {rid!r}",
            "ids: trajectories in out/trajectories.jsonl, raw as <file>#<n> (see `mint sample`)",
        )
    cmd = f"mint show {rid}"
    if field:
        rec = dotted(_view(rec), field)
    out: dict[str, Any] = {
        "id": rid,
        "kind": kind,
        "record": truncate(rec, max_chars, cmd, field or ""),
    }
    if attachments:
        if kind != "trajectory":
            raise MINTError(
                "--attachments applies to parsed trajectories", "parse first, or omit --attachments"
            )
        files = []
        att_root = (ws.out / "attachments").resolve()
        base = att_root / _safe_part(rid)
        for i, a in enumerate(traj["attachments"]):
            if not isinstance(a, dict):
                continue
            name = _safe_part(str(a.get("name") or f"attachment_{i}"))
            target = base / name
            if att_root not in target.resolve().parents:
                raise MINTError(f"attachment path for {rid!r}/{name!r} escapes out/attachments/")
            if "path" in a:
                files.append(
                    {"name": name, "path": str(a["path"]), "source": "original (read-only)"}
                )
            elif "data_b64" in a or "text" in a:
                if not dry:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if "data_b64" in a:
                        target.write_bytes(base64.b64decode(a["data_b64"]))
                    else:
                        target.write_text(str(a["text"]))
                files.append(
                    {
                        "name": name,
                        "path": str(target),
                        "source": "materialized copy" + (" (dry-run: not written)" if dry else ""),
                    }
                )
            else:
                files.append(
                    {
                        "name": name,
                        "path": None,
                        "source": "no path/text/data_b64 on this attachment",
                    }
                )
        out["attachments"] = files
    return out


# --------------------------------------------------------------------- query


def _view(rec: Any) -> Any:
    """Parsed trajectories are seen flat: ATIF keys plus id, meta, ended, source_refs, attachments."""
    return rec.flat() if isinstance(rec, Trajectory) else rec


class _R(dict):
    def __getattr__(self, n: str) -> Any:
        v = self.get(n)
        return _R(v) if isinstance(v, dict) else v


_SAFE = {
    k: getattr(builtins, k)
    for k in (
        "len",
        "str",
        "int",
        "float",
        "bool",
        "any",
        "all",
        "min",
        "max",
        "sum",
        "sorted",
        "set",
        "list",
        "dict",
        "isinstance",
        "abs",
        "round",
    )
}


def _scope() -> dict:
    return {"re": re, "get": _get, "__builtins__": _SAFE}


def _compile(expr: str):
    try:
        return compile(expr, "<query>", "eval")
    except SyntaxError as e:
        raise UsageError(
            f"bad expression {expr!r}: {e.msg}",
            "use Python: e.g. 'len(r.steps) > 10', r is the record",
        ) from e


def _get(r: Any, path: str) -> Any:
    try:
        return dotted(r, path)
    except MINTError:
        return None


def query_corpus(
    ws: Workspace,
    expr: str,
    count: bool,
    group_by: str | None,
    grep: str | None,
    raw: bool,
    limit: int,
) -> dict:
    if not raw and ws.trajectories.exists():
        source = "trajectories"
        records = ((t["id"], t) for t in load_trajectories(ws))
    else:
        source = "raw"
        logs = ws.logs_root()

        def gen():
            for p in list_log_files(logs):
                rf = rel(logs, p)
                for i, rec in iter_records(p):
                    yield record_id(rf, i), rec

        records = gen()
    code = _compile(expr or "True")
    gcode = None
    if group_by:
        gcode = None if re.fullmatch(r"[\w.]+", group_by) else _compile(group_by)
    try:
        rx = re.compile(grep) if grep else None
    except re.error as e:
        raise UsageError(
            f"bad --grep regex {grep!r}: {e}", "use a Python regular expression"
        ) from e
    scope = _scope()
    matched, groups, scanned, errors = [], Counter(), 0, Counter()
    for rid, rec in records:
        scanned += 1
        if rx and not rx.search(json.dumps(rec, default=str)):
            continue
        try:
            # r goes in globals: names in comprehensions don't see eval's locals
            ok = eval(code, {**scope, "r": _R(_view(rec)) if isinstance(rec, dict) else rec})
        except Exception as e:
            errors[f"{type(e).__name__}: {e}"] += 1
            continue
        if not ok:
            continue
        matched.append(rid)
        if group_by:
            try:
                key = (
                    eval(gcode, {**scope, "r": _R(_view(rec)) if isinstance(rec, dict) else rec})
                    if gcode
                    else _get(_view(rec), group_by)
                )
            except Exception as e:
                key = f"<error: {type(e).__name__}>"
            groups[json.dumps(key, default=str) if not isinstance(key, str) else key] += 1
    out: dict[str, Any] = {"source": source, "scanned": scanned, "matched": len(matched)}
    if errors:
        out["expression_errors"] = dict(errors.most_common(5))
    if group_by:
        out["groups"] = dict(groups.most_common())
    if not count and not group_by:
        out["ids"] = matched[:limit]
        out["shown"] = min(limit, len(matched))
        out["cut"] = len(matched) > limit
    return out
