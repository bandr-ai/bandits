"""Candidate environment evidence from a trajectory: what was read, written, run, fetched.

Input for rebuilding a task's starting state; not a rebuilder. Conservative by design:
- every item points back to its step and call;
- `complete` is True only when proven, False when the call itself shows a partial view
  (a line range, head/tail), and None (unknown) otherwise. A logged read is not assumed
  to be the whole file: outputs can be truncated by the tool or the logger;
- tools l2h has no interpreter for are listed under `unknown`, never guessed at.

Interpreters are a small registry (`register`): shell commands and common file/fetch tools
are built in; a workspace can add its own for other tools.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable

from .models import Call, Trajectory

KINDS = ("reads", "writes", "commands", "sources")

Interpreter = Callable[[Call], "list[dict] | None"]
_REGISTRY: list[tuple[str, Interpreter]] = []

_PATH_KEYS = ("file_path", "path", "filename", "file", "target_file", "notebook_path")
_URL_RE = re.compile(r"\b(?:https?|ftp|git)://[^\s'\"<>)]+|\bgit@[\w.-]+:[\w./-]+")
_TRUNC_RE = re.compile(r"truncated|\[\.\.\.\]|<\s*\d+\s+(?:more\s+)?(?:lines|chars|bytes)", re.I)
_SHELL_NAMES = re.compile(r"bash|shell|terminal|exec|run_?command|command|cmd|sh$", re.I)
_READ_NAMES = re.compile(r"read|view|open|cat|get_?file", re.I)
_WRITE_NAMES = re.compile(r"write|create|save|put_?file", re.I)
_EDIT_NAMES = re.compile(r"edit|replace|patch|insert|modify", re.I)
_FETCH_NAMES = re.compile(r"fetch|download|http|browse|navigate|open_?url|get_?url", re.I)
_RANGE_KEYS = ("offset", "limit", "view_range", "start_line", "end_line", "line_range", "lines")


def register(name: str, interpret: Interpreter, *, first: bool = True) -> None:
    """Add an interpreter. It returns items for calls it understands, or None to pass.

    Items are dicts with `kind` (one of KINDS) plus `path`, `command` or `url`, and
    optionally `complete` and `note`. Step and call refs are added by l2h.
    """
    entry = (name, interpret)
    _REGISTRY.insert(0, entry) if first else _REGISTRY.append(entry)


def _path_arg(args: dict) -> str | None:
    for k in _PATH_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _read_complete(call: Call) -> bool | None:
    if any(args_has(call.arguments, k) for k in _RANGE_KEYS):
        return False
    out = call.output
    if out is not None and _TRUNC_RE.search(out):
        return False
    return None  # nothing proves the logged output is the whole file


def args_has(args: dict, key: str) -> bool:
    return args.get(key) not in (None, "", [], {})


def _file_tool(call: Call) -> list[dict] | None:
    path = _path_arg(call.arguments)
    if path is None:
        return None
    name = call.name or ""
    if _EDIT_NAMES.search(name):
        return [{"kind": "writes", "path": path, "complete": False, "note": "edit"}]
    if _WRITE_NAMES.search(name):
        return [{"kind": "writes", "path": path, "complete": None}]
    if _READ_NAMES.search(name):
        return [{"kind": "reads", "path": path, "complete": _read_complete(call)}]
    return None


def _fetch_tool(call: Call) -> list[dict] | None:
    if not _FETCH_NAMES.search(call.name or ""):
        return None
    url = call.arguments.get("url") or call.arguments.get("uri")
    return [{"kind": "sources", "url": url}] if isinstance(url, str) and url else None


# Shell: only shapes whose meaning is unambiguous. Everything else stays a plain command.
_PARTIAL_READERS = {"head", "tail", "sed", "awk", "grep", "less", "more"}
_FULL_READERS = {"cat"}
_REDIRECT_RE = re.compile(r"(?<![<>&\d])(>>?)\s*([^\s;&|<>]+)")


def _shell_command(args: dict) -> str | None:
    for k in ("command", "cmd", "script"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return shlex.join(v)
    return None


def _shell(call: Call) -> list[dict] | None:
    if not _SHELL_NAMES.search(call.name or ""):
        return None
    cmd = _shell_command(call.arguments)
    if cmd is None:
        return None
    items: list[dict] = [{"kind": "commands", "command": cmd}]
    for url in _URL_RE.findall(cmd):
        items.append({"kind": "sources", "url": url.rstrip(".,;")})
    for op, target in _REDIRECT_RE.findall(cmd):
        if target.startswith(("/dev/", "&")):
            continue
        items.append(
            {
                "kind": "writes",
                "path": target,
                "complete": None,
                "note": "append" if op == ">>" else "redirect",
            }
        )
    for seg in re.split(r"&&|\|\||;|\n", cmd):
        for n, part in enumerate(seg.split("|")):
            try:
                words = shlex.split(part.strip())
            except ValueError:
                continue
            if not words:
                continue
            prog, rest = words[0].rsplit("/", 1)[-1], words[1:]
            files = [w for w in rest if not w.startswith("-") and w not in (">", ">>", "<")]
            if prog == "tee":
                for f in files:
                    items.append({"kind": "writes", "path": f, "complete": None, "note": "tee"})
            elif n > 0 or not files:
                continue  # later pipe stages read stdin, not files
            elif prog in _FULL_READERS:
                out = call.output
                trunc = out is not None and bool(_TRUNC_RE.search(out))
                for f in files:
                    items.append({"kind": "reads", "path": f, "complete": False if trunc else None})
            elif prog in _PARTIAL_READERS:
                # the last non-option word is usually the file; partial, never complete
                items.append({"kind": "reads", "path": files[-1], "complete": False, "note": prog})
    return items


register("shell", _shell, first=False)
register("file", _file_tool, first=False)
register("fetch", _fetch_tool, first=False)


def _interpret(call: Call) -> tuple[str | None, list[dict] | None]:
    for name, fn in _REGISTRY:
        try:
            got = fn(call)
        except Exception as e:  # an interpreter bug must not hide the call
            return name, [{"kind": "unknown", "note": f"interpreter {name} failed: {e}"}]
        if got:
            return name, got
    return None, None


def extract(t: Trajectory | dict, *, _prefix: str = "") -> dict:
    """Evidence lists for one trajectory (and its subagents, refs prefixed `sub<i>/`)."""
    t = t if isinstance(t, Trajectory) else Trajectory(t)
    ev: dict = {k: [] for k in KINDS} | {"unknown": []}
    for c in t.calls():
        ref = {"step": f"{_prefix}{c.step.get('step_id')}", "call": c.id, "tool": c.name}
        by, items = _interpret(c)
        if not items:
            ev["unknown"].append({**ref, "args": sorted(c.arguments)})
            continue
        for it in items:
            kind = it.pop("kind")
            ev.setdefault(kind, []).append({**ref, "by": by, **it})
    for i, sub in enumerate(t.subagents):
        sev = extract(sub, _prefix=f"{_prefix}sub{i}/")
        for k, v in sev.items():
            ev.setdefault(k, []).extend(v)
    return ev


def starting_files(ev: dict) -> list[dict]:
    """Paths read before anything in the run wrote them: candidate starting state.

    Paths the run wrote first are outputs (they belong in the hidden reference, not the
    starting environment). Order is by first read; `complete` is the best evidence seen.
    """
    order = _ordered(ev)
    written: set = set()
    seen: dict[str, dict] = {}
    for kind, item in order:
        p = item.get("path")
        if not p:
            continue
        if kind == "writes":
            written.add(p)
        elif kind == "reads" and p not in written:
            best = seen.get(p)
            if best is None:
                seen[p] = {"path": p, "complete": item.get("complete"), "refs": [item["step"]]}
            else:
                best["refs"].append(item["step"])
                if item.get("complete") is True:
                    best["complete"] = True
    return list(seen.values())


def outputs(ev: dict) -> list[str]:
    """Paths the run wrote (in first-write order)."""
    out: list[str] = []
    for item in ev.get("writes", []):
        if item.get("path") and item["path"] not in out:
            out.append(item["path"])
    return out


def _step_key(step: str) -> tuple:
    """Natural sort key for step refs like '12' or 'sub0/3' (numbers compare as numbers)."""
    return tuple(
        (0, int(x), "") if x.isdigit() else (1, 0, x) for x in re.findall(r"\d+|\D+", step)
    )


def _ordered(ev: dict) -> list[tuple[str, dict]]:
    """reads/writes in step order (a read and a write in one call: read first)."""
    pairs = [(k, i) for k in ("reads", "writes") for i in ev.get(k, [])]
    return sorted(pairs, key=lambda p: (_step_key(str(p[1].get("step"))), p[0] != "reads"))


def summary(ev: dict) -> dict:
    """Small overview: counts, candidate starting files, outputs, sources, unknown tools."""
    return {
        "counts": {k: len(ev.get(k, [])) for k in (*KINDS, "unknown")},
        "starting_files": starting_files(ev),
        "outputs": outputs(ev),
        "sources": sorted({i["url"] for i in ev.get("sources", []) if i.get("url")}),
        "unknown_tools": sorted({i.get("tool") or "?" for i in ev.get("unknown", [])}),
    }
