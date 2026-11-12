"""Candidate environment evidence from a trajectory: what was read, written, run, fetched.

Input for rebuilding a task's starting state; not a rebuilder. Conservative by design:
- every item points back to its step and call;
- `complete` is True only when proven, False when the call itself shows a partial view
  (a line range, head/tail), and None (unknown) otherwise. A logged read is not assumed
  to be the whole file: outputs can be truncated by the tool or the logger;
- tools mint has no interpreter for are listed under `unknown`, never guessed at.

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
    optionally `complete` and `note`. Step and call refs are added by mint.
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
# programs that never create files: a command made only of these (and no redirect) can't
# have produced a file a later step reads
_NO_CREATE = (
    _PARTIAL_READERS
    | _FULL_READERS
    | {
        "ls",
        "echo",
        "printf",
        "pwd",
        "cd",
        "wc",
        "find",
        "file",
        "stat",
        "which",
        "type",
        "diff",
        "cmp",
        "test",
        "[",
        "true",
        "false",
        "env",
        "whoami",
        "date",
        "sort",
        "uniq",
        "cut",
        "tr",
        "nl",
        "md5sum",
        "sha256sum",
        "du",
        "df",
        "ps",
        "id",
        "uname",
        "tree",
    }
)
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
    cmd_item: dict = {"kind": "commands", "command": cmd, "opaque": False}
    items: list[dict] = [cmd_item]
    for url in _URL_RE.findall(cmd):
        items.append({"kind": "sources", "url": url.rstrip(".,;")})
    if "`" in cmd or "$(" in cmd:
        cmd_item["opaque"] = True
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
            files = [w for w in rest if not w.startswith("-") and not re.search(r"[<>|]", w)]
            if prog not in _NO_CREATE or (prog == "sed" and "-i" in rest):
                cmd_item["opaque"] = True  # may create or change files we can't see
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


# Starting-state facts read from tool outputs and commands (shell-agnostic patterns only).
_ABSENT_RES = (
    re.compile(r"(?:^|[\s:])([\w.+-]+): (?:command )?not found", re.M),
    re.compile(r"No module named ['\"]?([\w.]+)"),
    re.compile(r"Cannot find module ['\"]([^'\"]+)"),
    re.compile(r"(?:^|\s)(?:E: )?Unable to locate package ([\w.+-]+)", re.M),
)
_SYSTEM_RES = (
    re.compile(r'PRETTY_NAME="([^"]+)"'),
    re.compile(r"^(Python \d+\.\d+(?:\.\d+)?)", re.M),
    re.compile(r"^(node v?\d+\.\d+\.\d+|v\d+\.\d+\.\d+)$", re.M),
    re.compile(r"^(gcc \([^)]*\) \d+\.\d+(?:\.\d+)?)", re.M),
    re.compile(r"^(go version go\d+\.\d+(?:\.\d+)?)", re.M),
    re.compile(r"^((?:openjdk|java) (?:version )?\"?\d+[\w.]*)", re.M),
    re.compile(r"^(Ubuntu \d+\.\d+[\w. ]*|Debian GNU/Linux \d+[\w. ()]*)", re.M),
)
# commands by which the agent changes its own machine: part of the job, never the start state
_SETUP_RE = re.compile(
    r"\b(?:apt(?:-get)?\s+(?:-\S+\s+)*(?:install|source|build-dep)|dpkg\s+-i|"
    r"(?:pip3?|uv\s+pip|python3?\s+-m\s+pip)\s+install|uv\s+(?:add|sync)|poetry\s+(?:add|install)|"
    r"conda\s+install|(?:npm|pnpm|yarn)\s+(?:i|install|add)\b|cargo\s+install|go\s+install|"
    r"gem\s+install|brew\s+install|apk\s+add|yum\s+install|dnf\s+install|"
    r"git\s+clone|make\s+install|update-alternatives|systemctl\s+(?:enable|start))"
    r"|(?:>>?|tee(?:\s+-a)?)\s*/(?:etc|usr|opt)/",
)


def _facts(call: Call, ref: dict, ev: dict) -> None:
    out = call.output or ""
    for rx in _ABSENT_RES:
        for m in rx.finditer(out):
            ev["absent"].append({**ref, "what": m.group(1), "line": m.group(0).strip()[:160]})
    for rx in _SYSTEM_RES:
        for m in rx.finditer(out):
            ev["system"].append({**ref, "fact": m.group(1).strip()[:120]})
    cmd = _shell_command(call.arguments) if _SHELL_NAMES.search(call.name or "") else None
    if cmd:
        for seg in re.split(r"&&|\|\||;|\n", cmd):
            if _SETUP_RE.search(seg):
                ev["agent_setup"].append({**ref, "command": seg.strip()[:200]})


def _interpret(call: Call) -> tuple[str | None, list[dict] | None]:
    for name, fn in _REGISTRY:
        try:
            got = fn(call)
        except Exception as e:  # an interpreter bug must not hide the call
            return name, [{"kind": "unknown", "note": f"interpreter {name} failed: {e}"}]
        if got:
            return name, got
    return None, None


def extract(t: Trajectory | dict, *, _prefix: str = "", _seq: list | None = None) -> dict:
    """Evidence lists for one trajectory (and its subagents, refs prefixed `sub<i>/`).

    Every item carries `seq`, the call's position in the run (subagents after the parent),
    so reads and writes can be ordered. Commands that may create or change files mint can't
    see are `opaque`; calls to tools without an interpreter are listed under `unknown`.
    """
    t = t if isinstance(t, Trajectory) else Trajectory(t)
    seq = _seq if _seq is not None else [0]
    ev: dict = {k: [] for k in KINDS} | {
        "unknown": [],
        "absent": [],
        "system": [],
        "agent_setup": [],
    }
    for c in t.calls():
        seq[0] += 1
        ref = {
            "step": f"{_prefix}{c.step.get('step_id')}",
            "call": c.id,
            "tool": c.name,
            "seq": seq[0],
        }
        _facts(c, ref, ev)
        by, items = _interpret(c)
        if not items:
            ev["unknown"].append({**ref, "args": sorted(c.arguments)})
            continue
        for it in items:
            kind = it.pop("kind")
            ev.setdefault(kind, []).append({**ref, "by": by, **it})
    for i, sub in enumerate(t.subagents):
        sev = extract(sub, _prefix=f"{_prefix}sub{i}/", _seq=seq)
        for k, v in sev.items():
            ev.setdefault(k, []).extend(v)
    return ev


def starting_files(ev: dict) -> list[dict]:
    """Paths read before the run was seen writing them: candidate starting state.

    Within one call, writes count first (`gen > out; cat out` reads its own output).
    `pre_existing`: True when nothing before the read could have created the file (no earlier
    opaque command or uninterpreted tool, and the read isn't inside an opaque command);
    None (unknown) otherwise: e.g. `tar xf a.tgz` then `cat src/x.c`. Paths the run wrote
    first are outputs and are left out. `complete` is the best evidence seen.
    """
    opaque_seqs = sorted(
        {i["seq"] for i in ev.get("commands", []) if i.get("opaque")}
        | {i["seq"] for i in ev.get("unknown", [])}
    )
    events = [(i["seq"], 0, "writes", i) for i in ev.get("writes", []) if i.get("path")]
    events += [(i["seq"], 1, "reads", i) for i in ev.get("reads", []) if i.get("path")]
    written: set = set()
    seen: dict[str, dict] = {}
    for sq, _, kind, item in sorted(events, key=lambda e: (e[0], e[1])):
        p = item["path"]
        if kind == "writes":
            written.add(p)
            continue
        if p in written:
            continue
        best = seen.get(p)
        if best is None:
            opaque_before = any(o < sq for o in opaque_seqs) or sq in opaque_seqs
            seen[p] = {
                "path": p,
                "complete": item.get("complete"),
                "pre_existing": None if opaque_before else True,
                "refs": [item["step"]],
            }
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


def summary(ev: dict) -> dict:
    """Small overview: counts, candidate starting files, outputs, sources, unknown tools, and the
    starting-state facts: what the agent lacked, what the system reported, what the agent set up
    itself (that setup is the job: it belongs in the solution, never the starting environment)."""
    first = {}
    for i in ev.get("system", []):
        first.setdefault(i["fact"], i["step"])
    absent = {}
    for i in ev.get("absent", []):
        absent.setdefault(i["what"], i["step"])
    return {
        "counts": {k: len(ev.get(k, [])) for k in (*KINDS, "unknown")},
        "absent_at_first_sight": [{"what": k, "step": v} for k, v in absent.items()],
        "system": [{"fact": k, "step": v} for k, v in first.items()],
        "agent_setup": [
            {"command": i["command"], "step": i["step"]} for i in ev.get("agent_setup", [])
        ],
        "starting_files": starting_files(ev),
        "outputs": outputs(ev),
        "sources": sorted({i["url"] for i in ev.get("sources", []) if i.get("url")}),
        "unknown_tools": sorted({i.get("tool") or "?" for i in ev.get("unknown", [])}),
    }
