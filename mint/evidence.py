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
# Each fact keeps its context: the command (segment) that produced it, step, tool, and
# `before_agent_setup`: True when no agent_setup action came earlier (earlier call, or earlier
# in the same command). Facts seen before any setup are starting-state evidence; later ones
# only describe that moment (the agent had changed the machine by then).
_NOT_FOUND_RE = re.compile(r"(?:^|[\s:])([\w.+-]+): (?:command )?not found", re.M)
_MODULE_RES = (
    re.compile(r"No module named ['\"]?([\w.]+)"),
    re.compile(r"Cannot find module ['\"]([^'\"]+)"),
)
# not absence: the package index lookup failed (no index, wrong name, stale lists)
_PKG_LOOKUP_RE = re.compile(r"(?:^|\s)(?:E: )?Unable to locate package ([\w.+-]+)", re.M)
_INTERPRETER_RE = re.compile(r"\b(?:python[\d.]*|nodejs?|ruby|perl|php|Rscript|julia|pytest)\b")

# (output pattern, pattern the producing command must match). A system fact needs both: output
# of other commands (e.g. `cat notes.txt` containing "Python 3.12") says nothing about the system.
_OS_Q = r"/etc/(?:os-release|issue|debian_version|lsb-release|redhat-release)|\blsb_release\b|\buname\b|\bhostnamectl\b"
_PY_Q = r"\bpython[\d.]*\b.*(?:--version|\s-V\b|sys\.version|platform\.)"
_NODE_Q = r"\bnodejs?\b.*(?:\s-v\b|--version|process\.version)"
_CC_Q = r"\b(?:gcc|g\+\+|cc|c\+\+|clang)\b.*(?:--version|\s-v\b|-dumpversion)"
_GO_Q = r"\bgo\s+version\b"
_JAVA_Q = r"\bjava\b.*(?:-version|--version)"
_SYSTEM_RES = tuple(
    (re.compile(o, re.M), re.compile(q))
    for o, q in (
        (r'PRETTY_NAME="([^"]+)"', _OS_Q),
        (r"^(Ubuntu \d+\.\d+[\w. ]*|Debian GNU/Linux \d+[\w. ()]*)", _OS_Q),
        (r"^(Python \d+\.\d+(?:\.\d+)?)", _PY_Q),
        (r"^(\d+\.\d+\.\d+) \(\w+, ", _PY_Q),  # print(sys.version)
        (r"^(node v?\d+\.\d+\.\d+|v\d+\.\d+\.\d+)$", _NODE_Q),
        (r"^(gcc \([^)]*\) \d+\.\d+(?:\.\d+)?)", _CC_Q),
        (r"^(go version go\d+\.\d+(?:\.\d+)?)", _GO_Q),
        (r"^((?:openjdk|java) (?:version )?\"?\d+[\w.]*)", _JAVA_Q),
    )
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


def _segments(cmd: str) -> list[str]:
    """Split a shell command on `&&`, `||`, `;`, `|` and newlines outside quotes."""
    segs: list[str] = []
    cur: list[str] = []
    quote = ""
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(cmd):
                cur.append(cmd[i : i + 2])
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "\\" and i + 1 < len(cmd):
            cur.append(cmd[i : i + 2])
            i += 2
            continue
        elif ch in ";\n|" or cmd.startswith("&&", i):
            segs.append("".join(cur))
            cur = []
            i += 2 if cmd.startswith(("&&", "||"), i) else 1
            continue
        cur.append(ch)
        i += 1
    segs.append("".join(cur))
    return [x.strip() for x in segs if x.strip()]


class _Ctx:
    """What one call's command looks like, for ordering facts against agent setup."""

    def __init__(self, call: Call, state: dict):
        self.state = state
        self.cmd = _shell_command(call.arguments) if _SHELL_NAMES.search(call.name or "") else None
        self.segs = _segments(self.cmd) if self.cmd else []
        self.setup = [i for i, g in enumerate(self.segs) if _SETUP_RE.search(g)]

    def locate(self, needle: str | None, *tiers) -> int | None:
        """Index of the first segment containing `needle` (or None); tiers are extra filters
        tried first, each a predicate over a segment."""
        if not needle and not tiers:
            return None
        for ok in tiers:
            for i, g in enumerate(self.segs):
                if i not in self.setup and ok(g):
                    return i
        for i, g in enumerate(self.segs):
            if needle and needle in g:
                return i
        return None

    def before(self, idx: int | None) -> bool:
        """True when no agent setup came earlier: in an earlier call, or an earlier segment of
        this command. An unlocated segment (None) counts as the end of the command."""
        if self.state["setup_seq"] is not None:
            return False
        if idx is None:
            idx = len(self.segs)
        return not any(s < idx for s in self.setup)

    def ctx(self, idx: int | None) -> dict:
        seg = self.segs[idx][:200] if idx is not None and idx < len(self.segs) else None
        return {
            "command": seg if seg is not None else (self.cmd or "")[:200] or None,
            "before_agent_setup": self.before(idx),
        }


def _facts(call: Call, ref: dict, ev: dict, cx: _Ctx) -> None:
    out = call.output or ""
    for m in _NOT_FOUND_RE.finditer(out):
        name = m.group(1)
        word = re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")
        idx = cx.locate(None, lambda g, w=word: bool(w.search(g)))
        ev["absent"].append(
            {**ref, "kind": "not_on_path", "what": name, "line": m.group(0).strip()[:160]}
            | cx.ctx(idx)
        )
    for rx in _MODULE_RES:
        for m in rx.finditer(out):
            name, top = m.group(1), m.group(1).split(".")[0]
            idx = cx.locate(
                top,
                lambda g, t=top: bool(_INTERPRETER_RE.search(g) and t in g),
                lambda g: bool(_INTERPRETER_RE.search(g)),
            )
            ev["absent"].append(
                {**ref, "kind": "module_missing", "what": name, "line": m.group(0).strip()[:160]}
                | cx.ctx(idx)
            )
    for m in _PKG_LOOKUP_RE.finditer(out):
        idx = cx.locate(m.group(1))
        ev["package_lookup_failed"].append(
            {**ref, "what": m.group(1), "line": m.group(0).strip()[:160]} | cx.ctx(idx)
        )
    if cx.cmd:  # system facts only from shell commands that are version/OS queries
        for out_rx, q_rx in _SYSTEM_RES:
            idx = next((i for i, g in enumerate(cx.segs) if q_rx.search(g)), None)
            if idx is None:
                continue
            for m in out_rx.finditer(out):
                ev["system"].append({**ref, "fact": m.group(1).strip()[:120]} | cx.ctx(idx))
        for i in cx.setup:
            ev["agent_setup"].append(
                {**ref, "command": cx.segs[i][:200], "before_agent_setup": cx.before(i)}
            )


def _interpret(call: Call) -> tuple[str | None, list[dict] | None]:
    for name, fn in _REGISTRY:
        try:
            got = fn(call)
        except Exception as e:  # an interpreter bug must not hide the call
            return name, [{"kind": "unknown", "note": f"interpreter {name} failed: {e}"}]
        if got:
            return name, got
    return None, None


def extract(
    t: Trajectory | dict,
    *,
    _prefix: str = "",
    _seq: list | None = None,
    _state: dict | None = None,
) -> dict:
    """Evidence lists for one trajectory (and its subagents, refs prefixed `sub<i>/`).

    Every item carries `seq`, the call's position in the run (subagents after the parent),
    so reads and writes can be ordered, and `before_agent_setup` (no agent_setup action at an
    earlier seq or earlier in the same command). Commands that may create or change files
    mint can't see are `opaque`; calls to tools without an interpreter are listed under
    `unknown`. `absent` (kind `not_on_path` / `module_missing`), `system` (only from shell
    version/OS queries), `package_lookup_failed` and `agent_setup` are starting-state facts.
    """
    t = t if isinstance(t, Trajectory) else Trajectory(t)
    seq = _seq if _seq is not None else [0]
    state = _state if _state is not None else {"setup_seq": None}
    ev: dict = {k: [] for k in KINDS} | {
        "unknown": [],
        "absent": [],
        "package_lookup_failed": [],
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
        cx = _Ctx(c, state)
        _facts(c, ref, ev, cx)
        by, items = _interpret(c)
        if items:
            for it in items:
                kind = it.pop("kind")
                idx = cx.locate(it.get("path") or it.get("url")) if cx.cmd else None
                if idx is None and cx.cmd:
                    idx = 0 if kind == "commands" else None
                ev.setdefault(kind, []).append(
                    {**ref, "by": by, **it, "before_agent_setup": cx.before(idx)}
                )
        else:
            ev["unknown"].append({**ref, "args": sorted(c.arguments)})
        if cx.setup and state["setup_seq"] is None:
            state["setup_seq"] = seq[0]
    for i, sub in enumerate(t.subagents):
        sev = extract(sub, _prefix=f"{_prefix}sub{i}/", _seq=seq, _state=state)
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
                "before_agent_setup": item.get("before_agent_setup", True),
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
    starting-state facts. `absent` (kind `not_on_path`: not on PATH for that shell at that step;
    `module_missing`: the interpreter command raised it), `system` (from version/OS queries only)
    and `package_lookup_failed` (not absence: the index lookup failed) each carry the producing
    `command`, `step` and `before_agent_setup`. True = nothing the agent installed or configured
    came earlier, so it is starting-state evidence; False = it only describes that moment.
    `agent_setup` is what the agent changed itself: that is the job (solution), never the
    starting environment."""

    def first(rows: list[dict], key: tuple[str, ...], keep: tuple[str, ...]) -> list[dict]:
        seen: dict = {}
        for r in rows:
            seen.setdefault(tuple(r.get(k) for k in key), {k: r.get(k) for k in keep})
        return list(seen.values())

    ctx = ("step", "command", "before_agent_setup")
    return {
        "counts": {k: len(ev.get(k, [])) for k in (*KINDS, "unknown")},
        "absent": first(ev.get("absent", []), ("kind", "what"), ("kind", "what", *ctx)),
        "system": first(ev.get("system", []), ("fact",), ("fact", *ctx)),
        "package_lookup_failed": first(
            ev.get("package_lookup_failed", []), ("what",), ("what", *ctx)
        ),
        "agent_setup": [
            {k: i.get(k) for k in ("command", "step", "before_agent_setup")}
            for i in ev.get("agent_setup", [])
        ],
        "starting_files": starting_files(ev),
        "outputs": outputs(ev),
        "sources": sorted({i["url"] for i in ev.get("sources", []) if i.get("url")}),
        "unknown_tools": sorted({i.get("tool") or "?" for i in ev.get("unknown", [])}),
    }
