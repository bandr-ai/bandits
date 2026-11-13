"""Anonymous usage telemetry for milestone commands. Stdlib only.

The embedded value is the PostHog project write key (``phc_``). A personal key
(``phx_``) is rejected. ``MINT_POSTHOG_API_KEY`` or ``POSTHOG_PROJECT_TOKEN``,
when set, replaces it. An empty value turns capture off.
Never includes paths, prompts, task ids, queries, or log contents.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from . import __version__
from .output import progress
from .workspace import atomic_write

MILESTONES = frozenset(
    {
        "init",
        "new",
        "run",
        "validate",
        "judge-calibrate",
        "decide",
        "report",
        "task try",
    }
)

NOTICE = (
    "mint sends anonymous usage for init, new, run, validate, judge-calibrate, decide, "
    "report, and task try: version, OS, command, duration, outcome, flags, and bucketed counts. "
    "No paths, prompts, task ids, queries, answers, or log contents.\n"
    "Turn it off with `mint telemetry off` or MINT_TELEMETRY_DISABLED=1. It is also off when "
    "CI or DO_NOT_TRACK=1 is set. `mint telemetry status` shows the current state."
)

_DEFAULT_HOST = "https://us.i.posthog.com"
_TIMEOUT = 0.2
# Public project write key. Replaced at build time from the local project token.
_EMBEDDED_KEY = "phc_CvhAGavirL5hbBfgmGzacMUV7v8w4gHpi749P7KJ4XnG"
_RUN_STAGES = frozenset({"parse", "checks", "generate"})
_RUNNERS = frozenset({"local", "harbor"})
_BUCKETS = frozenset({"0", "1", "2-9", "10-99", "100-999", "1000+"})
_LEVELS = frozenset({"1", "2", "3", "1,2", "1,3", "2,3", "1,2,3"})
# All names and string values are static; no user-defined names enter the payload.
_ENUM_FIELDS = {
    "stage": _RUN_STAGES,
    "runner": _RUNNERS,
    "requested_runner": _RUNNERS | {"auto"},
    "levels": _LEVELS,
    "kind": frozenset(
        {
            "parser",
            "jobs",
            "check",
            "tasktype",
            "note",
            "spot_check",
            "derivability",
            "audit",
            "exclusion",
        }
    ),
    "verdict": frozenset({"pass", "fail"}),
    "verifier_outcome": frozenset({"pass", "fail"}),
    "failure_phase": frozenset({"workspace", "execute", "render"}),
    "error_type": frozenset({"usage", "contract", "interrupted", "operation", "internal"}),
}
_BOOL_FIELDS = frozenset(
    {
        "dry_run",
        "docker_present",
        "harbor_present",
        "allow_missing_runtime",
        "judge",
        "limited",
        "cached",
        "any_failure",
        "any_infra_error",
    }
)
_COUNT_FIELDS = frozenset(
    {
        "outputs",
        "errors",
        "queued",
        "blocked",
        "level1_checked",
        "level1_failed",
        "sampled",
        "passed",
        "failed",
        "infra_error",
        "resumed",
    }
)
_FALSE_CI = frozenset({"", "0", "false", "no"})


def bucket(n: int) -> str:
    n = 0 if n < 0 else int(n)
    if n <= 1:
        return str(n)
    if n < 10:
        return "2-9"
    if n < 100:
        return "10-99"
    if n < 1000:
        return "100-999"
    return "1000+"


def os_family() -> str:
    plat = sys.platform
    if plat.startswith("linux"):
        return "linux"
    if plat == "darwin":
        return "darwin"
    if plat in ("win32", "cygwin"):
        return "windows"
    return "other"


def config_path() -> Path | None:
    try:
        base = os.environ.get("XDG_CONFIG_HOME")
        if base is None or not base.strip():
            base = str(Path.home() / ".config")
        return Path(base) / "mint" / "telemetry.json"
    except (RuntimeError, OSError):
        return None


def _read(path: Path) -> tuple[dict | None, str]:
    """Return ``(data, kind)`` where kind is ``missing``, ``ok``, or ``unreadable``."""
    if not path.exists():
        return None, "missing"
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "unreadable"
    if not isinstance(data, dict):
        return None, "unreadable"
    if "enabled" in data and not isinstance(data["enabled"], bool):
        return None, "unreadable"
    return data, "ok"


def _write(path: Path, data: dict) -> None:
    body = {
        "anonymous_id": data["anonymous_id"],
        "enabled": data["enabled"] is True,
        "notice_shown": data.get("notice_shown") is True,
    }
    atomic_write(path, json.dumps(body, indent=2, sort_keys=True) + "\n")


def project_key() -> str:
    """Env override, then the embedded project write key. Empty means off."""
    if "MINT_POSTHOG_API_KEY" in os.environ:
        return os.environ["MINT_POSTHOG_API_KEY"].strip()
    if "POSTHOG_PROJECT_TOKEN" in os.environ:
        return os.environ["POSTHOG_PROJECT_TOKEN"].strip()
    return _EMBEDDED_KEY.strip()


def _early_block() -> str | None:
    """Gate steps that must not touch the config file. ``None`` means keep going."""
    key = project_key()
    if not key:
        return "no API key"
    if not key.startswith("phc_"):
        return "API key is not a project key"
    ci = os.environ.get("CI")
    if ci is not None and ci.strip().lower() not in _FALSE_CI:
        return "CI"
    if os.environ.get("DO_NOT_TRACK") == "1":
        return "DO_NOT_TRACK"
    if os.environ.get("MINT_TELEMETRY_DISABLED") == "1":
        return "MINT_TELEMETRY_DISABLED"
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return "pytest"
    return None


def gate() -> tuple[bool, str]:
    early = _early_block()
    if early:
        return False, early
    path = config_path()
    if path is None:
        return False, "config unreadable"
    data, kind = _read(path)
    if kind == "unreadable":
        return False, "config unreadable"
    if kind == "ok" and data is not None and data.get("enabled") is False:
        return False, "persisted off"
    return True, "on"


def status() -> dict[str, Any]:
    effective, reason = gate()
    path = config_path()
    persisted = "default"
    notice_shown = False
    if path is not None:
        data, kind = _read(path)
        if kind == "ok" and data is not None:
            if data.get("enabled") is True:
                persisted = "on"
            elif data.get("enabled") is False:
                persisted = "off"
            notice_shown = data.get("notice_shown") is True
    return {
        "effective": effective,
        "reason": reason,
        "persisted": persisted,
        "notice_shown": notice_shown,
        "config": "" if path is None else str(path),
    }


def set_enabled(enabled: bool) -> None:
    path = config_path()
    if path is None:
        return
    data, kind = _read(path)
    anon = ""
    notice = False
    if kind == "ok" and data is not None and isinstance(data.get("anonymous_id"), str):
        anon = data["anonymous_id"].strip()
        notice = data.get("notice_shown") is True
    if not anon:
        anon = str(uuid.uuid4())
        notice = False
    _write(path, {"anonymous_id": anon, "enabled": enabled, "notice_shown": notice})


def _levels(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        normalized = ",".join(str(n) for n in sorted({int(x) for x in value.split(",")}))
    except ValueError:
        return None
    return normalized if normalized in _LEVELS else None


def extra_for(command: str, args: Any, result: Any) -> dict[str, Any]:
    """Aggregate the invocation's summary, never the workspace's saved results.

    Unknown/malformed summary fields are omitted rather than reported as zero.
    """
    if args is None:
        return {}
    extra: dict[str, Any] = {"dry_run": bool(getattr(args, "dry_run", False))}
    data = getattr(result, "data", None)
    data = data if isinstance(data, dict) else {}

    def count(key: str, value: Any) -> None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            extra[key] = bucket(value)

    if command == "init":
        extra.update(
            docker_present=bool(shutil.which("docker")),
            harbor_present=bool(shutil.which("harbor")),
            allow_missing_runtime=bool(getattr(args, "allow_missing_runtime", False)),
        )
    elif command in ("new", "decide"):
        extra["kind"] = getattr(args, "kind", None)
        if command == "decide":
            extra["verdict"] = getattr(args, "verdict", None)
    elif command == "run":
        stage = getattr(args, "stage", None)
        extra["stage"] = stage
        extra["limited"] = getattr(args, "limit", None) is not None or bool(
            stage == "generate" and getattr(args, "type", None)
        )
        if isinstance(data.get("cached"), bool):
            extra["cached"] = data["cached"]
        if stage == "parse":
            count("outputs", data.get("trajectories"))
            count("errors", data.get("parse_errors"))
        elif stage == "checks":
            count("outputs", data.get("findings"))
            groups = data.get("error_groups")
            if isinstance(groups, list) and all(
                isinstance(g, dict) and type(g.get("count")) is int and g["count"] >= 0
                for g in groups
            ):
                count("errors", sum(g["count"] for g in groups))
        elif stage == "generate":
            count("outputs", data.get("tasks"))
            count("queued", data.get("queued"))
            statuses = data.get("by_status")
            if isinstance(statuses, dict):
                count("blocked", statuses.get("blocked", 0))
    elif command == "validate":
        extra["judge"] = bool(getattr(args, "judge", False))
        extra["requested_runner"] = getattr(args, "runner", None)
        extra["levels"] = _levels(getattr(args, "level", None))
        extra["runner"] = data.get("runner")
        for key in ("any_failure", "any_infra_error"):
            extra[key] = data.get(key)
        level1 = data.get("level1")
        if isinstance(level1, dict):
            count("level1_checked", level1.get("checked"))
            count("level1_failed", level1.get("failed"))
        types = data.get("types")
        # Empty types for level 1/3 does not mean level 2 processed zero tasks.
        levels = data.get("levels")
        if isinstance(types, dict) and isinstance(levels, list) and 2 in levels:
            for key in ("sampled", "passed", "failed", "infra_error", "resumed"):
                values = [v.get(key, []) for v in types.values() if isinstance(v, dict)]
                if len(values) == len(types) and all(isinstance(v, list) for v in values):
                    count(key, sum(len(v) for v in values))
    elif command == "task try" and data.get("state") == "ok":
        reward = data.get("reward")
        if type(reward) in (int, float) and 0 <= reward <= 1:
            extra["verifier_outcome"] = "pass" if reward >= 1 else "fail"
    return extra


def origin() -> str:
    raw = os.environ.get("MINT_POSTHOG_HOST", "").strip() or _DEFAULT_HOST
    return raw.rstrip("/")


def post(url: str, body: bytes, timeout: float) -> None:
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"mint/{__version__}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def _properties(command: str, exit_class: str, duration_ms: int, extra: dict[str, Any]) -> dict:
    props: dict[str, Any] = {
        "mint_version": __version__,
        "python_version": platform.python_version(),
        "os": os_family(),
        "command": command,
        "duration_ms": duration_ms,
        "exit": exit_class,
        "$process_person_profile": False,
        "$geoip_disable": True,
    }
    props["schema_version"] = 2
    allowed = {
        "init": ("docker_present", "harbor_present", "allow_missing_runtime"),
        "new": ("kind",),
        "run": ("stage", "limited", "cached", "outputs", "errors", "queued", "blocked"),
        "validate": (
            "judge",
            "requested_runner",
            "runner",
            "levels",
            "level1_checked",
            "level1_failed",
            "sampled",
            "passed",
            "failed",
            "infra_error",
            "resumed",
            "any_failure",
            "any_infra_error",
        ),
        "decide": ("kind", "verdict"),
        "task try": ("verifier_outcome",),
    }.get(command, ())
    for key in (*allowed, "dry_run", "failure_phase", "error_type"):
        value = extra.get(key)
        if key in _BOOL_FIELDS and isinstance(value, bool):
            props[key] = value
        elif isinstance(value, str):
            if key in _ENUM_FIELDS and value in _ENUM_FIELDS[key]:
                props[key] = value
            elif key in _COUNT_FIELDS and value in _BUCKETS:
                props[key] = value
    return props


def _prepare(command: str) -> str | None:
    """Return the anonymous id when the send gate is open, else None."""
    ok, _reason = gate()
    if not ok:
        return None
    path = config_path()
    if path is None:
        return None
    data, kind = _read(path)
    if kind == "unreadable":
        return None
    current = data or {}
    anon = current.get("anonymous_id")
    if not isinstance(anon, str) or not anon.strip():
        anon = str(uuid.uuid4())
        current = {
            "anonymous_id": anon,
            "enabled": True,
            "notice_shown": False,
        }
        dirty = True
    else:
        anon = anon.strip()
        current = {
            "anonymous_id": anon,
            "enabled": current.get("enabled") is not False,
            "notice_shown": current.get("notice_shown") is True,
        }
        dirty = data is None or "anonymous_id" not in (data or {})
    show = (not current["notice_shown"]) and (sys.stderr.isatty() or command == "report")
    if show:
        progress(NOTICE)
        current["notice_shown"] = True
        dirty = True
    if dirty:
        _write(path, current)
    return anon


def emit(
    command: str, exit_class: str, duration_ms: int, extra: dict[str, Any] | None = None
) -> None:
    try:
        if command not in MILESTONES:
            return
        anon = _prepare(command)
        if not anon:
            return
        key = project_key()
        payload = {
            "api_key": key,
            "event": "command_finished",
            "distinct_id": anon,
            "properties": _properties(command, exit_class, max(0, int(duration_ms)), extra or {}),
        }
        post(f"{origin()}/i/v0/e/", json.dumps(payload).encode(), _TIMEOUT)
    except Exception:
        return
