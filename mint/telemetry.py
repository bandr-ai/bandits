"""Anonymous usage telemetry for milestone commands. Stdlib only.

Sends nothing unless ``MINT_POSTHOG_API_KEY`` is a project key (``phc_``).
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
    "report, and task try: version, OS, command, duration, and outcome. No paths, prompts, "
    "task ids, queries, or log contents.\n"
    "Turn it off with `mint telemetry off` or MINT_TELEMETRY_DISABLED=1. It is also off when "
    "CI or DO_NOT_TRACK=1 is set. `mint telemetry status` shows the current state."
)

_DEFAULT_HOST = "https://us.i.posthog.com"
_TIMEOUT = 0.2
_RUN_STAGES = frozenset({"parse", "checks", "generate"})
_RUNNERS = frozenset({"local", "harbor"})
_COUNT_KEYS = ("validated", "generated", "blocked", "exec_failed")
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


def _early_block() -> str | None:
    """Gate steps that must not touch the config file. ``None`` means keep going."""
    key = os.environ.get("MINT_POSTHOG_API_KEY")
    if key is None or not key.strip():
        return "no API key"
    if not key.strip().startswith("phc_"):
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


def extra_for(command: str, args: Any, result: Any, ws: Any) -> dict[str, Any]:
    """Allow-listed properties only. Never copies ``Result.data``."""
    if args is None:
        return {}
    if command == "init":
        return {
            "docker_present": bool(shutil.which("docker")),
            "harbor_present": bool(shutil.which("harbor")),
            "allow_missing_runtime": bool(getattr(args, "allow_missing_runtime", False)),
            "dry_run": bool(getattr(args, "dry_run", False)),
        }
    if command == "run":
        stage = getattr(args, "stage", None)
        return {"stage": stage} if stage in _RUN_STAGES else {}
    if command == "validate":
        extra: dict[str, Any] = {"judge": bool(getattr(args, "judge", False))}
        if result is None:
            return extra
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            runner = data.get("runner")
            if runner in _RUNNERS:
                extra["runner"] = runner
            levels = data.get("levels")
            if isinstance(levels, list) and levels:
                try:
                    nums = sorted(int(x) for x in levels)
                except (TypeError, ValueError):
                    nums = []
                if nums:
                    extra["levels"] = ",".join(str(n) for n in nums)
        if ws is not None:
            counts = {key: 0 for key in _COUNT_KEYS}
            for row in ws.result_rows():
                status = row.get("status")
                if status in counts:
                    counts[status] += 1
            for key, n in counts.items():
                extra[key] = bucket(n)
        return extra
    return {}


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
    allowed = {
        "init": ("docker_present", "harbor_present", "allow_missing_runtime", "dry_run"),
        "run": ("stage",),
        "validate": ("judge", "runner", "levels", *_COUNT_KEYS),
    }.get(command, ())
    for key in allowed:
        if key not in extra:
            continue
        value = extra[key]
        if key in ("docker_present", "harbor_present", "allow_missing_runtime", "dry_run", "judge"):
            if isinstance(value, bool):
                props[key] = value
        elif key == "stage" and value in _RUN_STAGES:
            props[key] = value
        elif key == "runner" and value in _RUNNERS:
            props[key] = value
        elif key == "levels" and isinstance(value, str):
            props[key] = value
        elif key in _COUNT_KEYS and isinstance(value, str):
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


def emit(command: str, exit_class: str, duration_ms: int, extra: dict[str, Any] | None = None) -> None:
    try:
        anon = _prepare(command)
        if not anon:
            return
        key = os.environ.get("MINT_POSTHOG_API_KEY", "").strip()
        payload = {
            "api_key": key,
            "event": "command_finished",
            "distinct_id": anon,
            "properties": _properties(command, exit_class, max(0, int(duration_ms)), extra or {}),
        }
        post(f"{origin()}/i/v0/e/", json.dumps(payload).encode(), _TIMEOUT)
    except BaseException:
        return
