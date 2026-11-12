"""Output conventions: --json result on stdout, progress on stderr, exit codes."""

from __future__ import annotations

import json
import sys
from typing import Any

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


class MINTError(Exception):
    """A user-facing error. Always carries a fix."""

    def __init__(self, message: str, fix: str = "", code: int = EXIT_ERROR, name: str = "error"):
        super().__init__(message)
        self.message, self.fix, self.code, self.name = message, fix, code, name


class UsageError(MINTError):
    def __init__(self, message: str, fix: str = ""):
        super().__init__(message, fix, EXIT_USAGE, "usage")


def progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Result:
    """What a command returns: data for --json, text for humans, next hints, exit code."""

    def __init__(
        self,
        data: dict[str, Any],
        text: str = "",
        next: list[str] | None = None,
        exit: int = EXIT_OK,
        warnings: list[str] | None = None,
    ):
        self.data, self.text, self.next, self.exit = data, text, next or [], exit
        self.warnings = warnings or []


def render(command: str, res: Result, as_json: bool) -> str:
    if as_json:
        body = {
            "ok": res.exit == EXIT_OK,
            "command": command,
            "data": res.data,
            "warnings": res.warnings,
            "next": res.next,
        }
        return json.dumps(body, indent=2, default=str)
    lines = [res.text.rstrip()] if res.text else []
    for w in res.warnings:
        lines.append(f"warning: {w}")
    for n in res.next:
        lines.append(f"next: {n}")
    return "\n".join(lines)


def render_error(command: str, err: MINTError, as_json: bool) -> str:
    if as_json:
        return json.dumps(
            {
                "ok": False,
                "command": command,
                "error": {"code": err.name, "message": err.message, "fix": err.fix},
                "next": [err.fix] if err.fix else [],
            },
            indent=2,
        )
    out = f"error: {err.message}"
    if err.fix:
        out += f"\nfix: {err.fix}"
    return out
