"""One definition of "the records in this file", shared by detection and the readers.

A trace export is a JSON document, or JSON lines, or a stream of JSON documents written one
after another (pretty-printed, or several to a line). Each reader used to decide this for
itself, and each got a different corner wrong: ``str.splitlines()`` also splits on U+2028,
U+2029 and U+0085, which a JSON string may contain raw, so a valid line was cut in two; a
BOM, a bad byte or a blank line made a reader refuse the whole file.

Here a file is decoded once and walked one JSON value at a time, so none of that depends on
where the lines fall. A stretch that is not JSON is skipped to the next line that starts a
value, and reported once with the lines it covered. Nothing is dropped without an issue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from bandits.traces import TraceIssue

_JSON_SPACE = " \t\n\r"
_DECODER = json.JSONDecoder()


@dataclass(frozen=True)
class Record:
    value: Any
    line: int
    """The 1-based line the record starts on."""


@dataclass
class Records:
    records: list[Record] = field(default_factory=list)
    issues: list[TraceIssue] = field(default_factory=list)
    layout: Literal["document", "stream"] = "document"
    """``document``: the file is one JSON value. ``stream``: several, one after another."""
    cut: bool = False
    """Reading stopped at ``limit`` before the end of the text."""
    bad_lines: int = 0
    """Lines skipped because they were not JSON."""

    @property
    def values(self) -> list[Any]:
        return [r.value for r in self.records]


def decode(data: bytes, location: str, issues: list[TraceIssue]) -> str:
    """UTF-8 text without a BOM. A bad byte is replaced and counted, never a reason to refuse."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", "replace")
        # A U+FFFD the file itself spelled out is not a replacement.
        replaced = text.count("�") - data.count(b"\xef\xbf\xbd")
        issues.append(
            TraceIssue(
                kind="invalid_utf8",
                detail=f"{replaced} byte sequence(s) were not valid UTF-8 and were replaced",
                location=location,
            )
        )
    return text[1:] if text.startswith("﻿") else text


def _skip_space(text: str, at: int) -> int:
    end = len(text)
    while at < end and text[at] in _JSON_SPACE:
        at += 1
    return at


def _next_record_start(text: str, after: int) -> int:
    """The start of the next line (after ``after``) that begins with ``{`` or ``[``, or -1."""
    position = text.find("\n", after)
    while position != -1:
        position += 1
        if position < len(text) and text[position] in "{[":
            return position
        position = text.find("\n", position)
    return -1


def read_records(
    data: bytes, location: str = "", *, limit: int | None = None, tolerate_cut: bool = False
) -> Records:
    """Every JSON value in ``data``, in order, with the problems met on the way.

    ``limit`` stops after that many records (a sample for detection). ``tolerate_cut`` treats a
    value that is cut off at the very end of ``data`` as the end of a sample, not a problem:
    use it when ``data`` is only the front of a larger file.
    """
    out = Records()
    text = decode(data, location, out.issues)
    end = len(text)
    at = _skip_space(text, 0)
    line = 1 + text.count("\n", 0, at)
    counted_to = at
    bad_from: int | None = None  # first line of a stretch that was not JSON
    bad_error = ""

    def flush_bad(until_line: int) -> None:
        nonlocal bad_from
        if bad_from is None:
            return
        span = (
            f"line {bad_from}"
            if until_line - 1 <= bad_from
            else f"lines {bad_from}-{until_line - 1}"
        )
        out.issues.append(
            TraceIssue(
                kind="malformed_json",
                detail=f"{span}: not readable as JSON ({bad_error})",
                location=f"{location}:{bad_from}" if location else str(bad_from),
            )
        )
        out.bad_lines += max(1, until_line - bad_from)
        bad_from = None

    while at < end:
        if limit is not None and len(out.records) >= limit:
            out.cut = True
            break
        line += text.count("\n", counted_to, at)
        counted_to = at
        try:
            value, stop = _DECODER.raw_decode(text, at)
        except json.JSONDecodeError as exc:
            if tolerate_cut and text.find("\n", at) == -1:
                # The final, partial line of a sample taken from the front of a larger file.
                out.cut = True
                break
            if bad_from is None:
                bad_from, bad_error = line, exc.msg
            # Resume at the next line that opens an object or array at its left edge. A line
            # inside a cut-off, pretty-printed document is indented, so it is not mistaken
            # for the start of a record.
            at = _next_record_start(text, at)
            if at == -1:
                at = end
                break
            continue
        flush_bad(line)
        out.records.append(Record(value, line))
        at = _skip_space(text, stop)
    flush_bad(1 + text.rstrip().count("\n") + 1)
    # Only a stretch that was not JSON makes it a stream; a replaced byte does not.
    malformed = any(i.kind == "malformed_json" for i in out.issues)
    out.layout = "document" if len(out.records) == 1 and not malformed else "stream"
    return out
