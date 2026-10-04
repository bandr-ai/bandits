"""Read a mining session back from its ledger, and check that it adds up.

The ledger is only worth having if a run can be reconstructed from it after the
fact — what was sent, what came back, what the sandbox did, and why the run
stopped — without another paid call. This module is that check: it pairs every
call start with its terminal row, every invocation and REPL step with its end,
sums what providers reported, and compares the result with the session's own
totals. A mismatch is reported, never smoothed over: an unterminated call is a
call the process did not live to finish; an uncounted one is a recording gap.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandits.ledger import read_events

_CALL_TERMINALS = ("model_call", "model_call_error")


@dataclass
class LedgerReport:
    """What one session's rows say, and where they fail to agree."""

    rows: int = 0
    calls_started: int = 0
    calls_completed: int = 0
    calls_failed: int = 0
    unterminated_calls: list[str] = field(default_factory=list)
    duplicate_terminals: list[str] = field(default_factory=list)
    calls_by_role: Counter = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_missing: int = 0
    cost_reported: float = 0.0
    cost_unknown: int = 0
    cache_hits: int = 0
    finish_reasons: Counter = field(default_factory=Counter)
    invocations: dict[str, dict[str, Any]] = field(default_factory=dict)
    unended_invocations: list[str] = field(default_factory=list)
    invocation_mismatches: list[str] = field(default_factory=list)
    repl_steps: int = 0
    unended_repl: list[str] = field(default_factory=list)
    events: Counter = field(default_factory=Counter)
    run_started: bool = False
    run_finished: dict[str, Any] | None = None
    truncated_rows: int = 0
    session_mismatches: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not (
            self.unterminated_calls
            or self.duplicate_terminals
            or self.unended_invocations
            or self.invocation_mismatches
            or self.unended_repl
            or self.truncated_rows
            or self.session_mismatches
        )

    def problems(self) -> list[str]:
        found = []
        if self.unterminated_calls:
            found.append(
                f"{len(self.unterminated_calls)} call(s) started and never ended (process "
                f"stopped mid-call): {', '.join(self.unterminated_calls[:5])}"
            )
        if self.duplicate_terminals:
            found.append(f"call(s) with more than one terminal row: {self.duplicate_terminals}")
        if self.unended_invocations:
            found.append(f"invocation(s) without an end: {self.unended_invocations}")
        found.extend(self.invocation_mismatches)
        if self.unended_repl:
            found.append(f"REPL step(s) without an end: {self.unended_repl}")
        if self.truncated_rows:
            found.append("the last row was cut short (a write interrupted by a crash)")
        found.extend(self.session_mismatches)
        return found


def _role(row: dict[str, Any]) -> str:
    """Root, subcall, adapter fallback or extraction, from where the call sat."""
    stage = row.get("stage")
    if stage == "rlm_extract":
        return "extract"
    if stage == "rlm_repl":
        return "subcall"
    settings = (row.get("request") or {}).get("settings") or {}
    if "response_format" in settings:
        return "adapter_fallback"
    if stage == "rlm_iteration":
        return "root"
    return stage or "other"


def session_rows(rows: Iterable[dict[str, Any]], session_id: str | None) -> list[dict[str, Any]]:
    """The rows one mining session wrote (all rows when ``session_id`` is None)."""
    if session_id is None:
        return list(rows)
    return [row for row in rows if row.get("session_id") == session_id]


def reconcile(rows: Sequence[dict[str, Any]], *, session: Any = None) -> LedgerReport:
    """Pair, sum and cross-check one session's rows; compare with its session state."""
    report = LedgerReport(rows=len(rows))
    starts: dict[str, dict[str, Any]] = {}
    terminals: dict[str, list[dict[str, Any]]] = {}
    calls_by_invocation: Counter = Counter()
    repl_open: dict[str, bool] = {}
    for row in rows:
        kind = row.get("event_type")
        report.events[kind] += 1
        if kind == "truncated_row":
            report.truncated_rows += 1
        elif kind == "model_call_start" and row.get("call_id"):
            starts[row["call_id"]] = row
        elif kind in _CALL_TERMINALS and row.get("call_id"):
            terminals.setdefault(row["call_id"], []).append(row)
            if row.get("invocation_id"):
                calls_by_invocation[row["invocation_id"]] += 1
        elif kind == "invocation_start":
            report.invocations[row["invocation_id"]] = {"start": row}
        elif kind == "invocation_end":
            report.invocations.setdefault(row["invocation_id"], {})["end"] = row
        elif kind == "repl_start":
            repl_open[row.get("repl_id", "?")] = True
            report.repl_steps += 1
        elif kind in ("repl_end", "repl_error"):
            repl_open[row.get("repl_id", "?")] = False
        elif kind == "run_started":
            report.run_started = True
        elif kind == "run_finished":
            report.run_finished = row

    for call_id in starts:
        ends = terminals.get(call_id, [])
        if not ends:
            report.unterminated_calls.append(call_id)
        elif len(ends) > 1:
            report.duplicate_terminals.append(call_id)
    for call_id, ends in terminals.items():
        report.calls_started += call_id in starts
        end = ends[-1]
        report.calls_by_role[_role(end)] += 1
        if end["event_type"] == "model_call_error":
            report.calls_failed += 1
            report.cost_unknown += 1
            continue
        report.calls_completed += 1
        report.finish_reasons[(end.get("response") or {}).get("finish_reason")] += 1
        usage = end.get("usage") or {}
        if isinstance(usage.get("prompt_tokens"), int):
            report.prompt_tokens += usage["prompt_tokens"]
            report.completion_tokens += usage.get("completion_tokens") or 0
        else:
            report.usage_missing += 1
        if end.get("cache_hit"):
            report.cache_hits += 1
        elif isinstance(end.get("cost_usd"), (int, float)):
            report.cost_reported += float(end["cost_usd"])
        else:
            report.cost_unknown += 1
    report.calls_started = len(starts)

    for invocation_id, pair in report.invocations.items():
        end = pair.get("end")
        if end is None:
            report.unended_invocations.append(invocation_id)
            continue
        reported = end.get("llm_calls")
        seen = calls_by_invocation.get(invocation_id, 0)
        if isinstance(reported, int) and reported != seen:
            report.invocation_mismatches.append(
                f"invocation {invocation_id} reports {reported} call(s); the ledger "
                f"holds {seen} terminal row(s) under it"
            )
    report.unended_repl = [repl_id for repl_id, open_ in repl_open.items() if open_]

    if session is not None:
        recorded = sum(len(ends) for ends in terminals.values())
        if session.llm_calls != recorded:
            report.session_mismatches.append(
                f"the session counts {session.llm_calls} call(s); the ledger holds "
                f"{recorded} terminal call row(s)"
            )
        if session.cost_usd and abs(session.cost_usd - report.cost_reported) > 1e-9:
            report.session_mismatches.append(
                f"the session reports ${session.cost_usd:.6f}; ledger rows report "
                f"${report.cost_reported:.6f}"
            )
    return report


def load_session_rows(ledger: Path, session_id: str | None) -> list[dict[str, Any]]:
    return session_rows(read_events(ledger), session_id)


def call_rows(rows: Sequence[dict[str, Any]], call_id: str) -> list[dict[str, Any]]:
    """Every row of one call, start to terminal, blobs restored."""
    return [row for row in rows if row.get("call_id") == call_id]
