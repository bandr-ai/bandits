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
class _Brackets:
    """Start and end rows of one kind, keyed by id, with every way they can disagree."""

    name: str
    starts: Counter = field(default_factory=Counter)
    ends: Counter = field(default_factory=Counter)

    def problems(self) -> list[str]:
        found = []
        unended = sorted(k for k in self.starts if k not in self.ends)
        unstarted = sorted(k for k in self.ends if k not in self.starts)
        repeated_starts = sorted(k for k, n in self.starts.items() if n > 1)
        repeated_ends = sorted(k for k, n in self.ends.items() if n > 1)
        if unended:
            found.append(
                f"{len(unended)} {self.name}(s) started and never ended (process stopped "
                f"mid-way): {', '.join(unended[:5])}"
            )
        if unstarted:
            found.append(
                f"{len(unstarted)} {self.name}(s) ended with no start row: "
                f"{', '.join(unstarted[:5])}"
            )
        if repeated_starts:
            found.append(f"{self.name}(s) with more than one start: {repeated_starts[:5]}")
        if repeated_ends:
            found.append(f"{self.name}(s) with more than one end: {repeated_ends[:5]}")
        return found


@dataclass
class LedgerReport:
    """What one session's rows say, and where they fail to agree."""

    rows: int = 0
    calls_started: int = 0
    calls_completed: int = 0
    calls_failed: int = 0
    calls_by_role: Counter = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_missing: int = 0
    cost_reported: float = 0.0
    cost_unknown: int = 0
    cache_hits: int = 0
    finish_reasons: Counter = field(default_factory=Counter)
    invocations: dict[str, dict[str, Any]] = field(default_factory=dict)
    repl_steps: int = 0
    events: Counter = field(default_factory=Counter)
    run_started: bool = False
    run_finished: dict[str, Any] | None = None
    truncated_rows: int = 0
    uncorrelated_calls: int = 0
    """Call rows with no call id: written outside the forward boundary, so they
    cannot be paired with a start or an invocation."""

    duplicate_reads: int = 0
    """Helper calls repeated with identical arguments in one invocation — the
    re-reading a well-navigated account should not need. Measured, not refused."""

    brackets: list[_Brackets] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not self.problems()

    def problems(self) -> list[str]:
        found = [problem for bracket in self.brackets for problem in bracket.problems()]
        if self.uncorrelated_calls:
            found.append(
                f"{self.uncorrelated_calls} call row(s) carry no call id and cannot be paired"
            )
        if self.truncated_rows:
            found.append("the last row was cut short (a write interrupted by a crash)")
        found.extend(self.mismatches)
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
    """Pair, sum and cross-check one session's rows; compare with its session state.

    Calls, invocations, REPL steps and runs are each checked as brackets: every
    start has exactly one end and every end exactly one start. A row that cannot
    be paired at all, a cut-short last row, an invocation whose reported call
    count differs from the calls recorded under it, or session totals the rows
    do not reproduce all make the report inconsistent.
    """
    report = LedgerReport(rows=len(rows))
    calls = _Brackets("call")
    invocations = _Brackets("invocation")
    repl = _Brackets("REPL step")
    runs = _Brackets("run")
    report.brackets = [calls, invocations, repl, runs]
    terminals: dict[str, dict[str, Any]] = {}
    calls_by_invocation: Counter = Counter()
    reads: Counter = Counter()
    run_key = 0
    for row in rows:
        kind = row.get("event_type")
        report.events[kind] += 1
        if kind == "truncated_row":
            report.truncated_rows += 1
        elif kind == "model_call_start":
            if row.get("call_id"):
                calls.starts[row["call_id"]] += 1
            else:
                report.uncorrelated_calls += 1
        elif kind in _CALL_TERMINALS:
            if not row.get("call_id"):
                report.uncorrelated_calls += 1
                continue
            calls.ends[row["call_id"]] += 1
            terminals[row["call_id"]] = row
            if row.get("invocation_id"):
                calls_by_invocation[row["invocation_id"]] += 1
        elif kind == "invocation_start":
            invocations.starts[row["invocation_id"]] += 1
            report.invocations.setdefault(row["invocation_id"], {})["start"] = row
        elif kind == "invocation_end":
            invocations.ends[row["invocation_id"]] += 1
            report.invocations.setdefault(row["invocation_id"], {})["end"] = row
        elif kind == "evidence_access":
            reads[
                (
                    row.get("invocation_id"),
                    row.get("tool"),
                    row.get("ref"),
                    row.get("cursor"),
                    row.get("limit"),
                    row.get("start"),
                    row.get("end"),
                )
            ] += 1
        elif kind == "repl_start":
            repl.starts[row.get("repl_id", "?")] += 1
            report.repl_steps += 1
        elif kind in ("repl_end", "repl_error"):
            repl.ends[row.get("repl_id", "?")] += 1
        elif kind == "run_started":
            # A resumed session runs again under the same id: runs pair in order.
            run_key += 1
            runs.starts[f"run {run_key}"] += 1
            report.run_started = True
        elif kind == "run_finished":
            runs.ends[f"run {run_key}"] += 1
            report.run_finished = row

    report.duplicate_reads = sum(n - 1 for n in reads.values() if n > 1)
    report.calls_started = sum(calls.starts.values())
    for row in terminals.values():
        report.calls_by_role[_role(row)] += 1
        if row["event_type"] == "model_call_error":
            report.calls_failed += 1
            report.cost_unknown += 1
            continue
        report.calls_completed += 1
        report.finish_reasons[(row.get("response") or {}).get("finish_reason")] += 1
        usage = row.get("usage") or {}
        if isinstance(usage.get("prompt_tokens"), int):
            report.prompt_tokens += usage["prompt_tokens"]
            report.completion_tokens += usage.get("completion_tokens") or 0
        else:
            report.usage_missing += 1
        if row.get("cache_hit"):
            report.cache_hits += 1
        elif isinstance(row.get("cost_usd"), (int, float)):
            report.cost_reported += float(row["cost_usd"])
        else:
            report.cost_unknown += 1

    for invocation_id, pair in report.invocations.items():
        end = pair.get("end")
        if end is None:
            continue  # reported as an unended invocation
        reported = end.get("llm_calls")
        seen = calls_by_invocation.get(invocation_id, 0)
        if isinstance(reported, int) and reported != seen:
            report.mismatches.append(
                f"invocation {invocation_id} reports {reported} call(s); the ledger "
                f"holds {seen} terminal row(s) under it"
            )

    if session is not None:
        recorded = sum(calls.ends.values())
        if session.llm_calls != recorded:
            report.mismatches.append(
                f"the session counts {session.llm_calls} call(s); the ledger holds "
                f"{recorded} terminal call row(s)"
            )
        if abs(session.cost_usd - report.cost_reported) > 1e-9:
            report.mismatches.append(
                f"the session reports ${session.cost_usd:.6f}; ledger rows report "
                f"${report.cost_reported:.6f}"
            )
    return report


def load_session_rows(ledger: Path, session_id: str | None) -> list[dict[str, Any]]:
    return session_rows(read_events(ledger), session_id)


def call_rows(rows: Sequence[dict[str, Any]], call_id: str) -> list[dict[str, Any]]:
    """Every row of one call, start to terminal, blobs restored."""
    return [row for row in rows if row.get("call_id") == call_id]
