"""A resumable mining session, written to disk as it goes.

Why this is not just the run artifact. A run is content-addressed and
immutable: it exists only once the run finishes, which is exactly when it stops
being useful for answering "what is this doing right now" and "what did it get
through before it died". A run over 160 traces makes dozens of model calls over
many minutes and can fail at any of them, and until now a failure at chunk three
of pass two lost everything the run had learned.

So the session is a mutable working directory beside the artifacts. It is
rewritten after every chunk — not every pass — so the state on disk is never
more than one model call behind what the run has actually done. A person
watching it can see which pass and chunk is in flight, how many traces have been
read, what the taxonomy currently holds and what it cost so far; a run that dies
can be resumed from the last chunk that completed rather than from the start.

The session is deliberately *not* evidence. It is a scratchpad that gets
overwritten, and nothing downstream may cite it. When a run reaches its pause,
the immutable ``RLMClusteringRun`` is written from it in the ordinary way, and that
is what an audit, an assignment or a task set is ever parented to.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from bandits.analyze.rlm_account import RunAccount, latest_by_run
from bandits.analyze.rlm_models import (
    ChunkResult,
    FamilyContract,
    PassResult,
    ResumeScope,
    TraceView,
)
from bandits.traces import Contract


class SessionState(Contract):
    """Everything a resumed run needs, and everything a watcher wants to see.

    One flat record rather than a log to replay: a resumed session should not
    have to reconstruct a taxonomy by re-applying every operation it once made,
    because that reconstruction can differ from what the run actually held and
    the difference would be silent.
    """

    schema_version: int = 1
    session_id: str
    analysis_id: str
    view: TraceView
    model: str
    seed: int
    requested_passes: int = Field(ge=1)

    status: str = "running"
    """One of running, awaiting_review, incomplete, interrupted, failed. Written
    before anything else so a crashed run is distinguishable from one still
    working; see :meth:`observed_status` for a run whose process is gone."""

    pid: int | None = None
    """The process writing this session, so a reader can tell a live run from
    one that died without closing its file."""

    pass_index: int = Field(default=0, ge=0)
    """The pass currently in flight, zero-based."""

    completed_passes: int = Field(default=0, ge=0)
    chunk_index: int = Field(default=0, ge=0)
    """Chunks finished across the whole run, not within the current pass."""

    traces_total: int = Field(default=0, ge=0)
    traces_seen_this_pass: int = Field(default=0, ge=0)
    """Progress within the pass in flight. The number a watcher reads to know
    how far through a half-finished pass the run actually is."""

    contracts: tuple[FamilyContract, ...] = ()
    assignments: dict[str, str] = Field(default_factory=dict)
    ambiguous_trace_ids: tuple[str, ...] = ()
    uncovered_trace_ids: tuple[str, ...] = ()
    seen_this_pass: tuple[str, ...] = ()
    """Traces already read in the pass in flight, so a resume does not reread
    them and does not count them twice toward pass completion."""

    pass_order: tuple[str, ...] = ()
    """The shuffled order this pass is working through. Persisted because the
    shuffle is seeded per pass, and a resume that reshuffled would read some
    traces twice and others not at all."""

    passes: tuple[PassResult, ...] = ()
    chunks: tuple[ChunkResult, ...] = ()
    llm_calls: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    tokens: dict[str, int] = Field(default_factory=dict)
    started_at: str = ""
    updated_at: str = ""
    elapsed_seconds: float = Field(default=0.0, ge=0)
    resumed_from: str | None = None
    resume_scope: ResumeScope | None = None
    last_error: str = ""

    accounts: tuple[RunAccount, ...] = ()
    """Every account attempt so far. Accepted ones survive an interruption and
    are reused on resume; nothing else is assumed complete."""

    selection: tuple[str, ...] = ()
    """Explicitly selected trace ids; empty means the whole corpus."""

    settings: dict[str, Any] = Field(default_factory=dict)
    settings_digest: str = ""
    """Digest of the generation/invocation settings. Resume refuses a mismatch."""

    budget_usage: dict[str, Any] = Field(default_factory=dict)
    """The admission guard's own counts: calls admitted before dispatch,
    reported and estimated spend, unknown-cost calls, refusals."""

    def observed_status(self, *, alive: Any = None) -> str:
        """``status``, except a "running" session whose process is gone is said so."""
        if self.status != "running":
            return self.status
        if self.pid is None:
            return "running (unverified: no process id recorded)"
        check = alive or _process_alive
        return self.status if check(self.pid) else "stale (process gone)"

    def coverage(self) -> dict[str, int]:
        """Counted separately; ``read`` never means an id merely appeared in a batch."""
        latest = latest_by_run(self.accounts)
        attempted = {t for c in self.chunks for t in c.trace_ids} | set(latest)
        return {
            "attempted": len(attempted),
            "source_accessed": sum(1 for a in latest.values() if a.completion.inspected_refs),
            "account_complete": sum(1 for a in latest.values() if a.status == "accepted"),
            "eligible": sum(1 for a in latest.values() if a.eligible_for_families),
            "quarantined": sum(1 for a in latest.values() if a.status == "quarantined")
            + sum(1 for c in self.chunks if c.status == "quarantined"),
            "failed": sum(1 for a in latest.values() if a.status in ("failed", "rejected"))
            + sum(1 for c in self.chunks if c.status == "error"),
            "assigned": len(self.assignments),
            "unresolved": len(set(self.ambiguous_trace_ids) | set(self.uncovered_trace_ids)),
        }

    @property
    def progress(self) -> str:
        """One line a person can read mid-run without parsing anything."""
        line = (
            f"pass {self.pass_index + 1}/{self.requested_passes} · "
            f"{self.traces_seen_this_pass}/{self.traces_total} traces · "
            f"chunk {self.chunk_index} · {len(self.contracts)} contracts · "
            f"{self.llm_calls} calls · ${self.cost_usd:.4f}"
        )
        if self.accounts:
            cov = self.coverage()
            line += (
                f" · accounts {cov['account_complete']}/{self.traces_total} "
                f"({cov['quarantined']} quarantined, {cov['failed']} failed)"
            )
        return line


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _summed_calls(chunks: Any, accounts: Any) -> int:
    return sum(record.llm_calls or 0 for record in [*chunks, *accounts])


def _summed_usd(chunks: Any, accounts: Any) -> float:
    """Reported cost only; unknown cost stays visible in each record, never as zero spend."""
    return sum(record.cost_usd or 0.0 for record in [*chunks, *accounts])


def _summed_tokens(chunks: Any, accounts: Any) -> dict[str, int]:
    """Top-level tokens summed from what each invocation reported; empty if none did."""
    totals: dict[str, int] = {}
    for record in [*chunks, *accounts]:
        for field, value in (record.tokens or {}).items():
            totals[field] = totals.get(field, 0) + value
    return totals


class SessionStore:
    """Mutable session state, one directory per session.

    Kept out of ``DerivedStore`` on purpose: that store is content-addressed and
    refuses to overwrite, which is the right guarantee for evidence and the
    wrong one for a file that must be rewritten after every chunk.
    """

    def __init__(self, project_dir: Path | str = Path(".bandits")) -> None:
        self._root = Path(project_dir) / "sessions"

    def _dir(self, session_id: str) -> Path:
        return self._root / session_id

    def path(self, session_id: str) -> Path:
        return self._dir(session_id) / "session.json"

    def write(self, state: SessionState) -> None:
        """Persist the session, atomically.

        Atomic because this is rewritten constantly and a crash during the write
        would otherwise leave a truncated file — losing the very state the file
        exists to preserve. The temporary file is replaced in one operation, so
        a reader either sees the previous state or the new one, never half.
        """
        directory = self._dir(state.session_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json(indent=2).encode("utf-8")
        target = directory / "session.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, target)

    def read(self, session_id: str) -> SessionState:
        return SessionState.model_validate_json(self.path(session_id).read_bytes())

    def exists(self, session_id: str) -> bool:
        return self.path(session_id).is_file()

    def list(self) -> list[SessionState]:
        if not self._root.exists():
            return []
        states = []
        for entry in sorted(self._root.iterdir()):
            if (entry / "session.json").is_file():
                try:
                    states.append(self.read(entry.name))
                except ValueError:
                    # A session written by an older schema is listed as
                    # unreadable elsewhere rather than crashing the listing.
                    continue
        return sorted(states, key=lambda s: s.updated_at, reverse=True)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        """Append one line to the session's own progress log.

        Separate from the state file because the state is overwritten and this
        is not: the state answers "where is it now", the log answers "how did it
        get here", and a run that behaved strangely needs the second.

        Best effort. A run must not die because its log could not be written.
        """
        directory = self._dir(session_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"at": datetime.now(UTC).isoformat(), **event}, default=str)
            with (directory / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def read_events(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        path = self._dir(session_id) / "progress.jsonl"
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        return events


def new_session_id(analysis_id: str, view: TraceView, seed: int) -> str:
    """A readable, collision-resistant id naming the run's own parameters."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"rlm-{view.value}-{analysis_id[-8:]}-s{seed}-{stamp}"


class SessionRecorder:
    """Writes a session's state after every chunk, and its log as it goes.

    Passed into :func:`~bandits.analyze.rlm_mine.mine_taxonomy` so the loop
    stays ignorant of storage: the miner calls ``checkpoint`` and this decides
    what that means. A run given no recorder behaves exactly as before, which
    keeps the tests free of a filesystem.
    """

    def __init__(
        self,
        store: SessionStore,
        *,
        session_id: str,
        analysis_id: str,
        view: TraceView,
        model: str,
        resumed_from: str | None = None,
        resume_scope: ResumeScope | None = None,
        selection: tuple[str, ...] = (),
        settings: dict[str, Any] | None = None,
        settings_digest: str = "",
        guard: Any = None,
        previous: SessionState | None = None,
    ) -> None:
        self._store = store
        self._guard = guard
        self._previous_accounts = previous.accounts if previous is not None else ()
        self._previous_chunks = previous.chunks if previous is not None else ()
        self._state = SessionState(
            session_id=session_id,
            analysis_id=analysis_id,
            view=view,
            model=model,
            seed=0,
            requested_passes=1,
            started_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
            resumed_from=resumed_from,
            resume_scope=resume_scope,
            selection=selection,
            settings=settings or {},
            settings_digest=settings_digest,
            pid=os.getpid(),
            accounts=tuple(self._previous_accounts),
        )

    @property
    def session_id(self) -> str:
        return self._state.session_id

    @property
    def state(self) -> SessionState:
        return self._state

    def begin(self, *, traces_total: int, requested_passes: int, seed: int) -> None:
        self._state = self._state.replace(
            traces_total=traces_total,
            requested_passes=requested_passes,
            seed=seed,
            status="running",
            pid=os.getpid(),
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "session_started",
                "traces": traces_total,
                "passes": requested_passes,
                "seed": seed,
                "view": self._state.view.value,
            },
        )

    def checkpoint(
        self,
        state: Any,
        *,
        pass_index: int,
        completed_passes: int,
        chunk: ChunkResult,
        chunks: Any,
        passes: Any,
        seen_this_pass: set[str],
        pass_order: tuple[str, ...],
        calls: int,
        usd: float,
        elapsed: float,
    ) -> None:
        """Persist everything a resume needs, after one chunk.

        The whole taxonomy is rewritten each time rather than a delta appended,
        because a resume that had to replay deltas could reconstruct a state the
        run never actually held, and the difference would be invisible.
        """
        self._state = self._state.replace(
            pass_index=pass_index,
            completed_passes=completed_passes,
            chunk_index=len(tuple(chunks)),
            traces_seen_this_pass=len(seen_this_pass),
            contracts=tuple(sorted(state.contracts.values(), key=lambda c: c.contract_id)),
            assignments=dict(state.assignments),
            ambiguous_trace_ids=tuple(sorted(state.ambiguous)),
            uncovered_trace_ids=tuple(sorted(state.uncovered)),
            seen_this_pass=tuple(sorted(seen_this_pass)),
            pass_order=pass_order,
            passes=tuple(passes),
            chunks=tuple(self._previous_chunks) + tuple(chunks),
            llm_calls=_summed_calls(
                tuple(self._previous_chunks) + tuple(chunks), self._state.accounts
            ),
            cost_usd=_summed_usd(
                tuple(self._previous_chunks) + tuple(chunks), self._state.accounts
            ),
            tokens=_summed_tokens(
                tuple(self._previous_chunks) + tuple(chunks), self._state.accounts
            ),
            elapsed_seconds=elapsed,
            updated_at=datetime.now(UTC).isoformat(),
            last_error=chunk.error if chunk.status != "success" else "",
            budget_usage=self._guard.summary() if self._guard is not None else {},
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "chunk_complete",
                "pass": pass_index,
                "chunk": chunk.index,
                "traces": len(chunk.trace_ids),
                "operations": [op.operation.value for op in chunk.operations],
                "contracts": len(self._state.contracts),
                "seen_this_pass": len(seen_this_pass),
                "of": self._state.traces_total,
                "calls": calls,
                "cost_usd": round(usd, 6),
                "status": chunk.status,
                "error": chunk.error,
                "completion_mode": chunk.completion_mode,
                "iterations_used": chunk.iterations_used,
                "failure_kind": chunk.failure_kind,
                "attempt": chunk.attempt,
            },
        )

    def checkpoint_account(self, account: RunAccount, accounts: Any, *, elapsed: float) -> None:
        """Persist every account attempt as it lands, accepted or not."""
        self._state = self._state.replace(
            accounts=tuple(accounts),
            llm_calls=_summed_calls(self._state.chunks, tuple(accounts)),
            cost_usd=_summed_usd(self._state.chunks, tuple(accounts)),
            tokens=_summed_tokens(self._state.chunks, tuple(accounts)),
            elapsed_seconds=elapsed,
            updated_at=datetime.now(UTC).isoformat(),
            last_error=account.completion.error or "; ".join(account.validation_errors[:2]),
            budget_usage=self._guard.summary() if self._guard is not None else {},
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "account_complete",
                "run_id": account.run_id,
                "attempt": account.attempt,
                "status": account.status,
                "completion_mode": account.completion.mode,
                "iterations_used": account.completion.iterations_used,
                "iterations_to_submit": account.completion.iterations_to_submit,
                "inspected": len(account.completion.inspected_ranges),
                "failure_kind": account.failure_kind,
                "errors": list(account.validation_errors[:3]),
                "calls": self._state.llm_calls,
                "cost_usd": round(self._state.cost_usd, 6),
            },
        )

    def finish(
        self,
        *,
        status: str,
        stop_reason: str,
        run_id: str = "",
        completed_passes: int | None = None,
    ) -> None:
        """Close the session with the run's own final counts.

        ``completed_passes`` is passed in rather than read from the last
        checkpoint: a pass increments only after its final chunk, so the state
        written during that chunk is always one behind, and a session closed on
        it would report fewer passes than the run beside it.
        """
        self._state = self._state.replace(
            status=status,
            completed_passes=(
                self._state.completed_passes if completed_passes is None else completed_passes
            ),
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "session_finished",
                "status": status,
                "stop_reason": stop_reason,
                "run_id": run_id,
                "completed_passes": self._state.completed_passes,
                "contracts": len(self._state.contracts),
            },
        )

    def fail(self, error: str, *, status: str = "failed") -> None:
        """Record that the run died or was interrupted, so it is not read as running.

        Whatever was checkpointed stays: accepted accounts and applied chunks
        are resumable; the in-flight invocation is not counted.
        """
        self._state = self._state.replace(
            status=status,
            last_error=error,
            updated_at=datetime.now(UTC).isoformat(),
            budget_usage=self._guard.summary() if self._guard is not None else {},
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "session_interrupted" if status == "interrupted" else "session_failed",
                "error": error,
            },
        )
