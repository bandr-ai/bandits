"""A resumable mining session, written to disk as it goes.

Why this is not just the draft artifact. A draft is content-addressed and
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
the immutable ``TaxonomyDraft`` is written from it in the ordinary way, and that
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
    """One of running, awaiting_review, failed. Written before anything else so
    a crashed run is distinguishable from one still working."""

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

    @property
    def progress(self) -> str:
        """One line a person can read mid-run without parsing anything."""
        return (
            f"pass {self.pass_index + 1}/{self.requested_passes} · "
            f"{self.traces_seen_this_pass}/{self.traces_total} traces · "
            f"chunk {self.chunk_index} · {len(self.contracts)} contracts · "
            f"{self.llm_calls} calls · ${self.cost_usd:.4f}"
        )


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
    ) -> None:
        self._store = store
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
            contracts=tuple(
                sorted(state.contracts.values(), key=lambda c: c.contract_id)
            ),
            assignments=dict(state.assignments),
            ambiguous_trace_ids=tuple(sorted(state.ambiguous)),
            uncovered_trace_ids=tuple(sorted(state.uncovered)),
            seen_this_pass=tuple(sorted(seen_this_pass)),
            pass_order=pass_order,
            passes=tuple(passes),
            chunks=tuple(chunks),
            llm_calls=calls,
            cost_usd=usd,
            elapsed_seconds=elapsed,
            updated_at=datetime.now(UTC).isoformat(),
            last_error=chunk.error if chunk.status == "error" else "",
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
            },
        )

    def finish(
        self,
        *,
        status: str,
        stop_reason: str,
        draft_id: str = "",
        completed_passes: int | None = None,
    ) -> None:
        """Close the session with the run's own final counts.

        ``completed_passes`` is passed in rather than read from the last
        checkpoint: a pass increments only after its final chunk, so the state
        written during that chunk is always one behind, and a session closed on
        it would report fewer passes than the draft beside it.
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
                "draft_id": draft_id,
                "completed_passes": self._state.completed_passes,
                "contracts": len(self._state.contracts),
            },
        )

    def fail(self, error: str) -> None:
        """Record that the run died, so a crashed session is not read as idle."""
        self._state = self._state.replace(
            status="failed", last_error=error, updated_at=datetime.now(UTC).isoformat()
        )
        self._store.write(self._state)
        self._store.append_event(self.session_id, {"event": "session_failed", "error": error})
