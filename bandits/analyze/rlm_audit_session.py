"""A resumable audit session, written to disk as it goes.

Mirrors mining's own :mod:`rlm_session` for the same reason: an audit over
forty-odd contracts makes one model call per contract and can die at any of
them, and until this existed a kill at contract seventeen of forty-four lost
every finding the session had already paid for.

Kept as its own module and its own on-disk shape rather than folded into
mining's ``SessionState``. Mining tracks passes, chunks and a taxonomy under
construction; an audit walks a fixed, already-frozen contract list once. A
single shared schema would carry passes/chunks fields that mean nothing for an
audit and a contract_order/findings shape that means nothing for mining, and a
resume would have no way to tell which arm's invariants apply to a given file.

Session files for the two kinds are told apart by a ``kind`` sub-key on disk
(see :func:`bandits.analyze.rlm_session.SessionStore.list`-equivalent below),
not by directory, so ``rlm-session`` can list and watch both from one root.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bandits.analyze.rlm_models import AuditFinding, TraceView
from bandits.traces import Contract


class AuditSessionState(Contract):
    schema_version: int = 1
    session_id: str
    run_id: str
    model: str
    view: TraceView
    prompt_digest: str

    status: str = "running"
    """One of running, awaiting_review, interrupted, failed."""

    contract_order: tuple[str, ...] = ()
    findings: tuple[AuditFinding, ...] = ()
    completed_contract_ids: tuple[str, ...] = ()

    llm_calls: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    started_at: str = ""
    updated_at: str = ""
    resumed_from: str | None = None
    stop_reason: str = ""
    last_error: str = ""

    @property
    def progress(self) -> str:
        done = len(self.completed_contract_ids)
        total = len(self.contract_order)
        latest = self.findings[-1] if self.findings else None
        verdict = f" · latest: {latest.contract_id}={latest.recommendation}" if latest else ""
        failures = sum(1 for f in self.findings if "failed and reached no verdict" in f.rationale)
        failed = f" · {failures} failed" if failures else ""
        return (
            f"contract {done}/{total}{verdict}{failed} · "
            f"{self.llm_calls} calls · ${self.cost_usd:.4f}"
        )


class AuditSessionStore:
    """Mutable audit session state, one directory per session.

    Lives beside mining's sessions under ``<project>/.bandits/sessions``, in
    its own subdirectory (``audits/``) so a listing that reads both kinds
    never has to guess a file's shape before parsing it.
    """

    def __init__(self, project_dir: Path | str = Path(".bandits")) -> None:
        self._root = Path(project_dir) / "sessions" / "audits"

    def _dir(self, session_id: str) -> Path:
        return self._root / session_id

    def path(self, session_id: str) -> Path:
        return self._dir(session_id) / "session.json"

    def write(self, state: AuditSessionState) -> None:
        directory = self._dir(state.session_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json(indent=2).encode("utf-8")
        target = directory / "session.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, target)

    def read(self, session_id: str) -> AuditSessionState:
        return AuditSessionState.model_validate_json(self.path(session_id).read_bytes())

    def exists(self, session_id: str) -> bool:
        return self.path(session_id).is_file()

    def list(self) -> list[AuditSessionState]:
        if not self._root.exists():
            return []
        states = []
        for entry in sorted(self._root.iterdir()):
            if (entry / "session.json").is_file():
                try:
                    states.append(self.read(entry.name))
                except ValueError:
                    continue
        return sorted(states, key=lambda s: s.updated_at, reverse=True)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
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


def new_audit_session_id(run_id: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"rlm-audit-{run_id[-8:]}-{stamp}"


class AuditSessionRecorder:
    """Writes an audit session's state after every contract.

    Passed into :func:`~bandits.analyze.rlm_audit.audit_clustering` so the loop
    stays ignorant of storage: it calls ``checkpoint`` after each finding and
    this decides what that means on disk. A run given no recorder behaves
    exactly as before, which keeps the tests free of a filesystem.
    """

    def __init__(
        self,
        store: AuditSessionStore,
        *,
        session_id: str,
        run_id: str,
        model: str,
        view: TraceView,
        prompt_digest: str,
        resumed_from: str | None = None,
        seed_state: AuditSessionState | None = None,
    ) -> None:
        self._store = store
        now = datetime.now(UTC).isoformat()
        if seed_state is not None:
            self._state = seed_state.replace(
                session_id=session_id,
                resumed_from=resumed_from,
                status="running",
                updated_at=now,
            )
        else:
            self._state = AuditSessionState(
                session_id=session_id,
                run_id=run_id,
                model=model,
                view=view,
                prompt_digest=prompt_digest,
                started_at=now,
                updated_at=now,
                resumed_from=resumed_from,
            )

    @property
    def session_id(self) -> str:
        return self._state.session_id

    @property
    def state(self) -> AuditSessionState:
        return self._state

    def begin(self, *, contract_order: tuple[str, ...]) -> None:
        self._state = self._state.replace(
            contract_order=contract_order,
            status="running",
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "session_started",
                "run_id": self._state.run_id,
                "contracts": len(contract_order),
                "resumed_from": self._state.resumed_from,
            },
        )

    def checkpoint(
        self,
        finding: AuditFinding,
        *,
        calls: int,
        usd: float,
        elapsed: float,
    ) -> None:
        """Persist one completed finding, atomically, immediately.

        The whole findings tuple is rewritten each time rather than an
        appended delta, for the same reason mining's session does this: a
        resume that replayed deltas could reconstruct a state the run never
        actually held, and the difference would be invisible.
        """
        findings = (*self._state.findings, finding)
        completed = (*self._state.completed_contract_ids, finding.contract_id)
        self._state = self._state.replace(
            findings=findings,
            completed_contract_ids=completed,
            llm_calls=calls,
            cost_usd=usd,
            elapsed_seconds=elapsed,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "contract_audited",
                "contract_id": finding.contract_id,
                "recommendation": finding.recommendation,
                "done": len(completed),
                "of": len(self._state.contract_order),
                "calls": calls,
                "cost_usd": round(usd, 6),
            },
        )

    def finish(self, *, status: str, stop_reason: str) -> None:
        self._state = self._state.replace(
            status=status,
            stop_reason=stop_reason,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._store.write(self._state)
        self._store.append_event(
            self.session_id,
            {
                "event": "session_finished",
                "status": status,
                "stop_reason": stop_reason,
                "findings": len(self._state.findings),
            },
        )

    def interrupt(self) -> None:
        """Mark the session interrupted rather than leaving it 'running'.

        Called from a SIGINT handler: whatever findings ``checkpoint`` already
        wrote stay on disk exactly as they were, this only changes ``status``
        so a later look does not mistake a killed session for one still alive.
        """
        self.finish(status="interrupted", stop_reason="interrupted")

    def fail(self, error: str) -> None:
        self._state = self._state.replace(
            status="failed", last_error=error, updated_at=datetime.now(UTC).isoformat()
        )
        self._store.write(self._state)
        self._store.append_event(self.session_id, {"event": "session_failed", "error": error})
