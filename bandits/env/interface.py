"""The environment session contract.

This file is the published boundary between a materialized environment and
everything that drives one: stage 6 (fidelity replay), the RL rollout loop, and
stage 5's verifiers. It is deliberately tiny. Four methods plus a lifecycle.

    with session:
        obs = session.execute("get_order", {"order_id": 7741})
        rows = session.snapshot()
        fx = session.effects()
        man = session.manifest()

The boundary rule
-----------------
**Every state change must go through :meth:`EnvSession.execute`.**

``snapshot()`` returns a *copy* of the store, never a live handle, and there is
no public setter. A rollout that reaches around ``execute()`` -- opening the
SQLite file directly, mutating the returned snapshot dicts and expecting it to
stick, or appending to the ledger by hand -- is **reward hacking**: it produces
the final state a verifier asserts on without performing the actions that were
supposed to produce it. Stage 5's anti-cheat depends on exactly this boundary,
and the only way it can depend on it is if the boundary is real. So:

* the store connection is private to the session,
* the ledger is append-only through the session and frozen at close,
* ``effects()`` and ``snapshot()`` hand back immutable / detached data.

Failure discipline
------------------
A tool the environment cannot faithfully reimplement raises
:class:`UnsupportedToolError`. It never returns a success-shaped
:class:`~bandits.contracts.Observation`. Faking success here would silently
corrupt every downstream reward: the agent would be trained to call a tool that
does nothing and be paid for it. An *observed* failure (a missing row, a
conflicting write) is different -- that is real environment dynamics and comes
back as ``Observation(status=ERROR, error_kind=...)``.
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, runtime_checkable

from bandits.contracts import Effect, EnvManifest, JsonObject, Observation


class EnvError(Exception):
    """Base for every environment runtime failure."""


class UnsupportedToolError(EnvError):
    """Raised for a tool the environment refuses to pretend it can run.

    Carries the reason so the manifest and any operator-facing report can say
    *why* the tool is unsupported rather than just that it is.
    """

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason


class ReadOnlyEntityError(UnsupportedToolError):
    """Raised when something tries to write an entity materialized as a static snapshot.

    A static entity's structure was never determined (PLAN.md Step 7, "Where
    this fails"), so a write to it cannot be modeled. We refuse rather than
    invent.
    """


class SessionClosedError(EnvError):
    """Raised when a closed session is used again. Teardown is final."""


class StoreError(EnvError):
    """Raised when the store cannot be materialized or seeded as specified."""


@runtime_checkable
class EnvSession(Protocol):
    """A live, materialized environment for exactly one :class:`TaskCase`.

    Implementations must be deterministic: same schema + same task + same
    action sequence => same ``snapshot()``, same ``digest``, same ``effects()``.
    No clocks, no randomness, no network.
    """

    def execute(self, tool: str, arguments: JsonObject) -> Observation:
        """Run one action. The only way to change the world.

        Raises :class:`UnsupportedToolError` for a tool that cannot be
        reimplemented. Returns an ERROR observation for a failure the real tool
        would also have returned.
        """
        ...

    def snapshot(self) -> dict[str, list[dict]]:
        """Full current state as ``entity -> rows``. A detached copy."""
        ...

    def digest(self) -> str:
        """Order-independent sha256 of ``snapshot()``. Exact state comparison."""
        ...

    def effects(self) -> tuple[Effect, ...]:
        """Every external side effect attempted, in order, none performed."""
        ...

    def manifest(self) -> EnvManifest:
        """Identity and honest limitations of this environment."""
        ...

    def open(self) -> EnvSession: ...

    def close(self) -> None:
        """Deterministic teardown. Idempotent."""
        ...

    def __enter__(self) -> EnvSession: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class BaseEnvSession:
    """Concrete lifecycle shared by session implementations.

    Subclasses implement ``_open``/``_close`` and the four data methods. This
    base only guarantees that open/close are idempotent, that use-after-close
    raises, and that ``with`` always tears down.
    """

    def __init__(self) -> None:
        self._opened = False
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> BaseEnvSession:
        if self._closed:
            raise SessionClosedError("session already closed; build a new one")
        if not self._opened:
            self._open()
            self._opened = True
        return self

    def close(self) -> None:
        if self._closed:
            return
        if self._opened:
            self._close()
        self._closed = True

    def _open(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _close(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _require_live(self) -> None:
        if self._closed:
            raise SessionClosedError("session is closed")
        if not self._opened:
            self.open()

    @property
    def is_open(self) -> bool:
        return self._opened and not self._closed

    def __enter__(self) -> BaseEnvSession:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "BaseEnvSession",
    "EnvError",
    "EnvSession",
    "ReadOnlyEntityError",
    "SessionClosedError",
    "StoreError",
    "UnsupportedToolError",
]
