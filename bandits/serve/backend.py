"""What the server serves: one episode per rollout, over a real env session.

The HTTP layer (:mod:`bandits.serve.server`) and the MCP layer
(:mod:`bandits.serve.mcp`) both drive an :class:`EnvBackend`. Keeping that
seam explicit means the wire protocol never reaches into
:class:`~bandits.env.session.BanditsSession` directly, and a different
episode source -- ``bandits.rl.BanditsEnv``, a task suite, a remote pool -- can
be plugged in without touching either transport.

Two guarantees this module owns:

**One session per episode.** :meth:`SessionBackend.new_episode` calls
``build_session`` afresh every time. Sessions share nothing: each materializes
its own SQLite store (in-memory by default) and its own effect ledger. There is
no module-level session, no cache keyed by task, no copy-on-write. Two episodes
of the same task are two different worlds.

**Every state change goes through ``execute``.** The episode calls
``session.execute`` and nothing else. It never opens the store, never mutates a
snapshot. That is the boundary rule from :mod:`bandits.env.interface`, and the
anticheat in stage 5 depends on it holding here too -- a served environment
that reached around ``execute`` would let a rollout produce the final state a
verifier asserts on without performing the actions.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from bandits.contracts import (
    CallStatus,
    JsonObject,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolSurface,
    Verifier,
)

from .protocol import EnvSpec, Observation, RewardRange, ToolSchema

__all__ = [
    "DEFAULT_MAX_STEPS",
    "Episode",
    "EnvBackend",
    "SessionBackend",
    "StepOutcome",
    "UnsupportedToolCall",
]

DEFAULT_MAX_STEPS = 32
"""Steps per episode when the caller does not say. Deliberately small: mined
tasks are single-goal and a rollout that has taken 32 actions is lost, not
thinking."""

_JSON_TYPES = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
    "null": "null",
}


class UnsupportedToolCall(Exception):
    """A tool the environment refuses to pretend it can run.

    Wraps ``bandits.env.interface.UnsupportedToolError`` so neither transport
    has to import the env package's exception hierarchy. It is deliberately
    *not* an error observation: a faked success here would silently corrupt
    every reward computed after it.
    """

    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason



class _SessionWorker:
    """Owns one session and the single thread allowed to touch it.

    SQLite connections are bound to the thread that created them, and an HTTP
    server hands consecutive requests for the same episode to *different*
    threads. So the session is built on, and only ever used from, a private
    worker thread; every caller submits a callable and waits for the result.

    That is not just a workaround for sqlite3's thread affinity -- it is also
    the strongest form of the isolation guarantee this layer sells. One session,
    one thread, one queue: actions against an episode are applied in a total
    order, and no two episodes can interleave inside one store because no two
    episodes share a thread.
    """

    def __init__(self, name: str) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f"bandits-episode-{name}", daemon=True)
        self._stopped = False
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, box, ready = item
            try:
                box.append(("ok", fn()))
            except BaseException as exc:  # returned to the caller, never swallowed
                box.append(("err", exc))
            finally:
                ready.set()

    def submit(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the owning thread and return its result here."""
        if self._stopped:
            raise RuntimeError("session worker is stopped")
        box: list[tuple[str, Any]] = []
        ready = threading.Event()
        self._queue.put((fn, box, ready))
        ready.wait()
        kind, value = box[0]
        if kind == "err":
            raise value
        return value

    def stop(self) -> None:
        """Retire the thread. Idempotent; safe after the session is closed."""
        if self._stopped:
            return
        self._stopped = True
        self._queue.put(None)
        self._thread.join(timeout=10)


@dataclass(frozen=True)
class StepOutcome:
    """Result of one action inside an episode."""

    observation: Observation
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    step: int = 0
    info: JsonObject = field(default_factory=dict)


@runtime_checkable
class EnvBackend(Protocol):
    """The episode source a transport talks to.

    Implementations must be safe to call from several threads at once, and must
    hand back an :class:`Episode` that shares no mutable state with any other
    live episode.
    """

    def task_ids(self) -> tuple[str, ...]:
        """Every task this backend can instantiate, in stable order."""
        ...

    def spec(self, task_id: str | None = None) -> EnvSpec:
        """Describe the environment for ``task_id`` (default task when None)."""
        ...

    def new_episode(
        self, episode_id: str, task_id: str | None = None, seed: int | None = None
    ) -> Episode:
        """Materialize a fresh, isolated environment. Raises ``KeyError`` for an
        unknown ``task_id``."""
        ...


class Episode:
    """One live environment bound to one rollout.

    Thread-safe by construction: every public method takes ``self._lock``, so
    two threads that (wrongly, but harmlessly) share an ``episode_id`` still see
    a serialized sequence of actions rather than a half-applied write. The lock
    is per-episode, never global -- parallel rollouts must not queue behind each
    other.
    """

    def __init__(
        self,
        episode_id: str,
        session: Any,
        *,
        task: TaskCase,
        tools: Sequence[ToolSchema],
        max_steps: int = DEFAULT_MAX_STEPS,
        verifier: Verifier | None = None,
        allow_unreviewed_verifier: bool = True,
        seed: int | None = None,
        worker: _SessionWorker | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.task = task
        self.task_id = task.task_id
        self.tools = tuple(tools)
        self.max_steps = max_steps
        self.seed = seed
        self._session = session
        self._worker = worker
        self._verifier = verifier
        self._allow_unreviewed = allow_unreviewed_verifier
        self._lock = threading.RLock()
        self._steps = 0
        self._closed = False
        self._done = False
        self._truncated = False
        self._final_digest: str | None = None

    # -- introspection -----------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def finished(self) -> bool:
        """True once the episode terminated or was truncated. Steps are refused."""
        return self._done or self._truncated

    @property
    def final_digest(self) -> str | None:
        return self._final_digest

    def opening_observation(self) -> Observation:
        """The observation handed back by ``/reset``.

        There has been no action, so this carries the framing a policy needs:
        the instruction and the names of the tools it may call.
        """
        payload = {
            "instruction": self.task.instruction,
            "task_id": self.task_id,
            "tools": [t.name for t in self.tools],
        }
        return Observation(response=payload, status=CallStatus.OK, text=_as_text(payload))

    def snapshot(self) -> dict[str, list[dict]]:
        with self._lock:
            self._require_open()
            return self._run(self._session.snapshot)

    def digest(self) -> str:
        with self._lock:
            self._require_open()
            return self._run(self._session.digest)

    def effects_count(self) -> int:
        with self._lock:
            self._require_open()
            return self._run(lambda: len(self._session.effects()))

    def _run(self, fn: Callable[[], Any]) -> Any:
        """Execute ``fn`` on the session's owning thread.

        Without a worker (in-process use from a single thread) it just runs
        here; the served path always has one.
        """
        if self._worker is None:
            return fn()
        return self._worker.submit(fn)

    # -- the one mutating path --------------------------------------------

    def step(self, name: str, arguments: Mapping[str, Any] | None = None) -> StepOutcome:
        """Run one action through ``session.execute`` and grade the result."""
        with self._lock:
            self._require_open()
            if self.finished:
                raise EpisodeFinished(self.episode_id, done=self._done)

            # One submission for the whole body: execute, grade and digest are
            # a single atomic unit on the owning thread, so no other caller can
            # observe a half-applied step.
            obs, reward, passed, assertions, digest, effects = self._run(
                lambda: self._apply(name, dict(arguments or {}))
            )
            self._steps += 1
            self._done = passed
            self._truncated = not passed and self._steps >= self.max_steps

            info: JsonObject = {
                "task_id": self.task_id,
                "seed": self.seed,
                "env_digest": digest,
                "effects": effects,
                "steps_remaining": max(0, self.max_steps - self._steps),
            }
            if assertions is not None:
                info["assertions"] = assertions
            if self._truncated:
                info["truncation_reason"] = "max_steps"

            return StepOutcome(
                observation=obs,
                reward=reward,
                done=self._done,
                truncated=self._truncated,
                step=self._steps,
                info=info,
            )

    def _apply(
        self, name: str, arguments: JsonObject
    ) -> tuple[Observation, float, bool, list[JsonObject] | None, str, int]:
        """The whole of one step, on the session's own thread."""
        obs = self._execute(name, arguments)
        reward, passed, assertions = self._grade()
        return obs, reward, passed, assertions, self._session.digest(), len(self._session.effects())

    def _execute(self, name: str, arguments: JsonObject) -> Observation:
        from bandits.env.interface import UnsupportedToolError

        try:
            raw = self._session.execute(name, arguments)
        except UnsupportedToolError as exc:
            # Never downgraded to an error observation. See UnsupportedToolCall.
            raise UnsupportedToolCall(exc.tool, exc.reason) from exc
        return Observation(
            response=raw.response,
            status=raw.status,
            error_kind=raw.error_kind,
            text=_as_text(raw.response),
        )

    def _grade(self) -> tuple[float, bool, list[JsonObject] | None]:
        """Score the current state. Pure: assertions over state + effects."""
        if self._verifier is None or not self._verifier.assertions:
            # No reward code for this task. Flat 0.0, and EnvSpec.verifier_digest
            # is None so the trainer can see why rather than infer it.
            return 0.0, False, None
        from bandits.verify.run import evaluate

        result = evaluate(
            self._verifier,
            self._session.snapshot(),
            self._session.effects(),
            allow_unreviewed=self._allow_unreviewed,
        )
        assertions = [
            {
                "kind": r.assertion.kind.value,
                "description": r.assertion.description,
                "passed": r.passed,
            }
            for r in result.results
        ]
        return float(result.reward), bool(result.passed), assertions

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Deterministic teardown. Idempotent, and safe to call concurrently.

        The state digest is captured *before* the store connection drops, so a
        closed episode can still be audited.
        """
        with self._lock:
            if self._closed:
                return
            try:
                self._final_digest = self._run(self._teardown)
            except Exception:  # pragma: no cover - store already unusable
                self._final_digest = None
            finally:
                self._closed = True
                if self._worker is not None:
                    self._worker.stop()
                    self._worker = None

    def _teardown(self) -> str | None:
        """Capture the final digest, then drop the store. Owning thread only."""
        try:
            return self._session.digest()
        finally:
            self._session.close()

    def _require_open(self) -> None:
        if self._closed:
            raise EpisodeClosed(self.episode_id)


class EpisodeClosed(Exception):
    """Raised when a closed episode is used again."""

    def __init__(self, episode_id: str) -> None:
        super().__init__(f"episode {episode_id!r} is closed")
        self.episode_id = episode_id


class EpisodeFinished(Exception):
    """Raised when a step is sent to an episode that already terminated."""

    def __init__(self, episode_id: str, *, done: bool) -> None:
        state = "done" if done else "truncated"
        super().__init__(f"episode {episode_id!r} is already {state}; reset for a new one")
        self.episode_id = episode_id
        self.done = done


# --------------------------------------------------------------------------
# the concrete backend
# --------------------------------------------------------------------------


class SessionBackend:
    """Serves environments built by :func:`bandits.env.session.build_session`.

    Parameters
    ----------
    schema:
        The reconstructed state schema (stage 3).
    tasks:
        One or more mined tasks (stage 4). The first is the default for a
        ``/reset`` that names none.
    tool_classes:
        ``tool -> ToolClass`` from stage 2.
    surface:
        The stage-2 tool surface. Strongly recommended: it is where the declared
        JSON Schemas come from, and it is what lets a READ tool return the field
        set it was observed to return instead of the whole table.
    verifiers:
        ``task_id -> Verifier``. A task with no verifier is servable but always
        scores 0.0 -- :attr:`EnvSpec.verifier_digest` is ``None`` so a trainer
        can detect that rather than silently train on a flat reward.
    max_steps:
        Per-episode action cap.
    session_kwargs:
        Passed through to ``build_session`` (e.g. ``rules``, ``external_stubs``).
    """

    def __init__(
        self,
        schema: StateSchema,
        tasks: Iterable[TaskCase],
        tool_classes: Mapping[str, ToolClass],
        *,
        surface: ToolSurface | None = None,
        verifiers: Mapping[str, Verifier] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        allow_unreviewed_verifiers: bool = True,
        **session_kwargs: Any,
    ) -> None:
        self.schema = schema
        self._tasks: dict[str, TaskCase] = {t.task_id: t for t in tasks}
        if not self._tasks:
            raise ValueError("SessionBackend needs at least one task to serve")
        self.tool_classes = dict(tool_classes)
        self.surface = surface
        self.verifiers = dict(verifiers or {})
        self.max_steps = max_steps
        self.allow_unreviewed_verifiers = allow_unreviewed_verifiers
        self._session_kwargs = session_kwargs
        self._default_task_id = next(iter(self._tasks))
        self._spec_cache: dict[str, EnvSpec] = {}
        self._spec_lock = threading.Lock()

    # -- description -------------------------------------------------------

    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def _task(self, task_id: str | None) -> TaskCase:
        key = task_id or self._default_task_id
        if key not in self._tasks:
            raise KeyError(key)
        return self._tasks[key]

    def spec(self, task_id: str | None = None) -> EnvSpec:
        """Describe one task's environment.

        Built by materializing a throwaway session, reading its manifest, and
        closing it. That costs one in-memory SQLite build per task, once, and
        buys a spec that reports what the environment *actually* refused to
        reimplement rather than what we hoped it would.
        """
        task = self._task(task_id)
        with self._spec_lock:
            cached = self._spec_cache.get(task.task_id)
            if cached is not None:
                return cached
            spec = self._build_spec(task)
            self._spec_cache[task.task_id] = spec
            return spec

    def _build_spec(self, task: TaskCase) -> EnvSpec:
        session = self._build_session(task)
        try:
            manifest = session.manifest()
            reasons = {
                name: session.unsupported_reason(name) or "not reimplementable"
                for name in manifest.unsupported_tools
            }
        finally:
            session.close()

        tools = self._tool_schemas(unsupported=reasons)
        verifier = self.verifiers.get(task.task_id)
        return EnvSpec(
            env_id=manifest.env_id,
            task_id=task.task_id,
            task_ids=self.task_ids(),
            instruction=task.instruction,
            tools=tools,
            max_steps=self.max_steps,
            reward_range=RewardRange(low=0.0, high=1.0),
            env_digest=manifest.schema_digest,
            verifier_digest=_verifier_digest(verifier),
            static_entities=manifest.static_entities,
            unsupported_tools=manifest.unsupported_tools,
        )

    def _tool_schemas(self, *, unsupported: Mapping[str, str]) -> tuple[ToolSchema, ...]:
        """The action space, declared schema first and observed shape second."""
        names = sorted(set(self.tool_classes) | set(unsupported))
        out: list[ToolSchema] = []
        for name in names:
            profile = self.surface.by_name(name) if self.surface is not None else None
            schema = None
            description = ""
            if profile is not None:
                schema = profile.declared_schema
                if schema is None:
                    schema = _schema_from_observed(profile)
            out.append(
                ToolSchema(
                    name=name,
                    description=description,
                    input_schema=dict(schema or {"type": "object", "properties": {}}),
                    tool_class=self.tool_classes.get(name, ToolClass.UNKNOWN),
                    supported=name not in unsupported,
                    unsupported_reason=unsupported.get(name),
                )
            )
        return tuple(out)

    # -- episodes ----------------------------------------------------------

    def _build_session(self, task: TaskCase) -> Any:
        from bandits.env.session import build_session

        return build_session(
            self.schema,
            task,
            self.tool_classes,
            surface=self.surface,
            **self._session_kwargs,
        )

    def new_episode(
        self, episode_id: str, task_id: str | None = None, seed: int | None = None
    ) -> Episode:
        """A brand-new world. Nothing is shared with any other live episode."""
        task = self._task(task_id)
        spec = self.spec(task.task_id)
        worker = _SessionWorker(episode_id)
        try:
            # Built *on* the worker thread: the SQLite connection must be
            # created by the thread that will use it.
            session = worker.submit(lambda: self._build_session(task))
        except BaseException:
            worker.stop()
            raise
        return Episode(
            episode_id,
            session,
            task=task,
            tools=spec.tools,
            max_steps=self.max_steps,
            verifier=self.verifiers.get(task.task_id),
            allow_unreviewed_verifier=self.allow_unreviewed_verifiers,
            seed=seed,
            worker=worker,
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _as_text(value: Any) -> str:
    """Flat JSON rendering of an observation, for text-in/text-out policies."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _verifier_digest(verifier: Verifier | None) -> str | None:
    """sha256 over the verifier's assertions. Two runs reporting the same digest
    were graded by the same reward code."""
    if verifier is None:
        return None
    payload = json.dumps(
        [a.model_dump(mode="json") for a in verifier.assertions], sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _schema_from_observed(profile: Any) -> JsonObject:
    """Synthesize a JSON Schema from the argument fields seen in traces.

    Used only when the corpus had no declared registry. Narrower than the real
    schema by construction -- it lists the parameters production actually sent,
    which is usually a handful of the declared dozens -- and that is the honest
    thing to advertise, since anything else was never exercised.
    """
    properties: JsonObject = {}
    required: list[str] = []
    for fp in profile.argument_fields:
        types = [_JSON_TYPES[t] for t in fp.json_types if t in _JSON_TYPES]
        prop: JsonObject = {}
        if len(types) == 1:
            prop["type"] = types[0]
        elif types:
            prop["type"] = types
        if fp.sample_values:
            prop["examples"] = list(fp.sample_values[:3])
        properties[fp.name] = prop
        if fp.occurrences >= profile.call_count > 0 and fp.null_count == 0:
            required.append(fp.name)
    schema: JsonObject = {"type": "object", "properties": properties}
    if required:
        schema["required"] = sorted(required)
    return schema
