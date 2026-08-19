"""``TraceEnv`` -- the gym-style RL environment over a rebuilt world.

This is the layer that turns the deterministic reconstruction pipeline into
something a trainer can actually roll out against::

    env = TraceEnv(schema=schema, task=task, verifier=verifier,
                   tool_classes=classes, surface=surface)
    obs = env.reset(seed=0)
    while True:
        step = env.step({"name": "get_order", "arguments": {"order_id": 7741}})
        ...
    final = env.step({"name": "respond", "arguments": {"message": "Refunded."}})
    final.reward   # 0.0 or 1.0, computed by the verifier, not by an opinion

Reward semantics
----------------
**Every non-terminal step returns exactly 0.0.** There is no shaping, no partial
credit for "getting closer", no bonus for calling the right tool. The whole
thesis of bandits is that reward is a state assertion rather than an opinion
(README, "The bet"), and a shaped intermediate reward is an opinion wearing a
number's clothes: somebody decided that reading the order before refunding it is
worth 0.1, and the policy will optimize that decision, not the task.

At the terminal step the reward is whatever
:func:`bandits.verify.run.evaluate` returns for the final store state and the
effect ledger -- 1.0 or 0.0 under the default all-or-nothing mode -- and then
:func:`bandits.verify.anticheat.enforce` is applied, so a rollout that tripped
a guard scores 0.0 no matter what the assertions said.

Truncation scores 0.0. An episode that ran out of steps did not finish the task;
paying it for the part it did is exactly the shaping this module refuses.

If a dense signal is genuinely needed for a research run, pass
``expose_progress=True`` and read ``info["progress"]``. It is deliberately
outside ``reward`` and it **must never be summed into the training reward**: the
"unscored" verification it carries is computed in ``partial`` mode, where
STATE_UNCHANGED assertions pass for free when the agent does nothing, so
optimizing it teaches inaction.

Failure handling
----------------
Three kinds of bad step, three deliberately different treatments:

``UnsupportedToolError``
    Comes back as an ERROR **observation** the agent can react to, and the
    attempt is counted in ``info["unsupported_tool_attempts"]``. Crashing the
    episode would make the environment unusable for RL -- one exploratory call
    to a tool we could not reimplement would kill an entire rollout and poison
    the batch. Returning a fake success would be worse: it would corrupt every
    reward computed afterwards, which is the exact failure
    ``bandits/env/interface.py`` refuses under "failure discipline". So we do
    the third thing: tell the truth to the agent, and keep the receipt for the
    trainer.

An *observed* failure (missing row, already-refunded)
    Ordinary environment dynamics. It comes back as an ERROR observation and is
    not counted as anything special. The agent is supposed to learn these.

An unrecoverable environment error (store failure, closed session, anything
unexpected out of the runtime)
    Ends the episode with ``done=True``, ``truncated=False``, reward 0.0 and
    ``info["env_error"]``. We never grade a world that broke halfway through.

The boundary rule
-----------------
``TraceEnv`` drives the session through :meth:`EnvSession.execute` and nothing
else. It holds no handle it hands out, mutates no snapshot, and builds the
``RolloutRecord`` from the actions it actually dispatched -- which is what makes
guard 1 in :mod:`bandits.verify.anticheat` able to notice a rollout that
reached around the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from bandits.contracts import (
    CallStatus,
    JsonObject,
    Observation,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolSurface,
    Verifier,
)
from bandits.env import EnvError, UnsupportedToolError, build_session
from bandits.env.session import BanditsSession
from bandits.verify import (
    AntiCheatReport,
    RolloutAction,
    RolloutRecord,
    check_rollout,
    enforce,
    evaluate,
)
from bandits.verify.run import UnreviewedVerifierError

from .spec import EnvSpec

__all__ = ["DEFAULT_FINISH_TOOL", "DEFAULT_MAX_STEPS", "StepResult", "TraceEnv"]

DEFAULT_MAX_STEPS = 30
DEFAULT_FINISH_TOOL = "respond"

#: Accepted spellings for the two fields of an action, so a trainer's own
#: serialization does not have to be rewritten to drive this env. Anything else
#: is a malformed action and gets an error observation, never a guess.
_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("arguments", "args", "parameters", "input", "tool_input")


@dataclass(frozen=True)
class StepResult:
    """One environment transition. Unpacks as the gym 5-tuple."""

    observation: JsonObject
    reward: float
    done: bool
    truncated: bool
    info: JsonObject = field(default_factory=dict)

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done
        yield self.truncated
        yield self.info

    @property
    def terminated(self) -> bool:
        """Gymnasium's name for ``done and not truncated``."""
        return self.done and not self.truncated


class EpisodeNotStartedError(RuntimeError):
    """Raised when :meth:`TraceEnv.step` is called before :meth:`TraceEnv.reset`."""


class TraceEnv:
    """A gym-style environment for exactly one :class:`TaskCase`.

    Parameters
    ----------
    schema, task, tool_classes, surface:
        The pipeline artifacts. Passed straight through to
        :func:`bandits.env.session.build_session`.
    verifier:
        The reward function. Its ``reviewed_by`` must be set unless
        ``allow_unreviewed=True`` -- and that check happens *here*, at
        construction, so an unreviewed verifier fails before a rollout burns
        compute rather than after.
    max_steps:
        Step budget. Reaching it truncates with reward 0.0.
    reward_mode:
        Passed to :func:`bandits.verify.run.evaluate`. Leave it
        ``"all_or_nothing"``; ``"partial"`` exists for verifier debugging and
        produces a reward that is not a pass/fail signal.
    network_calls:
        What the sandbox observed. ``None`` (the default) means "not reported",
        which raises a *warning* finding rather than passing the guard silently.
        Pass ``()`` only if something really did watch the process.
    expose_progress:
        Adds ``info["progress"]``. Diagnostics only -- see the module docstring.
    """

    def __init__(
        self,
        *,
        schema: StateSchema,
        task: TaskCase,
        verifier: Verifier,
        tool_classes: Mapping[str, ToolClass] | None = None,
        surface: ToolSurface | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        reward_mode: str = "all_or_nothing",
        allow_unreviewed: bool = False,
        finish_tool: str = DEFAULT_FINISH_TOOL,
        network_calls: Sequence[str] | None = None,
        expose_progress: bool = False,
        env_id: str | None = None,
        rules: Mapping[str, Any] | None = None,
        external_stubs: Mapping[str, Any] | None = None,
    ) -> None:
        if verifier.task_id != task.task_id:
            raise ValueError(
                f"verifier {verifier.verifier_id!r} grades task {verifier.task_id!r}, "
                f"not {task.task_id!r}; grading a task with another task's reward "
                f"function is silent nonsense"
            )
        if verifier.reviewed_by is None and not allow_unreviewed:
            raise UnreviewedVerifierError(
                f"verifier {verifier.verifier_id!r} has reviewed_by=None and would grade "
                f"every episode of task {task.task_id!r}. A generated reward function "
                f"nobody has read must not drive training; have a person read it and set "
                f"reviewed_by, or pass allow_unreviewed=True for a dry run."
            )
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        if tool_classes is None:
            if surface is None:
                raise ValueError("pass tool_classes, or a surface to derive them from")
            tool_classes = {p.name: p.tool_class for p in surface.tools}

        self.schema = schema
        self.task = task
        self.verifier = verifier
        self.tool_classes = dict(tool_classes)
        self.surface = surface
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.allow_unreviewed = allow_unreviewed
        self.finish_tool = finish_tool
        self.expose_progress = expose_progress
        self._network_calls = None if network_calls is None else tuple(network_calls)
        self._env_id = env_id
        self._rules = dict(rules or {})
        self._external_stubs = dict(external_stubs or {})

        self.session: BanditsSession | None = None
        self._seed: int | None = None
        self._steps = 0
        self._done = False
        self._actions: list[RolloutAction] = []
        self._unsupported: list[JsonObject] = []
        self._pre_state: dict[str, list[JsonObject]] = {}
        self._final_message: str | None = None
        self._last_result: JsonObject | None = None

    # -- identity ----------------------------------------------------------

    @property
    def primary_keys(self) -> dict[str, str]:
        return {e.name: e.primary_key for e in self.schema.entities if e.primary_key}

    def spec(self) -> EnvSpec:
        """The :class:`EnvSpec` for this environment. Available before ``reset``."""
        session = self.session
        opened_here = False
        if session is None:
            session = self._build_session()
            opened_here = True
        try:
            manifest = session.manifest()
            return EnvSpec.build(
                task=self.task,
                verifier=self.verifier,
                schema=self.schema,
                tool_classes=self.tool_classes,
                surface=self.surface,
                unsupported_tools=manifest.unsupported_tools,
                env_id=manifest.env_id,
                schema_digest=manifest.schema_digest,
                max_steps=self.max_steps,
                reward_mode=self.reward_mode,
                finish_tool=self.finish_tool,
                notes=tuple(self.task.provenance.get("solvability_warnings") or ()),
            )
        finally:
            if opened_here:
                session.close()

    # -- lifecycle ---------------------------------------------------------

    def _build_session(self) -> BanditsSession:
        return build_session(
            self.schema,
            self.task,
            self.tool_classes,
            surface=self.surface,
            rules=self._rules or None,
            external_stubs=self._external_stubs or None,
            env_id=self._env_id,
        )

    def reset(self, seed: int | None = None) -> JsonObject:
        """Materialize a fresh world and return the first observation.

        Deterministic given a seed -- and, in fact, deterministic *without* one:
        the store is seeded from the task's reconstructed pre-state and nothing
        in the runtime samples, clocks or reaches the network. ``seed`` is
        accepted because trainers pass it, is recorded in the observation for
        provenance, and is what :class:`~bandits.rl.batch.TaskSuite` uses to
        choose *which* task an episode gets.
        """
        self.close()
        self._seed = seed
        self._steps = 0
        self._done = False
        self._actions = []
        self._unsupported = []
        self._final_message = None
        self._last_result = None
        self.session = self._build_session()
        self._pre_state = self.session.snapshot()
        spec = self.spec()
        return {
            "task_id": self.task.task_id,
            "instruction": self.task.instruction,
            "tools": [t.to_json() for t in spec.tools],
            "action_space": spec.action_space,
            "finish_tool": self.finish_tool,
            "max_steps": self.max_steps,
            "step": 0,
            "seed": seed,
            "context": {
                "env_id": spec.env_id,
                "unsupported_tools": list(spec.unsupported_tools),
                "note": (
                    "External tools are recorded, never performed. Call "
                    f"{self.finish_tool!r} with {{'message': ...}} when the task is done."
                ),
            },
        }

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def __enter__(self) -> TraceEnv:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- stepping ----------------------------------------------------------

    @staticmethod
    def _parse(action: Any) -> tuple[str | None, JsonObject, str | None]:
        """``(name, arguments, error)``. Never guesses at a malformed action."""
        if isinstance(action, str):
            return action, {}, None
        payload: Any = action
        if not isinstance(payload, Mapping):
            for attr in ("name", "tool"):
                if hasattr(payload, attr):
                    name = getattr(payload, attr)
                    args = getattr(payload, "arguments", None)
                    if args is None:
                        args = getattr(payload, "args", None)
                    return (
                        str(name),
                        dict(args) if isinstance(args, Mapping) else {},
                        None,
                    )
            return None, {}, f"action is not an object: {type(action).__name__}"
        name = next((payload[k] for k in _NAME_KEYS if payload.get(k)), None)
        if isinstance(name, Mapping):  # {"function": {"name": ..., "arguments": ...}}
            inner = name
            name = inner.get("name")
            payload = {**payload, "arguments": inner.get("arguments", {})}
        if not name:
            return None, {}, f"action has no tool name (looked for {list(_NAME_KEYS)})"
        raw_args = next((payload[k] for k in _ARG_KEYS if k in payload), {})
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, Mapping):
            return str(name), {}, f"arguments must be an object, got {type(raw_args).__name__}"
        return str(name), dict(raw_args), None

    def step(self, action: Any) -> StepResult:
        """Take one action. Reward is 0.0 unless this step ends the episode."""
        if self.session is None:
            raise EpisodeNotStartedError("call reset() before step()")
        if self._done:
            raise EpisodeNotStartedError(
                "episode is over; call reset() to start another one"
            )

        name, arguments, parse_error = self._parse(action)
        self._steps += 1

        if parse_error is not None:
            obs = self._error_obs(name or "<malformed>", "malformed_action", parse_error)
            return self._continue_or_truncate(obs)

        assert name is not None
        if name == self.finish_tool:
            self._final_message = str(arguments.get("message", arguments.get("content", "")))
            return self._terminate(reason="finish")

        self._actions.append(
            RolloutAction(tool=name, arguments=dict(arguments), step=self._steps - 1)
        )

        try:
            observation = self.session.execute(name, arguments)
        except UnsupportedToolError as exc:
            # Documented in the module docstring: an observation, not a crash,
            # and not a fake success.
            self._unsupported.append(
                {"step": self._steps - 1, "tool": exc.tool, "reason": exc.reason}
            )
            obs = self._error_obs(
                name,
                "unsupported_tool",
                f"{exc.reason}. This environment cannot run {name!r}; it will never "
                f"succeed. Use a different tool.",
            )
            return self._continue_or_truncate(obs)
        except EnvError as exc:
            return self._env_failure(name, exc)
        except Exception as exc:  # unexpected: still must not kill the trainer
            return self._env_failure(name, exc)

        obs = self._observation(name, observation)
        return self._continue_or_truncate(obs)

    # -- observations ------------------------------------------------------

    def _observation(self, tool: str, observation: Observation) -> JsonObject:
        return {
            "tool": tool,
            "status": observation.status.value,
            "response": observation.response,
            "error_kind": observation.error_kind,
        }

    def _error_obs(self, tool: str, error_kind: str, detail: str) -> JsonObject:
        return {
            "tool": tool,
            "status": CallStatus.ERROR.value,
            "response": {"error": error_kind, "detail": detail},
            "error_kind": error_kind,
        }

    # -- transitions -------------------------------------------------------

    def _base_info(self) -> JsonObject:
        assert self.session is not None
        return {
            "task_id": self.task.task_id,
            "step": self._steps,
            "steps_remaining": max(0, self.max_steps - self._steps),
            "max_steps": self.max_steps,
            "effects": [e.model_dump(mode="json") for e in self.session.effects()],
            "unsupported_tool_attempts": list(self._unsupported),
            "unsupported_tool_attempt_count": len(self._unsupported),
            "tool_calls": len(self._actions),
            "seed": self._seed,
        }

    def _continue_or_truncate(self, observation: JsonObject) -> StepResult:
        """Non-terminal step: reward is 0.0, full stop. Truncate at the budget."""
        info = self._base_info()
        if self._steps >= self.max_steps:
            self._done = True
            info["terminal_reason"] = "max_steps"
            info["verification"] = None
            info["anticheat"] = None
            if self.expose_progress:
                info["progress"] = self._progress()
            return StepResult(observation, 0.0, True, True, info)
        return StepResult(observation, 0.0, False, False, info)

    def _env_failure(self, tool: str, exc: Exception) -> StepResult:
        """Unrecoverable: end at 0.0 and never grade a world that broke."""
        self._done = True
        obs = self._error_obs(tool, "env_error", str(exc))
        info = self._base_info()
        info["terminal_reason"] = "env_error"
        info["env_error"] = {"tool": tool, "type": type(exc).__name__, "detail": str(exc)}
        info["verification"] = None
        info["anticheat"] = None
        return StepResult(obs, 0.0, True, False, info)

    def _rollout_record(self, final_state: Mapping[str, Any]) -> RolloutRecord:
        assert self.session is not None
        return RolloutRecord(
            task_id=self.task.task_id,
            actions=tuple(self._actions),
            manifest=self.session.manifest(),
            pre_state=self._pre_state,
            final_state=dict(final_state),
            primary_keys=self.primary_keys,
            resource_reads=(),
            network_calls=self._network_calls,
            direct_store_writes=(),
        )

    def _terminate(self, *, reason: str) -> StepResult:
        """Grade the episode. The only place a non-zero reward can come from."""
        assert self.session is not None
        self._done = True
        final_state = self.session.snapshot()
        effects = self.session.effects()

        result = evaluate(
            self.verifier,
            final_state,
            effects,
            mode=self.reward_mode,  # type: ignore[arg-type]
            allow_unreviewed=self.allow_unreviewed,
        )
        record = self._rollout_record(final_state)
        report: AntiCheatReport = check_rollout(record, result)
        graded = enforce(result, report)

        info = self._base_info()
        info["terminal_reason"] = reason
        info["final_message"] = self._final_message
        info["digest"] = self.session.digest()
        info["verification"] = graded
        info["verification_json"] = graded.model_dump(mode="json")
        info["anticheat"] = report
        info["anticheat_json"] = report.to_json()
        info["anticheat_clean"] = report.clean
        info["anticheat_findings"] = [
            {"guard": f.guard, "severity": f.severity, "detail": f.detail}
            for f in report.findings
        ]
        info["reward_zeroed_by_anticheat"] = bool(
            result.reward > 0.0 and graded.reward == 0.0
        )
        if self.expose_progress:
            info["progress"] = self._progress(result)

        observation = {
            "tool": self.finish_tool,
            "status": CallStatus.OK.value,
            "response": {"final_message": self._final_message},
            "error_kind": None,
        }
        return StepResult(observation, float(graded.reward), True, False, info)

    # -- diagnostics -------------------------------------------------------

    def _progress(self, result: Any = None) -> JsonObject:
        """Diagnostics. **Never** add any of this to the training reward.

        The partial score below counts STATE_UNCHANGED assertions, which pass
        for free when the agent does nothing. Optimizing it teaches inaction;
        it exists so a human can see how far a truncated episode got.
        """
        assert self.session is not None
        partial = None
        try:
            partial = evaluate(
                self.verifier,
                self.session.snapshot(),
                self.session.effects(),
                mode="partial",
                allow_unreviewed=True,
            )
        except Exception:  # diagnostics must never break an episode
            partial = None
        return {
            "WARNING": "diagnostic only; must never be summed into the training reward",
            "assertions_passing_fraction": None if partial is None else partial.reward,
            "assertions_total": None if partial is None else len(partial.results),
            "tool_calls": len(self._actions),
            "graded_reward": None if result is None else result.reward,
        }
