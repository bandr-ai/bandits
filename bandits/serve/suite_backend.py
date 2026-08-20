"""Serve the *same* environment the in-process RL layer runs.

Why this module exists
----------------------
``backend.SessionBackend`` grades an episode by calling
:func:`bandits.verify.run.evaluate` directly. :class:`bandits.rl.BanditsEnv`
grades it by calling ``evaluate`` **and then** ``check_rollout`` + ``enforce``,
so a rollout that reached around ``session.execute`` scores zero.

Those are two reward functions. Left alone they drift, and the drift is
invisible and one-directional: a policy trained over HTTP would be graded
without anti-cheat while the same rollout in-process would be zeroed. An agent
that discovers a hack is then *rewarded* for it during distributed training and
punished for it in evaluation -- which reads as an unreproducible eval gap, not
as a reward bug.

So the wire path delegates to ``BanditsEnv`` rather than reimplementing it. There
is exactly one reward function in this repo and this module does not add a
second.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from bandits.contracts import JsonObject
from bandits.contracts import Observation as CoreObservation

from .backend import DEFAULT_MAX_STEPS, StepOutcome
from .protocol import EnvSpec as WireSpec
from .protocol import Observation as WireObservation
from .protocol import RewardRange
from .protocol import ToolSchema as WireTool


def _as_text(payload: Any) -> str:
    import json

    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(payload)



def _to_wire_observation(obs: Any) -> WireObservation:
    """Translate whatever the RL layer returned into the wire observation.

    ``bandits.rl`` may hand back a ``contracts.Observation`` (a tool result) or
    a plain dict (the reset framing). Both become a wire observation carrying a
    ``text`` rendering, because a text-in/text-out policy -- which is what an
    OSS model behind vLLM is -- needs something to read.
    """
    if isinstance(obs, CoreObservation):
        return WireObservation(
            response=obs.response,
            status=obs.status,
            error_kind=obs.error_kind,
            text=_as_text(obs.response),
        )
    return WireObservation(response=obs, status="ok", error_kind=None, text=_as_text(obs))


def _reward_range(value: Any) -> RewardRange:
    """Normalize a ``(low, high)`` pair into the wire's RewardRange."""
    low, high = (value.low, value.high) if hasattr(value, "low") else tuple(value)
    return RewardRange(low=float(low), high=float(high))


def _to_wire_spec(spec: Any, task_ids: tuple[str, ...]) -> WireSpec:
    """Translate ``bandits.rl.EnvSpec`` into the wire contract.

    The two are deliberately different types: the RL spec carries training
    concerns (reward mode, reviewer, outcome label) that no HTTP client needs,
    and the wire spec carries protocol identity the trainer does need. Mapping
    them here keeps the RL layer from having to know a transport exists.

    ``finish_tool`` is published as a real tool so a policy reading ``/spec``
    can see how to terminate. Omitting it would leave an agent able to act but
    with no way to say it is done, and every episode would truncate.
    """
    tools = [
        WireTool(
            name=t.name,
            description=getattr(t, "description", None) or "",
            input_schema=getattr(t, "parameters", None) or {"type": "object"},
            tool_class=str(getattr(t, "tool_class", "") or "unknown"),
            supported=True,
        )
        for t in getattr(spec, "tools", ())
    ]
    finish = getattr(spec, "finish_tool", None)
    if finish and not any(t.name == finish for t in tools):
        tools.append(
            WireTool(
                name=finish,
                description="End the episode and submit the final message.",
                input_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                tool_class="unknown",  # no ToolClass models "terminal"; it is not a reconstructed tool
                supported=True,
            )
        )
    unsupported = tuple(getattr(spec, "unsupported_tools", ()) or ())
    for name in unsupported:
        if not any(t.name == name for t in tools):
            tools.append(
                WireTool(
                    name=name,
                    description="",
                    input_schema={"type": "object"},
                    tool_class="unknown",
                    supported=False,
                    unsupported_reason="not reimplementable from the observed traces",
                )
            )
    return WireSpec(
        env_id=getattr(spec, "env_id", "") or "",
        task_id=spec.task_id,
        task_ids=list(task_ids),
        instruction=spec.instruction,
        tools=tools,
        max_steps=int(getattr(spec, "max_steps", DEFAULT_MAX_STEPS)),
        reward_range=_reward_range(getattr(spec, "reward_range", (0.0, 1.0))),
        env_digest=getattr(spec, "schema_digest", None),
        verifier_digest=getattr(spec, "verifier_digest", None),
        static_entities=list(getattr(spec, "static_entities", ()) or ()),
        unsupported_tools=list(unsupported),
    )


class SuiteEpisode:
    """One rollout, driven by a :class:`bandits.rl.BanditsEnv`.

    Mirrors the attribute surface the HTTP and MCP transports use, but owns a
    ``BanditsEnv`` instead of a raw session so reward, truncation and anti-cheat
    all come from the single implementation in ``bandits.rl``.
    """

    def __init__(self, episode_id: str, env: Any, task: Any, tools: tuple, seed: int | None):
        self.episode_id = episode_id
        self.task = task
        self.task_id = task.task_id
        self.tools = tools
        self.seed = seed
        self.max_steps = getattr(env, "max_steps", DEFAULT_MAX_STEPS)
        self._env = env
        self._lock = threading.RLock()
        self._closed = False
        self._steps = 0
        self._finished = False
        self._final_digest: str | None = None
        self._opening = env.reset(seed=seed)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def final_digest(self) -> str | None:
        return self._final_digest

    def opening_observation(self) -> WireObservation:
        payload = dict(self._opening) if isinstance(self._opening, dict) else self._opening
        return _to_wire_observation(payload)

    def step(self, name: str, arguments: Mapping[str, Any] | None = None) -> StepOutcome:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"episode {self.episode_id} is closed")
            if self._finished:
                raise RuntimeError(f"episode {self.episode_id} has already finished")
            result = self._env.step({"name": name, "arguments": dict(arguments or {})})
            self._steps += 1
            self._finished = bool(result.done)
            info: JsonObject = dict(result.info or {})
            if self._finished:
                self._final_digest = info.get("digest")
            return StepOutcome(
                observation=_to_wire_observation(result.observation),
                reward=float(result.reward),
                done=bool(result.done),
                truncated=bool(result.truncated),
                step=self._steps,
                info=info,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            closer = getattr(self._env, "close", None)
            if callable(closer):
                closer()


class SuiteBackend:
    """An :class:`~bandits.serve.backend.EnvBackend` over a ``TaskSuite``.

    The suite has already excluded tasks whose verifier is unreviewed, so this
    backend cannot serve an ungraded environment by accident -- the review gate
    from PLAN.md Step 11 holds over the wire, not just in-process.
    """

    def __init__(self, suite: Any):
        self._suite = suite
        self._lock = threading.Lock()

    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._suite.task_ids)

    def spec(self, task_id: str | None = None):
        ids = self.task_ids()
        if not ids:
            raise KeyError(
                "this suite serves no tasks; every task was excluded, most often "
                "because no verifier carries a human sign-off"
            )
        chosen = task_id or ids[0]
        if chosen not in ids:
            raise KeyError(chosen)
        specs = {s.task_id: s for s in self._suite.specs()}
        return _to_wire_spec(specs[chosen], ids)

    def new_episode(self, episode_id: str, task_id: str | None = None, seed: int | None = None):
        ids = self.task_ids()
        if not ids:
            raise KeyError("this suite serves no tasks")
        chosen = task_id or ids[0]
        if chosen not in ids:
            raise KeyError(chosen)
        with self._lock:
            env = self._suite.env_for(chosen)
        entry = next(e for e in self._suite if e.task.task_id == chosen)
        tools = tuple(getattr(entry, "tools", ()) or ())
        return SuiteEpisode(episode_id, env, entry.task, tools, seed)


__all__ = ["SuiteBackend", "SuiteEpisode"]
