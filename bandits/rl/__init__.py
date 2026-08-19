"""Stage 8 -- the RL environment layer.

Everything upstream of this package produces artifacts: a schema, tasks,
verifiers, a fidelity report. This is the package that makes them *runnable* --
a gym-style loop a policy can act in, a spec a trainer can read, and a suite it
can sample from.

    from bandits.rl import TaskSuite, make_env

    env = make_env(schema, task, verifier, surface=surface)
    obs = env.reset(seed=0)
    result = env.step({"name": "get_order", "arguments": {"order_id": 7741}})
    result.reward            # 0.0 -- always, on every non-terminal step
    final = env.step({"name": "respond", "arguments": {"message": "Refunded."}})
    final.reward             # verifier-computed, anti-cheat enforced

Two rules this package exists to enforce, both of them at the training boundary
because that is where they bite:

1. **Reward comes from the verifier at the terminal step and nowhere else.** No
   shaping, no partial credit, no bonus for a promising-looking trajectory. See
   :mod:`bandits.rl.episode`.
2. **An unreviewed verifier does not grade training.** See
   :class:`bandits.rl.batch.TaskSuite`.
"""

from __future__ import annotations

from collections.abc import Mapping

from bandits.contracts import StateSchema, TaskCase, ToolClass, ToolSurface, Verifier

from .batch import SuiteEntry, TaskSuite
from .episode import (
    DEFAULT_FINISH_TOOL,
    DEFAULT_MAX_STEPS,
    EpisodeNotStartedError,
    StepResult,
    TraceEnv,
)
from .spec import EnvSpec, ToolSpec, json_schema_for

__all__ = [
    "DEFAULT_FINISH_TOOL",
    "DEFAULT_MAX_STEPS",
    "EnvSpec",
    "EpisodeNotStartedError",
    "StepResult",
    "SuiteEntry",
    "TaskSuite",
    "ToolSpec",
    "TraceEnv",
    "json_schema_for",
    "make_env",
]


def make_env(
    schema: StateSchema,
    task: TaskCase,
    verifier: Verifier,
    *,
    tool_classes: Mapping[str, ToolClass] | None = None,
    surface: ToolSurface | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    reward_mode: str = "all_or_nothing",
    allow_unreviewed: bool = False,
    finish_tool: str = DEFAULT_FINISH_TOOL,
    **kwargs,
) -> TraceEnv:
    """Build one :class:`TraceEnv`. Call ``reset()`` to materialize the world.

    ``tool_classes`` may be omitted when ``surface`` is given -- stage 2 already
    decided which tool reads, writes and reaches outside, and re-deriving that
    here would be a second, weaker classifier.
    """
    return TraceEnv(
        schema=schema,
        task=task,
        verifier=verifier,
        tool_classes=tool_classes,
        surface=surface,
        max_steps=max_steps,
        reward_mode=reward_mode,
        allow_unreviewed=allow_unreviewed,
        finish_tool=finish_tool,
        **kwargs,
    )
