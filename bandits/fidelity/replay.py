"""Replay a recorded trace against the environment rebuilt from it.

This is the measurement the whole project turns on. An environment that cannot
reproduce its own source trace is not a model of anything (PLAN.md Step 12), and
the only way to find that out is to feed the recorded actions back in, in order,
and compare what comes out.

How a replay runs
-----------------
1. Build a session for the trace's :class:`~bandits.contracts.TaskCase` -- the
   same reconstructed pre-state an RL rollout would start from. If the pre-state
   is wrong, the replay says so, which is exactly what we want it to say.
2. Feed every :class:`~bandits.contracts.InvocationPoint` in ``step`` order via
   ``session.execute(tool, arguments)``.
3. Compare each returned :class:`~bandits.contracts.Observation` against the
   recorded ``response``/``status``/``error_kind``.

Three rules that are easy to get wrong
--------------------------------------
**Never stop at the first divergence.** Per-tool rates are the product. Halting
on the first bad call would report one broken tool and leave every other tool
unmeasured, which is precisely the aggregate-hides-the-problem failure the gate
exists to prevent. So a divergence is recorded and the replay continues -- and
it continues *against the environment as it now is*, divergence included, which
means a bad write shows up again in the reads that follow it. That is honest:
production did not get a fresh environment after each call either.

**Unsupported is not mismatched.** A tool that raises
:class:`~bandits.env.UnsupportedToolError` is counted as ``unsupported`` and
reported on its own line. "We could not model this" and "we modeled it wrong"
are different facts with different fixes: the first sends you to schema
inference or to an explicit rule, the second to the tool implementation. They
are still both failures -- see :mod:`bandits.fidelity.gate` for why unsupported
calls stay in the denominator.

**External tools are checked by effect, not by response.** ``send_email``
returned ``{"sent": true}`` in production; the stub returns whatever
acknowledgement it was configured with, and it never sent anything. Demanding
body equality would measure how well we guessed a vendor's acknowledgement
format, which is not a property of the world we rebuilt. What *is* a property of
the world we rebuilt is whether the attempt landed in the effect ledger with the
arguments it was called with -- so that is what is asserted, together with the
call status. Verifiers assert on the ledger too (``EFFECT_COUNT``), so this is
the same surface reward is computed from.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from bandits.contracts import (
    CallStatus,
    EnvManifest,
    InvocationPoint,
    JsonObject,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolFidelity,
    ToolSurface,
    Trace,
    TraceCorpus,
)
from bandits.env import EnvError, UnsupportedToolError, build_session

from .compare import Divergence, compare_observations

__all__ = [
    "CallReplay",
    "ReplayResult",
    "replay_corpus",
    "replay_trace",
    "tool_classes_from_surface",
]

DEFAULT_MAX_EXAMPLES = 3
"""How many example divergences to keep per tool. Enough to see the pattern."""


def tool_classes_from_surface(surface: ToolSurface) -> dict[str, ToolClass]:
    """``{tool: ToolClass}`` from a stage-2 surface, which is what a session wants."""
    return {profile.name: profile.tool_class for profile in surface.tools}


@dataclass(frozen=True)
class CallReplay:
    """The result of replaying exactly one recorded invocation."""

    step: int
    tool: str
    verdict: str
    """``matched`` | ``mismatched`` | ``unsupported``."""

    divergences: tuple[Divergence, ...] = ()
    unsupported_reason: str | None = None
    external: bool = False
    """True when compared by effect rather than by response body."""

    @property
    def matched(self) -> bool:
        return self.verdict == "matched"

    @property
    def blocking(self) -> tuple[Divergence, ...]:
        """The divergences that actually counted against the environment."""
        return tuple(d for d in self.divergences if not d.tolerated)

    def to_json(self) -> JsonObject:
        return {
            "step": self.step,
            "tool": self.tool,
            "verdict": self.verdict,
            "external": self.external,
            "unsupported_reason": self.unsupported_reason,
            "divergences": [d.to_json() for d in self.divergences],
        }


@dataclass
class ReplayResult:
    """Everything one trace's replay produced, before the gate is applied."""

    trace_id: str
    task_id: str
    env_id: str
    calls: list[CallReplay] = field(default_factory=list)
    manifest: EnvManifest | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def replayed(self) -> int:
        return len(self.calls)

    @property
    def matched(self) -> int:
        return sum(1 for c in self.calls if c.verdict == "matched")

    @property
    def mismatched(self) -> int:
        return sum(1 for c in self.calls if c.verdict == "mismatched")

    @property
    def unsupported(self) -> int:
        return sum(1 for c in self.calls if c.verdict == "unsupported")

    def tools(self) -> tuple[str, ...]:
        return tuple(sorted({c.tool for c in self.calls}))

    def per_tool(self, max_examples: int = DEFAULT_MAX_EXAMPLES) -> tuple[ToolFidelity, ...]:
        """Accumulate :class:`ToolFidelity` rows, sorted by tool name."""
        return per_tool_fidelity([self], max_examples=max_examples)


def per_tool_fidelity(
    results: Sequence[ReplayResult], *, max_examples: int = DEFAULT_MAX_EXAMPLES
) -> tuple[ToolFidelity, ...]:
    """Merge one or many replays into per-tool rows.

    Merging across traces is how a corpus-level number is produced: the counts
    add, and the examples are taken in trace-then-step order so the first thing
    an operator reads is the earliest failure.
    """
    counts: dict[str, dict[str, int]] = {}
    examples: dict[str, list[JsonObject]] = {}
    for result in results:
        for call in result.calls:
            row = counts.setdefault(
                call.tool, {"replayed": 0, "matched": 0, "mismatched": 0, "unsupported": 0}
            )
            row["replayed"] += 1
            row[call.verdict] += 1
            if call.verdict == "matched":
                continue
            bucket = examples.setdefault(call.tool, [])
            if len(bucket) >= max_examples:
                continue
            bucket.append(_example(result, call))
    return tuple(
        ToolFidelity(
            tool=tool,
            replayed=row["replayed"],
            matched=row["matched"],
            mismatched=row["mismatched"],
            unsupported=row["unsupported"],
            examples=tuple(examples.get(tool, ())),
        )
        for tool, row in sorted(counts.items())
    )


def _example(result: ReplayResult, call: CallReplay) -> JsonObject:
    example: JsonObject = {
        "trace_id": result.trace_id,
        "step": call.step,
        "verdict": call.verdict,
    }
    if call.unsupported_reason:
        example["reason"] = call.unsupported_reason
    blocking = call.blocking
    if blocking:
        example["divergences"] = [d.to_json() for d in blocking]
    return example


def _unbuildable(
    trace: Trace, task: TaskCase, env_id: str | None, exc: Exception
) -> ReplayResult:
    """Report a trace whose environment could not be constructed at all.

    Every recorded call becomes ``unsupported`` rather than ``mismatched``.
    The distinction is the whole point of the gate: "we could not model this"
    and "we modelled it wrong" have different fixes, and collapsing them would
    tell an operator to go debug a comparison that never ran.
    """
    reason = f"environment could not be built: {type(exc).__name__}: {exc}"
    result = ReplayResult(
        trace_id=trace.trace_id,
        task_id=task.task_id,
        env_id=env_id or f"env-{task.task_id}-unbuildable",
    )
    result.notes.append(reason)
    for inv in sorted(trace.invocations, key=lambda i: i.step):
        result.calls.append(
            CallReplay(
                step=inv.step,
                tool=inv.tool,
                verdict="unsupported",
                unsupported_reason=reason,
            )
        )
    return result


# -- the replay ------------------------------------------------------------


def replay_trace(
    trace: Trace,
    schema: StateSchema,
    task: TaskCase,
    tool_classes: Mapping[str, ToolClass],
    *,
    surface: ToolSurface | None = None,
    rules: Mapping[str, object] | None = None,
    external_stubs: Mapping[str, object] | None = None,
    env_id: str | None = None,
) -> ReplayResult:
    """Replay one trace against a freshly built environment for its task.

    ``task`` must have been mined from ``trace``; a mismatch is a caller bug and
    raises rather than silently measuring the wrong pair.

    Pass ``surface`` whenever it is available. Without it the environment falls
    back to omitting NULL columns from read responses, which is a strictly
    weaker guess than the observed response projection and shows up directly as
    lost fidelity -- a read starts emitting a key production never sent as soon
    as some write populates that column.
    """
    if task.trace_id != trace.trace_id:
        raise ValueError(
            f"task {task.task_id!r} was mined from trace {task.trace_id!r}, "
            f"not {trace.trace_id!r}; replaying them together would measure nothing"
        )

    try:
        session = build_session(
            schema,
            task,
            tool_classes,
            surface=surface,
            rules=rules,  # type: ignore[arg-type]
            external_stubs=external_stubs,  # type: ignore[arg-type]
        env_id=env_id,
        )
    except EnvError as exc:
        # The environment could not even be built for this task -- typically an
        # inferred schema that contradicts the data it must hold. That is a
        # reconstruction failure, not a per-call mismatch, so every recorded
        # call is reported unsupported rather than letting the exception escape
        # and take the whole gate run down with it.
        return _unbuildable(trace, task, env_id, exc)
    manifest = session.manifest()
    result = ReplayResult(
        trace_id=trace.trace_id,
        task_id=task.task_id,
        env_id=manifest.env_id,
        manifest=manifest,
    )
    if manifest.unsupported_tools:
        result.notes.append(
            "environment declares unsupported tools: " + ", ".join(manifest.unsupported_tools)
        )
    externals = {t for t, k in tool_classes.items() if k is ToolClass.EXTERNAL}

    try:
        for inv in sorted(trace.invocations, key=lambda i: i.step):
            result.calls.append(_replay_call(session, inv, external=inv.tool in externals))
    finally:
        session.close()
    return result


def _replay_call(session, inv: InvocationPoint, *, external: bool) -> CallReplay:
    """Execute one recorded call and classify the outcome. Never raises for a
    tool the environment refuses to run -- that is a measurement, not an error."""
    effects_before = len(session.effects())
    try:
        obs = session.execute(inv.tool, dict(inv.arguments or {}))
    except UnsupportedToolError as exc:
        return CallReplay(
            step=inv.step,
            tool=inv.tool,
            verdict="unsupported",
            unsupported_reason=exc.reason,
            external=external,
        )
    except EnvError as exc:
        # Any other environment failure is still "we could not run this here".
        # It is reported as unsupported with the reason attached rather than as
        # a wrong answer, because there is no answer to be wrong about.
        return CallReplay(
            step=inv.step,
            tool=inv.tool,
            verdict="unsupported",
            unsupported_reason=f"{type(exc).__name__}: {exc}",
            external=external,
        )

    divergences = list(
        compare_observations(
            inv.response,
            inv.status,
            inv.error_kind,
            obs,
            compare_response=not external,
        )
    )
    if external:
        divergences.extend(_check_effect(session, inv, effects_before))

    blocking = [d for d in divergences if not d.tolerated]
    return CallReplay(
        step=inv.step,
        tool=inv.tool,
        verdict="mismatched" if blocking else "matched",
        divergences=tuple(divergences),
        external=external,
    )


def _check_effect(session, inv: InvocationPoint, effects_before: int) -> list[Divergence]:
    """For an EXTERNAL tool: assert the attempt was *logged*, not that the body matched.

    The recorded response is a vendor acknowledgement. What the rebuilt world
    owes us is that the irreversible action was captured with its arguments and
    never performed -- which is the same ledger a verifier's ``EFFECT_COUNT``
    assertion reads.
    """
    effects = session.effects()
    if inv.status is not CallStatus.OK:
        # A recorded failure of an external tool: the status comparison above
        # already covers it, and nothing should have been logged as done.
        return []
    if len(effects) <= effects_before:
        return [
            Divergence(
                path="$effects",
                expected=f"one logged effect for {inv.tool!r}",
                actual="no effect was logged",
                reason="external tool ran without recording its attempt in the effect ledger",
                tolerated=False,
                field_class="exact",
            )
        ]
    logged = effects[-1]
    out: list[Divergence] = []
    if logged.tool != inv.tool:
        out.append(
            Divergence(
                path="$effects[-1].tool",
                expected=inv.tool,
                actual=logged.tool,
                reason="the logged effect names a different tool",
                tolerated=False,
                field_class="exact",
            )
        )
    if dict(logged.arguments) != dict(inv.arguments or {}):
        out.append(
            Divergence(
                path="$effects[-1].arguments",
                expected=dict(inv.arguments or {}),
                actual=dict(logged.arguments),
                reason="the logged effect does not carry the arguments the tool was called with",
                tolerated=False,
                field_class="exact",
            )
        )
    return out


def replay_corpus(
    corpus: TraceCorpus,
    schema: StateSchema,
    tasks: Iterable[TaskCase],
    tool_classes: Mapping[str, ToolClass],
    **kwargs,
) -> list[ReplayResult]:
    """Replay every task whose source trace is present in ``corpus``.

    A task whose trace is missing is not skipped quietly -- it raises. Silent
    dropping is the failure mode this project exists to avoid, and here it would
    inflate the fidelity number by measuring fewer traces than the operator
    thinks.
    """
    by_id = {t.trace_id: t for t in corpus.traces}
    results: list[ReplayResult] = []
    for task in tasks:
        trace = by_id.get(task.trace_id)
        if trace is None:
            raise KeyError(
                f"task {task.task_id!r} refers to trace {task.trace_id!r}, which is not in "
                f"this corpus; refusing to replay a partial set and call it a fidelity number"
            )
        results.append(replay_trace(trace, schema, task, tool_classes, **kwargs))
    return results
