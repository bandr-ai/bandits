"""``TaskSuite`` -- many tasks, one training distribution, with the review gate.

A trainer does not want one environment, it wants a distribution to sample from.
:class:`TaskSuite` is that distribution: a set of ``(task, verifier)`` pairs over
a shared schema and action space, sampled by seed, filterable by outcome label
and by review status.

The gate
--------
**A task whose verifier has ``reviewed_by=None`` is excluded from the suite by
default.** PLAN.md Step 11 says a generated reward function must be read by a
person before it grades anything, and this is the boundary where that rule
actually bites: :func:`bandits.verify.run.evaluate` can refuse a single call,
but training is where an unexamined reward function does its damage, silently,
across millions of steps.

Excluded tasks are never dropped quietly (BUILD_PLAN rule 6). They land in
:attr:`TaskSuite.excluded` with a reason, so an operator can see that their
suite of 40 tasks is training on 6 and why.

``allow_unreviewed=True`` opts back in, explicitly, in the caller's own code --
which is the point: the decision is visible at the call site instead of buried
in a default.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from bandits.contracts import (
    JsonObject,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolSurface,
    Trace,
    TraceCorpus,
    Verifier,
)
from bandits.verify import synthesize_verifier
from bandits.verify.synthesize import UnlabeledTraceError

from .episode import DEFAULT_FINISH_TOOL, DEFAULT_MAX_STEPS, TraceEnv
from .spec import EnvSpec

__all__ = ["SuiteEntry", "TaskSuite"]


@dataclass(frozen=True)
class SuiteEntry:
    """One trainable unit: a task plus the frozen reward function for it."""

    task: TaskCase
    verifier: Verifier

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def outcome(self) -> bool | None:
        """The source trace's ground-truth label. ``None`` means unlabeled."""
        return self.task.outcome

    @property
    def reviewed(self) -> bool:
        return self.verifier.reviewed_by is not None

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self.task.provenance.get("solvability_warnings") or ())


class TaskSuite:
    """A sampleable collection of tasks sharing one schema and action space."""

    def __init__(
        self,
        entries: Iterable[SuiteEntry],
        *,
        schema: StateSchema,
        tool_classes: Mapping[str, ToolClass] | None = None,
        surface: ToolSurface | None = None,
        allow_unreviewed: bool = False,
        max_steps: int = DEFAULT_MAX_STEPS,
        reward_mode: str = "all_or_nothing",
        finish_tool: str = DEFAULT_FINISH_TOOL,
        excluded: Sequence[JsonObject] = (),
    ) -> None:
        if tool_classes is None:
            if surface is None:
                raise ValueError("pass tool_classes, or a surface to derive them from")
            tool_classes = {p.name: p.tool_class for p in surface.tools}
        self.schema = schema
        self.tool_classes = dict(tool_classes)
        self.surface = surface
        self.allow_unreviewed = allow_unreviewed
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.finish_tool = finish_tool

        kept: list[SuiteEntry] = []
        dropped: list[JsonObject] = list(excluded)
        for entry in entries:
            if not entry.reviewed and not allow_unreviewed:
                dropped.append(
                    {
                        "task_id": entry.task_id,
                        "reason": "unreviewed_verifier",
                        "detail": (
                            f"verifier {entry.verifier.verifier_id!r} has reviewed_by=None. "
                            f"A generated reward function nobody has read must not drive "
                            f"training (PLAN.md Step 11). Review it, or construct the suite "
                            f"with allow_unreviewed=True."
                        ),
                    }
                )
                continue
            kept.append(entry)
        self._entries = tuple(kept)
        self.excluded = tuple(dropped)

    # -- collection protocol ----------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[SuiteEntry]:
        return iter(self._entries)

    def __getitem__(self, index: int) -> SuiteEntry:
        return self._entries[index]

    def __repr__(self) -> str:
        return (
            f"TaskSuite({len(self._entries)} task(s), {len(self.excluded)} excluded, "
            f"allow_unreviewed={self.allow_unreviewed})"
        )

    @property
    def entries(self) -> tuple[SuiteEntry, ...]:
        return self._entries

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(e.task_id for e in self._entries)

    def by_id(self, task_id: str) -> SuiteEntry | None:
        for entry in self._entries:
            if entry.task_id == task_id:
                return entry
        return None

    # -- filtering ---------------------------------------------------------

    def filter(
        self,
        *,
        outcome: bool | None = ...,  # type: ignore[assignment]
        reviewed: bool | None = None,
        task_ids: Iterable[str] | None = None,
        without_warnings: bool = False,
    ) -> TaskSuite:
        """A new suite holding the subset that matches.

        ``outcome`` filters on the source trace's ground-truth label, and
        accepts ``None`` as a real value (unlabeled), so its "no filter"
        sentinel is ``...``. ``reviewed=True`` keeps only reviewed verifiers,
        which is the filter to use when a suite was built with
        ``allow_unreviewed=True`` for inspection and now has to be narrowed for
        an actual run.
        """
        wanted = set(task_ids) if task_ids is not None else None
        kept: list[SuiteEntry] = []
        for entry in self._entries:
            if outcome is not ... and entry.outcome != outcome:
                continue
            if reviewed is not None and entry.reviewed != reviewed:
                continue
            if wanted is not None and entry.task_id not in wanted:
                continue
            if without_warnings and entry.warnings:
                continue
            kept.append(entry)
        return TaskSuite(
            kept,
            schema=self.schema,
            tool_classes=self.tool_classes,
            surface=self.surface,
            allow_unreviewed=self.allow_unreviewed,
            max_steps=self.max_steps,
            reward_mode=self.reward_mode,
            finish_tool=self.finish_tool,
            excluded=self.excluded,
        )

    # -- sampling ----------------------------------------------------------

    def sample(self, seed: int | None = None) -> SuiteEntry:
        """Pick one task. Same seed => same task, forever.

        ``seed=None`` returns the first task rather than a random one: an
        unseeded rollout that silently varied would make a training run
        irreproducible without anyone noticing.
        """
        if not self._entries:
            raise IndexError(
                "task suite is empty. If every task was dropped, check "
                "TaskSuite.excluded - an unreviewed verifier is excluded by design."
            )
        if seed is None:
            return self._entries[0]
        return self._entries[random.Random(seed).randrange(len(self._entries))]

    def make_env(self, seed: int | None = None, **kwargs) -> TraceEnv:
        """Build (but do not reset) the environment for the task ``seed`` selects."""
        entry = self.sample(seed)
        return self.env_for(entry, **kwargs)

    def env_for(self, entry: SuiteEntry | str, **kwargs) -> TraceEnv:
        if isinstance(entry, str):
            found = self.by_id(entry)
            if found is None:
                raise KeyError(f"no task {entry!r} in this suite")
            entry = found
        options: JsonObject = {
            "max_steps": self.max_steps,
            "reward_mode": self.reward_mode,
            "finish_tool": self.finish_tool,
            "allow_unreviewed": self.allow_unreviewed,
        }
        options.update(kwargs)
        return TraceEnv(
            schema=self.schema,
            task=entry.task,
            verifier=entry.verifier,
            tool_classes=self.tool_classes,
            surface=self.surface,
            **options,
        )

    def specs(self) -> tuple[EnvSpec, ...]:
        """One :class:`EnvSpec` per task. Serializable; hand this to a trainer."""
        return tuple(self.env_for(entry).spec() for entry in self._entries)

    def to_json(self) -> JsonObject:
        return {
            "task_count": len(self._entries),
            "allow_unreviewed": self.allow_unreviewed,
            "max_steps": self.max_steps,
            "reward_mode": self.reward_mode,
            "tasks": [spec.to_json() for spec in self.specs()],
            "excluded": list(self.excluded),
        }

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        *,
        corpus: TraceCorpus,
        schema: StateSchema,
        tasks: Sequence[TaskCase],
        surface: ToolSurface | None = None,
        tool_classes: Mapping[str, ToolClass] | None = None,
        reviewed_by: Mapping[str, str] | None = None,
        allow_unreviewed: bool = False,
        max_steps: int = DEFAULT_MAX_STEPS,
        reward_mode: str = "all_or_nothing",
        finish_tool: str = DEFAULT_FINISH_TOOL,
    ) -> TaskSuite:
        """Build a suite straight out of the deterministic pipeline.

        Synthesizes one verifier per task from that task's own trace. A trace
        that is not labeled correct yields no verifier at all -- that refusal
        lives in :func:`bandits.verify.synthesize.synthesize_verifier` and is
        surfaced here as an exclusion with a reason, never as a silent skip.

        ``reviewed_by`` maps ``task_id -> reviewer``, which is how a human's sign-off
        (however it is stored: a review file, a ticket, a signature) enters the
        suite. Anything not in that map stays unreviewed and is excluded unless
        ``allow_unreviewed=True``.
        """
        if tool_classes is None and surface is not None:
            tool_classes = {p.name: p.tool_class for p in surface.tools}
        externals = (
            [n for n, k in (tool_classes or {}).items() if k is ToolClass.EXTERNAL]
            if tool_classes
            else None
        )
        traces: dict[str, Trace] = {t.trace_id: t for t in corpus.traces}
        signed = dict(reviewed_by or {})

        entries: list[SuiteEntry] = []
        excluded: list[JsonObject] = []
        for task in tasks:
            trace = traces.get(task.trace_id)
            if trace is None:
                excluded.append(
                    {
                        "task_id": task.task_id,
                        "reason": "missing_trace",
                        "detail": f"task names trace {task.trace_id!r}, absent from the corpus",
                    }
                )
                continue
            try:
                verifier = synthesize_verifier(
                    task, trace, schema, external_tools=externals
                )
            except UnlabeledTraceError as exc:
                excluded.append(
                    {"task_id": task.task_id, "reason": "unlabeled_trace", "detail": str(exc)}
                )
                continue
            except ValueError as exc:
                excluded.append(
                    {"task_id": task.task_id, "reason": "synthesis_failed", "detail": str(exc)}
                )
                continue
            if not verifier.assertions:
                # evaluate() would refuse this one anyway: a verifier with no
                # assertions passes every rollout, including the empty one.
                excluded.append(
                    {
                        "task_id": task.task_id,
                        "reason": "no_assertions",
                        "detail": "synthesis produced no assertions; it would pass every rollout",
                    }
                )
                continue
            reviewer = signed.get(task.task_id)
            if reviewer:
                verifier = verifier.model_copy(update={"reviewed_by": reviewer})
            entries.append(SuiteEntry(task=task, verifier=verifier))

        return cls(
            entries,
            schema=schema,
            tool_classes=tool_classes,
            surface=surface,
            allow_unreviewed=allow_unreviewed,
            max_steps=max_steps,
            reward_mode=reward_mode,
            finish_tool=finish_tool,
            excluded=excluded,
        )

