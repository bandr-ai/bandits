"""Mine ``TaskCase``s out of a ``TraceCorpus``.

One trace becomes at most one task. The instruction is the trace's first user
message - the task statement as the customer actually phrased it, not a
paraphrase we invented. The starting state is whatever
:mod:`bandits.task.prestate` could prove from reads before the first write. The
outcome label rides along unchanged; stage 6 refuses to synthesize a reward
function from anything that is not labeled correct.

``provenance`` carries one convention worth naming here, because ``EntityRows``
has no field for it: ``provenance["partial_pre_state_rows"]`` maps an entity name
to the keys of its **partial** rows - rows production only *named* (an id in a
list) and never showed. Those rows sit in ``pre_state`` like any other, but only
their key and the query filters that produced them are known; the remaining
columns are unknown, not empty. Anything that would read an absent column as
evidence must consult this list first. See :mod:`bandits.task.prestate`.

Nothing is silently skipped (BUILD_PLAN rule 6). A trace we decline to turn into
a task lands in :attr:`MiningResult.skipped` with a reason.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from bandits.contracts import JsonObject, StateSchema, TaskCase, Trace, TraceCorpus
from bandits.task.prestate import PreState, reconstruct_pre_state

__all__ = ["MiningResult", "mine_task", "mine_tasks", "task_id_for"]


@dataclass
class MiningResult:
    tasks: list[TaskCase] = field(default_factory=list)
    skipped: list[JsonObject] = field(default_factory=list)
    """One entry per trace we refused, with ``trace_id`` and ``reason``."""

    @property
    def warned(self) -> list[TaskCase]:
        """Tasks carrying solvability warnings. Do not train on these unreviewed."""
        return [t for t in self.tasks if t.provenance.get("solvability_warnings")]


def task_id_for(trace: Trace) -> str:
    return f"task-{trace.trace_id}"


def mine_task(
    trace: Trace,
    schema: StateSchema,
    *,
    write_tools: Iterable[str] | None = None,
) -> tuple[TaskCase | None, str | None, PreState | None]:
    """Build one task. Returns ``(task, skip_reason, pre_state)``."""
    instruction = trace.instruction
    if not instruction or not instruction.strip():
        return None, "no user message: there is no task statement to give an agent", None
    if not trace.invocations:
        return None, "no invocations: nothing to reconstruct a starting state from", None

    pre = reconstruct_pre_state(trace, schema, write_tools=write_tools)
    tools = tuple(dict.fromkeys(inv.tool for inv in sorted(trace.invocations, key=lambda i: i.step)))
    provenance: JsonObject = {
        "source_trace_id": trace.trace_id,
        "source": trace.source,
        "source_digest": trace.source_digest,
        "invocation_count": len(trace.invocations),
        "first_write_step": pre.first_write_step,
        "pre_state_row_count": sum(len(b) for b in pre.rows.values()),
        "partial_pre_state_rows": {k: tuple(v) for k, v in pre.partial_row_keys().items()},
        "blocked_post_state_reads": tuple(pre.blocked),
        "solvability_warnings": tuple(pre.warnings),
        "notes": tuple(pre.notes),
    }
    task = TaskCase(
        task_id=task_id_for(trace),
        trace_id=trace.trace_id,
        instruction=instruction.strip(),
        pre_state=pre.to_entity_rows(),
        tools=tools,
        outcome=trace.outcome,
        provenance=provenance,
    )
    return task, None, pre


def mine_tasks(
    corpus: TraceCorpus,
    schema: StateSchema,
    *,
    write_tools: Iterable[str] | None = None,
) -> MiningResult:
    """Mine every trace in the corpus."""
    result = MiningResult()
    for trace in corpus.traces:
        task, reason, _pre = mine_task(trace, schema, write_tools=write_tools)
        if task is None:
            result.skipped.append({"trace_id": trace.trace_id, "reason": reason})
            continue
        result.tasks.append(task)
    return result
