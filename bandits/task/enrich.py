"""LLM-assisted task clarification, kept outside the reward authority.

Task mining preserves the customer wording from a trace as the canonical task
instruction.  That is deliberately deterministic.  This module gives an LLM a
useful but bounded job: turn a noisy conversation into a concise *draft*
instruction, list the evidence it used, and identify ambiguity a reviewer must
resolve.  It cannot propose a starting state, expected final state, or reward.

The caller supplies the model adapter as a callable.  Keeping provider code out
of this package makes the artifact portable and makes all LLM output testable
without credentials.  An adapter may call any hosted model, local model, or an
internal review service.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field

from bandits.contracts import Contract, JsonObject, TaskCase, Trace

__all__ = [
    "TaskDraft",
    "TaskEnrichmentRequest",
    "TaskEnricher",
    "enrich_task",
    "review_task_draft",
]


class TaskEnrichmentRequest(Contract):
    """The narrow, auditable evidence an LLM may use to clarify one task."""

    task_id: str
    trace_id: str
    canonical_instruction: str
    messages: tuple[JsonObject, ...] = ()
    observed_tools: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class TaskDraft(Contract):
    """A proposed instruction; never a proposed reward or world state."""

    instruction: str = Field(min_length=1)
    rationale: str = ""
    ambiguities: tuple[str, ...] = ()
    evidence_message_indexes: tuple[int, ...] = ()


TaskEnricher = Callable[[TaskEnrichmentRequest], TaskDraft]


def _request(task: TaskCase, trace: Trace) -> TaskEnrichmentRequest:
    if task.trace_id != trace.trace_id:
        raise ValueError(
            f"task {task.task_id!r} names trace {task.trace_id!r}, not {trace.trace_id!r}"
        )
    return TaskEnrichmentRequest(
        task_id=task.task_id,
        trace_id=trace.trace_id,
        canonical_instruction=task.instruction,
        messages=tuple(message.model_dump(mode="json") for message in trace.messages),
        observed_tools=task.tools,
        warnings=tuple(task.provenance.get("solvability_warnings") or ()),
    )


def enrich_task(task: TaskCase, trace: Trace, enrich: TaskEnricher) -> TaskCase:
    """Attach an unreviewed LLM draft without changing ``task.instruction``.

    The original instruction remains the executable task until
    :func:`review_task_draft` is called.  The serializable draft and the exact
    input evidence live in provenance, which makes a prompt/model integration
    auditable without expanding the frozen core contract.
    """
    request = _request(task, trace)
    draft = enrich(request)
    if not isinstance(draft, TaskDraft):
        raise TypeError("task enricher must return TaskDraft")
    invalid_indexes = [i for i in draft.evidence_message_indexes if i < 0 or i >= len(trace.messages)]
    if invalid_indexes:
        raise ValueError(f"draft cites message indexes outside this trace: {invalid_indexes}")
    provenance = dict(task.provenance)
    provenance["llm_task_draft"] = draft.model_dump(mode="json")
    provenance["llm_task_request"] = request.model_dump(mode="json")
    provenance["llm_task_reviewed_by"] = None
    return task.model_copy(update={"provenance": provenance})


def review_task_draft(task: TaskCase, reviewer: str, *, accept: bool) -> TaskCase:
    """Record a human decision; only an accepted draft changes the instruction."""
    if not reviewer.strip():
        raise ValueError("reviewer must be a non-empty name")
    raw = task.provenance.get("llm_task_draft")
    if not isinstance(raw, dict):
        raise ValueError(f"task {task.task_id!r} has no LLM task draft to review")
    draft = TaskDraft.model_validate(raw)
    provenance = dict(task.provenance)
    provenance["llm_task_reviewed_by"] = reviewer.strip()
    provenance["llm_task_decision"] = "accepted" if accept else "rejected"
    return task.model_copy(
        update={
            "instruction": draft.instruction.strip() if accept else task.instruction,
            "provenance": provenance,
        }
    )
