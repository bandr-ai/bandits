"""Tests for the bounded LLM task-clarification seam."""

from __future__ import annotations

import pytest

from bandits.task import TaskDraft, enrich_task, mine_task, review_task_draft
from bandits.task._testdata import RETAIL_SCHEMA, ep_refund_ok


def _task_and_trace():
    trace = ep_refund_ok()
    task, reason, _pre = mine_task(trace, RETAIL_SCHEMA)
    assert reason is None and task is not None
    return task, trace


def test_enrichment_is_a_draft_until_a_human_accepts_it():
    task, trace = _task_and_trace()

    enriched = enrich_task(
        task,
        trace,
        lambda request: TaskDraft(
            instruction="Refund order 7741 because it arrived damaged.",
            rationale="Combines the user's request with the observed refund workflow.",
            ambiguities=("Whether shipping is refundable is not stated.",),
            evidence_message_indexes=(0,),
        ),
    )

    assert enriched.instruction == task.instruction
    assert enriched.provenance["llm_task_reviewed_by"] is None
    assert enriched.provenance["llm_task_draft"]["instruction"].startswith("Refund")

    reviewed = review_task_draft(enriched, "laxman", accept=True)
    assert reviewed.instruction == "Refund order 7741 because it arrived damaged."
    assert reviewed.provenance["llm_task_reviewed_by"] == "laxman"
    assert reviewed.provenance["llm_task_decision"] == "accepted"


def test_rejected_draft_leaves_canonical_instruction_unchanged():
    task, trace = _task_and_trace()
    enriched = enrich_task(task, trace, lambda _request: TaskDraft(instruction="Invented request."))
    reviewed = review_task_draft(enriched, "reviewer", accept=False)
    assert reviewed.instruction == task.instruction
    assert reviewed.provenance["llm_task_decision"] == "rejected"


def test_draft_cannot_cite_messages_outside_its_trace():
    task, trace = _task_and_trace()
    with pytest.raises(ValueError, match="outside this trace"):
        enrich_task(task, trace, lambda _request: TaskDraft(instruction="x", evidence_message_indexes=(99,)))
