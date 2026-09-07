"""The audit is advisory. These tests exist mostly to prove it stays that way.

Nothing here reaches a model or a REPL sandbox: the predictor is injected, so
the suite runs without the ``audit`` extra installed, without credentials, and
without a network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bandits.analyze.audit import (
    AuditError,
    audit_family,
    audit_task_set,
    compute_audit_run_id,
    load_audit_run,
    prompt_digest,
    save_audit_run,
)
from bandits.analyze.families import mine_task_set
from bandits.analyze.models import (
    CorpusAnalysis,
    FamilyAudit,
    FamilyAuditRun,
    TaskCandidate,
)
from bandits.store import DerivedStore


def _analysis(instructions: dict[str, str]) -> CorpusAnalysis:
    return CorpusAnalysis(
        corpus_id="corpus-test",
        source="chat-json",
        tasks=tuple(
            TaskCandidate(
                task_id=f"task-{trace_id}",
                trace_id=trace_id,
                instruction=instruction,
                trajectory_span_ids=(f"search:{trace_id}",),
            )
            for trace_id, instruction in instructions.items()
        ),
        evidence=(),
    )


def _distance(groups: dict[str, str]):
    """Zero inside a declared group, far outside it."""

    def distance(left: str, right: str) -> float:
        if left == right:
            return 0.0
        return 0.1 if groups.get(left) == groups.get(right) else 0.9

    return distance


def _predictor(**fields):
    def predict(*, members: str, question: str):
        return SimpleNamespace(**fields)

    return predict


def _family_of(task_set, size: int = 2):
    return next(f for f in task_set.families if len(f.trace_ids) >= size)


@pytest.fixture
def two_family_set():
    analysis = _analysis(
        {
            "t1": "refund order 1111",
            "t2": "refund order 2222",
            "t3": "reset the password for user 3333",
        }
    )
    groups = {
        "refund order <order_id>": "refund",
        "reset the password for user <user_id>": "reset",
    }
    task_set = mine_task_set(
        analysis,
        "analysis-test",
        distance=_distance(groups),
        similarity=0.6,
        held_out=0.0,
    )
    return analysis, task_set


def test_audit_reports_an_incoherent_family_with_a_split(two_family_set):
    analysis, task_set = two_family_set
    family = _family_of(task_set)
    audit = audit_family(
        family,
        analysis,
        predict=_predictor(
            coherent=False,
            outlier_trace_ids=["t2"],
            proposed_subgroups=[["t1"], ["t2"]],
            generated_name="Refund an eligible order",
            rationale="Two different refund policies.",
        ),
    )

    assert audit.coherent is False
    assert audit.proposed_subgroups == (("t1",), ("t2",))
    assert audit.generated_name == "Refund an eligible order"
    assert audit.prompt_digest == prompt_digest(audit.model)


def test_a_coherent_family_reports_no_outliers(two_family_set):
    analysis, task_set = two_family_set
    audit = audit_family(
        _family_of(task_set),
        analysis,
        # A model that calls a family coherent and still names outliers is
        # contradicting itself; the verdict it stated is what survives.
        predict=_predictor(
            coherent=True,
            outlier_trace_ids=["t1"],
            proposed_subgroups=[],
            generated_name="Refund an order",
            rationale="Same task throughout.",
        ),
    )

    assert audit.coherent is True
    assert audit.outlier_trace_ids == ()


def test_hallucinated_trace_ids_are_dropped_rather_than_losing_the_audit(two_family_set):
    analysis, task_set = two_family_set
    audit = audit_family(
        _family_of(task_set),
        analysis,
        predict=_predictor(
            coherent=False,
            outlier_trace_ids=["t1", "nonexistent-trace"],
            proposed_subgroups=[["t1"], ["t2", "also-not-real"]],
            generated_name="Refund",
            rationale="One member is different.",
        ),
    )

    assert audit.outlier_trace_ids == ("t1",)
    assert audit.proposed_subgroups == (("t1",), ("t2",))


def test_a_trace_claimed_by_two_subgroups_is_placed_once(two_family_set):
    analysis, task_set = two_family_set
    audit = audit_family(
        _family_of(task_set),
        analysis,
        predict=_predictor(
            coherent=False,
            outlier_trace_ids=[],
            proposed_subgroups=[["t1", "t2"], ["t2"]],
            generated_name="Refund",
            rationale="Overlapping proposal.",
        ),
    )

    # The second group is emptied by the first claim, so what remains is a
    # single group, which is the family it already is and not a split.
    assert audit.proposed_subgroups == ()


def test_a_split_proposal_overrides_a_coherent_verdict(two_family_set):
    analysis, task_set = two_family_set
    audit = audit_family(
        _family_of(task_set),
        analysis,
        predict=_predictor(
            coherent=True,
            outlier_trace_ids=[],
            proposed_subgroups=[["t1"], ["t2"]],
            generated_name="Refund",
            rationale="Proposing a split.",
        ),
    )

    assert audit.coherent is False


def test_a_backend_failure_becomes_an_audit_error(two_family_set):
    analysis, task_set = two_family_set

    def explode(*, members: str, question: str):
        raise RuntimeError("connection reset")

    with pytest.raises(AuditError, match="connection reset"):
        audit_family(_family_of(task_set), analysis, predict=explode)


def test_singleton_families_are_skipped_and_said_to_be_skipped(two_family_set):
    analysis, task_set = two_family_set
    run = audit_task_set(
        task_set,
        "taskset-test",
        analysis,
        predict=_predictor(
            coherent=True,
            outlier_trace_ids=[],
            proposed_subgroups=[],
            generated_name="Refund an order",
            rationale="One task.",
        ),
    )

    skipped = {s.family_id for s in run.skipped}
    singletons = {f.family_id for f in task_set.families if len(f.trace_ids) == 1}
    assert singletons
    assert singletons == skipped
    assert all("nothing to split" in s.reason for s in run.skipped)


def test_one_failing_family_does_not_lose_the_others():
    analysis = _analysis(
        {
            "t1": "refund order 1111",
            "t2": "refund order 2222",
            "t3": "reset the password for user 3333",
            "t4": "reset the password for user 4444",
        }
    )
    groups = {
        "refund order <order_id>": "refund",
        "reset the password for user <user_id>": "reset",
    }
    task_set = mine_task_set(
        analysis, "analysis-test", distance=_distance(groups), similarity=0.6, held_out=0.0
    )
    assert len(task_set.families) == 2

    calls: list[str] = []

    def flaky(*, members: str, question: str):
        calls.append(members)
        if len(calls) == 1:
            raise RuntimeError("rate limited")
        return SimpleNamespace(
            coherent=True,
            outlier_trace_ids=[],
            proposed_subgroups=[],
            generated_name="Reset a password",
            rationale="One task.",
        )

    run = audit_task_set(task_set, "taskset-test", analysis, predict=flaky)

    assert len(run.audits) == 1
    assert any("rate limited" in s.reason for s in run.skipped)


def test_auditing_an_unknown_family_is_refused(two_family_set):
    analysis, task_set = two_family_set
    with pytest.raises(ValueError, match="unknown family id"):
        audit_task_set(
            task_set,
            "taskset-test",
            analysis,
            predict=_predictor(
                coherent=True, outlier_trace_ids=[], proposed_subgroups=[], rationale="x"
            ),
            family_ids=["family-does-not-exist"],
        )


def test_mining_is_byte_identical_whether_or_not_the_audit_runs(two_family_set, tmp_path):
    """The acceptance criterion the whole design turns on."""
    analysis, task_set = two_family_set
    store = DerivedStore(tmp_path)
    before = task_set.model_dump_json()

    run = audit_task_set(
        task_set,
        "taskset-test",
        analysis,
        predict=_predictor(
            coherent=False,
            outlier_trace_ids=["t1"],
            proposed_subgroups=[["t1"], ["t2"]],
            generated_name="Refund an eligible order",
            rationale="Two tasks in one family.",
        ),
    )
    save_audit_run(run, store)

    assert task_set.model_dump_json() == before
    # And the audit landed beside the task set rather than inside it.
    assert store.list(kind="family_audit")
    assert "family_audit" not in before


def test_the_audit_never_merges_or_renames_a_family(two_family_set):
    analysis, task_set = two_family_set
    family = _family_of(task_set)
    audit = audit_family(
        family,
        analysis,
        predict=_predictor(
            coherent=False,
            outlier_trace_ids=[],
            proposed_subgroups=[["t1"], ["t2"]],
            generated_name="Refund an eligible order",
            rationale="Split proposed.",
        ),
    )

    # A generated name is presentation only: identity stays mechanical.
    assert family.family_id.startswith("family-")
    assert audit.generated_name not in family.family_id
    assert family.descriptor != audit.generated_name
    # Nothing in the contract can express a merge.
    assert not hasattr(audit, "merge_with")


def test_an_audit_run_round_trips_through_the_store(two_family_set, tmp_path):
    analysis, task_set = two_family_set
    store = DerivedStore(tmp_path)
    run = audit_task_set(
        task_set,
        "taskset-test",
        analysis,
        predict=_predictor(
            coherent=True,
            outlier_trace_ids=[],
            proposed_subgroups=[],
            generated_name="Refund an order",
            rationale="One task.",
        ),
    )
    envelope = save_audit_run(run, store)

    assert envelope.artifact_id == compute_audit_run_id(run)
    assert envelope.parent_artifact_id == "taskset-test"
    assert load_audit_run(envelope.artifact_id, store) == run


def test_a_subgroup_naming_one_trace_twice_is_refused():
    with pytest.raises(ValidationError, match="two subgroups"):
        FamilyAudit(
            family_id="family-1",
            coherent=False,
            proposed_subgroups=(("t1", "t2"), ("t2",)),
            rationale="overlapping",
            model="m",
            prompt_digest="d",
        )


def test_a_coherent_family_may_not_also_be_split():
    with pytest.raises(ValidationError, match="still"):
        FamilyAudit(
            family_id="family-1",
            coherent=True,
            proposed_subgroups=(("t1",), ("t2",)),
            rationale="contradictory",
            model="m",
            prompt_digest="d",
        )


def test_a_single_subgroup_is_not_a_split():
    with pytest.raises(ValidationError, match="single subgroup"):
        FamilyAudit(
            family_id="family-1",
            coherent=False,
            proposed_subgroups=(("t1", "t2"),),
            rationale="not a split",
            model="m",
            prompt_digest="d",
        )


def test_an_audit_without_a_rationale_is_refused():
    with pytest.raises(ValidationError, match="no rationale"):
        FamilyAudit(
            family_id="family-1",
            coherent=True,
            rationale="   ",
            model="m",
            prompt_digest="d",
        )


def test_a_family_cannot_be_both_audited_and_skipped():
    audit = FamilyAudit(
        family_id="family-1", coherent=True, rationale="fine", model="m", prompt_digest="d"
    )
    with pytest.raises(ValidationError, match="both audited and skipped"):
        FamilyAuditRun(
            task_set_id="ts1",
            audits=(audit,),
            skipped=({"family_id": "family-1", "reason": "no"},),
            model="m",
        )


def test_the_prompt_digest_changes_with_the_model():
    assert prompt_digest("model-a") != prompt_digest("model-b")


def test_the_member_view_shows_the_tools_an_episode_actually_called():
    """The row the model reads must match the corpus it describes.

    Read over a real corpus rather than a hand-built analysis: the shape of a
    span id is exactly what this once got wrong, and a fixture that invents its
    own ids cannot catch that. `addr-1` calls three tools across eleven spans.
    """
    from pathlib import Path

    from bandits.analyze import analyze_corpus
    from bandits.analyze.audit import _member_view
    from bandits.ingest import load_corpus
    from bandits.traces import SpanKind

    corpus = load_corpus(Path("tests/fixtures/traces.support.otlp.jsonl"), "otlp")
    analysis = analyze_corpus(corpus)
    task_set = mine_task_set(
        analysis,
        "analysis-test",
        distance=lambda left, right: 0.0 if left == right else 0.9,
        similarity=0.6,
        held_out=0.0,
    )

    rows = {
        row["trace_id"]: row
        for family in task_set.families
        for row in _member_view(family, analysis)
    }

    for trace in corpus.traces:
        row = rows[trace.trace_id]
        assert row["tool_names"] == sorted({s.name for s in trace.spans if s.kind is SpanKind.TOOL})
        assert row["span_count"] == len(trace.spans)

    assert rows["addr-1"]["tool_names"] == [
        "lookup_order",
        "override_address_policy",
        "validate_address",
    ]
    assert rows["addr-1"]["span_count"] == 11


def test_every_key_the_prompt_promises_is_on_the_row(two_family_set):
    """A key the prompt names but the row omits is a KeyError in model-written
    REPL code, spent silently against the iteration budget."""
    import re

    from bandits.analyze.audit import _INSTRUCTION, _member_view

    analysis, task_set = two_family_set
    promised = re.search(r"each with keys: ([^.]+)\.", _INSTRUCTION).group(1)
    names = {re.match(r"\s*(\w+)", part).group(1) for part in promised.split(",")}

    row = _member_view(_family_of(task_set), analysis)[0]

    assert names <= set(row)
