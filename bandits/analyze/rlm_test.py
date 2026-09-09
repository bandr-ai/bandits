"""Tests for the RLM mining path.

Nothing here reaches a model, a REPL sandbox, or a network: every predictor is
injected, so the suite runs without the ``audit`` extra and without credentials.

Most of these exist to prove the path stays honest under a model that misbehaves.
A language model writing the taxonomy will hallucinate trace ids, contradict its
own status fields, propose families with no stated outcome, and — worst — force
a trace into a family rather than admit nothing fits. Each of those has a test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bandits.analyze.rlm_assign import (
    AssignmentError,
    _parse_result,
    assign_traces,
    load_assignment_run,
    save_assignment_run,
)
from bandits.analyze.rlm_audit import (
    FreezeRefused,
    audit_taxonomy,
    compute_taxonomy_id,
    freeze_taxonomy,
    resolve_findings,
)
from bandits.analyze.rlm_corpus import ReadOnlyCorpus, build_view
from bandits.analyze.rlm_mine import (
    MiningError,
    _parse_contract,
    compute_draft_id,
    load_draft,
    mine_taxonomy,
    save_draft,
)
from bandits.analyze.rlm_models import (
    AssignmentRun,
    AssignmentStatus,
    Budget,
    ChunkResult,
    FamilyContract,
    FrozenTaxonomy,
    Operation,
    StopReason,
    TaxonomyAudit,
    TaxonomyDraft,
    TaxonomyOperation,
    TraceAssignment,
    TraceView,
)
from bandits.analyze.rlm_stability import compare_runs, save_stability_report
from bandits.analyze.rlm_taskset import MaterializationError, materialize_task_set
from bandits.store import DerivedStore
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, UserTurn


def _trace(trace_id: str, *messages: str) -> Trace:
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    return Trace(
        trace_id=trace_id,
        source="chat-json",
        source_digest="0" * 64,
        task=messages[0] if messages else None,
        user_turns=tuple(UserTurn(text=text) for text in messages),
        spans=(
            Span(
                span_id=f"{trace_id}:span-0",
                kind=SpanKind.MODEL,
                name="model",
                started_at=moment,
                ended_at=moment,
            ),
        ),
    )


def _corpus(*traces: Trace) -> TraceCorpus:
    return TraceCorpus(source="chat-json", traces=traces)


def _contract(contract_id: str, definition: str = "refund an eligible order") -> FamilyContract:
    return FamilyContract(
        contract_id=contract_id,
        name="Refund an order",
        definition=definition,
        required_outcome_shape=("the order is refunded and the balance reflects it",),
    )


# --- input boundary ----------------------------------------------------------


def test_view_keeps_only_user_messages_in_order() -> None:
    view = build_view(
        _trace("t1", "book a flight", "actually make it business"), TraceView.USER_MESSAGES
    )
    assert view.messages == ("book a flight", "actually make it business")
    assert view.readable


def test_first_message_arm_drops_later_turns() -> None:
    """The two arms must differ, or the second arm measures nothing."""
    view = build_view(
        _trace("t1", "book a flight", "actually make it business"),
        TraceView.FIRST_USER_MESSAGE,
    )
    assert view.messages == ("book a flight",)


def test_trace_without_recorded_turns_is_unreadable_not_reconstructed() -> None:
    """A declared task must never be smuggled in as if it were a user message."""
    trace = _trace("t1")
    assert trace.task is None
    view = build_view(trace.replace(task="do the thing"), TraceView.USER_MESSAGES)
    assert not view.readable
    assert view.messages == ()
    assert "no user-role messages" in view.unreadable_reason


def test_readable_view_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="unreadable, not a trace"):
        build_view(_trace("t1"), TraceView.USER_MESSAGES).replace(readable=True)


def test_corpus_exposes_only_generic_operations() -> None:
    """The interface is the guarantee; anything reaching a span would break it."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "a"), _trace("t2", "b")))
    assert corpus.count_traces() == 2
    assert corpus.list_trace_ids(offset=1) == ("t2",)
    assert corpus.get_user_messages("t1").messages == ("a",)
    assert [v.trace_id for v in corpus.get_user_message_batch(["t2", "t1"])] == ["t2", "t1"]
    assert not hasattr(corpus, "get_spans")
    assert not hasattr(corpus, "get_outcome")


def test_sampling_is_reproducible_and_respects_exclusions() -> None:
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "ask") for i in range(10))))
    first = corpus.sample_trace_ids(3, seed=7)
    assert first == corpus.sample_trace_ids(3, seed=7)
    assert "t0" not in corpus.sample_trace_ids(5, seed=7, exclude=("t0",))


def test_sampling_does_not_disturb_the_global_generator() -> None:
    """Two runs must be comparable regardless of what else drew from random."""
    import random

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "ask") for i in range(10))))
    random.seed(1)
    before = random.random()
    random.seed(1)
    corpus.sample_trace_ids(3, seed=99)
    assert random.random() == before


def test_unreadable_traces_stay_visible() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "ask"), _trace("t2")))
    assert corpus.readable_trace_ids() == ("t1",)
    assert corpus.unreadable_trace_ids() == ("t2",)


# --- contracts ---------------------------------------------------------------


def test_contract_without_outcome_shape_is_rejected() -> None:
    """A family with no stated outcome is a topic, which is the failure mode."""
    with pytest.raises(ValidationError, match="names a topic rather than"):
        FamilyContract(contract_id="c1", name="Refunds", definition="refund things")


def test_contract_cannot_both_support_and_refute_with_one_trace() -> None:
    with pytest.raises(ValidationError, match="both support and counterexample"):
        _contract("c1").replace(supporting_trace_ids=("t1",), counterexample_trace_ids=("t1",))


def test_fingerprint_ignores_id_name_and_evidence() -> None:
    """Cross-run recurrence is measured on the claim, never on the generated name."""
    left = _contract("c1").replace(supporting_trace_ids=("t1",))
    right = _contract("c2").replace(name="Give money back", supporting_trace_ids=("t9",))
    assert left.fingerprint() == right.fingerprint()
    assert _contract("c3", "cancel a subscription").fingerprint() != left.fingerprint()


def test_marking_a_trace_is_not_a_taxonomy_mutation() -> None:
    """Marking records coverage; only these four change what the taxonomy claims."""
    from bandits.analyze.rlm_models import TaxonomyOperation

    def op(operation: Operation, **kw):
        return TaxonomyOperation(operation=operation, rationale="because", **kw)

    assert not op(Operation.MARK_AMBIGUOUS).mutating
    assert not op(Operation.MARK_UNCOVERED).mutating
    assert not op(Operation.KEEP).mutating
    assert op(Operation.CREATE).mutating
    assert op(Operation.SPLIT).mutating
    assert op(Operation.REVISE).mutating
    assert not op(Operation.REVISE, material=False).mutating


def test_operation_without_rationale_is_rejected() -> None:
    from bandits.analyze.rlm_models import TaxonomyOperation

    with pytest.raises(ValidationError, match="no rationale"):
        TaxonomyOperation(operation=Operation.CREATE, rationale="  ")


# --- discovery ---------------------------------------------------------------


class _ScriptedMiner:
    """A predictor replaying fixed chunk outputs, then going quiet.

    Going quiet is what lets convergence be tested: once it stops proposing
    operations, the loop must notice two clean sweeps and freeze on its own.
    """

    def __init__(self, script: list[dict]) -> None:
        self.script = script
        self.calls = 0
        self.seen_chunks: list[str] = []

    def __call__(self, *, chunk: str, taxonomy: str, question: str):
        self.seen_chunks.append(chunk)
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        import json

        traces = [row["trace_id"] for row in json.loads(chunk)]
        return SimpleNamespace(
            contracts=step.get("contracts", []),
            operations=step.get("operations", []),
            assignments={t: step["contract_id"] for t in traces} if step.get("contract_id") else {},
            ambiguous_trace_ids=step.get("ambiguous", []),
            uncovered_trace_ids=step.get("uncovered", []),
        )


_RAW_CONTRACT = {
    "contract_id": "c1",
    "name": "Refund an order",
    "definition": "refund an eligible order",
    "required_outcome_shape": ["the order is refunded"],
}


def test_discovery_runs_every_requested_pass_then_pauses() -> None:
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund my order") for i in range(6))))
    predict = _ScriptedMiner(
        [
            {
                "contracts": [_RAW_CONTRACT],
                "contract_id": "c1",
                "operations": [{"operation": "CREATE", "rationale": "these are all refunds"}],
            },
            {"contract_id": "c1"},
        ]
    )
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=3)
    assert draft.stop_reason is StopReason.PASSES_COMPLETE
    assert draft.complete
    assert draft.awaiting_review
    assert draft.completed_passes == 2
    assert [c.contract_id for c in draft.contracts] == ["c1"]


def test_every_eligible_trace_is_read_once_in_every_pass() -> None:
    """The bug this replaced: two quiet chunks could end a run having re-read
    a fraction of the corpus while reporting convergence."""
    seen: dict[str, int] = {}

    def predict(*, chunk: str, taxonomy: str, question: str):
        import json

        rows = json.loads(chunk)
        for row in rows:
            seen[row["trace_id"]] = seen.get(row["trace_id"], 0) + 1
        return SimpleNamespace(
            contracts=[_RAW_CONTRACT],
            operations=[],
            assignments={r["trace_id"]: "c1" for r in rows},
            ambiguous_trace_ids=[],
            uncovered_trace_ids=[],
        )

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(40))))
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=10)
    assert draft.completed_passes == 2
    assert set(seen) == {f"t{i}" for i in range(40)}
    assert all(count == 2 for count in seen.values()), seen
    for result in draft.passes:
        assert result.complete
        assert len(result.trace_ids) == 40


def test_each_pass_reshuffles_under_its_own_recorded_seed() -> None:
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(20))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=5,
        seed=7,
    )
    assert [p.seed for p in draft.passes] == [7, 8]
    assert draft.passes[0].trace_ids != draft.passes[1].trace_ids


def test_a_partial_pass_never_counts_toward_the_schedule() -> None:
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(40))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=10,
        budget=Budget(max_iterations=2),
    )
    assert draft.completed_passes == 0
    assert not draft.complete
    assert not draft.passes[0].complete


def test_a_failed_chunk_does_not_shrink_a_pass_coverage() -> None:
    """A provider blip must not silently cost a pass part of the corpus."""
    calls = {"n": 0}
    seen: dict[str, int] = {}

    def predict(*, chunk: str, taxonomy: str, question: str):
        import json

        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient outage")
        rows = json.loads(chunk)
        for row in rows:
            seen[row["trace_id"]] = seen.get(row["trace_id"], 0) + 1
        return SimpleNamespace(
            contracts=[_RAW_CONTRACT],
            operations=[],
            assignments={r["trace_id"]: "c1" for r in rows},
            ambiguous_trace_ids=[],
            uncovered_trace_ids=[],
        )

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(30))))
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=10)
    assert draft.completed_passes == 2
    assert all(count == 2 for count in seen.values())


def test_pass_diff_reports_what_the_second_look_changed() -> None:
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
    )
    diff = draft.pass_diff()
    assert diff
    assert any("placed differently" in line for line in diff)
    assert any("not itself a convergence test" in line for line in diff)


def test_a_complete_run_still_refuses_to_call_itself_converged() -> None:
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
    )
    assert any("not a convergence test" in limit for limit in draft.limitations)


def test_a_run_that_never_settles_stops_on_budget_and_is_incomplete() -> None:
    """A taxonomy that ran out of money must never read as one that converged."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    predict = _ScriptedMiner(
        [
            {
                "contracts": [_RAW_CONTRACT],
                "contract_id": "c1",
                "operations": [{"operation": "REVISE", "rationale": "still moving"}],
            }
        ]
    )
    draft = mine_taxonomy(
        corpus, "analysis-1", predict=predict, chunk_size=2, budget=Budget(max_iterations=3)
    )
    assert draft.stop_reason is StopReason.MAX_ITERATIONS
    assert not draft.complete
    assert any("before completing its" in limit for limit in draft.limitations)


def test_two_chunks_are_not_two_passes() -> None:
    """The precise defect: chunk cleanliness must not substitute for coverage."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(20))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=5,
    )
    # Four chunks per pass, two passes: never fewer, however quiet they were.
    assert len(draft.chunks) == 8
    assert draft.completed_passes == 2


def test_contracts_with_no_outcome_shape_are_dropped_and_reported() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))
    predict = _ScriptedMiner(
        [{"contracts": [{"contract_id": "c1", "name": "Stuff", "definition": "things"}]}]
    )
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    assert draft.contracts == ()
    assert any("the parser refused" in limit for limit in draft.limitations)
    # The rejected contract is kept verbatim, so a failed run can be diagnosed
    # without paying for another one.
    assert any(chunk.dropped_contracts for chunk in draft.chunks)


def test_hallucinated_trace_ids_are_dropped_rather_than_losing_the_chunk() -> None:
    contract = _parse_contract(
        {**_RAW_CONTRACT, "supporting_trace_ids": ["t1", "ghost", "t1"]},
        known_traces={"t1"},
    )
    assert contract is not None
    assert contract.supporting_trace_ids == ("t1",)


def test_a_trace_marked_unplaced_is_never_also_assigned() -> None:
    """Forcing a match is the one thing this stage must never do."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "something odd")))
    predict = _ScriptedMiner(
        [{"contracts": [_RAW_CONTRACT], "contract_id": "c1", "uncovered": ["t2"]}]
    )
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    assert "t2" in draft.uncovered_trace_ids
    assert all("t2" not in chunk.assignments for chunk in draft.chunks)


def test_a_failed_chunk_is_recorded_and_the_loop_continues() -> None:
    calls = {"n": 0}

    def predict(*, chunk: str, taxonomy: str, question: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the provider fell over")
        return SimpleNamespace(
            contracts=[_RAW_CONTRACT],
            operations=[],
            assignments={},
            ambiguous_trace_ids=[],
            uncovered_trace_ids=[],
        )

    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    failed = [c for c in draft.chunks if c.status == "error"]
    assert len(failed) == 1
    assert "fell over" in failed[0].error
    assert len(draft.chunks) > 1


def test_mining_an_unreadable_corpus_refuses_rather_than_inventing() -> None:
    with pytest.raises(MiningError, match="nothing to mine"):
        mine_taxonomy(
            ReadOnlyCorpus(_corpus(_trace("t1"))), "analysis-1", predict=_ScriptedMiner([{}])
        )


def test_chunks_mix_unseen_traces_with_review() -> None:
    """Chunk boundaries must not become family boundaries."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(8))))
    predict = _ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}])
    mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=4)
    import json

    later = [json.loads(c) for c in predict.seen_chunks[2:]]
    assert any(any(row["status"] != "unseen" for row in chunk) for chunk in later), (
        "later chunks never revisited an already-seen trace"
    )


def test_draft_round_trips_through_the_store(tmp_path) -> None:
    store = DerivedStore(tmp_path)
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
    )
    envelope = save_draft(draft, store)
    assert envelope.parent_artifact_id == "analysis-1"
    assert load_draft(envelope.artifact_id, store) == draft
    assert compute_draft_id(draft) == envelope.artifact_id


# --- adversarial audit and freezing -----------------------------------------


def _draft(**overrides):
    from bandits.analyze.rlm_models import TaxonomyDraft

    base = dict(
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        seed=42,
        contracts=(_contract("c1"),),
        stop_reason=StopReason.PASSES_COMPLETE,
        completed_passes=2,
        requested_passes=2,
        budget=Budget(),
        model="test-model",
        prompt_digest="digest",
    )
    return TaxonomyDraft(**{**base, **overrides})


def test_audit_challenges_every_contract() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))

    def predict(*, contract: str, members: str, outsiders: str, question: str):
        return SimpleNamespace(
            recommendation="split",
            least_compatible_pair=[],
            strongest_outsider_trace_id="",
            topical_only=True,
            rationale="these share a topic and need different verifiers",
        )

    audit = audit_taxonomy(_draft(), "draft-1", corpus, predict=predict)
    assert len(audit.findings) == 1
    assert audit.findings[0].topical_only
    assert audit.unresolved()


def test_an_unparseable_recommendation_becomes_uncertain_never_keep() -> None:
    """A parse failure must not silently clear the freeze gate."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))

    def predict(**_):
        return SimpleNamespace(recommendation="looks fine to me!", rationale="ok")

    audit = audit_taxonomy(_draft(), "draft-1", corpus, predict=predict)
    assert audit.findings[0].recommendation == "uncertain"


def test_a_failed_contract_audit_is_uncertain_not_absent() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))

    def predict(**_):
        raise RuntimeError("provider down")

    audit = audit_taxonomy(_draft(), "draft-1", corpus, predict=predict)
    assert audit.findings[0].recommendation == "uncertain"
    assert "provider down" in audit.findings[0].rationale


def test_freeze_refuses_while_a_split_is_unresolved() -> None:
    from bandits.analyze.rlm_models import AuditFinding

    audit = TaxonomyAudit(
        draft_id="draft-1",
        findings=(AuditFinding(contract_id="c1", recommendation="split", rationale="two tasks"),),
        model="m",
        prompt_digest="d",
    )
    with pytest.raises(FreezeRefused, match="have not been resolved"):
        freeze_taxonomy(_draft(), "draft-1", audit=audit, audit_id="audit-1")


def test_resolved_findings_permit_the_freeze() -> None:
    from bandits.analyze.rlm_models import AuditFinding

    audit = TaxonomyAudit(
        draft_id="draft-1",
        findings=(AuditFinding(contract_id="c1", recommendation="split", rationale="two tasks"),),
        model="m",
        prompt_digest="d",
    )
    resolved = resolve_findings(audit, {"c1": "split into c1 and c2 in the next sweep"})
    assert not resolved.unresolved()
    taxonomy = freeze_taxonomy(_draft(), "draft-1", audit=resolved, audit_id="audit-1")
    assert taxonomy.complete


def test_a_resolution_must_say_how() -> None:
    from bandits.analyze.rlm_models import AuditFinding

    with pytest.raises(ValidationError, match="indistinguishable from ignoring it"):
        AuditFinding(contract_id="c1", recommendation="split", rationale="two", resolved=True)


def test_forcing_a_freeze_records_that_it_was_forced() -> None:
    from bandits.analyze.rlm_models import AuditFinding

    audit = TaxonomyAudit(
        draft_id="draft-1",
        findings=(AuditFinding(contract_id="c1", recommendation="split", rationale="two"),),
        model="m",
        prompt_digest="d",
    )
    taxonomy = freeze_taxonomy(_draft(), "draft-1", audit=audit, audit_id="a1", force=True)
    assert any("unresolved audit finding" in limit for limit in taxonomy.limitations)


def test_freezing_without_an_audit_says_so() -> None:
    taxonomy = freeze_taxonomy(_draft(), "draft-1")
    assert taxonomy.audit_id is None
    assert any("no adversarial audit" in limit for limit in taxonomy.limitations)


def test_an_incomplete_draft_freezes_but_never_reads_as_converged() -> None:
    taxonomy = freeze_taxonomy(_draft(stop_reason=StopReason.MAX_SECONDS), "draft-1")
    assert not taxonomy.complete
    assert any("never converg" in x or "rather than converging" in x for x in taxonomy.limitations)


def test_taxonomy_id_tracks_contract_wording() -> None:
    """An assignment naming a taxonomy id must be naming exact wording."""
    first = freeze_taxonomy(_draft(), "draft-1")
    reworded = freeze_taxonomy(
        _draft(contracts=(_contract("c1", "refund an order the policy allows"),)), "draft-1"
    )
    assert compute_taxonomy_id(first) != compute_taxonomy_id(reworded)


def test_an_empty_taxonomy_cannot_be_frozen() -> None:
    with pytest.raises(ValidationError, match="cannot be assigned against"):
        freeze_taxonomy(_draft(contracts=()), "draft-1")


# --- fresh assignment --------------------------------------------------------


def _taxonomy(*contracts: FamilyContract) -> FrozenTaxonomy:
    return FrozenTaxonomy(
        draft_id="draft-1",
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        contracts=contracts or (_contract("c1"),),
        complete=True,
    )


def test_assignment_classifies_every_trace() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))

    def predict(*, taxonomy: str, batch: str, question: str):
        import json

        return SimpleNamespace(
            results=[
                {
                    "trace_id": row["trace_id"],
                    "matching_contract_ids": ["c1"],
                    "primary_contract_id": "c1",
                    "status": "assigned",
                    "reason": "asks for a refund",
                }
                for row in json.loads(batch)
            ]
        )

    run = assign_traces(_taxonomy(), "tax-1", corpus, predict=predict)
    assert len(run.assignments) == 2
    assert run.members() == {"c1": ("t1", "t2")}


def test_two_matches_stay_ambiguous_and_pick_no_primary() -> None:
    """Ambiguity is preserved for review rather than broken by a tiebreak."""
    parsed = _parse_result(
        {
            "trace_id": "t1",
            "matching_contract_ids": ["c1", "c2"],
            "primary_contract_id": "c1",
            "status": "assigned",
            "reason": "both fit",
        },
        known_contracts={"c1", "c2"},
    )
    assert parsed is not None
    assert parsed.status is AssignmentStatus.AMBIGUOUS
    assert parsed.primary_contract_id is None


def test_no_match_is_uncovered_even_when_the_model_says_assigned() -> None:
    parsed = _parse_result(
        {"trace_id": "t1", "matching_contract_ids": [], "status": "assigned", "reason": "eh"},
        known_contracts={"c1"},
    )
    assert parsed is not None
    assert parsed.status is AssignmentStatus.UNCOVERED
    assert parsed.matching_contract_ids == ()


def test_a_match_naming_an_unknown_contract_is_dropped() -> None:
    parsed = _parse_result(
        {"trace_id": "t1", "matching_contract_ids": ["ghost"], "reason": "x"},
        known_contracts={"c1"},
    )
    assert parsed is not None
    assert parsed.status is AssignmentStatus.UNCOVERED


def test_a_trace_the_model_skipped_becomes_uncovered_not_missing() -> None:
    """A missing trace would silently shrink the coverage denominator."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))

    def predict(**_):
        return SimpleNamespace(
            results=[
                {
                    "trace_id": "t1",
                    "matching_contract_ids": ["c1"],
                    "primary_contract_id": "c1",
                    "reason": "refund",
                }
            ]
        )

    run = assign_traces(_taxonomy(), "tax-1", corpus, predict=predict)
    assert {a.trace_id for a in run.assignments} == {"t1", "t2"}
    assert run.by_status(AssignmentStatus.UNCOVERED)[0].trace_id == "t2"
    assert any("no result from the model" in limit for limit in run.limitations)


def test_unreadable_traces_are_classified_as_unreadable() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2")))

    def predict(**_):
        return SimpleNamespace(
            results=[
                {
                    "trace_id": "t1",
                    "matching_contract_ids": ["c1"],
                    "primary_contract_id": "c1",
                    "reason": "refund",
                }
            ]
        )

    run = assign_traces(_taxonomy(), "tax-1", corpus, predict=predict)
    assert run.by_status(AssignmentStatus.UNREADABLE)[0].trace_id == "t2"


def test_a_failed_batch_does_not_lose_its_traces() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))

    def predict(**_):
        raise RuntimeError("provider down")

    run = assign_traces(_taxonomy(), "tax-1", corpus, predict=predict, batch_size=2)
    assert len(run.assignments) == 2
    assert all(a.status is AssignmentStatus.UNCOVERED for a in run.assignments)
    assert any("provider down" in limit for limit in run.limitations)


def test_assigning_across_views_is_refused() -> None:
    """The two arms are different experiments and must not be mixed."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")), view=TraceView.FIRST_USER_MESSAGE)
    with pytest.raises(AssignmentError, match="different experiments"):
        assign_traces(_taxonomy(), "tax-1", corpus, predict=lambda **_: None)


def test_ambiguous_traces_are_excluded_from_members() -> None:
    run = AssignmentRun(
        taxonomy_id="tax-1",
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        assignments=(
            TraceAssignment(
                trace_id="t1",
                matching_contract_ids=("c1",),
                primary_contract_id="c1",
                status=AssignmentStatus.ASSIGNED,
                reason="refund",
            ),
            TraceAssignment(
                trace_id="t2",
                matching_contract_ids=("c1", "c2"),
                status=AssignmentStatus.AMBIGUOUS,
                reason="both",
            ),
        ),
        model="m",
        prompt_digest="d",
    )
    assert run.members() == {"c1": ("t1",)}


def test_assignment_contract_rejects_a_status_that_disagrees_with_its_matches() -> None:
    with pytest.raises(ValidationError, match="assigned to nothing"):
        TraceAssignment(trace_id="t1", status=AssignmentStatus.ASSIGNED, reason="x")
    with pytest.raises(ValidationError, match="fewer than two matches"):
        TraceAssignment(
            trace_id="t1",
            matching_contract_ids=("c1",),
            status=AssignmentStatus.AMBIGUOUS,
            reason="x",
        )
    with pytest.raises(ValidationError, match="still names a contract"):
        TraceAssignment(
            trace_id="t1",
            matching_contract_ids=("c1",),
            status=AssignmentStatus.UNCOVERED,
            reason="x",
        )


def test_assignment_run_round_trips(tmp_path) -> None:
    store = DerivedStore(tmp_path)
    run = AssignmentRun(
        taxonomy_id="tax-1",
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        assignments=(
            TraceAssignment(
                trace_id="t1",
                matching_contract_ids=("c1",),
                primary_contract_id="c1",
                status=AssignmentStatus.ASSIGNED,
                reason="refund",
            ),
        ),
        model="m",
        prompt_digest="d",
    )
    envelope = save_assignment_run(run, store)
    assert envelope.parent_artifact_id == "tax-1"
    assert load_assignment_run(envelope.artifact_id, store) == run


# --- stability ---------------------------------------------------------------


def _run(taxonomy_id: str, groups: dict[str, list[str]], unplaced: dict[str, str] | None = None):
    assignments = [
        TraceAssignment(
            trace_id=trace_id,
            matching_contract_ids=(contract_id,),
            primary_contract_id=contract_id,
            status=AssignmentStatus.ASSIGNED,
            reason="grouped",
        )
        for contract_id, traces in groups.items()
        for trace_id in traces
    ]
    for trace_id, status in (unplaced or {}).items():
        assignments.append(
            TraceAssignment(
                trace_id=trace_id,
                matching_contract_ids=("c1", "c2") if status == "ambiguous" else (),
                status=AssignmentStatus(status),
                reason="unplaced",
            )
        )
    return AssignmentRun(
        taxonomy_id=taxonomy_id,
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        assignments=tuple(assignments),
        model="m",
        prompt_digest="d",
    )


def test_identical_grouping_under_different_names_is_perfect_agreement() -> None:
    """Names differ between runs and carry no information; partners do."""
    left = _run("tax-1", {"c1": ["t1", "t2"], "c2": ["t3"]})
    right = _run("tax-2", {"zzz": ["t1", "t2"], "yyy": ["t3"]})
    report = compare_runs([left, right], ["r1", "r2"], analysis_id="analysis-1")
    assert report.pairwise_agreement == 1.0
    assert report.stable_assignment_fraction == 1.0
    assert report.disagreements == ()


def test_a_split_disagreement_is_reported_as_a_contested_pair() -> None:
    left = _run("tax-1", {"c1": ["t1", "t2"]})
    right = _run("tax-2", {"c1": ["t1"], "c2": ["t2"]})
    report = compare_runs([left, right], ["r1", "r2"], analysis_id="analysis-1")
    assert report.pairwise_agreement == 0.0
    assert report.disagreements[0].trace_ids == ("t1", "t2")
    assert report.disagreements[0].together == 1
    assert report.disagreements[0].apart == 1


def test_a_run_that_left_a_trace_unplaced_does_not_vote_it_apart() -> None:
    """Declining to place is not the same claim as separating."""
    left = _run("tax-1", {"c1": ["t1", "t2"]})
    right = _run("tax-2", {"c1": ["t1"]}, unplaced={"t2": "uncovered"})
    report = compare_runs([left, right], ["r1", "r2"], analysis_id="analysis-1")
    assert report.disagreements == ()
    assert report.unplaced[0].trace_id == "t2"


def test_unplaced_traces_count_against_stability() -> None:
    left = _run("tax-1", {"c1": ["t1", "t2"]})
    right = _run("tax-2", {"c1": ["t1", "t2"]}, unplaced={"t3": "ambiguous"})
    report = compare_runs([left, right], ["r1", "r2"], analysis_id="analysis-1")
    assert report.stable_assignment_fraction < 1.0


def test_runs_sharing_a_taxonomy_are_refused() -> None:
    """That measures classifier repeatability, not discovery stability."""
    left = _run("tax-1", {"c1": ["t1"]})
    right = _run("tax-1", {"c1": ["t1"]})
    with pytest.raises(ValueError, match="not whether independent discovery"):
        compare_runs([left, right], ["r1", "r2"], analysis_id="analysis-1")


def test_a_single_run_cannot_report_stability() -> None:
    with pytest.raises(ValueError, match="at least two runs"):
        compare_runs([_run("tax-1", {"c1": ["t1"]})], ["r1"], analysis_id="analysis-1")


def test_recurring_contracts_are_counted_on_meaning_not_name() -> None:
    left = _taxonomy(_contract("c1"))
    right = _taxonomy(_contract("zz").replace(name="Money back"))
    report = compare_runs(
        [_run("tax-1", {"c1": ["t1"]}), _run("tax-2", {"zz": ["t1"]})],
        ["r1", "r2"],
        analysis_id="analysis-1",
        taxonomies=[left, right],
    )
    assert report.recurring_contracts[0][1] == 2


def test_stability_report_round_trips(tmp_path) -> None:
    store = DerivedStore(tmp_path)
    report = compare_runs(
        [_run("tax-1", {"c1": ["t1", "t2"]}), _run("tax-2", {"c1": ["t1", "t2"]})],
        ["r1", "r2"],
        analysis_id="analysis-1",
    )
    envelope = save_stability_report(report, store)
    assert envelope.parent_artifact_id == "analysis-1"


def test_a_report_cannot_name_the_same_run_twice() -> None:
    from bandits.analyze.rlm_stability import StabilityReport

    with pytest.raises(ValidationError, match="names the same run twice"):
        StabilityReport(
            analysis_id="a1",
            assignment_run_ids=("r1", "r1"),
            runs=2,
            stable_assignment_fraction=1.0,
            pairwise_agreement=1.0,
        )


# --- materialization ---------------------------------------------------------


def test_materialization_fabricates_no_similarity_geometry() -> None:
    """The task set's whole claim is that no distance was computed."""
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1", "t2"]})
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1")
    assert task_set.clustering is not None
    assert task_set.clustering.backend == "rlm-user-messages"
    assert task_set.clustering.embedding_model is None
    family = task_set.families[0]
    assert family.coherence is None
    assert family.medoid_trace_id in family.trace_ids
    assert any("no embedding distance" in limit for limit in family.limitations)


def test_unresolved_traces_never_enter_a_family() -> None:
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1"]}, unplaced={"t2": "ambiguous"})
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1")
    assert [t for f in task_set.families for t in f.trace_ids] == ["t1"]
    assert any("until a reviewer resolves them" in limit for limit in task_set.limitations)


def test_materializing_against_the_wrong_taxonomy_is_refused() -> None:
    run = _run("tax-does-not-match", {"c1": ["t1"]})
    with pytest.raises(MaterializationError, match="different taxonomy"):
        materialize_task_set(run, _taxonomy(), corpus_id="corpus-1")


def test_a_run_that_placed_nothing_yields_no_task_set() -> None:
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {}, unplaced={"t1": "uncovered"})
    with pytest.raises(MaterializationError, match="no family to materialize"):
        materialize_task_set(run, taxonomy, corpus_id="corpus-1")


def test_held_out_split_keeps_every_member_on_one_side() -> None:
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1", "t2", "t3", "t4"]})
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1", held_out=0.5)
    family = task_set.families[0]
    assert set(family.fit_trace_ids) | set(family.held_out_trace_ids) == set(family.trace_ids)
    assert not set(family.fit_trace_ids) & set(family.held_out_trace_ids)


# --- path F: the full-trajectory arm ----------------------------------------


def _tool_trace(trace_id: str, *, output: dict, message: str = "refund my order") -> Trace:
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    return Trace(
        trace_id=trace_id,
        source="chat-json",
        source_digest="0" * 64,
        user_turns=(UserTurn(text=message),),
        spans=(
            Span(
                span_id=f"{trace_id}:s1",
                kind=SpanKind.TOOL,
                name="refund",
                started_at=moment,
                ended_at=moment,
                arguments={"order": "A1"},
                output=output,
            ),
        ),
    )


def test_path_f_reads_agent_behavior_and_path_u_does_not() -> None:
    """The arms must differ in exactly this property, or the comparison is moot."""
    assert TraceView.FULL_TRAJECTORY.reads_agent_behavior
    assert not TraceView.USER_MESSAGES.reads_agent_behavior
    assert not TraceView.FIRST_USER_MESSAGE.reads_agent_behavior


def test_path_f_shows_tool_activity_path_u_hides_it() -> None:
    trace = _tool_trace("t1", output={"ok": True})
    full = build_view(trace, TraceView.FULL_TRAJECTORY)
    users = build_view(trace, TraceView.USER_MESSAGES)
    assert any("[tool:refund]" in line for line in full.messages)
    assert users.messages == ("refund my order",)


def test_path_f_strips_rewards_and_records_what_it_stripped() -> None:
    """Widening the input to the trajectory must not widen it to the answer."""
    view = build_view(
        _tool_trace("t1", output={"ok": True, "score": 0.98, "reward": 1, "success": True}),
        TraceView.FULL_TRAJECTORY,
    )
    body = " ".join(view.messages)
    assert "0.98" not in body and "reward" not in body and "success" not in body
    assert "ok" in body
    assert view.withheld_fields == ("reward", "score", "success")


def test_path_f_strips_rewards_nested_inside_a_payload() -> None:
    view = build_view(
        _tool_trace("t1", output={"result": {"detail": {"score": 7}, "id": "A1"}}),
        TraceView.FULL_TRAJECTORY,
    )
    assert "7" not in " ".join(view.messages)
    assert view.withheld_fields == ("score",)


def test_path_f_hides_whether_a_span_errored() -> None:
    """Otherwise 'episodes that failed' becomes a family."""
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    trace = Trace(
        trace_id="t1",
        source="chat-json",
        source_digest="0" * 64,
        user_turns=(UserTurn(text="refund"),),
        spans=(
            Span(
                span_id="s1",
                kind=SpanKind.TOOL,
                name="refund",
                started_at=moment,
                ended_at=moment,
                status=SpanStatus.ERROR,
            ),
        ),
    )
    assert "error" not in " ".join(build_view(trace, TraceView.FULL_TRAJECTORY).messages).lower()


def test_path_f_interleaves_user_turns_at_the_point_they_arrived() -> None:
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    trace = Trace(
        trace_id="t1",
        source="chat-json",
        source_digest="0" * 64,
        user_turns=(
            UserTurn(text="refund my order"),
            UserTurn(text="actually cancel it", after_span_id="s1"),
        ),
        spans=(
            Span(
                span_id="s1", kind=SpanKind.TOOL, name="lookup", started_at=moment, ended_at=moment
            ),
            Span(
                span_id="s2", kind=SpanKind.TOOL, name="refund", started_at=moment, ended_at=moment
            ),
        ),
    )
    messages = build_view(trace, TraceView.FULL_TRAJECTORY).messages
    assert messages[0] == "[user] refund my order"
    assert messages[2] == "[user] actually cancel it"
    assert "refund" in messages[3]


def test_path_f_never_drops_a_user_turn_with_a_dangling_anchor() -> None:
    """Losing a user message would make Path F a strictly worse Path U."""
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    trace = Trace(
        trace_id="t1",
        source="chat-json",
        source_digest="0" * 64,
        user_turns=(UserTurn(text="do the thing", after_span_id="gone"),),
        spans=(
            Span(
                span_id="s1", kind=SpanKind.TOOL, name="lookup", started_at=moment, ended_at=moment
            ),
        ),
    )
    assert "[user] do the thing" in build_view(trace, TraceView.FULL_TRAJECTORY).messages


def test_path_f_reads_an_episode_that_recorded_no_user_turn() -> None:
    """Recorded work with no recorded request is still a trajectory."""
    view = build_view(_trace("t1").replace(user_turns=()), TraceView.FULL_TRAJECTORY)
    assert view.readable
    assert view.messages


def test_path_f_is_unreadable_only_with_neither_turns_nor_spans() -> None:
    trace = Trace(trace_id="t1", source="chat-json", source_digest="0" * 64, spans=())
    view = build_view(trace, TraceView.FULL_TRAJECTORY)
    assert not view.readable
    assert "no trajectory to read" in view.unreadable_reason


def test_path_f_truncates_a_huge_payload_rather_than_refusing_it() -> None:
    view = build_view(_tool_trace("t1", output={"blob": "x" * 50_000}), TraceView.FULL_TRAJECTORY)
    assert "truncated" in " ".join(view.messages)


def test_path_f_mining_warns_that_families_may_be_behavior() -> None:
    """The whole risk of the permissive arm, stated on every artifact it makes."""
    corpus = ReadOnlyCorpus(
        _corpus(*(_tool_trace(f"t{i}", output={"ok": True, "score": 1}) for i in range(4))),
        view=TraceView.FULL_TRAJECTORY,
    )
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
    )
    assert draft.view is TraceView.FULL_TRAJECTORY
    assert any("what the agent did" in limit for limit in draft.limitations)
    assert any("score" in limit for limit in draft.limitations)


def test_path_u_mining_carries_no_behavior_warning() -> None:
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
    )
    assert not any("what the agent did" in limit for limit in draft.limitations)
    assert corpus.withheld_fields() == ()


def test_a_corpus_yielding_nothing_to_strip_is_reported_as_suspicious() -> None:
    """Silence here is more likely an unforeseen field name than a clean corpus."""
    corpus = ReadOnlyCorpus(
        _corpus(*(_tool_trace(f"t{i}", output={"ok": True}) for i in range(2))),
        view=TraceView.FULL_TRAJECTORY,
    )
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
    )
    assert any("under other names" in limit for limit in draft.limitations)


def test_the_two_paths_cannot_be_cross_assigned() -> None:
    """Comparing the arms requires that neither ever sees the other's taxonomy."""
    taxonomy = _taxonomy().replace(view=TraceView.FULL_TRAJECTORY)
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")), view=TraceView.USER_MESSAGES)
    with pytest.raises(AssignmentError, match="different experiments"):
        assign_traces(taxonomy, "tax-1", corpus, predict=lambda **_: None)


# --- review fixes: prompts, provenance, budgets, operations, coverage --------


def test_each_arm_gets_a_prompt_describing_what_it_actually_sees() -> None:
    """A Path F prompt claiming user-messages-only is internally inconsistent."""
    from bandits.analyze.rlm_assign import instruction_for as assign_instruction
    from bandits.analyze.rlm_audit import instruction_for as audit_instruction
    from bandits.analyze.rlm_mine import instruction_for as mine_instruction

    for build in (mine_instruction, audit_instruction, assign_instruction):
        full = build(TraceView.FULL_TRAJECTORY)
        users = build(TraceView.USER_MESSAGES)
        assert "FULL trajectory" in full
        assert "ONLY what the users asked for" not in full
        assert "ONLY what the users asked for" in users
        assert "[tool]" not in users


def test_path_f_prompt_forbids_grouping_by_agent_behavior() -> None:
    """The instruction is the only thing steering F away from behaviour groups."""
    from bandits.analyze.rlm_mine import instruction_for

    prompt = instruction_for(TraceView.FULL_TRAJECTORY)
    assert "never by how the agent went about it" in prompt
    assert "Never group by tool sequence" in prompt


def test_prompt_digest_covers_every_view_wording() -> None:
    """Changing an arm's preamble must change the digest artifacts are pinned to."""
    from bandits.analyze import rlm_models
    from bandits.analyze.rlm_mine import prompt_digest

    before = prompt_digest("m")
    original = rlm_models.VIEW_PREAMBLES[TraceView.FULL_TRAJECTORY]
    rlm_models.VIEW_PREAMBLES[TraceView.FULL_TRAJECTORY] = "something else"
    try:
        assert prompt_digest("m") != before
    finally:
        rlm_models.VIEW_PREAMBLES[TraceView.FULL_TRAJECTORY] = original


def test_task_set_backend_names_the_arm_that_produced_it() -> None:
    """Path F labelled rlm-user-messages corrupts every later comparison."""
    taxonomy = _taxonomy().replace(view=TraceView.FULL_TRAJECTORY)
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1"]}).replace(
        view=TraceView.FULL_TRAJECTORY
    )
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1")
    assert task_set.clustering is not None
    assert task_set.clustering.backend == "rlm-full-trajectory"
    assert any("full trajectories" in limit for limit in task_set.limitations)


def test_path_u_task_set_still_names_its_own_arm() -> None:
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1"]})
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1")
    assert task_set.clustering is not None
    assert task_set.clustering.backend == "rlm-user-messages"
    assert not any("full trajectories" in limit for limit in task_set.limitations)


def test_coverage_counts_unplaced_traces_against_the_task_set() -> None:
    """Placed/placed would report 100% for a run that reached almost nothing."""
    taxonomy = _taxonomy()
    run = _run(
        compute_taxonomy_id(taxonomy),
        {"c1": ["t1"]},
        unplaced={"t2": "uncovered", "t3": "ambiguous"},
    )
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1")
    assert task_set.total_workload_mass == 3
    assert task_set.workload_coverage == pytest.approx(1 / 3)


def test_coverage_is_one_only_when_everything_was_placed() -> None:
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1", "t2"]})
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1")
    assert task_set.workload_coverage == 1.0


def test_unreadable_traces_are_outside_the_coverage_denominator() -> None:
    """They were never classifiable, so they are not a coverage failure."""
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1"]}).replace(
        assignments=(
            TraceAssignment(
                trace_id="t1",
                matching_contract_ids=("c1",),
                primary_contract_id="c1",
                status=AssignmentStatus.ASSIGNED,
                reason="x",
            ),
            TraceAssignment(
                trace_id="t2", status=AssignmentStatus.UNREADABLE, reason="no messages"
            ),
        )
    )
    task_set = materialize_task_set(run, taxonomy, corpus_id="corpus-1")
    assert task_set.workload_coverage == 1.0


class _CostingMiner(_ScriptedMiner):
    """A predictor that reports a per-chunk price, the way DSPy history does."""

    def __init__(self, script: list[dict], price: float) -> None:
        super().__init__(script)
        self.price = price
        self.cost = lambda: self.price


def test_max_usd_actually_stops_the_run() -> None:
    """The ceiling was dead: cost was initialized to zero and never updated."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(20))))
    predict = _CostingMiner(
        [
            {
                "contracts": [_RAW_CONTRACT],
                "contract_id": "c1",
                "operations": [{"operation": "REVISE", "rationale": "still moving"}],
            }
        ],
        price=1.0,
    )
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=predict,
        chunk_size=2,
        budget=Budget(max_usd=3.0, max_iterations=50),
    )
    assert draft.stop_reason is StopReason.MAX_USD
    assert not draft.complete
    assert sum(c.cost_usd or 0 for c in draft.chunks) >= 3.0


def test_a_requested_ceiling_nothing_priced_is_reported_not_ignored() -> None:
    """Silently disabling --max-usd is how a run overspends unnoticed."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
        budget=Budget(max_usd=5.0),
    )
    assert any("no backend reported a cost" in limit for limit in draft.limitations)


def test_a_failed_chunk_still_counts_against_the_budget() -> None:
    """A failed call was billed like any other."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(8))))

    class _Failing(_CostingMiner):
        def __call__(self, **kw):
            raise RuntimeError("provider down")

    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_Failing([{}], price=2.0),
        chunk_size=2,
        budget=Budget(max_usd=3.0, max_iterations=50),
    )
    assert draft.stop_reason is StopReason.MAX_USD


def test_a_merge_actually_removes_the_contracts_it_consumed() -> None:
    """These were bookkeeping only: the log said merged, the taxonomy disagreed."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    second = {**_RAW_CONTRACT, "contract_id": "c2", "definition": "cancel an order"}
    predict = _ScriptedMiner(
        [
            {"contracts": [_RAW_CONTRACT, second], "contract_id": "c1"},
            {
                "contracts": [_RAW_CONTRACT],
                "contract_id": "c1",
                "operations": [
                    {
                        "operation": "MERGE",
                        "contract_ids": ["c2", "c1"],
                        "rationale": "one verifier covers both",
                    }
                ],
            },
            {"contract_id": "c1"},
        ]
    )
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    assert [c.contract_id for c in draft.contracts] == ["c1"]


def test_a_merge_moves_members_of_the_consumed_contract() -> None:
    from bandits.analyze.rlm_mine import _TaxonomyState
    from bandits.analyze.rlm_models import TaxonomyOperation

    state = _TaxonomyState()
    state.contracts = {"c1": _contract("c1"), "c2": _contract("c2", "cancel an order")}
    state.assignments = {"t1": "c2", "t2": "c1"}
    state.apply_operations(
        [TaxonomyOperation(operation=Operation.MERGE, contract_ids=("c2", "c1"), rationale="same")],
        [],
    )
    assert "c2" not in state.contracts
    assert state.assignments == {"t1": "c1", "t2": "c1"}


def test_a_split_retires_the_original_and_unplaces_its_orphans() -> None:
    from bandits.analyze.rlm_mine import _TaxonomyState
    from bandits.analyze.rlm_models import TaxonomyOperation

    state = _TaxonomyState()
    state.contracts = {"c1": _contract("c1")}
    state.assignments = {"t1": "c1"}
    state.apply_operations(
        [
            TaxonomyOperation(
                operation=Operation.SPLIT,
                contract_ids=("c1", "c1a", "c1b"),
                rationale="two outcomes",
            )
        ],
        [_contract("c1a"), _contract("c1b", "cancel an order")],
    )
    assert "c1" not in state.contracts
    # Never silently dropped: an unreassigned member becomes an open question.
    assert "t1" in state.uncovered
    assert "t1" not in state.assignments


def test_a_split_naming_no_replacement_does_not_delete_the_family() -> None:
    from bandits.analyze.rlm_mine import _TaxonomyState
    from bandits.analyze.rlm_models import TaxonomyOperation

    state = _TaxonomyState()
    state.contracts = {"c1": _contract("c1")}
    state.assignments = {"t1": "c1"}
    state.apply_operations(
        [TaxonomyOperation(operation=Operation.SPLIT, contract_ids=("c1",), rationale="x")],
        [],
    )
    assert "c1" in state.contracts
    assert state.assignments == {"t1": "c1"}


def test_an_operation_naming_a_contract_nobody_holds_is_reported() -> None:
    """A declared change that could not be applied must not pass silently."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))
    predict = _ScriptedMiner(
        [
            {
                "contracts": [_RAW_CONTRACT],
                "contract_id": "c1",
                "operations": [
                    {
                        "operation": "MERGE",
                        "contract_ids": ["ghost", "phantom"],
                        "rationale": "claimed but never carried out",
                    }
                ],
            }
        ]
    )
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    assert any("could not be applied" in limit for limit in draft.limitations)


def test_leakage_audit_catches_a_reward_under_an_unforeseen_key() -> None:
    """The denylist misses this by construction; the value check does not."""
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj = _corpus(_tool_trace("t1", output={"quality_metric": 0.87654321, "ok": True}))
    analysis = analyze_corpus(corpus_obj)
    corpus = ReadOnlyCorpus(corpus_obj, view=TraceView.FULL_TRAJECTORY)
    # The denylist did not know the name, so the value survived into the view.
    assert "0.87654321" in " ".join(corpus.get_user_messages("t1").messages)
    assert corpus.withheld_fields() == ()
    leaked = corpus.leakage_report(analysis)
    assert leaked and "t1" in leaked[0]


def test_leakage_audit_is_quiet_when_redaction_worked() -> None:
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj = _corpus(_tool_trace("t1", output={"score": 0.87654321, "ok": True}))
    corpus = ReadOnlyCorpus(corpus_obj, view=TraceView.FULL_TRAJECTORY)
    assert corpus.leakage_report(analyze_corpus(corpus_obj)) == ()


def test_leakage_audit_does_not_run_on_the_user_message_arms() -> None:
    """Those arms never read a payload that could carry a score."""
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj = _corpus(_tool_trace("t1", output={"quality_metric": 0.87654321}))
    corpus = ReadOnlyCorpus(corpus_obj, view=TraceView.USER_MESSAGES)
    assert corpus.leakage_report(analyze_corpus(corpus_obj)) == ()


def test_mining_reports_leakage_as_a_limitation() -> None:
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj = _corpus(
        *(_tool_trace(f"t{i}", output={"quality_metric": 0.87654321}) for i in range(4))
    )
    corpus = ReadOnlyCorpus(corpus_obj, view=TraceView.FULL_TRAJECTORY)
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
        analysis=analyze_corpus(corpus_obj),
    )
    assert any("OUTCOME LEAKAGE" in limit for limit in draft.limitations)


def test_mining_without_an_analysis_says_the_check_was_skipped() -> None:
    corpus = ReadOnlyCorpus(
        _corpus(*(_tool_trace(f"t{i}", output={"ok": True}) for i in range(2))),
        view=TraceView.FULL_TRAJECTORY,
    )
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=2,
    )
    assert any("never checked for outcome values" in limit for limit in draft.limitations)


def test_leakage_audit_ignores_identifiers_echoing_the_trace_id() -> None:
    """A confirmation number is not an outcome; flagging it drowns real findings."""
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj = _corpus(_tool_trace("t1", output={"confirmation": "t1-ok"}))
    corpus = ReadOnlyCorpus(corpus_obj, view=TraceView.FULL_TRAJECTORY)
    assert corpus.leakage_report(analyze_corpus(corpus_obj)) == ()


def test_leakage_audit_still_catches_a_real_score_beside_an_identifier() -> None:
    from bandits.analyze.analysis import analyze_corpus

    corpus_obj = _corpus(
        _tool_trace("t1", output={"confirmation": "t1-ok", "quality_metric": 0.87654321})
    )
    corpus = ReadOnlyCorpus(corpus_obj, view=TraceView.FULL_TRAJECTORY)
    assert corpus.leakage_report(analyze_corpus(corpus_obj))


# --- resumable sessions and incremental logging ------------------------------


def _recorder(tmp_path, session_id: str = "sess-1"):
    from bandits.analyze.rlm_session import SessionRecorder, SessionStore

    store = SessionStore(tmp_path)
    return store, SessionRecorder(
        store,
        session_id=session_id,
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        model="test-model",
    )


def test_state_is_written_after_every_chunk_not_every_pass(tmp_path) -> None:
    """A watcher must never be more than one model call behind."""
    store, recorder = _recorder(tmp_path)
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(20))))
    mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=5,
        session=recorder,
    )
    events = [e for e in store.read_events("sess-1") if e["event"] == "chunk_complete"]
    assert len(events) == 8
    # Progress within a half-finished pass is visible, not just at boundaries.
    first_pass = [e for e in events if e["pass"] == 0]
    assert [e["seen_this_pass"] for e in first_pass] == [5, 10, 15, 20]
    assert all(e["of"] == 20 for e in events)


def test_a_mid_pass_crash_leaves_resumable_state_on_disk(tmp_path) -> None:
    store, recorder = _recorder(tmp_path, "sess-crash")
    calls = {"n": 0}

    def predict(*, chunk: str, taxonomy: str, question: str):
        import json

        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("provider down for good")
        rows = json.loads(chunk)
        return SimpleNamespace(
            contracts=[_RAW_CONTRACT],
            operations=[],
            assignments={r["trace_id"]: "c1" for r in rows},
            ambiguous_trace_ids=[],
            uncovered_trace_ids=[],
        )

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(30))))
    mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=10, session=recorder)

    state = store.read("sess-crash")
    # The work that succeeded survived, and the pass order is pinned so a resume
    # reads what is left rather than reshuffling and rereading.
    assert len(state.assignments) == 20
    assert len(state.seen_this_pass) == 20
    assert len(state.pass_order) == 30
    assert state.contracts
    assert "provider down" in state.last_error


def test_progress_line_reads_without_parsing_anything(tmp_path) -> None:
    store, recorder = _recorder(tmp_path, "sess-p")
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(10))))
    mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=5,
        session=recorder,
    )
    progress = store.read("sess-p").progress
    assert "pass 2/2" in progress
    assert "10/10 traces" in progress
    assert "1 contracts" in progress


def test_a_failed_session_is_not_readable_as_idle(tmp_path) -> None:
    store, recorder = _recorder(tmp_path, "sess-f")
    recorder.begin(traces_total=5, requested_passes=2, seed=1)
    recorder.fail("the run died")
    state = store.read("sess-f")
    assert state.status == "failed"
    assert state.last_error == "the run died"
    assert [e["event"] for e in store.read_events("sess-f")][-1] == "session_failed"


def test_a_finished_session_is_marked_awaiting_review(tmp_path) -> None:
    store, recorder = _recorder(tmp_path, "sess-done")
    recorder.begin(traces_total=5, requested_passes=2, seed=1)
    recorder.finish(status="awaiting_review", stop_reason="passes_complete", draft_id="d1")
    assert store.read("sess-done").status == "awaiting_review"


def test_sessions_are_listed_newest_first(tmp_path) -> None:
    from bandits.analyze.rlm_session import SessionStore

    store = SessionStore(tmp_path)
    for name in ("a", "b"):
        _, recorder = _recorder(tmp_path, f"sess-{name}")
        recorder.begin(traces_total=1, requested_passes=1, seed=1)
    assert {s.session_id for s in store.list()} == {"sess-a", "sess-b"}


def test_session_state_survives_a_round_trip(tmp_path) -> None:
    store, recorder = _recorder(tmp_path, "sess-rt")
    recorder.begin(traces_total=3, requested_passes=2, seed=5)
    reloaded = store.read("sess-rt")
    assert reloaded.seed == 5
    assert reloaded.requested_passes == 2
    assert reloaded.view is TraceView.USER_MESSAGES


def test_a_finished_session_reports_the_runs_own_pass_count(tmp_path) -> None:
    """The last checkpoint is written before the final pass increments."""
    store, recorder = _recorder(tmp_path, "sess-count")
    recorder.begin(traces_total=4, requested_passes=2, seed=1)
    recorder.finish(status="awaiting_review", stop_reason="passes_complete", completed_passes=2)
    assert store.read("sess-count").completed_passes == 2


def test_each_family_names_the_view_it_was_mined_from() -> None:
    """A Path F family saying "user messages alone" misstates its own evidence."""
    taxonomy = _taxonomy().replace(view=TraceView.FULL_TRAJECTORY)
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1"]}).replace(
        view=TraceView.FULL_TRAJECTORY
    )
    family = materialize_task_set(run, taxonomy, corpus_id="corpus-1").families[0]
    assert any("full-trajectory view" in limit for limit in family.limitations)
    assert not any("user messages alone" in limit for limit in family.limitations)
    assert any("execution behaviour" in limit for limit in family.limitations)


def test_a_path_u_family_makes_no_behaviour_caveat() -> None:
    taxonomy = _taxonomy()
    run = _run(compute_taxonomy_id(taxonomy), {"c1": ["t1"]})
    family = materialize_task_set(run, taxonomy, corpus_id="corpus-1").families[0]
    assert any("user-messages view" in limit for limit in family.limitations)
    assert not any("execution behaviour" in limit for limit in family.limitations)


# --- viewing -----------------------------------------------------------------


def _render(renderable) -> str:
    from rich.console import Console

    console = Console(width=100, record=True, file=open("/dev/null", "w"))
    console.print(renderable)
    return console.export_text()


def test_a_family_card_shows_the_outcome_that_decides_membership() -> None:
    """Two families with similar definitions differ here, so it cannot be buried."""
    from bandits.analyze.rlm_view import family_card

    contract = _contract("c1").replace(
        inclusion_rules=("refund a damaged item",),
        exclusion_rules=("ask whether a refund is allowed",),
    )
    text = _render(family_card(contract, members=("t1",)))
    assert "Required outcome" in text
    assert "the order is refunded" in text
    assert "refund a damaged item" in text
    assert "ask whether a refund is allowed" in text


def test_a_family_card_shows_real_requests_beside_ids() -> None:
    """An id list is unreviewable; the point is judging without the corpus open."""
    from bandits.analyze.rlm_view import family_card

    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "please refund order A-1001")))
    text = _render(family_card(_contract("c1"), members=("t1",), corpus=corpus))
    assert "please refund order A-1001" in text


def test_a_card_renders_without_a_corpus() -> None:
    from bandits.analyze.rlm_view import family_card

    assert "t1" in _render(family_card(_contract("c1"), members=("t1",)))


def test_a_card_surfaces_an_audit_verdict_and_topical_flag() -> None:
    from bandits.analyze.rlm_models import AuditFinding
    from bandits.analyze.rlm_view import family_card

    audit = TaxonomyAudit(
        draft_id="d1",
        findings=(
            AuditFinding(
                contract_id="c1",
                recommendation="split",
                topical_only=True,
                rationale="members share a topic but need different verifiers",
                least_compatible_pair=("t1", "t2"),
            ),
        ),
        model="m",
        prompt_digest="d",
    )
    text = _render(family_card(_contract("c1"), members=("t1", "t2"), audit=audit))
    assert "split" in text
    assert "topical grouping" in text
    assert "t1 vs t2" in text


def test_the_overview_orders_by_size_and_shows_verdicts() -> None:
    from bandits.analyze.rlm_view import taxonomy_overview

    contracts = (_contract("c-small"), _contract("c-big", "cancel an order"))
    members = {"c-small": ("t1",), "c-big": ("t2", "t3", "t4")}
    text = _render(taxonomy_overview(contracts, members=members))
    assert text.index("c-big") < text.index("c-small")


def test_the_live_panel_shows_progress_within_an_unfinished_pass() -> None:
    """The question while watching is how far through it is, not just that it runs."""
    from bandits.analyze.rlm_session import SessionState
    from bandits.analyze.rlm_view import live_panel

    state = SessionState(
        session_id="s1",
        analysis_id="a1",
        view=TraceView.USER_MESSAGES,
        model="m",
        seed=42,
        requested_passes=2,
        traces_total=24,
        traces_seen_this_pass=12,
        chunk_index=3,
        llm_calls=6,
        cost_usd=0.0027,
        contracts=(_contract("c1"),),
        assignments={"t1": "c1"},
    )
    text = _render(live_panel(state))
    assert "12/24" in text
    assert "pass" in text and "1 of 2" in text
    assert "$0.0027" in text
    assert "Refund an order" in text


def test_the_live_panel_says_when_a_run_died() -> None:
    from bandits.analyze.rlm_session import SessionState
    from bandits.analyze.rlm_view import live_panel

    state = SessionState(
        session_id="s1",
        analysis_id="a1",
        view=TraceView.USER_MESSAGES,
        model="m",
        seed=1,
        requested_passes=2,
        status="failed",
        last_error="provider down",
    )
    text = _render(live_panel(state))
    assert "failed" in text
    assert "provider down" in text


def test_pass_history_marks_a_partial_pass() -> None:
    from rich.console import Console

    from bandits.analyze.rlm_view import print_pass_history

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(20))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [_RAW_CONTRACT], "contract_id": "c1"}]),
        chunk_size=5,
        budget=Budget(max_iterations=2),
    )
    console = Console(width=100, record=True, file=open("/dev/null", "w"))
    print_pass_history(draft, console)
    assert "partial" in console.export_text()


# --- resumption --------------------------------------------------------------


def _crash_after(chunks: int, seen: dict[str, int]):
    """A predictor that works for N chunks then dies, recording what it read."""
    state = {"n": 0}

    def predict(*, chunk: str, taxonomy: str, question: str):
        import json

        state["n"] += 1
        if chunks and state["n"] > chunks:
            raise KeyboardInterrupt("the run was killed")
        rows = json.loads(chunk)
        for row in rows:
            seen[row["trace_id"]] = seen.get(row["trace_id"], 0) + 1
        return SimpleNamespace(
            contracts=[_RAW_CONTRACT],
            operations=[],
            assignments={r["trace_id"]: "c1" for r in rows},
            ambiguous_trace_ids=[],
            uncovered_trace_ids=[],
        )

    return predict


def test_a_resumed_run_finishes_the_pass_it_died_in(tmp_path) -> None:
    """Without this the session was inspectable but restarting began again."""
    from bandits.analyze.rlm_session import SessionRecorder

    store, recorder = _recorder(tmp_path, "sess-r")
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(40))))
    seen: dict[str, int] = {}

    with pytest.raises(KeyboardInterrupt):
        mine_taxonomy(
            corpus, "analysis-1", predict=_crash_after(5, seen), chunk_size=10, session=recorder
        )
    crashed = store.read("sess-r")
    assert crashed.pass_index == 1
    assert crashed.traces_seen_this_pass == 10

    resumed = SessionRecorder(
        store,
        session_id="sess-r",
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        model="test-model",
        resumed_from="sess-r",
    )
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_crash_after(0, seen),
        chunk_size=10,
        session=resumed,
        resume=crashed,
    )
    assert draft.completed_passes == 2
    assert draft.stop_reason is StopReason.PASSES_COMPLETE
    # The point of resuming: no trace is read a third time, and none is skipped.
    assert all(count == 2 for count in seen.values()), seen
    assert sum(seen.values()) == 80


def test_a_resume_restores_the_taxonomy_rather_than_rebuilding_it(tmp_path) -> None:
    from bandits.analyze.rlm_session import SessionRecorder

    store, recorder = _recorder(tmp_path, "sess-t")
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(20))))
    seen: dict[str, int] = {}
    with pytest.raises(KeyboardInterrupt):
        mine_taxonomy(
            corpus, "analysis-1", predict=_crash_after(1, seen), chunk_size=10, session=recorder
        )
    crashed = store.read("sess-t")
    assert crashed.contracts and crashed.assignments

    def refuse(**_):
        raise AssertionError("a resumed run must not need the model to restore state")

    # Nothing is asked of the model before the first new chunk, so the restored
    # contracts and placements came from the checkpoint alone.
    resumed = SessionRecorder(
        store,
        session_id="sess-t",
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        model="test-model",
    )
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_crash_after(0, seen),
        chunk_size=10,
        session=resumed,
        resume=crashed,
    )
    assert [c.contract_id for c in draft.contracts] == ["c1"]


def test_a_resumed_pass_keeps_the_order_it_was_reading(tmp_path) -> None:
    """Reshuffling on resume would reread some traces and skip others."""
    from bandits.analyze.rlm_session import SessionRecorder

    store, recorder = _recorder(tmp_path, "sess-o")
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(30))))
    seen: dict[str, int] = {}
    with pytest.raises(KeyboardInterrupt):
        mine_taxonomy(
            corpus, "analysis-1", predict=_crash_after(1, seen), chunk_size=10, session=recorder
        )
    crashed = store.read("sess-o")
    resumed = SessionRecorder(
        store,
        session_id="sess-o",
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        model="test-model",
    )
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_crash_after(0, seen),
        chunk_size=10,
        session=resumed,
        resume=crashed,
    )
    assert draft.passes[0].trace_ids == crashed.pass_order
    assert all(count == 2 for count in seen.values())


# --- membership after a merge ------------------------------------------------


def _merged_draft() -> TaxonomyDraft:
    return TaxonomyDraft(
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        seed=1,
        contracts=(_contract("c1"),),
        chunks=(
            ChunkResult(
                index=0,
                pass_index=0,
                trace_ids=("t0", "t1"),
                assignments={"t0": "c2", "t1": "c2"},
            ),
            ChunkResult(
                index=1,
                pass_index=0,
                trace_ids=("t2",),
                assignments={"t2": "c1"},
                operations=(
                    TaxonomyOperation(
                        operation=Operation.MERGE,
                        contract_ids=("c2", "c1"),
                        rationale="one verifier covers both",
                    ),
                ),
            ),
        ),
        assignments={"t0": "c1", "t1": "c1", "t2": "c1"},
        stop_reason=StopReason.PASSES_COMPLETE,
        completed_passes=2,
        budget=Budget(),
        model="m",
        prompt_digest="d",
    )


def test_a_merged_family_shows_the_members_it_absorbed() -> None:
    """Replaying chunk assignments strands them under the consumed contract."""
    from bandits.analyze.rlm_audit import _draft_members

    assert _draft_members(_merged_draft()) == {"c1": ("t0", "t1", "t2")}


def test_a_consumed_contract_is_not_rendered_as_a_family() -> None:
    """A ghost family is something a reviewer could try to act on."""
    from bandits.analyze.rlm_audit import _draft_members

    assert "c2" not in _draft_members(_merged_draft())


# --- the live-model boundary -------------------------------------------------


def test_the_spend_wrapper_accepts_every_stage_signature() -> None:
    """The bug the smoke test caught: it hardcoded the family audit's arguments.

    Every RLM stage wraps its predictor in ``scoped_to_history``, and the three
    stages pass different keywords. Nothing caught this because the tests inject
    plain functions and never reach the wrapper, so it failed only against a
    live model — on the first real call of all three paths at once.
    """
    from bandits.analyze.audit import scoped_to_history

    language_model = SimpleNamespace(history=[])
    for kwargs in (
        {"members": "m", "question": "q"},  # family audit
        {"chunk": "c", "taxonomy": "t", "question": "q"},  # mining
        {"contract": "c", "members": "m", "outsiders": "o", "question": "q"},  # taxonomy audit
        {"taxonomy": "t", "batch": "b", "question": "q"},  # assignment
    ):
        wrapped = scoped_to_history(lambda **received: received, language_model)
        assert wrapped(**kwargs) == kwargs


def test_the_cost_wrapper_survives_a_predictor_it_did_not_wrap() -> None:
    """`with_cost` reads the spend recorder's entries, which may not exist."""
    from bandits.analyze.rlm_mine import with_cost

    assert with_cost(lambda **_: None).cost() is None


def test_a_reply_arriving_as_json_text_is_not_thrown_away() -> None:
    """The first real run's failure: three CREATEs recorded, zero contracts kept.

    DSPy usually returns the declared types, but when the root model runs out of
    iterations it falls back to an ``extract`` pass whose fields arrive as
    strings. Every parser type-checked its input, so the whole chunk was
    silently dropped and the run ended reporting an empty taxonomy it had
    already paid to build.
    """
    import json

    def predict(*, chunk: str, taxonomy: str, question: str):
        rows = json.loads(chunk)
        return SimpleNamespace(
            contracts=json.dumps([_RAW_CONTRACT]),
            operations=json.dumps([{"operation": "CREATE", "rationale": "new family"}]),
            assignments=json.dumps({r["trace_id"]: "c1" for r in rows}),
            ambiguous_trace_ids="[]",
            uncovered_trace_ids="[]",
        )

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(6))))
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=3)
    assert [c.contract_id for c in draft.contracts] == ["c1"]
    assert len(draft.assignments) == 6
    assert any(op.operation is Operation.CREATE for p in draft.passes for op in p.operations)


def test_a_fenced_json_reply_is_decoded() -> None:
    from bandits.analyze.rlm_mine import _decoded

    assert _decoded('```json\n{"a": 1}\n```') == {"a": 1}
    assert _decoded('{"a": 1}') == {"a": 1}


def test_a_single_contract_returned_bare_is_still_read() -> None:
    """A model that returns one object rather than a list of one."""
    import json

    def predict(*, chunk: str, taxonomy: str, question: str):
        rows = json.loads(chunk)
        return SimpleNamespace(
            contracts=_RAW_CONTRACT,
            operations=[],
            assignments={r["trace_id"]: "c1" for r in rows},
            ambiguous_trace_ids=[],
            uncovered_trace_ids=[],
        )

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    assert [c.contract_id for c in draft.contracts] == ["c1"]


def test_unparseable_text_is_still_dropped_rather_than_crashing() -> None:
    """Decoding must not turn a malformed reply into an exception."""

    def predict(*, chunk: str, taxonomy: str, question: str):
        return SimpleNamespace(
            contracts="I could not complete this task",
            operations="n/a",
            assignments="none",
            ambiguous_trace_ids="",
            uncovered_trace_ids="",
        )

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    assert draft.contracts == ()


def test_assignment_results_arriving_as_json_text_are_read() -> None:
    import json

    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "refund")))

    def predict(*, taxonomy: str, batch: str, question: str):
        return SimpleNamespace(
            results=json.dumps(
                [
                    {
                        "trace_id": row["trace_id"],
                        "matching_contract_ids": ["c1"],
                        "primary_contract_id": "c1",
                        "reason": "refund",
                    }
                    for row in json.loads(batch)
                ]
            )
        )

    run = assign_traces(_taxonomy(), "tax-1", corpus, predict=predict)
    assert run.members() == {"c1": ("t1", "t2")}


def test_a_decoded_scalar_is_not_iterated() -> None:
    """`"1"` decodes to an int, which raises; `'"txt"'` to a string, which
    iterates one character at a time and silently yields garbage."""
    from bandits.analyze.rlm_mine import _rows

    for scalar in ('"1"', "1", "true", '"some text"', "null", "", None, 3, True):
        assert _rows(scalar) == [], scalar
    assert _rows('[{"a": 1}]') == [{"a": 1}]
    assert _rows('{"a": 1}') == [{"a": 1}]
    assert _rows([{"a": 1}]) == [{"a": 1}]


def test_a_scalar_reply_does_not_crash_a_chunk() -> None:
    def predict(*, chunk: str, taxonomy: str, question: str):
        return SimpleNamespace(
            contracts="1",
            operations="true",
            assignments='"nope"',
            ambiguous_trace_ids='"t1"',
            uncovered_trace_ids="42",
        )

    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(4))))
    draft = mine_taxonomy(corpus, "analysis-1", predict=predict, chunk_size=2)
    assert draft.contracts == ()
    assert all(chunk.status == "success" for chunk in draft.chunks)
    assert any("no contracts at all" in limit for limit in draft.limitations)


def test_a_bare_string_is_not_read_as_a_list_of_ids() -> None:
    from bandits.analyze.rlm_mine import _string_tuple

    assert _string_tuple("t1") == ()
    assert _string_tuple(["t1", "t2"]) == ("t1", "t2")


def test_assignment_survives_a_scalar_reply() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))
    run = assign_traces(
        _taxonomy(), "tax-1", corpus, predict=lambda **_: SimpleNamespace(results="1")
    )
    assert run.by_status(AssignmentStatus.UNCOVERED)[0].trace_id == "t1"


def test_every_chunk_keeps_what_the_model_actually_returned() -> None:
    """Two paid runs were lost guessing at why contracts were rejected."""
    corpus = ReadOnlyCorpus(_corpus(*(_trace(f"t{i}", "refund") for i in range(2))))
    draft = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=_ScriptedMiner([{"contracts": [{"name": "Topic only"}]}]),
        chunk_size=2,
    )
    chunk = draft.chunks[0]
    assert "Topic only" in chunk.raw_reply
    assert chunk.dropped_contracts
    assert "Topic only" in chunk.dropped_contracts[0]


def test_a_contract_is_kept_however_the_model_spells_its_fields() -> None:
    """The real failure: 22 contracts proposed, all refused on field spelling."""
    from bandits.analyze.rlm_mine import _parse_contract

    for raw in (
        {"name": "C", "definition": "cancel a reservation", "required_outcome_shape": "cancelled"},
        {"name": "C", "definition": "cancel a reservation", "required_outcome": ["cancelled"]},
        {"name": "C", "definition": "cancel a reservation", "outcome_shape": "cancelled"},
        {"name": "C", "definition": "cancel a reservation", "requiredOutcomeShape": ["cancelled"]},
        {"name": "C", "description": "cancel a reservation", "required_outcome_shape": ["x"]},
        {"definition": "cancel a reservation for a user", "required_outcome_shape": ["x"]},
    ):
        assert _parse_contract(raw, known_traces=set()) is not None, raw


def test_a_contract_with_nothing_to_verify_is_still_refused() -> None:
    """Loosening the spelling must not admit a topic with no stated outcome."""
    from bandits.analyze.rlm_mine import _parse_contract

    assert _parse_contract({"name": "Refunds"}, known_traces=set()) is None
    assert _parse_contract(
        {"name": "Refunds", "definition": "refund things"}, known_traces=set()
    ) is None
