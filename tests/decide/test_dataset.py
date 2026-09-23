from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from bandits.analyze.models import TaskFamily, TaskSet
from bandits.decide.dataset import (
    ACTION_OUTCOME_OPTIONS,
    ACTION_OUTCOME_QUESTION,
    DecisionSchema,
    DecisionTarget,
    build_decision_dataset,
    build_decision_dataset_from_corpus,
    load_decision_dataset,
    save_decision_dataset,
    source_from_trace,
)
from bandits.store import DerivedStore
from bandits.traces import Span, SpanKind, SpanStatus, Trace
from bandits.verify.nextstate import Archetype, TraceSignal, TurnJudgeRun, TurnVerdict, _vote_shares
from bandits.verify.turns import extract_turns

_T = datetime(2025, 1, 1, tzinfo=UTC)


def _span(span_id: str, kind: SpanKind, name: str, output, status=SpanStatus.OK) -> Span:
    return Span(
        span_id=span_id, kind=kind, name=name, started_at=_T, ended_at=_T, output=output, status=status
    )


def _trace(trace_id: str, task: str = "fix the bug") -> Trace:
    return Trace(
        trace_id=trace_id,
        source="test",
        source_digest="d",
        task=task,
        spans=(
            _span("m1", SpanKind.MODEL, "gpt", "open schema.py"),
            _span("t1", SpanKind.TOOL, "execute", "schema.py not found", SpanStatus.ERROR),
            _span("m2", SpanKind.MODEL, "gpt", "run tests"),
            _span("t2", SpanKind.TOOL, "execute", "3 passed"),
            _span("m3", SpanKind.MODEL, "gpt", "done"),
        ),
    )


def _verdict(trace_id: str, index: int, action_span_id: str, *, votes: tuple[int, ...]) -> TurnVerdict:
    return TurnVerdict(
        trace_id=trace_id,
        index=index,
        action_span_id=action_span_id,
        observed=True,
        score=votes[0],
        judge_votes=_vote_shares(votes),
        votes=votes,
    )


def _run(traces: list[Trace], verdicts: list[TurnVerdict], *, votes: int = 1) -> TurnJudgeRun:
    signals = tuple(
        TraceSignal(trace_id=t.trace_id, turns=len(extract_turns(t)), scored=0) for t in traces
    )
    return TurnJudgeRun(
        corpus_id="corpus-1",
        archetype=Archetype.CODING,
        model="m",
        prompt_digest="p",
        votes=votes,
        trace_ids=tuple(t.trace_id for t in traces),
        verdicts=tuple(verdicts),
        signals=signals,
    )


def _all_observed_verdicts(trace: Trace, votes: tuple[int, ...] = (1,)) -> list[TurnVerdict]:
    return [_verdict(trace.trace_id, t.index, t.action_span_id, votes=votes) for t in extract_turns(trace) if t.observed]


def test_action_outcome_row_carries_state_options_and_soft_target() -> None:
    trace = _trace("a")
    verdicts = _all_observed_verdicts(trace)
    run = _run([trace], verdicts)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    assert dataset.counts.examples == 2  # turn 0 has an error reaction, turn 1 does not; both observed
    assert dataset.counts.quarantined == 0
    row = dataset.examples[0]
    assert row.primitive == "choice"
    assert set(row.options) == set(ACTION_OUTCOME_OPTIONS)
    assert row.label_source == "judge_votes"
    assert row.split == "within_family_fit"
    assert abs(sum(row.target.probabilities.values()) - 1.0) < 1e-9
    assert row.judge.votes_requested == 1
    assert row.judge.votes_valid == 1
    assert row.judge.settings_digest  # stored, not just computable
    assert row.lineage.trace_id == "a"
    assert row.lineage.corpus_id == "corpus-1"
    assert row.lineage.judge_run_id == "judge-run-1"
    assert row.lineage.source_kind == "action_outcome_judge_votes"
    assert row.lineage.record_id == "a:0"
    assert "judge-run-1" in row.lineage.source_artifact_ids


def test_state_carries_task_and_previous_action_not_teacher_prompt() -> None:
    """The compiled state must give the student the same evidence the judge
    prompt used -- task, previous action, current action, reaction -- and
    nothing of the judge's own instructions, boxed-score format or reply."""
    trace = _trace("a", task="Refund the customer's order")
    turns = [t for t in extract_turns(trace) if t.observed]
    verdicts = [_verdict("a", t.index, t.action_span_id, votes=(1,)) for t in turns]
    run = _run([trace], verdicts)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    first, second = sorted(dataset.examples, key=lambda r: r.lineage.turn_index)
    assert "Refund the customer's order" in first.state
    assert "open schema.py" in first.state
    assert "schema.py not found" in first.state
    # first turn has no previous action -- must not invent one
    assert "Previous action" not in first.state

    assert "Previous action" in second.state
    assert "open schema.py" in second.state  # the first turn's action, carried as context
    assert "run tests" in second.state  # the second turn's own action
    assert "\\boxed" not in second.state
    assert "HINT" not in second.state


def test_missing_observed_verdict_is_quarantined_not_silently_dropped() -> None:
    trace = _trace("a")
    turns = [t for t in extract_turns(trace) if t.observed]
    # only judge the second turn; the run is silent about the first
    verdicts = [_verdict("a", turns[1].index, turns[1].action_span_id, votes=(1,))]
    run = _run([trace], verdicts)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    assert dataset.counts.examples == 1
    assert dataset.counts.quarantined == 1
    assert dataset.quarantined[0].turn_index == turns[0].index
    assert "incomplete judge artifact" in dataset.quarantined[0].reasons[0]


def test_majority_tie_reports_the_split_vote_not_unclear() -> None:
    trace = _trace("a")
    turns = [t for t in extract_turns(trace) if t.observed]
    verdicts = [_verdict("a", turns[0].index, turns[0].action_span_id, votes=(-1, 1))]
    verdicts += [_verdict("a", t.index, t.action_span_id, votes=(1,)) for t in turns[1:]]
    run = _run([trace], verdicts, votes=2)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    row = next(r for r in dataset.examples if r.lineage.turn_index == turns[0].index)
    assert row.target.probabilities["failure"] == 0.5
    assert row.target.probabilities["success"] == 0.5
    assert row.target.probabilities["unclear"] == 0.0


def test_unobserved_turns_are_excluded_not_quarantined() -> None:
    trace = _trace("a")
    turns = extract_turns(trace)
    unobserved = [t for t in turns if not t.observed]
    assert unobserved
    observed = [t for t in turns if t.observed]
    verdicts = [_verdict("a", t.index, t.action_span_id, votes=(1,)) for t in observed]
    run = _run([trace], verdicts)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    assert dataset.counts.quarantined == 0
    assert {r.lineage.turn_index for r in dataset.examples} == {t.index for t in observed}


def test_failed_judgment_is_quarantined_not_turned_into_unclear() -> None:
    trace = _trace("a")
    turns = [t for t in extract_turns(trace) if t.observed]
    failed = TurnVerdict(
        trace_id="a",
        index=turns[0].index,
        action_span_id=turns[0].action_span_id,
        observed=True,
        score=None,
        judge_votes=None,
        failure="unparseable: no boxed score",
    )
    verdicts = [failed] + [_verdict("a", t.index, t.action_span_id, votes=(1,)) for t in turns[1:]]
    run = _run([trace], verdicts)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    assert dataset.counts.quarantined == 1
    assert dataset.quarantined[0].turn_index == turns[0].index
    assert "unparseable" in dataset.quarantined[0].reasons[0]
    assert turns[0].index not in {r.lineage.turn_index for r in dataset.examples}


def test_missing_trace_is_quarantined() -> None:
    trace = _trace("a")
    verdicts = [_verdict("a", 0, "m1", votes=(1,))]
    run = _run([trace], verdicts)
    dataset = build_decision_dataset([], run, "judge-run-1")

    assert dataset.counts.quarantined == 1
    assert dataset.counts.examples == 0
    assert dataset.quarantined[0].trace_id == "a"
    assert "not found among the sources" in dataset.quarantined[0].reasons[0]


def test_minimum_valid_votes_quarantines_underpowered_verdicts() -> None:
    trace = _trace("a")
    turns = [t for t in extract_turns(trace) if t.observed]
    verdicts = [_verdict("a", t.index, t.action_span_id, votes=(1,)) for t in turns]
    run = _run([trace], verdicts, votes=3)
    dataset = build_decision_dataset_from_corpus(
        [trace], run, "judge-run-1", minimum_valid_votes=2
    )

    assert dataset.counts.examples == 0
    assert dataset.counts.quarantined == len(turns)
    assert all("valid vote" in q.reasons[0] for q in dataset.quarantined)


def _family_task_set(fit_trace: Trace, held_out_trace: Trace) -> TaskSet:
    family = TaskFamily(
        family_id="family-1",
        descriptor="fix a bug",
        trace_ids=(fit_trace.trace_id, held_out_trace.trace_id),
        medoid_trace_id=fit_trace.trace_id,
        workload_mass=2,
        fit_trace_ids=(fit_trace.trace_id,),
        held_out_trace_ids=(held_out_trace.trace_id,),
    )
    return TaskSet(
        corpus_id="corpus-1",
        analysis_id="analysis-1",
        families=(family,),
        selected=(),
        total_workload_mass=2,
        workload_coverage=1.0,
    )


def test_family_split_follows_the_task_set_never_splits_within_a_trace() -> None:
    fit_trace, held_out_trace = _trace("a"), _trace("b")
    task_set = _family_task_set(fit_trace, held_out_trace)
    verdicts = []
    for trace in (fit_trace, held_out_trace):
        verdicts += _all_observed_verdicts(trace)
    run = _run([fit_trace, held_out_trace], verdicts)
    dataset = build_decision_dataset_from_corpus(
        [fit_trace, held_out_trace], run, "judge-run-1", task_set=task_set, task_set_id="taskset-1"
    )

    splits = {r.lineage.trace_id: r.split for r in dataset.examples}
    assert splits["a"] == "within_family_fit"
    assert splits["b"] == "within_family_held_out"
    by_trace: dict[str, set[str]] = {}
    for row in dataset.examples:
        by_trace.setdefault(row.lineage.trace_id, set()).add(row.split)
    assert all(len(sides) == 1 for sides in by_trace.values())


def test_no_task_set_defaults_every_example_to_within_family_fit() -> None:
    trace = _trace("a")
    verdicts = _all_observed_verdicts(trace)
    run = _run([trace], verdicts)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    assert all(r.split == "within_family_fit" for r in dataset.examples)
    assert dataset.source_task_set_id is None


def test_task_set_from_a_different_corpus_is_rejected() -> None:
    fit_trace, held_out_trace = _trace("a"), _trace("b")
    family = TaskFamily(
        family_id="family-1",
        descriptor="fix a bug",
        trace_ids=("a", "b"),
        medoid_trace_id="a",
        workload_mass=2,
        fit_trace_ids=("a",),
        held_out_trace_ids=("b",),
    )
    mismatched_task_set = TaskSet(
        corpus_id="corpus-OTHER",
        analysis_id="analysis-1",
        families=(family,),
        selected=(),
        total_workload_mass=2,
        workload_coverage=1.0,
    )
    verdicts = _all_observed_verdicts(fit_trace) + _all_observed_verdicts(held_out_trace)
    run = _run([fit_trace, held_out_trace], verdicts)

    with pytest.raises(ValueError, match="not the judge run's corpus"):
        build_decision_dataset_from_corpus(
            [fit_trace, held_out_trace],
            run,
            "judge-run-1",
            task_set=mismatched_task_set,
            task_set_id="taskset-1",
        )


def test_trace_unplaced_by_the_task_set_is_quarantined_not_defaulted_to_fit() -> None:
    placed = _trace("a")
    unplaced = _trace("orphan")
    family = TaskFamily(
        family_id="family-1",
        descriptor="fix a bug",
        trace_ids=("a",),
        medoid_trace_id="a",
        workload_mass=1,
        fit_trace_ids=("a",),
    )
    task_set = TaskSet(
        corpus_id="corpus-1",
        analysis_id="analysis-1",
        families=(family,),
        selected=(),
        total_workload_mass=1,
        workload_coverage=1.0,
    )
    verdicts = _all_observed_verdicts(placed) + _all_observed_verdicts(unplaced)
    run = _run([placed, unplaced], verdicts)
    dataset = build_decision_dataset_from_corpus(
        [placed, unplaced], run, "judge-run-1", task_set=task_set, task_set_id="taskset-1"
    )

    assert all(r.lineage.trace_id != "orphan" for r in dataset.examples)
    assert any(
        q.trace_id == "orphan" and "does not place this trace" in q.reasons[0]
        for q in dataset.quarantined
    )


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        DecisionTarget(kind="soft", probabilities={"success": 0.5, "failure": 0.2})


def test_probabilities_must_be_finite_and_nonnegative() -> None:
    with pytest.raises(ValidationError):
        DecisionTarget(kind="soft", probabilities={"success": float("nan"), "failure": 1.0})
    with pytest.raises(ValidationError):
        DecisionTarget(kind="soft", probabilities={"success": -0.5, "failure": 1.5})


def test_hard_target_must_concentrate_on_one_option() -> None:
    with pytest.raises(ValidationError):
        DecisionTarget(kind="hard", probabilities={"success": 0.5, "failure": 0.5})
    DecisionTarget(kind="hard", probabilities={"success": 1.0, "failure": 0.0})


def test_choice_needs_at_least_two_nonempty_options() -> None:
    from bandits.decide.dataset import DecisionExample, DecisionLineage

    def _example(options: dict[str, str], probabilities: dict[str, float]) -> None:
        DecisionExample(
            decision_id="decision-1",
            family_id="f",
            state="s",
            question="q",
            primitive="choice",
            options=options,
            target=DecisionTarget(kind="soft", probabilities=probabilities),
            label_source="judge_votes",
            split="within_family_fit",
            lineage=DecisionLineage(source_kind="k", record_id="r"),
        )

    with pytest.raises(ValidationError):
        _example({"a": "only one"}, {"a": 1.0})
    with pytest.raises(ValidationError):
        _example({"a": "", "b": "fine"}, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValidationError):
        _example({"a": "fine", "b": "fine"}, {"a": 1.0})  # target missing option "b"


def test_dataset_rejects_a_row_that_does_not_match_its_shared_schema() -> None:
    from bandits.decide.dataset import (
        DecisionDataset,
        DecisionDatasetCounts,
        DecisionExample,
        DecisionLineage,
    )

    mismatched = DecisionExample(
        decision_id="decision-1",
        family_id="f",
        state="s",
        question="a different question",
        primitive="choice",
        options={"a": "x", "b": "y"},
        target=DecisionTarget(kind="soft", probabilities={"a": 1.0, "b": 0.0}),
        label_source="judge_votes",
        split="within_family_fit",
        lineage=DecisionLineage(source_kind="k", record_id="r"),
    )
    with pytest.raises(ValidationError, match="do not match decision_schema"):
        DecisionDataset(
            producer="test",
            source_artifact_ids=("x",),
            decision_schema=DecisionSchema(
                primitive="choice", options=dict(ACTION_OUTCOME_OPTIONS), question=ACTION_OUTCOME_QUESTION
            ),
            examples=(mismatched,),
            counts=DecisionDatasetCounts(
                examples=1,
                within_family_fit=1,
                within_family_held_out=0,
                quarantined=0,
                votes_requested=1,
                votes_valid_min=1,
                votes_valid_max=1,
            ),
        )


def test_dataset_rejects_a_wrong_examples_count() -> None:
    from bandits.decide.dataset import DecisionDataset, DecisionDatasetCounts

    with pytest.raises(ValidationError, match="counts.examples"):
        DecisionDataset(
            producer="test",
            source_artifact_ids=("x",),
            decision_schema=DecisionSchema(
                primitive="choice", options=dict(ACTION_OUTCOME_OPTIONS), question=ACTION_OUTCOME_QUESTION
            ),
            examples=(),
            counts=DecisionDatasetCounts(
                examples=5,  # claims 5 rows while examples=() has none
                within_family_fit=0,
                within_family_held_out=0,
                quarantined=0,
                votes_requested=1,
                votes_valid_min=0,
                votes_valid_max=0,
            ),
        )


def test_source_from_trace_matches_extract_turns() -> None:
    trace = _trace("a")
    source = source_from_trace(trace)
    assert source.trace_id == "a"
    assert source.task == "fix the bug"
    assert source.turns == extract_turns(trace)


def test_dataset_round_trips(tmp_path) -> None:
    trace = _trace("a")
    verdicts = _all_observed_verdicts(trace)
    run = _run([trace], verdicts)
    dataset = build_decision_dataset_from_corpus([trace], run, "judge-run-1")

    store = DerivedStore(tmp_path)
    envelope = save_decision_dataset(dataset, store)
    loaded = load_decision_dataset(envelope.artifact_id, store)
    assert loaded == dataset
    assert envelope.summary["examples"] == dataset.counts.examples
