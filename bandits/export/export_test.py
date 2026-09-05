from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from bandits.analyze import analyze_corpus, compute_analysis_id, compute_task_set_id
from bandits.analyze.models import TaskFamily, TaskSet
from bandits.export import (
    Partition,
    SFTExample,
    ToolCall,
    ToolFunction,
    TrainingMessage,
    build_eval_export,
    build_sft_export,
    build_transcript,
    load_export,
    save_export,
    write_jsonl,
)
from bandits.export.sft import _quality_reasons
from bandits.ingest.chat_json import load_chat_json
from bandits.ingest.claude_code import load_claude_code
from bandits.ingest.otlp import load_otlp
from bandits.store import DerivedStore, compute_artifact_id
from bandits.traces import Span, SpanKind, SpanStatus, Trace, TraceCorpus, UserTurn
from bandits.verify.models import (
    CheckOperator,
    CheckSpec,
    VerifierDraft,
    VerifierMode,
    VerifierSpec,
    VerifierStatus,
)
from bandits.verify.review import compute_reviewed_verifier_id, review_verifier
from bandits.verify.validate import Agreement, Validation


def _trace(
    trace_id: str,
    order: int,
    status: str,
    *,
    tool_status: SpanStatus = SpanStatus.OK,
    extra_models: int = 0,
    carrier_args: bool = False,
) -> Trace:
    """One episode: a model turn that calls a tool, the result, then a closing turn.

    ``carrier_args`` switches between the two shapes real adapters produce. With
    it, the call arguments sit on the model span and the tool span holds only the
    result, as the chat-JSON and Claude Code adapters record them; without it the
    tool span carries its own arguments, as OTLP does.
    """
    started = datetime(2026, 1, 1, tzinfo=UTC)
    call_span_id = f"{trace_id}-call"
    spans: list[Span] = [
        Span(
            span_id=call_span_id,
            kind=SpanKind.MODEL,
            name="change_order" if carrier_args else "model-v1",
            started_at=started,
            ended_at=started + timedelta(seconds=1),
            # In the OTLP shape this is the prompt, not a call. A builder that
            # reads it as call arguments would export the prompt as the action.
            arguments={"order_id": order} if carrier_args else {"prompt": "change it"},
            attributes={"tool_call": True} if carrier_args else {},
        )
    ]
    for index in range(extra_models):
        spans.append(
            Span(
                span_id=f"{trace_id}-extra-{index}",
                parent_span_id=call_span_id,
                kind=SpanKind.MODEL,
                name="model-v1",
                started_at=started + timedelta(seconds=2 + index),
                ended_at=started + timedelta(seconds=3 + index),
                output=f"extra {index}",
            )
        )
    spans.append(
        Span(
            span_id=f"{trace_id}-tool",
            parent_span_id=call_span_id,
            kind=SpanKind.TOOL,
            name="change_order",
            started_at=started + timedelta(seconds=20),
            ended_at=started + timedelta(seconds=21),
            status=tool_status,
            arguments={} if carrier_args else {"order_id": order},
            output={"status": status, "order_id": order},
        )
    )
    spans.append(
        Span(
            span_id=f"{trace_id}-final",
            kind=SpanKind.MODEL,
            name="model-v1",
            started_at=started + timedelta(seconds=22),
            ended_at=started + timedelta(seconds=23),
            output="completed",
        )
    )
    return Trace(
        trace_id=trace_id,
        source="otlp",
        source_digest=trace_id.ljust(64, "0"),
        task=f"Change order {order}",
        spans=tuple(spans),
    )


@pytest.fixture
def export_case():
    corpus = TraceCorpus(
        source="otlp",
        traces=(
            _trace("good-1", 100, "changed"),
            _trace("good-2", 200, "changed"),
            _trace("failed", 300, "pending"),
            _trace("recovered", 400, "changed", tool_status=SpanStatus.ERROR),
        ),
    )
    analysis = analyze_corpus(corpus)
    analysis_id = compute_analysis_id(analysis)
    family = TaskFamily(
        family_id="family-change",
        descriptor="change order <id>",
        trace_ids=tuple(trace.trace_id for trace in corpus.traces),
        medoid_trace_id="good-1",
        workload_mass=4,
        fit_trace_ids=("good-1", "failed"),
        held_out_trace_ids=("good-2", "recovered"),
    )
    task_set = TaskSet(
        corpus_id=compute_artifact_id(corpus),
        analysis_id=analysis_id,
        families=(family,),
        selected=(),
        total_workload_mass=4,
        workload_coverage=1,
    )
    task_set_id = compute_task_set_id(task_set)
    evidence = tuple(item for item in analysis.evidence if item.claim == "final_state_field")
    spec = VerifierSpec(
        verifier_id="verifier-status-changed",
        family_id=family.family_id,
        task_set_id=task_set_id,
        mode=VerifierMode.REPLAY,
        status=VerifierStatus.EXECUTABLE,
        inputs=("terminal_evidence:final_state_field",),
        checks=(
            CheckSpec(
                check_id="check-status",
                claim="final_state_field:status",
                operator=CheckOperator.EQUALS,
                expected="changed",
                supporting_evidence_ids=tuple(item.evidence_id for item in evidence),
                description="Terminal status changed.",
            ),
        ),
        unknown_when=("terminal status absent",),
        blind_spots=("status can be forged",),
        gaming_hypotheses=("write status directly",),
    )
    draft = VerifierDraft(
        task_set_id=task_set_id,
        analysis_id=analysis_id,
        family_id=family.family_id,
        verifiers=(spec,),
    )
    validation = Validation(
        source_draft_id="draft-1",
        family_id=family.family_id,
        label_set_id="labels-1",
        agreements=(
            Agreement(
                verifier_id=spec.verifier_id,
                split="fit",
                labeled=2,
                agreed=2,
                disagreed=0,
                unscored=0,
                agreement=1,
            ),
            Agreement(
                verifier_id=spec.verifier_id,
                split="held_out",
                labeled=2,
                agreed=2,
                disagreed=0,
                unscored=0,
                agreement=1,
            ),
        ),
        labels_used=4,
        success_labels=3,
        failure_labels=1,
    )
    reviewed = review_verifier(
        draft, "draft-1", validation, "validation-1", spec.verifier_id, "owner-ticket-7"
    )
    return corpus, analysis, task_set, task_set_id, draft, validation, reviewed


def test_eval_exports_every_prompt_safe_case_with_full_lineage(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    reviewed_id = compute_reviewed_verifier_id(reviewed)

    bundle = build_eval_export(corpus, task_set, task_set_id, analysis, reviewed, reviewed_id)

    # Held out by default: these are the traces SFT is not allowed to train on.
    assert bundle.manifest.partition is Partition.HELD_OUT
    assert {row.trace_id for row in bundle.rows} == {"good-2", "recovered"}
    assert bundle.manifest.success_threshold == reviewed.success_threshold
    assert bundle.manifest.verifier_status == "reviewed"
    assert not bundle.unresolved
    assert all(row.corpus_id == task_set.corpus_id for row in bundle.rows)
    assert all(row.verifier_id == reviewed.spec.verifier_id for row in bundle.rows)
    assert all(row.grader["status"] == "reviewed" for row in bundle.rows)


def test_sft_requires_success_and_imitation_quality_and_deduplicates(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    reviewed_id = compute_reviewed_verifier_id(reviewed)

    bundle = build_sft_export(
        corpus, task_set, task_set_id, analysis, reviewed, reviewed_id, partition=Partition.ALL
    )

    assert [row.trace_id for row in bundle.rows] == ["good-1"]
    rejected = {item.trace_id: item.reasons for item in bundle.unresolved}
    assert any("near-duplicate" in reason for reason in rejected["good-2"])
    assert any("did not establish success" in reason for reason in rejected["failed"])
    assert any("error or recovery" in reason for reason in rejected["recovered"])
    assert bundle.rows[0].messages[0].role == "user"
    assert bundle.rows[0].generating_policy["models"] == ("model-v1",)


def test_export_rejects_cross_artifact_lineage(export_case) -> None:
    corpus, analysis, task_set, _, _, _, reviewed = export_case

    with pytest.raises(ValueError, match="another task set"):
        build_eval_export(corpus, task_set, "taskset-wrong", analysis, reviewed, "reviewed-1")


def test_jsonl_writer_is_deterministic_and_always_writes_quarantine(export_case, tmp_path) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    bundle = build_sft_export(
        corpus,
        task_set,
        task_set_id,
        analysis,
        reviewed,
        compute_reviewed_verifier_id(reviewed),
    )
    output = tmp_path / "sft.jsonl"

    accepted, unresolved = write_jsonl(bundle, output)
    first = accepted.read_bytes(), unresolved.read_bytes()
    write_jsonl(bundle, output)

    assert first == (accepted.read_bytes(), unresolved.read_bytes())
    assert len(accepted.read_text().splitlines()) == len(bundle.rows)
    assert len(unresolved.read_text().splitlines()) == len(bundle.unresolved)
    assert json.loads(accepted.read_text().splitlines()[0])["trace_id"] == "good-1"


def test_export_artifact_round_trips_and_records_review_parent(export_case, tmp_path) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    reviewed_id = compute_reviewed_verifier_id(reviewed)
    bundle = build_eval_export(corpus, task_set, task_set_id, analysis, reviewed, reviewed_id)
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_export(bundle, store)

    assert envelope.parent_artifact_id == reviewed_id
    assert load_export(envelope.artifact_id, store) == bundle


def test_unknown_verifier_result_is_quarantined_not_failed_or_exported(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    missing = reviewed.spec.checks[0].replace(claim="final_state_field:not_recorded")
    blind = reviewed.replace(spec=reviewed.spec.replace(checks=(missing,)))

    bundle = build_sft_export(corpus, task_set, task_set_id, analysis, blind, "reviewed-blind")

    assert not bundle.rows
    assert all(
        any("could not score" in reason for reason in rejected.reasons)
        for rejected in bundle.unresolved
    )


def test_non_start_prompt_evidence_is_quarantined(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    terminal_id = next(
        item.evidence_id
        for item in analysis.evidence
        if item.trace_id == "good-1" and item.claim == "final_state_field"
    )
    tasks = tuple(
        task.replace(prompt_evidence_ids=(terminal_id,)) if task.trace_id == "good-1" else task
        for task in analysis.tasks
    )
    unsafe = analysis.replace(tasks=tasks)

    bundle = build_eval_export(
        corpus,
        task_set,
        task_set_id,
        unsafe,
        reviewed,
        compute_reviewed_verifier_id(reviewed),
        partition=Partition.ALL,
    )

    rejected = next(item for item in bundle.unresolved if item.trace_id == "good-1")
    assert any("unavailable at start" in reason for reason in rejected.reasons)


def test_transcript_puts_the_action_on_the_assistant_turn(export_case) -> None:
    """The call is a target, not context: it must not arrive as a tool message."""
    messages, defects, warnings = build_transcript(_trace("good-1", 100, "changed"))

    assert not defects and not warnings
    assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant"]
    call = messages[1].tool_calls[0]
    assert call.function.name == "change_order"
    assert json.loads(call.function.arguments) == {"order_id": 100}
    assert messages[2].tool_call_id == call.id
    assert not messages[2].tool_calls


def test_a_later_user_turn_is_replayed_where_it_arrived() -> None:
    """The correction has to sit before the action it caused, not be dropped."""
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]

    messages, defects, warnings = build_transcript(trace)

    assert not defects and not warnings
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2].content == "Use version 1.9 instead"
    assert json.loads(messages[3].tool_calls[0].function.arguments) == {"version": "1.9"}


def test_a_multi_turn_session_log_replays_its_correction_too() -> None:
    messages, defects, _ = build_transcript(load_claude_code(MULTI_TURN_SESSION).traces[0])

    assert not defects
    assert [message.content for message in messages if message.role == "user"] == [
        "Update the dependency",
        "Use version 1.9 instead",
    ]


def test_a_trace_that_lost_a_user_turn_is_refused() -> None:
    """Fail closed: the actions answered something this transcript cannot show."""
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]

    _, defects, _ = build_transcript(trace.replace(unrepresented_user_turns=1))

    assert any("does not represent" in defect for defect in defects)


def test_a_user_turn_with_no_recorded_response_is_refused() -> None:
    """A dangling instruction has no behavior to imitate after it."""
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]
    dangling = trace.replace(
        user_turns=(
            *trace.user_turns,
            UserTurn(text="also bump the lockfile", after_span_id=trace.spans[-1].span_id),
        )
    )

    _, defects, _ = build_transcript(dangling)

    assert any("no recorded response" in defect for defect in defects)


def test_a_user_turn_anchored_to_no_span_is_refused() -> None:
    trace = load_chat_json(MULTI_TURN_FIXTURE).traces[0]
    orphaned = trace.replace(
        user_turns=(*trace.user_turns, UserTurn(text="and pin it", after_span_id="not-a-span"))
    )

    _, defects, _ = build_transcript(orphaned)

    assert any("does not sit anywhere" in defect for defect in defects)


def test_a_single_turn_transcript_is_unchanged_by_recorded_turns() -> None:
    """A source that records its one user turn must export what it always did."""
    trace = _trace("good-1", 100, "changed")

    recorded = trace.replace(user_turns=(UserTurn(text=trace.task),))

    assert build_transcript(recorded) == build_transcript(trace)


@pytest.mark.parametrize("carrier_args", [False, True])
def test_both_recorded_call_shapes_yield_the_same_transcript(carrier_args: bool) -> None:
    """OTLP puts arguments on the tool span; chat JSON puts them on the model span."""
    messages, defects, _ = build_transcript(_trace("t", 100, "changed", carrier_args=carrier_args))

    assert not defects
    assert json.loads(messages[1].tool_calls[0].function.arguments) == {"order_id": 100}


def test_a_model_prompt_is_never_exported_as_the_tool_call(export_case) -> None:
    """The OTLP model span carries a prompt; only the tool span carries the action."""
    messages, _, _ = build_transcript(_trace("t", 100, "changed"))

    arguments = [json.loads(call.function.arguments) for call in messages[1].tool_calls]
    assert arguments == [{"order_id": 100}]
    assert all("prompt" not in item for item in arguments)


def test_transcript_reports_a_truncated_episode_instead_of_exporting_it() -> None:
    trace = _trace("t", 100, "changed")
    truncated = trace.replace(spans=trace.spans[:-1])

    messages, defects, warnings = build_transcript(truncated)

    # Not a defect: the actions are still exactly what the agent did.
    assert messages[-1].role == "tool"
    assert not defects
    assert any("closing turn" in warning for warning in warnings)


def test_transcript_reports_a_call_whose_result_was_never_recorded() -> None:
    trace = _trace("t", 100, "changed")
    spans = tuple(
        span.replace(output=None) if span.span_id == "t-tool" else span for span in trace.spans
    )

    _, defects, _ = build_transcript(trace.replace(spans=spans))

    assert any("no recorded result" in defect for defect in defects)


def test_sft_example_refuses_a_result_that_answers_no_call() -> None:
    with pytest.raises(ValidationError, match="answers no announced call"):
        SFTExample(
            example_id="sft-1",
            messages=(
                TrainingMessage(role="user", content="do it"),
                TrainingMessage(role="assistant", content="done"),
                TrainingMessage(role="tool", tool_call_id="call-ghost", content="{}"),
            ),
            generating_policy={"models": ("model-v1",)},
            corpus_id="corpus-1",
            task_set_id="taskset-1",
            family_id="family-1",
            trace_id="t",
            verifier_id="v",
            validation_id="validation-1",
        )


def test_sft_example_refuses_a_call_with_no_result() -> None:
    call = ToolCall(id="call-1", function=ToolFunction(name="change_order", arguments="{}"))
    with pytest.raises(ValidationError, match="no recorded result"):
        SFTExample(
            example_id="sft-1",
            messages=(
                TrainingMessage(role="user", content="do it"),
                TrainingMessage(role="assistant", tool_calls=(call,)),
            ),
            generating_policy={"models": ("model-v1",)},
            corpus_id="corpus-1",
            task_set_id="taskset-1",
            family_id="family-1",
            trace_id="t",
            verifier_id="v",
            validation_id="validation-1",
        )


def test_tool_message_may_not_announce_tool_calls() -> None:
    call = ToolCall(id="call-1", function=ToolFunction(name="x", arguments="{}"))
    with pytest.raises(ValidationError, match="only an assistant message"):
        TrainingMessage(role="tool", tool_call_id="call-1", content="{}", tool_calls=(call,))


def test_emitted_jsonl_is_clean_chat_completions(export_case, tmp_path) -> None:
    """The stored artifact stays lossless; the emitted row carries no empty padding."""
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    bundle = build_sft_export(
        corpus, task_set, task_set_id, analysis, reviewed, compute_reviewed_verifier_id(reviewed)
    )
    accepted, _ = write_jsonl(bundle, tmp_path / "sft.jsonl")

    messages = json.loads(accepted.read_text().splitlines()[0])["messages"]
    assert messages[0] == {"role": "user", "content": "Change order 100"}
    assert "tool_calls" not in messages[0]
    assert "content" not in messages[1] and messages[1]["tool_calls"][0]["type"] == "function"
    assert set(messages[2]) == {"role", "content", "name", "tool_call_id"}
    assert all(isinstance(message.get("content", ""), str) for message in messages)


def test_repeated_action_is_detected_when_arguments_live_on_the_model_span() -> None:
    """The carrier shape leaves tool spans argument-less; repeats must still be seen."""
    trace = _trace("t", 100, "changed", carrier_args=True)
    doubled = trace.replace(
        spans=trace.spans
        + tuple(span.replace(span_id=f"{span.span_id}-again") for span in trace.spans[:2])
    )
    messages, _, _ = build_transcript(doubled)

    assert len(_quality_reasons(doubled, messages, max_steps=99)) == 1
    assert "repeats the same tool action" in _quality_reasons(doubled, messages, 99)[0]


def test_step_count_gate_quarantines_a_long_episode(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    long_trace = _trace("good-1", 100, "changed", extra_models=6)
    grown = corpus.replace(
        traces=tuple(long_trace if t.trace_id == "good-1" else t for t in corpus.traces)
    )
    regrown = analyze_corpus(grown)
    retargeted = task_set.replace(
        corpus_id=compute_artifact_id(grown), analysis_id=compute_analysis_id(regrown)
    )

    bundle = build_sft_export(
        grown,
        retargeted,
        task_set_id,
        regrown,
        reviewed.replace(spec=reviewed.spec.replace(task_set_id=task_set_id)),
        "reviewed-long",
    )

    rejected = {item.trace_id: item.reasons for item in bundle.unresolved}
    assert any("family quality limit" in reason for reason in rejected["good-1"])


FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
SUPPORT_FIXTURE = FIXTURES / "traces.support.otlp.jsonl"
MULTI_TURN_FIXTURE = FIXTURES / "traces.multiturn.chat.json"
MULTI_TURN_SESSION = FIXTURES / "session.multiturn.jsonl"


def test_a_real_corpus_still_yields_trainable_rows() -> None:
    """A gate strict enough to quarantine every real trace is a broken gate.

    Every episode in this corpus ends on a tool result, because that is what the
    exporter recorded. Treating that as disqualifying rather than as a warning
    would silently reduce the whole export to nothing.
    """
    corpus = load_otlp(SUPPORT_FIXTURE)
    refunds = [trace for trace in corpus.traces if trace.trace_id.startswith("refund-")]

    exported = 0
    for trace in refunds:
        messages, defects, warnings = build_transcript(trace)
        if defects:
            continue
        exported += 1
        assert messages[-1].role == "tool"
        assert any("closing turn" in warning for warning in warnings)
        # lookup then refund, as two turns: the second call must follow the
        # observation that justified it, never ride alongside the first.
        calls = [call.function.name for message in messages for call in message.tool_calls]
        assert calls == ["lookup_order", "refund_order"]
        assert all(len(message.tool_calls) <= 1 for message in messages)

    assert exported == len(refunds) - 2, "only the two traces with no recorded result are defective"


def test_sequential_calls_are_never_batched_into_one_turn() -> None:
    """A shared parent span means 'emitted during', not 'issued together'."""
    trace = next(item for item in load_otlp(SUPPORT_FIXTURE).traces if item.trace_id == "refund-1")
    assert {span.parent_span_id for span in trace.spans[1:]} == {trace.spans[0].span_id}

    messages, _, _ = build_transcript(trace)

    announcing = [message for message in messages if message.tool_calls]
    assert len(announcing) == 2
    assert messages.index(announcing[1]) > messages.index(
        next(m for m in messages if m.role == "tool")
    )


def test_demonstrations_and_evaluation_cases_never_share_a_trace(export_case) -> None:
    """The property the whole split exists for: no task is both trained on and scored."""
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    reviewed_id = compute_reviewed_verifier_id(reviewed)

    trained = build_sft_export(corpus, task_set, task_set_id, analysis, reviewed, reviewed_id)
    scored = build_eval_export(corpus, task_set, task_set_id, analysis, reviewed, reviewed_id)

    assert trained.manifest.partition is Partition.FIT
    assert scored.manifest.partition is Partition.HELD_OUT
    assert not {row.trace_id for row in trained.rows} & {row.trace_id for row in scored.rows}


def test_held_out_traces_are_not_offered_to_training(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    family = task_set.families[0]

    bundle = build_sft_export(
        corpus, task_set, task_set_id, analysis, reviewed, compute_reviewed_verifier_id(reviewed)
    )

    considered = {row.trace_id for row in bundle.rows} | {
        item.trace_id for item in bundle.unresolved
    }
    assert not considered & set(family.held_out_trace_ids)
    assert bundle.manifest.partition_trace_count == len(family.fit_trace_ids)


def test_drawing_from_the_whole_family_is_warned_about(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case

    bundle = build_sft_export(
        corpus,
        task_set,
        task_set_id,
        analysis,
        reviewed,
        compute_reviewed_verifier_id(reviewed),
        partition=Partition.ALL,
    )

    assert any("overlap" in warning for warning in bundle.manifest.warnings)


def test_an_empty_partition_says_so_rather_than_looking_like_a_clean_run(export_case) -> None:
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    family = task_set.families[0]
    unsplit = task_set.replace(families=(family.replace(fit_trace_ids=(), held_out_trace_ids=()),))

    bundle = build_sft_export(
        corpus, unsplit, task_set_id, analysis, reviewed, compute_reviewed_verifier_id(reviewed)
    )

    assert not bundle.rows and not bundle.unresolved
    assert any("no fit traces" in warning for warning in bundle.manifest.warnings)


def test_export_cannot_re_choose_the_threshold_review_was_measured_at(export_case) -> None:
    """The threshold is frozen at review, and export has no way to override it."""
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    strict = reviewed.replace(success_threshold=0.99)

    bundle = build_sft_export(
        corpus, task_set, task_set_id, analysis, strict, "reviewed-strict", partition=Partition.ALL
    )

    assert bundle.manifest.success_threshold == 0.99

    # Not merely defaulted from the reviewed artifact — unreachable from the call
    # site, so no caller can admit a run the validation recorded as a failure.
    with pytest.raises(TypeError):
        build_sft_export(
            corpus,
            task_set,
            task_set_id,
            analysis,
            strict,
            "reviewed-strict",
            success_threshold=0.1,  # type: ignore[call-arg]
        )


def test_a_split_verdict_is_scored_against_the_frozen_threshold(export_case) -> None:
    """Two checks, one passing and one failing, score 0.5 — either side of a threshold."""
    corpus, analysis, task_set, task_set_id, _, _, reviewed = export_case
    passing = reviewed.spec.checks[0]
    failing = passing.replace(
        check_id="check-order", claim="final_state_field:order_id", expected=-1
    )
    composite = reviewed.replace(spec=reviewed.spec.replace(checks=(passing, failing)))

    lenient = build_sft_export(
        corpus,
        task_set,
        task_set_id,
        analysis,
        composite.replace(success_threshold=0.5),
        "reviewed-lenient",
        partition=Partition.ALL,
    )
    strict = build_sft_export(
        corpus,
        task_set,
        task_set_id,
        analysis,
        composite.replace(success_threshold=0.8),
        "reviewed-strict",
        partition=Partition.ALL,
    )

    assert [row.trace_id for row in lenient.rows] == ["good-1"]
    assert not strict.rows
    assert any(
        "did not establish success" in reason
        for item in strict.unresolved
        if item.trace_id == "good-1"
        for reason in item.reasons
    )
