from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from bandits.analyze import analyze_corpus, compute_analysis_id, mine_task_set
from bandits.analyze.models import EvidenceKind
from bandits.ingest import load_corpus
from bandits.store import DerivedStore
from bandits.traces import Span, SpanKind, Trace, TraceCorpus
from bandits.verify import (
    VerifierMode,
    VerifierStatus,
    draft_verifiers,
    load_verifier_draft,
    save_verifier_draft,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _test_distance(left: str, right: str) -> float:
    return 0.0 if left.partition(" ")[0] == right.partition(" ")[0] else 1.0


def _task_set(path: Path):
    analysis = analyze_corpus(load_corpus(path, "otlp"))
    analysis_id = compute_analysis_id(analysis)
    return analysis, mine_task_set(
        analysis, analysis_id, distance=_test_distance, similarity=0.7, budget=10
    )


def test_coding_family_proposes_exit_code_check() -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.coding.otlp.jsonl")
    family = next(f for f in task_set.families if "parser" in f.descriptor)

    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)

    spec = next(v for v in draft.verifiers if v.checks[0].claim == "command_exit_code")
    assert spec.mode is VerifierMode.REPLAY
    assert spec.status is VerifierStatus.EXECUTABLE
    assert spec.checks[0].expected == 0
    assert spec.blind_spots
    assert spec.gaming_hypotheses


def test_support_family_proposes_structured_state_check() -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.support.otlp.jsonl")
    family = next(f for f in task_set.families if "refund" in f.descriptor)

    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)

    checks = [v.checks[0] for v in draft.verifiers]
    assert any(check.claim == "final_state_field:status" for check in checks)
    assert all(check.supporting_evidence_ids for check in checks)


def test_no_deterministic_signal_is_explicitly_unresolved() -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.support.otlp.jsonl")
    family = next(f for f in task_set.families if "help" in f.descriptor)

    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)

    assert draft.verifiers == ()
    assert any("no deterministic" in reason for reason in draft.unresolved)


def test_draft_round_trips_with_task_set_as_parent(tmp_path) -> None:
    analysis, task_set = _task_set(FIXTURES / "traces.support.otlp.jsonl")
    family = next(f for f in task_set.families if "refund" in f.descriptor)
    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id)
    store = DerivedStore(tmp_path / ".bandits")

    envelope = save_verifier_draft(draft, store)

    assert envelope.parent_artifact_id == "taskset-test"
    assert envelope.kind == "verifier_draft"
    assert load_verifier_draft(envelope.artifact_id, store) == draft


def test_exact_output_is_proposed_only_when_the_instruction_requires_exact_text() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    corpus = TraceCorpus(
        source="test",
        traces=(
            Trace(
                trace_id="exact-one",
                source="test",
                source_digest="a" * 64,
                task="Return exactly OK",
                spans=(
                    Span(
                        span_id="reply",
                        kind=SpanKind.MODEL,
                        name="assistant",
                        started_at=now,
                        ended_at=now,
                        output="OK",
                    ),
                ),
            ),
        ),
    )
    analysis = analyze_corpus(corpus)
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=1
    )

    draft = draft_verifiers(task_set, "taskset-test", analysis, task_set.families[0].family_id)

    assert draft.verifiers[0].checks[0].operator.value == "exact_output"
    assert "agent output" in draft.verifiers[0].blind_spots[0]


def test_a_recorded_evaluator_score_becomes_a_check(tmp_path) -> None:
    """It is the strongest evidence extraction produces, and was previously unusable."""
    source = tmp_path / "scored.jsonl"
    source.write_text(
        "\n".join(
            f'{{"trace_id": "s-{i}", "span_id": "s-{i}-0", "parent_span_id": null,'
            f' "name": "gpt-5", "start_time": "2026-04-01T00:0{i}:00Z",'
            f' "end_time": "2026-04-01T00:0{i}:30Z", "attributes":'
            f' {{"gen_ai.operation.name": "chat", "task": "Summarize ticket {i}",'
            f' "gen_ai.completion": "done", "score": 1.0}}}}'
            for i in range(1, 5)
        )
        + "\n"
    )
    analysis = analyze_corpus(load_corpus(source, "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=5
    )
    family = task_set.families[0]

    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id)
    claims = {check.claim: check for spec in draft.verifiers for check in spec.checks}

    assert "recorded_score:score" in claims
    assert claims["recorded_score:score"].evidence_kind is EvidenceKind.TRUSTED_EVALUATOR


def test_stronger_evidence_survives_the_draft_limit(tmp_path) -> None:
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        '{"trace_id": "m-1", "span_id": "m-1-0", "parent_span_id": null, "name": "gpt-5",'
        ' "start_time": "2026-04-01T00:00:00Z", "end_time": "2026-04-01T00:00:30Z",'
        ' "attributes": {"gen_ai.operation.name": "chat", "task": "Run the suite",'
        ' "gen_ai.completion": "all green", "score": 1.0}}\n'
        '{"trace_id": "m-1", "span_id": "m-1-1", "parent_span_id": "m-1-0", "name": "pytest",'
        ' "start_time": "2026-04-01T00:00:31Z", "end_time": "2026-04-01T00:00:40Z",'
        ' "attributes": {"gen_ai.operation.name": "execute_tool",'
        ' "gen_ai.tool.call.arguments": {}, "gen_ai.tool.call.result": {"exit_code": 0}}}\n'
    )
    analysis = analyze_corpus(load_corpus(source, "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=5
    )

    draft = draft_verifiers(task_set, "ts-1", analysis, task_set.families[0].family_id, limit=1)

    kept = draft.verifiers[0].checks[0]
    assert kept.evidence_kind is EvidenceKind.STRUCTURED_EXTERNAL_RESULT


def _varying_amounts(tmp_path) -> Path:
    """Six refunds whose amounts all differ — the shape real corpora actually have."""
    rows = []
    for i in range(1, 7):
        base = f"2026-05-01T00:0{i}"
        rows += [
            f'{{"trace_id": "v-{i}", "span_id": "v-{i}-0", "parent_span_id": null,'
            f' "name": "gpt-5", "start_time": "{base}:00Z", "end_time": "{base}:10Z",'
            f' "attributes": {{"gen_ai.operation.name": "chat", "task": "Refund order {7000 + i}"}}}}',
            f'{{"trace_id": "v-{i}", "span_id": "v-{i}-1", "parent_span_id": "v-{i}-0",'
            f' "name": "lookup_order", "start_time": "{base}:11Z", "end_time": "{base}:12Z",'
            f' "attributes": {{"gen_ai.operation.name": "execute_tool",'
            f' "gen_ai.tool.call.arguments": {{}},'
            f' "gen_ai.tool.call.result": {{"charged_amount": {10.0 * i}, "status": "paid"}}}}}}',
            f'{{"trace_id": "v-{i}", "span_id": "v-{i}-2", "parent_span_id": "v-{i}-0",'
            f' "name": "refund_order", "start_time": "{base}:13Z", "end_time": "{base}:14Z",'
            f' "attributes": {{"gen_ai.operation.name": "execute_tool",'
            f' "gen_ai.tool.call.arguments": {{}},'
            f' "gen_ai.tool.call.result": {{"refunded_amount": {10.0 * i}, "status": "refunded"}}}}}}',
        ]
    path = tmp_path / "varying.jsonl"
    path.write_text("\n".join(rows) + "\n")
    return path


def _drafted(path, limit: int = 8):
    analysis = analyze_corpus(load_corpus(path, "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=5
    )
    return draft_verifiers(task_set, "ts-1", analysis, task_set.families[0].family_id, limit=limit)


def test_an_invariant_is_drafted_from_the_before_and_after_pair(tmp_path) -> None:
    claims = {
        check.claim
        for spec in _drafted(_varying_amounts(tmp_path)).verifiers
        for check in spec.checks
    }

    assert "invariant:refunded_amount==charged_amount" in claims


def test_a_field_that_differs_every_run_gets_no_equality_check(tmp_path) -> None:
    """Otherwise a corpus of N refunds drafts N verifiers, each passing only its own."""
    claims = {
        check.claim
        for spec in _drafted(_varying_amounts(tmp_path)).verifiers
        for check in spec.checks
    }

    assert "final_state_field:refunded_amount" not in claims
    assert "final_state_field:status" in claims, "a real status must survive"


def test_competing_values_for_one_field_are_both_drafted() -> None:
    """Two hypotheses for what success looks like is the question worth asking."""
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    analysis = analyze_corpus(load_corpus(fixtures / "traces.support.otlp.jsonl", "otlp"))
    task_set = mine_task_set(
        analysis, compute_analysis_id(analysis), distance=_test_distance, similarity=0.7, budget=10
    )
    address = next(f for f in task_set.families if "address" in f.descriptor)

    draft = draft_verifiers(task_set, "ts-1", analysis, address.family_id, limit=6)
    expected = {
        check.expected
        for spec in draft.verifiers
        for check in spec.checks
        if check.claim == "final_state_field:status"
    }

    assert expected == {"overridden", "paid"}
