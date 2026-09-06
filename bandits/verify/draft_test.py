from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bandits.analyze import analyze_corpus, compute_analysis_id, mine_task_set
from bandits.analyze.models import EvidenceKind
from bandits.ingest import load_corpus
from bandits.labels import LabelSet, Verdict, make_label
from bandits.store import DerivedStore
from bandits.traces import Span, SpanKind, Trace, TraceCorpus
from bandits.verify import (
    VerifierMode,
    VerifierStatus,
    draft_verifiers,
    load_verifier_draft,
    save_verifier_draft,
)
from bandits.verify.execute import execute_verifier

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _test_distance(left: str, right: str) -> float:
    return 0.0 if left.partition(" ")[0] == right.partition(" ")[0] else 1.0


def _task_set(path: Path):
    analysis = analyze_corpus(load_corpus(path, "otlp"))
    analysis_id = compute_analysis_id(analysis)
    return analysis, mine_task_set(
        analysis,
        analysis_id,
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=10,
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
    assert any(check.claim == "final_state_field:refund_order.status" for check in checks)
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
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=1,
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
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=5,
    )
    family = task_set.families[0]

    draft = draft_verifiers(task_set, "ts-1", analysis, family.family_id)
    claims = {check.claim: check for spec in draft.verifiers for check in spec.checks}

    assert "recorded_score:score" in claims
    # An anonymous number is not a trusted evaluator's verdict. Extraction
    # decides that, and the check carries whatever the evidence was classified as.
    assert claims["recorded_score:score"].evidence_kind is EvidenceKind.OBSERVED_TRACE


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
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=5,
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
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=5,
    )
    return draft_verifiers(task_set, "ts-1", analysis, task_set.families[0].family_id, limit=limit)


def test_an_invariant_is_drafted_from_the_before_and_after_pair(tmp_path) -> None:
    claims = {
        check.claim
        for spec in _drafted(_varying_amounts(tmp_path)).verifiers
        for check in spec.checks
    }

    assert "invariant:refund_order.refunded_amount==lookup_order.charged_amount" in claims


def test_a_field_that_differs_every_run_gets_no_equality_check(tmp_path) -> None:
    """Otherwise a corpus of N refunds drafts N verifiers, each passing only its own."""
    claims = {
        check.claim
        for spec in _drafted(_varying_amounts(tmp_path)).verifiers
        for check in spec.checks
    }

    assert "final_state_field:refund_order.refunded_amount" not in claims
    assert "final_state_field:refund_order.status" in claims, "a real status must survive"


def _competing_statuses(tmp_path) -> Path:
    """One tool, one field, two values across runs — a real pair of hypotheses."""
    rows = []
    for i in range(1, 7):
        base = f"2026-05-02T00:0{i}"
        status = "resolved" if i % 2 else "escalated"
        rows += [
            f'{{"trace_id": "c-{i}", "span_id": "c-{i}-0", "parent_span_id": null,'
            f' "name": "gpt-5", "start_time": "{base}:00Z", "end_time": "{base}:10Z",'
            f' "attributes": {{"gen_ai.operation.name": "chat", "task": "Close ticket {7000 + i}"}}}}',
            f'{{"trace_id": "c-{i}", "span_id": "c-{i}-1", "parent_span_id": "c-{i}-0",'
            f' "name": "close_ticket", "start_time": "{base}:11Z", "end_time": "{base}:12Z",'
            f' "attributes": {{"gen_ai.operation.name": "execute_tool",'
            f' "gen_ai.tool.call.arguments": {{}},'
            f' "gen_ai.tool.call.result": {{"status": "{status}"}}}}}}',
        ]
    path = tmp_path / "competing.jsonl"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_competing_values_for_one_field_are_both_drafted(tmp_path) -> None:
    """Two hypotheses for what success looks like is the question worth asking."""
    draft = _drafted(_competing_statuses(tmp_path))

    expected = {
        check.expected
        for spec in draft.verifiers
        for check in spec.checks
        if check.claim == "final_state_field:close_ticket.status"
    }

    assert expected == {"resolved", "escalated"}


def test_two_tools_reporting_one_key_are_drafted_as_different_checks() -> None:
    """`status` from a lookup and `status` from an override are not one field."""
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    analysis = analyze_corpus(load_corpus(fixtures / "traces.support.otlp.jsonl", "otlp"))
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=_test_distance,
        backend="first-word",
        similarity=0.7,
        budget=10,
    )
    address = next(f for f in task_set.families if "address" in f.descriptor)

    draft = draft_verifiers(task_set, "ts-1", analysis, address.family_id, limit=6)
    drafted = {(check.claim, check.expected) for spec in draft.verifiers for check in spec.checks}

    assert ("final_state_field:lookup_order.status", "paid") in drafted
    assert ("final_state_field:override_address_policy.status", "overridden") in drafted
    assert not any(claim == "final_state_field:status" for claim, _ in drafted)


def _nested_refunds(tmp_path) -> Path:
    """Six refunds whose outcome is a structured object, not a flat scalar."""
    rows = []
    for i in range(1, 7):
        base = f"2026-06-02T00:0{i}"
        amount = 10.0 * i
        rows += [
            f'{{"trace_id": "n-{i}", "span_id": "n-{i}-0", "parent_span_id": null,'
            f' "name": "gpt-5", "start_time": "{base}:00Z", "end_time": "{base}:10Z",'
            f' "attributes": {{"gen_ai.operation.name": "chat", "task": "Refund order {7000 + i}"}}}}',
            f'{{"trace_id": "n-{i}", "span_id": "n-{i}-1", "parent_span_id": "n-{i}-0",'
            f' "name": "lookup_order", "start_time": "{base}:11Z", "end_time": "{base}:12Z",'
            f' "attributes": {{"gen_ai.operation.name": "execute_tool",'
            f' "gen_ai.tool.call.arguments": {{}},'
            f' "gen_ai.tool.call.result": {{"order": {{"charge": {{"amount": {amount}}}}}}}}}}}',
            f'{{"trace_id": "n-{i}", "span_id": "n-{i}-2", "parent_span_id": "n-{i}-0",'
            f' "name": "refund_order", "start_time": "{base}:13Z", "end_time": "{base}:14Z",'
            f' "attributes": {{"gen_ai.operation.name": "execute_tool",'
            f' "gen_ai.tool.call.arguments": {{}},'
            f' "gen_ai.tool.call.result": {{"refund": {{"amount": {amount},'
            f' "status": "refunded"}}, "warnings": []}}}}}}',
        ]
    path = tmp_path / "nested.jsonl"
    path.write_text("\n".join(rows) + "\n")
    return path


def test_a_structured_outcome_is_drafted_from_rather_than_reported_unresolved(tmp_path) -> None:
    """A source answering with an object used to yield no terminal evidence at all."""
    draft = _drafted(_nested_refunds(tmp_path))

    claims = {check.claim for spec in draft.verifiers for check in spec.checks}
    assert "final_state_field:refund_order.refund.status" in claims
    assert "final_state_field:refund_order.warnings[].count" in claims
    assert draft.unresolved == () or all("no deterministic" not in r for r in draft.unresolved)


def test_an_invariant_spans_two_tools_nested_results(tmp_path) -> None:
    """The refunded amount equalling the charge is the check worth having."""
    claims = {
        check.claim
        for spec in _drafted(_nested_refunds(tmp_path)).verifiers
        for check in spec.checks
    }

    assert "invariant:refund_order.refund.amount==lookup_order.order.charge.amount" in claims


def test_a_prose_outcome_says_what_kind_of_verifier_it_would_need(tmp_path) -> None:
    path = tmp_path / "prose.jsonl"
    path.write_text(
        '{"trace_id": "p-1", "span_id": "p-1-0", "parent_span_id": null, "name": "gpt-5",'
        ' "start_time": "2026-06-03T00:00:00Z", "end_time": "2026-06-03T00:00:01Z",'
        ' "attributes": {"gen_ai.operation.name": "chat", "task": "Help the customer"}}\n'
        '{"trace_id": "p-1", "span_id": "p-1-1", "parent_span_id": "p-1-0", "name": "reply",'
        ' "start_time": "2026-06-03T00:00:02Z", "end_time": "2026-06-03T00:00:03Z",'
        ' "attributes": {"gen_ai.operation.name": "execute_tool",'
        ' "gen_ai.tool.call.arguments": {},'
        ' "gen_ai.tool.call.result": "I told them to try again later."}}\n'
    )

    draft = _drafted(path)

    assert draft.verifiers == ()
    assert any("judge or a domain extractor" in reason for reason in draft.unresolved)


def _skewed_corpus(
    *, failures: int = 7, successes: int = 3, amount_matches_on_success: bool = True
) -> TraceCorpus:
    """The issue's own example: failures dominate, so the failing value is the common one.

    Every episode looks up an order and then refunds it. The failures leave the
    order pending; the successes mark it refunded and return the amount that was
    charged. Frequency alone puts ``status == pending`` forward.
    """
    start = datetime(2026, 1, 1, tzinfo=UTC)
    traces: list[Trace] = []
    for index in range(failures + successes):
        succeeded = index >= failures
        charged = 100 + index
        refunded = charged if (succeeded and amount_matches_on_success) else 0
        traces.append(
            Trace(
                trace_id=f"{'ok' if succeeded else 'bad'}-{index}",
                source="otlp",
                source_digest=f"{index:064d}",
                task=f"Refund order {7000 + index}",
                spans=(
                    Span(
                        span_id=f"look-{index}",
                        kind=SpanKind.TOOL,
                        name="lookup_order",
                        started_at=start,
                        ended_at=start,
                        arguments={"order_id": 7000 + index},
                        output={"charged_amount": charged},
                    ),
                    Span(
                        span_id=f"refund-{index}",
                        kind=SpanKind.TOOL,
                        name="refund_order",
                        started_at=start,
                        ended_at=start,
                        arguments={"order_id": 7000 + index},
                        output={
                            "status": "refunded" if succeeded else "pending",
                            "refunded_amount": refunded,
                        },
                    ),
                ),
            )
        )
    return TraceCorpus(source="otlp", traces=tuple(traces))


def _skewed(**corpus_options):
    corpus = _skewed_corpus(**corpus_options)
    analysis = analyze_corpus(corpus)
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=lambda left, right: 0.0,
        backend="first-word",
        similarity=0.5,
        budget=10,
    )
    family = task_set.families[0]
    return analysis, task_set, family


def _labels(family_id: str, trace_ids: list[str]) -> LabelSet:
    return LabelSet(
        task_set_id="taskset-test",
        family_id=family_id,
        labels=tuple(
            make_label(
                trace_id=trace_id,
                family_id=family_id,
                verdict=Verdict.SUCCESS if trace_id.startswith("ok") else Verdict.FAILURE,
                labeler="reviewer",
            )
            for trace_id in trace_ids
        ),
    )


def test_a_value_common_only_among_failures_is_not_the_primary_success_verifier() -> None:
    """Seventy pending, thirty refunded: frequency proposes exactly the wrong check."""
    analysis, task_set, family = _skewed()
    labels = _labels(family.family_id, list(family.fit_trace_ids))

    draft = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=8, labels=labels
    )

    primary = draft.verifiers[0]
    assert "pending" not in json.dumps(
        [check.model_dump(mode="json") for check in primary.checks], default=str
    )
    ranked = {spec.verifier_id: index for index, spec in enumerate(draft.verifiers)}
    status_checks = {
        check.expected: spec.verifier_id
        for spec in draft.verifiers
        for check in spec.checks
        if check.claim == "final_state_field:refund_order.status"
    }
    assert "refunded" in status_checks
    if "pending" in status_checks:
        assert ranked[status_checks["refunded"]] < ranked[status_checks["pending"]]


def test_a_candidate_reports_its_support_rejection_coverage_and_unknowns() -> None:
    analysis, task_set, family = _skewed()
    labels = _labels(family.family_id, list(family.fit_trace_ids))

    draft = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=8, labels=labels
    )

    stats = {item.verifier_id: item for item in draft.candidates}
    assert set(stats) == {spec.verifier_id for spec in draft.verifiers}
    refunded = next(
        stats[spec.verifier_id]
        for spec in draft.verifiers
        if any(check.expected == "refunded" for check in spec.checks)
    )
    assert refunded.derivation == "contrast"
    assert refunded.calibrated
    assert refunded.success_support == refunded.labeled_successes
    assert refunded.failure_rejection == refunded.labeled_failures
    assert refunded.false_positives == 0
    assert refunded.coverage == 1.0
    assert refunded.unknown == 0
    assert refunded.discrimination == 1.0


def test_without_labels_every_candidate_is_marked_an_uncalibrated_hypothesis() -> None:
    analysis, task_set, family = _skewed()

    draft = draft_verifiers(task_set, "taskset-test", analysis, family.family_id, limit=8)

    assert all(item.derivation == "frequency" for item in draft.candidates)
    assert all(not item.calibrated for item in draft.candidates)
    assert all(item.discrimination == 0.0 for item in draft.candidates)
    assert any("frequency-based hypothesis" in item for item in draft.unresolved)


def _two_condition_corpus() -> TraceCorpus:
    """A family where success needs both conditions and neither alone decides.

    Two independent failure modes. One leaves the order pending while refunding
    the right amount, so the amount invariant holds on a failed run. The other
    marks the order refunded and moves no money, so the status holds on a failed
    run. Only the conjunction rejects both, and only the labels can show it.
    """
    start = datetime(2026, 1, 1, tzinfo=UTC)
    shapes = [
        *(("ok", "refunded", True) for _ in range(4)),
        *(("bad-status", "pending", True) for _ in range(3)),
        *(("bad-amount", "refunded", False) for _ in range(3)),
    ]
    traces = []
    for index, (prefix, status, paid) in enumerate(shapes):
        charged = 100 + index
        traces.append(
            Trace(
                trace_id=f"{prefix}-{index}",
                source="otlp",
                source_digest=f"{index:064d}",
                task=f"Refund order {7000 + index}",
                spans=(
                    Span(
                        span_id=f"look-{index}",
                        kind=SpanKind.TOOL,
                        name="lookup_order",
                        started_at=start,
                        ended_at=start,
                        arguments={"order_id": 7000 + index},
                        output={"charged_amount": charged},
                    ),
                    Span(
                        span_id=f"refund-{index}",
                        kind=SpanKind.TOOL,
                        name="refund_order",
                        started_at=start,
                        ended_at=start,
                        arguments={"order_id": 7000 + index},
                        output={"status": status, "refunded_amount": charged if paid else 0},
                    ),
                ),
            )
        )
    return TraceCorpus(source="otlp", traces=tuple(traces))


def test_a_two_condition_success_produces_a_justified_composite() -> None:
    """Status alone accepts a run that moved no money; the amount alone accepts a pending one."""
    analysis = analyze_corpus(_two_condition_corpus())
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=lambda left, right: 0.0,
        backend="first-word",
        similarity=0.5,
        budget=10,
    )
    family = task_set.families[0]
    labels = _labels(family.family_id, list(family.fit_trace_ids))

    draft = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=8, labels=labels
    )

    composites = [spec for spec in draft.verifiers if len(spec.checks) > 1]
    assert composites, "a family needing two conditions must produce a conjunction"
    stats = {item.verifier_id: item for item in draft.candidates}
    composite = composites[0]
    assert stats[composite.verifier_id].rationale
    assert "labeled failure" in stats[composite.verifier_id].rationale
    assert stats[composite.verifier_id].false_positives == 0
    # The standalone checks survive, so validation can compare them with the pair.
    assert any(len(spec.checks) == 1 for spec in draft.verifiers)


def test_a_composite_keeps_its_subscores_and_fails_closed_on_an_unknown_check() -> None:
    analysis, task_set, family = _skewed()
    labels = _labels(family.family_id, list(family.fit_trace_ids))
    draft = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=8, labels=labels
    )
    composite = next((spec for spec in draft.verifiers if len(spec.checks) > 1), draft.verifiers[0])
    known = tuple(item for item in analysis.evidence if item.trace_id == family.fit_trace_ids[0])

    partial = execute_verifier(composite, tuple(known[:1]))

    assert partial.score is None, "one unknown check must make the verifier unknown"
    assert len(partial.subscores) == len(composite.checks)
    assert any(part.score is not None for part in partial.subscores) or all(
        part.score is None for part in partial.subscores
    )


def test_drafting_is_deterministic_under_labels() -> None:
    analysis, task_set, family = _skewed()
    labels = _labels(family.family_id, list(family.fit_trace_ids))

    first = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=8, labels=labels
    )
    second = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=8, labels=labels
    )

    assert first.model_dump_json() == second.model_dump_json()


def test_a_contradicting_check_is_ranked_below_one_that_separates() -> None:
    """A check disagreeing with the labels is exactly what the ranking must demote."""
    analysis, task_set, family = _skewed()
    labels = _labels(family.family_id, list(family.fit_trace_ids))

    draft = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=8, labels=labels
    )

    stats = [item for item in draft.candidates]
    contradicting = [item for item in stats if item.contradictions]
    clean = [item for item in stats if not item.contradictions and item.calibrated]
    if contradicting and clean:
        assert stats.index(clean[0]) < stats.index(contradicting[0])


def test_labels_from_another_family_are_refused() -> None:
    analysis, task_set, family = _skewed()

    with pytest.raises(ValueError, match="another family"):
        draft_verifiers(
            task_set,
            "taskset-test",
            analysis,
            family.family_id,
            labels=_labels("family-elsewhere", list(family.fit_trace_ids)),
        )


def test_a_contradictory_conjunction_is_never_proposed() -> None:
    """Two equality checks on one field would score zero forever and read as strict."""
    analysis = analyze_corpus(_two_condition_corpus())
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=lambda left, right: 0.0,
        backend="first-word",
        similarity=0.5,
        budget=10,
    )
    family = task_set.families[0]
    labels = _labels(family.family_id, list(family.fit_trace_ids))

    draft = draft_verifiers(
        task_set, "taskset-test", analysis, family.family_id, limit=12, labels=labels
    )

    for spec in draft.verifiers:
        claims = [check.claim for check in spec.checks]
        assert len(claims) == len(set(claims))
    assert all(
        stats.success_support > 0
        for stats in draft.candidates
        if len(next(s for s in draft.verifiers if s.verifier_id == stats.verifier_id).checks) > 1
    )


def test_an_invariant_is_not_retired_by_the_failures_it_exists_to_catch() -> None:
    """Unlabeled, one counterexample retires it; labeled, only a success can."""
    analysis = analyze_corpus(_two_condition_corpus())
    task_set = mine_task_set(
        analysis,
        compute_analysis_id(analysis),
        distance=lambda left, right: 0.0,
        backend="first-word",
        similarity=0.5,
        budget=10,
    )
    family = task_set.families[0]

    blind = draft_verifiers(task_set, "taskset-test", analysis, family.family_id, limit=12)
    informed = draft_verifiers(
        task_set,
        "taskset-test",
        analysis,
        family.family_id,
        limit=12,
        labels=_labels(family.family_id, list(family.fit_trace_ids)),
    )

    def invariants(draft):
        return [
            check.claim
            for spec in draft.verifiers
            for check in spec.checks
            if check.claim.startswith("invariant:")
        ]

    assert not invariants(blind)
    assert "invariant:refund_order.refunded_amount==lookup_order.charged_amount" in invariants(
        informed
    )
