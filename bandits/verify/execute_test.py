from __future__ import annotations

from bandits.analyze.models import Evidence, EvidenceKind, Visibility
from bandits.verify import (
    CheckOperator,
    CheckSpec,
    VerifierMode,
    VerifierSpec,
    execute_verifier,
)


def _evidence(claim: str, value: object) -> Evidence:
    return Evidence(
        evidence_id=f"ev-{claim}",
        claim=claim,
        value=value,
        visibility=Visibility.TERMINAL,
        provenance="observed",
        strength="strong",
        kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
        trace_id="trace-one",
    )


def _spec(operator: CheckOperator, claim: str, expected: object) -> VerifierSpec:
    return VerifierSpec(
        verifier_id="verifier-one",
        family_id="family-one",
        task_set_id="taskset-one",
        mode=VerifierMode.REPLAY,
        status="executable",
        inputs=(f"terminal_evidence:{claim}",),
        checks=(
            CheckSpec(
                check_id="check-one",
                claim=claim,
                operator=operator,
                expected=expected,
                supporting_evidence_ids=("ev-support",),
                description="A deterministic check.",
            ),
        ),
        unknown_when=("evidence absent",),
        blind_spots=("may be insufficient",),
        gaming_hypotheses=("fake the checked value",),
    )


def test_exit_code_check_executes() -> None:
    spec = _spec(CheckOperator.EXIT_CODE_ZERO, "command_exit_code", 0)

    passed = execute_verifier(
        spec, (_evidence("command_exit_code", {"key": "exit_code", "value": 0}),)
    )
    failed = execute_verifier(
        spec, (_evidence("command_exit_code", {"key": "exit_code", "value": 1}),)
    )

    assert passed.score == 1.0
    assert failed.score == 0.0


def test_structured_field_check_executes() -> None:
    spec = _spec(CheckOperator.EQUALS, "final_state_field:status", "refunded")
    evidence = _evidence(
        "final_state_field", {"key": "status", "value": "refunded", "tool": "refund"}
    )

    assert execute_verifier(spec, (evidence,)).score == 1.0


def test_a_check_is_not_answered_by_another_tools_field_of_the_same_name() -> None:
    """Two tools both reporting `status` are two facts, not one shared one."""
    spec = _spec(CheckOperator.EQUALS, "final_state_field:refund_order.status", "refunded")
    other_tool = _evidence(
        "final_state_field",
        {
            "key": "status",
            "field": "lookup_order.status",
            "value": "refunded",
            "tool": "lookup_order",
        },
    )

    assert execute_verifier(spec, (other_tool,)).score is None


def test_missing_required_evidence_is_unknown() -> None:
    spec = _spec(CheckOperator.EQUALS, "final_state_field:status", "refunded")

    result = execute_verifier(spec, ())

    assert result.score is None
    assert result.subscores[0].score is None


def test_exact_output_check_executes_without_semantic_guessing() -> None:
    spec = _spec(CheckOperator.EXACT_OUTPUT, "final_output", "OK")

    assert execute_verifier(spec, (_evidence("final_output", {"output": "OK"}),)).score == 1.0
    assert execute_verifier(spec, (_evidence("final_output", {"output": "Okay"}),)).score == 0.0


def test_a_recorded_evaluator_score_is_scored() -> None:
    spec = _spec(CheckOperator.EQUALS, "recorded_score:score", 1.0)

    matching = _evidence("recorded_score", {"key": "score", "value": 1.0})
    differing = _evidence("recorded_score", {"key": "score", "value": 0.0})

    assert execute_verifier(spec, (matching,)).score == 1.0
    assert execute_verifier(spec, (differing,)).score == 0.0


def test_a_score_under_another_key_is_not_this_checks_evidence() -> None:
    """Absent evidence for the named key is unknown, not a failure."""
    spec = _spec(CheckOperator.EQUALS, "recorded_score:score", 1.0)

    result = execute_verifier(spec, (_evidence("recorded_score", {"key": "rating", "value": 1.0}),))

    assert result.score is None


def test_an_unknown_check_keeps_the_breakdown_of_the_ones_that_ran() -> None:
    spec = VerifierSpec(
        verifier_id="verifier-two",
        family_id="family-one",
        task_set_id="taskset-one",
        mode=VerifierMode.REPLAY,
        inputs=(),
        checks=(
            CheckSpec(
                check_id="check-exit",
                claim="command_exit_code",
                operator=CheckOperator.EXIT_CODE_ZERO,
                supporting_evidence_ids=(),
                description="d",
            ),
            CheckSpec(
                check_id="check-state",
                claim="final_state_field:status",
                operator=CheckOperator.EQUALS,
                expected="refunded",
                supporting_evidence_ids=(),
                description="d",
            ),
        ),
        unknown_when=(),
        blind_spots=(),
        gaming_hypotheses=(),
    )

    result = execute_verifier(
        spec, (_evidence("command_exit_code", {"key": "exit_code", "value": 0}),)
    )

    assert result.score is None, "one unscorable check makes the aggregate unknown"
    breakdown = {part.check_id: part.score for part in result.subscores}
    assert breakdown == {"check-exit": 1.0, "check-state": None}
    assert result.details["unknown"] == ["check-state"]


def _state(claim: str, key: str, value: object) -> Evidence:
    return _evidence(claim, {"key": key, "value": value, "tool": "t"}).replace(
        evidence_id=f"ev-{claim}-{key}"
    )


def test_an_invariant_compares_terminal_state_against_initial_state() -> None:
    spec = _spec(CheckOperator.STATE_INVARIANT, "invariant:refunded_amount==charged_amount", None)

    full = (
        _state("initial_state_field", "charged_amount", 48.0),
        _state("final_state_field", "refunded_amount", 48.0),
    )
    partial = (
        _state("initial_state_field", "charged_amount", 48.0),
        _state("final_state_field", "refunded_amount", 20.0),
    )

    assert execute_verifier(spec, full).score == 1.0
    assert execute_verifier(spec, partial).score == 0.0, "a partial refund must not pass"


def test_an_invariant_without_its_initial_state_is_unknown() -> None:
    spec = _spec(CheckOperator.STATE_INVARIANT, "invariant:refunded_amount==charged_amount", None)

    result = execute_verifier(spec, (_state("final_state_field", "refunded_amount", 48.0),))

    assert result.score is None


def test_no_span_error_passes_a_clean_episode_and_fails_a_broken_one() -> None:
    spec = _spec(CheckOperator.NO_SPAN_ERROR, "no_span_error", None)
    anchor = _evidence("episode_span_count", 3)

    assert execute_verifier(spec, (anchor,)).score == 1.0
    assert execute_verifier(spec, (anchor, _evidence("span_error", {"name": "x"}))).score == 0.0


def test_no_span_error_is_unknown_when_no_evidence_exists_for_the_episode() -> None:
    """Absence of a recorded error is only meaningful if we hold the episode at all."""
    spec = _spec(CheckOperator.NO_SPAN_ERROR, "no_span_error", None)

    assert execute_verifier(spec, ()).score is None
