from rlm_draft_verifiers import (
    ProposedCheck,
    ProposedVerifier,
    build_spec,
    field_catalog,
    ground_check,
    rank_key,
)

from bandits.analyze.models import Evidence, EvidenceKind, Visibility
from bandits.labels import Verdict
from bandits.verify.models import CheckOperator


def _ev(trace_id: str, claim: str, field: str, value, index: int) -> Evidence:
    return Evidence(
        evidence_id=f"ev-{trace_id}-{index}",
        claim=claim,
        value={
            "key": field.split(".")[-1],
            "field": field,
            "value": value,
            "tool": field.split(".")[0],
        },
        visibility=Visibility.TERMINAL if claim == "final_state_field" else Visibility.DURING,
        provenance="observed",
        strength="strong",
        kind=EvidenceKind.STRUCTURED_EXTERNAL_RESULT,
        trace_id=trace_id,
    )


def _evidence() -> dict[str, list[Evidence]]:
    return {
        "t1": [
            _ev("t1", "initial_state_field", "get_reservation_details.insurance", "yes", 0),
            _ev("t1", "final_state_field", "cancel_reservation.status", "cancelled", 1),
        ],
        "t2": [
            _ev("t2", "initial_state_field", "get_reservation_details.insurance", "no", 0),
            _ev("t2", "final_state_field", "cancel_reservation.status", "cancelled", 1),
        ],
    }


def test_equals_is_grounded_in_observed_fields_and_values() -> None:
    evidence = _evidence()
    ok = ground_check(
        ProposedCheck(
            claim="final_state_field:cancel_reservation.status",
            operator="equals",
            expected="cancelled",
        ),
        evidence,
    )
    assert ok.operator is CheckOperator.EQUALS
    assert set(ok.supporting_evidence_ids) == {"ev-t1-1", "ev-t2-1"}

    unseen_value = ground_check(
        ProposedCheck(
            claim="final_state_field:cancel_reservation.status",
            operator="equals",
            expected="refunded",
        ),
        evidence,
    )
    assert isinstance(unseen_value, str) and "never observed" in unseen_value

    unseen_field = ground_check(
        ProposedCheck(
            claim="final_state_field:cancel_reservation.refund", operator="equals", expected=1
        ),
        evidence,
    )
    assert isinstance(unseen_field, str) and "no fit trace recorded" in unseen_field


def test_invariant_needs_both_sides_and_drops_expected() -> None:
    evidence = _evidence()
    ok = ground_check(
        ProposedCheck(
            claim="invariant:cancel_reservation.status==get_reservation_details.insurance",
            operator="state_invariant",
            expected="ignored",
        ),
        evidence,
    )
    assert ok.operator is CheckOperator.STATE_INVARIANT and ok.expected is None

    bad = ground_check(
        ProposedCheck(
            claim="invariant:cancel_reservation.status==nope.field", operator="state_invariant"
        ),
        evidence,
    )
    assert isinstance(bad, str) and "initial_state_field" in bad


def test_unknown_operator_and_empty_verifier_are_rejected() -> None:
    evidence = _evidence()
    assert isinstance(ground_check(ProposedCheck(claim="x", operator="regex"), evidence), str)
    spec, rejections = build_spec(
        ProposedVerifier(name="v", checks=[ProposedCheck(claim="x", operator="regex")]),
        evidence,
        "family-a",
        "taskset-a",
    )
    assert spec is None
    assert any("no executable check survived" in item for item in rejections)


def test_build_spec_is_model_provenance_and_stable() -> None:
    evidence = _evidence()
    proposal = ProposedVerifier(
        name="cancel-status",
        checks=[
            ProposedCheck(
                claim="final_state_field:cancel_reservation.status",
                operator="equals",
                expected="cancelled",
            ),
            ProposedCheck(
                claim="final_state_field:cancel_reservation.status",
                operator="equals",
                expected="cancelled",
            ),
        ],
        blind_spots=["cannot see refund"],
    )
    first, _ = build_spec(proposal, evidence, "family-a", "taskset-a")
    second, _ = build_spec(proposal, evidence, "family-a", "taskset-a")
    assert first.verifier_id == second.verifier_id
    assert first.provenance == "model" and len(first.checks) == 1
    assert first.blind_spots == ("cannot see refund",)


def test_catalog_splits_values_by_label() -> None:
    catalog = field_catalog(_evidence(), {"t1": Verdict.SUCCESS, "t2": Verdict.FAILURE})
    entry = catalog["initial_state_field:get_reservation_details.insurance"]
    assert entry["traces_recording"] == 2
    assert entry["values"]['"yes"'] == {"success": 1}
    assert entry["values"]['"no"'] == {"failure": 1}


def test_rank_prefers_zero_false_positives_then_catch_rate_then_coverage() -> None:
    clean_narrow = {
        "false_positives": 0,
        "failure_catch_rate": 1.0,
        "coverage": 0.2,
        "agreement": 1.0,
    }
    clean_wide = {
        "false_positives": 0,
        "failure_catch_rate": 1.0,
        "coverage": 0.9,
        "agreement": 1.0,
    }
    leaky_wide = {
        "false_positives": 2,
        "failure_catch_rate": 0.5,
        "coverage": 1.0,
        "agreement": 0.8,
    }
    assert sorted([leaky_wide, clean_narrow, clean_wide], key=rank_key) == [
        clean_wide,
        clean_narrow,
        leaky_wide,
    ]
