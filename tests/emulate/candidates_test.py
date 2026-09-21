"""Candidate adapters: the view boundary, parsing, and the controls."""

from __future__ import annotations

from bandits.emulate.candidates import (
    CandidateSpec,
    giving_up_candidate,
    looping_candidate,
    offered_tool_names,
    parse_action,
    render_candidate_prompt,
    render_history,
    render_tools,
    scripted_candidate,
    unoffered_calls,
)
from bandits.emulate.models import (
    ActionCall,
    CandidateView,
    ExpectedEffect,
    Partition,
    Scenario,
    ScenarioKind,
    SealedSuccessContract,
    SuccessShape,
)
from bandits.emulate.rollout import CandidateAction


def _view(**updates) -> CandidateView:
    base = dict(
        scenario_id="scenario-1",
        task="cancel reservation ABC",
        system_policy="You must obtain explicit confirmation before any change.",
        offered_tools=(
            {
                "name": "cancel_reservation",
                "parameters": {
                    "type": "object",
                    "properties": {"reservation_id": {"type": "string"}},
                },
            },
        ),
        prefix=(),
    )
    return CandidateView(**{**base, **updates})


# --- the view boundary --------------------------------------------------


def test_the_prompt_is_built_only_from_the_candidate_view() -> None:
    """A private field cannot reach the candidate by being added to Scenario."""
    scenario = Scenario(
        scenario_id="scenario-1",
        kind=ScenarioKind.TASK_START,
        task="cancel reservation ABC",
        success_contract=SealedSuccessContract(
            contract_id="contract-secret",
            source_task_id="7",
            shape=SuccessShape.MUTATION,
            required_effects=(ExpectedEffect(effect_id="e1", tool="cancel_reservation"),),
        ),
        source_trace_id="airline-10",
        source_task_id="7",
        family_id="f",
        partition=Partition.FIT,
        retrieval_excluded_trace_ids=("airline-10",),
    )
    prompt = render_candidate_prompt(scenario.candidate_view(), [])
    assert "contract-secret" not in prompt
    assert "cancel reservation ABC" in prompt


def test_the_toolset_is_rendered_with_its_schemas() -> None:
    """Which tool to reach for, out of what was offered, is the decision."""
    rendered = render_tools(_view())
    assert "cancel_reservation" in rendered
    assert "reservation_id" in rendered


def test_an_empty_toolset_is_stated_rather_than_omitted() -> None:
    assert "no tools are available" in render_tools(_view(offered_tools=()))


def test_history_is_clipped_from_the_front() -> None:
    """The most recent observation is what the next action answers."""
    history = [{"role": "tool", "content": "x" * 500} for _ in range(40)]
    history.append({"role": "tool", "content": "THE-LATEST-RESULT"})
    rendered = render_history(history, limit=1000)
    assert "THE-LATEST-RESULT" in rendered
    assert "earlier turns omitted" in rendered


def test_an_empty_conversation_says_so() -> None:
    assert "has not started" in render_history([])


# --- parsing ------------------------------------------------------------


def test_a_wire_shaped_tool_call_is_parsed() -> None:
    action = parse_action(
        {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": '{"reservation_id": "ABC"}',
                    },
                }
            ]
        }
    )
    assert action.calls[0].tool == "cancel_reservation"
    assert action.calls[0].arguments == {"reservation_id": "ABC"}
    assert action.calls[0].call_id == "c1"


def test_a_flat_call_shape_is_parsed_too() -> None:
    action = parse_action({"calls": [{"tool": "cancel_reservation", "arguments": {"id": "A"}}]})
    assert action.calls[0].tool == "cancel_reservation"


def test_several_calls_in_one_reply_stay_one_action() -> None:
    """54 recorded tau2 actions carry several calls; a candidate may too."""
    action = parse_action(
        {
            "tool_calls": [
                {"id": "c1", "function": {"name": "cancel_reservation", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "get_user_details", "arguments": "{}"}},
            ]
        }
    )
    assert len(action.calls) == 2
    assert [call.call_id for call in action.calls] == ["c1", "c2"]


def test_plain_text_is_a_message_not_a_failed_parse() -> None:
    action = parse_action("Could I have your reservation number?")
    assert action.calls == ()
    assert "reservation number" in str(action.content)


def test_unparseable_arguments_are_kept_rather_than_discarded() -> None:
    """A malformed argument is a capability finding; dropping it hides one."""
    action = parse_action(
        {"tool_calls": [{"function": {"name": "cancel_reservation", "arguments": "not json"}}]}
    )
    assert action.calls[0].arguments == {"_unparsed": "not json"}


def test_done_is_read_from_either_spelling() -> None:
    assert parse_action({"done": True}).done
    assert parse_action({"finished": True}).done


# --- a tool that was never offered --------------------------------------


def test_a_call_to_an_unoffered_tool_is_kept_as_a_real_action() -> None:
    """Dropping it would turn the candidate's mistake into silence."""
    action = parse_action(
        {"tool_calls": [{"function": {"name": "refund_everything", "arguments": "{}"}}]},
        offered=frozenset({"cancel_reservation"}),
    )
    assert action.calls[0].tool == "refund_everything"


def test_unoffered_calls_are_reported() -> None:
    action = CandidateAction(
        calls=(ActionCall(tool="refund_everything"), ActionCall(tool="cancel_reservation"))
    )
    assert unoffered_calls(action, frozenset({"cancel_reservation"})) == ("refund_everything",)


def test_every_call_is_unoffered_when_the_offered_set_is_empty() -> None:
    action = CandidateAction(calls=(ActionCall(tool="invented_tool"),))
    assert unoffered_calls(action, frozenset()) == ("invented_tool",)


def test_offered_names_are_read_from_either_schema_shape() -> None:
    view = _view(
        offered_tools=(
            {"name": "a"},
            {"type": "function", "function": {"name": "b"}},
        )
    )
    assert offered_tool_names(view) == frozenset({"a", "b"})


# --- controls -----------------------------------------------------------


def test_the_scripted_candidate_replays_then_finishes() -> None:
    planned = CandidateAction(calls=(ActionCall(tool="cancel_reservation"),))
    candidate = scripted_candidate([planned])
    assert candidate(view=_view(), history=[]) is planned
    assert candidate(view=_view(), history=[]).done


def test_the_giving_up_control_does_nothing_and_finishes() -> None:
    """It must score a clean failure on any mutation task."""
    action = giving_up_candidate()(view=_view(), history=[])
    assert action.done
    assert action.calls == ()


def test_the_looping_control_never_stops_on_its_own() -> None:
    candidate = looping_candidate(ActionCall(tool="cancel_reservation"))
    for _ in range(5):
        action = candidate(view=_view(), history=[])
        assert not action.done
        assert action.calls


# --- spec ---------------------------------------------------------------


def test_the_spec_digest_changes_with_decoding_settings() -> None:
    """Two runs differing in temperature are two measurements, not one."""
    cold = CandidateSpec(candidate_id="c", model="m", temperature=0.0)
    warm = CandidateSpec(candidate_id="c", model="m", temperature=0.7)
    assert cold.prompt_digest != warm.prompt_digest


def test_the_spec_digest_pins_endpoint_tokens_and_seed() -> None:
    base = CandidateSpec(candidate_id="c", model="m", endpoint="provider/m", seed=1)
    assert base.prompt_digest != base.replace(endpoint="other/m").prompt_digest
    assert base.prompt_digest != base.replace(max_tokens=999).prompt_digest
    assert base.prompt_digest != base.replace(seed=2).prompt_digest


def test_the_spec_digest_is_stable_for_one_configuration() -> None:
    first = CandidateSpec(candidate_id="c", model="m")
    second = CandidateSpec(candidate_id="c", model="m")
    assert first.prompt_digest == second.prompt_digest
