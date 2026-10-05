"""Hand-calculated rollouts for the step-reward shaping rules."""

from __future__ import annotations

import pytest

from bandits_jev.step_shaping import StepEvent, shape_rollout

# idx:   0 1 2 | 3 4 5 6 | 7 8 | 9 10 11 | 12 13 14 15
# bit:   1 1 1 | 0 0 0 0 | 1 1 | 0  0  0  | 1  1  1  1
# part:  action 1 | reply | action 2 | reply | final answer
MASK = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
FIRST, SECOND, FINAL = 2, 8, 15


def event(position, action="q one", score=0.8, observation="a real reply", tool="browser.search"):
    return StepEvent(position, tool, action, observation, score)


def test_action_reward_lands_on_the_action_token_and_outcome_on_the_last_token():
    shaped = shape_rollout(MASK, [event(FIRST), event(SECOND, "q two", 0.5)], 1.0, step_weight=0.3)
    expected = [0.0] * 16
    expected[FIRST] = pytest.approx(0.24)
    expected[SECOND] = pytest.approx(0.15)
    expected[FINAL] = 1.0
    assert shaped.rewards == pytest.approx(expected)
    assert shaped.total == pytest.approx(1.39)
    assert shaped.step_sum_capped == pytest.approx(0.39)


def test_tool_reply_tokens_never_get_reward():
    shaped = shape_rollout(MASK, [event(FIRST), event(SECOND, "q two", -1.0)], 1.0, step_weight=0.3)
    assert all(r == 0.0 for r, bit in zip(shaped.rewards, MASK, strict=True) if not bit)


def test_outcome_is_separate_from_the_step_reward():
    outcome_only = shape_rollout(MASK, [], 1.0, step_weight=0.3)
    with_steps = shape_rollout(MASK, [event(FIRST)], 1.0, step_weight=0.3)
    assert outcome_only.rewards[FINAL] == with_steps.rewards[FINAL] == 1.0
    assert outcome_only.total == 1.0
    assert with_steps.total - 1.0 == pytest.approx(with_steps.step_sum_capped)


def test_a_repeated_action_is_never_rewarded_even_if_the_judge_likes_it():
    shaped = shape_rollout(MASK, [event(FIRST), event(SECOND, " Q  ONE ", 0.6)], 0.0, step_weight=0.3)
    assert shaped.rewards[SECOND] == 0.0
    assert shaped.rejected["repeat"] == 1
    assert shaped.total == pytest.approx(0.24)


def test_a_repeated_action_can_still_be_penalised():
    shaped = shape_rollout(MASK, [event(FIRST), event(SECOND, "q one", -0.5)], 0.0, step_weight=0.3)
    assert shaped.rewards[SECOND] == pytest.approx(-0.15)


def test_unobserved_and_unscored_steps_earn_nothing():
    steps = [event(FIRST, observation=""), event(SECOND, "q two", score=None)]
    shaped = shape_rollout(MASK, steps, 1.0, step_weight=0.3)
    assert shaped.rewards[FIRST] == shaped.rewards[SECOND] == 0.0
    assert shaped.rejected == {"no_provenance": 1, "unscored": 1}
    assert shaped.total == 1.0


def test_only_the_first_call_in_a_turn_counts():
    shaped = shape_rollout(MASK, [event(FIRST), event(FIRST, "q two", 0.9)], 0.0, step_weight=0.3)
    assert shaped.rejected["extra_call_in_turn"] == 1
    assert shaped.total == pytest.approx(0.24)


def test_steps_after_truncation_are_dropped_and_outcome_moves_to_the_new_last_token():
    mask = MASK[:8]  # policy tokens end at index 7
    shaped = shape_rollout(mask, [event(FIRST), event(SECOND, "q two", 0.9)], 1.0, step_weight=0.3)
    assert shaped.rejected["truncated"] == 1
    assert shaped.rewards[7] == 1.0
    assert shaped.rewards[FIRST] == pytest.approx(0.24)


def test_step_total_is_capped_and_the_uncapped_value_is_reported():
    mask = [1, 0] * 6 + [1]
    steps = [event(p, f"query {p}", 1.0) for p in (0, 2, 4, 6, 8)]
    shaped = shape_rollout(mask, steps, 1.0, step_weight=0.3, step_cap=1.0)
    assert shaped.step_sum_uncapped == pytest.approx(1.5)
    assert shaped.step_sum_capped == pytest.approx(1.0)
    assert shaped.total == pytest.approx(2.0)


def test_judge_scores_are_clipped_to_one():
    shaped = shape_rollout(MASK, [event(FIRST, score=1.7)], 0.0, step_weight=0.3)
    assert shaped.rewards[FIRST] == pytest.approx(0.3)


def test_an_event_on_a_tool_token_is_a_harness_bug():
    with pytest.raises(ValueError, match="not a policy token"):
        shape_rollout(MASK, [event(4)], 0.0, step_weight=0.3)
