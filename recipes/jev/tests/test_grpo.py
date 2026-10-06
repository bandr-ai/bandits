"""Rewards, advantages, segments and the loss sign for the scoped GRPO trainer."""

from __future__ import annotations

import pytest

from bandits_jev.grpo import group_advantages, rollout_reward, training_segments


def judged(score, action="a", observation="reply"):
    return {"tool": "browser.search", "action": action, "observation": observation, "judge": {"score": score}}


def test_outcome_arm_ignores_the_judge():
    result = {"correct": True, "events": [judged(0.9)]}
    assert rollout_reward(result, "outcome", step_weight=0.3, step_cap=1.0, baseline=0.15)["reward"] == 1.0


def test_step_arm_centres_scores_on_the_baseline_before_weighting():
    result = {"correct": False, "events": [judged(0.65, "a"), judged(0.15, "b"), judged(-0.35, "c")]}
    out = rollout_reward(result, "step", step_weight=0.3, step_cap=1.0, baseline=0.15)
    # centred: +0.5, 0.0, -0.5 -> weighted +0.15, 0, -0.15
    assert out["reward"] == pytest.approx(0.0) and out["outcome"] == 0.0


def test_a_useless_step_at_the_baseline_earns_nothing_and_repeats_are_not_rewarded():
    one = rollout_reward({"correct": False, "events": [judged(0.6, "a")]}, "step", step_weight=0.3, step_cap=1.0, baseline=0.1)
    padded = rollout_reward(
        {"correct": False, "events": [judged(0.6, "a"), judged(0.1, "b"), judged(0.9, "a")]},
        "step", step_weight=0.3, step_cap=1.0, baseline=0.1,
    )
    assert padded["reward"] == pytest.approx(one["reward"])


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        rollout_reward({"correct": True, "events": []}, "both", step_weight=0.3, step_cap=1.0, baseline=0.0)


def test_dr_grpo_advantages_are_centred_but_not_scaled():
    assert group_advantages([1.0, 0.0, 0.0, 1.0]) == [0.5, -0.5, -0.5, 0.5]
    assert group_advantages([0.0, 0.0]) == [0.0, 0.0]
    assert group_advantages([2.0, 0.0]) == [1.0, -1.0]


def test_turns_that_extend_the_previous_prompt_merge_into_one_masked_sequence():
    turns = [
        {"prompt_token_ids": [1, 2, 3], "token_ids": [10, 11]},
        {"prompt_token_ids": [1, 2, 3, 10, 11, 20, 21], "token_ids": [12]},
    ]
    assert training_segments(turns) == [([1, 2, 3, 10, 11, 20, 21, 12], [0, 0, 0, 1, 1, 0, 0, 1])]


def test_a_re_rendered_prompt_starts_a_new_sequence_instead_of_training_on_it():
    turns = [
        {"prompt_token_ids": [1, 2], "token_ids": [10, 11]},
        {"prompt_token_ids": [1, 2, 99, 11, 20], "token_ids": [12]},  # template re-rendered token 10 as 99
    ]
    assert training_segments(turns) == [([1, 2, 10, 11], [0, 0, 1, 1]), ([1, 2, 99, 11, 20, 12], [0, 0, 0, 0, 0, 1])]
