"""Rewards, advantages, segments and the loss sign for the scoped GRPO trainer."""

from __future__ import annotations

import pytest

from bandits_jev.grpo import (
    group_advantages,
    group_rewards,
    judged_step_quality,
    rollout_reward,
    training_segments,
)


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


def rollout(correct, *steps):
    return {"correct": correct, "events": [judged(score, action) for action, score in steps]}


V2 = {"step_weight": 0.3, "step_cap": 1.0, "baseline": 0.2, "repeat_penalty": 0.5}


def test_step2_uses_outcome_alone_when_any_rollout_is_correct():
    group = [rollout(True, ("a", -0.9)), rollout(False, ("b", 0.9)), rollout(False)]
    assert [r["reward"] for r in group_rewards(group, "step2", **V2)] == [1.0, 0.0, 0.0]


def test_step2_ranks_an_all_wrong_group_by_mean_step_quality():
    good, poor = rollout(False, ("a", 0.7)), rollout(False, ("b", 0.0))
    rewards = [r["reward"] for r in group_rewards([good, poor], "step2", **V2)]
    assert rewards == pytest.approx([0.3 * 0.5, 0.3 * -0.2])
    assert group_advantages(rewards)[0] > 0


def test_padding_with_baseline_steps_earns_nothing_and_repeats_cost():
    one = judged_step_quality(rollout(False, ("a", 0.7)), baseline=0.2, repeat_penalty=0.5)
    padded = judged_step_quality(rollout(False, ("a", 0.7), ("b", 0.2), ("c", 0.2)), baseline=0.2, repeat_penalty=0.5)
    repeated = judged_step_quality(rollout(False, ("a", 0.7), ("A ", 0.9)), baseline=0.2, repeat_penalty=0.5)
    assert padded < one
    assert repeated == pytest.approx((0.5 - 0.5) / 2)


def test_not_searching_is_the_worst_in_an_all_wrong_group():
    assert judged_step_quality(rollout(False), baseline=0.2, repeat_penalty=0.5) == -1.0
    rewards = [r["reward"] for r in group_rewards([rollout(False), rollout(False, ("a", 0.1))], "step2", **V2)]
    assert rewards[0] < rewards[1]


def test_outcome_and_step_arms_are_unchanged_by_group_rewards():
    group = [rollout(True, ("a", 0.9)), rollout(False, ("b", 0.4))]
    for arm in ("outcome", "step"):
        expected = [rollout_reward(r, arm, step_weight=0.3, step_cap=1.0, baseline=0.2) for r in group]
        assert group_rewards(group, arm, **V2) == expected
    with pytest.raises(ValueError):
        group_rewards(group, "both", **V2)


def test_oracle_adds_weighted_gold_evidence_recall_to_the_outcome():
    group = [{"correct": True, "evidence_seen_recall": 0.5, "events": []},
             {"correct": False, "evidence_seen_recall": 1.0, "events": [judged(0.9)] * 5}]
    rewards = [r["reward"] for r in group_rewards(group, "oracle", **V2)]
    assert rewards == pytest.approx([1.0 + 0.3 * 0.5, 0.3 * 1.0])
