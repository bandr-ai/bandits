"""Dr. GRPO for the scoped step-reward experiment: rewards, advantages, token segments, loss.

Both arms share everything here except the reward: ``outcome`` uses the answer
alone; ``step`` adds the frozen judge's per-step scores, centred on a baseline
fixed from training-pool rollouts and shaped by ``shape_rollout`` (cap,
repeats never rewarded, unobserved steps dropped). Training uses the exact
token ids vLLM sampled, never a re-rendered transcript.
"""

from __future__ import annotations

from collections.abc import Sequence

from bandits_jev.step_shaping import StepEvent, action_key, shape_rollout


def rollout_reward(result: dict, arm: str, *, step_weight: float, step_cap: float, baseline: float) -> dict:
    """Reward for one finished, judged rollout, with the parts kept for logging."""
    outcome = 1.0 if result["correct"] else 0.0
    if arm == "outcome":
        return {"reward": outcome, "outcome": outcome, "step_sum": 0.0}
    if arm != "step":
        raise ValueError(f"unknown arm {arm!r}")
    events = []
    for position, event in enumerate(result["events"]):
        judged = event.get("judge") or {}
        score = judged.get("score")
        events.append(
            StepEvent(
                position, event["tool"], event["action"], event["observation"],
                None if score is None else score - baseline,
            )
        )
    shaped = shape_rollout([1] * (len(events) + 1), events, outcome, step_weight=step_weight, step_cap=step_cap)
    return {"reward": shaped.total, "outcome": outcome, "step_sum": shaped.step_sum_capped}


def judged_step_quality(result: dict, *, baseline: float, repeat_penalty: float) -> float:
    """Mean of (judge score - baseline) over a rollout's observed steps, in [-1, 1].

    A repeated action scores -repeat_penalty whatever the judge says. A mean, not
    a sum, so extra searches earn nothing by themselves. A rollout that never
    searched scores -1: in a group where nobody answered correctly, not looking is
    the worst thing it could have done.
    """
    seen: set[tuple[str, str]] = set()
    values: list[float] = []
    for event in result["events"]:
        key = action_key(event["tool"], event["action"])
        score = (event.get("judge") or {}).get("score")
        if key in seen:
            values.append(-repeat_penalty)
        elif score is not None and event["observation"].strip():
            values.append(score - baseline)
        seen.add(key)
    if not values:
        return -1.0
    return max(-1.0, min(1.0, sum(values) / len(values)))


def group_rewards(group: Sequence[dict], arm: str, *, step_weight: float, step_cap: float, baseline: float,
                  repeat_penalty: float = 0.5) -> list[dict]:
    """Rewards for one question's rollouts.

    ``step2`` uses the judge only where outcome reward is blind: in a group with
    any correct answer every rollout gets its outcome alone (exactly the
    outcome-only reward); in a group where all are wrong, rollouts are ranked by
    ``judged_step_quality``. ``outcome`` and ``step`` score each rollout alone.
    """
    if arm in ("outcome", "step"):
        return [
            rollout_reward(r, arm, step_weight=step_weight, step_cap=step_cap, baseline=baseline) for r in group
        ]
    if arm != "step2":
        raise ValueError(f"unknown arm {arm!r}")
    outcomes = [1.0 if r["correct"] else 0.0 for r in group]
    if any(outcomes):
        return [{"reward": o, "outcome": o, "step_sum": 0.0} for o in outcomes]
    return [
        {
            "reward": step_weight * judged_step_quality(r, baseline=baseline, repeat_penalty=repeat_penalty),
            "outcome": 0.0,
            "step_sum": step_weight * judged_step_quality(r, baseline=baseline, repeat_penalty=repeat_penalty),
        }
        for r in group
    ]


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Dr. GRPO: reward minus the group mean, with no division by the group's spread."""
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def training_segments(turns: Sequence[dict]) -> list[tuple[list[int], list[int]]]:
    """Token sequences with a mask of 1 on tokens the policy sampled.

    Each turn is the prompt vLLM actually saw and the ids it sampled. A turn is
    appended to the current sequence when its prompt starts with that sequence
    (the usual case); otherwise it starts a new one, so a template that
    re-renders earlier turns never puts a re-rendered token under the loss.
    """
    segments: list[tuple[list[int], list[int]]] = []
    ids: list[int] = []
    mask: list[int] = []
    for turn in turns:
        prompt, sampled = list(turn["prompt_token_ids"]), list(turn["token_ids"])
        if ids and prompt[: len(ids)] == ids:
            extra = prompt[len(ids) :]
            ids += extra
            mask += [0] * len(extra)
        else:
            if ids:
                segments.append((ids, mask))
            ids, mask = prompt, [0] * len(prompt)
        ids += sampled
        mask += [1] * len(sampled)
    if ids:
        segments.append((ids, mask))
    return segments


def policy_gradient_step(model, items: Sequence[tuple[list[int], list[int], float]], normalizer: float, device: str) -> dict:
    """Accumulate -advantage * log p(sampled token) / normalizer over all items.

    One sequence at a time; logits are computed only where a sampled token is
    predicted, so memory does not scale with vocabulary times length. The
    caller runs the optimizer. Returns the mean log-prob of sampled tokens,
    used to check that vLLM and the trained model agree.
    """
    import torch

    total_logprob, total_tokens = 0.0, 0
    for ids, mask, advantage in items:
        targets = [i for i in range(1, len(ids)) if mask[i]]
        if not targets or advantage == 0:
            continue
        input_ids = torch.tensor([ids], device=device)
        keep = torch.tensor([t - 1 for t in targets], device=device)
        logits = model(input_ids=input_ids, logits_to_keep=keep).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1).gather(-1, input_ids[0, keep + 1].unsqueeze(-1)).squeeze(-1)
        loss = -(advantage * logprobs).sum() / normalizer
        loss.backward()
        total_logprob += float(logprobs.detach().sum())
        total_tokens += len(targets)
    return {"trained_tokens": total_tokens, "mean_logprob": total_logprob / total_tokens if total_tokens else None}
