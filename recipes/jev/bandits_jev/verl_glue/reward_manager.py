"""Outcome reward from veRL's own manager, plus capped judge rewards for observed steps.

The outcome comes from ``NaiveRewardManager.run_single``, the code the
outcome-only arm uses, so the two arms differ only in the step term. Stock
GRPO sums token rewards, so the returned scalar is what trains. Steps are
scored in one call to the frozen judge; a failed judge call costs that
rollout its step term (flagged in ``judge_failed``), not the run.
"""

from __future__ import annotations

import json
import logging

from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

from bandits_jev.step_shaping import StepEvent, shape_rollout
from bandits_jev.verl_glue.judge_client import JudgeClient, ModalJudgeClient
from bandits_jev.verl_glue.tools import EVENTS_KEY

logger = logging.getLogger(__name__)


def stand_in_mask(response_length: int, positions: list[int]) -> list[int]:
    """The reward loop does not receive ``response_mask``, only the length.

    Marks the step positions and the last token as policy tokens. That is exact
    for the total reward, which is all stock GRPO uses; the token vector is
    only an audit aid here.
    """
    mask = [0] * response_length
    for position in positions:
        if 0 <= position < response_length:
            mask[position] = 1
    if response_length:
        mask[-1] = 1
    return mask


@register("jev_step")
class JevStepRewardManager(NaiveRewardManager):
    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
        judge_client: JudgeClient | None = None,
    ):
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)
        options = config.reward.get("reward_kwargs") or {}
        self.step_weight = float(options.get("step_weight", 0.1))
        self.step_cap = float(options.get("step_cap", 1.0))
        self.judge = judge_client or ModalJudgeClient()

    async def run_single(self, data) -> dict:
        base = await super().run_single(data)
        item = data[-1:][0]
        outcome = float(base["reward_score"])
        raw_events = sorted(
            (item.non_tensor_batch.get("tool_extra_fields") or {}).get(EVENTS_KEY, []),
            key=lambda event: event["position"],
        )
        question = (item.non_tensor_batch.get("extra_info") or {}).get("question", "")

        requests, request_positions, history = [], [], []
        for event in raw_events:
            observed = bool(event["observation"] and event["observation"].strip())
            if observed and event["position"] not in request_positions:
                requests.append(
                    {
                        "question": question,
                        "previous": [list(step) for step in history[-2:]],
                        "tool": event["tool"],
                        "action": event["action"],
                        "observation": event["observation"],
                    }
                )
                request_positions.append(event["position"])
            history.append((event["tool"], event["action"], event["observation"]))

        scores: dict[int, float | None] = {}
        judge_failed, digest = 0, ""
        if requests:
            try:
                reply = await self.judge.score(requests)
                digest = reply["judge"]["adapter_digest"]
                scores = {p: r["score"] for p, r in zip(request_positions, reply["results"], strict=True)}
            except Exception:
                logger.exception("judge call failed; this rollout gets no step reward")
                judge_failed = 1

        steps = [
            StepEvent(e["position"], e["tool"], e["action"], e["observation"], scores.get(e["position"]))
            for e in raw_events
        ]
        response_length = int(item.non_tensor_batch["response_len"])
        mask = stand_in_mask(response_length, [e["position"] for e in raw_events])
        shaped = shape_rollout(mask, steps, outcome, step_weight=self.step_weight, step_cap=self.step_cap)

        info = dict(base["reward_extra_info"])
        info.update(
            outcome=outcome,
            step_sum_capped=shaped.step_sum_capped,
            step_sum_uncapped=shaped.step_sum_uncapped,
            n_steps=len(raw_events),
            n_rejected=sum(shaped.rejected.values()),
            rejected=json.dumps(dict(shaped.rejected), sort_keys=True),
            judge_failed=judge_failed,
            judge_adapter_digest=digest,
        )
        return {"reward_score": shaped.total, "reward_extra_info": info}
