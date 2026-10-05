# Step reward in veRL: what the code does and how we hook it

Status: design note from reading veRL at commit
`8718ca30a3f002f93b7c4fd99b9b2506718681bc` (ignored checkout
`work/verl-upstream/`). Nothing here is implemented or run yet. Paths are
relative to `verl/` in that checkout.

## What veRL does today

1. **Mask and token positions.** In `experimental/agent_loop/tool_agent_loop.py`
   the assistant turn is merged into the token stream at line 274
   (`ct_merge_assistant_token`, mask 1) and the tool response at line 386
   (`ct_merge_context_msg`, mask 0). Right after line 282 and before tools run
   (line 310), `len(agent_data.response_mask) - 1` is the last policy token of
   the action. This is the position the runbook needs, and it is available
   before the tool response is appended.
2. **Parallel calls share one position.** One assistant turn can hold several
   tool calls (`max_parallel_calls`, line 317), so several actions would map to
   one token. `step_reward.reward_vector` rejects duplicate positions.
3. **`tool_rewards` is not used for training.** It is a list of scalars with no
   positions (lines 375-376), copied into `extra_fields` (line 212) and into
   the non-tensor batch (`agent_loop.py` 1138-1152). The trainer never reads it.
4. **The reward manager returns one scalar.** `reward_loop/reward_manager/naive.py`
   returns `reward_score`; `agent_loop.py` 1110-1116 writes it at the last valid
   response token of `rm_scores`. Only a terminal reward exists today.
5. **Stock GRPO sums token rewards.** `trainer/ppo/core_algos.py:304` does
   `scores = token_level_rewards.sum(dim=-1)`, normalises within the group,
   and line 329 broadcasts that one advantage to every response token.
   `norm_adv_by_std_in_grpo=False` gives Dr. GRPO (lines 289-296).

## What that means for the experiment

Under stock GRPO, **where** a step reward lands in the token vector does not
matter; only the trajectory total does. Adding the step rewards means
"outcome + capped sum of judge steps" shifts the whole trajectory's advantage.
It is step-reward *shaping*, not per-step credit assignment. This is the main
correction to the runbook, which implies each reward credits its own action.

It still helps the case we care about: a group whose rollouts all fail the
outcome has zero advantage, but with a step term the rollouts differ and the
group gives gradient.

## Options

- **A. Trajectory-sum shaping (recommended first).** A custom agent loop
  records `(token position, action, observed result, provenance)` per browser
  action in `extra_fields` during the rollout. A custom reward manager (the
  async reward loop already passes `tool_extra_fields` to it) scores those
  steps with the frozen judge, applies the lambda, the +/-1 caps, duplicate
  rejection and provenance checks, adds the outcome, and returns one scalar.
  No change to veRL core, stock GRPO, and the smallest diff from the
  outcome-only arm. `reward_vector(...).sum()` must equal that scalar, which
  gives a test.
- **B. Per-step credit.** Register a custom advantage estimator
  (`register_adv_est`, `core_algos.py:116`) that gives each action's tokens an
  advantage from the step rewards at and after that step, normalised across the
  group. Needs rewards as a per-token tensor, so a patch where `rm_scores` is
  built, plus new advantage code. Tests the claim as originally worded but
  adds risk and is no longer a matched GRPO comparison.

## Decisions needed before Phase 2

1. Run A first and keep per-token vectors saved so B stays possible, and word
   the launch claim as "step-reward shaping" unless B is run.
2. Enforce one browser action per assistant turn in the search tool env
   (`max_parallel_calls=1`), counting any extra calls as invalid and logging
   them. This matches the judge input (a single action).
3. Where the judge runs during training (a served endpoint beside the trainer
   versus after rollouts). To settle in Phase 2 with a latency and cost number.

## Edge cases to test

- Response truncation at `response_length` (lines 195-196): drop events past
  the truncated mask.
- Tool response that overflows the length cap (line 394): the loop terminates
  without appending it.
- A terminal reward index (`agent_loop.py:1113`) uses the last attention-mask
  token, which is only a policy token if the final turn is an assistant turn.
  Irrelevant under A, but matters for B.
