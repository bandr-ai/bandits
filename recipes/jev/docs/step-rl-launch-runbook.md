# Step reward RL launch runbook

Status: judge pilot passed a limited offline transfer gate on 2026-10-05. No
policy RL result exists yet. The experiment log is `step-reward-rl-pilot.md`.

## Fixed question

Does adding reward from our search-step judge improve a Qwen3-4B search
agent over outcome-only GRPO at matched policy updates, rollout budget,
retriever, prompt, and final-answer grader?

The judge is a separate pinned Qwen3.5-4B-Base LoRA model. It sees the task,
two preceding observed tool interactions, the current browser action, and
the *actual tool response*. It never sees a gold final answer. The policy
checkpoint and the judge checkpoint are different models.

## Data boundary

1. Reserve all 830 BrowseComp-Plus questions and answers for evaluation. Keep
   its fixed corpus/index and official semantic answer grader. Generate policy
   training questions from other documents or a separate public search corpus;
   deduplicate by question and answer against BrowseComp-Plus before training.
   Do not train on the 830 benchmark questions and then report all 830 as
   held-out evaluation.
2. Freeze the judge adapter and rubric before any live policy rollout. The
   current frozen judge is recorded in the pilot report on the `jev-runs`
   Modal volume. A further judge revision requires a new held-out audit.
3. Save task IDs, corpus/index hashes, retrieval top-k, policy and judge model
   revisions, prompts, tool schema, seeds, and every rollout. Never put
   BrowseComp-Plus gold answers in a judge prompt or training example.

## Harness contract

Use one multi-turn rollout implementation for both RL arms and for the base
evaluation. Search, open, and find return genuine results from the same fixed
retriever. The tool layer records the action and result with provenance before
requesting any judge score. A missing, malformed, or unobserved result cannot
receive positive step reward. Final-answer-only turns receive terminal reward
only. Detect identical repeats from the trajectory and never give them a
positive reward, even if the model predicts one. Bound tool turns and tokens.

The outcome-only arm receives the task's verified final-answer reward at the
terminal policy token. The step arm receives the *same* outcome reward plus a
step signal attached to the policy token that finished each browser action:

`R_t = lambda * clip(P_judge(positive) - P_judge(negative), -1, 1)`.

Start with `lambda = 0.1` and `0.3` on a separate pilot task set. Cap positive
and negative step sums at 1.0 in magnitude per trajectory so extra low-value
tool calls cannot dominate the final-answer reward; log uncapped and capped
values. The pure reward mapping is implemented in
`bandits_jev/step_reward.py` and has token-position tests; veRL integration
is still needed. Inspect tool-count
and duplicate-query drift. If the judge learns to reward verbosity or repeated
searches, stop and try progress-difference scoring or refresh the judge on
fresh policy rollouts. Use the same RL framework, GRPO settings, policy
compute, and update count for each paired arm. The framework's tool reward
mapping must be checked against actual assistant token positions on saved
rollouts before training.

The inspected veRL checkout is commit
`8718ca30a3f002f93b7c4fd99b9b2506718681bc`. Its multi-turn tool loop
records tool return rewards in `tool_rewards`, but its default naive reward
manager assembles a **terminal-only** reward tensor. Therefore simply
returning a judge score from a tool would not implement step-wise RL. Build a
custom reward manager that maps each observed tool action to the exact
assistant-token boundary, adds the terminal answer reward, and verifies the
result against a hand-calculated rollout. The older Search-R1 checkout is a
retrieval/RL reference and likewise has a terminal-only reward manager.

## Training hygiene (from Lu, "Training Search Agents with GRPO", 2026-09-17)

That post trains with ground-truth-derived reward only (no learned judge), so
it is a reference for RL settings, not for the step-reward idea. Apply to both
arms identically:

- **Algorithm and batch:** Dr. GRPO (no std normalisation), no KL penalty,
  one update per batch. Start from 64 groups x 8 rollouts per step. The post's
  8 groups/step runs hit high train reward by degenerate behaviour (curating
  everything, entropy 1.1 -> 0.34) and had the worst eval.
- **Learning rate:** run a short sweep on the training-domain validation tasks
  before the pilots. In the post, LoRA at 1e-4 was best and 5e-4 collapsed the
  policy. Our 4B policy under veRL needs its own sweep; do not copy the number.
- **Always log:** train reward, entropy, share of no-signal (all-equal reward)
  groups, valid-rollout share, tool calls per episode, duplicate calls, and the
  off-form tool-call rate. Stop a run on entropy collapse.
- **Strict tool-call parsing** in the shared harness. The post's lenient parser
  let malformed calls get rewarded; 99% of episodes ended up with one. Count
  malformed calls in every arm and apply the same small penalty to both arms.
- **Floor reward** for episodes that never produce an answer, so degenerate
  non-answers are separable from wrong answers. Same value in both arms.
- **Do not read a longer trajectory as hacking by itself.** The post's best run
  went from 18 to 30 turns with better recall. Judge it against ground-truth
  success and the duplicate-call rate.
- **Headroom check first.** The base 8-rollout run doubles as the best-of-N
  check. Groups that are all zero give no outcome-only gradient; that is where
  step reward should help, so report the share of such groups per arm.

Scope note: the required comparison is outcome-only vs outcome + Jev step
reward. A judge-free shaped arm (for example a trajectory-evidence bonus) is
optional follow-up only. It needs gold evidence per training question, which a
Jev user trained from their own traces would not have. Without it, a win does
not show Jev beats other shaping, only that Jev step reward beats outcome-only.

## Testing sequence

1. **Integration smoke:** 10 training-only tasks, one rollout each. Verify
   tool provenance, no gold leakage, answer grader, judge prompts, exact
   token reward placement, length caps, and reward totals by hand.
2. **Base bucket:** run 8 independent base-policy rollouts per evaluation
   question using the frozen harness. Save the `0/8` bucket *before* viewing
   any RL outcome. Keep final-answer grading separate from judge scoring.
3. **Pilot:** same short training budget for `lambda=0.1` and `0.3`. Select
   based on separate training-domain validation tasks and reward integrity,
   not the BrowseComp-Plus test score.
4. **Paired runs:** two seeds for outcome-only and two matched seeds for the
   selected step arm. Same prompts, tool limits, policy steps, rollout count,
   hardware budget, and final grader. Track ground-truth success, judge
   reward, tool calls, duplicates, invalid calls, and judge latency/cost.
5. **Frontier cost slice:** only after the paired runs are informative, score
   the same saved early rollout steps using the frontier judge. Label any
   full-run cost as an extrapolation from measured calls, tokens, and prices.
6. **Launch decision:** publish only if the step arm beats outcome-only in
   both paired seeds on overall BrowseComp-Plus success and the frozen `0/8`
   bucket, with no reward/success divergence. Report task counts and paired
   uncertainty intervals. TRACE 35.6 is a differently configured published
   reference line, not a head-to-head result.

Judge-label agreement (78/100 on the fresh audit) is a prerequisite, not a
proof of RL gains. Until the paired runs finish, there is no launch claim,
hero chart, or cost superiority measurement.
