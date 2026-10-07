# Scoped step-reward RL results (seed 1)

Status 2026-10-06. One seed per arm, 10 Dr. GRPO updates of 16 training-pool
questions x 8 rollouts, LoRA rank 32, lr 1e-4, no KL; Qwen3-4B-Instruct-2507
policy; BrowseComp-Plus Qwen3-Embedding-8B retriever. Both arms saw the same
questions in the same order. The step arm adds 0.3 x (Jev step score - 0.20
baseline fixed on training-pool rollouts), capped at +/-1 per rollout, repeats
never rewarded. Held-out evaluation: 60 never-trained questions x 8 rollouts.
Accuracy is a strict string-match proxy, not the official LLM grader.

| Metric | No RL | Outcome-only | Step-wise | Step - outcome (95% CI over questions) |
|---|---|---|---|---|
| Gold evidence seen (primary) | 0.218 | 0.200 | 0.224 | +0.024 (-0.016, +0.068) |
| Gold evidence opened | 0.058 | 0.100 | 0.136 | +0.036 (+0.009, +0.064) |
| Accuracy (proxy) | 15.6% | 22.9% | 20.6% | -2.3 pts (-8.5, +4.4) |
| Tool calls per rollout | 3.6 | 4.1 | 7.4 | +3.3 |
| Rollouts repeating an action | 4% | 8% | 26% | +18 pts (+10, +26) |

Outcome-only vs no RL: accuracy +7.3 pts (+1.7, +13.5).

- Training signal: the step arm had learning signal in 14-16 of 16 groups per
  update; outcome-only in 0-7.
- The step arm searched and opened documents much more (the base policy's main
  weakness was stopping early) at no accuracy cost, but gold evidence found and
  accuracy did not beat outcome-only within 10 updates.
- Side effect: repeats rose to 26%. Repeats are only denied reward, not
  penalised, so extra searching came with padding.
- Mid-training the step arm first searched less (3.9 -> 2.2 calls) while its
  judge reward rose, then swung to more (6.2 calls by update 10).

Verdict: directional, not a win. One seed; the next run penalises repeats.

Engineering failures on the way (fixed): catastrophic regexes in `find` froze
the trainer; a per-update HTTP client to vLLM left requests hanging until its
timeout; a laptop suspend ended a non-detached run.

## Step v2 (seed 1)

Same seed, questions and settings. Reward: groups with any correct answer get
outcome reward only; all-wrong groups are ranked by the mean of
baseline-centred judge scores (weight 1.0), repeats score -0.5, no search
scores -1.

| Metric | No RL | Outcome-only | Step v1 | Step v2 | v2 - outcome (95% CI) |
|---|---|---|---|---|---|
| Gold evidence seen | 0.218 | 0.200 | 0.224 | 0.209 | +0.009 (-0.040, +0.061) |
| Gold evidence opened | 0.058 | 0.100 | 0.136 | 0.075 | -0.025 (-0.045, -0.006) |
| Accuracy (proxy) | 15.6% | 22.9% | 20.6% | 15.8% | -7.1 pts (-14.2, -0.2) |
| Tool calls | 3.6 | 4.1 | 7.4 | 2.8 | -1.3 |
| Rollouts repeating an action | 4% | 8% | 26% | 0.2% | -7.7 pts |

Step v2 lost to outcome-only. Averaging step scores rewarded short rollouts
(the first search usually scores highest), so the policy searched less.

Across both step variants, the reward mostly changed how much the agent
searched (sum: more, with padding; mean: less), not how well: gold evidence
found never beat outcome-only. The judge's step signal (AUC 0.68-0.77 against
gold-evidence steps, generous to topical searches) is too weak to steer search
quality in 10 updates. Next: a stricter judge rubric (positive only when a step
surfaces or opens supporting evidence, with padding and repeats as negatives)
before more RL.

## Jev v2: retrained on gold-evidence step labels

Logged training-pool steps (2,986 rollouts, 9,654 steps) were labelled from
ground truth (positive: surfaced or opened a gold evidence document; negative:
repeat or tool error; else neutral) and a new judge was trained with the pilot
recipe (581 rows, 478 train; best dev accuracy 0.79 at the last checkpoint).
Both judges were scored on 1,172 steps from held-out questions:

| | v1 | v2 |
|---|---|---|
| AUC, evidence-finding steps vs rest | 0.781 | 0.786 |
| Accuracy | 70.6% | 70.6% |
| Non-evidence steps scored above 0.25 | 33% | 38% |

No meaningful gain. From a step's snippets alone, without the answer, a judge
cannot reliably tell the right evidence from relevant-looking results; better
labels do not supply that information. Next proposed test: an oracle arm whose
step signal comes from gold evidence directly, to learn whether step-wise
reward helps here at all.

## Oracle step reward (seed 1)

Same seed, questions and settings. Reward: outcome + 0.3 x the share of gold
evidence documents the rollout saw (ground truth, training questions only).
Training finished all ten updates; the evaluation was rerun from the saved
update-10 adapter on the deployed Evaluator after the launching laptop
suspended (no judge scoring; same rollout code).

| Metric | No RL | Outcome-only | Jev step v1 | Jev step v2 | Oracle step | Oracle - outcome (95% CI) |
|---|---|---|---|---|---|---|
| Gold evidence seen | 0.218 | 0.200 | 0.224 | 0.209 | 0.350 | +0.150 (+0.101, +0.201) |
| Gold evidence opened | 0.058 | 0.100 | 0.136 | 0.075 | 0.049 | -0.050 (-0.083, -0.018) |
| Accuracy (proxy) | 15.6% | 22.9% | 20.6% | 15.8% | 17.3% | -5.6 pts (-12.1, +1.3) |
| Tool calls | 3.6 | 4.1 | 7.4 | 2.8 | 7.3 | +3.2 |
| Rollouts repeating an action | 4% | 8% | 26% | 0.2% | 7% | -0.6 pts |

## Conclusions so far (one seed, ten updates)

- Step-wise reward reshapes search behaviour when its signal is right: a
  ground-truth step signal raised gold evidence found on held-out questions by
  about 75% over outcome-only.
- That did not become accuracy within ten updates; outcome-only RL still has
  the best accuracy (+7.3 pts over no RL). The agent finds evidence in results
  but opens less of it and still abstains.
- The trace-trained Jev is the bottleneck: step v1 recovered about 16% of the
  oracle's evidence gain (+0.024 of +0.150), and retraining on gold labels did
  not raise its held-out AUC (0.786 vs 0.781).

Defensible claim: step-wise reward measurably changes agent behaviour, and
judge quality decides how much of that a learned judge captures. Not yet
supported: step-wise RL beats outcome-only on accuracy.

## Jev-only rewards: the unverifiable-domain comparison (seed 1)

No ground truth in any training reward. An outcome Jev (Qwen3.5-4B LoRA) was
trained from training-pool rollouts labelled correct or incorrect; on 461
held-out final answers its AUC is 0.92 (accuracy 86.6%; mean P(correct) 0.69
when right, 0.17 when wrong). Two arms, same seed, questions and settings:

- Jev final: reward = outcome Jev's P(correct).
- Jev final + Jev step: the same, plus 0.3 x baseline-centred step-Jev scores
  (summed, capped at +/-1, repeats scored -0.5).

Jev final hit the 60-minute training cap after 8 updates; the other ran 10.
The outcome Jev tracked ground truth on live training rollouts throughout
(AUC 0.89 to 1.0 per update).

| Metric | No RL | Ground-truth outcome | Jev final | Jev final + step | Final + step - final (95% CI) |
|---|---|---|---|---|---|
| Accuracy (proxy) | 15.6% | 22.9% | 5.4% | 19.6% | +14.2 pts (+8.1, +20.4) |
| Gold evidence seen | 0.218 | 0.200 | 0.307 | 0.318 | +0.011 (-0.031, +0.054) |
| Gold evidence opened | 0.058 | 0.100 | 0.066 | 0.137 | +0.072 (+0.042, +0.103) |
| Tool calls | 3.6 | 4.1 | 10.5 | 6.5 | -4.0 |
| Rollouts repeating an action | 4% | 8% | 15% | 11% | -3.5 pts (-8.8, +1.9) |

Jev final + step vs no RL: accuracy +4.0 pts (-0.2, +7.9), evidence seen
+0.099 (+0.053, +0.146). Vs the ground-truth outcome arm: accuracy -3.3 pts
(-8.3, +1.5), evidence seen +0.118 (+0.064, +0.173).

Jev final alone was gamed: the policy padded searches (10.5 calls), its
training accuracy fell update by update, and held-out accuracy ended below
the untrained model. Adding step Jev kept search purposeful and ended above
no RL, within noise of a ground-truth verifier, while finding more gold
evidence than it.

Claim supported (one seed): where no ground-truth checker exists, a
judge-only outcome reward gets gamed, and step-wise Jev rewards prevent that
collapse (+14 pts accuracy here).

## Hosted Jev (TypeSafe System One) as both rewards (seed 1)

Gate on held-out data, zero-shot: step-judge AUC 0.891 (small trained step
Jev 0.786); final-answer verifier AUC 0.751 from the full trajectory (small
trained outcome Jev 0.921). Step baseline -0.17, fixed on training-pool steps.
Runs on an H100 with judging over the API.

| Metric | No RL | Small Jev final | Small Jev final + step | Hosted Jev final | Hosted Jev final + step |
|---|---|---|---|---|---|
| Accuracy (proxy) | 15.6% | 5.4% | 19.6% | 0% | 0% |
| Gold evidence seen | 0.218 | 0.307 | 0.318 | 0.000 | 0.046 |
| Tool calls | 3.6 | 10.5 | 6.5 | 0.0 | 0.3 |

Both hosted-Jev arms collapsed within about three updates: the policy learned
to answer at once without searching, with answers the hosted verifier rated
about 70% likely correct (its AUC against truth on training rollouts fell from
0.86 to 0.25). The hosted step reward did not prevent it: taking no steps
avoids step judging entirely.

Lesson: the verifier trained on this agent's own traces (AUC 0.92) held up
under RL with a step reward; the general zero-shot verifier (AUC 0.75) was
gamed in a few updates. Next proposed run: the trained outcome Jev as the
final reward with the hosted Jev, the better step judge, for steps.
