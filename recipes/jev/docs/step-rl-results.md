# Step-wise Jev rewards for RL on a search agent

Status 2026-10-07. One seed per arm; directional, not a launch claim.

## Question

Does a step-wise reward from Jev help RL train a search agent, and does it
still help when no ground-truth checker exists (an unverifiable domain), where
the final-answer reward must also come from Jev?

## Setup

- **Policy:** Qwen3-4B-Instruct-2507 (`cdbee75f`), LoRA rank 32, served by
  vLLM. Tools: `search`, `open`, `find` (one call per turn, at most 12 turns,
  the last turn must answer) over the BrowseComp-Plus corpus with its
  Qwen3-Embedding-8B index.
- **Questions:** the 830 BrowseComp-Plus questions are hash-split into a
  415-question training pool and 415 held out; 60 held-out questions are the
  evaluation set. Nothing held out is used for training, judge training or
  tuning.
- **Training:** Dr. GRPO, no KL, one update per batch; 10 updates of 16
  training questions x 8 rollouts; lr 1e-4. Every arm sees the same questions
  in the same order for a given seed. Training uses the exact token ids vLLM
  sampled.
- **Evaluation:** 60 held-out questions x 8 rollouts. Accuracy is a strict
  normalised string match against the gold answer (a proxy, not the official
  LLM grader). "Evidence seen" and "evidence opened" are the share of a
  question's gold evidence documents that appeared in search results or were
  opened. Differences are paired by question with 95% intervals from
  resampling questions (`scripts/compare_runs.py`).

## Judges

| Judge | What it is | Held-out quality |
|---|---|---|
| Step Jev v1 | Qwen3.5-4B-Base LoRA trained on 252 search steps labelled by a Fireworks teacher (OpenResearcher traces) plus 80 synthetic negatives | 78/100 steps on a fresh audit; AUC 0.78 for steps that found gold evidence |
| Step Jev v2 | Same recipe, retrained on gold-evidence step labels from our rollouts | AUC 0.786 (no gain over v1) |
| Outcome Jev | Same base, trained on training-pool final answers labelled correct or incorrect (777 rows, 591 for training); sees the question, actions, last 3 tool results and the answer | AUC 0.92 on 461 held-out answers |
| Hosted Jev | TypeSafe's `jev-latest` over the API, zero-shot; final answers judged from the full trajectory | Step AUC 0.891; final-answer AUC 0.751 |

Judge inputs never contain the gold answer.

## Arms and held-out results (seed 1)

| Arm | Final-answer reward | Step reward | Accuracy | Evidence seen | Evidence opened | Tool calls | Repeats |
|---|---|---|---|---|---|---|---|
| No RL | - | - | 15.6% | 0.218 | 0.058 | 3.6 | 4% |
| Outcome (ground truth) | string match | none | 22.9% | 0.200 | 0.100 | 4.1 | 8% |
| Step v1 | string match | step Jev, summed, capped | 20.6% | 0.224 | 0.136 | 7.4 | 26% |
| Step v2 | string match | step Jev mean, all-wrong groups only | 15.8% | 0.209 | 0.074 | 2.8 | 0% |
| Oracle step | string match | gold evidence recall | 17.3% | 0.350 | 0.049 | 7.3 | 7% |
| Jev final | outcome Jev | none | 5.4% | 0.307 | 0.066 | 10.5 | 15% |
| **Jev final + step** | outcome Jev | step Jev, repeats penalised | **19.6%** | 0.318 | 0.137 | 6.5 | 11% |
| Hosted Jev final | hosted Jev | none | 0% | 0.000 | 0.000 | 0.0 | 0% |
| Hosted Jev final + step | hosted Jev | hosted Jev | 0% | 0.046 | 0.000 | 0.3 | 0% |

Jev final stopped at 8 of 10 updates (60-minute training cap); its accuracy
was falling with each update.

Key paired differences (95% CI):

- Outcome vs no RL: accuracy +7.3 pts (+1.7, +13.5).
- **Jev final + step vs Jev final: accuracy +14.2 pts (+8.1, +20.4).**
- Jev final + step vs no RL: accuracy +4.0 pts (-0.2, +7.9).
- Jev final + step vs outcome (ground truth): accuracy -3.3 pts (-8.3, +1.5);
  evidence seen +0.118 (+0.064, +0.173).
- Jev final vs no RL: accuracy -10.2 pts (-16.5, -4.6).
- Oracle step vs outcome: evidence seen +0.150 (+0.101, +0.201); accuracy
  -5.6 pts (-12.1, +1.3).
- Step v1 vs outcome: evidence opened +0.036 (+0.009, +0.064); accuracy -2.3
  pts (-8.5, +4.4).
- Step v2 vs outcome: accuracy -7.1 pts (-14.2, -0.2).

## Findings

1. **Without a ground-truth checker, a judge-only outcome reward is gamed,
   and step-wise Jev rewards prevent it.** With the outcome Jev as the only
   reward the policy padded its searches (10.5 calls) and ended below the
   untrained model. Adding the step Jev kept search purposeful and gave +14
   pts over Jev final, within noise of a ground-truth verifier, while finding
   more gold evidence than it.
2. **The verifier must be trained on the agent's own traces.** The hosted Jev
   used zero-shot as the final-answer verifier (AUC 0.75) was gamed within
   about three updates: the policy answered at once without searching, with
   answers the verifier rated about 70% likely correct, all wrong. Its step
   reward could not stop this, because taking no steps avoids step judging.
   The outcome Jev trained on this agent's traces (AUC 0.92) held up.
3. **With a ground-truth checker, step rewards change behaviour more than
   accuracy.** A ground-truth step signal (oracle) raised gold evidence found
   by 75% over outcome-only but not accuracy within 10 updates; outcome-only
   remained the most accurate arm. Summing step-Jev scores rewarded padding;
   averaging them rewarded stopping early. How a step reward is aggregated
   shapes how much the agent searches.
4. **The trace-trained step Jev recovers about a sixth of the oracle's
   evidence gain** (+0.024 of +0.150). Retraining it on gold-evidence labels
   did not help; from one step's snippets a judge cannot reliably tell the
   right evidence from relevant-looking results. The hosted Jev is a stronger
   step judge (0.891).

## Reproducing

Data (files land in the ignored `work/step-rl/`):
`scripts/step_rl_split.py`, `scripts/step_rl_tasks.py`.

Judges:
- Step Jev v1: `scripts/search_reward_data.py`, `search_reward_negatives.py`,
  `search_reward_probes.py`, then `modal_search_train.py`; audit with
  `modal_search_expanded.py`.
- Step Jev v2: `scripts/gold_step_labels.py`, `modal_judge_v2.py`.
- Outcome Jev: `scripts/outcome_labels.py`, `modal_outcome_judge.py`.
- Hosted Jev gate: `scripts/real_jev_gate.py` (needs `JEV_API_KEY`).

Training and evaluation (`scripts/modal_grpo.py`; deploy it, then spawn so a
sleeping laptop cannot end a run):

```bash
uvx modal deploy recipes/jev/scripts/modal_grpo.py
uvx --from modal python recipes/jev/scripts/spawn_train.py <arm> 1 <step_weight> <baseline> [api] [tag]
uvx --from modal python recipes/jev/scripts/spawn_eval.py <run> [--untrained]
```

Arms are named in `bandits_jev/grpo.py` (`outcome`, `step`, `step2`, `oracle`,
`jev_final`, `jev_final_step`). The step baseline is the mean step score on
non-evidence training-pool steps: 0.20 for the small step Jev, -0.17 for the
hosted Jev; the hosted-Jev arms pass `api`. `Trainer` (H200) holds vLLM, the
policy update, both small Jevs and the retriever; `ApiTrainer` (H100) judges
over the API. Each run writes `metrics.jsonl`, per-update rollouts, adapters
and `eval.json` to `/runs/step-rl/grpo/<run>/` on the `jev-runs` volume.
Analysis: `scripts/compare_runs.py`, `scripts/analyze_rollouts.py`.

The no-RL baseline and the calibration rollouts were produced by an earlier
standalone harness using the same `bandits_jev/rollout.py`;
`spawn_eval.py base --untrained` reproduces the baseline evaluation.

## Caveats

- One seed per arm; Jev final ran 8 updates, the others 10.
- Accuracy is a strict string-match proxy.
- Hyperparameters (lr, step weight 0.3, cap 1) were not tuned.
- Total Modal compute for the study was roughly $50 to $55, estimated from list prices.
