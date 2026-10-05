# Search step reward pilot

2026-10-05. This is an experiment log, not a launch result.

## Question

Can a small Jev-style decision model identify useful search actions from the
actual tool result well enough to serve as a step reward for Qwen3-4B agent RL?
The terminal answer reward remains separate.

## Data boundary

- Judge training and development: actions and tool responses from
  [OpenResearcher-Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset),
  `seed_42`, dataset row offsets 62–161 (100 distinct task IDs, 69–191). The teacher is
  `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`.
- Search transfer audit: distinct OpenResearcher row offsets 0–61 (task IDs
  0–68), with 33 sampled
  steps labeled under rubric `search-reward-v2`. No audit row enters training.
- Attack probes: six hand-authored observed search steps, in
  `scripts/search_reward_probes.py`. They do not enter training.
- BrowseComp-Plus: all 830 queries are evaluation-only. The decrypted JSONL
  SHA-256 is `0374946363a5c59f4d3707fa9c31b9e5152269c0f024668a71687dda4e48a85e`.
  The upstream checkout is at `046949032b0328319cc9a02663a759ec601d9402`.
  Plaintext questions and answers stay under ignored `work/step-rl/`.

The extractor drops the gold answer, final outcome, and any tool call without
an actual tool response. It includes the last two tool interactions as context.
The teacher saw the same action and response as the eventual small judge,
plus instructions to treat tool text as untrusted data. Exact normalized
question overlap between the 162 sampled OpenResearcher tasks and the 830
BrowseComp-Plus questions was zero. This does not rule out semantic overlap.

## First gate: existing September judge

The AgentProcessBench LoRA adapter, step 250, on pinned
`Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`, scored
12/33 = **36.4%** against the revised teacher labels. The neutral-majority
baseline was 20/33 = **60.6%**. It called 17 neutral steps positive. No rows
were rejected by the scorer. Full predictions are saved locally at
`work/step-rl/search-reward-gate.json` and on the `jev-runs` Modal volume at
`/search-reward-gate.json`. This checkpoint fails the search reward gate; do
not use it for policy RL.

The first teacher rubric was inconsistent on reasonable searches that found no
answer. Rubric v2 treats those as neutral unless the action itself is clearly
bad. On the same eight-step failure sequence, v2 changed two such labels from
negative to neutral and kept the off-task query negative.

## Search-specific judge pilot

The pilot import has 332 decisions: 252 teacher-labeled observed steps and 80
programmatically constructed training-only negatives (an identical repeat and
an off-task search on each of 40 training tasks). The task-disjoint splits are
272 train, 29 dev, 23 calibration, and 8 internal test; import rejected zero.
The hand-authored attack probes remain outside these counts. Provider rate
limits caused 16 initial teacher calls to fail, and two were later recovered;
failed calls never became training labels.

Pilot job: `uvx modal run recipes/jev/scripts/modal_search_train.py`. It trained
the 4B model for two epochs with LoRA on an L40S; step 50 was selected on dev
(26/29 = 89.7%). The search-specific model scored **28/33 = 84.8%** on the
v2 transfer audit and **5/6** hand-authored attack probes. It gave no attack
a positive verdict. The one probe error was a fabricated result claim judged
neutral rather than negative. Full predictions and the training log are at
`work/step-rl/search-judge-pilot-report.json` locally and
`jev-runs:/search-judge-pilot/report.json` on Modal. This is a pilot diagnostic.
The immutable input snapshots are `work/step-rl/search-train-combined.jsonl`
(SHA-256 `5222876d1de3deca5e23c7d32eed9a953b66cb6d849eb2110053ef8305813e09`),
`work/step-rl/pilot-audit-33.jsonl`
(`dcf207fcef1e2ac16b8d89a2aa25fc9a34650247a627168dea6240b2bf7f139b`),
and `work/step-rl/search-reward-probes.jsonl`
(`bde8c83608f0aba415d51c59e8c861739bd2fe527747fe61b167984ae800254f`).

Reviewing its apparent false positives exposed a defect in rubric v2: a
search that found the **exact requested article** was labeled neutral because
the answer was not yet visible in the search snippet. Another search that
found the exact film page was treated likewise. That is useful partial
progress, which the eventual reward needs to recognize. Rubric v3 explicitly
credits a specific credible source to inspect next. On these examples it
labels both positive while retaining a merely topical search as neutral.
This revision was made after inspecting v2 predictions, so the v2 audit is
spent. A new v3 audit came from separate OpenResearcher row offsets 162–241
(80 candidate task IDs, 193–276), with no overlap with judge training or the
earlier audit. The model was frozen before this audit. One teacher call failed
to produce a parseable label and was excluded; the 100 scored steps span 62
tasks. The immutable labeled file is
`work/step-rl/openresearcher-v3-audit-labeled.jsonl`.
Its SHA-256 is `437240a45865514bd7a44664b23582d4f1f4d31265408aa98d2052469231e8f9`.

The new judge got **78/100 = 78%**, versus **60/100 = 60%** for the old
general judge and **59/100 = 59%** for always predicting the majority class.
The new judge recognized 50/59 positive steps and correctly withheld positive
reward on 33/41 neutral steps. Its eight false-positive rewards are mostly
topical searches without a specific source or evidence. A task-clustered
bootstrap (10,000 resamples, seed 20261005) gives a 69.4–86.3% interval for
new-judge accuracy and a 5.1–30.9 percentage-point interval for its gain over
the old judge. This audit has no natural negative examples, so it does not
measure negative-class recall. Full predictions are in
`work/step-rl/search-judge-v3-audit.json` locally and
`jev-runs:/search-judge-pilot/v3-audit.json` on Modal. The local report SHA-256
is `2a2ba5aa732a1d1fc424c813a4744988f016660aa2cd72f8a98a84094d40978f`.
The six attack probes
remain a small, separate diagnostic: none got positive reward, and one
fabricated-result claim was neutral rather than negative.

## Next gate

The search judge passes the limited transfer gate above. Before using it as a
policy reward, add action/provenance validation at the reward boundary and
run a live-trajectory smoke test; the offline audit cannot reveal failures
caused by the policy adapting to the judge. Monitor false-positive rewards and
ground-truth success together during any RL pilot.
Final-answer-only actions receive no step score under the current Jev contract;
premature completion and agreement instead of solving must be penalized by
the terminal outcome reward. If the judge gate fails, revise labels and
retrain; do not run outcome-plus-step RL.

If it passes, build one pinned multi-turn search harness and run paired
outcome-only and outcome-plus-step RL from the same Qwen3-4B policy checkpoint.
Use the official BrowseComp-Plus answer-grading procedure for held-out eval;
the small step judge must never grade the final benchmark result.
