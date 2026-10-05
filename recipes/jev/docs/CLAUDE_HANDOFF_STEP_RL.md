# Claude Code handoff: step reward RL launch pilot

Date: 2026-10-05  
Workspace: `/home/laxman/Desktop/bandits`

## Objective

Continue the preliminary Twitter launch experiment. Train a small Jev style
judge on observed browser action/result traces, freeze it, then test whether
adding its per-step reward improves a Qwen 4B search policy over matched
outcome-only GRPO. BrowseComp-Plus is evaluation only. Do not announce a policy
RL result until both paired seeds pass the stated gates.

## Where the code is

All paths below are repository relative.

- `recipes/jev/scripts/search_reward_data.py`: fetches observed
  OpenResearcher browser action/result steps and labels them with the existing
  Fireworks verifier. The prompt excludes final answers and treats tool text as
  untrusted. Current rubric is v3.
- `recipes/jev/scripts/search_reward_negatives.py`: creates repeat/off-topic
  negative training rows using training task IDs only.
- `recipes/jev/scripts/search_reward_probes.py`: six fixed hand-authored judge
  probes; these are not training rows.
- `recipes/jev/scripts/modal_search_train.py` and
  `recipes/jev/scripts/search_train_worker.py`: Modal LoRA training of the
  search-specific 4B judge.
- `recipes/jev/scripts/modal_search_score.py` and
  `recipes/jev/scripts/search_score_worker.py`: evaluates the prior Agent
  Process Bench judge on the first search audit.
- `recipes/jev/scripts/modal_search_expanded.py` and
  `recipes/jev/scripts/search_expanded_worker.py`: compares the prior judge
  and the new search judge on the fresh rubric-v3 audit.
- `recipes/jev/bandits_jev/step_reward.py`: pure token reward accounting
  helper, with tests in `recipes/jev/tests/test_step_reward.py`. It is not yet
  wired into veRL.
- `recipes/jev/tests/test_search_reward_data.py`: checks extraction, answer
  leakage boundary, and synthetic negatives.
- `recipes/jev/docs/step-reward-rl-pilot.md`: exact judge data boundaries,
  metrics, limitations, and artifact hashes.
- `recipes/jev/docs/step-rl-launch-runbook.md`: intended RL experiment, gates,
  and harness contract.
- `README.md`: two stale `vektori-ai` URLs were changed to `bandr-ai`.

## What actually ran

1. Existing APB Jev transfer check: 12/33 (36.4%) against the v2 teacher
   labels, below the 20/33 neutral-majority baseline. Do not use that adapter
   as the step reward.
2. Search-specific Qwen3.5-4B-Base LoRA judge trained on 332 rows (252 teacher
   labels plus 80 training-only synthetic negatives). Modal L40S, two epochs;
   selected step 50 with 26/29 dev accuracy.
3. Initial v2 check: 28/33 (84.8%); 5/6 attack probes classified exactly.
   No attack probe received positive reward. One fabricated-result probe got
   neutral instead of negative.
4. Reviewing v2 errors found the rubric was too strict about exact relevant
   source discovery. Rubric v3 now credits a specific credible page to inspect
   next even if the answer is not in the search snippet. The new audit was
   labeled after that rubric change, on fresh tasks; the judge remained frozen.
5. Fresh v3 audit: 100 observed steps from 62 tasks. New judge 78/100 (78%),
   old judge 60/100, majority baseline 59/100. New judge: 50/59 positive steps
   detected and 33/41 neutral steps correctly not given positive reward. It
   has no natural negative examples, so this does not establish negative
   recall. The eight false-positive rewards were mostly broadly topical
   searches. This is a limited transfer gate, not proof of RL gain.
6. `uv run pytest tests/test_search_reward_data.py tests/test_step_reward.py
   -q` passed (9 tests); Ruff passed on these scripts/tests; `git diff
   --check` passed.

No policy RL, base 830-question run, official answer evaluation, or frontier
judge cost run has happened. Do not imply otherwise.

## Artifacts and reproducibility

Large/private-to-the-workspace artifacts are under ignored `work/step-rl/` and
are not committed. Inspect before running; do not assume another checkout or
machine has them.

- `search-train-combined.jsonl`: training/dev/calibration/test judge rows;
  SHA-256 `5222876d1de3deca5e23c7d32eed9a953b66cb6d849eb2110053ef8305813e09`
- `pilot-audit-33.jsonl`: spent rubric-v2 audit; SHA-256
  `dcf207fcef1e2ac16b8d89a2aa25fc9a34650247a627168dea6240b2bf7f139b`
- `openresearcher-v3-audit-labeled.jsonl`: fresh judge audit; SHA-256
  `437240a45865514bd7a44664b23582d4f1f4d31265408aa98d2052469231e8f9`
- `search-reward-probes.jsonl`: six fixed probes; SHA-256
  `bde8c83608f0aba415d51c59e8c861739bd2fe527747fe61b167984ae800254f`
- `search-judge-pilot-report.json`: pilot metrics/predictions, if still present.
- `search-judge-v3-audit.json`: fresh audit metrics/predictions; SHA-256
  `2a2ba5aa732a1d1fc424c813a4744988f016660aa2cd72f8a98a84094d40978f`
- BrowseComp-Plus decrypted answers are in ignored work storage. They must
  stay evaluation-only and must never enter judge inputs or policy training.
- Fixed BrowseComp-Plus corpus and Qwen3-Embedding-8B index are downloaded
  under `work/step-rl/corpus/` and `work/step-rl/indexes/` (about 3.3 GB).
- Upstream ignored checkouts: `work/browsecomp-plus-upstream/`,
  `work/search-r1-upstream/`, `work/trace-repro/`, `work/verl-upstream/`.
  The inspected veRL commit is
  `8718ca30a3f002f93b7c4fd99b9b2506718681bc`.
- The trained adapter and source training report are on Modal volume
  `jev-runs` under `/search-judge-pilot/`; the fresh audit report is at
  `/search-judge-pilot/v3-audit.json`. Re-download with `uvx modal volume get`
  if local copies are missing. Modal profile used was `laxmansriv`.

## First commands for Claude Code

Run from repo root. First check `git status --short`, ignored files, and that
Modal and Fireworks authentication exist without printing secrets. Keep the
pre-existing untracked `scripts/trail_signal.py` untouched.

```bash
cd /home/laxman/Desktop/bandits
git status --short
cd /home/laxman/Desktop/bandits/recipes/jev
uv run pytest tests/test_search_reward_data.py tests/test_step_reward.py -q
uv run ruff check bandits_jev/step_reward.py tests/test_step_reward.py \
  scripts/search_reward_data.py scripts/search_reward_negatives.py \
  scripts/search_reward_probes.py scripts/search_score_worker.py \
  scripts/search_train_worker.py scripts/search_expanded_worker.py \
  scripts/modal_search_score.py scripts/modal_search_train.py \
  scripts/modal_search_expanded.py
```

Prior Modal invocations, for reference (training re-runs consume GPU time):

```bash
cd /home/laxman/Desktop/bandits
uvx modal run recipes/jev/scripts/modal_search_train.py
uvx modal run recipes/jev/scripts/modal_search_expanded.py
uvx modal volume get jev-runs /search-judge-pilot/report.json \
  work/step-rl/search-judge-pilot-report.json
uvx modal volume get jev-runs /search-judge-pilot/v3-audit.json \
  work/step-rl/search-judge-v3-audit.json
```

## Required next work

Do these in order. Do not jump straight to the 830-question run.

1. Inspect the actual installed/current veRL API against
   `work/verl-upstream/`. The default naive reward manager is terminal-only;
   `tool_rewards` being recorded does not itself put those rewards in policy
   token scores.
2. Implement the multi-turn search tool environment and custom reward manager.
   Capture the last policy token position for every browser action *before*
   appending the tool response. Score only action plus real result; reject
   missing provenance and exact repeats. Put final-answer reward on the
   terminal policy token. Use `bandits_jev.step_reward.reward_vector` or revise
   it with tests if the pinned veRL response-mask contract differs.
3. Add a deterministic hand-built rollout test proving the action reward lands
   on the right generated token, tool tokens get zero, and outcome reward is
   separate. Then run 10 training-only online smoke tasks and inspect prompts,
   full traces, token masks, scores, and errors manually.
4. Decide and generate policy training questions from a distinct task set.
   Do not train on BrowseComp-Plus's 830 questions. Save task provenance and
   exact policy checkpoint revision. The current OpenResearcher examples train
   the **judge**, not the search policy.
5. Run the base policy with 8 rollouts per BrowseComp-Plus question using the
   frozen harness and official ground-truth grader. Freeze the never-solves
   bucket before viewing any RL results.
6. Run short lambda 0.1/0.3 pilots on separate training-domain validation
   tasks, then two matched seeds each for outcome-only and outcome-plus-step.
   Same policy checkpoint, tool limits, updates, rollouts, and final grader.
7. Only consider the public launch if our arm wins both paired seeds on total
   success and on the frozen never-solves bucket, with judge reward tracking
   ground-truth success. Report preliminary scope and paired uncertainty.

## Known cautions

- The fresh v3 audit shares the external OpenResearcher corpus domain with
  training; exact normalized task-question overlap with all 830 benchmark
  questions was zero, but semantic overlap was not ruled out.
- The v3 test has positive/neutral labels only. Keep attack probes and add
  independently human-reviewed negatives before treating negative judgments
  as reliable.
- Do not revise rubric/model after seeing audit errors and continue reporting
  the same audit as held out; it becomes spent and a new audit is needed.
- The small judge cannot grade final answers. Use only official BrowseComp-Plus
  ground truth for benchmark success.
- A launch cost comparison needs measured calls/tokens and actual prices; any
  extrapolation from a partial frontier-judge run must be labeled.
- The user asked for a Modal compute spending cap. A multi-GPU policy run has
  not started; ask for or read the user's cap before launching the expensive
  paired runs. Judge pilot GPU jobs already ran on Modal.

## Workspace hygiene

At handoff, all step-RL changes are uncommitted and show as untracked except
`README.md`. Keep changes scoped. Do not add the ignored `work/` data, private
keys, or the unrelated `scripts/trail_signal.py`. Before finishing, report
which code is implemented, which tests ran, which runs actually completed,
and any remaining gates. Do not claim RL success before the policy runs.
