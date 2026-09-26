# Jev recipe: test plan and handoff

2026-09-25 · How we test the launch claim: what each run is for, how to start it, what we decide from it, and where things stand. The claim, the success bars (B1–B6) and their fallbacks are fixed in [launch-plan.md](launch-plan.md); this file doesn't change them. It is written so someone new (or a new chat) can pick the work up from here.

## 1. What we are proving

> Train a small model on your verifier's labels, and it gives the same answers as your verifier, much faster and cheaper, so you can check every step of your agent live.

- **The verifier** is Bandits' next-state judge: an LLM (Fireworks `nemotron-lightning-3p5-30b-a3b`) that reads one agent step (action + what happened next) and answers success / unclear / failure.
- **The student** is Qwen3.5-4B with a small LoRA adapter, trained to give the verifier's answers in one forward pass, with a probability for each answer.
- **Each run** below answers one part of the claim, or shows that part is false.

## 2. Where things stand

**Merged into `feat/jev`** (PR #64 is the umbrella into `main`; nothing merges to `main` without a decision): the whole pipeline, as the `jev` command in `recipes/jev/`.

| Piece | Command / file |
| --- | --- |
| Judge run → decision dataset, split by lineage into train/dev/calibration/test | `jev dataset`, `jev merge` |
| Train (LoRA), with the best checkpoint picked on dev | `jev train` |
| Score, calibrate (temperature), report | `jev score`, `jev calibrate`, `jev report` |
| Verifier cost from its ledger | `jev verifier-cost` |
| All of it, one command, resumable | `jev run --eval-split dev\|test` |
| A phase on a GPU box | `recipes/jev/scripts/gpu_run.sh` (`PHASE=1\|2`) |
| A phase on Modal | `recipes/jev/scripts/modal_run.py` |
| Real-model smoke test | `recipes/jev/scripts/smoke.py` |

Tested on CPU with a tiny model (`hf-internal-testing/tiny-random-gpt2`): 202 recipe tests, and full rehearsals of `gpu_run.sh` on the real data. **Nothing has trained on a real model at scale yet.** Qwen3.5-0.8B has been checked on CPU: it loads, the option letters are single tokens, and the loss falls.

**Not built** (issues filed):
- **#105** cheaper-LLM-judge baseline, needed for B3;
- **#106** verifier self-agreement, needed for B2;
- **#107** check against TRAIL human labels, needed for B4;
- **#109** learning curve;
- **#110** launch chart;
- **#98** more judged traces with 3 votes;
- **#97** real Jev API column;
- **#96** hosted training for the UI;
- **#70 / #71 / #72** API, UI, launch.

## 3. The data

Three Bandits turn-judge runs, in the gitignored `work/` folder. They share no traces.

| Source | Judge run | Ledger (for cost) | Traces | Steps |
| --- | --- | --- | --- | --- |
| TRAIL GAIA (web agents) | `turn-judge-0044b8f5c041155c` | `work/trail-ns/ledger-gaia-v2.jsonl` | 109 | 715 |
| TRAIL SWE (coding agents) | `turn-judge-b83480800ad90a25` | `work/trail-ns/ledger-swe-v3.jsonl` | 31 | 461 |
| tau2 (support agents) | `turn-judge-d450cf34d97b27a2` | `work/tau2-ns/ledger-judge-sub.jsonl` | 120 | 687 |

- **Split** (merged, by lineage, so retries of one session stay together): **train 1,097 · dev 271 · calibration 323 · test 172**.
- **Labels** are the verifier's vote shares, trained as soft targets. Today every step has **1 vote**, so every label is one-hot. #98 adds 3-vote runs.
- **Tied votes** are left out of accuracy and counted separately.
- **Label mix** is skewed on tau2 (~70% "success" in test), which is why the majority baseline is always shown.

Other ledgers in those folders belong to earlier or partial runs. `ledger-judge.jsonl` in `work/tau2-ns` is one; don't pass it with the others, or the cost is double-counted (the report warns).

## 4. The recipe

Copied on purpose from what worked in public:

| Piece | From |
| --- | --- |
| One forward pass, softmax over the answer letters (A/B/C) | Nimble / AutoJev |
| Cross-entropy loss, LoRA rank 16 on all linear layers, LR 5e-5, batch 8, 1 epoch | Nimble (Bespoke Labs) |
| Shuffle option order every time; best checkpoint on dev; one temperature on its own split | AutoJev |
| Train from the Base model | Kev |

**Ours:** 4B instead of Nimble's 9B (9B is the fallback), 8k-token prompts instead of 2k, and **labels from the verifier on real agent traces** instead of synthetic data.

## 5. The runs

### Run 1: does training work, and on which model? (phase 1, dev only)

- **What:** train **Qwen3.5-4B-Base** and **Qwen3.5-4B** (instruct) once each (seed 1) on the 1,097 train steps. Pick the best checkpoint on dev, fit the temperature on calibration, and score both on the 271 dev steps. Nothing touches test.
- **Why:** the first real-scale run. It shows:
  - whether the student beats the majority baseline at all;
  - which model to use;
  - the real speed and cost per decision.
- **Cost:** one L40S, about 1–2 h. That's ~$2–4 on Modal ($1.95/h; the Starter plan has $30/month free).
- **Command** (from the main checkout, with `work/` present):
  ```bash
  uvx modal run recipes/jev/scripts/modal_run.py --phase 1
  ```
  Or on any GPU box:
  ```bash
  PHASE=1 PROJECT_DIR=runs/jev-project \
  PROJECTS="work/trail-ns work/tau2-ns" \
  JUDGE_RUNS="turn-judge-0044b8f5c041155c turn-judge-b83480800ad90a25 turn-judge-d450cf34d97b27a2" \
  LEDGERS="work/tau2-ns/ledger-judge-sub.jsonl work/trail-ns/ledger-gaia-v2.jsonl work/trail-ns/ledger-swe-v3.jsonl" \
  INPUT_USD_PER_MTOK=0.05 CACHED_INPUT_USD_PER_MTOK=0.01 OUTPUT_USD_PER_MTOK=0.20 GPU_USD_PER_HOUR=<box price> \
  recipes/jev/scripts/gpu_run.sh
  ```
- **You get:** per model, a smoke result (`smoke-*.json`) and a dev report (`reports/<model>/report.md` + `report.json`), all in one tarball. The console streams live and is saved beside the tarball as `<run-dir>.console.log`. Its last lines print each model pinned to the exact snapshot it trained (`model@sha`); keep that line for run 2.
- **Decide:**
  1. **Stop check:** if neither model's `trained + calibrated − majority` agreement interval is above 0 on dev, the claim doesn't hold as built. Stop and choose between more data (#98), 9B, or a different claim before spending more.
  2. **Model:** the higher dev agreement (coverage-adjusted) wins; a tie goes to Base. This is launch plan §6.
  3. **Read the smoke results:**
     - letter mass (how weak the untrained baseline is);
     - rejected rows (should be ~0; prompts are under 4k tokens);
     - real latency and cost per decision, used to check B6 early (see §7).

### Run 2: the real numbers (phase 2, test, scored once)

- **What:** the chosen model × seeds **1, 2, 3**, each trained, calibrated and scored once on the 172 locked test steps. The same project is reused, so seed 1 is not retrained.
- **Why:** the launch numbers, with error ranges, and 3 seeds to show the result isn't luck.
- **Command:**
  ```bash
  uvx modal run recipes/jev/scripts/modal_run.py --phase 2 --models <chosen>@<sha from run 1> --seed <1|2|3>
  ```
  Run it once per seed, **one after another**, and add `--held-out-swe` to one of them for run 3. The pin keeps every seed on run 1's snapshot (an unpinned model takes whatever the Hub has that day, and seed 1 would then retrain). Running them one at a time lets each run reuse what the last one saved: a container sees the shared project as it was when it started.
- **Checks bars:** B1 (beats majority), B5 (calibrated), B6 (cheap and fast). B2, B3 and B4 need #106, #105 and #107 first (§6).

### Run 3: does it work on an agent it never saw? (part of phase 2)

- **What:** train on GAIA + tau2 only, and test on all of TRAIL SWE (461 steps). The report shows SWE as its own section.
- **Why:** answers "does this only work on your own data?". It has no bar and is reported as measured.

### Before run 2: the missing comparisons

| Build | Gives | Needs |
| --- | --- | --- |
| #98 more judged traces, 3 votes | ≥ 1,000 test steps (±~3 points instead of ±~7), and the votes #106 needs | Fireworks spend (~3k tokens per step per vote; key in `.env`) |
| #106 verifier self-agreement | B2's ceiling: how often the verifier agrees with itself | #98's 3-vote runs |
| #105 cheaper-LLM-judge column | B3: does the student beat just using a cheaper LLM? | a judge run with a cheaper Fireworks model |
| #107 TRAIL human labels | B4: student vs verifier vs humans | TRAIL annotations (Hugging Face, gated: accept the terms) |

The test split is scored once. If run 2 happens before these exist, B2–B4 can never be checked on that test split without cutting a new one.

## 6. Order

1. **Run 1** (phase 1). ~$3. Decide go/stop and pick the model.
2. In parallel: **#98** (judge spend), then **#106, #105, #107** (code).
3. **Run 2 + run 3** (phase 2) on the bigger dataset, once. That means a new dataset, so run 1 on it again first (dev only) to confirm the model choice.
4. **#109, #110** (curve, chart), then write-up and launch (#70–#72).

## 7. Known issues to read the results against

- **The untrained baseline is weak.** Untrained Qwen puts only ~5% of its probability on the answer letters, and the instruct model answers "unclear" almost every time. "Trained beats untrained" will look large and means little; read the student against the **majority** and **verifier** columns.
- **The verifier is already cheap.** At list price (verified 2026-09-25: $0.05 in / $0.01 cached in / $0.20 out per Mtok) the verifier costs about **$0.47 per 1,000 decisions** (tau2 ledger; output dominates, since the judge writes its reasoning). The student on a $1.95/h L40S is likely around $0.03 per 1k, **~15×, below B6's 50×**. Latency is the strong side: the verifier takes ~6.7 s per step (summed calls), the student likely ~0.05 s. If B6's cost half fails, launch plan §8 applies: lead on speed and running locally, not on cost.
- **The test set is small.** 172 steps gives ±~7-point intervals until #98 lands.
- **All labels are the verifier's opinion.** The claim is "copies your verifier", not "is right". Measured earlier against TRAIL's human error labels, the verifier itself scores F1 0.44 (SWE) and 0.53 (GAIA). #107 re-measures this for the student.

## 8. Picking this up

- **Code:** `git fetch && git checkout feat/jev`. The recipe lives in `recipes/jev/`, and core Bandits (`bandits/`) is left untouched on purpose: the recipe reads core, core never imports it.
- **Environment:** `cd recipes/jev && uv sync --extra dev` (add `--extra train` for torch). For CPU tests, install the CPU torch wheel as `.github/workflows/quality.yml` does. `uv run pytest` and `uv run ruff check .` must pass.
- **Data:** the `work/` folder in the main checkout (gitignored; ~83 MB for the two projects used). Git worktrees don't have it: run Modal from the main checkout with this branch checked out, or from a worktree with `JEV_WORK=<main checkout>/work`. The image is built from the checkout the script sits in; `.env` and other gitignored data are left out of it.
- **Modal:** the `modal` command on Laxman's machine is broken; use `uvx modal ...`. Two saved profiles, `laxmansrivast` (active) and `laxmanvidushi`. Launching spends money, so confirm the profile first.
- **Fireworks:** the key is in `.env` (`FIREWORKS_API_KEY`), used by `bandits judge-turns` (#98, #105).
- **Workflow:**
  - Each issue gets a branch off `feat/jev` and a PR into `feat/jev`.
  - For stacked PRs, **retarget each one to `feat/jev` before merging it**. #100 once merged into its stack branch and never reached `feat/jev`.
  - Reviews come from Alex (`Alex-Hunterz`) and CodeRabbit. Reply on each thread with what changed and in which commit.
- **Key docs:** [launch-plan.md](launch-plan.md) (claim, bars, fallbacks) · this file (runs) · [decision-models-learnings.md](decision-models-learnings.md) (research on Jev and its clones) · [decision-models-plan.md](decision-models-plan.md) (original plan, superseded for launch).

## 9. Open decisions

1. Launch run 1 now: which Modal profile, and who watches it?
2. Approve the Fireworks spend for #98 (and #105's cheaper-judge run).
3. If B6's cost half fails: lead the launch on speed / running locally (the §8 fallback), or look for a claim where cost wins (e.g. against a frontier judge as the verifier)?
