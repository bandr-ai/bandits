# Bandits Decision Models: Plan V3

2026-09-23 · the build plan for Decision Models (tracking issue #73, features #65–#72). Research background: [decision-models-learnings.md](decision-models-learnings.md).

## 1. What we are building

A feature inside Bandits that lets anyone **train their own small Jev-style decision model** and see if it's any good:

1. bring labeled decisions (a JSONL file, or Bandits traces later);
2. score them with an untrained model (the baseline);
3. fine-tune a small model with SFT;
4. fix its confidence (temperature calibration);
5. compare untrained vs trained vs real Jev on the same held-out questions;
6. try it in a local UI and call it from a local API.

Then launch it publicly (Twitter). No deadline; the goal is working software.

A "Jev-style model" here means: a normal LLM that reads a state, a question and 2–26 options, does **one forward pass** (two if we average two option orders), and returns a probability for every option (softmax over the option-letter logits). No text generation.

## 2. Locked decisions

| Decision | Choice |
| --- | --- |
| Where | A recipe on top of Bandits: `recipes/jev/` (package `bandits_jev`, command `jev`). It reads Bandits core (traces, judge runs, the artifact store) and core never imports it. Heavy deps (torch, transformers, peft) in the recipe's own `train` extra. |
| Training method | **SFT only**: cross-entropy on the option logits, LoRA. No RL, no "RLCD". |
| Recipe | **Nimble-style core** (LoRA r16 on all linear layers of a 4B/9B Qwen, cross-entropy over A–Z option-letter logits, LR 5e-5, batch 8, 1 epoch; full recipe in A2) **+ AutoJev's calibration and discipline** (option shuffling, temperature on its own split, checkpoint picked on dev). Not AutoJev's 27B full-weight training or 255-slot head (needs an H200 + ~389 GB RAM). |
| Confidence | One temperature per trained model, fitted on a separate calibration split. |
| First use case | **Bring your own labeled data** (a public human-labeled dataset for the demo). |
| Bandits judge distillation | Track B, after the launch. |
| Question type | `choice` only (2–26 options, letters A–Z). |
| UI | One polished page (FastAPI + plain HTML/JS) served by Bandits. No React, no Gradio. |
| Compute | Local single GPU first. Colab/Modal launch button later, as a thin wrapper around the same code. |
| Branching | Nothing gets merged into `main` for this work until we decide. See §8. |

## 3. What exists today

- PR #64 (`feat/jev`, replaces #63): the `DecisionDataset` format + a compiler from Bandits judge verdicts, plus dense `judge_votes`. **Not merged. Not to be merged yet.**
- **Limit:** that format was built for the judge only. Every row must share one question and one option set, splits are only `within_family_fit` / `within_family_held_out`, `family_id` is required, and rejected rows need a `trace_id`. It can't hold user datasets with a different question or options per row (e.g. CaseHOLD), or train/dev/calibration/test splits. A1 must generalize it first.
- Datasets already in the Bandits repo (all agent traces, gitignored): SWE-bench Lite (291 SWE-agent trajectories, resolved/unresolved per trace), tau-bench airline (200 Claude Sonnet 4.5 runs, 13 tools, 144 pass / 56 fail), tau2 handoff (16 mined families). Good for Track B; see A0 for one Track A idea.
- Nothing else: no scorer, no trainer, no UI.

### Reuse from Bandits (checked in the repo)

| Existing piece | Where | Use in V3 |
| --- | --- | --- |
| Content-addressed artifact store | `bandits/store.py` (`DerivedStore`) | Save datasets, scorer runs, training runs, calibrations, bundles |
| Typer CLI | the recipe's own `jev` command (`bandits_jev/cli.py`) | `jev dataset`, `jev import`, `jev score`, `jev train`, … |
| Optional-extra pattern | `pyproject.toml` (`audit`, `emulate` extras) | New `decide` extra for torch/transformers/peft |
| Tests inject a fake predictor | existing test suite | Fake tiny model for scorer/trainer tests |
| Brier, ECE, reliability bins | `scripts/trail_nextstate_eval.py` (binary, script-only) | Move into `bandits_jev/metrics.py`, extend to multi-option |
| Percentile bootstrap | `bandits/emulate/report.py` (`_bootstrap_mean_interval`) | Generalize to paired, grouped bootstrap |
| Retrying HTTP client | `bandits/transport.py` (Fireworks-specific) | Pattern for the Jev API client |

Watch-outs:
- **No web server or UI exists in Bandits.** The UI and `/v1/decisions` API add the first server dependency (FastAPI/uvicorn) → put them in the `decide` extra or a `ui` extra, not core.
- **No torch anywhere yet.** Core stays at three dependencies; everything heavy is lazy-imported.
- **"SFT" already means something in Bandits**: `build-sft` / `export-nextstate` *export* chat SFT rows for training an agent. Our feature *trains* a decision model. Use distinct names (`jev train`, never `sft`) to avoid confusion.

## 4. Track A: the launch path (build this now)

Each step has a "done when" check. The next step starts only after that check passes.

**Four splits, fixed before any training:**

| Split | Used for |
| --- | --- |
| train | training |
| dev | picking the dataset, the go/no-go, settings, early stopping |
| calibration | fitting the temperature only |
| test (locked) | the final chart, run once |

If we change anything after looking at test results, that test set becomes a dev set and we cut a new locked test set.

### A0. Shortlist demo datasets (no model needed)

Criteria:
- 26 or fewer options;
- labels checked by humans (not made by an LLM);
- enough rows for train / calibration / test (a few thousand minimum);
- **room to improve**: the untrained model must score clearly below where trained models usually get;
- a license that allows redistribution of results;
- easy to understand in a screenshot;
- **shows why the Jev shape matters** (a state + a question + choices that change per item + probabilities), not just a fixed-label classifier in a wrapper. Pick the winner on both improvement and this.

Shortlist (exist on Hugging Face; licenses match the dataset cards; **human-labeled still to verify for each**):

| Dataset | Task | Options | License | Why |
| --- | --- | --- | --- | --- |
| `zeroshot/twitter-financial-news-topic` | Topic of a finance tweet | 20 | MIT | Finance is where Jev beats open models; label taxonomy is dataset-specific, so SFT should gain a lot |
| LexGLUE `case_hold` (`coastalcph/lex_glue`) | Pick the correct legal holding | 5 | CC-BY-4.0 | Law is Jev's strongest area vs open models; hard; impressive if it moves |
| `jackhhao/jailbreak-classification` / `deepset/prompt-injections` | Is this prompt an attack? | 2 | Apache-2.0 | Agent-safety angle, very tweetable; but small datasets (wide error bars) |
| `tals/vitaminc` | Does the evidence support / refute / not decide the claim? | 3 | CC-BY-SA-3.0 | Better as the external check set than as the demo |

Rejected: Banking77 (77 options, over the limit), ContractNLI (non-commercial license).

Bandits-native idea (optional): **next-tool prediction** from the tau-bench airline traces ("which of the 13 tools does the agent call next?"). The labels are real agent actions, not human judgments, so it teaches "copy Claude's routing", not "be correct". It's also small. Better kept as a secondary demo than the headline.

Done when: each candidate's label source and license checked, and the success bar written down. Suggested bar: trained beats untrained on **dev** by a margin whose 95% interval excludes zero, and loses nothing meaningful on the external set. The winner is picked after A1's scorer runs on each candidate's **dev** split.

### A1. Import + prompt + untrained scorer

- **Generalize the `DecisionDataset` format first** (new schema version):
  - question and options per row (the one-shared-schema check becomes optional, for producers like the judge compiler);
  - splits `train` / `dev` / `calibration` / `test`; the judge compiler maps fit → train and divides held-out lineages across dev/calibration/test;
  - `family_id` becomes an optional `group_id` (keeps related rows in one split and drives bootstrap resampling);
  - rejected rows point at a source record (file + line number, or trace + turn) instead of always a trace;
  - judge-only count fields become optional.
  The existing judge compiler stays and keeps passing its tests as one producer.
- `jev import file.jsonl` → that format. Hard or soft labels, optional split/group/source/license fields, bad lines quarantined with line number and reason.
- A JevBench importer built on top of it (keeps each item's source and license).
- One versioned prompt builder used everywhere (scoring, training, serving). The prompt text is hashed and saved with every result.
- Scorer: one forward pass → option-letter logits → softmax. Raw logits kept. Checks that each letter is one token. Rejects prompts over the length limit (8k tokens to start, configurable) instead of cutting them.
- Two modes: one option order, or two orders averaged (two passes; measured, not assumed to help; report its extra latency).

Done when: tests pass using a fake tiny model (no 4B download in CI), the judge compiler still passes, and a real run on each shortlisted dataset's dev split produces a saved result. Then pick the demo dataset.

### A2. SFT spike: go / no-go

**The recipe (SFT only, no RL):**

1. **Input:** each training row becomes the exact A1 prompt (state, question, options lettered A, B, C…, then `Answer:`). The correct answer is never in the prompt.
2. **Shuffle** the option order for every row each time it is seen; the target letter moves with its option.
3. **One forward pass** of the base model + LoRA. Take the logits at the last position, keep only the letters for this row's options, softmax them.
4. **Loss:** cross-entropy between those probabilities and the correct option (one-hot). Soft targets use the same formula; not used in the launch path.
5. **Update only the LoRA weights**; the base model stays frozen.
6. **Every N steps**, score dev; keep the checkpoint with the best dev result, not the last one.
7. **After training**, fit the temperature on the calibration split (A3).

**Default settings (Nimble's published config):**

| Setting | Value |
| --- | --- |
| Base | Qwen3.5-4B or 9B, pinned revision (Q2) |
| LoRA | rank 16, alpha 32, dropout 0.05, on all linear layers (attention + MLP) |
| Learning rate | 5e-5, linear schedule with warmup |
| Effective batch | 8 (e.g. 2 × 4 gradient accumulation) |
| Epochs | 1 |
| Precision | BF16 |
| Max length | 8k tokens (Nimble used 2k; long states are rejected, not cut) |

Fallback if the external set shows a regression: attention-only LoRA and/or lower LR (reflex / decision-head advice).

- Written as the real trainer from day one, not a throwaway script.

Done when: result on **dev** compared with the success bar from A0. **If it fails, we change dataset or setup before building the UI.** Test is not touched here.

### A3. Calibration + report

- Fit one temperature on the calibration split. Never on test.
- Report: accuracy, macro F1, NLL, Brier, ECE, reliability chart, option-order sensitivity. Before and after calibration.
- Paired bootstrap 95% intervals for "trained minus untrained" (resample by group when rows are related, by row otherwise).
- Three columns: untrained, trained, **real Jev** on the same test items. The Jev column runs only if API access works; the report records the actual bill. Without access it shows "not run" and nothing else is blocked. (We have ≈ $3–4 of credit, estimated enough for a few thousand decisions.)
- An external set the model never trained on (e.g. JevBench public + VitaminC), shown next to the main result.

Done when: one command regenerates the full report from saved artifacts. The test split is scored once, here.

### A4. Model bundle + local API

- A bundle = base model + revision, LoRA weights, prompt hash, temperature, eval summary, file hashes.
- `POST /v1/decisions`: state, question, options → chosen option + all probabilities.

Done when: the API gives the same numbers as offline scoring.

### A5. UI

Screens: **Data** (import, splits, rejected rows) → **Train** (settings, live loss, stop/resume) → **Evaluate** (untrained vs trained vs Jev, reliability chart) → **Playground** (type a state + options, see probabilities).

The UI only calls the same code the CLI uses; no hidden logic.

Stack: one polished page, FastAPI + plain HTML/JS, served by Bandits (`decide` extra).

Done when: the whole cookbook can be done from the UI without editing files.

### A6. Cookbook + launch

README section, one reproducible demo, a results chart, a screen recording, one-command start, a hardware note and an honest limitations list.

Claims allowed only if measured: "beats the untrained model on X", and "beats Jev on X" only if A3 showed it. No general "beats Jev" claim.

## 5. Track B: Bandits judge path (after launch)

Train from agent traces instead of user labels. Needs extra work because judge labels are only the judge's opinion:

1. Inventory: how many judged turns, how many votes, label balance, lengths.
2. Correct labels: map TRAIL human error annotations to turns + a small hand-labeled set (two labelers, disagreements resolved).
3. Targeted 3-vote judge runs, then soft-label SFT.
4. One shared "evidence view" so judge and student see the exact same (truncated) state.
5. Baselines: existing checks (by coverage and precision), the judge, untrained 4B/9B.
6. Optional: decision model as a cheap first-pass judge, falling back to the LLM judge when unsure.

Track B results are never used to support Track A's launch claims.

## 6. Rules for honest results

- Test rows are never used for training, prompt tuning, early stopping, dataset choice, go/no-go or temperature fitting.
- Splits are fixed before training; related rows stay in the same split.
- Public benchmarks are final checks only, never tuned against.
- Rejected inputs are reported, never silently dropped.
- Untrained softmax output is never called "calibrated".

## 7. Explicitly out of scope

RL of any kind, synthetic data engine, Noul/Score/structured question types, more than 26 options, pointer/slot heads, shared-prefix serving tricks, 27B training, hosted/multi-user infra.

## 8. Branching

- Done: the branch was renamed `feat/decide-dataset` → `feat/jev`. GitHub closed PR #63 automatically on the rename, so it was reopened as **PR #64** (same commit, base `feat/awm-env`). Not merged.
- `feat/awm-env` (PR #45) has since been merged into `main`, so `feat/jev` is effectively `main` + the one dataset commit. PR #64's base stays `feat/awm-env` (same content as `main`).
- All Jev work lives on `feat/jev`. Each issue gets a branch off `feat/jev` and a PR back into it. Nothing goes to `main` until we decide.

## 9. Issue map

**Filed 2026-09-23 as 8 feature issues on bandr-ai/bandits:** tracking #73 · #65 dataset + import · #66 prompt + frozen scorer · #67 demo dataset + success bar · #68 LoRA SFT + go/no-go · #69 calibration + report · #70 bundle + API · #71 UI · #72 cookbook + launch. The finer list below is folded into those.

Track A:

1. Demo dataset shortlist: label-source + license audit, success bar (A0)
2. Generalize `DecisionDataset`: per-row question/options, four splits, optional group id, generic rejections; judge compiler kept (A1)
3. JSONL import + JevBench importer with source/license kept (A1)
4. Versioned prompt builder + tokenizer checks (A1)
5. Untrained scorer, one-order and two-order modes, saved scorer runs; dev runs on shortlist → pick dataset (A1)
6. LoRA SFT trainer + go/no-go on dev (A2)
7. Temperature calibration (A3)
8. Evaluation report: metrics, paired bootstrap intervals, external set, optional Jev column, one locked test run (A3)
9. Model bundle + local `/v1/decisions` API (A4)
10. UI: Data / Train / Evaluate / Playground (A5)
11. Cookbook, README, demo recording (A6)
12. Colab/Modal launch button (A6, optional)

Track B (created later): inventory, gold turn labels, multi-vote runs + soft SFT, evidence view, baseline panel, judge backend/cascade, judge cookbook.

Order: 1 + 2 + 4 in parallel → 3 → 5 → 6 → 7 + 8 → 9 → 10 → 11 → 12.

## 10. Open questions

- **Q1. Demo dataset:** which of the shortlist? (Decided after dev-split scoring in A1.)
- **Q2. Base model:** Qwen3.5-4B (fits a 24 GB GPU for LoRA, matches the #2 JevBench system) or 9B (more accurate, needs ~40 GB)? Depends on the GPU we have.
- **Q3. GPU:** what's available locally?

## 11. What changed from V2

| V2 | V3 | Why |
| --- | --- | --- |
| First use case: copy the Bandits judge | First use case: your own labeled data | Judge labels have no ground truth yet; weaker launch story |
| Hard + soft SFT recipes | Hard-label SFT only in Track A | No soft labels in the demo data |
| UI near the end, after cascade | UI right after the SFT spike succeeds | UI is the product for the launch |
| No external or Jev comparison in the core path | External set + Jev column in the report | Needed for honest launch claims |
| No confidence intervals | Paired bootstrap intervals | Small test sets are noisy |
| No demo-dataset step | A0 shortlists; dataset picked on dev after the scorer exists | Avoid assuming a win we haven't measured |
| Dataset format only fit/held-out, one shared question | Per-row questions/options, train/dev/calibration/test | Needed for user datasets; protects the locked test |
| Jev comparison assumed | Jev column optional, shows "not run" without access | Launch can't depend on the API |
| Judge integration in core milestones | Moved to Track B | Needs gold labels first |
