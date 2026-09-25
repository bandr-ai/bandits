# Jev recipe: launch plan

2026-09-25 · **The one plan for testing and launch.** It replaces the launch sections of [decision-models-plan.md](decision-models-plan.md) (the CaseHOLD-first "Track A"); that file stays as background. This plan changes only through the rules in §8. Tracking issue: #73.

## 1. What we launch

**Train your own Jev from your agent's traces.** Bandits' verifier labels every step your agent took, a small open model (Qwen3.5-4B + LoRA) learns to give the same answers, and you get a model you run yourself. It is fast and cheap enough to check every step of every live conversation, and it comes with an honest scorecard.

For: people building AI agents who already grade them with an LLM judge and can't afford to run it on every live step.

Second entry point, same product: bring your own labeled file (`jev import`) instead of traces.

## 2. The claim

The launch post says only what §5 measured. Its template:

> "Trained on **N** of your agent's steps, a 4B model agrees with your verifier **A%** of the time (the verifier agrees with itself **S%**), at **C× lower cost** and **L× lower latency**, well calibrated (ECE **E**). On human-labeled errors (TRAIL) it scores within **D** points of the verifier."

Every bold number comes from the locked test run (§6, phase 2). A number whose bar failed is dropped from the post, not reworded.

## 3. What exists (all in `recipes/jev/`)

| Piece | Command | Status |
| --- | --- | --- |
| Traces → dataset, split by trace | `jev dataset`, `jev merge` | PR #100, #104 |
| Untrained scoring, training, calibration | `jev score`, `jev train`, `jev calibrate` | #89 merged; PR #99 |
| Report: verifier, majority, untrained, trained, calibrated, Jev; paired intervals; cost and latency | `jev report`, `jev verifier-cost` | PR #99, #101, #102 |
| One command, resumable | `jev run` | PR #103 |
| GPU script + real-model smoke test | `scripts/gpu_run.sh`, `scripts/smoke.py` | PR #104 |

Rehearsed end to end on the real data with a tiny model on CPU. Nothing has trained on a real model at scale yet.

## 4. The benchmark

### 4.1 Data

- **Sources:** Bandits turn-judge runs over TRAIL GAIA, TRAIL SWE and tau2 (1,863 judged steps today), plus #98: the same judge with **3 votes**, over more traces, aiming for **≥ 1,000 test steps** (±~3-point intervals; today's 172 give ±~7).
- **Labels:** the verifier's vote shares, trained as soft targets (a 2-of-3 vote trains toward 2/3, not toward the winner). Rows whose votes tie are left out of accuracy and counted; NLL and Brier still use them.
- **Splits:** by trace, ~70/10/10/10 train/dev/calibration/test, fixed at compile time (#91). Every step of a conversation stays in one split. Intervals resample whole traces.
- **Unseen source:** a second run holds out **TRAIL SWE** entirely as test-only (#108).
- **Human labels:** TRAIL's error annotations, mapped onto the same steps (#107).

### 4.2 Systems compared (one column each, same test steps)

| System | What it is | Built |
| --- | --- | --- |
| **Verifier** | Bandits' judge (Fireworks `nemotron-lightning-3p5-30b-a3b`), the labels' source | yes |
| **Verifier self-agreement** | the ceiling: one vote vs the other two (#106) | no |
| **Majority** | always the train split's most common answer | yes |
| **Untrained** | Qwen3.5-4B, no training (one and two option orders) | yes |
| **Trained**, **trained + calibrated** | Qwen3.5-4B + LoRA, temperature from the calibration split | yes |
| **Cheaper LLM judge** | same judge prompt on a cheaper model (#105). Rule: the cheapest Fireworks model whose judge output parses on ≥ 95% of 50 dev steps; the model id is recorded | no |
| **Real Jev** | TypeSafe's API on the same steps (#97); "not run" without a key | client no |

### 4.3 Measurements

1. **Agreement with the verifier** (accuracy vs its majority vote; tied votes left out and counted), with paired intervals against every baseline.
2. **Calibration:** ECE, NLL and Brier, before and after temperature.
3. **Cost per 1k decisions and latency** (p50/p95): the verifier's and the cheaper judge's from their ledgers, ours from measured latency × the GPU's hourly price, Jev from its bill.
4. **Against humans:** F1 of "failure" vs TRAIL's human error marks, for the verifier, the student and the cheaper judge (#107).
5. **Unseen source:** agreement on held-out TRAIL SWE (#108).
6. **Learning curve:** dev agreement after 100 / 300 / 1,000 / all labeled steps (#109).
7. **Launch chart:** cost (log scale) vs agreement, with self-agreement as the ceiling line (#110).

## 5. Success bars (fixed now, before any real run)

| # | Claim | Pass if (on the locked test split) |
| --- | --- | --- |
| B1 | It learned something | trained + calibrated − majority: agreement interval lower bound > 0 |
| B2 | Close to the verifier | trained + calibrated agreement ≥ 0.9 × verifier self-agreement |
| B3 | Beats the cheap alternative | trained + calibrated − cheaper judge: agreement interval lower bound ≥ 0 |
| B4 | Same quality against humans | student F1 − verifier F1 on TRAIL humans: interval lower bound ≥ −3 points |
| B5 | Calibrated | ECE ≤ 0.05 after temperature |
| B6 | Cheap and fast | ≥ 50× lower cost per 1k and ≥ 20× lower p50 latency than the verifier |

The unseen-source result and the learning curve have no bar; they are reported as measured. "Beats Jev" is claimed only if trained + calibrated − Jev has an agreement interval lower bound > 0.

## 6. Runs

**Phase 0: no GPU (Laxman; judge API spend)**
- Commit this plan (done in this PR).
- #98 more 3-vote judge runs · #105 cheaper judge · #106 self-agreement · #107 TRAIL humans · #108 held-out source · #109 learning curve · #110 chart.
- Prices recorded: the verifier's and the cheaper judge's per-token prices, and the GPU's hourly price.

**Phase 1: GPU session 1, dev only (Alex; ~1–2 h on one 48 GB+ GPU)**
- `gpu_run.sh` smoke test for `Qwen/Qwen3.5-4B-Base` and `Qwen/Qwen3.5-4B`.
- One seed each, scored on **dev**. **Pick the model by dev agreement** (higher wins; a tie goes to Base). Nothing is scored on test.
- Learning curve on dev.

**Phase 2: GPU session 2, the locked test run, once (Alex; ~2–3 h)**
- The chosen model × **3 seeds** (1, 2, 3), each through `jev run`, plus the held-out-SWE run.
- Every bar in §5 checked from the reports. The test split is not scored again for this dataset.

**Phase 3: write-up (Laxman)**
- The chart, one table (agreement, ECE, cost/1k, p50 latency, F1 vs humans; one row per system; mean ± range over seeds), the learning curve, the report files and the exact command to reproduce.

**Budget:** about 3–5 GPU-hours in total, plus judge API spend of about 3k tokens per step per vote (measured from the ledgers).

## 7. Launch deliverables

1. The results above, published with the reports.
2. Model bundle + local `/v1/decisions` API (#70).
3. One-page UI: traces or file → verifier labels → Train (hosted on Modal, #96) → scorecard → playground (#71).
4. README, cookbook, demo recording, launch thread (#72).

Build order: 1 needs phase 2; 2 and 3 are built in parallel with phases 0–2; 4 comes last.

## 8. Rules

- **This plan changes only when a bar fails**, and then only by its pre-written fallback:
  - B1 or B2 fails at 4B → one extra dev run at `Qwen/Qwen3.5-9B-Base`. If that passes on dev, phase 2 repeats at 9B on a **freshly cut** test split. Otherwise launch without that claim.
  - B3 fails → no "better than a cheaper judge" claim; the post leads on running locally and on latency only if B6 passes.
  - B4 fails → no quality claim against humans; say "copies your verifier", not "as good as".
  - B5 fails → report calibration as measured; no "calibrated" claim.
  - Fewer than 1,000 test steps → run anyway and print the wider intervals.
- **The test split is scored once** (phase 2). If anything is changed after seeing test results, that test becomes dev and a new test split is cut before any claim.
- **Claims only from measured numbers.** Unknown costs read "unknown", never 0.
- Anything not in this document is out of scope for launch (§9). New ideas go into issues for after launch.

## 9. Out of scope for launch

Improving the verifier itself (it improves separately; users retrain), RL, question types other than choice, more than 26 options, 27B+ models or full fine-tuning, multi-user hosting, the CaseHOLD demo as a headline (it stays as the "bring your own file" example), JevBench as a claim (our model is task-specific).

## 10. Inputs we need

| Input | From | For |
| --- | --- | --- |
| Verifier and cheaper-judge prices per million tokens | Fireworks billing | B6, cost columns |
| A rented GPU (48 GB+) and its hourly price | Alex | phases 1–2, B6 |
| `work/trail-ns`, `work/tau2-ns` on the GPU box | Laxman | all runs |
| Jev API key and docs | whoever has it | Jev column (optional) |
