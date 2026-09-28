# Your own Jev: locked test results

2026-09-28 · Qwen3.5-4B-Base + LoRA · one seed. What the launch may claim, the numbers behind it, and what it may not claim.

## Claims

**1. Trained on labeled steps, your own Jev beats TypeSafe's Jev on AgentProcessBench.**

> On 1,428 held-out AgentProcessBench steps (30 tasks), a 4B model trained on the benchmark's other tasks agrees with the human labels **79.7%** of the time. TypeSafe's Jev (`jev-1.13.0`) agrees **66.3%** of the time on the same steps with the same input: **+13.4 points** (95% CI +8.0 to +19.1).

The run's test split had 1,920 steps, but 492 of them belong to one τ²-bench task that also appears in train and dev (twelve query indices, 34 and 39–49, share its text, and the split went by index). The claim uses the 1,428 steps without it. On all 1,920 the gap is +12.3 [+7.3, +17.3]; on the leaked task alone, 77.4% vs 67.9%, so the leak did not flatter the model. `scripts/agentprocessbench.py` now groups by task text.

This supports "beats Jev on AgentProcessBench". It does not support a general "beats Jev": see the in-distribution caveat below.

**2. Trained on your verifier's labels, your own Jev agrees with the verifier as often as the verifier agrees with itself.**

> Trained on 1,097 steps labeled by Bandits' verifier (Nemotron-Lightning-3.5-30B-A3B), a 4B model agrees with the verifier on **77.9%** of 172 unseen steps. The verifier, re-run on the same steps, agrees with its own labels **75.6%** of the time. It answers **36× faster** (0.12 s vs 4.2 s median), with calibration error **0.047**.

Real Jev has not been run on this split yet (#97, 172 steps).

**3. Your own Jev beats DeepSeek-V4.1-Flash, a 763B-parameter model, at judging agent steps.**

> Given the same input for each step: on your verifier's labels (172 test steps) it agrees 77.9% vs DeepSeek-V4.1-Flash's 69.2%, **+8.7** (95% CI +3.6 to +14.1). On human labels (AgentProcessBench, clean 493-step sample, 30 tasks) 79.7% vs 65.3%, **+14.4** (CI +7.5 to +21.6). About 90× faster per step (0.13 s vs 12.3 s).

DeepSeek-V4.1-Flash is a mixture-of-experts model: 763B parameters in total (552B backbone, 196B Engram memory), 8–16B active per token ([model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)).

## 1. AgentProcessBench (human labels)

[AgentProcessBench](https://arxiv.org/abs/2603.14465) labels every step of 1,000 agent trajectories as +1 (advances the task), 0 (little effect) or −1 (wrong), with 89.1% agreement between annotators. We split it by task index: 5,380 train steps, 556 dev, 653 calibration, 1,920 test. One τ²-bench task spans twelve indices and so all four splits (see Claim 1); the numbers below are on the full test split unless marked **clean** (1,428 steps, 30 tasks, that task removed).

| System | Accuracy | Macro F1 | ECE | p50 latency |
| --- | ---: | ---: | ---: | ---: |
| Majority label | 60.8% | 25.2% | 0.013 | — |
| Qwen3.5-4B-Base, untrained | 60.7% | 35.9% | 0.106 | 0.092 s |
| **Your own Jev (Qwen3.5-4B-Base + LoRA)** | **79.1%** | 52.6% | **0.012** | 0.135 s |
| TypeSafe Jev 1.13.0 (API) | 66.8% | 52.7% | 0.073 | 0.449 s |

Paired differences, 95% bootstrap intervals over the 36 test tasks (10,000 draws):

| Difference | Accuracy |
| --- | ---: |
| Your own Jev − TypeSafe Jev | +12.3 [+7.3, +17.3] |
| Your own Jev − TypeSafe Jev, **clean** | **+13.4 [+8.0, +19.1]** |
| Your own Jev − DeepSeek-V4.1-Flash, **clean** (493-step sample) | +14.4 [+7.5, +21.6] |
| Your own Jev − untrained | +18.4 [+11.5, +25.3] |

By source:

| Source | Steps | Your own Jev | TypeSafe Jev |
| --- | ---: | ---: | ---: |
| GAIA | 333 | 77.5% | 57.7% |
| τ²-bench | 958 | 78.8% | 67.3% |
| τ²-bench, clean | 466 | 80.3% | 66.5% |
| BFCL | 571 | 81.3% | 72.9% |
| HotpotQA | 58 | 72.4% | 51.7% |

## 2. Verifier labels (Bandits traces)

1,863 steps from TRAIL GAIA, TRAIL SWE-bench and τ²-bench, labeled by the verifier; split by trace into 1,097 train / 271 dev / 323 calibration / 172 test (25 traces). The launch plan's bars ([launch-plan.md](launch-plan.md) §5), on the test split:

| Bar | Result | Status |
| --- | --- | --- |
| B1 beats the majority answer | +33.1 [+21.8, +46.1] | pass |
| B2 ≥ 0.9 × verifier self-agreement (68.0%) | 77.9% | pass |
| B3 beats a cheaper LLM judge | +7.0 [−0.8, +14.3] vs GLM-5.3-Flash | fail (lower bound below 0) |
| B4 within 3 points of the verifier on TRAIL human labels | not run (#107) | — |
| B5 ECE ≤ 0.05 | 0.047 | pass |
| B6 ≥ 50× cheaper and ≥ 20× faster than the verifier | 5.3× cheaper, 36× faster | fail (cost) |

- Self-agreement came from re-running the verifier once on the test steps (temperature 0), not from #106's three-vote runs. Your own Jev − verifier self-agreement: +2.3 [−3.1, +7.8].
- B3's cheap judge is picked by rule: the cheapest Fireworks model whose judge output parses on at least 95% of dev steps. The cheapest is the verifier itself (Nemotron, $0.05 / $0.20 per million tokens), so the rule takes the next: GLM-5.3-Flash ($0.15 / $0.50), which scored 271 of 271 dev steps. On test it agrees with the verifier 70.9% [64.4, 77.1]; ours 77.9%. The interval's lower bound is −0.8, so B3 fails and the post makes no claim against a cheaper judge. Measured, GLM-5.3-Flash was not cheaper than the verifier: its long reasoning cost $1.22 per 1,000 decisions, at 12 s per step. An earlier comparison against DeepSeek-V4.1-Flash (+8.7 [+3.6, +14.1]) is not a substitute, because the rule did not pick it.
- B6 failed on cost, so the post drops the cost claim (§2: "a number whose bar failed is dropped").

## Speed and cost

| Per 1,000 decisions | Cost | p50 latency |
| --- | ---: | ---: |
| Your own Jev, one L40S, one step at a time | $0.077 | 0.135 s |
| TypeSafe Jev, list price ($0.042 per million input tokens, output free) | ~$0.058 | 0.449 s |
| Verifier (Nemotron, Fireworks) | $0.46 | 4.2 s |

TypeSafe Jev is cheaper than our current setup, so the launch makes no cost claim against it. Its figure comes from the published price and the tokens the API reported (2,647,082 input on this run), not from a bill. Latency is not like for like: Jev's includes the network from this machine, ours is measured on the GPU.

## Not claimed: TRAIL

The AgentProcessBench-trained model and TypeSafe Jev were both scored on 1,174 TRAIL steps (139 traces) against TRAIL's human error marks (834 of 836 marks map onto a step).

- Neither is useful on its own there: flagging every step as an error gets F1 0.558, above both (TypeSafe Jev 0.451, ours 0.444).
- Ranking steps by P(failure): AUROC 0.604 (ours) vs 0.572 (Jev), difference +0.032 [+0.008, +0.058]. On SWE-bench both are at chance (0.540 vs 0.536).
- Our GAIA result is contaminated: 37 of AgentProcessBench's 50 GAIA tasks are TRAIL questions, and 26 of those were in our training split.

## Caveats

- **In-distribution advantage.** Our model trained on the benchmark's other tasks; TypeSafe Jev saw the benchmark cold. The clean test tasks were never trained on, but the advantage remains.
- **Input format.** Both systems got the same input, built for our model: the task, the agent's instructions (clipped to 1,500 characters), the last eight turns, the step and what came back, with our question and option wording. Jev may do better with inputs written for it.
- **Unclear.** Our model never answers "unclear" (5% of the human labels); TypeSafe Jev answers it about 230 times. That is why the two have the same macro F1 despite the accuracy gap.
- **Small verifier-label test.** 172 steps from 25 traces, with 2 traces each for coding and support agents.
- **One seed** for each model.
- **Test splits are spent.** Any model, prompt or threshold changed after these results needs a fresh test split.

## What ran

- Model: `Qwen/Qwen3.5-4B-Base` at `1001bb4d826a52d1f399e183466143f4da7b741b`, LoRA rank 16 on all linear layers.
  - AgentProcessBench adapter `794b83263275b9fd` (step 250), temperature 1.0475.
  - Verifier-label adapter (step 138), temperature 1.0774.
- TypeSafe Jev: `jev-latest`, resolved to `jev-1.13.0`; 1,920/1,920 answers, 2,647,082 input and 75,109 output tokens.
- Reports and raw predictions are local (gitignored): `work/apb-jev-eval/report-with-jev/`, `work/apb-jev-eval/jev-test.jsonl`, and the run tarballs under `runs/`.

## Reproduce the Jev column

```bash
export JEV_API_KEY=...
uv run jev score-api decision-dataset-a5d18c1c963246f9 \
  --split test --allow-test --model jev-latest --workers 16 \
  --output work/apb-jev-eval/jev-test.jsonl \
  --project work/apb-jev-eval/jev-project-import-2fbd164382b9
```

Each successful answer is appended as it arrives, so rerunning the command resumes after an interruption. Import the file with `jev import-predictions`, then pass the scorer-run id to `jev report --jev`.
