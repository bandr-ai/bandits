# Your own Jev: locked test results

2026-09-28 · Qwen3.5-4B-Base + LoRA · one seed. What the launch may claim, the numbers behind it, and what it may not claim.

## Claim

**Trained on labeled steps, your own Jev beats TypeSafe's Jev on AgentProcessBench.**

> On 1,920 held-out AgentProcessBench steps (36 tasks), a 4B model trained on the benchmark's other tasks agrees with the human labels **79.1%** of the time. TypeSafe's Jev (`jev-1.13.0`) agrees **66.8%** of the time on the same steps with the same input: **+12.3 points** (95% CI +7.3 to +17.3).

This supports "beats Jev on AgentProcessBench". It does not support a general "beats Jev": see the in-distribution caveat below.

## AgentProcessBench (human labels)

[AgentProcessBench](https://arxiv.org/abs/2603.14465) labels every step of 1,000 agent trajectories as +1 (advances the task), 0 (little effect) or −1 (wrong), with 89.1% agreement between annotators. We split it by task, so a task's five trajectories share a split: 5,380 train steps (135 tasks), 556 dev (13), 653 calibration (16), 1,920 test (36 tasks, 180 trajectories).

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
| Your own Jev − untrained | +18.4 [+11.5, +25.3] |

By source:

| Source | Steps | Your own Jev | TypeSafe Jev |
| --- | ---: | ---: | ---: |
| GAIA | 333 | 77.5% | 57.7% |
| τ²-bench | 958 | 78.8% | 67.3% |
| BFCL | 571 | 81.3% | 72.9% |
| HotpotQA | 58 | 72.4% | 51.7% |

## Speed and cost

On the same 1,920 test steps:

| Per 1,000 decisions | Cost | p50 latency |
| --- | ---: | ---: |
| Your own Jev, one L40S, one step at a time | $0.077 | 0.135 s |
| TypeSafe Jev, list price ($0.042 per million input tokens, output free) | ~$0.058 | 0.449 s |

TypeSafe Jev is cheaper than our current setup, so the launch makes no cost claim against it. Its figure comes from the published price and the tokens the API reported (2,647,082 input on this run), not from a bill. Latency is not like for like: Jev's includes the network from this machine, ours is measured on the GPU.

## Not claimed: training on verifier labels

A second model was trained on 1,097 steps from TRAIL GAIA, TRAIL SWE-bench and τ²-bench labeled by Bandits' verifier (Fireworks Nemotron-Lightning-3.5-30B-A3B), and tested on 172 steps from 25 traces. It is not claimed:

- The verifier was unstable. At temperature 0 it looped until its token budget ran out on up to 10 of 25 identical prompts, so the re-run meant to measure its self-agreement needed retries with changed sampling settings on 51 of the 172 test steps. That is not a clean self-agreement baseline.
- The planned cheaper-judge comparison (DeepSeek) was not run under the plan's rule.
- The test split is small, with 2 traces each for coding and support agents.

Bandits' default judge is now DeepSeek V4.1 Flash. A verifier-label result needs a new run with it.

## Not claimed: TRAIL

The AgentProcessBench-trained model and TypeSafe Jev were both scored on 1,174 TRAIL steps (139 traces) against TRAIL's human error marks (834 of 836 marks map onto a step).

- Neither is useful on its own there: flagging every step as an error gets F1 0.558, above both (TypeSafe Jev 0.451, ours 0.444).
- Ranking steps by P(failure): AUROC 0.604 (ours) vs 0.572 (Jev), difference +0.032 [+0.008, +0.058]. On SWE-bench both are at chance (0.540 vs 0.536).
- Our GAIA result is contaminated: 37 of AgentProcessBench's 50 GAIA tasks are TRAIL questions, and 26 of those were in our training split.

## Caveats

- **In-distribution advantage.** Our model trained on 135 AgentProcessBench tasks; TypeSafe Jev saw the benchmark cold. Held-out tasks prevent leakage, but not the advantage.
- **Input format.** Both systems got the same input, built for our model: the task, the agent's instructions (clipped to 1,500 characters), the last eight turns, the step and what came back, with our question and option wording. Jev may do better with inputs written for it.
- **Unclear.** Our model never answers "unclear" (5% of the human labels); TypeSafe Jev answers it about 230 times. That is why the two have the same macro F1 despite the accuracy gap.
- **One seed.**
- **The test split is spent.** Any model, prompt or threshold changed after these results needs a fresh test split.

## What ran

- Model: `Qwen/Qwen3.5-4B-Base` at `1001bb4d826a52d1f399e183466143f4da7b741b`, LoRA rank 16 on all linear layers; adapter `794b83263275b9fd` (step 250), temperature 1.0475.
- Data: `scripts/agentprocessbench.py` builds the import file from AgentProcessBench's Hugging Face release.
- TypeSafe Jev: `jev-latest`, resolved to `jev-1.13.0`; 1,920/1,920 answers, 2,647,082 input and 75,109 output tokens.
- Reports and raw predictions are local (gitignored): `work/apb-jev-eval/report-with-jev/`, `work/apb-jev-eval/jev-test.jsonl`, and the run tarballs under `runs/`.

## Reproduce

```bash
cd recipes/jev
uv run python scripts/agentprocessbench.py --out apb.jsonl
uv run jev import apb.jsonl --project work/apb          # prints the dataset id
uv run jev run <dataset-id> --model Qwen/Qwen3.5-4B-Base --revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --seed 1 --two-order --eval-split test --allow-test \
  --checkpoint-dir runs/apb-ckpt --output runs/apb-report --project work/apb

export JEV_API_KEY=...
uv run jev score-api <dataset-id> --split test --allow-test --model jev-latest --workers 16 \
  --output work/apb/jev-test.jsonl --project work/apb
```

`jev score-api` appends each answer as it arrives, so rerunning it resumes after an interruption. Import its file with `jev import-predictions`, then pass the scorer-run id to `jev report --jev`.
