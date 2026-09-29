# Jev recipe: train your own decision model on Bandits

A recipe on top of Bandits. It turns your agent's traces, labeled by Bandits' verifiers or by you, into a small Jev-style decision model: one forward pass, a probability for every option, and no text generation. It then produces an honest scorecard for that model.

## How it sits next to Bandits

- Package `bandits_jev`, command `jev`, and its own environment and dependencies (torch, transformers and peft live in the `train` extra).
- It **reads** Bandits core (traces, turn-judge runs, task sets, the `.bandits` artifact store) and writes its own artifacts beside them.
- **Core never imports this recipe.** Deleting `recipes/jev/` leaves Bandits exactly as it was.

## Install

```bash
cd recipes/jev
uv sync --extra dev              # data, calibration, report, tests
uv sync --extra dev --extra train  # plus scoring and training a real model (GPU)
```

## One command

```bash
uv run jev run <judge-run-id> --model Qwen/Qwen3.5-4B-Base --revision <sha> --seed 1 \
  --eval-split dev --checkpoint-dir runs/ckpt --output runs/report \
  --ledger <judge ledger.jsonl> --input-usd-per-mtok <price> --output-usd-per-mtok <price> \
  --gpu-usd-per-hour <price>
```

This takes a Bandits turn-judge run (or a dataset id from `jev import`) through the whole chain: dataset, untrained score, training, calibration, trained score and the report. Every step is saved as it finishes, so rerunning it reuses finished steps. If training is interrupted, `--resume-from-step N` continues it.

`--eval-split dev` reports on dev, for choosing a model and settings. Once those are fixed, `--eval-split test --allow-test` gives the final numbers; each such run is a recorded look at the locked test split. The 4B model trains on one 48 GB GPU (an L40S).

## Your own labeled data

`jev import <file.jsonl>` prints a dataset id to pass to `jev run` in place of a judge run. One JSON object per line:

```json
{"state": "Task: refund order 1182\nAgent called refund(order=1182)\nTool returned: error, order not found", "question": "Did this step advance the task?", "options": {"success": "Yes, it advanced the task.", "failure": "No, it was wrong or had no effect."}, "target": "failure", "group_id": "task-1182"}
```

| Field | |
| --- | --- |
| `state` | What the model sees: the situation to decide on |
| `question` | What to decide |
| `options` | 2 to 26 options, option id → description |
| `target` | The right option id, or a probability per option id (e.g. `{"success": 0.7, "failure": 0.3}`) |
| `group_id` | Optional. Rows that share one (a trace, a task) always land in the same split, so related rows never sit across train and test |
| `split` | Optional: `train`, `dev`, `calibration` or `test`. By default a hash of `group_id` (or the row's content) picks it |
| `id`, `source`, `license`, `label_source` | Optional, recorded with the row |

By default rows split about 70/10/10/10 into train, dev, calibration and test, and `jev run` needs rows in each, so give it at least a few dozen groups. Bad rows are set aside with their line number and the reason, not silently dropped; `--output rows.jsonl` writes the imported rows there and the set-aside ones to `rows.jsonl.quarantined.jsonl`. `scripts/agentprocessbench.py` is a full example: it turns AgentProcessBench's human labels into this format.

## Commands

| Command | What it does |
| --- | --- |
| `jev run <judge-run or dataset>` | Everything below, in order, resuming from finished steps |
| `jev dataset <judge-run>` | Turn a Bandits turn-judge run into a decision dataset (the verifier's vote shares become the labels) |
| `jev import <file.jsonl>` | Import your own labeled decisions |
| `jev score <dataset> --split dev` | Score a split with a frozen model, optionally with `--adapter` for a trained one |
| `jev train <dataset>` | LoRA fine-tune on train; the best checkpoint is picked on dev |
| `jev calibrate <scorer-run>` | Fit one temperature on the calibration split |
| `jev import-predictions <file>` | Bring in another system's answers (e.g. real Jev) with its bill |
| `jev score-api <dataset> --output jev.jsonl` | Call real Jev; resumable, reads `JEV_API_KEY` |
| `jev verifier-cost <dataset> --ledger ...` | Price the verifier per decision from its ledger (prices are given, never guessed) |
| `jev report ...` | The scorecard: verifier, majority baseline, untrained, trained, calibrated and Jev, with paired intervals, cost and latency |

Run `uv run jev <command> --help` for options.

## Docs

- **[Locked test results](docs/launch-results.md): measured launch claims and caveats**
- **[Launch plan](docs/launch-plan.md): the one plan for testing and launch**
- **[Test plan and handoff](docs/run-plan.md): what each run is for, how to start it, where things stand**
- [Original plan](docs/decision-models-plan.md) and [research notes](docs/decision-models-learnings.md)
- [Demo dataset and success bar](docs/decision-models-demo-dataset.md)
