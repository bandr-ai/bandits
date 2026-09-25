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
uv run jev run <judge-run-id> --model Qwen/Qwen3.5-4B --revision <sha> --seed 1 \
  --checkpoint-dir runs/ckpt --output runs/report \
  --ledger <judge ledger.jsonl> --input-usd-per-mtok <price> --output-usd-per-mtok <price> \
  --gpu-usd-per-hour <price>
```

This takes a Bandits turn-judge run (or a dataset id from `jev import`) through the whole chain: dataset, untrained score, training, calibration, trained score and the report. Every step is saved as it finishes, so rerunning it reuses finished steps. If training is interrupted, `--resume-from-step N` continues it.

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
| `jev verifier-cost <dataset> --ledger ...` | Price the verifier per decision from its ledger (prices are given, never guessed) |
| `jev report ...` | The scorecard: verifier, majority baseline, untrained, trained, calibrated and Jev, with paired intervals, cost and latency |

Run `uv run jev <command> --help` for options.

## Docs

- **[Launch plan](docs/launch-plan.md): the one plan for testing and launch**
- **[Test plan and handoff](docs/run-plan.md): what each run is for, how to start it, where things stand**
- [Original plan](docs/decision-models-plan.md) and [research notes](docs/decision-models-learnings.md)
- [Demo dataset and success bar](docs/decision-models-demo-dataset.md)
