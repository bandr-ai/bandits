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

## Commands

| Command | What it does |
| --- | --- |
| `jev dataset <judge-run>` | Turn a Bandits turn-judge run into a decision dataset (the verifier's vote shares become the labels) |
| `jev import <file.jsonl>` | Import your own labeled decisions |
| `jev score <dataset> --split dev` | Score a split with a frozen (untrained) model |
| `jev train <dataset>` | LoRA fine-tune on train; the best checkpoint is picked on dev |

Run `uv run jev <command> --help` for options.

## Docs

- [Plan](docs/decision-models-plan.md) and [research notes](docs/decision-models-learnings.md)
