#!/usr/bin/env python3
"""Real-model smoke test: does a pinned checkpoint work with this recipe
before a long run is spent on it?

Scores up to --rows dev rows and takes a few training steps on a handful of
train rows, then prints one JSON summary. Fails (exit 1) when:

- any row is rejected for a tokenizer reason (an option letter is not one
  distinct token in the answer context);
- a training loss is not finite, or the loss does not fall on the same batch.

Warns (exit 0) when the untrained model puts little of its probability on
the answer letters at all (`letter_mass`), or picks one option for nearly
every row: the untrained baseline is then weak, and "trained beats
untrained" says less than it seems.

    uv run python scripts/smoke.py --project <dir> --dataset <id> \
        --model Qwen/Qwen3.5-4B-Base --revision <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

LOW_LETTER_MASS = 0.2
COLLAPSED_SHARE = 0.9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    from bandits.store import DerivedStore
    from bandits_jev.dataset import load_decision_dataset
    from bandits_jev.hf_predictor import HFPredictor
    from bandits_jev.hf_trainer import HFTrainer
    from bandits_jev.metrics import gold_option
    from bandits_jev.scorer import score_dataset
    from bandits_jev.trainer import build_training_config, reject_unscorable

    dataset = load_decision_dataset(args.dataset, DerivedStore(args.project / ".bandits"))
    # A spread of dev rows (ordered by id hash, not file order), so a merged
    # dataset's smoke sample covers every source, not only the first one.
    dev = sorted(
        (e for e in dataset.examples if e.split == "dev"),
        key=lambda e: hashlib.sha256(e.decision_id.encode()).hexdigest(),
    )[: args.rows]
    train_candidates = [e for e in dataset.examples if e.split == "train"]
    if not dev or not train_candidates:
        print(f"dataset {args.dataset} needs dev and train rows", file=sys.stderr)
        return 1

    predictor = HFPredictor(args.model, revision=args.revision, device=args.device, dtype=args.dtype)
    run = score_dataset(predictor, dev)
    del predictor
    by_id = {e.decision_id: e for e in dev}
    tokenizer_rejections = [
        r.reasons for r in run.rejections if not any("over the" in reason for reason in r.reasons)
    ]
    chosen = Counter(r.chosen_option_id for r in run.results)
    agree = [
        r.chosen_option_id == gold
        for r in run.results
        if (gold := gold_option(by_id[r.decision_id].target.probabilities)) is not None
    ]
    letter_mass = sum(r.candidate_token_mass for r in run.results) / len(run.results) if run.results else None

    config = build_training_config(
        base_model_id=args.model,
        base_revision=args.revision,
        dataset_id=args.dataset,
        seed=1,
        eval_every_steps=0,
        effective_batch=4,
        dtype=args.dtype,
        device=args.device,
    )
    trainer = HFTrainer.from_config(config)
    # The same filter real training applies: rows over the token limit or the
    # model's own context, or failing the letter check, are never trained on.
    kept, _ = reject_unscorable(
        trainer, [(e, dict(e.options)) for e in train_candidates], max_prompt_tokens=config.max_prompt_tokens
    )
    train_rows = [e for e, _ in kept][:4]
    if not train_rows:
        print("no train row fits this model; nothing to take a training step on", file=sys.stderr)
        return 1
    trainer.configure_schedule(total_steps=args.train_steps, warmup_steps=0)
    losses = [trainer.train_step(train_rows) for _ in range(args.train_steps)]

    failures = []
    if not run.results:
        failures.append(f"no dev row could be scored ({len(run.rejections)} rejected)")
    if tokenizer_rejections:
        failures.append(f"{len(tokenizer_rejections)} row(s) rejected by the tokenizer check: {tokenizer_rejections[:3]}")
    if not all(math.isfinite(loss) for loss in losses):
        failures.append(f"non-finite training loss: {losses}")
    elif losses[-1] >= losses[0]:
        failures.append(f"loss did not fall on the same batch: {losses}")
    warnings = []
    if letter_mass is not None and letter_mass < LOW_LETTER_MASS:
        warnings.append(
            f"untrained model puts only {letter_mass:.1%} of its probability on the answer letters; "
            "its baseline is weak"
        )
    if run.results and chosen.most_common(1)[0][1] / len(run.results) >= COLLAPSED_SHARE:
        warnings.append(f"untrained model picks {chosen.most_common(1)[0][0]!r} for nearly every row")

    print(
        json.dumps(
            {
                "model": f"{args.model}@{args.revision}",
                "dev_rows_scored": len(run.results),
                "rejected": len(run.rejections),
                "letter_mass": round(letter_mass, 4) if letter_mass is not None else None,
                "untrained_agreement_with_verifier": round(sum(agree) / len(agree), 4) if agree else None,
                "untrained_choices": dict(chosen),
                "train_losses": [round(loss, 4) for loss in losses],
                "failures": failures,
                "warnings": warnings,
                "ok": not failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
