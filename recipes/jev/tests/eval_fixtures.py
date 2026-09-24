"""Synthetic datasets and scorer runs for calibration and report tests.

Rows are generated from known "true" logits so tests can assert on things
that have a right answer: a model whose logits are the truth scaled by k is
overconfident by exactly k, so the fitted temperature must come out near k.
"""

from __future__ import annotations

import json
import math
import random

from bandits.store import DerivedStore
from bandits_jev.dataset import DecisionDataset
from bandits_jev.importer import import_jsonl, save_imported_dataset
from bandits_jev.prompt import PROMPT_VERSION, template_digest
from bandits_jev.scorer import (
    DecisionScoreResult,
    OptionScore,
    RejectedScore,
    ScorerRun,
    save_scorer_run,
)

OPTIONS = {"a": "first", "b": "second", "c": "third"}
BASE_MODEL = "fake/base"
BASE_REVISION = "rev1"
ADAPTER = "adapter-digest-1"


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    top = max(logits.values())
    exps = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def build_dataset(
    store: DerivedStore,
    *,
    per_split: dict[str, int],
    seed: int = 0,
    group_size: int = 1,
    tag: str = "main",
) -> tuple[str, DecisionDataset, dict[str, dict[str, float]]]:
    """Rows whose gold option is sampled from softmax(true logits). Returns
    the dataset id, the dataset, and each decision's true logits."""
    rng = random.Random(seed)
    lines = []
    truths_by_state: dict[str, dict[str, float]] = {}
    index = 0
    for split, count in per_split.items():
        for i in range(count):
            true_logits = {o: rng.gauss(0.0, 1.5) for o in OPTIONS}
            probs = _softmax(true_logits)
            gold = rng.choices(list(probs), weights=list(probs.values()))[0]
            state = f"{tag} {split} item {i}"
            truths_by_state[state] = true_logits
            row = {
                "id": f"{tag}-{index}",
                "state": state,
                "question": "which option?",
                "options": OPTIONS,
                "target": gold,
                "split": split,
            }
            if group_size > 1:
                row["group_id"] = f"{tag}-{split}-g{i // group_size}"
            lines.append(json.dumps(row))
            index += 1
    dataset = import_jsonl("\n".join(lines), source_file=f"{tag}.jsonl")
    assert not dataset.quarantined
    envelope = save_imported_dataset(dataset, store, source_file=f"{tag}.jsonl")
    truths = {e.decision_id: truths_by_state[e.state] for e in dataset.examples}
    return envelope.artifact_id, dataset, truths


def make_run(
    dataset_id: str,
    dataset: DecisionDataset,
    truths: dict[str, dict[str, float]],
    *,
    split: str,
    scale: float,
    adapter_digest: str | None = None,
    two_order: bool = False,
    order_bias: float = 0.0,
    reject: set[str] = frozenset(),
    noise_seed: int = 0,
    noise: float = 0.0,
) -> ScorerRun:
    """A run whose logits are ``scale`` × the truth (+ optional noise). In
    two-order mode, pass 1 favours the first option by ``order_bias`` and
    pass 2 favours the last one, mimicking a position bias."""
    rng = random.Random(noise_seed)
    results = []
    rejections = []
    for example in dataset.examples:
        if example.split != split:
            continue
        if example.decision_id in reject:
            rejections.append(
                RejectedScore(
                    decision_id=example.decision_id,
                    reasons=("prompt is 9001 tokens, over the 8000 limit",),
                )
            )
            continue
        logits = {o: scale * v + rng.gauss(0.0, noise) for o, v in truths[example.decision_id].items()}
        options = list(logits)
        pass1 = dict(logits)
        pass2 = None
        if two_order:
            pass1 = {o: v + (order_bias if o == options[0] else 0.0) for o, v in logits.items()}
            pass2 = {o: v + (order_bias if o == options[-1] else 0.0) for o, v in logits.items()}
            p1, p2 = _softmax(pass1), _softmax(pass2)
            probs = {o: (p1[o] + p2[o]) / 2 for o in options}
        else:
            probs = _softmax(pass1)
        scores = tuple(
            OptionScore(
                option_id=o,
                probability=probs[o],
                raw_logit_pass1=pass1[o],
                raw_logit_pass2=pass2[o] if pass2 is not None else None,
            )
            for o in options
        )
        results.append(
            DecisionScoreResult(
                decision_id=example.decision_id,
                scores=scores,
                chosen_option_id=max(scores, key=lambda s: s.probability).option_id,
                candidate_token_mass=0.9,
                mode="two_order_average" if two_order else "single_order",
                prompt_digest="p",
                latency_seconds=0.02 if two_order else 0.01,
            )
        )
    return ScorerRun(
        model_id=BASE_MODEL,
        revision=BASE_REVISION,
        dtype="float32",
        device="cpu",
        dataset_id=dataset_id,
        split=split,
        prompt_version=PROMPT_VERSION,
        template_digest=template_digest(),
        mode="two_order_average" if two_order else "single_order",
        max_prompt_tokens=8000,
        results=tuple(results),
        rejections=tuple(rejections),
        adapter_digest=adapter_digest,
    )


def save(run: ScorerRun, store: DerivedStore) -> str:
    return save_scorer_run(run, store).artifact_id
