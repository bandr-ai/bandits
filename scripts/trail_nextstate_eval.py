#!/usr/bin/env python3
"""Score a turn-judge run (and optionally verifier scores) against TRAIL's annotations.

TRAIL annotates every error by span id, category and impact, and gives each
trace a 1-5 ``overall`` reliability score. Two questions, both answered here:

  turn level  — does a turn the judge scored -1 (or a check flagged) sit where a
                human placed an error?  precision / recall / F1, vs. the base rate
                and vs. the "tool span errored" baseline.
  trace level — does the trace's share of clean turns track the human overall
                score?  Spearman, plus AUC of top vs bottom tertile.

Usage:
    uv run python scripts/trail_nextstate_eval.py --project work/trail-ns \
        --judge-run turn-judge-… [--scores verifier-scores-…] \
        --trail-dir /tmp/trail-benchmark/benchmarking --split gaia --out work/trail-ns/eval-gaia.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bandits.store import ArtifactStore, DerivedStore
from bandits.verify.nextstate import load_turn_judge_run
from bandits.verify.propose import load_verifier_scores
from bandits.verify.turns import extract_turns

_ANN = {"gaia": "processed_annotations_gaia", "swe_bench": "processed_annotations_swe_bench"}


def load_annotations(trail_dir: Path, split: str) -> dict[str, dict]:
    out = {}
    for path in sorted((trail_dir / _ANN[split]).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        out[data.get("trace_id") or path.stem] = data
    return out


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = rank
            i = j + 1
        return out

    xs, ys = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if not vx or not vy else cov / (vx * vy)


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brier(pairs: list[tuple[float, bool]]) -> float | None:
    """Mean squared error of a predicted probability against a 0/1 outcome."""
    if not pairs:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / len(pairs)


def reliability_bins(pairs: list[tuple[float, bool]], n_bins: int = 10) -> list[dict]:
    """Fixed-width probability bins: each bin's range, count, mean prediction
    and observed rate. Every bin from 0 to ``n_bins`` is reported, including
    empty ones (``n``: 0, ``mean_predicted``/``observed_rate``: None) -- an
    ECE number is easy to overread on a small slice without seeing which
    bins actually held data.
    """
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    out = []
    for i, bucket in enumerate(bins):
        out.append(
            {
                "range": [i / n_bins, (i + 1) / n_bins],
                "n": len(bucket),
                "mean_predicted": sum(p for p, _ in bucket) / len(bucket) if bucket else None,
                "observed_rate": sum(1.0 for _, y in bucket if y) / len(bucket)
                if bucket
                else None,
            }
        )
    return out


def ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float | None:
    """Expected calibration error: bin-size-weighted mean |predicted - observed|."""
    if not pairs:
        return None
    bins = reliability_bins(pairs, n_bins)
    total = len(pairs)
    return sum(
        b["n"] / total * abs(b["mean_predicted"] - b["observed_rate"]) for b in bins if b["n"]
    )


def calibration_pairs(
    verdicts: dict[tuple[str, int], object], observed: set, truth_any: set
) -> list[tuple[float, bool]]:
    """(p(-1), was actually an error) for every observed, judged turn.

    Every turn the judge produced a vote distribution for is included, even
    one where every vote was "0" or "1" -- ``judge_votes`` is now dense, so
    ``p(-1) = 0.0`` there is a real observed frequency, not a missing key.
    Excluding those turns would restrict prevalence, Brier and ECE to the
    subset that happened to draw at least one negative vote, which biases
    every number the calibration report prints.
    """
    return [
        (v.judge_votes.get("-1", 0.0), key in truth_any)
        for key, v in verdicts.items()
        if key in observed and v.judge_votes is not None
    ]


def prf(predicted: set, truth: set, universe: set) -> dict:
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else None
    recall = tp / len(truth) if truth else None
    # `None` means undefined (an empty denominator set), never a computed
    # 0.0 -- disjoint but non-empty predicted/truth sets give real 0.0
    # precision and recall, and F1 there is 0.0, not undefined.
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "predicted": len(predicted),
        "truth": len(truth),
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "base_rate": len(truth) / len(universe) if universe else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--judge-run", required=True)
    parser.add_argument(
        "--scores", default=None, help="verifier-scores artifact to evaluate as well"
    )
    parser.add_argument("--trail-dir", type=Path, required=True)
    parser.add_argument("--split", choices=list(_ANN), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    derived = DerivedStore(args.project / ".bandits")
    run = load_turn_judge_run(args.judge_run, derived)
    corpus = ArtifactStore(args.project / ".bandits").read(run.corpus_id)
    annotations = load_annotations(args.trail_dir, args.split)
    wanted = set(run.trace_ids)
    traces = [t for t in corpus.traces if t.trace_id in wanted and t.trace_id in annotations]

    # span id -> (trace_id, turn index): an error on the action or on any of its reactions
    span_to_turn: dict[str, tuple[str, int]] = {}
    turns_of = {}
    for trace in traces:
        turns = extract_turns(trace)
        turns_of[trace.trace_id] = turns
        for turn in turns:
            span_to_turn[turn.action_span_id] = (trace.trace_id, turn.index)
            for reaction in turn.reactions:
                if reaction.span_id:
                    span_to_turn.setdefault(reaction.span_id, (trace.trace_id, turn.index))

    observed = {(t.trace_id, t.index) for turns in turns_of.values() for t in turns if t.observed}
    all_turns = {(t.trace_id, t.index) for turns in turns_of.values() for t in turns}
    truth_any: set = set()
    truth_high: set = set()
    unmapped = 0
    categories: Counter = Counter()
    for trace in traces:
        for error in annotations[trace.trace_id].get("errors", []):
            key = span_to_turn.get(error.get("location"))
            if key is None:
                unmapped += 1
                continue
            truth_any.add(key)
            categories[error.get("category")] += 1
            if error.get("impact") in {"HIGH", "MEDIUM"}:
                truth_high.add(key)

    verdicts = run.verdict_by_key()
    judged_negative = {k for k, v in verdicts.items() if v.score == -1 and k in observed}
    calib_pairs = calibration_pairs(verdicts, observed, truth_any)
    prevalence = sum(1.0 for _, y in calib_pairs if y) / len(calib_pairs) if calib_pairs else None
    baseline_pairs = [(prevalence, y) for _, y in calib_pairs] if prevalence is not None else []
    votes_valid = [
        len(v.votes) for k, v in verdicts.items() if k in observed and v.judge_votes is not None
    ]
    errored = {(t.trace_id, t.index) for turns in turns_of.values() for t in turns if t.errored}
    flagged = None
    if args.scores:
        scores = load_verifier_scores(args.scores, derived)
        flagged = {(s.trace_id, f.index) for s in scores.scores for f in s.flagged}

    turn_level = {
        "observed_turns": len(observed),
        "all_turns": len(all_turns),
        "annotated_errors_unmapped": unmapped,
        "categories": dict(categories.most_common()),
        "judge_-1 vs any error (observed turns)": prf(
            judged_negative, truth_any & observed, observed
        ),
        "judge_-1 vs HIGH/MEDIUM error (observed turns)": prf(
            judged_negative, truth_high & observed, observed
        ),
        "tool_errored baseline vs any error": prf(
            errored & observed, truth_any & observed, observed
        ),
        "annotated errors on unobserved turns": len(truth_any - observed),
        "calibration": {
            "n": len(calib_pairs),
            "error_prevalence": prevalence,
            "votes_requested": run.votes,
            "votes_valid_per_example": {
                "min": min(votes_valid) if votes_valid else None,
                "max": max(votes_valid) if votes_valid else None,
                "mean": sum(votes_valid) / len(votes_valid) if votes_valid else None,
            },
            "brier(p(-1), any_error)": brier(calib_pairs),
            "brier_baseline(prevalence, any_error)": brier(baseline_pairs),
            "ece(p(-1), any_error)": ece(calib_pairs),
            "ece_n_bins": 10,
            "reliability_bins": reliability_bins(calib_pairs),
        },
    }
    if flagged is not None:
        turn_level["verifier flagged vs any error (observed)"] = prf(
            flagged & observed, truth_any & observed, observed
        )
        turn_level["verifier flagged vs HIGH/MEDIUM (observed)"] = prf(
            flagged & observed, truth_high & observed, observed
        )

    overall = {t: (annotations[t].get("scores") or [{}])[0].get("overall") for t in turns_of}
    n_errors = {t: len(annotations[t].get("errors", [])) for t in turns_of}
    signals = run.signal_by_trace()
    pairs_score = [
        (s.score, overall[t])
        for t, s in signals.items()
        if t in overall and s.score is not None and overall[t] is not None
    ]
    pairs_neg = [
        (-s.negative, overall[t])
        for t, s in signals.items()
        if t in overall and overall[t] is not None
    ]
    pairs_errs = [
        (s.score, -n_errors[t]) for t, s in signals.items() if t in n_errors and s.score is not None
    ]
    values = sorted(o for o in overall.values() if o is not None)
    lo, hi = values[len(values) // 3], values[2 * len(values) // 3]
    tert = [(s, o >= hi) for s, o in pairs_score if o >= hi or o <= lo]
    trace_level = {
        "traces": len(pairs_score),
        "spearman(clean_share, overall)": spearman(pairs_score),
        "spearman(-negatives, overall)": spearman(pairs_neg),
        "spearman(clean_share, -n_errors)": spearman(pairs_errs),
        "auc top-vs-bottom tertile": auc(tert),
        "passing traces": sum(1 for s in signals.values() if s.passes),
    }
    if flagged is not None:
        by_trace = {s.trace_id: s for s in scores.scores}
        pairs_v = [
            (by_trace[t].score, overall[t])
            for t in by_trace
            if t in overall and by_trace[t].score is not None and overall[t] is not None
        ]
        trace_level["verifier spearman(score, overall)"] = spearman(pairs_v)
        trace_level["verifier spearman(score, -n_errors)"] = spearman(
            [
                (by_trace[t].score, -n_errors[t])
                for t in by_trace
                if t in n_errors and by_trace[t].score is not None
            ]
        )
        trace_level["verifier auc top-vs-bottom tertile"] = auc(
            [(s, o >= hi) for s, o in pairs_v if o >= hi or o <= lo]
        )

    report = {
        "judge_run": args.judge_run,
        "scores": args.scores,
        "split": args.split,
        "archetype": run.archetype.value,
        "model": run.model,
        "turn_level": turn_level,
        "trace_level": trace_level,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
