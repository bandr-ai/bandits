"""Does the step judge track real progress? Measurements over a saved rollout file.

    python recipes/jev/scripts/analyze_rollouts.py work/step-rl/rollouts/base.json

Pure Python. Step labels come from ground truth (a step found or opened a gold
evidence document) and are used only here, never in a prompt. Intervals come
from resampling whole questions, since rollouts of one question are not
independent.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ABSTAIN = re.compile(
    r"do(?:es)? not (?:contain|provide|mention|specify|include)|not explicitly|no verified|cannot (?:be )?(?:determine|identif|find)"
    r"|insufficient|unable to|not (?:enough|sufficient) information|no (?:document|information)",
    re.IGNORECASE,
)


def auc(positive: list[float], negative: list[float]) -> float | None:
    """P(a random positive scores above a random negative); ties count half."""
    if not positive or not negative:
        return None
    ranked = sorted([(s, 1) for s in positive] + [(s, 0) for s in negative])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        average_rank = (i + j + 1) / 2
        rank_sum += average_rank * sum(label for _, label in ranked[i:j])
        i = j
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def step_rows(results: list[dict]) -> list[tuple[str, float, bool]]:
    return [
        (r["query_id"], e["judge"]["score"], bool(e["new_evidence_seen"] or e["evidence_opened"]))
        for r in results
        for e in r["events"]
        if e.get("judge") and e["judge"]["score"] is not None
    ]


def interval(values: list[float]) -> list[float]:
    values = sorted(values)
    return [round(values[int(0.025 * len(values))], 3), round(values[int(0.975 * len(values)) - 1], 3)]


def bootstrap(by_question: dict[str, list], statistic, rounds: int = 400, seed: int = 0) -> list[float]:
    rng, ids, out = random.Random(seed), list(by_question), []
    for _ in range(rounds):
        sample = [row for q in rng.choices(ids, k=len(ids)) for row in by_question[q]]
        value = statistic(sample)
        if value is not None:
            out.append(value)
    return interval(out)


def step_auc(rows: list[tuple[str, float, bool]]) -> float | None:
    return auc([s for _, s, hit in rows if hit], [s for _, s, hit in rows if not hit])


def within_question_auc(results: list[dict], score) -> dict:
    """For each question whose rollouts differ in outcome, how often does the higher-scored
    rollout also have the better outcome? This is what a trajectory-level shaping term acts on."""
    groups = defaultdict(list)
    for r in results:
        groups[r["query_id"]].append(r)
    values = []
    for rollouts in groups.values():
        good = [score(r) for r in rollouts if r["correct"]]
        bad = [score(r) for r in rollouts if not r["correct"]]
        value = auc(good, bad)
        if value is not None:
            values.append(value)
    return {"questions_with_mixed_outcomes": len(values), "mean_auc": round(sum(values) / len(values), 3) if values else None}


def analyze(data: dict, rounds: int = 400) -> dict:
    results = data["results"]
    n = len(results)
    per_question = defaultdict(list)
    for r in results:
        per_question[r["query_id"]].append(r)
    correct_counts = Counter(sum(r["correct"] for r in rs) for rs in per_question.values())
    group_size = len(next(iter(per_question.values())))

    rows = step_rows(results)
    rows_by_question = defaultdict(list)
    for row in rows:
        rows_by_question[row[0]].append(row)

    def mean_judge(r: dict) -> float:
        scores = [e["judge"]["score"] for e in r["events"] if e.get("judge") and e["judge"]["score"] is not None]
        return sum(scores) / len(scores) if scores else 0.0

    finals = [r["final"] or "" for r in results]
    abstained = [bool(ABSTAIN.search(f)) for f in finals]
    return {
        "rollouts": n,
        "questions": len(per_question),
        "group_size": group_size,
        "accuracy_proxy": round(sum(r["correct"] for r in results) / n, 4),
        "correct_per_question_histogram": {k: correct_counts.get(k, 0) for k in range(group_size + 1)},
        "share_of_questions_all_wrong": round(correct_counts.get(0, 0) / len(per_question), 3),
        "share_of_questions_with_outcome_variance": round(
            sum(v for k, v in correct_counts.items() if 0 < k < group_size) / len(per_question), 3
        ),
        "tool_calls_mean": round(sum(r["tool_calls"] for r in results) / n, 2),
        "tool_calls_histogram": dict(sorted(Counter(r["tool_calls"] for r in results).items())),
        "abstain_phrasing_share": round(sum(abstained) / n, 3),
        "accuracy_when_abstain_phrasing": round(
            sum(r["correct"] for r, a in zip(results, abstained, strict=True) if a) / max(1, sum(abstained)), 3
        ),
        "answer_format_share": round(sum("answer:" in f.lower() for f in finals) / n, 3),
        "evidence_seen_recall": round(sum(r["evidence_seen_recall"] for r in results) / n, 3),
        "evidence_opened_recall": round(sum(r["evidence_opened_recall"] for r in results) / n, 3),
        "accuracy_if_all_evidence_seen": _accuracy_if(results, lambda r: r["evidence_seen_recall"] >= 0.999),
        "accuracy_if_no_evidence_seen": _accuracy_if(results, lambda r: r["evidence_seen_recall"] == 0),
        "judge_step_auc_vs_evidence": {
            "steps": len(rows),
            "evidence_steps": sum(hit for _, _, hit in rows),
            "auc": round(step_auc(rows), 3),
            "ci95_over_questions": bootstrap(rows_by_question, step_auc, rounds),
        },
        "judge_positive_share_on_non_evidence_steps": round(
            sum(s > 0.25 for _, s, hit in rows if not hit) / max(1, sum(not hit for _, _, hit in rows)), 3
        ),
        "within_question_auc_mean_judge_score_vs_correct": within_question_auc(results, mean_judge),
        "within_question_auc_evidence_seen_vs_correct": within_question_auc(results, lambda r: r["evidence_seen_recall"]),
        "within_question_auc_tool_calls_vs_correct": within_question_auc(results, lambda r: r["tool_calls"]),
        "repeat_rollouts_share": round(sum(len({e["action"] for e in r["events"]}) < len(r["events"]) for r in results) / n, 3),
    }


def _accuracy_if(results: list[dict], keep) -> dict:
    chosen = [r for r in results if keep(r)]
    return {"rollouts": len(chosen), "accuracy": round(sum(r["correct"] for r in chosen) / len(chosen), 3) if chosen else None}


if __name__ == "__main__":
    print(json.dumps(analyze(json.loads(Path(sys.argv[1]).read_text())), indent=2))
