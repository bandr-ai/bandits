from __future__ import annotations

import pytest
from recipes.jev.scripts.analyze_rollouts import analyze, auc


def test_auc_counts_ties_half_and_handles_empty_sides():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([], [1]) is None


def rollout(query_id, correct, scores, evidence_hits, seen=0.0, final="Answer: x"):
    events = [
        {"action": f"a{i}", "new_evidence_seen": int(hit), "evidence_opened": False,
         "judge": {"score": s, "probabilities": {"positive": 0.5, "neutral": 0.4, "negative": 0.1}}}
        for i, (s, hit) in enumerate(zip(scores, evidence_hits, strict=True))
    ]
    return {"query_id": query_id, "correct": correct, "tool_calls": len(events), "events": events,
            "evidence_seen_recall": seen, "evidence_opened_recall": 0.0, "final": final}


def test_analyze_reports_headroom_judge_auc_and_within_question_ranking():
    results = []
    for q in ("q1", "q2"):
        results += [
            rollout(q, True, [0.9, 0.8], [True, True], seen=1.0),
            rollout(q, False, [0.1, 0.0], [False, False], final="The documents do not contain this."),
        ]
    results += [rollout("q3", False, [0.2], [False]), rollout("q3", False, [0.3], [False])]
    out = analyze({"results": results}, rounds=20)
    assert out["share_of_questions_all_wrong"] == pytest.approx(1 / 3, abs=1e-3)
    assert out["share_of_questions_with_outcome_variance"] == pytest.approx(2 / 3, abs=1e-3)
    assert out["judge_step_auc_vs_evidence"]["auc"] == 1.0
    assert out["within_question_auc_mean_judge_score_vs_correct"] == {"questions_with_mixed_outcomes": 2, "mean_auc": 1.0}
    assert out["abstain_phrasing_share"] == pytest.approx(2 / 6, abs=1e-3)
    assert out["accuracy_if_no_evidence_seen"]["accuracy"] == 0.0
