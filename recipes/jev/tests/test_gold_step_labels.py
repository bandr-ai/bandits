from __future__ import annotations

import json

from recipes.jev.scripts.gold_step_labels import build, step_rows

from bandits_jev.step_judge import OPTIONS, QUESTION, step_state


def event(action, observation="reply", new=0, opened=False, tool="browser.search"):
    return {"tool": tool, "action": action, "observation": observation, "new_evidence_seen": new, "evidence_opened": opened}


ROLLOUT = {
    "query_id": "7", "sample": 2,
    "events": [
        event('{"query": "a"}', new=2),
        event('{"query": "b"}'),
        event('{"query": "A"}'),
        event('{"id": "9"}', tool="browser.open", opened=True),
        event('{"pattern": "("}', "Error: invalid pattern", tool="browser.find"),
    ],
}


def test_labels_come_from_gold_evidence_repeats_and_errors():
    rows = step_rows(ROLLOUT, "Q?", "run")
    assert [r["target"] for r in rows] == ["positive", "neutral", "negative", "positive", "negative"]


def test_the_judge_input_is_the_state_the_current_judge_reads():
    rows = step_rows(ROLLOUT, "Q?", "run")
    previous = [("browser.search", '{"query": "a"}', "reply")]
    assert rows[1]["state"] == step_state("Q?", previous, "browser.search", '{"query": "b"}', "reply")
    assert rows[1]["question"] == QUESTION and rows[1]["options"] == OPTIONS
    assert "gold" not in rows[1]["state"].lower()


def test_build_dedupes_balances_and_splits_by_question(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps([ROLLOUT, ROLLOUT]))
    rows = build([path], {"7": "Q?"}, per_class=1, splits=True)
    assert sorted(r["target"] for r in rows) == ["negative", "neutral", "positive"]
    assert len({r["split"] for r in rows}) == 1
