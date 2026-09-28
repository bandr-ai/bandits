from __future__ import annotations

import json

import pytest

from bandits_jev.dataset import DecisionExample, DecisionLineage, DecisionTarget
from bandits_jev.jev_api import load_cache, score_jev_api


def example() -> DecisionExample:
    return DecisionExample(
        decision_id="decision-1",
        family_id="family-1",
        state="The tool returned an error.",
        question="Judge the step.",
        primitive="choice",
        options={"success": "advanced", "failure": "failed"},
        target=DecisionTarget(kind="hard", probabilities={"success": 0.0, "failure": 1.0}),
        label_source="test",
        split="test",
        lineage=DecisionLineage(source_kind="jsonl_import", record_id="1"),
    )


def test_score_api_requires_a_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="JEV_API_KEY"):
        score_jev_api([example()], tmp_path / "out.jsonl")


def test_score_api_resumes_without_calling_network(tmp_path, monkeypatch) -> None:
    path = tmp_path / "out.jsonl"
    row = {"decision_id": "decision-1", "probabilities": {"success": 0.2, "failure": 0.8}}
    path.write_text(json.dumps(row) + "\n")

    def fail(*args, **kwargs):
        raise AssertionError("network should not be called for a cached row")

    monkeypatch.setattr("bandits_jev.jev_api._request", fail)
    rows, errors = score_jev_api([example()], path, api_key="secret")
    assert rows == {"decision-1": row}
    assert errors == []
    assert load_cache(path) == rows


def test_score_api_normalizes_rounded_cached_probabilities(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"decision_id": "decision-1", "probabilities": {"success": 0.94, "failure": 0.05}}) + "\n"
    )
    rows, errors = score_jev_api([example()], path, api_key="secret")
    assert errors == []
    assert sum(rows["decision-1"]["probabilities"].values()) == pytest.approx(1.0)
    written = json.loads(path.read_text())
    assert sum(written["probabilities"].values()) == pytest.approx(1.0)


def test_score_api_rejects_a_cache_from_another_split(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps({"decision_id": "other"}) + "\n")
    with pytest.raises(ValueError, match="outside this dataset split"):
        score_jev_api([example()], path, api_key="secret")
