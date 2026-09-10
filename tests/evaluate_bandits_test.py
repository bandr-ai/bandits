from __future__ import annotations

import json
from pathlib import Path

from bandits.labels import Verdict
from scripts.evaluate_bandits import (
    _load_local_env,
    _model_label_prompt,
    _parse_model_label,
    prepare,
    render_html,
    score,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_prepare_creates_a_blind_project_and_runbook(tmp_path: Path) -> None:
    project = tmp_path / "blind"

    manifest = prepare(FIXTURES / "traces.otlp.jsonl", "otlp", project)

    assert manifest["trace_count"] > 0
    assert manifest["analysis_id"].startswith("analysis-")
    assert (project / ".bandits" / "artifacts" / manifest["corpus_id"]).is_dir()
    runbook = (project / "RUNBOOK.md").read_text()
    assert "Do not open the benchmark truth" in runbook
    assert manifest["analysis_id"] in runbook


def test_score_requires_completed_human_review_but_still_writes_zero_state(
    tmp_path: Path,
) -> None:
    project = tmp_path / "blind"
    prepare(FIXTURES / "traces.otlp.jsonl", "otlp", project)
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"checkout-1": {"success": True, "task_id": "1"}}))

    report = score(project, truth)

    assert report["workflow"]["reviewed_families"] == 0
    assert report["held_out"]["scored"] == 0
    assert report["scores"]["trust_score"] is None


def test_viewer_is_self_contained_and_handles_an_unfinished_run(tmp_path: Path) -> None:
    project = tmp_path / "blind"
    prepare(FIXTURES / "traces.otlp.jsonl", "otlp", project)
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"checkout-1": {"success": True}}))

    page = render_html(score(project, truth))

    assert "Bandits HITL evaluation" in page
    assert "No completed reviewed verifier yet" in page
    assert "<script" not in page


def test_reviewer_loads_credentials_from_the_repo_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    (tmp_path / ".env").write_text("FIREWORKS_API_KEY='test-key'\n")

    loaded = _load_local_env(tmp_path / "review")

    assert loaded == (tmp_path / ".env",)
    assert __import__("os").environ["FIREWORKS_API_KEY"] == "test-key"


def test_model_judge_requires_outcome_evidence_and_structured_reasoning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "scripts.evaluate_bandits.render_transcript", lambda trace: "REQUEST and TOOL RESULT"
    )

    prompt = _model_label_prompt(object())

    assert "every requested outcome and constraint" in prompt
    assert "External tool results" in prompt
    assert "stronger evidence than the agent's own claims" in prompt
    assert "unrequested irreversible change" in prompt
    assert '"supporting_evidence"' in prompt


def test_model_judge_keeps_structured_rationale_and_fails_closed() -> None:
    verdict, rationale = _parse_model_label(
        '{"verdict":"failure","confidence":0.91,"rationale":"wrong action"}'
    )
    assert verdict is Verdict.FAILURE
    assert rationale.startswith("auto-label-v3:")
    assert "wrong action" in rationale

    verdict, rationale = _parse_model_label("not json")
    assert verdict is Verdict.UNCLEAR
    assert "unparseable response" in rationale
