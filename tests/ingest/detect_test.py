from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.ingest.detect import DetectionError, detect_source

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (FIXTURES / "traces.chat.jsonl", "chat-json"),
        (FIXTURES / "upstream/interlingua/openinference.otlp.json", "otlp-std"),
        (FIXTURES / "upstream/interlingua/openllmetry.otlp.json", "otlp-std"),
    ],
)
def test_detect_known_fixtures(path: Path, expected: str) -> None:
    assert detect_source(path).source == expected


def test_detect_flat_otlp_and_reject_malformed_jsonl(tmp_path: Path) -> None:
    source = FIXTURES / "traces.otlp.jsonl"
    clean = tmp_path / "flat.jsonl"
    clean.write_text(source.read_text().splitlines()[0] + "\n")
    assert detect_source(clean).source == "otlp"
    with pytest.raises(DetectionError, match="malformed JSONL"):
        detect_source(source)


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"trace_id": "t", "observations": []}, "langfuse"),
        ({"id": "r", "run_type": "chain"}, "langsmith"),
        ({"run_id": "r", "run_type": "chain"}, "langsmith"),
        ({"context": {"trace_id": "t", "span_id": "s"}}, "phoenix"),
        ({"spans": [{"span_id": "s", "span_attributes": {}}]}, "trail"),
        ({"sessionId": "s", "type": "user", "message": {}}, "claude-code"),
    ],
)
def test_detect_native_shapes(tmp_path: Path, record: object, expected: str) -> None:
    assert detect_source(_write(tmp_path / "source.json", record)).source == expected


def test_unknown_and_mixed_shapes_stop(tmp_path: Path) -> None:
    _write(tmp_path / "unknown.json", {"items": [{"text": "hello"}]})
    with pytest.raises(DetectionError, match="unrecognized"):
        detect_source(tmp_path)
    (tmp_path / "unknown.json").unlink()
    _write(tmp_path / "a.json", {"id": "r", "run_type": "chain"})
    _write(tmp_path / "b.json", {"trace_id": "t", "observations": []})
    with pytest.raises(DetectionError, match="mixed source schemas"):
        detect_source(tmp_path)


def test_generic_batches_key_is_not_taken_for_otlp(tmp_path: Path) -> None:
    source = _write(tmp_path / "batches.json", {"batches": [{"items": [1, 2]}]})
    with pytest.raises(DetectionError, match="unrecognized"):
        detect_source(source)


def test_auto_ingest_and_read_only_check(tmp_path: Path) -> None:
    source = FIXTURES / "traces.chat.jsonl"
    assert load_corpus(source, "auto") == load_corpus(source, "chat-json")
    runner = CliRunner()
    result = runner.invoke(app, ["check-source", str(source)])
    assert result.exit_code == 0, result.output
    assert "detected:    chat-json" in result.output
    assert not (tmp_path / ".bandits").exists()
    result = runner.invoke(app, ["ingest", str(source), "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "detected:    chat-json" in result.output


def test_unknown_auto_ingest_stops_before_artifact(tmp_path: Path) -> None:
    source = _write(tmp_path / "unknown.json", {"items": [1]})
    result = CliRunner().invoke(app, ["ingest", str(source), "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "unrecognized trace structure" in result.output
    assert not (tmp_path / ".bandits").exists()


def test_auto_otlp_requires_declared_interaction_mode(tmp_path: Path) -> None:
    source = FIXTURES / "upstream/interlingua/openinference.otlp.json"
    with pytest.raises(ValueError, match="interaction mode is not in the file format"):
        load_corpus(source, "auto")
    check = CliRunner().invoke(app, ["check-source", str(source)])
    assert check.exit_code == 1
    assert "detected:    otlp-std" in check.output
    assert "choose --mode conversation or --mode workflow" in check.output
    result = CliRunner().invoke(app, ["ingest", str(source), "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "choose --mode conversation or --mode workflow" in result.output
    assert not (tmp_path / ".bandits").exists()


def test_check_reports_the_model_kind_used_by_the_reader() -> None:
    source = FIXTURES / "upstream/interlingua/vercel.json"
    result = CliRunner().invoke(app, ["check-source", str(source), "--mode", "conversation"])
    assert "detected:    otlp-std" in result.output
    assert "via ai.operationId=ai.generateText.doGenerate" in result.output
