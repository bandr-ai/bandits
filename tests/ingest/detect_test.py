from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.ingest.detect import DetectionError, detect_source
from bandits.store import ArtifactStore

FIXTURES = Path(__file__).parents[1] / "fixtures"


def plain(text: str) -> str:
    """CLI output with terminal line wrapping undone."""
    return " ".join(text.split())


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
    result = runner.invoke(app, ["ingest", str(source), "--dry-run", "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "format:   chat-json" in result.output
    assert "nothing saved" in result.output
    assert not (tmp_path / ".bandits").exists()
    result = runner.invoke(app, ["ingest", str(source), "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "artifact_id: corpus-" in result.output


def test_unknown_auto_ingest_stops_before_artifact(tmp_path: Path) -> None:
    source = _write(tmp_path / "unknown.json", {"items": [1]})
    result = CliRunner().invoke(app, ["ingest", str(source), "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "could not tell what format this file is" in result.output
    assert "fix: pass --source NAME" in result.output
    assert not (tmp_path / ".bandits").exists()


def test_auto_otlp_defaults_to_workflow_mode(tmp_path: Path) -> None:
    """The library API still refuses to guess; the CLI picks the safe mode and says so."""
    source = FIXTURES / "upstream/interlingua/openinference.otlp.json"
    with pytest.raises(ValueError, match="interaction mode is not in the file format"):
        load_corpus(source, "auto")
    result = CliRunner().invoke(app, ["ingest", str(source), "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "mode: workflow (default" in plain(result.output)
    store = ArtifactStore(tmp_path / ".bandits")
    corpus = store.read(store.list()[0].artifact_id)
    assert corpus.workflow is not None
    assert all(not trace.user_turns for trace in corpus.traces)


def test_workflow_options_in_conversation_mode_are_explained(tmp_path: Path) -> None:
    source = FIXTURES / "traces.chat.jsonl"
    result = CliRunner().invoke(
        app, ["ingest", str(source), "--task-field", "input.q", "--project", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "only apply to workflows" in result.output
    assert "why:" in result.output and "fix:" in result.output


def test_check_reports_the_model_kind_used_by_the_reader() -> None:
    source = FIXTURES / "upstream/interlingua/vercel.json"
    result = CliRunner().invoke(app, ["ingest", str(source), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "format:   otlp-std" in result.output
    assert "read:     1 traces, 1 model calls" in result.output


def _declared_chat(tmp_path: Path, input_messages: object) -> Path:
    def attr(key: str, value: str) -> dict:
        return {"key": key, "value": {"stringValue": value}}

    span = {
        "traceId": "0" * 31 + "1",
        "spanId": "0" * 15 + "1",
        "name": "chat",
        "startTimeUnixNano": "1000",
        "endTimeUnixNano": "2000",
        "attributes": [
            attr("gen_ai.operation.name", "chat"),
            attr("gen_ai.input.messages", json.dumps(input_messages)),
            attr(
                "gen_ai.output.messages",
                json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}]),
            ),
        ],
    }
    return _write(
        tmp_path / "chat.otlp.json",
        {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]},
    )


def test_check_reads_declared_messages_as_json(tmp_path: Path) -> None:
    """Producers write gen_ai.*.messages as a JSON string; that is not malformed."""
    source = _declared_chat(
        tmp_path, [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
    )
    result = CliRunner().invoke(app, ["ingest", str(source), "--dry-run"])
    assert "malformed" not in result.output
    assert result.exit_code == 0


def test_check_flags_messages_without_roles_and_parts(tmp_path: Path) -> None:
    source = _declared_chat(tmp_path, ["hello"])
    result = CliRunner().invoke(app, ["ingest", str(source), "--dry-run"])
    output = plain(result.output)
    assert "1 model call(s) have malformed message lists" in output
    assert "why: a message has no valid role" in output
    assert "nothing was saved" in output
    assert result.exit_code == 1


def test_check_does_not_call_an_errored_model_response_lost(tmp_path: Path) -> None:
    source = _declared_chat(
        tmp_path, [{"role": "user", "parts": [{"type": "text", "content": "hello"}]}]
    )
    record = json.loads(source.read_text())
    span = record["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["attributes"] = [
        item for item in span["attributes"] if item["key"] != "gen_ai.output.messages"
    ]
    span["status"] = {"code": 2, "message": "provider unavailable"}
    source.write_text(json.dumps(record))
    result = CliRunner().invoke(app, ["ingest", str(source), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "no output in a field Bandits reads" not in result.output
