from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.ingest.native import _ns
from bandits.store import ArtifactStore
from bandits.traces import SpanKind, SpanStatus, WorkflowDeclaration


def _file(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


def _assert_archived(tmp_path: Path, path: Path, corpus) -> None:
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(path))
    manifest = store.source_manifest(artifact.artifact_id)
    assert len(manifest) == 1
    assert store.read_source(artifact.artifact_id, str(manifest[0]["archive"])) == path.read_bytes()


def test_native_langfuse_keeps_request_and_observation(tmp_path: Path) -> None:
    observation = {
        "id": "a1",
        "traceId": "trace-1",
        "type": "SPAN",
        "name": "run",
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:02Z",
        "input": {"query": "What is covered?", "empty": "", "nil": None},
        "output": {"answer": "Dental."},
        "metadata": {
            "x": 0,
            "list": [],
            "resourceAttributes": {"service.name": "hr"},
            "scope": {"name": "langfuse-sdk", "version": "3.14"},
        },
    }
    model = {
        "id": "a2",
        "traceId": "trace-1",
        "parentObservationId": "a1",
        "type": "GENERATION",
        "name": "answer",
        "startTime": "2026-01-01T00:00:01Z",
        "endTime": "2026-01-01T00:00:02Z",
        "input": "Question: What is covered?",
        "output": "Dental.",
        "model": "model-1",
    }
    path = _file(
        tmp_path,
        "langfuse.json",
        {
            "trace_id": "trace-1",
            "trace_name": "help",
            "tags": [],
            "observations": [observation, model],
        },
    )
    corpus = load_corpus(
        path,
        "langfuse",
        workflow=WorkflowDeclaration(task_fields=("input.query",), delivered_field="output.answer"),
    )
    trace = corpus.traces[0]
    assert trace.request is not None
    assert trace.request.task == "What is covered?"
    assert trace.request.delivered == "Dental."
    assert trace.user_turns == ()
    assert len(trace.spans) == 1 and trace.spans[0].kind == SpanKind.MODEL
    assert json.loads(trace.spans[0].attributes["bandits.native.record"]) == model
    assert trace.request.raw_input == observation["input"]
    assert trace.spans[0].attributes["bandits.otlp.source_context"]["resource"] == {}
    _assert_archived(tmp_path, path, corpus)
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(path),
            "--source",
            "langfuse",
            "--mode",
            "workflow",
            "--task-field",
            "input.query",
            "--delivered-field",
            "output.answer",
            "--project",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "traces:      1" in result.output


def test_native_langsmith_maps_run_tree_and_messages(tmp_path: Path) -> None:
    root_id = "11111111-1111-1111-1111-111111111111"
    model_id = "22222222-2222-2222-2222-222222222222"
    root = {
        "id": root_id,
        "trace_id": root_id,
        "run_type": "chain",
        "name": "pipeline",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:03Z",
        "inputs": {"query": "hello"},
        "outputs": {"answer": "hi"},
    }
    llm = {
        "id": model_id,
        "trace_id": root_id,
        "parent_run_id": root_id,
        "run_type": "llm",
        "name": "chat",
        "start_time": "2026-01-01T00:00:01Z",
        "end_time": "2026-01-01T00:00:02Z",
        "inputs": {"messages": [{"role": "user", "content": "hello"}]},
        "outputs": {"messages": [{"role": "assistant", "content": "hi"}]},
        "extra": {"metadata": {"flag": False}},
    }
    path = _file(tmp_path, "langsmith.json", {"runs": [root, llm]})
    corpus = load_corpus(path, "langsmith")
    trace = corpus.traces[0]
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.kind == SpanKind.MODEL
    assert json.loads(span.attributes["gen_ai.input.messages"])[0]["role"] == "user"
    assert json.loads(span.attributes["bandits.native.record"]) == llm
    assert (
        span.parent_span_id == hashlib.sha256(f"langsmith:span:{root_id}".encode()).hexdigest()[:16]
    )
    _assert_archived(tmp_path, path, corpus)


def test_native_langsmith_error_status(tmp_path: Path) -> None:
    run = {
        "id": "run-1",
        "run_type": "llm",
        "name": "failed",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "status": "error",
        "error": "timeout",
        "inputs": {},
        "outputs": None,
    }
    span = load_corpus(_file(tmp_path, "run.json", run), "langsmith").traces[0].spans[0]
    assert span.status == SpanStatus.ERROR


def test_native_langsmith_nested_child_runs(tmp_path: Path) -> None:
    child = {
        "id": "model",
        "run_type": "llm",
        "name": "reply",
        "start_time": "2026-01-01T00:00:01Z",
        "end_time": "2026-01-01T00:00:02Z",
        "inputs": {"text": "hello"},
        "outputs": {"text": "hi"},
    }
    root = {
        "id": "root",
        "trace_id": "root",
        "run_type": "chain",
        "name": "agent",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:03Z",
        "inputs": {"query": "hello"},
        "outputs": {"answer": "hi"},
        "child_runs": [child],
    }
    trace = load_corpus(_file(tmp_path, "nested.json", root), "langsmith").traces[0]
    assert len(trace.spans) == 1
    assert trace.spans[0].parent_span_id == hashlib.sha256(b"langsmith:span:root").hexdigest()[:16]


def test_native_phoenix_preserves_openinference_and_links(tmp_path: Path) -> None:
    raw = {
        "name": "llm",
        "context": {"trace_id": "0x" + "1" * 32, "span_id": "0x" + "2" * 16},
        "parent_id": "0x" + "3" * 16,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "attributes": {
            "openinference.span.kind": "LLM",
            "input.value": "hello",
            "output.value": "hi",
        },
        "links": [{"traceId": "4" * 32, "spanId": "5" * 16, "attributes": {"why": "recorded"}}],
        "events": [
            {
                "name": "checkpoint",
                "timeUnixNano": "1767225600000000000",
                "attributes": {"ok": True},
            }
        ],
        "resource": {"attributes": {"service.name": "demo"}},
        "status": {"status_code": "ERROR", "message": "failed"},
    }
    path = _file(tmp_path, "phoenix.json", {"spans": [raw]})
    corpus = load_corpus(path, "phoenix")
    span = corpus.traces[0].spans[0]
    assert span.kind == SpanKind.MODEL
    assert json.loads(span.attributes["bandits.native.record"]) == raw
    assert span.attributes["bandits.otlp.source_context"]["links"] == raw["links"]
    assert span.attributes["bandits.otlp.source_context"]["resource"] == {"service.name": "demo"}
    assert span.status == SpanStatus.ERROR
    _assert_archived(tmp_path, path, corpus)


def test_native_timestamp_keeps_nanoseconds() -> None:
    assert _ns("2026-01-01T00:00:00.123456789Z") == 1767225600123456789


@pytest.mark.parametrize(
    "source_name,fixture",
    [
        ("otlp", "traces.otlp.jsonl"),
        ("chat-json", "traces.chat.jsonl"),
        ("claude-code", "session.multiturn.jsonl"),
    ],
)
def test_source_archive_preserves_redacted_input(
    tmp_path: Path, source_name: str, fixture: str
) -> None:
    path = Path(__file__).resolve().parents[1] / "fixtures" / fixture
    corpus = load_corpus(path, source_name)
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(path))
    manifest = store.source_manifest(artifact.artifact_id)
    assert len(manifest) == 1
    assert store.read_source(artifact.artifact_id, str(manifest[0]["archive"]))


def test_archive_redacts_source_and_retains_null(tmp_path: Path) -> None:
    path = _file(
        tmp_path, "chat.jsonl", {"messages": [], "nil": None, "secret": "sk-abcdefghijklmnop"}
    )
    corpus = load_corpus(path, "chat-json")
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(path))
    archive = store.read_source(artifact.artifact_id, "000000.json")
    assert b"sk-abcdefghijklmnop" not in archive
    assert b'"nil": null' in archive


def test_trail_reader_does_not_clip_recorded_text(tmp_path: Path) -> None:
    long_reply = "evidence " * 1200
    path = _file(
        tmp_path,
        "trail.json",
        {
            "trace_id": "trail-one",
            "spans": [
                {
                    "span_id": "agent",
                    "span_name": "agent",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "span_attributes": {
                        "openinference.span.kind": "AGENT",
                        "input.value": '{"task":"question"}',
                    },
                    "child_spans": [
                        {
                            "span_id": "step",
                            "span_name": "Step 1",
                            "timestamp": "2026-01-01T00:00:01Z",
                            "span_attributes": {
                                "openinference.span.kind": "CHAIN",
                                "output.value": long_reply,
                            },
                            "child_spans": [
                                {
                                    "span_id": "model",
                                    "span_name": "model",
                                    "timestamp": "2026-01-01T00:00:01Z",
                                    "span_attributes": {
                                        "openinference.span.kind": "LLM",
                                        "llm.output_messages.0.message.content": long_reply,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    corpus = load_corpus(path, "trail")
    assert corpus.traces[0].spans[0].output == long_reply
    assert corpus.traces[0].spans[1].output == long_reply
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(path))
    assert long_reply.encode() in store.read_source(artifact.artifact_id, "000000.json")


def test_otlp_preserves_separate_namespaces_and_links(tmp_path: Path) -> None:
    attr = lambda key, value: {"key": key, "value": value}  # noqa: E731
    event = {
        "name": "checkpoint",
        "timeUnixNano": "1767225600000000000",
        "attributes": [attr("ok", {"boolValue": True})],
        "droppedAttributesCount": 3,
    }
    link = {
        "traceId": "3" * 32,
        "spanId": "4" * 16,
        "attributes": [attr("edge", {"stringValue": "declared"})],
    }
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": [attr("same", {"intValue": "0"})]},
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "instrumentor",
                            "version": "1.0",
                            "attributes": [attr("empty", {"stringValue": ""})],
                        },
                        "spans": [
                            {
                                "traceId": "1" * 32,
                                "spanId": "2" * 16,
                                "name": "call",
                                "startTimeUnixNano": "1767225600000000000",
                                "endTimeUnixNano": "1767225601000000000",
                                "attributes": [
                                    attr("gen_ai.operation.name", {"stringValue": "chat"}),
                                    attr("same", {"stringValue": ""}),
                                ],
                                "links": [link],
                                "events": [event],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    path = _file(tmp_path, "otlp.json", payload)
    span = load_corpus(path, "otlp-std").traces[0].spans[0]
    context = span.attributes["bandits.otlp.source_context"]
    assert span.attributes["same"] == ""
    assert context["resource"]["same"] == 0
    assert context["span_attributes"]["same"] == ""
    assert context["scope"]["attributes"][0]["value"]["stringValue"] == ""
    assert context["links"] == [link]
    assert context["events"] == [event]


@pytest.mark.parametrize(
    "kind,model_key,input_key,output_key",
    [
        (
            "gen_ai.operation.name",
            "gen_ai.response.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        ),
        (
            "openinference.span.kind",
            "llm.response.model_name",
            "llm.token_count.prompt",
            "llm.token_count.completion",
        ),
        (
            "llm.request.type",
            "gen_ai.request.model",
            "gen_ai.usage.prompt_tokens",
            "gen_ai.usage.completion_tokens",
        ),
        (
            "langfuse.observation.type",
            "gen_ai.request.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        ),
    ],
)
def test_convention_scalar_keys_keep_zero_and_source(
    tmp_path: Path, kind: str, model_key: str, input_key: str, output_key: str
) -> None:
    kind_value = {
        "gen_ai.operation.name": "response",
        "openinference.span.kind": "LLM",
        "llm.request.type": "chat",
        "langfuse.observation.type": "GENERATION",
    }[kind]
    attributes = {
        kind: kind_value,
        model_key: "model-x",
        input_key: 0,
        output_key: 7,
    }
    payload = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "1" * 32,
                                "spanId": "2" * 16,
                                "name": "model",
                                "startTimeUnixNano": "1767225600000000000",
                                "endTimeUnixNano": "1767225601000000000",
                                "attributes": [
                                    {
                                        "key": k,
                                        "value": {"intValue": str(v)}
                                        if isinstance(v, int)
                                        else {"stringValue": v},
                                    }
                                    for k, v in attributes.items()
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    span = load_corpus(_file(tmp_path, "one.json", payload), "otlp-std").traces[0].spans[0]
    normalized = span.attributes["bandits.normalized_scalars"]
    assert normalized["model"] == {"source": model_key, "value": "model-x"}
    assert normalized["input_tokens"] == {"source": input_key, "value": 0}
    assert normalized["output_tokens"] == {"source": output_key, "value": 7}


@pytest.mark.parametrize("dialect", ["openinference", "openllmetry"])
def test_upstream_interlingua_otlp_fixture_maps_without_loss(tmp_path: Path, dialect: str) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "upstream"
        / "interlingua"
        / f"{dialect}.otlp.json"
    )
    corpus = load_corpus(source, "otlp-std")
    assert not corpus.issues
    assert len(corpus.traces) == 1
    assert [span.kind for span in corpus.traces[0].spans] == [SpanKind.MODEL, SpanKind.TOOL]
    model = corpus.traces[0].spans[0]
    normalized = model.attributes["bandits.normalized_scalars"]
    assert normalized["input_tokens"]["value"] == 412
    assert normalized["output_tokens"]["value"] == 27
    assert normalized["reasoning_tokens"]["value"] == 8
    assert model.attributes["gen_ai.input.messages"]
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(source))
    assert store.read_source(artifact.artifact_id, "000000.json") == source.read_bytes()
