from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import detect_source, load_corpus
from bandits.ingest.native import _ns
from bandits.ingest.otlp_standard import _attributes
from bandits.store import ArtifactStore, resolve_record
from bandits.traces import SpanKind, SpanStatus, WorkflowDeclaration
from tests.cli_test import plain


def _file(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


def test_native_reader_rejects_a_directory_through_its_public_error_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="reads one export file"):
        load_corpus(tmp_path, "langfuse")


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
    pointer = trace.spans[0].attributes["bandits.source.record"]
    assert json.loads(pointer) == {"line": 1, "observation_id": "a2"}
    assert resolve_record(path.read_bytes(), pointer) == model
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
    assert "read:     1 traces" in plain(result.output)


def test_upstream_langfuse_trace_wrapper_preserves_all_observations() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "fixtures/upstream/langfuse/agno-2025-06-11.trace.json"
    )
    raw = json.loads(source.read_text())
    assert detect_source(source).source == "langfuse"
    corpus = load_corpus(source, "langfuse", workflow=WorkflowDeclaration(task_fields=()))
    assert len(corpus.traces) == 1
    trace = corpus.traces[0]
    assert trace.request is not None and trace.request.task_status == "unresolved"
    assert trace.user_turns == ()
    represented = {span.span_id for span in trace.spans} | {
        node.span_id for node in trace.workflow_nodes
    }
    represented.add(trace.request.source_span_id)
    assert {observation["id"] for observation in raw["observations"]} <= represented
    assert sum(span.kind == SpanKind.MODEL for span in trace.spans) == 2
    wrappers = {
        node.span_id: node.attributes.get("bandits.duplicate_model_of")
        for node in trace.workflow_nodes
        if node.attributes.get("bandits.duplicate_model_of")
    }
    assert wrappers == {
        "b38a82eaa62b551e": "fc7f61b57ecebbaf",
        "ca136de468e156c9": "1f883bb9668167c0",
    }
    assert sum(issue.kind == "duplicate_model_instrumentation" for issue in corpus.issues) == 2
    assert not [
        issue
        for issue in corpus.issues
        if issue.kind not in ("task_unresolved", "redaction", "duplicate_model_instrumentation")
    ]


def test_phoenix_sdk_server_export_retains_request_model_and_parent() -> None:
    # Generated by Phoenix client 3.5.0 log_spans, Phoenix server 13.9.0,
    # then exported by client.get_spans. The client input was constructed;
    # the server's exported record shape is independent of this reader.
    source = (
        Path(__file__).resolve().parents[1] / "fixtures/upstream/phoenix/sdk-server-getspans.json"
    )
    raw = json.loads(source.read_text())
    assert detect_source(source).source == "phoenix"
    corpus = load_corpus(
        source,
        "phoenix",
        workflow=WorkflowDeclaration(task_fields=("input",), delivered_field="output"),
    )
    assert len(corpus.traces) == 1
    trace = corpus.traces[0]
    assert trace.request is not None
    assert trace.request.source_span_id == "1111111111111111"
    assert trace.request.task == "Where is the order?"
    assert trace.request.delivered == "Order 123 shipped."
    assert len(trace.spans) == 1
    model = trace.spans[0]
    assert model.span_id == "2222222222222222"
    assert model.parent_span_id == trace.request.source_span_id
    assert model.kind == SpanKind.MODEL
    assert model.attributes["openinference.span.kind"] == "LLM"
    record = resolve_record(source.read_bytes(), model.attributes["bandits.source.record"])
    assert record["attributes"]["input.value"] == (
        '[{"role":"user","content":"Where is the order?"}]'
    )
    assert {span["context"]["span_id"] for span in raw["data"]} == {
        trace.request.source_span_id,
        model.span_id,
    }
    assert not corpus.issues


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
    assert span.attributes["gen_ai.input.messages"] == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    assert span.attributes["gen_ai.output.messages"] == [
        {"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert resolve_record(path.read_bytes(), span.attributes["bandits.source.record"]) == llm
    assert (
        span.parent_span_id == hashlib.sha256(f"langsmith:span:{root_id}".encode()).hexdigest()[:16]
    )
    _assert_archived(tmp_path, path, corpus)


def test_langsmith_nested_langchain_batches_and_generations(tmp_path: Path) -> None:
    """Shape observed in a public LangSmith RunTree export, including nested lists."""
    human = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "HumanMessage"],
        "kwargs": {"content": "hello"},
    }
    assistant = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {"content": "hi"},
    }
    run = {
        "id": "run-1",
        "run_type": "llm",
        "name": "ChatModel",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "inputs": {"messages": [[human]]},
        "outputs": {"generations": [[{"type": "ChatGeneration", "message": assistant}]]},
    }
    span = load_corpus(_file(tmp_path, "run.json", run), "langsmith").traces[0].spans[0]
    assert span.attributes["gen_ai.input.messages"] == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    assert span.attributes["gen_ai.output.messages"] == [
        {"role": "assistant", "parts": [{"type": "text", "content": "hi"}]}
    ]


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


def test_langsmith_cli_export_run_id_shape(tmp_path: Path) -> None:
    """LangSmith's trace export uses run_id in its JSONL example."""
    run = {
        "run_id": "run-1",
        "trace_id": "trace-1",
        "run_type": "llm",
        "name": "chat",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "inputs": {"messages": [{"role": "user", "content": "hello"}]},
        "outputs": {"messages": [{"role": "assistant", "content": "hi"}]},
    }
    path = _file(tmp_path, "langsmith.jsonl", run)
    assert detect_source(path).source == "langsmith"
    corpus = load_corpus(path, "langsmith")
    assert len(corpus.traces) == 1
    assert len(corpus.traces[0].spans) == 1
    assert not [issue for issue in corpus.issues if issue.kind != "redaction"]
    pointer = corpus.traces[0].spans[0].attributes["bandits.source.record"]
    assert resolve_record(path.read_bytes(), pointer) == run


def test_phoenix_getspans_response_uses_top_level_span_kind(tmp_path: Path) -> None:
    """Phoenix's getSpans response has data[] and span_kind outside attributes."""
    span = {
        "id": "global-span-1",
        "name": "chat_completion",
        "context": {"trace_id": "trace-1", "span_id": "span-1"},
        "span_kind": "LLM",
        "parent_id": None,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "status_code": "ERROR",
        "status_message": "rate limited",
        "attributes": {"input.value": "hello", "output.value": "hi"},
        "events": [],
    }
    source = _file(tmp_path, "phoenix.json", {"data": [span], "next_cursor": None})
    assert detect_source(source).source == "phoenix"
    corpus = load_corpus(source, "phoenix")
    assert len(corpus.traces) == 1
    assert len(corpus.traces[0].spans) == 1
    decoded = corpus.traces[0].spans[0]
    assert decoded.kind == SpanKind.MODEL
    assert decoded.status == SpanStatus.ERROR
    assert decoded.attributes["openinference.span.kind"] == "LLM"
    assert not corpus.issues


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
    assert resolve_record(path.read_bytes(), span.attributes["bandits.source.record"]) == raw
    assert span.attributes["bandits.otlp.source_context"]["links"] == raw["links"]
    assert span.attributes["bandits.otlp.source_context"]["resource"] == {"service.name": "demo"}
    assert span.status == SpanStatus.ERROR
    _assert_archived(tmp_path, path, corpus)


def test_openinference_logical_span_status_code(tmp_path: Path) -> None:
    """The OpenInference logical span example has top-level status_code."""
    raw = {
        "name": "query",
        "context": {"trace_id": "a" * 32, "span_id": "b" * 16},
        "parent_id": None,
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "status_code": "ERROR",
        "status_message": "failed",
        "attributes": {
            "openinference.span.kind": "LLM",
            "input.value": "hello",
            "output.value": "error",
        },
        "events": [],
    }
    path = _file(tmp_path, "phoenix.json", raw)
    assert detect_source(path).source == "phoenix"
    corpus = load_corpus(path, "phoenix")
    assert corpus.traces[0].spans[0].status == SpanStatus.ERROR


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


@pytest.mark.parametrize(
    ("dialect", "models", "tools"),
    [
        ("vercel", 1, 1),
        ("braintrust", 1, 1),
        ("litellm", 1, 0),
        ("langchain", 1, 2),
        ("openllmetry-legacy", 1, 1),
    ],
)
def test_independent_captured_otel_dialects(
    tmp_path: Path, dialect: str, models: int, tools: int
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "upstream"
        / "interlingua"
        / f"{dialect}.json"
    )
    assert detect_source(source).source == "otlp-std"
    corpus = load_corpus(source, "otlp-std")
    spans = [span for trace in corpus.traces for span in trace.spans]
    assert sum(span.kind == SpanKind.MODEL for span in spans) == models
    assert sum(span.kind == SpanKind.TOOL for span in spans) == tools
    model = next(span for span in spans if span.kind == SpanKind.MODEL)
    assert model.attributes.get("gen_ai.input.messages") is not None
    assert model.attributes.get("gen_ai.output.messages") is not None
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(source))
    assert store.read_source(artifact.artifact_id, "000000.json") == source.read_bytes()
    if dialect == "vercel":
        assert model.attributes["bandits.normalized_scalars"]["input_tokens"]["value"] == 412
        assert any(
            part["type"] == "tool_call"
            for message in model.attributes["gen_ai.output.messages"]
            for part in message["parts"]
        )
    if dialect == "braintrust":
        assert not any(span.name == "scoring_span" for span in spans)


@pytest.mark.parametrize(
    "dialect",
    [
        "openinference.otlp",
        "openllmetry.otlp",
        "vercel",
        "braintrust",
        "litellm",
        "langchain",
        "openllmetry-legacy",
    ],
)
def test_captured_otel_keeps_every_attribute_of_retained_spans(dialect: str) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "upstream"
        / "interlingua"
        / f"{dialect}.json"
    )
    raw = json.loads(source.read_text())
    original = {
        span["spanId"]: _attributes(span.get("attributes"))
        for resource in raw["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    }
    corpus = load_corpus(source, "otlp-std")
    kept = [span for trace in corpus.traces for span in trace.spans]
    assert kept
    for span in (span for span in kept if span.span_id in original):
        # Every declared attribute is either in the attributes as declared, or
        # kept in the source context with its declared value; nothing else is.
        declared = original[span.span_id]
        context = span.attributes["bandits.otlp.source_context"]["span_attributes"]
        assert set(context) <= set(declared)
        for key, value in declared.items():
            assert span.attributes.get(key) == value or context.get(key) == value, key

    workflow = load_corpus(source, "otlp-std", workflow=WorkflowDeclaration(task_fields=()))
    represented = {span.span_id for trace in workflow.traces for span in trace.spans} | {
        node.span_id for trace in workflow.traces for node in trace.workflow_nodes
    }
    represented.update(
        trace.request.source_span_id
        for trace in workflow.traces
        if trace.request and trace.request.source_span_id
    )
    missing = set(original) - represented
    if dialect == "braintrust":
        assert len(missing) == 1
        assert any(issue.kind == "excluded_evaluator" for issue in workflow.issues)
    else:
        assert not missing


def test_langfuse_null_io_is_absent_not_the_text_null(tmp_path: Path) -> None:
    """Langfuse's public trace API returns input/output null until fetched."""
    generation = {
        "id": "g1",
        "traceId": "t1",
        "type": "GENERATION",
        "name": "chat",
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "input": None,
        "output": None,
        "model": "gpt-4o-mini",
    }
    path = _file(tmp_path, "lf.json", {"id": "t1", "observations": [generation]})
    span = load_corpus(path, "langfuse").traces[0].spans[0]
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes
    assert span.output is None
    assert (
        resolve_record(path.read_bytes(), span.attributes["bandits.source.record"])["input"] is None
    )


def test_langfuse_embedding_generation_is_not_a_model_call(tmp_path: Path) -> None:
    def generation(oid: str, model: str, start: str) -> dict:
        return {
            "id": oid,
            "traceId": "t1",
            "type": "GENERATION",
            "name": oid,
            "startTime": f"2026-01-01T00:00:0{start}Z",
            "endTime": f"2026-01-01T00:00:0{start}Z",
            "input": [{"role": "user", "content": "hi"}],
            "output": {"role": "assistant", "content": "hello"},
            "model": model,
        }

    path = _file(
        tmp_path,
        "lf.json",
        {
            "id": "t1",
            "observations": [
                generation("embed", "text-embedding-ada-002", "1"),
                generation("chat", "gpt-4o-mini", "2"),
            ],
        },
    )
    spans = load_corpus(path, "langfuse").traces[0].spans
    assert [s.name for s in spans if s.kind == SpanKind.MODEL] == ["chat"]


def _repeated_records(tmp_path: Path, source: str) -> Path:
    """One native record written twice, so its spans repeat across lines."""
    if source == "langfuse":
        upstream = Path(__file__).resolve().parents[1] / "fixtures/upstream/langfuse"
        record = json.loads((upstream / "agno-2025-06-11.trace.json").read_text())
    else:
        record = {
            "id": "11111111-1111-1111-1111-111111111111",
            "trace_id": "11111111-1111-1111-1111-111111111111",
            "run_type": "llm",
            "name": "chat",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T00:00:01Z",
            "inputs": {"messages": [{"role": "user", "content": "hello"}]},
            "outputs": {"messages": [{"role": "assistant", "content": "hi"}]},
        }
    path = tmp_path / f"{source}.jsonl"
    path.write_text("\n".join([json.dumps(record)] * 2) + "\n")
    return path


@pytest.mark.parametrize("source", ["langfuse", "langsmith"])
def test_native_issue_locations_name_the_source_file_not_the_conversion(
    tmp_path: Path, source: str
) -> None:
    # The conversion's temporary path in a stored issue made the corpus id
    # differ on every ingest of the same file.
    path = _repeated_records(tmp_path, source)
    first, second = load_corpus(path, source), load_corpus(path, source)
    located = [issue.location for issue in first.issues if issue.kind == "duplicate_span"]
    assert located
    assert all(location.startswith(f"{path}:2 observation ") for location in located)
    assert first.model_dump_json() == second.model_dump_json()


def _unmapped(span) -> dict:
    value = span.attributes.get("bandits.unmapped") or {}
    return json.loads(value) if isinstance(value, str) else value


def _every_step(corpus):
    return [s for t in corpus.traces for s in (*t.spans, *t.workflow_nodes)]


def test_fields_no_converter_knows_are_kept_not_dropped(tmp_path: Path) -> None:
    # A field nobody listed yet (a vendor adds one tomorrow) must come through
    # every native converter, in the same place.
    future = {"brand_new_field": {"nested": [1, 2]}}
    langfuse = {
        "id": "t1",
        "user_id": "u-9",
        "observations": [
            {
                "id": "o1",
                "type": "GENERATION",
                "name": "chat",
                "parentObservationId": "never-exported",
                "startTime": "2026-01-01T00:00:00Z",
                "endTime": "2026-01-01T00:00:01Z",
                "input": [{"role": "user", "content": "hi"}],
                "output": "hello",
                "usageDetails": {"input": 313, "output": 64, "total": 377},
                "modelParameters": {"temperature": 0, "top_p": 0.7},
                "costDetails": {"total": 4.1e-06},
                **future,
            }
        ],
    }
    langsmith = {
        "id": "r1",
        "run_type": "llm",
        "name": "chat",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "inputs": {"messages": [{"role": "user", "content": "hi"}]},
        "outputs": {"messages": [{"role": "assistant", "content": "hello"}]},
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "extra": {"invocation_params": {"temperature": 0.2}},
        "tags": ["prod"],
        **future,
    }
    phoenix = {
        "name": "chat",
        "context": {"trace_id": "t" * 8, "span_id": "s1"},
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "attributes": {"openinference.span.kind": "LLM", "input.value": "hi"},
        **future,
    }
    for source, record in (("langfuse", langfuse), ("langsmith", langsmith), ("phoenix", phoenix)):
        corpus = load_corpus(_file(tmp_path, f"{source}.json", record), source)
        (step,) = [s for s in _every_step(corpus) if _unmapped(s)]
        assert _unmapped(step)["brand_new_field"] == {"nested": [1, 2]}, source

    corpus = load_corpus(_file(tmp_path, "usage.json", langfuse), "langfuse")
    (step,) = _every_step(corpus)
    assert _unmapped(step)["costDetails"] == {"total": 4.1e-06}
    assert step.attributes["gen_ai.usage.input_tokens"] == 313
    assert step.attributes["gen_ai.usage.output_tokens"] == 64
    assert step.attributes["gen_ai.request.temperature"] == 0
    # Its parent was never exported: it is a top step and carries the trace's fields.
    trace_record = step.attributes["bandits.native.trace_record"]
    trace_record = json.loads(trace_record) if isinstance(trace_record, str) else trace_record
    assert trace_record["user_id"] == "u-9"

    corpus = load_corpus(_file(tmp_path, "ls.json", langsmith), "langsmith")
    (step,) = [s for s in _every_step(corpus) if _unmapped(s)]
    assert step.attributes["gen_ai.usage.input_tokens"] == 5
    assert step.attributes["gen_ai.request.temperature"] == 0.2
    assert _unmapped(step)["tags"] == ["prod"]


def test_unmapped_keeps_nulls_empties_nested_leftovers_and_odd_shapes(tmp_path: Path) -> None:
    from bandits.ingest.native import USED_FIELDS, unmapped_fields

    observation = {
        "id": "o1",
        "type": "GENERATION",
        "name": "step",
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
        "input": "q",
        "output": "a",
        "custom_null": None,
        "custom_empty": {},
        "custom_zero": 0,
        "custom_false": False,
        "metadata": "not json",  # an odd shape: kept whole, not read
        "input.value": "a field named like an attribute",  # a name collision
    }
    corpus = load_corpus(
        _file(tmp_path, "lf.json", {"id": "t", "observations": [observation]}), "langfuse"
    )
    (step,) = [s for t in corpus.traces for s in (*t.spans, *t.workflow_nodes)]
    kept = _unmapped(step)
    assert kept == {
        "custom_null": None,
        "custom_empty": {},
        "custom_zero": 0,
        "custom_false": False,
        "metadata": "not json",
        "input.value": "a field named like an attribute",
    }
    assert kept == unmapped_fields(observation, USED_FIELDS["langfuse"](observation))

    span = {
        "name": "chat",
        "context": {"trace_id": "t" * 8, "span_id": "s1", "trace_state": "vendor=1"},
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:01Z",
        "attributes": {"openinference.span.kind": "LLM", "input.value": "hi"},
        "status": {"status_code": "OK", "detail": {"retries": 2}},
        "resource": "flat-string",
    }
    corpus = load_corpus(_file(tmp_path, "px.json", {"spans": [span]}), "phoenix")
    (step,) = [s for t in corpus.traces for s in (*t.spans, *t.workflow_nodes)]
    assert _unmapped(step) == {
        "context": {"trace_state": "vendor=1"},  # the unread key inside a read object
        "status": {"detail": {"retries": 2}},
        "resource": "flat-string",  # not the object the converter reads: kept whole
    }
