from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.ingest.otlp_standard import load_otlp_standard
from bandits.traces import SpanKind, SpanStatus

TRACE = "5b8efff798038103d269b633813fc60c"
T0 = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z in unix nanoseconds


def _value(value: object) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_value(v) for v in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": _attrs(value)}}
    return {"stringValue": value}


def _attrs(attributes: dict) -> list[dict]:
    return [{"key": key, "value": _value(value)} for key, value in attributes.items()]


def _span(
    span_id: str,
    name: str,
    attributes: dict,
    *,
    parent: str | None = None,
    at: int = 0,
    trace: str = TRACE,
    **extra: object,
) -> dict:
    span = {
        "traceId": trace,
        "spanId": span_id,
        "name": name,
        "kind": 1,
        # OTLP/JSON allows either a string or a number here; both are exercised.
        "startTimeUnixNano": str(T0 + at * 1_000_000_000),
        "endTimeUnixNano": T0 + at * 1_000_000_000 + 500_000_000,
        "attributes": _attrs(attributes),
        **extra,
    }
    if parent is not None:
        span["parentSpanId"] = parent
    return span


def _request(spans: list[dict], resource: dict | None = None) -> dict:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs(resource or {"service.name": "agent"})},
                "scopeSpans": [{"scope": {"name": "test-instrumentation"}, "spans": spans}],
            }
        ]
    }


def _write(path: Path, *requests: dict) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in requests) + "\n")
    return path


def _only_trace(corpus):
    assert len(corpus.traces) == 1
    return corpus.traces[0]


def test_reads_genai_semconv_spans(tmp_path) -> None:
    messages = [{"role": "user", "parts": [{"type": "text", "content": "Refund order 7741"}]}]
    output = [{"role": "assistant", "parts": [{"type": "text", "content": "Refunded."}]}]
    path = _write(
        tmp_path / "genai.jsonl",
        _request(
            [
                _span(
                    "a1",
                    "chat gpt-5",
                    {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": "gpt-5",
                        "gen_ai.input.messages": json.dumps(messages),
                        "gen_ai.output.messages": json.dumps(output),
                        "gen_ai.request.temperature": 0.2,
                    },
                ),
                _span(
                    "a2",
                    "execute_tool refund",
                    {
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": "refund",
                        "gen_ai.tool.call.arguments": '{"order_id": "7741"}',
                        "gen_ai.tool.call.result": '{"status": "refunded"}',
                    },
                    parent="a1",
                    at=1,
                    status={"code": 2, "message": "boom"},
                ),
            ],
            resource={"service.name": "agent", "session.id": "sess-9"},
        ),
    )

    corpus = load_otlp_standard(path)

    assert corpus.source == "otlp-std"
    trace = _only_trace(corpus)
    assert trace.trace_id == TRACE
    assert trace.source == "otlp-std"
    assert trace.task == "Refund order 7741"
    assert trace.lineage_id == "sess-9"
    model, tool = trace.spans
    assert model.kind is SpanKind.MODEL
    assert model.output == "Refunded."
    assert model.attributes["gen_ai.request.temperature"] == 0.2
    assert model.started_at.isoformat() == "2026-01-01T00:00:00+00:00"
    assert tool.kind is SpanKind.TOOL
    assert tool.name == "refund"
    assert tool.parent_span_id == "a1"
    assert tool.arguments == {"order_id": "7741"}
    assert tool.output == {"status": "refunded"}
    assert tool.status is SpanStatus.ERROR
    assert tool.call_recorded is True
    assert [t.text for t in trace.user_turns] == ["Refund order 7741"]


def test_reads_openinference_flattened_messages_and_steps(tmp_path) -> None:
    llm = {
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-5",
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": "Be brief.",
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.contents.0.message_content.type": "text",
        "llm.input_messages.1.message.contents.0.message_content.text": "Weather in Paris?",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call-1",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "weather",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": (
            '{"city": "Paris"}'
        ),
    }
    path = _write(
        tmp_path / "oi.jsonl",
        _request(
            [
                _span("root", "agent", {"openinference.span.kind": "AGENT"}),
                _span(
                    "r1",
                    "retrieve",
                    {
                        "openinference.span.kind": "RETRIEVER",
                        "input.value": "Paris climate",
                        "output.value": '[{"doc": "mild"}]',
                    },
                    parent="root",
                ),
                _span("l1", "ChatOpenAI", llm, parent="root", at=1),
                _span(
                    "t1",
                    "weather",
                    {
                        "openinference.span.kind": "TOOL",
                        "tool.name": "weather",
                        "input.value": '{"city": "Paris"}',
                        "output.value": "18C",
                    },
                    parent="root",
                    at=2,
                ),
            ]
        ),
    )

    trace = _only_trace(load_otlp_standard(path))

    assert [s.span_id for s in trace.spans] == ["r1", "l1", "t1"]
    step, model, tool = trace.spans
    assert step.kind is SpanKind.TOOL
    assert step.call_recorded is False
    assert step.attributes["bandits.pipeline_step"] is True
    assert step.arguments == {"input": "Paris climate"}
    assert step.output == [{"doc": "mild"}]
    assert model.kind is SpanKind.MODEL
    assert model.attributes["bandits.input_messages_from"] == "llm.input_messages.N"
    assert model.attributes["gen_ai.output.messages"][0]["parts"][0] == {
        "type": "tool_call",
        "name": "weather",
        "arguments": '{"city": "Paris"}',
        "id": "call-1",
    }
    assert trace.task == "Weather in Paris?"
    assert tool.arguments == {"city": "Paris"}
    assert tool.output == "18C"
    assert tool.call_recorded is True


def test_reads_openllmetry_legacy_prompt_attributes(tmp_path) -> None:
    path = _write(
        tmp_path / "traceloop.jsonl",
        _request(
            [
                _span("w", "workflow", {"traceloop.span.kind": "workflow"}),
                _span(
                    "c",
                    "openai.chat",
                    {
                        "llm.request.type": "chat",
                        "gen_ai.prompt.0.role": "user",
                        "gen_ai.prompt.0.content": "List files",
                        "gen_ai.completion.0.role": "assistant",
                        "gen_ai.completion.0.content": "Running ls.",
                    },
                    parent="w",
                ),
                _span(
                    "t",
                    "ls.tool",
                    {
                        "traceloop.span.kind": "tool",
                        "traceloop.entity.name": "ls",
                        "traceloop.entity.input": '{"args": [], "kwargs": {"path": "."}}',
                        "traceloop.entity.output": '"a.py b.py"',
                    },
                    parent="w",
                    at=1,
                ),
            ]
        ),
    )

    trace = _only_trace(load_otlp_standard(path))

    model, tool = trace.spans
    assert trace.task == "List files"
    assert model.output == "Running ls."
    assert tool.name == "ls"
    assert tool.arguments == {"args": [], "kwargs": {"path": "."}}
    assert tool.output == "a.py b.py"


def test_langfuse_pipeline_steps_are_outermost_action_free_nodes(tmp_path) -> None:
    generation_input = [
        {"role": "system", "content": "Answer HR questions."},
        {"role": "user", "content": "How many leave days do I get?"},
    ]
    spans = [
        _span("root", "answer", {"langfuse.observation.type": "SPAN"}),
        _span("graph", "LangGraph", {"langfuse.observation.type": "CHAIN"}, parent="root"),
        _span(
            "classify",
            "classify",
            {
                "langfuse.observation.type": "CHAIN",
                "input.value": '{"question": "leave?"}',
            },
            parent="graph",
        ),
        _span(
            "gen",
            "CLASSIFY",
            {
                "langfuse.observation.type": "GENERATION",
                "gen_ai.request.model": "gpt-5",
                "input.value": json.dumps(generation_input),
                "output.value": '{"role": "assistant", "content": "policy"}',
            },
            parent="classify",
        ),
        _span(
            "search",
            "search_node",
            {
                "langfuse.observation.type": "CHAIN",
                "input.value": '{"query": "leave days"}',
                "output.value": '{"docs": ["20 days"]}',
            },
            parent="graph",
            at=1,
        ),
        _span("inner", "rerank", {"langfuse.observation.type": "CHAIN"}, parent="search", at=1),
        _span("score", "judge", {"langfuse.observation.type": "EVALUATOR"}, parent="root", at=2),
    ]
    path = _write(tmp_path / "langfuse.jsonl", _request(spans))

    corpus = load_otlp_standard(path)
    trace = _only_trace(corpus)

    assert [s.span_id for s in trace.spans] == ["gen", "search"]
    generation, step = trace.spans
    assert generation.output == "policy"
    assert trace.task == "How many leave days do I get?"
    assert step.name == "search_node"
    assert step.arguments == {"query": "leave days"}
    assert step.output == {"docs": ["20 days"]}
    assert step.call_recorded is False
    unrepresented = [i for i in corpus.issues if i.kind == "unrepresented_span"]
    assert [i.detail.split(" carry")[0] for i in unrepresented] == [
        "1 span(s) with langfuse.observation.type=EVALUATOR"
    ]

    without = load_otlp_standard(path, pipeline_steps=False)
    assert [s.span_id for s in _only_trace(without).spans] == ["gen"]
    assert any("1 span(s) with langfuse.observation.type=CHAIN" in i.detail for i in without.issues)


def test_openai_tool_calls_become_embedded_tool_spans_once(tmp_path) -> None:
    first_in = [{"role": "user", "content": "Weather in Paris?"}]
    first_out = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }
    second_in = [*first_in, first_out, {"role": "tool", "tool_call_id": "c1", "content": "18C"}]
    spans = [
        _span(
            "m1",
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "input.value": json.dumps(first_in),
                "output.value": json.dumps(first_out),
            },
        ),
        _span(
            "m2",
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "input.value": json.dumps(second_in),
                "output.value": "It is 18C.",
            },
            at=1,
        ),
    ]
    path = _write(tmp_path / "openai.jsonl", _request(spans))

    trace = _only_trace(load_otlp_standard(path))

    assert [s.kind for s in trace.spans] == [SpanKind.MODEL, SpanKind.TOOL, SpanKind.MODEL]
    embedded = trace.spans[1]
    assert embedded.name == "weather"
    assert embedded.arguments == {"city": "Paris"}
    assert embedded.output == "18C"
    assert embedded.parent_span_id == "m1"

    # The same call also exported as its own execute_tool span is not doubled.
    explicit = _span(
        "t1",
        "execute_tool weather",
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "weather",
            "gen_ai.tool.call.id": "c1",
            "gen_ai.tool.call.result": "18C",
        },
        parent="m1",
        at=0,
    )
    path = _write(tmp_path / "both.jsonl", _request([*spans, explicit]))
    trace = _only_trace(load_otlp_standard(path))
    assert [s.span_id for s in trace.spans if s.kind is SpanKind.TOOL] == ["t1"]


def test_reads_legacy_genai_span_events(tmp_path) -> None:
    events = [
        {
            "timeUnixNano": str(T0),
            "name": "gen_ai.system.message",
            "attributes": _attrs({"content": "Be kind."}),
        },
        {
            "timeUnixNano": str(T0),
            "name": "gen_ai.user.message",
            "attributes": _attrs({"content": "Hello"}),
        },
        {
            "timeUnixNano": str(T0),
            "name": "gen_ai.choice",
            "attributes": _attrs({"message": '{"content": "Hi there"}'}),
        },
    ]
    path = _write(
        tmp_path / "events.jsonl",
        _request([_span("m", "chat", {"gen_ai.operation.name": "chat"}, events=events)]),
    )

    trace = _only_trace(load_otlp_standard(path))

    model = trace.spans[0]
    assert trace.task == "Hello"
    assert model.output == "Hi there"
    assert model.attributes["bandits.input_messages_from"] == "events"


def test_directory_joins_a_trace_split_across_files(tmp_path) -> None:
    export = tmp_path / "export"
    (export / "nested").mkdir(parents=True)
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    _write(export / "part_000.jsonl", _request([_span("a", "chat", chat)]))
    _write(
        export / "nested" / "part_001.jsonl",
        _request(
            [
                _span("b", "chat", chat, at=1),
                _span("a", "chat", chat),  # re-exported by a retrying exporter
            ]
        ),
    )
    (export / "notes.txt").write_text("not an export")

    corpus = load_otlp_standard(export)

    trace = _only_trace(corpus)
    assert [s.span_id for s in trace.spans] == ["a", "b"]
    assert [i.kind for i in corpus.issues] == ["duplicate_span"]
    assert load_otlp_standard(export).traces[0].source_digest == trace.source_digest
    (export / "part_000.jsonl").write_text((export / "part_000.jsonl").read_text() + "\n")
    assert load_otlp_standard(export).traces[0].source_digest != trace.source_digest


def test_reads_a_pretty_printed_request_and_an_array_of_them(tmp_path) -> None:
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    single = tmp_path / "single.json"
    single.write_text(json.dumps(_request([_span("a", "chat", chat)]), indent=2))
    many = tmp_path / "many.json"
    many.write_text(
        json.dumps(
            [
                _request([_span("a", "chat", chat)]),
                _request([_span("b", "chat", chat, trace="f" * 32)]),
            ]
        )
    )

    assert len(load_otlp_standard(single).traces) == 1
    assert len(load_otlp_standard(many).traces) == 2


def test_bad_records_become_issues_not_failures(tmp_path) -> None:
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    path = tmp_path / "bad.jsonl"
    no_times = _span("t", "chat", chat)
    del no_times["endTimeUnixNano"]
    no_trace = _span("n", "chat", chat)
    del no_trace["traceId"]
    path.write_text(
        "\n".join(
            [
                "{not json",
                json.dumps({"spans": []}),
                json.dumps(_request([no_times, no_trace, _span("ok", "chat", chat)])),
                json.dumps(
                    _request([_span("h", "GET /health", {"http.method": "GET"}, trace="e" * 32)])
                ),
            ]
        )
    )

    corpus = load_otlp_standard(path)

    assert [t.trace_id for t in corpus.traces] == [TRACE]
    kinds = [i.kind for i in corpus.issues]
    assert kinds.count("malformed_json") == 1
    assert kinds.count("malformed_record") == 1
    assert kinds.count("malformed_span") == 2
    assert kinds.count("empty_trace") == 1
    assert any(
        i.kind == "unrepresented_span" and "no declared kind" in i.detail for i in corpus.issues
    )


def test_dispatch_and_cli(tmp_path) -> None:
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    path = _write(tmp_path / "t.jsonl", _request([_span("a", "chat", chat)]))

    assert load_corpus(path, "otlp-std") == load_otlp_standard(path)
    with pytest.raises(ValueError):
        load_corpus(path, "otlp", pipeline_steps=False)

    result = CliRunner().invoke(
        app, ["ingest", str(path), "--source", "otlp-std", "--project", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "traces:      1" in result.stdout
