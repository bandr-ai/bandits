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


def test_a_filtered_agent_root_keeps_its_episode_context(tmp_path) -> None:
    tools = [
        {
            "type": "function",
            "name": "refund",
            "description": "Refund an order",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        }
    ]
    root = _span(
        "agent",
        "invoke_agent support",
        {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.system_instructions": json.dumps(
                [{"type": "text", "content": "Refund only paid orders."}]
            ),
            "gen_ai.tool.definitions": json.dumps(tools),
            "gen_ai.request.model": "gpt-5",
        },
    )
    chat = _span(
        "c",
        "chat",
        {"gen_ai.operation.name": "chat", "input.value": "Refund 7741"},
        parent="agent",
    )
    path = _write(tmp_path / "agent.jsonl", _request([root, chat]))

    trace = _only_trace(load_otlp_standard(path))

    assert [s.span_id for s in trace.spans] == ["c"]
    assert trace.system_prompt == "Refund only paid orders."
    assert [t.name for t in trace.tools_available or ()] == ["refund"]
    assert trace.runtime_context == {"gen_ai.request.model": "gpt-5"}


def test_cyclic_parent_ids_are_issues_not_a_hang(tmp_path) -> None:
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    spans = [
        _span("self", "chat", chat, parent="self"),
        _span("x", "chain", {"openinference.span.kind": "CHAIN"}, parent="y"),
        _span("y", "chain", {"openinference.span.kind": "CHAIN"}, parent="x"),
        _span("child", "chat", chat, parent="x", at=1),
        _span("ok", "chat", chat, at=2),
    ]
    path = _write(tmp_path / "cycle.jsonl", _request(spans))

    corpus = load_otlp_standard(path)

    assert [s.span_id for s in _only_trace(corpus).spans] == ["child", "ok"]
    looped = [i for i in corpus.issues if i.kind == "malformed_span"]
    assert sorted(i.detail.split()[1] for i in looped) == ["self", "x", "y"]


# ---------- review round 2 ----------


def test_an_evaluator_declaration_wins_over_any_other_convention(tmp_path) -> None:
    judged = _span(
        "e",
        "judge",
        {
            "gen_ai.operation.name": "chat",
            "langfuse.observation.type": "EVALUATOR",
            "input.value": "Score this answer",
        },
        at=1,
    )
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    path = _write(tmp_path / "eval.jsonl", _request([_span("a", "chat", chat), judged]))

    corpus = load_otlp_standard(path)

    assert [s.span_id for s in _only_trace(corpus).spans] == ["a"]
    assert any("EVALUATOR" in i.detail for i in corpus.issues if i.kind == "unrepresented_span")


def test_nothing_beneath_an_evaluator_enters_the_corpus(tmp_path) -> None:
    spans = [
        _span("root", "run", {"openinference.span.kind": "AGENT"}),
        _span("eval", "grade", {"openinference.span.kind": "EVALUATOR"}, parent="root"),
        # An LLM-as-judge call: its prompt is the grading rubric, not the task.
        _span(
            "judge",
            "ChatOpenAI",
            {"openinference.span.kind": "LLM", "input.value": "Grade: was it right?"},
            parent="eval",
        ),
        _span(
            "work",
            "ChatOpenAI",
            {"openinference.span.kind": "LLM", "input.value": "Refund 7741"},
            parent="root",
            at=1,
        ),
    ]
    path = _write(tmp_path / "judge.jsonl", _request(spans))

    trace = _only_trace(load_otlp_standard(path))

    assert [s.span_id for s in trace.spans] == ["work"]
    assert trace.task == "Refund 7741"


def test_a_recorded_system_message_becomes_the_system_prompt(tmp_path) -> None:
    from bandits.export import build_transcript

    first = [
        {"role": "system", "content": "Refund only paid orders."},
        {"role": "user", "content": "Refund 7741"},
    ]
    spans = [
        _span(
            "m1",
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "input.value": json.dumps(first),
                "output.value": "Done.",
            },
        )
    ]
    path = _write(tmp_path / "sys.jsonl", _request(spans))

    trace = _only_trace(load_otlp_standard(path))
    messages, defects, _ = build_transcript(trace)

    assert trace.system_prompt == "Refund only paid orders."
    assert messages[0].role == "system"
    assert messages[0].content == "Refund only paid orders."
    assert defects == ()


def test_a_call_under_a_different_system_prompt_refuses_the_row(tmp_path) -> None:
    from bandits.export import build_transcript

    def call(span_id: str, system: str, at: int) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "q"}]
        return _span(
            span_id,
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "input.value": json.dumps(messages),
                "output.value": "a",
            },
            at=at,
        )

    path = _write(
        tmp_path / "two.jsonl", _request([call("m1", "Classify.", 0), call("m2", "Answer.", 1)])
    )

    trace = _only_trace(load_otlp_standard(path))
    _, defects, _ = build_transcript(trace)

    assert trace.system_prompt == "Classify."
    assert any("system prompt" in d for d in defects)


def test_an_openinference_tool_call_id_is_not_recovered_twice(tmp_path) -> None:
    first_in = [{"role": "user", "content": "Weather?"}]
    first_out = {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "function": {"name": "weather", "arguments": "{}"}}],
    }
    second_in = [*first_in, first_out, {"role": "tool", "tool_call_id": "c1", "content": "18C"}]
    spans = [
        _span(
            "m1",
            "llm",
            {
                "openinference.span.kind": "LLM",
                "input.value": json.dumps(first_in),
                "output.value": json.dumps(first_out),
            },
        ),
        _span(
            "t1",
            "weather",
            {
                "openinference.span.kind": "TOOL",
                "tool.name": "weather",
                "tool_call.id": "c1",
                "output.value": "18C",
            },
            parent="m1",
            at=1,
        ),
        _span(
            "m2",
            "llm",
            {
                "openinference.span.kind": "LLM",
                "input.value": json.dumps(second_in),
                "output.value": "18C.",
            },
            at=2,
        ),
    ]
    path = _write(tmp_path / "oi-dup.jsonl", _request(spans))

    trace = _only_trace(load_otlp_standard(path))

    assert [s.span_id for s in trace.spans if s.kind is SpanKind.TOOL] == ["t1"]


def test_a_tool_only_response_is_a_call_not_assistant_text(tmp_path) -> None:
    from bandits.export import build_transcript

    response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c9", "function": {"name": "lookup", "arguments": '{"id": 7}'}}],
    }
    spans = [
        _span(
            "m1",
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "input.value": json.dumps([{"role": "user", "content": "Find 7"}]),
                "output.value": json.dumps(response),
            },
        )
    ]
    path = _write(tmp_path / "toolonly.jsonl", _request(spans))

    trace = _only_trace(load_otlp_standard(path))
    messages, defects, _ = build_transcript(trace)

    model, call = trace.spans
    assert model.output is None
    # The call happened; its result was never recorded. Both facts are kept.
    assert call.kind is SpanKind.TOOL
    assert call.name == "lookup"
    assert call.arguments == {"id": 7}
    assert call.output is None
    assert not any(
        m.role == "assistant" and m.content and "tool_calls" in m.content for m in messages
    )
    assert any("no recorded result" in d for d in defects)


def test_pipeline_steps_are_left_out_of_a_transcript_not_fatal_to_it(tmp_path) -> None:
    from bandits.export import build_transcript

    spans = [
        _span("root", "answer", {"langfuse.observation.type": "SPAN"}),
        _span(
            "search",
            "search_node",
            {
                "langfuse.observation.type": "CHAIN",
                "input.value": '{"q": "leave"}',
                "output.value": '{"docs": ["20 days"]}',
            },
            parent="root",
        ),
        _span(
            "gen",
            "ANSWER",
            {
                "langfuse.observation.type": "GENERATION",
                "input.value": json.dumps([{"role": "user", "content": "Leave? Docs: 20 days"}]),
                "output.value": "20 days.",
            },
            parent="root",
            at=1,
        ),
    ]
    path = _write(tmp_path / "rag.jsonl", _request(spans))

    trace = _only_trace(load_otlp_standard(path))
    messages, defects, warnings = build_transcript(trace)

    assert [s.span_id for s in trace.spans] == ["search", "gen"]
    assert defects == ()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert any("search_node" in w for w in warnings)


@pytest.mark.parametrize("nanos", ["9" * 30, str(10**22), "-5", "1e18"])
def test_an_unrepresentable_timestamp_is_an_issue_not_a_crash(tmp_path, nanos) -> None:
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    bad = _span("bad", "chat", chat)
    bad["startTimeUnixNano"] = nanos
    path = _write(tmp_path / "ts.jsonl", _request([bad, _span("ok", "chat", chat)]))

    corpus = load_otlp_standard(path)

    assert [s.span_id for s in _only_trace(corpus).spans] == ["ok"]
    assert [i.kind for i in corpus.issues] == ["malformed_span"]


def test_a_non_string_kind_attribute_is_not_a_crash(tmp_path) -> None:
    odd = _span("odd", "x", {"openinference.span.kind": ["LLM"], "gen_ai.operation.name": 3})
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    path = _write(tmp_path / "odd.jsonl", _request([odd, _span("ok", "chat", chat)]))

    corpus = load_otlp_standard(path)

    assert [s.span_id for s in _only_trace(corpus).spans] == ["ok"]


def test_invalid_utf8_is_reported_not_silently_replaced(tmp_path) -> None:
    chat = {"gen_ai.operation.name": "chat", "input.value": "hi"}
    good = json.dumps(_request([_span("ok", "chat", chat)])).encode()
    bad = json.dumps(_request([_span("bad", "chat", chat, trace="a" * 32)])).encode()
    bad = bad.replace(b'"hi"', b'"h\xff"')
    path = tmp_path / "utf8.jsonl"
    path.write_bytes(good + b"\n" + bad + b"\n")

    corpus = load_otlp_standard(path)

    assert [t.trace_id for t in corpus.traces] == [TRACE]
    assert [i.kind for i in corpus.issues] == ["malformed_json"]
    assert corpus.issues[0].location.endswith(":2")


def test_truncated_json_messages_are_not_read_as_user_text(tmp_path) -> None:
    truncated = json.dumps([{"role": "user", "content": "Refund 7741 please"}])[:25]
    spans = [
        _span(
            "m",
            "chat",
            {"gen_ai.operation.name": "chat", "input.value": truncated + "...[truncated]"},
        )
    ]
    path = _write(tmp_path / "trunc.jsonl", _request(spans))

    corpus = load_otlp_standard(path)
    trace = _only_trace(corpus)

    assert trace.task is None
    assert trace.user_turns == ()
    assert any(i.kind == "unparsed_value" for i in corpus.issues)


# ---------- review round 3 ----------


def test_an_id_less_execution_is_not_recovered_again_from_history(tmp_path) -> None:
    from bandits.export import build_transcript

    first_in = [{"role": "user", "content": "Find leave policy"}]
    first_out = {
        "role": "assistant",
        "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": "{}"}}],
    }
    second_in = [
        *first_in,
        first_out,
        {"role": "tool", "tool_call_id": "c1", "content": "20 days"},
    ]
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
            "t1",
            "search",
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "search",
                "output.value": "20 days",
            },
            parent="m1",
            at=1,
        ),
        _span(
            "m2",
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "input.value": json.dumps(second_in),
                "output.value": "20 days.",
            },
            at=2,
        ),
    ]
    path = _write(tmp_path / "noid.jsonl", _request(spans))

    trace = _only_trace(load_otlp_standard(path))
    messages, _, _ = build_transcript(trace)

    assert [s.span_id for s in trace.spans if s.kind is SpanKind.TOOL] == ["t1"]
    assert [m.role for m in messages].count("tool") == 1


def test_input_the_transcript_cannot_show_refuses_the_row(tmp_path) -> None:
    from bandits.export import build_transcript

    # Retrieval evidence handed to the model as a tool-role message with no call
    # to answer. The transcript has nowhere to put it, so the row must go.
    answer_in = [
        {"role": "user", "content": "How many days?"},
        {"role": "tool", "content": "Policy: employees get 20 days."},
    ]
    spans = [
        _span("root", "rag", {"langfuse.observation.type": "SPAN"}),
        _span(
            "search",
            "retrieve",
            {
                "langfuse.observation.type": "RETRIEVER",
                "input.value": "days",
                "output.value": "Policy: employees get 20 days.",
            },
            parent="root",
        ),
        _span(
            "gen",
            "answer",
            {
                "langfuse.observation.type": "GENERATION",
                "input.value": json.dumps(answer_in),
                "output.value": "20 days.",
            },
            parent="root",
            at=1,
        ),
    ]
    path = _write(tmp_path / "evidence.jsonl", _request(spans))

    _, defects, _ = build_transcript(_only_trace(load_otlp_standard(path)))

    assert any("input" in d and "does not show" in d for d in defects)


def test_evidence_the_transcript_carries_keeps_the_row(tmp_path) -> None:
    from bandits.export import build_transcript

    answer_in = [{"role": "user", "content": "How many days?\nDocs: employees get 20 days."}]
    spans = [
        _span("root", "rag", {"langfuse.observation.type": "SPAN"}),
        _span(
            "search",
            "retrieve",
            {"langfuse.observation.type": "RETRIEVER", "output.value": "20 days"},
            parent="root",
        ),
        _span(
            "gen",
            "answer",
            {
                "langfuse.observation.type": "GENERATION",
                "input.value": json.dumps(answer_in),
                "output.value": "20 days.",
            },
            parent="root",
            at=1,
        ),
    ]
    path = _write(tmp_path / "carried.jsonl", _request(spans))

    messages, defects, _ = build_transcript(_only_trace(load_otlp_standard(path)))

    assert defects == ()
    assert messages[0].content == "How many days?\nDocs: employees get 20 days."


# ---------- review round 4 ----------


def _chat_call(span_id: str, messages: list[dict], output: object, at: int) -> dict:
    return _span(
        span_id,
        "chat",
        {
            "gen_ai.operation.name": "chat",
            "input.value": json.dumps(messages),
            "output.value": output if isinstance(output, str) else json.dumps(output),
        },
        at=at,
    )


def test_an_omitted_example_is_not_excused_by_matching_words(tmp_path) -> None:
    from bandits.export import build_transcript

    # A few-shot assistant turn no span produced. Its text also occurs inside
    # the user's question, which must not count as the example being shown.
    call = [
        {"role": "assistant", "content": "20 days"},
        {"role": "user", "content": "Is it 20 days of leave?"},
    ]
    path = _write(tmp_path / "fewshot.jsonl", _request([_chat_call("m", call, "Yes.", 0)]))

    _, defects, _ = build_transcript(_only_trace(load_otlp_standard(path)))

    assert any("does not show" in d for d in defects)


def test_a_repeated_turn_must_appear_as_often_as_recorded(tmp_path) -> None:
    from bandits.export import build_transcript

    second = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "go"},
    ]
    spans = [
        _chat_call("m1", [{"role": "user", "content": "go"}], "done", 0),
        _chat_call("m2", second, "done", 1),
    ]
    path = _write(tmp_path / "repeat.jsonl", _request(spans))

    _, defects, _ = build_transcript(_only_trace(load_otlp_standard(path)))

    assert any("does not show" in d for d in defects)


def test_a_faithful_agent_loop_keeps_its_row(tmp_path) -> None:
    from bandits.export import build_transcript

    system = {"role": "system", "content": "Use tools."}
    first_in = [system, {"role": "user", "content": "Weather in Paris?"}]
    first_out = {
        "role": "assistant",
        "content": "Checking.",
        "tool_calls": [
            {"id": "c1", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}}
        ],
    }
    second_in = [
        *first_in,
        first_out,
        {"role": "tool", "tool_call_id": "c1", "content": "18C"},
        {"role": "user", "content": "And tomorrow?"},
    ]
    spans = [
        _chat_call("m1", first_in, first_out, 0),
        _chat_call("m2", second_in, "Also mild.", 1),
    ]
    path = _write(tmp_path / "loop.jsonl", _request(spans))

    messages, defects, _ = build_transcript(_only_trace(load_otlp_standard(path)))

    assert defects == ()
    assert [m.role for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
