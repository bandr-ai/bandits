from __future__ import annotations

import json
from pathlib import Path

from bandits.ingest import load_corpus
from bandits.ingest.trail import load_trail
from bandits.traces import SpanKind, SpanStatus


def _span(
    span_id: str,
    name: str,
    kind: str | None,
    ts: str,
    attrs: dict,
    children=(),
    status="Ok",
    parent=None,
):
    attributes = dict(attrs)
    if kind:
        attributes["openinference.span.kind"] = kind
    return {
        "timestamp": ts,
        "trace_id": "t1",
        "span_id": span_id,
        "parent_span_id": parent,
        "span_name": name,
        "span_attributes": attributes,
        "duration": "PT0.5S",
        "status_code": status,
        "status_message": "boom" if status == "Error" else "",
        "child_spans": list(children),
    }


def _write_trace(tmp_path: Path) -> Path:
    llm_plan = _span(
        "llm-plan",
        "LiteLLMModel.__call__",
        "LLM",
        "2025-01-01T00:00:01Z",
        {"llm.model_name": "o3-mini", "llm.output_messages.0.message.content": "Plan: search"},
    )
    llm_step = _span(
        "llm-1",
        "LiteLLMModel.__call__",
        "LLM",
        "2025-01-01T00:00:02Z",
        {
            "llm.model_name": "o3-mini",
            "llm.input_messages.0.message.content": "sys",
            "llm.input_messages.1.message.content": "obs",
            "llm.output_messages.0.message.content": "Thought: run it\nCode: x()",
        },
    )
    tool = _span(
        "tool-1",
        "SearchTool",
        "TOOL",
        "2025-01-01T00:00:03Z",
        {
            "tool.name": "web_search",
            "input.value": json.dumps({"args": [], "kwargs": {"query": "q"}}),
            "output.value": "result text",
        },
    )
    step = _span(
        "step-1",
        "Step 1",
        "CHAIN",
        "2025-01-01T00:00:02Z",
        {"output.value": "Execution logs:\nfile not found"},
        children=[llm_step, tool],
        status="Error",
    )
    agent = _span(
        "agent",
        "CodeAgent.run",
        "AGENT",
        "2025-01-01T00:00:00Z",
        {"input.value": json.dumps({"task": "Find the thing"})},
        children=[llm_plan, step],
    )
    root = _span("root", "main", None, "2025-01-01T00:00:00Z", {}, children=[agent])
    path = tmp_path / "abc123.json"
    path.write_text(json.dumps({"trace_id": "abc123", "spans": [root]}))
    return path


def test_trail_tree_becomes_action_reaction_order(tmp_path: Path) -> None:
    path = _write_trace(tmp_path)
    corpus = load_trail(tmp_path)
    assert corpus.source == "trail"
    assert len(corpus.traces) == 1
    trace = corpus.traces[0]
    assert trace.trace_id == "abc123"
    assert trace.task == "Find the thing"
    assert trace.user_turns == ()
    assert [s.span_id for s in trace.spans] == ["llm-plan", "llm-1", "tool-1", "step-1"]
    assert [s.kind for s in trace.spans] == [
        SpanKind.MODEL,
        SpanKind.MODEL,
        SpanKind.TOOL,
        SpanKind.TOOL,
    ]
    execute = trace.spans[-1]
    assert execute.name == "execute"
    assert execute.status is SpanStatus.ERROR
    assert execute.output == "Execution logs:\nfile not found"
    assert execute.started_at > trace.spans[1].started_at
    assert trace.spans[2].arguments == {"query": "q"}
    assert trace.spans[1].arguments == {"observation": "obs"}
    assert path.exists()


def test_trail_registered_as_source(tmp_path: Path) -> None:
    _write_trace(tmp_path)
    corpus = load_corpus(tmp_path, "trail")
    assert len(corpus.traces) == 1


def test_trail_malformed_file_is_an_issue_not_a_trace(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    corpus = load_trail(tmp_path)
    assert corpus.traces == ()
    assert corpus.issues[0].kind == "malformed_json"


def test_a_top_level_json_array_is_quarantined_not_raised(tmp_path: Path) -> None:
    """A well-formed JSON document that isn't the object this adapter expects
    must not crash the directory read; ``payload.get(...)`` on a list raises
    ``AttributeError`` if this isn't checked first."""
    (tmp_path / "array.json").write_text(json.dumps(["not", "an", "object"]))
    _write_trace(tmp_path)

    corpus = load_trail(tmp_path)

    assert len(corpus.traces) == 1
    assert any(issue.kind == "malformed_record" for issue in corpus.issues)


def test_a_span_missing_its_id_is_quarantined_not_raised(tmp_path: Path) -> None:
    """A malformed span tree -- here, a child with no ``span_id`` -- must
    quarantine only this file, not abort every other file in the directory."""
    broken_child = _span("child", "x", "LLM", "2025-01-01T00:00:01Z", {})
    del broken_child["span_id"]
    root = _span("root", "main", None, "2025-01-01T00:00:00Z", {}, children=[broken_child])
    (tmp_path / "broken.json").write_text(json.dumps({"trace_id": "broken", "spans": [root]}))
    _write_trace(tmp_path)

    corpus = load_trail(tmp_path)

    assert len(corpus.traces) == 1
    assert any(issue.kind == "malformed_span" for issue in corpus.issues)


def test_trail_tool_calls_render_into_output(tmp_path: Path) -> None:
    llm = _span(
        "llm-1",
        "x",
        "LLM",
        "2025-01-01T00:00:02Z",
        {
            "llm.output_messages.0.message.content": "",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "web_search",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": json.dumps(
                {"query": "q"}
            ),
        },
    )
    root = _span("root", "main", None, "2025-01-01T00:00:00Z", {}, children=[llm])
    (tmp_path / "t.json").write_text(json.dumps({"trace_id": "t", "spans": [root]}))
    trace = load_trail(tmp_path).traces[0]
    assert trace.spans[0].output == {
        "content": "",
        "tool_calls": [{"name": "web_search", "arguments": {"query": "q"}}],
    }
