"""Every documented export shape of each provider reads to the same traces.

The Langfuse UI and blob-storage exports write one observation per line, not a
trace with ``observations[]``; LangSmith and Phoenix export flat runs and spans
too. Each shape must give the traces its nested twin gives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.ingest import load_corpus
from bandits.ingest.detect import detect_source

_OBSERVATIONS = [
    {
        "id": "o1",
        "type": "SPAN",
        "name": "agent",
        "startTime": "2026-10-01T10:00:00.000Z",
        "endTime": "2026-10-01T10:00:03.000Z",
        "input": "where is my order?",
        "output": "It ships tomorrow.",
    },
    {
        "id": "o2",
        "parentObservationId": "o1",
        "type": "GENERATION",
        "name": "chat",
        "model": "gpt-4o",
        "startTime": "2026-10-01T10:00:00.100Z",
        "endTime": "2026-10-01T10:00:01.000Z",
        "input": [{"role": "user", "content": "where is my order?"}],
        "output": {"role": "assistant", "content": "let me check"},
        "usageDetails": {"input": 5, "output": 3, "total": 8},
    },
    {
        "id": "o3",
        "parentObservationId": "o1",
        "type": "TOOL",
        "name": "lookup",
        "startTime": "2026-10-01T10:00:01.000Z",
        "endTime": "2026-10-01T10:00:02.000Z",
        "input": {"order": "A1"},
        "output": {"eta": "tomorrow"},
    },
]


def _trace(trace_id: str, offset: str = "") -> dict:
    return {
        "id": trace_id,
        "name": "support-agent",
        "observations": [
            {
                **o,
                "id": o["id"] + offset,
                "parentObservationId": (o.get("parentObservationId") or "")
                + (offset if o.get("parentObservationId") else "")
                or None,
            }
            for o in _OBSERVATIONS
        ],
    }


def _flat_lines(trace_id: str, offset: str = "", *, snake: bool = False) -> list[dict]:
    lines = []
    for o in _trace(trace_id, offset)["observations"]:
        line = {**o, "traceId": trace_id, "traceName": "support-agent"}
        if snake:
            line = {
                "trace_id": trace_id,
                "trace_name": "support-agent",
                **{
                    {
                        "parentObservationId": "parent_observation_id",
                        "startTime": "start_time",
                        "endTime": "end_time",
                        "usageDetails": "usage_details",
                        "model": "provided_model_name",
                    }.get(k, k): v
                    for k, v in o.items()
                },
            }
        lines.append(line)
    return lines


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _shape(corpus) -> list[tuple[str, int]]:
    return sorted((t.trace_id, len(t.spans)) for t in corpus.traces)


def _nested(tmp_path: Path, ids: list[str]) -> list[tuple[str, int]]:
    path = _write_jsonl(tmp_path / "nested.jsonl", [_trace(i) for i in ids])
    return _shape(load_corpus(path, "langfuse"))


@pytest.mark.parametrize("snake", [False, True], ids=["ui-camel", "blob-snake"])
def test_flat_langfuse_lines_read_like_nested(tmp_path: Path, snake: bool) -> None:
    path = _write_jsonl(tmp_path / "flat.jsonl", _flat_lines("t1", snake=snake))
    corpus = load_corpus(path, "langfuse")
    assert len(corpus.traces) == 1
    # The root observation becomes the request, not a step: 3 observations, 2 steps.
    assert _shape(corpus)[0][1] == 2
    assert _shape(corpus) == _nested(tmp_path, ["t1"])
    assert not [i for i in corpus.issues if i.kind not in ("redaction",)]


def test_flat_langfuse_interleaved_traces_are_grouped(tmp_path: Path) -> None:
    a, b = _flat_lines("tA"), _flat_lines("tB", "-b")
    interleaved = [x for pair in zip(a, b, strict=True) for x in pair]
    path = _write_jsonl(tmp_path / "flat.jsonl", interleaved)
    corpus = load_corpus(path, "langfuse")
    assert [n for _, n in _shape(corpus)] == [2, 2]
    assert len(corpus.traces) == 2


def test_flat_langfuse_json_array_and_autodetect(tmp_path: Path) -> None:
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(_flat_lines("t1")))
    assert detect_source(path).source == "langfuse"
    assert _shape(load_corpus(path, "langfuse"))[0][1] == 2


def test_flat_jsonl_autodetects_as_langfuse(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "flat.jsonl", _flat_lines("t1"))
    assert detect_source(path).source == "langfuse"


def test_nested_and_flat_agree_on_shape(tmp_path: Path) -> None:
    flat = _write_jsonl(tmp_path / "flat.jsonl", _flat_lines("t1") + _flat_lines("t2", "-2"))
    assert _shape(load_corpus(flat, "langfuse")) == _nested(tmp_path, ["t1", "t2"])


def test_flat_lines_without_a_trace_id_are_refused_not_dropped_silently(tmp_path: Path) -> None:
    rows = [{k: v for k, v in o.items()} for o in _OBSERVATIONS]
    path = _write_jsonl(tmp_path / "noid.jsonl", rows)
    corpus = load_corpus(path, "langfuse")
    assert not corpus.traces
    assert any(i.kind == "unsupported_native_record" for i in corpus.issues)


# --- LangSmith: nested child_runs, flat runs (one per line), and a JSON array ----------------


def _runs(trace: str, suffix: str = "") -> list[dict]:
    base = {
        "trace_id": trace,
        "start_time": "2026-10-01T10:00:00Z",
        "end_time": "2026-10-01T10:00:02Z",
    }
    root = {
        **base,
        "id": f"r1{suffix}",
        "name": "agent",
        "run_type": "chain",
        "inputs": {"q": "hi"},
        "outputs": {"a": "yo"},
    }
    llm = {
        **base,
        "id": f"r2{suffix}",
        "parent_run_id": root["id"],
        "name": "llm",
        "run_type": "llm",
        "inputs": {"messages": [{"role": "user", "content": "hi"}]},
        "outputs": {"text": "yo"},
    }
    tool = {
        **base,
        "id": f"r3{suffix}",
        "parent_run_id": root["id"],
        "name": "t",
        "run_type": "tool",
        "inputs": {"x": 1},
        "outputs": {"y": 2},
    }
    return [root, llm, tool]


def test_langsmith_flat_runs_nested_and_array_agree(tmp_path: Path) -> None:
    runs = _runs("0b0b0b0b-0000-4000-8000-000000000001")
    nested = {**runs[0], "child_runs": runs[1:]}
    a = _write_jsonl(tmp_path / "nested.jsonl", [nested])
    b = _write_jsonl(tmp_path / "flat.jsonl", runs)
    c = tmp_path / "array.json"
    c.write_text(json.dumps(runs))
    shapes = [_shape(load_corpus(p, "langsmith")) for p in (a, b, c)]
    assert shapes[0] == shapes[1] == shapes[2]
    assert len(shapes[0]) == 1


def test_langsmith_flat_interleaved_traces(tmp_path: Path) -> None:
    one, two = (
        _runs("0b0b0b0b-0000-4000-8000-000000000001"),
        _runs("0b0b0b0b-0000-4000-8000-000000000002", "x"),
    )
    rows = [x for pair in zip(one, two, strict=True) for x in pair]
    corpus = load_corpus(_write_jsonl(tmp_path / "flat.jsonl", rows), "langsmith")
    assert len(corpus.traces) == 2


# --- Phoenix: span per line, getSpans data[], spans[] wrapper ------------------------------


def _phoenix_spans(trace: str, suffix: str = "") -> list[dict]:
    def span(sid: str, parent: str | None, kind: str, name: str) -> dict:
        return {
            "name": name,
            "context": {"trace_id": trace, "span_id": sid + suffix},
            "parent_id": None if parent is None else parent + suffix,
            "span_kind": kind,
            "start_time": "2026-10-01T10:00:00.000Z",
            "end_time": "2026-10-01T10:00:01.000Z",
            "status_code": "OK",
            "attributes": {
                "openinference.span.kind": kind,
                "input.value": "hi",
                "output.value": "yo",
            },
        }

    return [
        span("a1", None, "AGENT", "agent"),
        span("a2", "a1", "LLM", "llm"),
        span("a3", "a1", "TOOL", "tool"),
    ]


def test_phoenix_per_line_data_wrapper_and_spans_wrapper_agree(tmp_path: Path) -> None:
    spans = _phoenix_spans("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b01")
    lines = _write_jsonl(tmp_path / "lines.jsonl", spans)
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"data": spans}))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"spans": spans}))
    shapes = [_shape(load_corpus(p, "phoenix")) for p in (lines, data, wrapped)]
    assert shapes[0] == shapes[1] == shapes[2]
    assert len(shapes[0]) == 1


# --- Langfuse: saved API responses and tree exports ----------------------------------------


def test_langfuse_api_list_response_reads_each_trace(tmp_path: Path) -> None:
    path = tmp_path / "api.json"
    path.write_text(json.dumps({"data": [_trace("t1"), _trace("t2", "-2")], "meta": {"page": 1}}))
    assert detect_source(path).source == "langfuse"
    assert _shape(load_corpus(path, "langfuse")) == _nested(tmp_path, ["t1", "t2"])


def test_langfuse_api_observations_response_reads_flat_observations(tmp_path: Path) -> None:
    path = tmp_path / "obs.json"
    path.write_text(json.dumps({"data": _flat_lines("t1"), "meta": {}}))
    assert detect_source(path).source == "langfuse"
    assert _shape(load_corpus(path, "langfuse")) == _nested(tmp_path, ["t1"])


def test_langfuse_trace_list_of_observation_ids_says_what_to_export(tmp_path: Path) -> None:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"data": [{"id": "t1", "observations": ["o1", "o2"]}]}))
    corpus = load_corpus(path, "langfuse")
    assert not corpus.traces
    (issue,) = [i for i in corpus.issues if i.kind == "unsupported_native_record"]
    assert "observation ids only" in issue.detail


def test_langfuse_children_tree_matches_flat(tmp_path: Path) -> None:
    obs = _trace("t1")["observations"]
    tree = {"id": "t1", "observations": [{**obs[0], "children": obs[1:]}]}
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(tree))
    assert _shape(load_corpus(path, "langfuse")) == _nested(tmp_path, ["t1"])
