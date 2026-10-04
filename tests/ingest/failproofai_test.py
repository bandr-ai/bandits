from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.ingest import detect_source, load_corpus
from bandits.ingest.health import check
from bandits.ingest.report import IngestReport
from bandits.inspect import check_fidelity
from bandits.store import ArtifactStore
from bandits.traces import SpanKind, SpanStatus, WorkflowDeclaration

FIXTURE = Path(__file__).parents[1] / "fixtures" / "failproofai.session.json"


def _ts(second: int) -> str:
    return f"2026-05-10T12:00:{second:02d}.000000Z"


def _event(event_id: object, event_type: str, second: int, **payload: object) -> dict:
    return {
        "id": event_id,
        "ts": _ts(second),
        "event_type": event_type,
        "payload": {"session_id": "s1", "agent_id": "a", "type": event_type, **payload},
    }


ANCHOR = [
    _event(900, "model_request", 50, request_id="anchor", messages=[]),
    _event(901, "model_response", 51, request_id="anchor", content="ok"),
]
"""A proven pair, so a trace under test has an action and is kept."""


def _write(tmp_path: Path, events: list[dict], **top: object) -> Path:
    document = {
        "schema_version": "1",
        "session_id": "s1",
        "agent_id": "a",
        "environment": "test",
        "started_at": _ts(0),
        "ended_at": _ts(59),
        "events": events,
        **top,
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(document))
    return path


def _load(path: Path, report: IngestReport | None = None):
    corpus = load_corpus(path, "failproofai", report=report)
    assert len(corpus.traces) == 1
    return corpus, corpus.traces[0]


def _kept(trace) -> dict[object, str]:
    return {
        entry["event"]["id"]: entry["reason"]
        for entry in (trace.source_record or {}).get("unpaired_events", [])
    }


def test_fixture_is_detected() -> None:
    assert detect_source(FIXTURE).source == "failproofai"


def test_fixture_accounts_for_every_event() -> None:
    report = IngestReport()
    corpus, trace = _load(FIXTURE, report)
    assert report.accounting_errors == []
    assert report.dropped == 0
    assert report.buckets["kept_on_trace"] == 11
    kept = _kept(trace)
    # Ambiguous correlation_id: two requests share it, so none of the three pairs.
    assert {110, 111, 112} <= set(kept)
    assert "share correlation_id 'c-dup'" in kept[110]
    # Standalone and undocumented events are kept, never attached to a span.
    assert "standalone" in kept[121] and "standalone" in kept[122]
    assert "not in FailproofAI's documented catalog" in kept[124]
    # A tool_result whose tool_use is absent.
    assert "0 tool_use and 1 tool_result" in kept[123]
    # Paired, but no span kind represents them: kept whole, as recorded.
    human_input = next(
        e["event"]
        for e in trace.source_record["unpaired_events"]
        if e["event"]["event_type"] == "human_input"
    )
    assert human_input["payload"]["response"] == "yes"
    assert "paired by input_id" in kept[118]
    assert any(i.kind == "record_kept_on_trace" for i in corpus.issues)


def test_pairs_name_their_basis() -> None:
    _, trace = _load(FIXTURE)
    basis = {
        (span.kind, span.attributes.get("failproofai.pairing"))
        for span in trace.spans
        if span.attributes.get("failproofai.pairing")
    }
    assert (SpanKind.MODEL, "request_id") in basis
    assert (SpanKind.MODEL, "source_extension:correlation_id") in basis
    assert (SpanKind.TOOL, "tool_call_id") in basis


def test_overlapping_calls_pair_by_correlation_id_not_arrival_order() -> None:
    _, trace = _load(FIXTURE)
    by_opener = {
        json.loads(span.attributes["bandits.source.record"])["observation_id"]: span
        for span in trace.spans
        if span.attributes.get("failproofai.pairing") == "source_extension:correlation_id"
    }
    # Request 106 (c-A) was answered by 109, after 107 (c-B) was answered by 108.
    assert by_opener["106"].attributes["failproofai.closer_event_id"] == 109
    assert by_opener["107"].attributes["failproofai.closer_event_id"] == 108


def test_documented_id_is_never_overridden_by_correlation_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            _event(1, "model_request", 1, request_id="r1", correlation_id="c1", messages=[]),
            _event(2, "model_response", 2, correlation_id="c1", content="hi"),
            *ANCHOR,
        ],
    )
    _, trace = _load(path)
    assert len([s for s in trace.spans if s.kind == SpanKind.MODEL]) == 1
    assert set(_kept(trace)) == {1, 2}


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        (
            [
                _event(1, "model_request", 5, request_id="r1", messages=[]),
                _event(2, "model_response", 4, request_id="r1", content="x"),
            ],
            "before its model_request",
        ),
        (
            [
                _event(1, "tool_use", 1, tool_name="a", tool_call_id="t1"),
                _event(2, "tool_result", 2, tool_name="b", tool_call_id="t1"),
            ],
            "joins different tool_names",
        ),
    ],
)
def test_incompatible_halves_stay_unpaired(tmp_path: Path, events: list, reason: str) -> None:
    _, trace = _load(_write(tmp_path, [*events, *ANCHOR]))
    kept = _kept(trace)
    assert set(kept) == {1, 2}
    assert all(reason in r for r in kept.values())


def _session(path: Path, report: IngestReport | None = None):
    """The one trace of a session that recorded no model or tool call."""
    corpus, trace = _load(path, report)
    assert trace.spans == ()
    return corpus, trace


def test_a_session_without_calls_is_kept_with_its_events(tmp_path: Path) -> None:
    report = IngestReport()
    path = _write(tmp_path, [_event(1, "error", 1, error_type="E", message="m")])
    _, trace = _session(path, report)
    assert report.accounting_errors == []
    assert report.dropped == 0
    assert report.buckets["kept_on_trace"] == 1
    assert "standalone" in _kept(trace)[1]
    assert trace.source_record["session_id"] == "s1"


def test_v1_contract_example_with_null_end_is_kept(tmp_path: Path) -> None:
    # The agenteye-evaluator wire-format example: one agent_start, ended_at null.
    events = [_event(1, "agent_start", 0, goal="help the user")]
    path = _write(tmp_path, events, ended_at=None)
    assert detect_source(path).source == "failproofai"
    _, trace = _session(path)
    assert trace.source_record["ended_at"] is None  # never an invented end
    assert "1 agent_start and 0 agent_end" in _kept(trace)[1]
    from bandits.inspect import _trace_view

    assert _trace_view(trace)["record"] == trace.source_record  # what inspect shows


def test_v1_ended_at_may_be_absent(tmp_path: Path) -> None:
    path = _write(tmp_path, [])
    document = json.loads(path.read_text())
    del document["ended_at"]
    path.write_text(json.dumps(document))
    _, trace = _session(path)
    assert "ended_at" not in trace.source_record


def test_empty_session_is_detected_and_kept(tmp_path: Path) -> None:
    path = _write(tmp_path, [])
    assert detect_source(path).source == "failproofai"
    _, trace = _session(path)
    assert "unpaired_events" not in trace.source_record


def test_lifecycle_pair_is_kept_on_the_trace_without_a_session_span(tmp_path: Path) -> None:
    events = [_event(1, "agent_start", 1, goal="g"), _event(2, "agent_end", 2, outcome="done")]
    _, trace = _session(_write(tmp_path, events, ended_at=None))
    kept = _kept(trace)
    assert "paired by agent_id with event 2" in kept[1]
    assert "paired by agent_id with event 1" in kept[2]


def test_lifecycle_pair_with_an_end_is_a_step(tmp_path: Path) -> None:
    events = [_event(1, "agent_start", 1, goal="g"), _event(2, "agent_end", 2, outcome="done")]
    corpus = load_corpus(_write(tmp_path, events), "failproofai", workflow=WorkflowDeclaration())
    (step,) = corpus.traces[0].spans
    assert (step.name, step.attributes["failproofai.pairing"]) == ("a", "agent_id")


def test_null_payload_is_kept_with_its_reason(tmp_path: Path) -> None:
    events = [*ANCHOR, {"id": 1, "ts": _ts(1), "event_type": "agent_end", "payload": None}]
    path = _write(tmp_path, events)
    assert detect_source(path).source == "failproofai"
    _, trace = _load(path)
    assert _kept(trace)[1] == "invalid v1 event: payload must be an object (it is null)"


def test_null_end_keeps_calls_without_a_session_span(tmp_path: Path) -> None:
    report = IngestReport()
    _, trace = _load(_write(tmp_path, ANCHOR, ended_at=None), report)
    assert [s.kind for s in trace.spans] == [SpanKind.MODEL]
    assert trace.source_record["ended_at"] is None
    assert report.accounting_errors == []
    assert report.dropped == 0


@pytest.mark.parametrize(
    ("top", "detail"),
    [
        ({"agent_id": None}, "v1 transcript requires agent_id"),
        ({"started_at": "soon"}, "started_at 'soon' is not a time"),
        ({"ended_at": "later"}, "ended_at 'later' is neither a time nor null"),
        ({"events": None}, "requires events[]"),
    ],
)
def test_v1_envelope_contract(tmp_path: Path, top: dict, detail: str) -> None:
    corpus = load_corpus(_write(tmp_path, top.pop("events", ANCHOR), **top), "failproofai")
    assert not corpus.traces
    assert any(detail in i.detail for i in corpus.issues)


def test_v1_event_ids_are_integers(tmp_path: Path) -> None:
    events = [*ANCHOR, _event("e1", "error", 1, error_type="E", message="m")]
    _, trace = _load(_write(tmp_path, events))
    assert _kept(trace)["e1"] == "invalid v1 event: id must be an integer"


def test_failed_outcome_marks_the_agent_span(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            _event(1, "agent_start", 1, goal="g"),
            _event(2, "model_request", 2, request_id="r", messages=[]),
            _event(3, "model_response", 3, request_id="r", content="c"),
            _event(4, "agent_end", 4, outcome="timeout"),
        ],
    )
    corpus = load_corpus(path, "failproofai", workflow=WorkflowDeclaration())
    agent = next(
        n for n in corpus.traces[0].workflow_nodes if n.attributes.get("failproofai.pairing")
    )
    assert agent.status == SpanStatus.ERROR


V2 = {
    "schema_version": "2",
    "assignment_id": "as1",
    "session_revision_id": "rev1",
}


def test_v2_string_ids_and_event_count(tmp_path: Path) -> None:
    events = [
        _event("e1", "model_request", 1, request_id="r", messages=[]),
        _event("e2", "model_response", 2, request_id="r", content="c"),
    ]
    _, trace = _load(_write(tmp_path, events, **V2, event_count=2))
    assert [s.kind for s in trace.spans if s.kind == SpanKind.MODEL] == [SpanKind.MODEL]
    bad = load_corpus(_write(tmp_path, events, **V2, event_count=3), "failproofai")
    assert not bad.traces
    assert any("event_count is 3" in i.detail for i in bad.issues)


@pytest.mark.parametrize(
    ("top", "detail"),
    [
        ({"assignment_id": None}, "v2 transcript requires assignment_id"),
        ({"ended_at": None}, "v2 transcript requires ended_at"),
        ({"event_count": None}, "requires event_count as an integer"),
    ],
)
def test_v2_envelope_contract(tmp_path: Path, top: dict, detail: str) -> None:
    events = [_event("e1", "error", 1, error_type="E", message="m")]
    corpus = load_corpus(_write(tmp_path, events, **{**V2, "event_count": 1, **top}), "failproofai")
    assert not corpus.traces
    assert any(detail in i.detail for i in corpus.issues)


def test_v2_event_ids_are_strings(tmp_path: Path) -> None:
    events = [_event(7, "error", 1, error_type="E", message="m")]
    _, trace = _session(_write(tmp_path, events, **V2, event_count=1))
    assert _kept(trace)[7] == "invalid v2 event: id must be a string"


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    corpus = load_corpus(_write(tmp_path, [], schema_version="9"), "failproofai")
    assert not corpus.traces
    assert any(i.kind == "unsupported_native_record" for i in corpus.issues)


def test_pointers_resolve_and_fields_are_kept_exactly(tmp_path: Path) -> None:
    corpus = load_corpus(FIXTURE, "failproofai", workflow=WorkflowDeclaration())
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(FIXTURE))
    fidelity = check_fidelity(store, artifact.artifact_id, corpus.traces, "failproofai")
    assert fidelity["steps"] > 0
    assert fidelity["carried_wrong"] == 0, fidelity["mismatches"]
    # The producer's own fields stay under their names.
    model = next(s for s in corpus.traces[0].spans if s.name == "model-a")
    closer = json.loads(model.attributes["failproofai.closer_unmapped"])
    assert closer["payload"]["fw_cost"] == 0.01


def test_a_session_without_model_calls_is_saved_with_a_warning(tmp_path: Path) -> None:
    corpus = load_corpus(_write(tmp_path, []), "failproofai")
    health = check(corpus, "failproofai")
    assert health.fatal == []
    assert any("record no model call" in w for w in health.warnings)


def test_other_sources_without_model_calls_still_refuse(tmp_path: Path) -> None:
    # A tool call alone: kept as a trace, but its source may hold model calls
    # in a convention not recognized, so the file is refused, not saved empty.
    span = {
        "traceId": "0" * 31 + "1",
        "spanId": "0" * 15 + "1",
        "name": "search",
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000001000000000",
        "attributes": [{"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}}],
    }
    path = tmp_path / "otlp.json"
    path.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}))
    corpus = load_corpus(path, "otlp-std")
    assert len(corpus.traces) == 1
    assert any("no model calls were found" in f for f in check(corpus, "otlp-std").fatal)


def test_a_pair_across_agents_has_no_owner(tmp_path: Path) -> None:
    events = [
        _event(1, "agent_start", 1, goal="g"),
        _event(2, "model_request", 2, request_id="r", messages=[]),
        {**_event(3, "model_response", 3, request_id="r", content="c")},
        _event(4, "agent_end", 4, outcome="done"),
    ]
    events[2]["payload"]["agent_id"] = "b"
    _, trace = _load(_write(tmp_path, events))
    model = next(s for s in trace.spans if s.kind == SpanKind.MODEL)
    agent = next(s for s in trace.spans if s.attributes.get("failproofai.pairing") == "agent_id")
    assert model.parent_span_id != agent.span_id  # under the session, not agent a
    opener = json.loads(model.attributes["bandits.unmapped"])["payload"]["agent_id"]
    closer = json.loads(model.attributes["failproofai.closer_unmapped"])["payload"]["agent_id"]
    assert (opener, closer) == ("a", "b")


def test_each_model_call_keeps_the_tools_it_was_offered(tmp_path: Path) -> None:
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    path = _write(
        tmp_path,
        [
            _event(1, "model_request", 1, request_id="r1", messages=[], tools=tools),
            _event(2, "model_response", 2, request_id="r1", content="ok"),
            _event(3, "model_request", 3, request_id="r2", messages=[]),
            _event(4, "model_response", 4, request_id="r2", content="ok"),
        ],
    )
    _, trace = _load(path)
    offered = [
        s.attributes.get("gen_ai.request.tools") for s in trace.spans if s.kind is SpanKind.MODEL
    ]
    assert [json.loads(o) if o else None for o in offered] == [tools, None]


def test_an_unresolved_task_keeps_tentative_clues_not_a_task(tmp_path: Path) -> None:
    prompt = [{"role": "system", "content": "s"}, {"role": "human", "content": " Fix step 4 "}]
    path = _write(
        tmp_path,
        [
            _event(1, "agent_start", 1, goal="analyze_failure"),
            _event(2, "model_request", 2, request_id="r1", messages=prompt),
            _event(3, "model_response", 3, request_id="r1", content="done"),
            _event(4, "agent_end", 4, outcome="success"),
        ],
    )
    corpus = load_corpus(path, "failproofai", workflow=WorkflowDeclaration())
    request = corpus.traces[0].request
    assert request.task is None and request.task_status == "unresolved"
    assert [(t.clue, t.value) for t in request.tentative_tasks] == [
        ("step_input", "analyze_failure"),
        ("first_model_prompt", "Fix step 4"),
    ]


def test_a_pointer_survives_an_escaped_secret_after_a_blank_line(tmp_path: Path) -> None:
    key = "sk-ant-api03-" + "A" * 48
    events = [
        _event(1, "model_request", 1, request_id="r", messages=[{"role": "user", "content": key}])
    ]
    events.append(_event(2, "model_response", 2, request_id="r", content="ok"))
    path = _write(tmp_path, events)
    # One line after a blank one, the secret hidden behind a JSON escape.
    path.write_text("\n" + path.read_text().replace("sk-ant", "sk\\u002dant") + "\n")
    corpus = load_corpus(path, "failproofai")
    store = ArtifactStore(tmp_path / ".bandits")
    artifact = store.write(corpus, source_path=str(path))
    pointers = [
        s.attributes["bandits.source.record"]
        for s in corpus.traces[0].spans
        if s.kind is SpanKind.MODEL
    ]
    record = store.read_native_record(artifact.artifact_id, pointers[0])
    assert record["id"] == 1 and key not in json.dumps(record)
