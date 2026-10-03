from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.ingest import detect_source, load_corpus
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


def test_events_kept_on_a_dropped_trace_are_counted_as_lost(tmp_path: Path) -> None:
    report = IngestReport()
    corpus = load_corpus(
        _write(tmp_path, [_event(1, "error", 1, error_type="E", message="m")]),
        "failproofai",
        report=report,
    )
    assert not corpus.traces
    assert report.accounting_errors == []
    assert report.buckets["kept_on_trace"] == 0
    assert report.buckets["unconvertible"] == 1
    assert not any(i.kind == "record_kept_on_trace" for i in corpus.issues)


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


def test_v2_string_ids_and_event_count(tmp_path: Path) -> None:
    events = [
        _event("e1", "model_request", 1, request_id="r", messages=[]),
        _event("e2", "model_response", 2, request_id="r", content="c"),
    ]
    _, trace = _load(_write(tmp_path, events, schema_version="2", event_count=2))
    assert [s.kind for s in trace.spans if s.kind == SpanKind.MODEL] == [SpanKind.MODEL]
    bad = load_corpus(_write(tmp_path, events, schema_version="2", event_count=3), "failproofai")
    assert not bad.traces
    assert any("event_count is 3" in i.detail for i in bad.issues)


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
