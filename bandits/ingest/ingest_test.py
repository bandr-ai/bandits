"""Tests for stage 1 -- ingest, developed against the golden corpus.

The load-bearing test here is :func:`test_encodings_are_equivalent`. Everything else
checks that we read one encoding correctly; that one checks that the *recovery* path
(chat transcripts, PLAN.md Step 6) reconstructs exactly what an instrumented OTLP
export recorded directly. If that ever fails, every downstream stage is reasoning
about a different world depending on which export a customer happens to have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.contracts import CallStatus, Trace, TraceCorpus
from bandits.ingest import (
    CANONICAL_SOURCES,
    UnknownSourceError,
    load_corpus,
    load_corpus_and_registry,
)
from bandits.ingest.errors import infer_error_kind, looks_like_error, normalize_error_kind
from bandits.ingest.otlp import normalize_attributes, parse_otlp_record
from bandits.ingest.registry import RegistryError, load_registry, parse_registry

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
OTLP_PATH = FIXTURES / "traces.otlp.jsonl"
CHAT_PATH = FIXTURES / "traces.chat.jsonl"
TOOLS_PATH = FIXTURES / "tools.json"


@pytest.fixture(scope="module")
def expected() -> dict:
    """The ground truth every stage must reproduce (``tests/fixtures/expected.json``)."""
    return json.loads((FIXTURES / "expected.json").read_text())


@pytest.fixture(scope="module")
def otlp_corpus() -> TraceCorpus:
    """The golden corpus read through the OTLP adapter."""
    return load_corpus(OTLP_PATH, "otlp", tools_path=TOOLS_PATH)


@pytest.fixture(scope="module")
def chat_corpus() -> TraceCorpus:
    """The same episodes read through the chat-json recovery adapter."""
    return load_corpus(CHAT_PATH, "chat-json", tools_path=TOOLS_PATH)


def _signature(trace: Trace) -> list[tuple]:
    """The replayable content of a trace: ``(tool, arguments, response, status, step)``.

    Deliberately excludes latency, span ids and call ids -- those are provenance and
    legitimately differ between encodings. What must not differ is anything the
    rebuilt environment would replay.
    """
    return [
        (
            inv.tool,
            json.dumps(inv.arguments, sort_keys=True),
            json.dumps(inv.response, sort_keys=True),
            inv.status.value,
            inv.step,
        )
        for inv in trace.invocations
    ]


# ---------------------------------------------------------------- shape


def test_otlp_matches_expected_counts(otlp_corpus: TraceCorpus, expected: dict) -> None:
    """OTLP ingest reproduces the episode and invocation counts in expected.json."""
    assert len(otlp_corpus.traces) == expected["episode_count"] == 7
    total = sum(len(t.invocations) for t in otlp_corpus.traces)
    assert total == expected["invocation_count"] == 23


def test_chat_matches_expected_counts(chat_corpus: TraceCorpus, expected: dict) -> None:
    """Recovery from plain chat finds every invocation the spans recorded."""
    assert len(chat_corpus.traces) == expected["episode_count"] == 7
    total = sum(len(t.invocations) for t in chat_corpus.traces)
    assert total == expected["invocation_count"] == 23


def test_golden_corpus_ingests_without_issues(
    otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus
) -> None:
    """A well-formed export with a matching registry produces no issues at all.

    Guards the opposite failure from silent dropping: an adapter that flags healthy
    records would train reviewers to ignore the issue list.
    """
    assert otlp_corpus.issues == ()
    assert chat_corpus.issues == ()


def test_steps_are_dense_and_zero_based(otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus) -> None:
    """Steps run 0..n-1 in both encodings; a hole would break replay ordering."""
    for corpus in (otlp_corpus, chat_corpus):
        for trace in corpus.traces:
            assert [inv.step for inv in trace.invocations] == list(range(len(trace.invocations)))
            assert all(inv.trace_id == trace.trace_id for inv in trace.invocations)


# ---------------------------------------------------------------- the equivalence test


def test_encodings_are_equivalent(otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus) -> None:
    """THE test: both encodings of the same episodes yield identical invocation points.

    ``(tool, arguments, response, status, step)`` recovered from a plain chat
    transcript must be byte-for-byte the same as what the OTLP spans declared. This
    is what makes the chat recovery path trustworthy enough to build an environment
    on.
    """
    otlp_by_id = {t.trace_id: t for t in otlp_corpus.traces}
    chat_by_id = {t.trace_id: t for t in chat_corpus.traces}
    assert set(otlp_by_id) == set(chat_by_id), "episode ids must line up across encodings"

    for trace_id in sorted(otlp_by_id):
        assert _signature(otlp_by_id[trace_id]) == _signature(chat_by_id[trace_id]), (
            f"invocation points diverge for episode {trace_id}"
        )


def test_error_kinds_are_equivalent(otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus) -> None:
    """Error labels agree too, even though they are derived from different evidence.

    OTLP reads span status; chat has only the response body. Both must land on the
    same ``(status, error_kind)`` or stage 2 would infer different error modes from
    the same episodes.
    """
    otlp_by_id = {t.trace_id: t for t in otlp_corpus.traces}
    chat_by_id = {t.trace_id: t for t in chat_corpus.traces}
    for trace_id, otlp_trace in otlp_by_id.items():
        a = [(i.status, i.error_kind) for i in otlp_trace.invocations]
        b = [(i.status, i.error_kind) for i in chat_by_id[trace_id].invocations]
        assert a == b


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize("source,path", [("otlp", OTLP_PATH), ("chat-json", CHAT_PATH)])
def test_error_episodes_carry_kinds(source: str, path: Path, expected: dict) -> None:
    """not_found and already_refunded survive ingest with status=ERROR.

    The rebuilt environment has to be able to reproduce adversity; it can only do
    that if the failure modes are labeled here.
    """
    corpus = load_corpus(path, source)
    errors = [
        (inv.tool, inv.error_kind)
        for trace in corpus.traces
        for inv in trace.invocations
        if inv.status is CallStatus.ERROR
    ]
    assert sorted(errors) == [("get_order", "not_found"), ("refund_order", "already_refunded")]

    observed: dict[str, list[str]] = {}
    for tool, kind in errors:
        observed.setdefault(tool, []).append(kind)
    assert observed == expected["expected_error_modes"]


def test_successful_calls_have_no_error_kind(otlp_corpus: TraceCorpus) -> None:
    """error_kind is only ever set on failures; a kind on an OK call is noise."""
    for trace in otlp_corpus.traces:
        for inv in trace.invocations:
            if inv.status is CallStatus.OK:
                assert inv.error_kind is None


def test_infer_error_kind_is_explicit() -> None:
    """The kind inference reads declared labels and refuses to guess at anything else."""
    assert infer_error_kind({"error": "not_found", "order_id": 9999}) == "not_found"
    assert infer_error_kind({"error": "NOT FOUND"}) == "not_found"
    assert infer_error_kind({"error": {"code": "already_refunded"}}) == "already_refunded"
    assert infer_error_kind({"error": True}) is None, "a bare flag says that, not how"
    assert infer_error_kind({"ok": True}) is None
    assert infer_error_kind("something went wrong") is None
    assert normalize_error_kind("  ") is None


def test_looks_like_error_only_fires_on_declared_failure() -> None:
    """The chat adapter's status evidence must not fire on ordinary payloads."""
    assert looks_like_error({"error": "not_found"})
    assert looks_like_error({"status": "error"})
    assert not looks_like_error({"order_id": 7741, "status": "refunded"})
    assert not looks_like_error({"sent": True})


# ---------------------------------------------------------------- labels and provenance


@pytest.mark.parametrize("source,path", [("otlp", OTLP_PATH), ("chat-json", CHAT_PATH)])
def test_outcome_labels(source: str, path: Path, expected: dict) -> None:
    """5 pass / 2 fail, carried from the OTLP attribute and the chat top-level field."""
    corpus = load_corpus(path, source)
    outcomes = [t.outcome for t in corpus.traces]
    assert outcomes.count(True) == expected["labeled_pass"] == 5
    assert outcomes.count(False) == expected["labeled_fail"] == 2
    assert outcomes.count(None) == 0


def test_outcomes_agree_across_encodings(otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus) -> None:
    """The same episode carries the same label in both exports."""
    otlp_labels = {t.trace_id: t.outcome for t in otlp_corpus.traces}
    chat_labels = {t.trace_id: t.outcome for t in chat_corpus.traces}
    assert otlp_labels == chat_labels


def test_source_digest_is_sha256_of_the_record_line(otlp_corpus: TraceCorpus) -> None:
    """Each trace's digest is the sha256 of its own source line, not of the file."""
    import hashlib

    lines = [ln for ln in OTLP_PATH.read_bytes().split(b"\n") if ln.strip()]
    digests = [hashlib.sha256(ln).hexdigest() for ln in lines]
    assert [t.source_digest for t in otlp_corpus.traces] == digests
    assert len(set(digests)) == len(digests)


def test_instruction_and_answer_survive(otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus) -> None:
    """The task statement reaches the corpus in both encodings; stage 5 mines it."""
    otlp_by_id = {t.trace_id: t for t in otlp_corpus.traces}
    chat_by_id = {t.trace_id: t for t in chat_corpus.traces}
    for trace_id, trace in otlp_by_id.items():
        assert trace.instruction
        assert trace.instruction == chat_by_id[trace_id].instruction
    assert otlp_by_id["ep-refund-ok"].instruction == (
        "I want a refund for my order, my customer id is 88."
    )


def test_latency_present_on_otlp_only(otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus) -> None:
    """Spans give durations; a chat transcript honestly has none. Neither is invented."""
    assert all(i.latency_ms == 1.0 for t in otlp_corpus.traces for i in t.invocations)
    assert all(i.latency_ms is None for t in chat_corpus.traces for i in t.invocations)
    assert all(i.source_span_id for t in otlp_corpus.traces for i in t.invocations)


# ---------------------------------------------------------------- OTLP attribute encodings


def _to_verbose_attributes(value: object) -> object:
    """Re-encode plain-dict span attributes into the verbose protobuf-JSON form."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_to_verbose_attributes(v) for v in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": [
            {"key": k, "value": _to_verbose_attributes(v)} for k, v in value.items()
        ]}}
    return {"stringValue": json.dumps(value)}


def test_verbose_attributes_parse_identically(otlp_corpus: TraceCorpus, tmp_path: Path) -> None:
    """The verbose OTLP attribute encoding yields exactly the same traces as plain dicts.

    Vendors differ on this purely mechanically; if the two forms diverged, the same
    agent would produce two different environments depending on its collector.
    """
    verbose_lines = []
    for line in OTLP_PATH.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for resource in record["resourceSpans"]:
            for scope in resource["scopeSpans"]:
                for span in scope["spans"]:
                    span["attributes"] = [
                        {"key": k, "value": _to_verbose_attributes(v)}
                        for k, v in span["attributes"].items()
                    ]
        verbose_lines.append(json.dumps(record))
    verbose_path = tmp_path / "traces.verbose.jsonl"
    verbose_path.write_text("\n".join(verbose_lines) + "\n")

    verbose_corpus = load_corpus(verbose_path, "otlp")
    assert verbose_corpus.issues == ()
    assert len(verbose_corpus.traces) == len(otlp_corpus.traces)
    for plain, verbose in zip(otlp_corpus.traces, verbose_corpus.traces, strict=True):
        assert verbose.trace_id == plain.trace_id
        assert verbose.outcome == plain.outcome
        assert verbose.messages == plain.messages
        assert _signature(verbose) == _signature(plain)
        assert [i.call_id for i in verbose.invocations] == [i.call_id for i in plain.invocations]


def test_normalize_attributes_handles_both_forms() -> None:
    """normalize_attributes is the single place the two encodings converge."""
    plain = {"gen_ai.tool.name": "get_order", "retries": 2, "ok": True}
    verbose = [
        {"key": "gen_ai.tool.name", "value": {"stringValue": "get_order"}},
        {"key": "retries", "value": {"intValue": "2"}},
        {"key": "ok", "value": {"boolValue": True}},
    ]
    assert normalize_attributes(plain) == normalize_attributes(verbose) == plain
    assert normalize_attributes(None) == {}


def test_spans_are_ordered_by_start_time(tmp_path: Path) -> None:
    """Step order follows span start time, not the order spans happen to be written."""
    line = OTLP_PATH.read_text().splitlines()[0]
    record = json.loads(line)
    spans = record["resourceSpans"][0]["scopeSpans"][0]["spans"]
    shuffled = [spans[0]] + list(reversed(spans[1:]))
    record["resourceSpans"][0]["scopeSpans"][0]["spans"] = shuffled
    path = tmp_path / "shuffled.jsonl"
    path.write_text(json.dumps(record) + "\n")

    trace = load_corpus(path, "otlp").traces[0]
    assert [i.tool for i in trace.invocations] == [
        "get_customer", "search_orders", "get_order", "get_store_policy",
        "refund_order", "get_order", "send_email",
    ]


# ---------------------------------------------------------------- failure handling


def test_malformed_jsonl_line_becomes_an_issue(tmp_path: Path) -> None:
    """A broken line is reported, never dropped and never fatal.

    Rule 6 of the build plan: fail loudly. The surrounding good records still ingest,
    so one corrupt line does not cost a customer their whole export.
    """
    good = OTLP_PATH.read_text().splitlines()[:2]
    path = tmp_path / "broken.jsonl"
    path.write_text(good[0] + "\n" + "{not valid json,,,\n" + good[1] + "\n")

    corpus = load_corpus(path, "otlp")
    assert len(corpus.traces) == 2
    kinds = [i.kind for i in corpus.issues]
    assert kinds == ["malformed_json"]
    assert corpus.issues[0].location == f"{path}:1"


def test_malformed_chat_line_becomes_an_issue(tmp_path: Path) -> None:
    """The same guarantee on the recovery path."""
    good = CHAT_PATH.read_text().splitlines()[:1]
    path = tmp_path / "broken.chat.jsonl"
    path.write_text(good[0] + "\n" + "]]not json[[\n")

    corpus = load_corpus(path, "chat-json")
    assert len(corpus.traces) == 1
    assert [i.kind for i in corpus.issues] == ["malformed_json"]


def test_record_without_messages_becomes_an_issue(tmp_path: Path) -> None:
    """A chat record with no transcript is unusable and says so."""
    path = tmp_path / "empty.chat.jsonl"
    path.write_text(json.dumps({"conversation_id": "ep-x", "outcome": True}) + "\n")

    corpus = load_corpus(path, "chat-json")
    assert corpus.traces == ()
    assert [i.kind for i in corpus.issues] == ["missing_messages"]


def test_record_without_spans_becomes_an_issue(tmp_path: Path) -> None:
    """An OTLP record with no spans is likewise reported rather than yielding an empty trace."""
    path = tmp_path / "empty.otlp.jsonl"
    path.write_text(json.dumps({"resourceSpans": []}) + "\n")

    corpus = load_corpus(path, "otlp")
    assert corpus.traces == ()
    assert [i.kind for i in corpus.issues] == ["no_spans"]


def test_recovery_without_tool_call_id_records_an_issue(tmp_path: Path) -> None:
    """Falling back to name + source order is allowed, but never silent.

    This is the ambiguous branch of PLAN.md Step 6: without an id we are guessing at
    the pairing, and the corpus has to show where a guess was made.
    """
    record = json.loads(CHAT_PATH.read_text().splitlines()[0])
    for message in record["messages"]:
        if message["role"] == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                call.pop("id", None)
        if message["role"] == "tool":
            message.pop("tool_call_id", None)
    path = tmp_path / "no_ids.chat.jsonl"
    path.write_text(json.dumps(record) + "\n")

    corpus = load_corpus(path, "chat-json")
    trace = corpus.traces[0]
    reference = load_corpus(CHAT_PATH, "chat-json").traces[0]
    assert _signature(trace) == _signature(reference), "pairing must still be correct"

    kinds = {i.kind for i in corpus.issues}
    assert "recovered_by_name_and_order" in kinds
    assert "synthesized_call_id" in kinds
    assert all(i.call_id == f"ep-refund-ok:{i.step}" for i in trace.invocations)


def test_orphan_tool_result_is_reported(tmp_path: Path) -> None:
    """A tool reply that answers no known call is an issue, not a dropped record."""
    record = json.loads(CHAT_PATH.read_text().splitlines()[0])
    record["messages"].append({"role": "tool", "tool_call_id": "call_nope", "content": "{}"})
    path = tmp_path / "orphan.chat.jsonl"
    path.write_text(json.dumps(record) + "\n")

    corpus = load_corpus(path, "chat-json")
    assert [i.kind for i in corpus.issues] == ["orphan_tool_result"]
    assert len(corpus.traces[0].invocations) == 7


def test_unanswered_tool_call_is_reported(tmp_path: Path) -> None:
    """A call with no reply keeps its slot with a null response, and is flagged."""
    record = json.loads(CHAT_PATH.read_text().splitlines()[0])
    record["messages"] = [m for m in record["messages"] if m.get("role") != "tool"]
    path = tmp_path / "unanswered.chat.jsonl"
    path.write_text(json.dumps(record) + "\n")

    corpus = load_corpus(path, "chat-json")
    trace = corpus.traces[0]
    assert len(trace.invocations) == 7
    assert all(i.response is None for i in trace.invocations)
    assert [i.kind for i in corpus.issues] == ["tool_call_without_result"] * 7


def test_tool_span_without_a_name_is_reported(tmp_path: Path) -> None:
    """An execute_tool span with no tool name cannot be replayed, so it is flagged."""
    record = json.loads(OTLP_PATH.read_text().splitlines()[0])
    record["resourceSpans"][0]["scopeSpans"][0]["spans"][1]["attributes"].pop("gen_ai.tool.name")
    path = tmp_path / "nameless.otlp.jsonl"
    path.write_text(json.dumps(record) + "\n")

    corpus = load_corpus(path, "otlp")
    assert [i.kind for i in corpus.issues] == ["tool_span_missing_name"]
    trace = corpus.traces[0]
    assert len(trace.invocations) == 6
    assert [i.step for i in trace.invocations] == list(range(6))


# ---------------------------------------------------------------- source declaration


def test_unknown_source_raises() -> None:
    """The source is declared, never sniffed. An unknown name is a hard error."""
    with pytest.raises(UnknownSourceError):
        load_corpus(OTLP_PATH, "openinference")
    with pytest.raises(UnknownSourceError):
        load_corpus(OTLP_PATH, "auto")


def test_canonical_sources_are_the_supported_adapters() -> None:
    """CANONICAL_SOURCES is the contract callers dispatch on. Declared, never sniffed."""
    assert CANONICAL_SOURCES == ("otlp", "chat-json", "langsmith")


def test_declared_source_is_recorded_on_every_trace(
    otlp_corpus: TraceCorpus, chat_corpus: TraceCorpus
) -> None:
    """Trace.source is the declared adapter name, so provenance survives merging."""
    assert otlp_corpus.source == "otlp"
    assert all(t.source == "otlp" for t in otlp_corpus.traces)
    assert chat_corpus.source == "chat-json"
    assert all(t.source == "chat-json" for t in chat_corpus.traces)


# ---------------------------------------------------------------- registry


def test_registry_loads_declared_schemas(expected: dict) -> None:
    """The registry gives the full action space, including never-called tools."""
    registry = load_registry(TOOLS_PATH)
    assert list(registry) == expected["declared_tools"]
    assert registry["get_order"]["input_schema"]["required"] == ["order_id"]
    assert registry["get_order"]["description"] == "Fetch one order by id."
    assert "escalate_to_human" in registry


def test_registry_covers_every_observed_tool(otlp_corpus: TraceCorpus, expected: dict) -> None:
    """Every tool the traces call is declared, and one declared tool is never called."""
    registry = load_registry(TOOLS_PATH)
    observed = {i.tool for t in otlp_corpus.traces for i in t.invocations}
    assert observed <= set(registry)
    assert sorted(set(registry) - observed) == expected["never_called_tools"]


def test_undeclared_tool_is_reported(tmp_path: Path) -> None:
    """A tool called but not declared means export and registry disagree. Say so."""
    tools = json.loads(TOOLS_PATH.read_text())
    trimmed = [t for t in tools if t["name"] != "send_email"]
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(trimmed))

    corpus = load_corpus(OTLP_PATH, "otlp", tools_path=path)
    assert [i.kind for i in corpus.issues] == ["undeclared_tool"]
    assert "send_email" in corpus.issues[0].detail


def test_malformed_registry_entries_become_issues() -> None:
    """Entry-level registry problems are issues; a non-list payload is fatal."""
    registry, issues = parse_registry([
        {"name": "ok_tool", "input_schema": {"type": "object"}},
        {"description": "nameless"},
        "not an object",
        {"name": "ok_tool", "input_schema": {"type": "object"}},
    ])
    assert list(registry) == ["ok_tool"]
    assert [i.kind for i in issues] == [
        "tool_missing_name", "malformed_tool_entry", "duplicate_tool",
    ]
    with pytest.raises(RegistryError):
        parse_registry({"not": "a list"})


def test_load_corpus_and_registry_returns_both() -> None:
    """Stage 2 needs the corpus and the declared schemas; TraceCorpus holds only one."""
    corpus, registry = load_corpus_and_registry(OTLP_PATH, "otlp", tools_path=TOOLS_PATH)
    assert len(corpus.traces) == 7
    assert len(registry) == 9
    _, empty = load_corpus_and_registry(OTLP_PATH, "otlp")
    assert empty == {}


# ---------------------------------------------------------------- direct record parsing


def test_parse_otlp_record_is_usable_directly() -> None:
    """The per-record entry point works without going through a file."""
    record = json.loads(OTLP_PATH.read_text().splitlines()[0])
    trace, issues = parse_otlp_record(
        record, source_digest="deadbeef", location="<memory>", fallback_trace_id="fallback"
    )
    assert issues == []
    assert trace is not None
    assert trace.trace_id == "ep-refund-ok"
    assert trace.source_digest == "deadbeef"
    assert trace.metadata["otlp_trace_id"] == f"{0:032x}"
    assert len(trace.invocations) == 7
