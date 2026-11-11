from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.ingest import load_corpus
from bandits.ingest.detect import DetectionError, detect_source

FIXTURES = Path(__file__).parents[1] / "fixtures"


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


_CONVERSATION = [
    {"role": "user", "content": "Refund order 7741"},
    {"role": "assistant", "content": "Done."},
]


@pytest.mark.parametrize(
    "text",
    [
        json.dumps([_CONVERSATION, _CONVERSATION]),
        "\n".join(json.dumps(c) for c in (_CONVERSATION, _CONVERSATION)),
        "\n".join(json.dumps({"id": str(i), "messages": _CONVERSATION}) for i in range(2)),
        "\n".join(json.dumps(message) for message in _CONVERSATION),
    ],
    ids=["array-of-arrays", "jsonl-bare-arrays", "jsonl-wrapped", "one-message-per-line"],
)
def test_detect_every_chat_export_shape_the_loader_reads(tmp_path: Path, text: str) -> None:
    path = tmp_path / "chat.json"
    path.write_text(text)
    assert detect_source(path).source == "chat-json"
    assert len(load_corpus(path, "auto").traces) >= 1


def _jsonl(path: Path, *records: object) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


CLAUDE_CODE_TURN = [
    {"type": "user", "sessionId": "s", "message": {"role": "user", "content": "hi"}},
    {"type": "assistant", "sessionId": "s", "message": {"role": "assistant", "content": []}},
]
NOISE = [
    {"type": "summary", "summary": "x", "leafUuid": "u"},
    {"type": "file-history-snapshot", "messageId": "m", "snapshot": {}},
    {"type": "queue-operation", "operation": "enqueue", "sessionId": "s"},
]


def test_claude_code_events_between_turns_do_not_hide_the_format(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "s.jsonl", *NOISE, *CLAUDE_CODE_TURN, NOISE[0])
    assert detect_source(path).source == "claude-code"


def test_a_file_of_only_noise_is_not_recognized(tmp_path: Path) -> None:
    with pytest.raises(DetectionError, match="unrecognized"):
        detect_source(_jsonl(tmp_path / "s.jsonl", *NOISE))


def test_a_langchain_system_message_is_not_skipped_as_noise(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "c.json", [{"type": "system", "content": "s"}, {"type": "human", "content": "q"}]
    )
    assert detect_source(path).source == "chat-json"


def test_tolerant_detection_lets_an_unreadable_line_through_strict_does_not(tmp_path: Path) -> None:
    good = {"id": "1", "messages": [{"role": "user", "content": "q"}]}
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join([json.dumps(good)] * 5 + ["{broken"] + [json.dumps(good)] * 4) + "\n")
    with pytest.raises(DetectionError, match="malformed JSONL"):
        detect_source(path)
    assert detect_source(path, tolerant=True).source == "chat-json"


def test_tolerant_detection_still_refuses_when_most_of_the_sample_is_unreadable(
    tmp_path: Path,
) -> None:
    good = {"id": "1", "messages": [{"role": "user", "content": "q"}]}
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join([json.dumps(good), "x", "y", "z"]) + "\n")
    with pytest.raises(DetectionError, match="malformed JSONL"):
        detect_source(path, tolerant=True)


def test_tolerant_detection_lets_a_minority_of_unknown_records_through(tmp_path: Path) -> None:
    good = {"id": "1", "messages": [{"role": "user", "content": "q"}]}
    path = _jsonl(tmp_path / "c.jsonl", *([good] * 6), {"unrelated": True})
    with pytest.raises(DetectionError, match="unrecognized"):
        detect_source(path)
    assert detect_source(path, tolerant=True).source == "chat-json"


def test_a_leading_blank_line_and_a_bom_do_not_matter(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_bytes(
        b"\xef\xbb\xbf\n" + json.dumps({"messages": [{"role": "user", "content": "q"}]}).encode()
    )
    assert detect_source(path).source == "chat-json"


def test_a_byte_that_is_not_utf8_is_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_bytes(b'{"messages": [{"role": "user", "content": "bad \xff"}]}\n')
    assert detect_source(path).source == "chat-json"
