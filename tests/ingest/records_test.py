"""The shared record reader: every way a file can lay out its JSON must give the same records."""

from __future__ import annotations

import json

import pytest

from bandits.ingest.records import read_records

RECORDS = [{"id": 1, "text": "hello"}, {"id": 2, "text": "a b \u0085 c"}, {"id": 3, "text": "x"}]


def _lines(ea: bool = True, end: str = "\n") -> bytes:
    return (end.join(json.dumps(r, ensure_ascii=ea) for r in RECORDS) + end).encode()


LAYOUTS = {
    "lines": _lines(),
    "lines_raw_unicode_separators": _lines(ea=False),
    "crlf": _lines(end="\r\n"),
    "bom": b"\xef\xbb\xbf" + _lines(),
    "blank_lines_and_spaces": (
        "\n\n" + "\n   \n".join(json.dumps(r) for r in RECORDS) + "\n\n"
    ).encode(),
    "no_trailing_newline": _lines().rstrip(b"\n"),
    "pretty_printed_one_after_another": "\n".join(
        json.dumps(r, indent=2) for r in RECORDS
    ).encode(),
    "pretty_printed_raw_unicode": "\n".join(
        json.dumps(r, indent=2, ensure_ascii=False) for r in RECORDS
    ).encode(),
    "several_on_one_line": "".join(json.dumps(r) for r in RECORDS).encode() + b"\n",
}


@pytest.mark.parametrize("name", LAYOUTS)
def test_every_layout_gives_the_same_records(name: str) -> None:
    result = read_records(LAYOUTS[name], "f")
    assert result.values == RECORDS
    assert result.issues == []
    assert result.layout == "stream"


def test_one_value_is_a_document() -> None:
    result = read_records(json.dumps([RECORDS]).encode(), "f")
    assert result.layout == "document"
    assert result.values == [[RECORDS]]


def test_records_know_their_line() -> None:
    result = read_records(_lines(ea=False), "f")
    assert [r.line for r in result.records] == [1, 2, 3]
    pretty = read_records(LAYOUTS["pretty_printed_one_after_another"], "f")
    assert [r.line for r in pretty.records] == [1, 5, 9]


def test_a_bad_line_is_one_issue_with_its_line_and_the_rest_is_kept() -> None:
    data = _lines().replace(b"\n", b"\n{oops\n", 1)
    result = read_records(data, "f.jsonl")
    assert result.values == RECORDS
    assert [(i.kind, i.location) for i in result.issues] == [("malformed_json", "f.jsonl:2")]
    assert "line 2" in result.issues[0].detail


def test_a_run_of_bad_lines_is_reported_once_with_its_range() -> None:
    data = b'{"a": 1}\nnot json\nstill not\n{"a": 2}\n'
    result = read_records(data, "f")
    assert result.values == [{"a": 1}, {"a": 2}]
    assert len(result.issues) == 1
    assert "lines 2-3" in result.issues[0].detail


def test_a_bad_stretch_at_the_end_is_reported() -> None:
    result = read_records(b'{"a": 1}\n{"a": 2, "b":', "f")
    assert result.values == [{"a": 1}]
    assert [i.kind for i in result.issues] == ["malformed_json"]


def test_an_invalid_byte_is_replaced_and_counted_not_fatal() -> None:
    data = b'{"t": "ok"}\n{"t": "bad \xff byte"}\n'
    result = read_records(data, "f")
    assert [r["t"][:3] for r in result.values] == ["ok", "bad"]
    assert [i.kind for i in result.issues] == ["invalid_utf8"]
    assert "1 byte" in result.issues[0].detail


def test_a_literal_replacement_character_is_not_reported_as_damage() -> None:
    result = read_records('{"t": "�"}'.encode(), "f")
    assert result.issues == []


def test_a_sample_stops_at_the_limit() -> None:
    result = read_records(_lines(), "f", limit=2)
    assert result.values == RECORDS[:2]
    assert result.cut is True


def test_a_cut_final_line_of_a_sample_is_not_a_problem() -> None:
    data = _lines()[: -len(b'"x"}\n') - 3]
    result = read_records(data, "f", tolerate_cut=True)
    assert result.values == RECORDS[:2]
    assert result.issues == []
    assert result.cut is True


def test_an_empty_file_has_no_records() -> None:
    result = read_records(b"  \n\n", "f")
    assert result.records == [] and result.issues == []


def test_a_cut_off_pretty_printed_document_is_one_issue_and_leaves_no_fragments() -> None:
    result = read_records(b'{\n  "messages": [\n    {"role": "user"},\n', "f")
    assert result.records == []
    assert [i.kind for i in result.issues] == ["malformed_json"]


def test_reading_resumes_at_the_next_record_after_a_bad_one() -> None:
    result = read_records(b'{"a": 1}\n{"b": \n{"c": 3}\n', "f")
    assert result.values == [{"a": 1}, {"c": 3}]
    assert [i.kind for i in result.issues] == ["malformed_json"]
