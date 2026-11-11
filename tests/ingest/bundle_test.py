from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bandits.ingest.bundle import bundle_json_documents

FIXTURE = Path(__file__).parents[1] / "fixtures" / "failproofai.session.json"


def _sessions(folder: Path) -> list[Path]:
    """Two sessions, one pretty-printed, as a session export folder holds them."""
    folder.mkdir()
    document = json.loads(FIXTURE.read_text())
    second = {**document, "session_id": "s2"}
    (folder / "session-b.json").write_text(json.dumps(second, indent=2))
    (folder / "session-a.json").write_text(json.dumps(document))
    return sorted(folder.iterdir())


def test_one_document_per_line_in_path_order(tmp_path: Path) -> None:
    files = _sessions(tmp_path / "export")
    out = tmp_path / "out.jsonl"
    listing = bundle_json_documents(tmp_path / "export", out)
    lines = out.read_bytes().splitlines()
    assert [entry["path"] for entry in listing] == ["session-a.json", "session-b.json"]
    assert lines[0] == files[0].read_bytes()  # already one line: copied byte for byte
    assert json.loads(lines[1]) == json.loads(files[1].read_text())  # re-encoded, same value
    assert listing[1]["sha256"] == hashlib.sha256(files[1].read_bytes()).hexdigest()


def test_byte_order_mark_is_not_carried_into_a_line(tmp_path: Path) -> None:
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "a.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"k": 1}).encode())
    bundle_json_documents(folder, tmp_path / "out.jsonl")
    assert (tmp_path / "out.jsonl").read_bytes() == b'{"k":1}\n'


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("a.json", "{cut", "is not one JSON document"),
        ("a.json", "3", "holds a JSON int, not a document"),
        ("a.jsonl", "{}\n", "holds .jsonl files as well"),
    ],
)
def test_refuses_what_is_not_one_document_per_file(
    tmp_path: Path, name: str, content: str, reason: str
) -> None:
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "ok.json").write_text("{}")
    (folder / name).write_text(content)
    with pytest.raises(ValueError, match=reason):
        bundle_json_documents(folder, tmp_path / "out.jsonl")
