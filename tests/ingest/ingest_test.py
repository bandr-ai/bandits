from __future__ import annotations

from pathlib import Path

import pytest

from bandits.ingest import UnknownSourceError, load_corpus
from bandits.ingest.chat_json import load_chat_json
from bandits.ingest.otlp import load_otlp

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_dispatches_otlp() -> None:
    assert load_corpus(FIXTURES / "traces.otlp.jsonl", "otlp") == load_otlp(
        FIXTURES / "traces.otlp.jsonl"
    )


def test_dispatches_chat_json() -> None:
    assert load_corpus(FIXTURES / "traces.chat.jsonl", "chat-json") == load_chat_json(
        FIXTURES / "traces.chat.jsonl"
    )


def test_unknown_source_raises() -> None:
    with pytest.raises(UnknownSourceError):
        load_corpus(FIXTURES / "traces.otlp.jsonl", "not-a-real-source")
