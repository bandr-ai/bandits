"""Ingest adapters: turn a raw trace export into a :class:`~bandits.traces.TraceCorpus`.

Format is always declared by the caller, never guessed from file content — a
mislabeled export would otherwise parse into a corpus with the wrong spans in it
and look like a valid, if odd, trace rather than an error.
"""

from __future__ import annotations

from pathlib import Path

from bandits.ingest.chat_json import load_chat_json
from bandits.ingest.claude_code import load_claude_code
from bandits.ingest.otlp import load_otlp
from bandits.ingest.otlp_standard import load_otlp_standard
from bandits.ingest.trail import load_trail
from bandits.redact import DEFAULT_RULESET, RedactionRuleset
from bandits.traces import TraceCorpus

CANONICAL_SOURCES: tuple[str, ...] = ("otlp", "otlp-std", "chat-json", "claude-code", "trail")

_LOADERS = {
    "otlp": load_otlp,
    "otlp-std": load_otlp_standard,
    "chat-json": load_chat_json,
    "claude-code": load_claude_code,
    "trail": load_trail,
}


class UnknownSourceError(ValueError):
    """Raised for a source name that is not in :data:`CANONICAL_SOURCES`."""


def load_corpus(
    path: str | Path,
    source: str,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    pipeline_steps: bool = True,
) -> TraceCorpus:
    """Read a raw export into a :class:`TraceCorpus` using the declared adapter.

    ``pipeline_steps`` is read only by ``otlp-std``, the one source that
    declares workflow steps apart from model and tool calls.
    """
    loader = _LOADERS.get(source)
    if loader is None:
        raise UnknownSourceError(
            f"unknown source {source!r}; declare one of {list(CANONICAL_SOURCES)}"
        )
    if source == "otlp-std":
        return load_otlp_standard(Path(path), ruleset, pipeline_steps=pipeline_steps)
    if not pipeline_steps:
        raise ValueError(f"pipeline steps are an otlp-std option; source {source!r} has none")
    return loader(Path(path), ruleset)


__all__ = [
    "CANONICAL_SOURCES",
    "UnknownSourceError",
    "load_corpus",
]
