"""Stage 1 -- ingest. Raw vendor exports become a normalized :class:`TraceCorpus`.

Everything downstream reads invocation points, so this is where the pipeline either
gets real state to reconstruct or degrades to text (PLAN.md, "The key concept:
invocation points"). Two adapters are supported today:

``otlp``
    OpenTelemetry GenAI spans, where invocation points are recorded explicitly.
``chat-json``
    OpenAI-style transcripts, where they must be recovered from ``tool_calls``
    blocks paired with ``tool``-role replies (PLAN.md Step 6).
``langsmith``
    LangSmith run trees, where ``run_type="tool"`` runs are invocation points
    recorded explicitly and llm runs carry the transcript. This is the adapter a
    real customer's telemetry usually needs (docs/PRODUCT.md, "Who this is for").

**The source is always declared, never sniffed.** Format detection is a silent
failure waiting to happen: a chat export misread as OTLP would yield zero
invocations and look like a tool-free episode rather than an error. An unknown
source name raises :class:`UnknownSourceError`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from bandits.contracts import (
    CallStatus,
    IngestIssue,
    InvocationPoint,
    JsonObject,
    Message,
    Trace,
    TraceCorpus,
)
from bandits.ingest.chat_json import parse_chat_record
from bandits.ingest.errors import infer_error_kind, looks_like_error, normalize_error_kind
from bandits.ingest.langsmith import parse_langsmith_record
from bandits.ingest.otlp import normalize_attributes, parse_otlp_record
from bandits.ingest.registry import RegistryError, load_registry, load_registry_with_issues

#: The only adapter names :func:`load_corpus` accepts. Declared, never inferred.
CANONICAL_SOURCES: tuple[str, ...] = ("otlp", "chat-json", "langsmith")

#: Adapter name -> record parser. The single dispatch point for ingest.
_PARSERS = {
    "otlp": parse_otlp_record,
    "chat-json": parse_chat_record,
    "langsmith": parse_langsmith_record,
}


class UnknownSourceError(ValueError):
    """Raised for a source name that is not in :data:`CANONICAL_SOURCES`."""


def _digest(payload: bytes) -> str:
    """sha256 hex of the exact source bytes for one record.

    Provenance for the fidelity gate: stage 7 replays a trace against a rebuilt
    environment and has to be able to prove which bytes it replayed.
    """
    return hashlib.sha256(payload).hexdigest()


def _iter_records(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield ``(index, raw bytes)`` for every record in a JSONL or JSON-array file.

    JSONL is the primary form and the one the digest contract is written for: the
    bytes yielded are the exact line bytes, newline excluded. A whole-file JSON
    array is also accepted for convenience, in which case each record's bytes are
    its compact canonical re-encoding -- there are no per-record source bytes to
    point at, and that substitution is stated here rather than hidden.
    """
    raw = path.read_bytes()
    stripped = raw.lstrip()
    if stripped.startswith(b"["):
        payload = json.loads(raw.decode("utf-8"))
        for index, record in enumerate(payload):
            yield index, json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return
    index = 0
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        yield index, line
        index += 1


def _tool_names(traces: tuple[Trace, ...]) -> set[str]:
    """Every tool name observed anywhere in the corpus."""
    return {inv.tool for trace in traces for inv in trace.invocations}


def load_corpus(
    path: str | Path,
    source: str,
    tools_path: str | Path | None = None,
) -> TraceCorpus:
    """Read a raw export into a :class:`TraceCorpus` using the declared adapter.

    ``source`` must be one of :data:`CANONICAL_SOURCES`. ``tools_path`` is the
    declared tool registry: it is optional, but supplying it lets ingest report
    tools that were called yet never declared, which is the first sign an export and
    its registry came from different systems.

    A record that cannot be normalized never disappears -- it becomes an
    :class:`IngestIssue` on the returned corpus. That is the whole point of the
    ``issues`` field: an ingest run that quietly drops the failure paths produces an
    environment that is confidently wrong.
    """
    if source not in CANONICAL_SOURCES:
        raise UnknownSourceError(
            f"unknown source {source!r}; declare one of {list(CANONICAL_SOURCES)}. "
            "Formats are never sniffed."
        )
    path = Path(path)
    parse = _PARSERS[source]

    traces: list[Trace] = []
    issues: list[IngestIssue] = []

    for index, raw in _iter_records(path):
        location = f"{path}:{index}"
        try:
            record = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            issues.append(IngestIssue(
                kind="malformed_json",
                detail=f"record is not valid JSON: {exc}",
                location=location,
            ))
            continue
        if not isinstance(record, dict):
            issues.append(IngestIssue(
                kind="malformed_record",
                detail=f"expected a JSON object, got {type(record).__name__}",
                location=location,
            ))
            continue
        trace, record_issues = parse(
            record,
            source_digest=_digest(raw),
            location=location,
            fallback_trace_id=f"{path.name}:{index}",
        )
        issues.extend(record_issues)
        if trace is not None:
            traces.append(trace)

    if tools_path is not None:
        registry, registry_issues = load_registry_with_issues(tools_path)
        issues.extend(registry_issues)
        for name in sorted(_tool_names(tuple(traces)) - set(registry)):
            issues.append(IngestIssue(
                kind="undeclared_tool",
                detail=f"tool {name!r} is called in the traces but absent from the registry",
                location=str(tools_path),
            ))

    return TraceCorpus(source=source, traces=tuple(traces), issues=tuple(issues))


def load_corpus_and_registry(
    path: str | Path,
    source: str,
    tools_path: str | Path | None = None,
) -> tuple[TraceCorpus, dict[str, JsonObject]]:
    """:func:`load_corpus` plus the declared registry mapping.

    ``TraceCorpus`` has no field for the registry, but stage 2 needs both to fill in
    ``ToolProfile.declared_schema`` and ``declared_only``. Rather than smuggling the
    registry into ``Trace.metadata``, it is returned alongside. Returns an empty
    mapping when no registry was supplied.
    """
    corpus = load_corpus(path, source, tools_path=tools_path)
    registry: dict[str, JsonObject] = {}
    if tools_path is not None:
        registry, _ = load_registry_with_issues(tools_path)
    return corpus, registry


__all__ = [
    "CANONICAL_SOURCES",
    "CallStatus",
    "IngestIssue",
    "InvocationPoint",
    "Message",
    "RegistryError",
    "Trace",
    "TraceCorpus",
    "UnknownSourceError",
    "infer_error_kind",
    "load_corpus",
    "load_corpus_and_registry",
    "load_registry",
    "load_registry_with_issues",
    "looks_like_error",
    "normalize_attributes",
    "normalize_error_kind",
]
