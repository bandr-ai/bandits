"""Ingest adapters: turn a raw trace export into a :class:`~bandits.traces.TraceCorpus`.

Known export shapes may be recognized from their structure. Ambiguous inputs
require an explicit source rather than a guessed interpretation.
"""

from __future__ import annotations

from pathlib import Path

from bandits.ingest.chat_json import load_chat_json
from bandits.ingest.claude_code import load_claude_code
from bandits.ingest.detect import DetectionError, detect_source
from bandits.ingest.mapping import IngestMapping
from bandits.ingest.native import load_native
from bandits.ingest.otlp import load_otlp
from bandits.ingest.otlp_standard import load_otlp_standard
from bandits.ingest.report import IngestReport
from bandits.ingest.trail import load_trail
from bandits.redact import DEFAULT_RULESET, RedactionRuleset
from bandits.traces import TraceCorpus, WorkflowDeclaration

CANONICAL_SOURCES: tuple[str, ...] = (
    "otlp",
    "otlp-std",
    "chat-json",
    "claude-code",
    "trail",
    "langfuse",
    "langsmith",
    "phoenix",
)

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
    source: str = "auto",
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    pipeline_steps: bool = True,
    workflow: WorkflowDeclaration | None = None,
    report: IngestReport | None = None,
    mapping: IngestMapping | None = None,
) -> TraceCorpus:
    """Read a raw export into a :class:`TraceCorpus` using a recognized adapter.

    ``pipeline_steps`` and ``workflow`` apply to OTLP and the native readers
    routed through it; so do ``report``, which the other readers leave empty,
    and a confirmed ``mapping``.
    """
    if source == "auto":
        source = detect_source(Path(path)).source
        # Legacy flat OTLP has no workflow mode, so there is nothing to declare.
        if workflow is None and source in (
            "otlp-std",
            "langfuse",
            "langsmith",
            "phoenix",
        ):
            raise ValueError(
                f"auto-detected {source!r}, but interaction mode is not in the file format; "
                "declare the source and whether this is a conversation or workflow"
            )
    if source in ("langfuse", "langsmith", "phoenix"):
        return load_native(
            Path(path),
            source,
            ruleset,
            pipeline_steps=pipeline_steps,
            workflow=workflow,
            report=report,
            mapping=mapping,
        )
    loader = _LOADERS.get(source)
    if loader is None:
        raise UnknownSourceError(
            f"unknown source {source!r}; declare one of {list(CANONICAL_SOURCES)}"
        )
    if source == "otlp-std":
        return load_otlp_standard(
            Path(path),
            ruleset,
            pipeline_steps=pipeline_steps,
            workflow=workflow,
            report=report,
            mapping=mapping,
        )
    if workflow is not None or mapping is not None:
        raise ValueError(f"workflow mode is an otlp-std option; source {source!r} has none")
    if not pipeline_steps:
        raise ValueError(f"pipeline steps are an otlp-std option; source {source!r} has none")
    return loader(Path(path), ruleset)


__all__ = [
    "CANONICAL_SOURCES",
    "DetectionError",
    "IngestReport",
    "UnknownSourceError",
    "detect_source",
    "load_corpus",
]
