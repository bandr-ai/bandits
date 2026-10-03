"""Readers for documented native trace exports, routed through the OTLP reader.

OpenInference is an OTLP convention, not a separate file format. Phoenix's
Span JSON and LangSmith's Run JSON are native formats; Langfuse's bundled
trace/observation JSON is another. This module converts their recorded fields
to OTLP internally. Each span carries ``bandits.source.record``, a pointer to
its native record in the redacted source archive (``ArtifactStore.read_native_record``),
rather than a copy of it.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

from bandits.ingest.mapping import IngestMapping
from bandits.ingest.otlp_standard import iter_otlp_standard, load_otlp_standard
from bandits.ingest.report import EXAMPLES, IngestReport, aggregate_issues
from bandits.jsonarray import iter_array
from bandits.redact import DEFAULT_RULESET, RedactionRuleset, redact_bytes
from bandits.traces import Trace, TraceCorpus, TraceIssue, WorkflowDeclaration

Position = dict[str, Any]
"""Where a native record sits in its file: ``{"line": n}`` (1-based, physical),
plus ``"index"`` when that line holds a JSON array; ``{"index": i}`` in a
whole-file array; ``{"document": True}`` for a single JSON document."""

Skipped = list[tuple[str, str | None]]
"""``(reason, observation id)`` for each observation a converter could not
turn into a span. Reasons starting ``duplicate`` are duplicates; the rest are
unconvertible."""

_LINE_LIMIT = 64 << 20
"""Bytes read looking for the end of the first line; past this, a file
starting with ``[`` is read as one streamed array."""

_DUPLICATE = "duplicate id in this record; the first copy is kept"
_INSIDE_SKIPPED = "inside an observation that could not be converted"


class _Unconvertible(ValueError):
    """One observation that cannot become a span; its record is still read."""


def _hex_id(value: object, size: int, namespace: str) -> str:
    raw = str(value or "")
    normalized = raw.removeprefix("0x").replace("-", "").lower()
    if len(normalized) == size and all(c in "0123456789abcdef" for c in normalized):
        return normalized
    return hashlib.sha256(f"{namespace}:{raw}".encode()).hexdigest()[:size]


def _ns(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        # datetime only stores microseconds; keep the remaining three digits
        # separately so an ISO timestamp with nanoseconds is not rounded away.
        fraction = re.search(r"\.(\d+)", value)
        extra = 0
        if fraction:
            digits = fraction.group(1)[:9].ljust(9, "0")
            extra = int(digits[6:])
            value = value.replace(fraction.group(0), "." + digits[:6], 1)
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = dt.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000 + extra


def _av(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, ensure_ascii=False)}


def _attrs(values: dict[str, Any]) -> list[dict[str, Any]]:
    # An unrecorded value is an absent attribute. Encoding None would write the
    # string "null", which the decoder reads as a message saying "null". The
    # null itself stays visible in the source archive.
    return [{"key": k, "value": _av(v)} for k, v in values.items() if v is not None]


Used = dict[str, frozenset[str] | None]
"""The fields a converter consumes: ``None`` for a field consumed whole, or the
keys it reads inside an object (the rest of that object is not consumed)."""


def unmapped_fields(record: dict[str, Any], used: Used) -> dict[str, Any]:
    """What a converter does not consume of *record*, kept under original names.

    A field it never reads is kept whole (nulls and empty values too). Inside
    an object it reads only partly, the keys it does not read are kept. A
    partly read field that is not an object is kept whole: its shape was
    unexpected, so nothing in it was read.
    """
    unmapped: dict[str, Any] = {}
    for key, value in record.items():
        if key not in used:
            unmapped[key] = value
            continue
        inner = used[key]
        if inner is None:
            continue
        if not isinstance(value, dict):
            unmapped[key] = value
            continue
        rest = {k: v for k, v in value.items() if k not in inner}
        if rest:
            unmapped[key] = rest
    return unmapped


def _carry(attributes: dict[str, Any], record: dict[str, Any], used: Used) -> None:
    """Keep :func:`unmapped_fields` on the span as ``bandits.unmapped``.

    Converters name only what they consume; everything else comes along by
    default, so a field nobody listed (usage, cost, a vendor's new field) is
    never silently dropped, and every source puts such fields in the same
    place. Their bytes also stay in the source archive.
    """
    unmapped = unmapped_fields(record, used)
    if unmapped:
        attributes["bandits.unmapped"] = unmapped


def _standard_usage(
    attributes: dict[str, Any], tokens: dict[str, Any], params: dict[str, Any]
) -> None:
    """GenAI names for token counts and request parameters a source recorded
    under its own names, so readers find them in one place."""
    for name, value in tokens.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attributes.setdefault(f"gen_ai.usage.{name}", value)
    for name in ("temperature", "top_p", "max_tokens", "top_k", "seed"):
        value = params.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attributes.setdefault(f"gen_ai.request.{name}", value)


def _records(data: bytes) -> Iterator[tuple[Position, dict[str, Any]]]:
    stripped = data.lstrip()
    if stripped.startswith(b"["):
        parsed = json.loads(data)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON array")
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError("expected objects in JSON array")
            yield {"index": index}, item
    elif stripped.startswith(b"{") and (
        b"\n" not in stripped or not stripped.split(b"\n", 1)[0].rstrip().endswith(b"}")
    ):
        parsed = json.loads(data)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        yield {"document": True}, parsed
    else:
        for number, line in enumerate(io.BytesIO(data), start=1):
            if line.strip():
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("expected one object per JSONL line")
                yield {"line": number}, parsed


def _span(
    trace_id: object,
    span_id: object,
    parent_id: object,
    name: object,
    start: object,
    end: object,
    attributes: dict[str, Any],
    *,
    namespace: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One OTLP span, or :class:`_Unconvertible` naming what is missing."""
    if not span_id:
        raise _Unconvertible("empty id")
    beginning, ending = _ns(start), _ns(end)
    if beginning is None:
        raise _Unconvertible("start time missing" if start is None else "start time unparseable")
    if ending is None:
        raise _Unconvertible(
            "end time missing (still running?)" if end is None else "end time unparseable"
        )
    result: dict[str, Any] = {
        "traceId": _hex_id(trace_id, 32, namespace + ":trace"),
        "spanId": _hex_id(span_id, 16, namespace + ":span"),
        "name": str(name or "unnamed"),
        "startTimeUnixNano": str(beginning),
        "endTimeUnixNano": str(ending),
        "attributes": _attrs(attributes),
    }
    if parent_id:
        result["parentSpanId"] = _hex_id(parent_id, 16, namespace + ":span")
    if extra:
        result.update(extra)
    return result


def _langfuse(record: dict[str, Any], position: Position) -> tuple[list[dict[str, Any]], Skipped]:
    if not isinstance(record.get("observations"), list):
        raise ValueError("Langfuse input requires a trace with observations[]")
    trace_record = record.get("trace") if isinstance(record.get("trace"), dict) else record
    trace_id = trace_record.get("trace_id") or trace_record.get("id")
    if not trace_id:
        raise ValueError("Langfuse trace has no trace_id/id")
    spans: list[dict[str, Any]] = []
    skipped: Skipped = []
    seen: set[str] = set()
    # A top step is one whose parent is not in this record (absent from the
    # export, or none at all); each carries the trace-level fields.
    present = {str(o.get("id")) for o in _langfuse_observations(record["observations"])}
    # The flag marks observations nested in one that was skipped whole: they
    # cannot be placed, so they are counted rather than silently lost.
    stack = [(o, None, False) for o in reversed(record["observations"])]
    while stack:
        observation, enclosing_id, orphaned = stack.pop()
        if not isinstance(observation, dict):
            skipped.append(("not an object", None))
            continue
        native_id = observation.get("id")
        reason = (
            _DUPLICATE
            if native_id is not None and str(native_id) in seen
            else _INSIDE_SKIPPED
            if orphaned
            else "empty id"
            if native_id is None or native_id == ""
            else None
        )
        if reason is not None:
            skipped.append((reason, None if native_id is None else str(native_id)))
            stack.extend((c, None, True) for c in reversed(observation.get("children") or []))
            continue
        seen.add(str(native_id))
        attributes = {
            "langfuse.observation.type": observation.get("type"),
            "langfuse.observation.id": native_id,
            "langfuse.observation.parent_id": observation.get("parentObservationId"),
            "input.value": observation.get("input"),
            "output.value": observation.get("output"),
            "gen_ai.request.model": observation.get("model"),
            "langfuse.observation.level": observation.get("level"),
            "bandits.source.record": {**position, "observation_id": str(native_id)},
        }
        parent = observation.get("parentObservationId") or enclosing_id
        if parent is None or str(parent) not in present:
            # On every top step: the first may become the trace's request,
            # which is not stored as a step.
            attributes["bandits.native.trace_record"] = {
                k: v for k, v in trace_record.items() if k != "observations"
            }
        metadata = observation.get("metadata")
        if isinstance(metadata, str):
            # Some exports serialize metadata as JSON text; read it the same way.
            try:
                decoded = json.loads(metadata)
            except ValueError:
                decoded = None
            metadata = decoded if isinstance(decoded, dict) else metadata
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if key in _PROMOTED_METADATA and isinstance(value, dict):
                    # The span's resource or scope (below), not stored twice;
                    # the marker lets readers find it under its own name.
                    moved = attributes.setdefault("bandits.native.metadata_moved", {})
                    moved[key] = _PROMOTED_METADATA[key]
                    continue
                attributes[f"metadata.{key}"] = value
        # Metadata that is not an object is not consumed: it is carried as-is.
        _carry(
            attributes,
            observation,
            langfuse_used(observation),
        )
        usage = observation.get("usageDetails") or observation.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        params = observation.get("modelParameters")
        _standard_usage(
            attributes,
            {
                "input_tokens": usage.get("input"),
                "output_tokens": usage.get("output"),
                "total_tokens": usage.get("total"),
            },
            params if isinstance(params, dict) else {},
        )
        try:
            span = _span(
                trace_id,
                native_id,
                observation.get("parentObservationId") or enclosing_id,
                observation.get("name"),
                observation.get("startTime"),
                observation.get("endTime"),
                attributes,
                namespace="langfuse",
            )
        except _Unconvertible as exc:
            # Its children still have an id and times of their own; they keep
            # this one as their (now absent) parent.
            skipped.append((str(exc), str(native_id)))
        else:
            if isinstance(metadata, dict):
                if isinstance(metadata.get("resourceAttributes"), dict):
                    span["_native_resource"] = metadata["resourceAttributes"]
                if isinstance(metadata.get("scope"), dict):
                    span["_native_scope"] = metadata["scope"]
            spans.append(span)
        for child in reversed(observation.get("children") or []):
            stack.append((child, native_id, False))
    return spans, skipped


_LANGFUSE_USED: Used = {
    "id": None,
    "type": None,
    "name": None,
    "parentObservationId": None,
    "startTime": None,
    "endTime": None,
    "input": None,
    "output": None,
    "model": None,
    "level": None,
    "metadata": None,
    "children": None,
}
"""Langfuse observation fields the converter turns into span fields or the
attributes above; every other field is kept in ``bandits.unmapped``."""


_PROMOTED_METADATA = {"resourceAttributes": "resource", "scope": "scope"}
"""Langfuse metadata keys holding the span's OTel resource and scope: they
become the span's resource and scope, read back as ``metadata.*`` by
``bandits.fields``."""


def langfuse_used(observation: dict[str, Any]) -> Used:
    """Langfuse's consumed fields for *observation*: metadata is consumed only
    when it is an object (or JSON text of one), whose keys become ``metadata.*``."""
    metadata = observation.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            pass
    if isinstance(metadata, dict):
        return _LANGFUSE_USED
    return {k: v for k, v in _LANGFUSE_USED.items() if k != "metadata"}


USED_FIELDS = {
    "langfuse": langfuse_used,
    "langsmith": lambda record: _LANGSMITH_USED,
    "phoenix": lambda record: _PHOENIX_USED,
}
"""Per source, the fields its converter consumes for one native record."""


def _langfuse_observations(observations: list[Any]) -> Iterator[dict[str, Any]]:
    stack = list(observations)
    while stack:
        observation = stack.pop()
        if isinstance(observation, dict):
            yield observation
            stack.extend(observation.get("children") or [])


_LANGSMITH_KIND = {
    "llm": "LLM",
    "chat_model": "LLM",
    "tool": "TOOL",
    "retriever": "RETRIEVER",
    "chain": "CHAIN",
    "prompt": "PROMPT",
}


_LANGSMITH_USED: Used = {
    "id": None,
    "run_id": None,
    "run_type": None,
    "name": None,
    "trace_id": None,
    "parent_run_id": None,
    "start_time": None,
    "end_time": None,
    "inputs": None,
    "outputs": None,
    "status": None,
    "error": None,
    "child_runs": None,
}
"""LangSmith run fields the converter consumes; the rest are kept in ``bandits.unmapped``."""


def _langsmith(record: dict[str, Any], position: Position) -> tuple[list[dict[str, Any]], Skipped]:
    run_id = record.get("id") or record.get("run_id")
    if not run_id or not record.get("run_type"):
        raise ValueError("LangSmith run requires id/run_id and run_type")
    attributes = {
        "openinference.span.kind": _LANGSMITH_KIND.get(str(record["run_type"]).lower(), "UNKNOWN"),
        "input.value": record.get("inputs"),
        "output.value": record.get("outputs"),
        "bandits.source.record": {**position, "observation_id": str(run_id)},
    }
    _carry(attributes, record, _LANGSMITH_USED)
    extra_fields = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    params = extra_fields.get("invocation_params")
    _standard_usage(
        attributes,
        {
            "input_tokens": record.get("prompt_tokens"),
            "output_tokens": record.get("completion_tokens"),
            "total_tokens": record.get("total_tokens"),
        },
        params if isinstance(params, dict) else {},
    )
    # Keep the producer's structure in input.value/output.value. The shared
    # decoder normalizes LangChain's nested message batches and generations;
    # assigning them directly to gen_ai.* would falsely mark raw lists as
    # already-normalized GenAI messages.
    status = str(record.get("status") or "").lower()
    extra = (
        {"status": {"code": 2, "message": str(record.get("error") or "")}}
        if status == "error" or record.get("error")
        else None
    )
    try:
        span = _span(
            record.get("trace_id") or run_id,
            run_id,
            record.get("parent_run_id"),
            record.get("name"),
            record.get("start_time"),
            record.get("end_time"),
            attributes,
            namespace="langsmith",
            extra=extra,
        )
    except _Unconvertible as exc:
        return [], [(str(exc), str(run_id))]
    return [span], []


_PHOENIX_USED: Used = {
    "context": frozenset({"trace_id", "span_id"}),
    "attributes": None,
    "name": None,
    "parent_id": None,
    "start_time": None,
    "end_time": None,
    "span_kind": None,
    "status": frozenset({"status_code", "code", "message"}),
    "status_code": None,
    "status_message": None,
    "events": None,
    "links": None,
    "resource": frozenset({"attributes"}),
    "instrumentation_scope": None,
    "instrumentationScope": None,
}
"""Phoenix span fields the converter consumes; the rest are kept in ``bandits.unmapped``."""


def _phoenix(record: dict[str, Any], position: Position) -> tuple[list[dict[str, Any]], Skipped]:
    context = record.get("context")
    if not isinstance(context, dict) or not context.get("trace_id") or not context.get("span_id"):
        raise ValueError("Phoenix span requires context.trace_id and context.span_id")
    attributes = record.get("attributes") or {}
    if not isinstance(attributes, dict):
        raise ValueError("Phoenix span attributes must be an object")
    attributes = {
        **attributes,
        "bandits.source.record": {**position, "observation_id": str(context["span_id"])},
    }
    if "openinference.span.kind" not in attributes and isinstance(record.get("span_kind"), str):
        attributes["openinference.span.kind"] = record["span_kind"]
    _carry(attributes, record, _PHOENIX_USED)
    resource = record.get("resource") or {}
    scope = record.get("instrumentation_scope") or record.get("instrumentationScope") or {}
    status = record.get("status")
    if status is None and record.get("status_code") is not None:
        status = {
            "status_code": record["status_code"],
            "message": record.get("status_message"),
        }
    if isinstance(status, dict) and str(
        status.get("status_code") or status.get("code") or ""
    ).upper() in ("ERROR", "STATUS_CODE_ERROR"):
        status = {"code": 2, "message": status.get("message")}
    extra = {
        "events": record.get("events") or [],
        "links": record.get("links") or [],
        "status": status,
    }
    try:
        span = _span(
            context["trace_id"],
            context["span_id"],
            record.get("parent_id"),
            record.get("name"),
            record.get("start_time"),
            record.get("end_time"),
            attributes,
            namespace="phoenix",
            extra=extra,
        )
    except _Unconvertible as exc:
        return [], [(str(exc), str(context["span_id"]))]
    if isinstance(resource, dict):
        span["_native_resource"] = resource.get("attributes") or {}
    if isinstance(scope, dict):
        span["_native_scope"] = scope
    return [span], []


_CONVERTERS = {"langfuse": _langfuse, "langsmith": _langsmith, "phoenix": _phoenix}


def _langsmith_runs(record: dict[str, Any], skipped: Skipped) -> Iterator[dict[str, Any]]:
    """Flatten documented child_runs without turning nesting into time order.

    Children that are not objects, and repeated run ids, go to *skipped*; so
    do the children of a repeated run, which have no single place to go.
    """
    stack = [
        (record, None, record.get("trace_id") or record.get("id") or record.get("run_id"), False)
    ]
    seen: set[str] = set()
    while stack:
        run, enclosing, trace_id, orphaned = stack.pop()
        if not isinstance(run, dict):
            skipped.append(("not an object", None))
            continue
        native_id = str(run.get("id") or run.get("run_id"))
        if native_id in seen or orphaned:
            skipped.append((_DUPLICATE if native_id in seen else _INSIDE_SKIPPED, native_id))
            stack.extend((c, None, trace_id, True) for c in reversed(run.get("child_runs") or []))
            continue
        seen.add(native_id)
        if enclosing and not run.get("parent_run_id"):
            run = {**run, "parent_run_id": enclosing}
        if trace_id and not run.get("trace_id"):
            run = {**run, "trace_id": trace_id}
        yield run
        for child in reversed(run.get("child_runs") or []):
            stack.append((child, run.get("id") or run.get("run_id"), trace_id, False))


def _where(path: Path, position: Position) -> str:
    if "line" in position:
        return f"{path}:{position['line']}" + (
            f"[{position['index']}]" if "index" in position else ""
        )
    return f"{path}[{position['index']}]" if "index" in position else str(path)


def _origin(path: Path, span: dict[str, Any]) -> str:
    """Where a converted span came from in the native file, for issue locations."""
    for attribute in span.get("attributes") or []:
        if attribute.get("key") == "bandits.source.record":
            pointer = json.loads(attribute["value"]["stringValue"])
            position = {k: v for k, v in pointer.items() if k != "observation_id"}
            return f"{_where(path, position)} observation {pointer['observation_id']}"
    return str(path)


class NativeConversion:
    """One native file converted to OTLP in chunks, counting every observation.

    :meth:`chunks` yields the path of a temporary OTLP/JSONL file holding the
    next batch; whoever consumes it (the loader, or request discovery) must
    finish reading before asking for the next. Langfuse records are batched
    by about 400 spans so a large export never has to fit in memory.
    """

    def __init__(
        self,
        path: Path,
        source_name: str,
        ruleset: RedactionRuleset,
        *,
        scratch_dir: Path | None = None,
    ) -> None:
        if source_name not in _CONVERTERS:
            raise ValueError(f"unknown native source {source_name!r}")
        if path.is_dir():
            raise ValueError(
                f"{source_name} reads one export file; {path} is a directory — ingest each file"
            )
        self.path, self.source_name, self.ruleset = path, source_name, ruleset
        self.scratch_dir = scratch_dir
        self.issues: list[TraceIssue] = []
        self.source_hash = hashlib.sha256()
        # Converter-level counts; a caller adds the decoded chunks' own.
        self.report = IngestReport()
        self.observations = 0
        # Native origin of each line in the current converted chunk.
        self.origins: list[str] = []

    def _source_records(self) -> Iterator[tuple[Position, dict[str, Any]]]:
        path, ruleset = self.path, self.ruleset
        with path.open("rb") as stream:
            first = stream.readline(_LINE_LIMIT)
            first_number = 1
            while first and not first.strip():
                self.source_hash.update(first)
                first = stream.readline(_LINE_LIMIT)
                first_number += 1
            # A first line cut at the limit is a single-line array export
            # (streamed below); any other long first line is read whole.
            cut = len(first) == _LINE_LIMIT and not first.endswith(b"\n")
            if cut and not first.lstrip().startswith(b"["):
                first += stream.readline()
                cut = False
            # Complete JSON objects/arrays are independent JSONL records.
            try:
                jsonl = not cut and isinstance(json.loads(first), (dict, list))
            except (UnicodeDecodeError, json.JSONDecodeError):
                jsonl = False
            if jsonl:
                for number, line in enumerate(chain((first,), stream), start=first_number):
                    self.source_hash.update(line)
                    if not line.strip():
                        continue
                    safe = redact_bytes(line, f"{path}:record{number}", ruleset)
                    self.issues.extend(safe.issues)
                    for inner, record in _records(safe.data):
                        index = {"index": inner["index"]} if "index" in inner else {}
                        yield {"line": number, **index}, record
            elif first.lstrip().startswith(b"["):
                # One array element in memory at a time, not the whole file.
                stream.seek(0)
                self.source_hash = hashlib.sha256()
                for index, (_, element, line) in enumerate(iter_array(stream, self.source_hash)):
                    if element is None:
                        break
                    safe = redact_bytes(element, str(path), ruleset, first_line=line)
                    self.issues.extend(safe.issues)
                    record = json.loads(safe.data)
                    if not isinstance(record, dict):
                        raise ValueError("expected objects in JSON array")
                    yield {"index": index}, record
            else:
                stream.seek(0)
                original = stream.read()
                self.source_hash = hashlib.sha256(original)
                safe = redact_bytes(original, str(path), ruleset)
                self.issues.extend(safe.issues)
                yield from _records(safe.data)

    def _skip(self, skipped: Skipped, position: Position) -> None:
        self.observations += len(skipped)
        for reason, observation_id in skipped:
            bucket = "duplicate_native" if reason == _DUPLICATE else "unconvertible"
            self.report.buckets[bucket] += 1
            if len(self.report.unconvertible_examples) < EXAMPLES:
                self.report.unconvertible_examples.append(
                    f"{_where(self.path, position)} observation {observation_id or '?'}: {reason}"
                )

    def chunks(self) -> Iterator[Path]:
        source_name = self.source_name
        with tempfile.TemporaryDirectory(
            prefix=".bandits-native-", dir=self.scratch_dir or Path.cwd()
        ) as temporary:
            converted = Path(temporary) / "converted.jsonl"
            with converted.open("w+", encoding="utf-8") as output:
                pending = 0
                for position, outer in self._source_records():
                    records = (
                        outer.get("runs")
                        if source_name == "langsmith"
                        else outer.get("spans") or outer.get("data")
                        if source_name == "phoenix"
                        else None
                    )
                    if not isinstance(records, list):
                        records = [outer]
                    skipped: Skipped = []
                    expanded = (
                        (
                            run
                            for record in records
                            for run in (
                                _langsmith_runs(record, skipped)
                                if isinstance(record, dict)
                                else [record]
                            )
                        )
                        if source_name == "langsmith"
                        else iter(records)
                    )
                    for record in expanded:
                        try:
                            if not isinstance(record, dict):
                                raise ValueError("expected an object")
                            spans, dropped = _CONVERTERS[source_name](record, position)
                        except (TypeError, ValueError) as exc:
                            self.issues.append(
                                TraceIssue(
                                    kind="unsupported_native_record",
                                    detail=str(exc),
                                    location=_where(self.path, position),
                                )
                            )
                            self.report.unreadable_items["unsupported_native_record"] += 1
                            continue
                        self.observations += len(spans)
                        self._skip(dropped, position)
                        for span in spans:
                            self.origins.append(_origin(self.path, span))
                            output.write(
                                json.dumps(_request(span, source_name), ensure_ascii=False)
                            )
                            output.write("\n")
                            pending += 1
                    self._skip(skipped, position)
                    if source_name == "langfuse" and pending >= 400:
                        output.flush()
                        yield converted
                        output.seek(0)
                        output.truncate(0)
                        self.origins = []
                        pending = 0
                if pending:
                    output.flush()
                    yield converted

    def relocate(self, issue: TraceIssue, chunk: Path) -> TraceIssue:
        """*issue* located in the native file instead of the converted *chunk*.

        The chunk lives in a random temporary directory, so its path in a
        stored issue would make the corpus id differ on every ingest.
        """
        location = issue.location or ""
        if not location.startswith(str(chunk)):
            return issue
        line = location[len(str(chunk)) :].removeprefix(":").split("#", 1)[0]
        if line.isdigit() and 0 < int(line) <= len(self.origins):
            return issue.replace(location=self.origins[int(line) - 1])
        return issue.replace(location=str(self.path))

    def finish(self, decoded: IngestReport) -> IngestReport:
        """Converter counts plus the decoded chunks' (*decoded*), checked to add up."""
        total = IngestReport()
        total.merge(self.report)
        total.merge(decoded)
        lost = self.report.buckets["unconvertible"] + self.report.buckets["duplicate_native"]
        if self.observations != lost + decoded.spans_seen:
            total.accounting_errors.append(
                f"{self.path}: {self.observations} observation(s) seen but {lost} skipped and "
                f"{decoded.spans_seen} converted; this is a bandits bug"
            )
        total.spans_seen = self.observations
        return total


def _request(span: dict[str, Any], source_name: str) -> dict[str, Any]:
    """One converted span as its own OTLP export request."""
    native_resource = span.pop("_native_resource", {})
    native_scope = span.pop("_native_scope", {"name": source_name})
    if not isinstance(native_scope, dict):
        native_scope = {"name": source_name}
    if not isinstance(native_resource, dict):
        native_resource = {}
    scope = {k: v for k, v in native_scope.items() if k != "attributes"}
    scope["attributes"] = (
        _attrs(native_scope.get("attributes") or {})
        if isinstance(native_scope.get("attributes"), dict)
        else native_scope.get("attributes") or []
    )
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs(native_resource)},
                "scopeSpans": [{"scope": scope, "spans": [span]}],
            }
        ]
    }


def iter_native(
    path: Path,
    source_name: str,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    workflow: WorkflowDeclaration | None = None,
    pipeline_steps: bool = True,
    report: IngestReport | None = None,
    mapping: IngestMapping | None = None,
    scratch_dir: Path | None = None,
) -> Iterator[Trace | TraceCorpus]:
    """Yield traces, then a trace-free corpus footer containing issues and metadata.

    The report is complete only when the iterator is exhausted. Bundled Langfuse
    releases decoded chunks; interleaved native spans are grouped on disk.

    Every observation the converters see is counted: converted, skipped as
    unconvertible or duplicate, and the converted ones through the OTLP
    reader's own buckets. ``report`` receives the totals.
    """
    conversion = NativeConversion(path, source_name, ruleset, scratch_dir=scratch_dir)
    # Always collected: the per-ingest summary issues are built from it once,
    # not once per chunk.
    decoded = IngestReport()
    with path.open("rb") as original:
        digest = hashlib.file_digest(original, "sha256").hexdigest()
    converted_issues: list[TraceIssue] = []
    seen_ids: set[str] = set()
    for chunk_path in conversion.chunks():
        ids = set()
        reader = iter_otlp_standard if source_name != "langfuse" else None
        if reader is None:
            chunk = load_otlp_standard(
                chunk_path,
                ruleset,
                pipeline_steps=pipeline_steps,
                workflow=workflow,
                report=decoded,
                defer_aggregate_issues=True,
                mapping=mapping,
            )
            items = (*chunk.traces, chunk.replace(traces=()))
        else:
            items = reader(
                chunk_path,
                ruleset,
                pipeline_steps=pipeline_steps,
                workflow=workflow,
                report=decoded,
                defer_aggregate_issues=True,
                mapping=mapping,
                scratch_dir=scratch_dir,
            )
        for item in items:
            if isinstance(item, TraceCorpus):
                converted_issues.extend(conversion.relocate(i, chunk_path) for i in item.issues)
            else:
                ids.add(item.trace_id)
                yield item.replace(source=source_name, source_digest=digest)
        decoded.split_trace_ids += len(ids & seen_ids)
        seen_ids |= ids
        if reader is None:
            del chunk, items
    if conversion.source_hash.hexdigest() != digest:
        raise ValueError(f"source file changed during ingest: {path}")
    internal = conversion.finish(decoded)
    deferred = aggregate_issues(
        internal,
        str(path),
        workflow=workflow is not None,
        mapping_name=workflow.mapping_name if workflow is not None else None,
        step_kinds=mapping.step_kinds if mapping is not None else (),
    )
    if report is not None:
        report.merge(internal)
    yield TraceCorpus(
        source=source_name,
        traces=(),
        issues=tuple(conversion.issues) + tuple(converted_issues) + tuple(deferred),
        workflow=workflow,
        redaction_ruleset=ruleset.name,
    )


def load_native(
    path: Path,
    source_name: str,
    ruleset: RedactionRuleset = DEFAULT_RULESET,
    *,
    workflow: WorkflowDeclaration | None = None,
    pipeline_steps: bool = True,
    report: IngestReport | None = None,
    mapping: IngestMapping | None = None,
) -> TraceCorpus:
    """Materialize the streaming native reader for library callers."""
    traces = []
    for item in iter_native(
        path,
        source_name,
        ruleset,
        workflow=workflow,
        pipeline_steps=pipeline_steps,
        report=report,
        mapping=mapping,
    ):
        if isinstance(item, TraceCorpus):
            return item.replace(traces=tuple(traces))
        traces.append(item)
    raise RuntimeError("native reader ended without a footer")
