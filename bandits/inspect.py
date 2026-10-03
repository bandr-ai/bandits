"""A self-contained HTML page showing what one ingest produced.

Written beside each saved corpus (``inspect.html``) so the parsed structure can
be looked at without ``jq``: the counts and accounting, the trace shapes, each
sampled trace as a tree of steps, and the notes the ingest wrote. The page
makes no network requests; it holds the corpus's (redacted) data, so it stays
on disk with the corpus.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bandits.traces import Trace, TraceCorpus, TraceIssue

TRACES_SHOWN = 50
"""Traces embedded in the page, besides one example per shape."""

_TEXT_LIMIT = 4000
_VALUE_LIMIT = 600
_LOCATION_FILE = re.compile(r"^(.*?)(?::record\d+)?(?::\d+)?$")


def issue_rows(issues: Iterable[TraceIssue], *, all_redactions: bool) -> list[tuple[str, str, str]]:
    """Issue rows with redactions counted per kind of value, not one row each.

    A redaction is routine (one per hidden value, often hundreds per file);
    listed one by one they bury the issues that need reading.
    """
    rows: list[tuple[str, str, str]] = []
    redactions: dict[str, list[str]] = {}
    for issue in issues:
        if issue.kind == "redaction" and not all_redactions:
            redactions.setdefault(issue.detail, []).append(issue.location or "")
            continue
        rows.append((issue.kind, issue.location or "", issue.detail))
    grouped = []
    for detail, locations in redactions.items():
        files = sorted({Path(_LOCATION_FILE.match(loc).group(1)).name for loc in locations if loc})
        grouped.append(
            (
                "redaction",
                f"{len(set(locations))} location(s) in {', '.join(files) or 'the source'}",
                f"{detail} ({len(locations)} value(s); --all-redactions lists each)",
            )
        )
    return grouped + rows


class TraceSample:
    """The traces a page shows: the first :data:`TRACES_SHOWN`, plus the
    first trace of each step layout seen later, so rare shapes have an example.

    Fed one trace at a time, so a streamed ingest never holds more than this.
    """

    def __init__(self, limit: int = TRACES_SHOWN, wanted: Iterable[str] = ()) -> None:
        self.limit = limit
        self.wanted = set(wanted)
        self.kept: dict[str, Trace] = {}
        self.layouts: set[tuple[tuple[str, str], ...]] = set()
        self.seen = 0

    def add(self, trace: Trace) -> None:
        self.seen += 1
        layout = tuple(sorted({(s.kind.value, s.name) for s in trace.spans}))
        new_layout = layout not in self.layouts and len(self.layouts) < self.limit
        self.layouts.add(layout)
        if len(self.kept) < self.limit or new_layout or trace.trace_id in self.wanted:
            self.kept[trace.trace_id] = trace


def _clip(value: Any, limit: int = _TEXT_LIMIT) -> Any:
    """*value*, or a cut-off rendering of it when its JSON is longer than
    *limit*, so one huge payload (hundreds of retrieved documents) cannot swell the page."""
    if isinstance(value, str):
        return (
            value if len(value) <= limit else value[:limit] + f"… [{len(value) - limit} more chars]"
        )
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"… [{len(text) - limit} more chars of JSON]"


def _clip_fields(attributes: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _clip(value, _VALUE_LIMIT) for key, value in attributes.items()}


def _messages_text(messages: Any) -> list[dict[str, str]] | None:
    """Normalized GenAI messages as role/text pairs, for reading."""
    if not isinstance(messages, list):
        return None
    shown = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        texts = []
        for part in message.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(str(part.get("content", "")))
            elif part.get("type") == "tool_call":
                texts.append(
                    f"→ {part.get('name')}({json.dumps(part.get('arguments'), default=str)})"
                )
            else:
                texts.append(json.dumps(part, default=str))
        shown.append({"role": str(message.get("role")), "text": _clip("\n".join(texts))})
    return shown


def _misc(attributes: dict[str, Any]) -> dict[str, Any]:
    """The fields Bandits keeps without interpreting: the converter's
    ``bandits.unmapped`` and the app's own ``metadata.*``, as readable objects."""
    misc: dict[str, Any] = {}
    unmapped = attributes.get("bandits.unmapped")
    if isinstance(unmapped, str):
        try:
            unmapped = json.loads(unmapped)
        except ValueError:
            pass
    if isinstance(unmapped, dict):
        misc.update(unmapped)
    elif unmapped is not None:
        misc["(unmapped)"] = unmapped
    metadata = {
        key.removeprefix("metadata."): value
        for key, value in attributes.items()
        if key.startswith("metadata.")
    }
    if metadata:
        misc["metadata"] = metadata
    return _clip(misc, 30_000) if misc else {}


def _steps(trace: Trace) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    request = trace.request
    seen: set[str] = set()
    for span in trace.spans:
        attributes = span.attributes
        steps.append(
            {
                "id": span.span_id,
                "parent": span.parent_span_id,
                "type": span.kind.value,
                "name": span.name,
                "status": span.status.value,
                "start": span.started_at.isoformat(),
                "ms": round((span.ended_at - span.started_at).total_seconds() * 1000),
                "model": attributes.get("gen_ai.request.model"),
                "input_messages": _messages_text(attributes.get("gen_ai.input.messages")),
                "output_messages": _messages_text(attributes.get("gen_ai.output.messages")),
                "input": _clip(span.arguments) if span.arguments else None,
                "output": _clip(span.output),
                "attributes": _clip_fields(attributes),
                "misc": _misc(attributes),
            }
        )
        seen.add(span.span_id)
    for node in trace.workflow_nodes:
        if node.span_id in seen:
            continue
        steps.append(
            {
                "id": node.span_id,
                "parent": node.parent_span_id,
                "type": "step",
                "name": node.name,
                "status": node.status.value,
                "start": node.started_at.isoformat(),
                "ms": round((node.ended_at - node.started_at).total_seconds() * 1000),
                "input": _clip(node.input),
                "output": _clip(node.output),
                "attributes": _clip_fields(node.attributes),
                "misc": _misc(node.attributes),
            }
        )
        seen.add(node.span_id)
    if request is not None and request.source_span_id and request.source_span_id not in seen:
        steps.append(
            {
                "id": request.source_span_id,
                "parent": None,
                "type": "invocation",
                "name": "the run (its input holds the question)",
                "status": request.status.value if request.status else "ok",
                "start": min((s["start"] for s in steps), default=""),
                "ms": None,
                "input": _clip(request.raw_input),
                "output": _clip(request.raw_output),
                "attributes": {},
            }
        )
    return steps


def _trace_view(trace: Trace) -> dict[str, Any]:
    request = trace.request
    return {
        "trace_id": trace.trace_id,
        "task": _clip(trace.task) if trace.task else None,
        "task_status": request.task_status if request else None,
        "task_path": request.task_path if request else None,
        "task_reason": request.task_reason if request else None,
        "invocation_basis": request.invocation_basis if request else None,
        "task_candidates": [
            {"span_id": c.span_id, "path": c.path, "value": _clip(c.value, 500)}
            for c in (request.task_candidates if request else ())
        ],
        "delivered": _clip(request.delivered) if request else None,
        "system_prompt": _clip(trace.system_prompt) if trace.system_prompt else None,
        "evidence": len(trace.evidence),
        "steps": _steps(trace),
    }


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every scalar in a raw record with its path; nested observations are
    their own steps and are checked there."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key != "children":
                yield from _leaves(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, f"{path}[{index}]")
    elif isinstance(value, str) and value[:1] in "{[":
        # JSON held in text is checked by its values, as the parsed side reads it.
        try:
            decoded = json.loads(value)
        except ValueError:
            yield path, value
        else:
            yield from _leaves(decoded, path)
    else:
        yield path, value


_ABSENT = object()

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")


def _instant(text: str) -> float | None:
    """*text* as a point in time, when it is an ISO timestamp, so that
    ``…32.598Z`` and ``…32.598000+00:00`` compare equal."""
    if len(text) > 40 or not _TIMESTAMP.match(text):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


def _parsed_values(value: Any, texts: list[str], numbers: set[float], flags: set[bool]) -> None:
    """Every value in a parsed step, reading JSON held in strings as well;
    timestamps also count as instants (in *numbers*, negated to stay apart)."""
    if isinstance(value, dict):
        for child in value.values():
            _parsed_values(child, texts, numbers, flags)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _parsed_values(child, texts, numbers, flags)
    elif isinstance(value, bool):
        flags.add(value)
    elif isinstance(value, (int, float)):
        numbers.add(float(value))
    elif isinstance(value, str):
        texts.append(value)
        if (moment := _instant(value)) is not None:
            numbers.add(-moment - 1)
        if value[:1] in "{[":
            try:
                _parsed_values(json.loads(value), texts, numbers, flags)
            except ValueError:
                pass


def missing_values(raw: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    """Paths of values in a *raw* record that the *parsed* step holds nowhere.

    Informational, not proof: a value counts as found if the parsed step
    holds an equal value anywhere (as a field, in a message, in JSON text), so
    it cannot tell a moved value from a coincidence. Carried fields are
    checked exactly by :func:`check_fidelity`. Empty values are not checked.
    """
    texts: list[str] = []
    numbers: set[float] = set()
    flags: set[bool] = set()
    _parsed_values(parsed, texts, numbers, flags)
    exact = set(texts)
    missing = []
    for path, leaf in _leaves(raw):
        if leaf is None or leaf == "":
            continue
        if isinstance(leaf, bool):
            kept = leaf in flags or json.dumps(leaf) in exact
        elif isinstance(leaf, (int, float)):
            kept = float(leaf) in numbers or json.dumps(leaf) in exact
        else:
            moment = _instant(str(leaf))
            kept = str(leaf) in exact or (moment is not None and -moment - 1 in numbers)
        if not kept:
            missing.append(path)
    return missing


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def check_fidelity(
    store: Any, artifact_id: str, traces: Iterable[Trace], source: str = ""
) -> dict[str, Any]:
    """Each sampled step against the raw record it was parsed from.

    Exact: the fields the converter does not consume must sit in
    ``bandits.unmapped`` under their original names, with the same type and
    value. Informational: whether the consumed fields' values appear anywhere
    in the parsed step. The raw record comes from the (redacted) source
    archive through the step's ``bandits.source.record`` pointer, so
    redaction is not a mismatch.
    """
    from bandits.ingest.native import USED_FIELDS, unmapped_fields

    used_for = USED_FIELDS.get(source)
    steps = []
    for trace in traces:
        for step in (*trace.spans, *trace.workflow_nodes):
            pointer = step.attributes.get("bandits.source.record")
            if isinstance(pointer, str):
                try:
                    pointer = json.loads(pointer)
                except ValueError:
                    continue
            if isinstance(pointer, dict):
                steps.append((trace.trace_id, step, pointer))
    if not steps:
        return {"steps": 0}
    try:
        raws = store.read_native_records(artifact_id, [p for _, _, p in steps])
    except (ValueError, FileNotFoundError, LookupError) as exc:
        return {"steps": 0, "error": str(exc)}
    checked = values = carried = 0
    per_step: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for trace_id, step, pointer in steps:
        raw = raws.get(json.dumps(pointer, sort_keys=True))
        if raw is None:
            continue
        checked += 1
        values += sum(1 for _, leaf in _leaves(raw) if leaf is not None and leaf != "")
        wrong: list[str] = []
        if used_for is not None:
            expected = unmapped_fields(raw, used_for(raw))
            actual = step.attributes.get("bandits.unmapped") or {}
            if isinstance(actual, str):
                try:
                    actual = json.loads(actual)
                except ValueError:
                    actual = {"(unreadable)": actual}
            carried += len(expected)
            wrong = sorted(
                key
                for key in set(expected) | set(actual)
                if _canonical(expected.get(key, _ABSENT)) != _canonical(actual.get(key, _ABSENT))
            )
            for key in wrong[: max(0, 50 - len(mismatches))]:
                mismatches.append(
                    {
                        "trace_id": trace_id,
                        "step": step.name,
                        "field": key,
                        "raw": _clip(expected.get(key, "(absent)"), 300),
                        "kept": _clip(actual.get(key, "(absent)"), 300),
                    }
                )
        missing = missing_values(raw, step.model_dump(mode="json"))
        per_step[step.span_id] = {"raw": _clip(raw, 20_000), "missing": missing, "wrong": wrong}
        if missing and len(examples) < 50:
            examples.append({"trace_id": trace_id, "step": step.name, "paths": missing[:20]})
    return {
        "steps": checked,
        "values": values,
        "carried": carried,
        "carried_wrong": sum(len(v["wrong"]) for v in per_step.values()),
        "mismatches": mismatches,
        "exact": used_for is not None,
        "missing": sum(len(v["missing"]) for v in per_step.values()),
        "examples": examples,
        "per_step": per_step,
    }


def fidelity_line(fidelity: dict[str, Any]) -> str:
    if fidelity.get("error"):
        return f"not checked ({fidelity['error']})"
    if not fidelity.get("steps"):
        return "not checked (no step points to a raw record)"
    parts = [f"{fidelity['steps']} stored steps compared with their raw records"]
    if fidelity.get("exact"):
        wrong = fidelity["carried_wrong"]
        parts.append(
            f"{fidelity['carried']} unmapped fields kept exactly"
            if not wrong
            else f"{wrong} unmapped field(s) NOT kept exactly (see inspect.html)"
        )
    found = fidelity["values"] - fidelity["missing"]
    parts.append(
        f"{found}/{fidelity['values']} raw values found in the parsed steps (informational)"
    )
    return "; ".join(parts)


def _misc_counts(traces: Iterable[Trace]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for trace in traces:
        for step in (*trace.spans, *trace.workflow_nodes):
            misc = _misc(step.attributes)
            for key in misc.get("metadata", {}) if isinstance(misc.get("metadata"), dict) else ():
                counts[f"metadata.{key}"] = counts.get(f"metadata.{key}", 0) + 1
            for key in misc:
                if key != "metadata":
                    counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def page_data(
    envelope: dict[str, Any],
    report: dict[str, Any] | None,
    footer: TraceCorpus,
    sample: TraceSample,
    fidelity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the page shows, as plain JSON data."""
    issues = footer.issues
    redactions: dict[str, int] = {}
    for issue in issues:
        if issue.kind == "redaction":
            redactions[issue.detail] = redactions.get(issue.detail, 0) + 1
    workflow = footer.workflow
    shapes = (report or {}).get("shapes") or []
    total = sum(shape.get("traces", 0) for shape in shapes) or 1
    return {
        "envelope": envelope,
        "report": report,
        "workflow": workflow.model_dump(mode="json") if workflow else None,
        "redaction_ruleset": footer.redaction_ruleset,
        "problems": envelope.get("problem_count"),
        "notes": len(issues),
        "redactions": redactions,
        "note_rows": issue_rows(issues, all_redactions=False),
        "shapes": [
            {
                **shape,
                "share": round(100 * shape.get("traces", 0) / total),
                "in_page": shape.get("example_trace_id") in sample.kept,
            }
            for shape in shapes
        ],
        "shape_count": (report or {}).get("shape_count", len(shapes)),
        "traces_seen": sample.seen,
        "traces": [_trace_view(trace) for trace in sample.kept.values()],
        "misc_fields": _misc_counts(sample.kept.values()),
        "fidelity": fidelity or {"steps": 0},
        "fidelity_line": fidelity_line(fidelity or {"steps": 0}),
    }


def render(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    title = f"Ingest · {data['envelope'].get('artifact_id', '')}"
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)


def write_page(
    directory: Path,
    envelope: dict[str, Any],
    report: dict[str, Any] | None,
    footer: TraceCorpus,
    sample: TraceSample,
    fidelity: dict[str, Any] | None = None,
) -> Path:
    """Write ``inspect.html`` into an artifact *directory*; returns its path."""
    path = directory / "inspect.html"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        render(page_data(envelope, report, footer, sample, fidelity)), encoding="utf-8"
    )
    temporary.replace(path)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfaf8;--panel:#fff;--ink:#1d1d1b;--muted:#6b6862;--line:#e6e3dd;--accent:#e8541e;
--model:#2f6fdb;--tool:#7a4fd6;--step:#5d7a3a;--inv:#b8721d;--bad:#c62f2f;--code:#f4f2ee}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe8;--muted:#a19e97;
--line:#33322f;--code:#262523;--model:#7aa7ff;--tool:#b49aff;--step:#a3c47a;--inv:#e8a85a;--bad:#ff7a7a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:20px 24px 0}h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);font-size:12px;word-break:break-all}
nav{display:flex;gap:4px;padding:12px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
nav button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}
nav button.on{border-color:var(--accent);color:var(--accent)}
main{padding:16px 24px 48px;max-width:1200px}section{display:none}section.on{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card b{display:block;font-size:22px}.card span{color:var(--muted);font-size:12px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:12px}
.box h3{margin:0 0 6px;font-size:14px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;font-size:12px}tr.click{cursor:pointer}tr.click:hover{background:var(--code)}
.bar{height:6px;background:var(--line);border-radius:3px;min-width:80px}.bar i{display:block;height:6px;background:var(--accent);border-radius:3px}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid currentColor;margin-right:6px;min-width:64px;text-align:center}
.t-model{color:var(--model)}.t-tool{color:var(--tool)}.t-step{color:var(--step)}.t-invocation{color:var(--inv)}
.err{color:var(--bad)}code,pre{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:8px 10px;border-radius:6px;overflow:auto;max-height:420px;white-space:pre-wrap;word-break:break-word;margin:4px 0}
details.step{margin:2px 0}details.step>summary{cursor:pointer;padding:3px 4px;border-radius:4px;list-style:none}
details.step>summary:hover{background:var(--code)}details.step>summary::before{content:"▸ ";color:var(--muted)}
details.step[open]>summary::before{content:"▾ "}.kids{margin-left:18px;border-left:1px dashed var(--line);padding-left:8px}
.body{margin:4px 0 8px 22px}.label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-top:6px}
.msg{margin:2px 0}.role{font-size:11px;color:var(--muted)}.layout{display:grid;grid-template-columns:minmax(260px,1fr) 2fr;gap:12px}
@media (max-width:800px){.layout{grid-template-columns:1fr}main,header,nav{padding-left:16px;padding-right:16px}}
input[type=search]{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);margin-bottom:8px}
.list{max-height:70vh;overflow:auto}.dur{color:var(--muted);font-size:12px;margin-left:6px}
</style>
</head>
<body>
<header><h1 id="h"></h1><div class="sub" id="sub"></div></header>
<nav><button data-tab="overview" class="on">Overview</button><button data-tab="shapes">Shapes</button>
<button data-tab="traces">Traces</button><button data-tab="notes">Notes</button></nav>
<main>
<section id="overview" class="on"></section><section id="shapes"></section>
<section id="traces"></section><section id="notes"></section>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const E = D.envelope, R = D.report || {};
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = v => typeof v === "string" ? v : JSON.stringify(v, null, 2);
const pre = v => v === null || v === undefined || v === "" ? '<span class="muted">none</span>' : `<pre>${esc(fmt(v))}</pre>`;
const el = id => document.getElementById(id);
function tab(name) {
  document.querySelectorAll("nav button, section").forEach(x => x.classList.remove("on"));
  document.querySelector(`nav button[data-tab="${name}"]`).classList.add("on"); el(name).classList.add("on");
}
document.querySelectorAll("nav button").forEach(b => b.onclick = () => tab(b.dataset.tab));
el("h").textContent = `${E.source} ingest · ${E.artifact_id}`;
el("sub").textContent = `${E.source_path} · ${E.created_at} · bandits ${E.bandits_version} @ ${(E.git_commit||"").slice(0,8)}${E.git_dirty?" (dirty)":""}`;

// Overview
const B = R.buckets || {};
const card = (n, label) => `<div class="card"><b>${esc(n)}</b><span>${esc(label)}</span></div>`;
const redactions = Object.entries(D.redactions).map(([d,n]) => `${n} × ${esc(d.replace("redacted ",""))}`).join("<br>") || "none";
const wf = D.workflow;
el("overview").innerHTML = `
<div class="cards">${card(E.trace_count,"traces")}${card(E.span_count,"steps kept as spans")}
${card(B.model ?? "–","model calls")}${card(D.problems,"problems (saved anyway)")}
${card(D.notes,"notes (informational)")}${card(D.shape_count ?? "–","trace shapes")}</div>
<div class="box"><h3>Raw vs parsed</h3><div class="${D.fidelity.carried_wrong ? "err" : ""}">${esc(D.fidelity_line)}</div>
${(D.fidelity.mismatches||[]).slice(0,10).map(m => `<div class="err"><code>${esc(m.trace_id.slice(0,12))}</code> ${esc(m.step)} · <code>${esc(m.field)}</code>: raw ${esc(JSON.stringify(m.raw))} → kept ${esc(JSON.stringify(m.kept))}</div>`).join("")}
<div class="muted">Each stored step in the traces on this page is compared with the raw record it was parsed from (in the redacted source copy).
<b>Exact:</b> every field the converter does not interpret must be kept under Misc with its original name, type and value.
<b>Informational:</b> how many raw values appear anywhere in the parsed step; a value can be found by coincidence, so this is a signal, not proof.
The run's own record (the invocation) is kept as the trace's request, not as a step, and is not compared here. Open a step to see its raw record.</div>
${(D.fidelity.examples||[]).slice(0,5).map(e => `<div class="muted"><code>${esc(e.trace_id.slice(0,12))}</code> ${esc(e.step)}: not found anywhere: ${e.paths.map(esc).join(", ")}</div>`).join("")}</div>
${D.misc_fields && D.misc_fields.length ? `<div class="box"><h3>Misc fields (kept, not interpreted)</h3><div class="muted">Fields Bandits keeps under their original names without giving them a meaning, in the steps on this page. Open a step to read them.</div>
<table><tr><th>field</th><th>steps</th></tr>${D.misc_fields.map(([k,n]) => `<tr><td><code>${esc(k)}</code></td><td>${n}</td></tr>`).join("")}</table></div>` : ""}
<div class="box"><h3>Problems vs notes</h3><div class="muted">A <b>problem</b> is something that may make part of the data wrong or incomplete: a run whose question could not be read, a model call with no recorded reply. The corpus is still saved; ingest refuses to save only when nothing usable was read or the records do not add up.
A <b>note</b> is recorded for information: a hidden value, a parent the export did not include. ${D.problems ? "" : "This corpus has no problems."}</div></div>
${R.spans_seen !== undefined ? `<div class="box"><h3>Every record accounted for</h3>
<div>${R.spans_seen} records seen → ${Object.entries(B).filter(([,n])=>n).map(([k,n])=>`${n} ${esc(k.replace("_"," "))}`).join(" · ")} · <b>${R.dropped} dropped</b></div>
${R.accounting_errors && R.accounting_errors.length ? `<div class="err">${R.accounting_errors.map(esc).join("<br>")}</div>` : ""}
${R.traces_with_absent_parents ? `<div class="muted">${R.traces_with_absent_parents} trace(s) have top-level steps whose parent was not exported (max ${R.max_top_steps} per trace).</div>` : ""}</div>` : ""}
${wf ? `<div class="box"><h3>Fields read</h3><table>
<tr><th>question (task)</th><td><code>${esc((wf.task_fields||[]).join(", ") || "none")}</code></td></tr>
<tr><th>answer (delivered)</th><td><code>${esc(wf.delivered_field || "none")}</code></td></tr>
${wf.mapping_name ? `<tr><th>mapping</th><td><code>${esc(wf.mapping_name)}</code></td></tr>` : ""}</table></div>` : ""}
<div class="box"><h3>Redaction (${esc(D.redaction_ruleset || "")})</h3><div>${redactions}</div></div>
${R.evidence_links ? `<div class="box"><h3>Evidence links between steps</h3><div>${Object.entries(R.evidence_links).map(([k,n])=>`${n} ${esc(k.replace("_"," "))}`).join(" · ")}</div></div>` : ""}
<div class="box"><h3>Files beside this page</h3><table>
<tr><td><code>corpus.json</code></td><td>the parsed traces</td></tr>
<tr><td><code>report.json</code></td><td>the accounting, shapes and evidence counts above</td></tr>
<tr><td><code>envelope.json</code></td><td>summary: source, counts, code version</td></tr>
<tr><td><code>source/</code>, <code>source-manifest.json</code></td><td>the redacted original export and its checksums</td></tr></table></div>`;

// Trace tree
function stepBody(s) {
  let h = '<div class="body">';
  if (s.model) h += `<div class="label">model</div><code>${esc(s.model)}</code>`;
  if (s.input_messages) h += `<div class="label">input messages</div>` + s.input_messages.map(m => `<div class="msg"><span class="role">${esc(m.role)}</span>${pre(m.text)}</div>`).join("");
  else if (s.input !== undefined && s.input !== null) h += `<div class="label">input</div>${pre(s.input)}`;
  if (s.output_messages) h += `<div class="label">output messages</div>` + s.output_messages.map(m => `<div class="msg"><span class="role">${esc(m.role)}</span>${pre(m.text)}</div>`).join("");
  else h += `<div class="label">output</div>${pre(s.output)}`;
  h += `<details><summary class="label">all parsed fields (${Object.keys(s.attributes||{}).length})</summary>${pre(s.attributes)}</details>`;
  const misc = s.misc || {};
  if (Object.keys(misc).length) h += `<details><summary class="label">misc: kept, not interpreted (${Object.keys(misc).length})</summary>${pre(misc)}</details>`;
  const f = (D.fidelity.per_step || {})[s.id];
  if (f) h += (f.wrong.length ? `<div class="label err">unmapped fields not kept exactly (${f.wrong.length})</div><pre class="err">${esc(f.wrong.join("\n"))}</pre>` : "")
    + (f.missing.length ? `<div class="label">raw values not found anywhere in the parsed step (informational, ${f.missing.length})</div><pre>${esc(f.missing.join("\n"))}</pre>` : "")
    + `<details><summary class="label">raw record (as exported, redacted)</summary>${pre(f.raw)}</details>`;
  h += `</div>`;
  return h;
}
function tree(t) {
  const ids = new Set(t.steps.map(s => s.id)), kids = {};
  t.steps.forEach(s => { const p = ids.has(s.parent) ? s.parent : ""; (kids[p] = kids[p] || []).push(s); });
  Object.values(kids).forEach(a => a.sort((x, y) => (x.start || "").localeCompare(y.start || "")));
  // The tree is always shown in full; a click opens one step's input, output and fields.
  const node = s => `<details class="step"><summary><span class="badge t-${esc(s.type)}">${esc(s.type)}</span>${esc(s.name)}
${s.status !== "ok" ? `<span class="err"> ${esc(s.status)}</span>` : ""}${s.ms !== null && s.ms !== undefined ? `<span class="dur">${s.ms} ms</span>` : ""}</summary>
${stepBody(s)}</details>${kids[s.id] ? `<div class="kids">${kids[s.id].map(node).join("")}</div>` : ""}`;
  return (kids[""] || []).map(node).join("");
}
function traceDetail(t) {
  return `<div class="box"><h3>Trace <code>${esc(t.trace_id)}</code></h3><table>
<tr><th>question</th><td>${t.task ? esc(t.task) : '<span class="muted">not found</span>'}${t.task_path ? ` <span class="muted">from <code>${esc(t.task_path)}</code></span>` : ""}
${t.task_reason ? `<div class="muted">${esc(t.task_reason)}</div>` : ""}
${t.invocation_basis ? `<div class="muted">run: ${esc(t.invocation_basis)}</div>` : ""}
${t.task_candidates.length > 1 ? `<details><summary class="label">${t.task_status === "conflict" ? "different questions recorded (none chosen)" : "recorded in " + t.task_candidates.length + " fields (same text)"}</summary>
<table>${t.task_candidates.map(c => `<tr><td><code>${esc(c.path)}</code><div class="muted"><code>${esc(c.span_id)}</code></div></td><td>${esc(c.value)}</td></tr>`).join("")}</table></details>` : ""}</td></tr>
<tr><th>answer</th><td>${pre(t.delivered)}</td></tr>
${t.system_prompt ? `<tr><th>system prompt</th><td>${pre(t.system_prompt)}</td></tr>` : ""}
<tr><th>steps</th><td>${t.steps.length} (${["model","tool","step","invocation"].map(k => `${t.steps.filter(s=>s.type===k).length} ${k}`).join(", ")}) · ${t.evidence} evidence links</td></tr></table>
<div class="label">step tree (click a step to see its input, output and fields)</div>${tree(t)}</div>`;
}
const byId = Object.fromEntries(D.traces.map(t => [t.trace_id, t]));

// Shapes
el("shapes").innerHTML = `<div class="box muted">A shape is one layout of steps. Traces with the same shape took the same path through the app,
so different versions of an app (or different routes through it) show up as different shapes. Every shape found is listed (${D.shape_count}),
with where its question was read and its step outline (a repeated step shows as ×N).</div>
<div class="layout"><div class="box list"><table><tr><th>shape</th><th>traces</th><th>question read from</th></tr>
${D.shapes.map((s, i) => `<tr class="click" data-shape-index="${i}"><td><code>${esc(s.shape_id)}</code><div class="bar"><i style="width:${s.share}%"></i></div>
<div class="muted">${s.model_calls} model calls</div></td><td>${s.traces} (${s.share}%)</td>
<td>${Object.keys(s.task_paths||{}).length ? Object.entries(s.task_paths).map(([p,n])=>`<code>${esc(p)}</code> ${n}`).join("<br>") : `<span class="muted">${esc(Object.entries(s.task_status||{}).map(([k,n])=>`${n} ${k}`).join(", ") || "none")}</span>`}</td></tr>`).join("")}</table></div>
<div id="shapeview"></div></div>`;
function showShape(i) {
  const s = D.shapes[i], t = byId[s.example_trace_id];
  el("shapeview").innerHTML = `<div class="box"><h3>Shape <code>${esc(s.shape_id)}</code> · ${s.traces} trace(s)</h3>
<div class="label">outline</div>${s.outline ? pre(s.outline.join("\n")) : '<span class="muted">not saved for this shape</span>'}
<div class="muted">example trace <code>${esc(s.example_trace_id)}</code>${t ? "" : " (not among the traces in this page; <code>bandits inspect</code> adds every shape's example)"}</div></div>`
  + (t ? traceDetail(t) : "");
}
document.querySelectorAll("[data-shape-index]").forEach(r => r.onclick = () => showShape(+r.dataset.shapeIndex));
if (D.shapes.length) showShape(0);

// Traces
el("traces").innerHTML = `<div class="box muted">${D.traces.length} of ${E.trace_count} traces are in this page.</div>
<div class="layout"><div class="box list"><input type="search" id="q" placeholder="filter by question or id"><table id="tl"></table></div>
<div id="traceview"><div class="box muted">Pick a trace.</div></div></div>`;
function listTraces(q) {
  q = (q || "").toLowerCase();
  el("tl").innerHTML = `<tr><th>question</th><th>steps</th></tr>` + D.traces.filter(t => !q || (t.task||"").toLowerCase().includes(q) || t.trace_id.includes(q))
    .map(t => `<tr class="click" data-trace="${esc(t.trace_id)}"><td>${t.task ? esc(t.task.slice(0,90)) : '<span class="muted">no question found</span>'}<div class="muted"><code>${esc(t.trace_id)}</code></div></td><td>${t.steps.length}</td></tr>`).join("");
  document.querySelectorAll("[data-trace]").forEach(r => r.onclick = () => { el("traceview").innerHTML = traceDetail(byId[r.dataset.trace]); });
}
listTraces(""); el("q").oninput = e => listTraces(e.target.value);

// Links: #shapes, #notes, #trace=<id> (opens that trace)
function follow() {
  const h = decodeURIComponent(location.hash.slice(1));
  if (h.startsWith("trace=") && byId[h.slice(6)]) { tab("traces"); el("traceview").innerHTML = traceDetail(byId[h.slice(6)]); }
  else if (["overview","shapes","traces","notes"].includes(h)) tab(h);
}
window.addEventListener("hashchange", follow);

// Notes
el("notes").innerHTML = `<div class="box muted">Notes are informational; redactions are counted per kind of value.
<code>bandits show ${esc(E.artifact_id)} --issues --all-redactions</code> lists every one.</div>
<div class="box"><table><tr><th>kind</th><th>where</th><th>detail</th></tr>
${D.note_rows.map(([k,l,d]) => `<tr><td><code>${esc(k)}</code></td><td class="muted">${esc(l)}</td><td>${esc(d)}</td></tr>`).join("") || '<tr><td colspan="3" class="muted">none</td></tr>'}</table></div>`;
follow();
</script>
</body>
</html>
"""
