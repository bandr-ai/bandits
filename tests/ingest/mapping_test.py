"""Confirmed ingest mappings: propose, confirm, apply, and flag what they don't cover."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from bandits.cli import app
from bandits.ingest.mapping import (
    Identity,
    IngestMapping,
    MappingError,
    ShapeRef,
    applicable,
    confirm,
    load_mapping,
    mapping_path,
)
from bandits.ingest.otlp_standard import load_otlp_standard
from bandits.ingest.report import IngestReport
from bandits.store import ArtifactStore
from bandits.traces import SpanKind, WorkflowDeclaration
from tests.ingest.discovery_test import _new, _old
from tests.ingest.otlp_standard_test import _request, _span, _write

runner = CliRunner()
SPAN = "langfuse.observation.type=SPAN"
CHAIN = "langfuse.observation.type=CHAIN"


def _out(result) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)


def _cli(*args: str):
    return runner.invoke(app, list(args))


def _file(tmp_path, *traces):
    return _write(tmp_path / "export.jsonl", *(_request(t) for t in traces))


def _mapping(**fields) -> IngestMapping:
    return confirm(IngestMapping(source="otlp-std", task_fields=("input.query",), **fields))


def _load(path, mapping, **declared):
    report = IngestReport()
    corpus = load_otlp_standard(
        path,
        workflow=WorkflowDeclaration(
            task_fields=declared.get("task_fields", mapping.task_fields),
            delivered_field=declared.get("delivered_field", mapping.delivered_field),
            mapping_name="m",
            mapping_digest=mapping.confirmed_digest,
        ),
        report=report,
        mapping=mapping,
    )
    return corpus, report


# ---- propose / confirm / show


def test_propose_a_single_shape_export(tmp_path) -> None:
    path = _file(tmp_path, _old("1" * 32), _old("2" * 32))
    result = _cli(
        "mapping",
        "propose",
        str(path),
        "--source",
        "otlp-std",
        "--name",
        "shop",
        "--project",
        str(tmp_path),
    )
    assert result.exit_code == 0, _out(result)
    mapping = load_mapping(tmp_path, "shop")
    assert mapping.task_fields == ("input.query",)
    assert mapping.delivered_field == "output.answer"
    assert mapping.invocation == (Identity(kind_label=SPAN, name="handle"),)
    assert [(s.trace_count, s.example_trace_id) for s in mapping.shapes] == [(2, "1" * 32)]
    assert not mapping.confirmed
    confirmed = _cli("mapping", "confirm", "shop", "--project", str(tmp_path))
    assert confirmed.exit_code == 0, _out(confirmed)
    shown = _cli("mapping", "show", "shop", "--project", str(tmp_path))
    assert "confirmed, unmodified" in _out(shown)


def test_agreeing_candidates_need_no_invocation_and_an_explicit_one_still_works(
    tmp_path,
) -> None:
    path = _file(tmp_path, _new("1" * 32), _new("2" * 32))
    project = str(tmp_path)
    result = _cli(
        "mapping", "propose", str(path), "--source", "otlp-std", "--name", "m", "--project", project
    )
    out = _out(result)
    assert result.exit_code == 0, out
    # Both runs hold the same question: the fields are declared together.
    assert load_mapping(tmp_path, "m").task_fields == ("input.payload.query", "input.question")
    assert _cli("mapping", "confirm", "m", "--project", project).exit_code == 0
    ingested = _cli(
        "ingest", str(path), "--source", "otlp-std", "--mapping", "m", "--project", project
    )
    assert ingested.exit_code == 0, _out(ingested)
    artifact = re.search(r"artifact_id: (\S+)", _out(ingested)).group(1)
    corpus = ArtifactStore(tmp_path / ".bandits").read(artifact)
    assert [t.request.task_status for t in corpus.traces] == ["declared", "declared"]

    again = _cli(
        "mapping",
        "propose",
        str(path),
        "--source",
        "otlp-std",
        "--name",
        "m",
        "--invocation",
        f"{SPAN}|handle",
        "--force",
        "--project",
        project,
    )
    assert again.exit_code == 0, _out(again)
    mapping = load_mapping(tmp_path, "m")
    assert mapping.task_fields == ("input.payload.query",)
    assert mapping.invocation == (Identity(kind_label=SPAN, name="handle"),)
    assert _cli("mapping", "confirm", "m", "--project", project).exit_code == 0
    mapping = load_mapping(tmp_path, "m")

    ingested = _cli(
        "ingest", str(path), "--source", "otlp-std", "--mapping", "m", "--project", project
    )
    assert ingested.exit_code == 0, _out(ingested)
    artifact = re.search(r"artifact_id: (\S+)", _out(ingested)).group(1)
    corpus = ArtifactStore(tmp_path / ".bandits").read(artifact)
    assert [t.request.task_status for t in corpus.traces] == ["declared", "declared"]
    assert {t.request.invocation_basis for t in corpus.traces} == {"mapping m: invocation identity"}
    assert corpus.workflow.mapping_name == "m"
    assert corpus.workflow.mapping_digest == mapping.confirmed_digest


def test_propose_never_overwrites_without_force(tmp_path) -> None:
    path = _file(tmp_path, _old("1" * 32))
    args = [
        "mapping",
        "propose",
        str(path),
        "--source",
        "otlp-std",
        "--name",
        "m",
        "--project",
        str(tmp_path),
    ]
    assert _cli(*args).exit_code == 0
    assert _cli("mapping", "confirm", "m", "--project", str(tmp_path)).exit_code == 0
    refused = _cli(*args)
    assert refused.exit_code == 1 and "already exists" in _out(refused)
    assert load_mapping(tmp_path, "m").confirmed
    assert _cli(*args, "--force").exit_code == 0
    assert not load_mapping(tmp_path, "m").confirmed  # replacing clears confirmation


@pytest.mark.parametrize("name", ["Upper", "-dash", "has space", "a" * 65, "../x", ""])
def test_bad_names_are_refused(tmp_path, name) -> None:
    with pytest.raises(MappingError):
        mapping_path(tmp_path, name)


def test_edited_unconfirmed_and_wrong_source_mappings_are_refused() -> None:
    mapping = _mapping()
    assert applicable(mapping, "m", "otlp-std") is mapping
    with pytest.raises(MappingError, match="not confirmed"):
        applicable(mapping.replace(confirmed=False), "m", "otlp-std")
    with pytest.raises(MappingError, match="changed since it was confirmed"):
        applicable(mapping.replace(task_fields=("input.text",)), "m", "otlp-std")
    with pytest.raises(MappingError, match="--source otlp-std, not langfuse"):
        applicable(mapping, "m", "langfuse")
    # What confirm writes is outside the digest.
    assert mapping.replace(bandits_version="9.9", derivation_version=7).unmodified


def test_an_invalid_file_is_an_error_never_partly_applied(tmp_path) -> None:
    path = mapping_path(tmp_path, "m")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"source": "otlp-std", "step_kinds": {"k|n": "model"}}))
    with pytest.raises(MappingError, match="not valid"):
        load_mapping(tmp_path, "m")


def test_cli_refuses_an_edited_mapping_and_flags_still_override(tmp_path) -> None:
    path = _file(tmp_path, _old("1" * 32))
    project = str(tmp_path)
    _cli(
        "mapping", "propose", str(path), "--source", "otlp-std", "--name", "m", "--project", project
    )
    _cli("mapping", "confirm", "m", "--project", project)
    flagged = _cli(
        "ingest",
        str(path),
        "--source",
        "otlp-std",
        "--mapping",
        "m",
        "--task-field",
        "input.missing",
        "--dry-run",
        "--project",
        project,
    )
    assert flagged.exit_code == 0 and "task:     input.missing" in _out(flagged)
    assert "overridden by flags" in " ".join(_out(flagged).split())
    assert "1 trace(s) have no task" in _out(flagged)

    file = mapping_path(tmp_path, "m")
    file.write_text(file.read_text().replace('"input.query"', '"input.text"'))
    refused = _cli(
        "ingest",
        str(path),
        "--source",
        "otlp-std",
        "--mapping",
        "m",
        "--dry-run",
        "--project",
        project,
    )
    assert refused.exit_code == 1
    assert "changed since it was confirmed" in " ".join(_out(refused).split())
    wrong = _cli(
        "ingest",
        str(path),
        "--source",
        "langfuse",
        "--mapping",
        "m",
        "--dry-run",
        "--project",
        project,
    )
    assert wrong.exit_code == 1


def test_ingest_without_a_mapping_hints_how_to_save_one(tmp_path) -> None:
    path = _file(tmp_path, _old("1" * 32))
    result = _cli("ingest", str(path), "--source", "otlp-std", "--dry-run")
    assert "to save these choices: bandits mapping propose" in _out(result)


# ---- applying


def test_an_identity_matching_two_candidates_chooses_nothing(tmp_path) -> None:
    trace = _old("1" * 32) + [
        _span(
            "twin",
            "handle",
            {"langfuse.observation.type": "SPAN", "input.value": json.dumps({"query": "other"})},
            parent="gone-2",
            trace="1" * 32,
        )
    ]
    path = _file(tmp_path, trace)
    corpus, _ = _load(path, _mapping(invocation=(Identity(kind_label=SPAN, name="handle"),)))
    assert corpus.traces[0].request.source_span_id is None
    (issue,) = [i for i in corpus.issues if i.kind == "ambiguous_invocation"]
    assert "mapping m matches 2 of 2" in issue.detail and "run, twin" in issue.detail


def test_mixed_shapes_resolve_with_one_mapping(tmp_path) -> None:
    traces = [_old("1" * 32), _new("2" * 32), _old("3" * 32), _new("4" * 32)]
    path = _file(tmp_path, *traces)
    mapping = _mapping(
        invocation=(Identity(kind_label=SPAN, name="handle"),),
        delivered_field="output.answer",
    ).replace(task_fields=("input.query", "input.payload.query"))
    corpus, _ = _load(path, confirm(mapping))
    assert [t.request.task_status for t in corpus.traces] == ["declared"] * 4
    assert all(t.request.delivered == "Yes, until it ships." for t in corpus.traces)


def test_a_task_only_mapping_is_valid() -> None:
    assert confirm(IngestMapping(source="otlp-std", task_fields=("input",))).delivered_field is None


def _pipeline(trace: str = "1" * 32) -> list[dict]:
    def attrs(kind: str, **io: object) -> dict:
        return {
            "langfuse.observation.type": kind,
            **{f"{k}.value": json.dumps(v) for k, v in io.items()},
        }

    return [
        _span("run", "handle", attrs("SPAN", input={"query": "q"}), parent="gone", trace=trace),
        _span(
            "rank",
            "rank",
            {
                "langfuse.observation.type": "GENERATION",
                "input.value": "q",
                "output.value": "[1, 2]",
            },
            parent="run",
            at=1,
            trace=trace,
        ),
        _span("plan", "plan", attrs("CHAIN"), parent="run", at=2, trace=trace),
        _span(
            "draft",
            "draft",
            {
                "langfuse.observation.type": "GENERATION",
                "input.value": "q",
                "output.value": "a draft",
            },
            parent="plan",
            at=2,
            trace=trace,
        ),
        _span("audit", "audit", attrs("CHAIN"), parent="run", at=3, trace=trace),
        _span(
            "judge",
            "judge",
            {"langfuse.observation.type": "GENERATION", "input.value": "q", "output.value": "fine"},
            parent="audit",
            at=3,
            trace=trace,
        ),
        _span(
            "answer",
            "answer",
            {
                "langfuse.observation.type": "GENERATION",
                "input.value": "q",
                "output.value": "final",
            },
            parent="run",
            at=4,
            trace=trace,
        ),
    ]


def test_step_kind_overrides(tmp_path) -> None:
    path = _file(tmp_path, _pipeline())
    kinds = {
        "langfuse.observation.type=GENERATION|rank": "tool",  # nothing beneath: applies
        "langfuse.observation.type=CHAIN|plan": "tool",  # a call beneath: not applicable
        "langfuse.observation.type=GENERATION|answer": "step",
        "langfuse.observation.type=CHAIN|audit": "exclude",
    }
    corpus, report = _load(path, _mapping(step_kinds=kinds))
    trace = corpus.traces[0]
    spans = {s.span_id: s for s in trace.spans}
    assert spans["rank"].kind is SpanKind.TOOL and not spans["rank"].call_recorded
    assert spans["draft"].kind is SpanKind.MODEL  # plan stayed structure
    assert "plan" in {n.span_id for n in trace.workflow_nodes}
    assert "answer" not in spans and "answer" in {n.span_id for n in trace.workflow_nodes}
    assert "audit" not in spans and "judge" not in spans
    assert not {"audit", "judge"} & {n.span_id for n in trace.workflow_nodes}
    assert report.buckets["excluded"] == 2
    kinds_found = {i.kind: i for i in corpus.issues}
    assert "2 span(s) of langfuse.observation.type=CHAIN|audit" in (
        kinds_found["excluded_by_mapping"].detail
    )
    assert "CHAIN|plan" in kinds_found["mapping_override_not_applicable"].detail
    assert "excluded_evaluator" not in kinds_found
    assert sum(report.buckets.values()) == report.spans_seen


def test_a_shape_the_mapping_was_not_confirmed_on_is_flagged_not_dropped(tmp_path) -> None:
    path = _file(tmp_path, _old("1" * 32), _new("2" * 32))
    old_only = load_otlp_standard(_file(tmp_path, _old("9" * 32)), workflow=WorkflowDeclaration())
    assert old_only.traces
    from bandits.ingest.discovery import discover_requests

    known = {
        t.shape_id for t in discover_requests(_file(tmp_path, _old("9" * 32)), "otlp-std").traces
    }
    mapping = _mapping(
        invocation=(Identity(kind_label=SPAN, name="handle"),),
        shapes=tuple(ShapeRef(shape_id=s, example_trace_id="x", trace_count=1) for s in known),
    )
    path = _file(tmp_path, _old("1" * 32), _new("2" * 32))
    corpus, report = _load(path, mapping)
    assert report.unmapped_shapes == 1
    (issue,) = [i for i in corpus.issues if i.kind == "shape_not_in_mapping"]
    assert issue.detail.startswith("1 trace(s) have shapes mapping m")
    assert len(corpus.traces) == 2


def test_drift_compares_the_export_before_overrides(tmp_path) -> None:
    path = _file(tmp_path, _pipeline())
    plain = IngestReport()
    load_otlp_standard(path, workflow=WorkflowDeclaration(), report=plain)
    (shape,) = plain.shapes
    mapping = _mapping(
        step_kinds={"langfuse.observation.type=CHAIN|audit": "exclude"},
        shapes=(ShapeRef(shape_id=shape, example_trace_id="1" * 32, trace_count=1),),
    )
    _, report = _load(path, mapping)
    assert report.unmapped_shapes == 0 and set(report.shapes) == {shape}


def _shared_name(trace: str) -> list[dict]:
    # Two top-level runs with one name; only one records the request field.
    return [
        _span(
            "a",
            "run",
            {"langfuse.observation.type": "SPAN", "input.value": json.dumps({"query": "q"})},
            parent="gone-1",
            trace=trace,
        ),
        _span(
            "b",
            "run",
            {"langfuse.observation.type": "SPAN", "input.value": json.dumps({"other": "x"})},
            parent="gone-2",
            trace=trace,
        ),
        _span(
            "m",
            "call",
            {"gen_ai.operation.name": "chat", "input.value": "q", "output.value": "a"},
            parent="a",
            trace=trace,
        ),
    ]


def test_a_proposal_applies_as_it_was_resolved(tmp_path) -> None:
    path = _file(tmp_path, _shared_name("1" * 32), _shared_name("2" * 32))
    project = str(tmp_path)
    proposed = _cli(
        "mapping", "propose", str(path), "--source", "otlp-std", "--name", "m", "--project", project
    )
    assert "invocation left open" in _out(proposed)
    mapping = load_mapping(tmp_path, "m")
    assert mapping.task_fields == ("input.query",) and mapping.invocation == ()
    assert _cli("mapping", "confirm", "m", "--project", project).exit_code == 0
    corpus, _ = _load(path, load_mapping(tmp_path, "m"))
    assert [t.request.source_span_id for t in corpus.traces] == ["a", "a"]


def test_an_explicit_invocation_that_cannot_choose_is_warned(tmp_path) -> None:
    path = _file(tmp_path, _shared_name("1" * 32))
    result = _cli(
        "mapping",
        "propose",
        str(path),
        "--source",
        "otlp-std",
        "--name",
        "m",
        "--invocation",
        f"{SPAN}|run",
        "--project",
        str(tmp_path),
    )
    assert "picks exactly one run in only 0/1 traces" in " ".join(_out(result).split())


@pytest.mark.parametrize("key", ["invalid-key", "|name", "label|", ""])
def test_step_kind_keys_must_name_an_identity(tmp_path, key) -> None:
    with pytest.raises(ValueError, match="KIND_LABEL"):
        IngestMapping(source="otlp-std", step_kinds={key: "exclude"})
    path = mapping_path(tmp_path, "m")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": "otlp-std", "step_kinds": {key: "exclude"}}))
    with pytest.raises(MappingError, match="not valid"):
        load_mapping(tmp_path, "m")


def test_a_key_that_matches_nothing_is_reported(tmp_path) -> None:
    path = _file(tmp_path, _pipeline())
    corpus, _ = _load(path, _mapping(step_kinds={"kind=SPAN|absent": "exclude"}))
    (issue,) = [i for i in corpus.issues if i.kind == "mapping_key_unmatched"]
    assert "kind=SPAN|absent" in issue.detail
