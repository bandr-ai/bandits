"""Workflow mode: a program-driven trace read as recorded, not as a conversation."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from bandits.analyze.rlm_corpus import build_view
from bandits.analyze.rlm_models import TraceView
from bandits.cli import app
from bandits.ingest import load_corpus
from bandits.ingest.otlp_standard import load_otlp_standard
from bandits.ingest.workflow import delivery_status, resolve_task
from bandits.traces import WorkflowDeclaration
from bandits.verify.turns import WorkflowTurnsError, extract_turns
from tests.ingest.otlp_standard_test import _request, _span, _write

QUESTION = "If an employee leaves mid month, does health insurance stop on the last day?"
HYDE = "Under the Benefits Continuity Policy, coverage ends on the last day of the month of exit."
FOCUSED = '{"focused_queries": ["health coverage end date after termination mid month"]}'
ANSWER = "Coverage ends on the last day of the month you leave, unless you elect COBRA."

DECLARED = WorkflowDeclaration(
    task_fields=("input.query", "input.payload.query"), delivered_field="output.answer"
)


def _gen(span_id: str, name: str, prompt: str, output: str, *, parent: str, at: int) -> dict:
    messages = [
        {"role": "system", "content": f"You are the {name} stage."},
        {"role": "user", "content": prompt},
    ]
    return _span(
        span_id,
        name,
        {
            "langfuse.observation.type": "GENERATION",
            "input.value": json.dumps(messages),
            "output.value": output,
        },
        parent=parent,
        at=at,
    )


def _chain(span_id: str, name: str, *, parent: str, at: int, step: int | None = None, **io) -> dict:
    attributes = {"langfuse.observation.type": "CHAIN"}
    if step is not None:
        attributes["metadata.langgraph_step"] = step
        attributes["metadata.langgraph_node"] = name
    for key, value in io.items():
        attributes[f"{key}.value"] = json.dumps(value)
    return _span(span_id, name, attributes, parent=parent, at=at)


def _workflow(
    *, container: bool = True, framework: bool = True, invocation: dict | None = None
) -> list[dict]:
    step = (lambda n: n) if framework else (lambda n: None)
    spans = [
        _span(
            "inv",
            "generate_answer",
            {
                "langfuse.observation.type": "SPAN",
                "input.value": json.dumps(
                    invocation if invocation is not None else {"query": QUESTION}
                ),
                "output.value": json.dumps({"answer": ANSWER, "category": "benefits"}),
            },
            parent="box" if container else None,
        ),
        _chain("graph", "LangGraph", parent="inv", at=0),
        _chain(
            "n_hyde", "hyde", parent="graph", at=1, step=step(2), output={"hyde_document": HYDE}
        ),
        _gen(
            "m_hyde",
            "HYDE",
            f"Employee question: {QUESTION}\n\nWrite the passage:",
            HYDE,
            parent="n_hyde",
            at=1,
        ),
        _chain("n_focused", "focused", parent="graph", at=1, step=step(2)),
        _gen(
            "m_focused",
            "FOCUSED",
            f"Original question: {QUESTION}",
            FOCUSED,
            parent="n_focused",
            at=1,
        ),
        _chain(
            "merge",
            "merge_expansions",
            parent="graph",
            at=3,
            step=step(3),
            input={
                "hyde_document": HYDE,
                "focused_queries": json.loads(FOCUSED)["focused_queries"],
            },
        ),
        _gen(
            "m_answer",
            "ANSWER",
            f"Question: {QUESTION}\n[DOCUMENT] ...",
            ANSWER,
            parent="inv",
            at=4,
        ),
    ]
    if container:
        spans.insert(
            0, _span("box", "generate_answer", {"langfuse.synthetic_span": "trace_record"})
        )
    return spans


def _load(tmp_path, spans, declaration=DECLARED):
    corpus = load_otlp_standard(
        _write(tmp_path / "wf.jsonl", _request(spans)), workflow=declaration
    )
    assert len(corpus.traces) == 1
    return corpus, corpus.traces[0]


def _links(trace, kind):
    return [link for link in trace.evidence if link.kind == kind]


def test_task_is_the_invocation_request_never_a_model_prompt(tmp_path) -> None:
    corpus, trace = _load(tmp_path, _workflow())

    assert corpus.workflow == DECLARED
    assert trace.interaction == "workflow"
    assert trace.task == QUESTION
    assert trace.user_turns == ()
    request = trace.request
    assert request.source_span_id == "inv"
    assert request.invocation_basis == "sole outermost non-container span"
    assert (request.task_status, request.task_path) == ("declared", "input.query")
    assert request.delivered == ANSWER
    assert request.raw_output == {"answer": ANSWER, "category": "benefits"}


def test_conversation_mode_is_unchanged_on_the_same_export(tmp_path) -> None:
    path = _write(tmp_path / "wf.jsonl", _request(_workflow()))
    trace = load_otlp_standard(path).traces[0]

    # The existing reading, kept as it was: the first prompt is taken as the task.
    assert trace.interaction == "conversation"
    assert trace.request is None
    assert trace.task.startswith("Employee question:")
    assert trace.user_turns


def test_steps_containing_calls_are_kept_as_structure_not_actions(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())

    nodes = {node.name: node for node in trace.workflow_nodes}
    assert {"LangGraph", "hyde", "focused"} <= set(nodes)
    assert nodes["hyde"].output == {"hyde_document": HYDE}
    assert nodes["hyde"].framework["langgraph_step"] == 2
    # The invocation is the request record, the container is nothing: neither is a node.
    assert "inv" not in {n.span_id for n in trace.workflow_nodes}
    assert "box" not in {n.span_id for n in trace.workflow_nodes}
    assert not {n.span_id for n in trace.workflow_nodes} & {s.span_id for s in trace.spans}


def test_evidence_links_record_their_basis(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())

    enclosing = {
        (link.call_span_id, link.target_span_id) for link in _links(trace, "enclosing_result")
    }
    assert ("m_hyde", "n_hyde") in enclosing and ("m_focused", "n_focused") in enclosing

    rounds = {(link.call_span_id, link.target_span_id) for link in _links(trace, "same_round")}
    assert rounds == {("m_hyde", "m_focused"), ("m_focused", "m_hyde")}
    assert all("not proof" in link.basis for link in _links(trace, "same_round"))

    merged = {
        link.call_span_id: link
        for link in _links(trace, "shared_result")
        if link.target_span_id == "merge"
    }
    assert set(merged) == {"m_hyde", "m_focused"}
    assert merged["m_hyde"].shared_with == ("m_focused",)
    assert not merged["m_hyde"].ambiguous

    delivery = _links(trace, "delivery")
    assert [link.call_span_id for link in delivery] == ["m_answer"]
    assert delivery_status(trace.request, trace.evidence) == "delivered by m_answer"


def test_nothing_later_is_linked_as_something_a_call_received(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())
    started = {s.span_id: s.started_at for s in trace.spans}
    started.update({n.span_id: n.started_at for n in trace.workflow_nodes})
    ended = {s.span_id: s.ended_at for s in trace.spans}

    for link in trace.evidence:
        if link.kind in ("text_match", "shared_result"):
            assert started[link.target_span_id] >= ended[link.call_span_id]


@pytest.mark.parametrize(
    ("record", "fields", "expected"),
    [
        ({"input": {"query": QUESTION}}, (), (None, "unresolved", None)),
        ({"input": {"query": ""}}, ("input.query",), (None, "unresolved", None)),
        ({"input": {"query": None}}, ("input.query",), (None, "unresolved", None)),
        ({"input": {"query": {"text": QUESTION}}}, ("input.query",), (None, "unresolved", None)),
        (
            {"input": {"payload": {"query": QUESTION}}},
            ("input.query", "input.payload.query"),
            (QUESTION, "declared", "input.payload.query"),
        ),
        ({"input": {"query": "a", "q": "b"}}, ("input.query", "input.q"), (None, "conflict", None)),
        (
            {"input": {"query": "a", "q": "a"}},
            ("input.query", "input.q"),
            ("a", "declared", "input.query"),
        ),
    ],
)
def test_task_selectors(record, fields, expected) -> None:
    task, status, path, reason = resolve_task(record, fields)
    assert (task, status, path) == expected
    assert (reason is None) == (status == "declared")


def test_type_error_is_named_in_the_reason() -> None:
    _, status, _, reason = resolve_task({"input": {"query": {"text": "x"}}}, ("input.query",))
    assert status == "unresolved" and "dict, not a string" in reason


def test_conflicting_task_is_reported_and_left_unset(tmp_path) -> None:
    declared = WorkflowDeclaration(task_fields=("input.query", "input.q"))
    corpus, trace = _load(tmp_path, _workflow(invocation={"query": "a", "q": "b"}), declared)

    assert trace.task is None
    assert trace.request.task_status == "conflict"
    assert any(issue.kind == "task_conflict" for issue in corpus.issues)


def test_no_declared_field_keeps_the_raw_request(tmp_path) -> None:
    corpus, trace = _load(tmp_path, _workflow(), WorkflowDeclaration())

    assert trace.task is None
    assert trace.request.task_status == "unresolved"
    assert trace.request.raw_input == {"query": QUESTION}
    assert any(issue.kind == "task_unresolved" for issue in corpus.issues)


def test_partial_export_without_container_still_finds_the_invocation(tmp_path) -> None:
    spans = _workflow(container=False)
    spans[0]["parentSpanId"] = "never-exported"
    _, trace = _load(tmp_path, spans)

    assert trace.request.source_span_id == "inv"
    assert trace.task == QUESTION


def test_several_candidates_resolved_by_the_task_field(tmp_path) -> None:
    spans = _workflow(container=False) + [
        _span(
            "other",
            "Generate Answer",
            {
                "langfuse.observation.type": "CHAIN",
                "input.value": json.dumps({"question": QUESTION}),
            },
        )
    ]
    _, trace = _load(tmp_path, spans)

    assert trace.request.source_span_id == "inv"
    assert trace.request.invocation_basis == "only candidate where a declared task field resolves"
    assert set(trace.request.candidate_span_ids) == {"inv", "other"}


def test_ambiguous_invocation_is_recorded_not_guessed(tmp_path) -> None:
    spans = _workflow(container=False) + [
        _span(
            "twin",
            "generate_answer",
            {"langfuse.observation.type": "SPAN", "input.value": json.dumps({"query": "other"})},
        )
    ]
    corpus, trace = _load(tmp_path, spans)

    assert trace.request.source_span_id is None
    assert trace.task is None
    assert any(issue.kind == "ambiguous_invocation" for issue in corpus.issues)


def test_a_lone_model_call_is_not_an_invocation(tmp_path) -> None:
    corpus, trace = _load(
        tmp_path, [_gen("m", "ANSWER", "Question: x", ANSWER, parent="gone", at=0)]
    )

    assert trace.request.source_span_id is None
    assert trace.request.candidate_span_ids == ()
    assert any(issue.kind == "ambiguous_invocation" for issue in corpus.issues)


@pytest.mark.parametrize(("origin", "turns"), [("human", 1), ("machine", 0), ("unknown", 0)])
def test_only_a_declared_human_request_becomes_a_turn(tmp_path, origin, turns) -> None:
    declared = DECLARED.replace(request_origin=origin)
    _, trace = _load(tmp_path, _workflow(), declared)

    assert len(trace.user_turns) == turns
    if turns:
        assert trace.user_turns[0].text == QUESTION
        assert trace.user_turns[0].origin == "declared"


def test_identical_outputs_are_ambiguous(tmp_path) -> None:
    spans = _workflow()
    spans.append(_chain("n_twin", "twin", parent="graph", at=1, step=2))
    spans.append(_gen("m_twin", "TWIN", "again", HYDE, parent="n_twin", at=1))
    _, trace = _load(tmp_path, spans)

    to_merge = {
        link.call_span_id: link for link in trace.evidence if link.target_span_id == "merge"
    }
    assert to_merge["m_hyde"].ambiguous and to_merge["m_twin"].ambiguous


def test_short_outputs_do_not_make_a_result_shared(tmp_path) -> None:
    spans = _workflow()
    merge_at = next(i for i, span in enumerate(spans) if span["spanId"] == "merge")
    spans.append(_gen("m_label", "CLASSIFY", "Query: x", "benefits", parent="graph", at=0))
    spans[merge_at] = _chain(  # merge also quotes the label
        "merge",
        "merge_expansions",
        parent="graph",
        at=3,
        step=3,
        input={"hyde_document": HYDE, "category": "benefits"},
    )
    _, trace = _load(tmp_path, spans)

    to_merge = {
        link.call_span_id: link for link in trace.evidence if link.target_span_id == "merge"
    }
    assert to_merge["m_label"].ambiguous
    assert "m_label" not in to_merge["m_hyde"].shared_with


def test_a_node_with_several_calls_encloses_each(tmp_path) -> None:
    spans = _workflow()
    spans.append(_gen("m_hyde2", "HYDE", "second attempt", HYDE + " Again.", parent="n_hyde", at=2))
    _, trace = _load(tmp_path, spans)

    enclosing = {
        (link.call_span_id, link.target_span_id) for link in _links(trace, "enclosing_result")
    }
    assert {("m_hyde", "n_hyde"), ("m_hyde2", "n_hyde")} <= enclosing


def test_without_framework_metadata_nothing_is_called_the_same_round(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow(framework=False))

    assert _links(trace, "same_round") == []
    assert trace.task == QUESTION


def test_upstream_truncated_output_is_kept_and_not_matched(tmp_path) -> None:
    spans = _workflow()
    spans[1]["attributes"] = [
        a
        if a["key"] != "output.value"
        else {
            "key": "output.value",
            "value": {"stringValue": '{"answer": "Coverage ends...[truncated]'},
        }
        for a in spans[1]["attributes"]
    ]
    _, trace = _load(tmp_path, spans)

    assert trace.request.raw_output == '{"answer": "Coverage ends...[truncated]'
    assert trace.request.delivered is None
    assert (
        delivery_status(trace.request, trace.evidence) == "no delivered value declared or recorded"
    )


def test_turn_extraction_refuses_a_workflow(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())
    with pytest.raises(WorkflowTurnsError):
        extract_turns(trace)


def test_request_view_reads_the_request_never_prompts(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())
    view = build_view(trace, TraceView.REQUEST)
    assert view.messages == (QUESTION,)
    assert not build_view(trace, TraceView.USER_MESSAGES).readable

    _, unresolved = _load(tmp_path, _workflow(), WorkflowDeclaration())
    (raw,) = build_view(unresolved, TraceView.REQUEST).messages
    assert raw.startswith("[raw request record, task unresolved]") and QUESTION in raw


def test_request_view_is_unreadable_on_a_conversation(tmp_path) -> None:
    trace = load_otlp_standard(_write(tmp_path / "c.jsonl", _request(_workflow()))).traces[0]
    assert not build_view(trace, TraceView.REQUEST).readable


def test_cli_ingests_a_workflow_and_refuses_turn_judging(tmp_path) -> None:
    path = _write(tmp_path / "wf.jsonl", _request(_workflow()))
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            str(path),
            "--source",
            "otlp-std",
            "--mode",
            "workflow",
            "--task-field",
            "input.query",
            "--delivered-field",
            "output.answer",
            "--project",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    corpus_id = re.search(r"artifact_id: (\S+)", result.output).group(1)

    judged = runner.invoke(
        app, ["judge-turns", corpus_id, "--archetype", "generic", "--project", str(tmp_path)]
    )
    assert judged.exit_code == 1
    assert "ingested as a workflow" in judged.output


def test_cli_rejects_workflow_options_without_workflow_mode(tmp_path) -> None:
    path = _write(tmp_path / "wf.jsonl", _request(_workflow()))
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(path),
            "--source",
            "otlp-std",
            "--task-field",
            "input.query",
            "--project",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1


def test_workflow_mode_is_otlp_std_only(tmp_path) -> None:
    with pytest.raises(ValueError, match="otlp-std"):
        load_corpus(tmp_path / "x.jsonl", "chat-json", workflow=DECLARED)


def test_a_failed_step_keeps_its_status_and_metadata(tmp_path) -> None:
    spans = _workflow()
    hyde = next(span for span in spans if span["spanId"] == "n_hyde")
    hyde["status"] = {"code": 2, "message": "retrieval timed out"}
    hyde["attributes"].append({"key": "metadata.retry_count", "value": {"intValue": "3"}})
    _, trace = _load(tmp_path, spans)

    node = next(n for n in trace.workflow_nodes if n.span_id == "n_hyde")
    assert node.status.value == "error"
    assert node.attributes["metadata.retry_count"] == 3
    assert "output.value" not in node.attributes  # already in node.output


def test_code_only_records_inside_a_kept_step_are_preserved(tmp_path) -> None:
    spans = _workflow()
    # A pipeline step with no model call beneath it is kept as a span; its child
    # has an output of its own that nothing else records.
    spans.append(_chain("rerank", "rerank", parent="graph", at=5))
    spans.append(
        _chain("scoring", "rerank_scoring", parent="rerank", at=5, output={"scores": [0.91, 0.12]})
    )
    corpus, trace = _load(tmp_path, spans)

    assert "rerank" in {s.span_id for s in trace.spans}
    scoring = next(n for n in trace.workflow_nodes if n.span_id == "scoring")
    assert scoring.output == {"scores": [0.91, 0.12]}
    assert scoring.parent_span_id == "rerank"
    assert not any(
        "CHAIN" in issue.detail for issue in corpus.issues if issue.kind == "unrepresented_span"
    )


def test_every_recorded_span_is_kept_somewhere(tmp_path) -> None:
    spans = _workflow()
    spans.append(_chain("rerank", "rerank", parent="graph", at=5))
    spans.append(_chain("scoring", "rerank_scoring", parent="rerank", at=5, output={"s": 1}))
    _, trace = _load(tmp_path, spans)

    kept = {s.span_id for s in trace.spans} | {n.span_id for n in trace.workflow_nodes}
    kept |= {trace.request.source_span_id, "box"}  # the container is reported, not kept
    assert {span["spanId"] for span in spans} <= kept


def test_request_view_never_cuts_the_request(tmp_path) -> None:
    padded = {"payload": {"config": "x" * 5000, "query": QUESTION}}
    _, trace = _load(tmp_path, _workflow(invocation=padded), WorkflowDeclaration())

    (raw,) = build_view(trace, TraceView.REQUEST).messages
    assert QUESTION in raw
    assert "[truncated]" not in raw


def test_a_code_only_invocation_is_the_request_not_an_action(tmp_path) -> None:
    # The answer call sits beside the invocation (its parent was never exported),
    # so nothing beneath the invocation is a model call.
    spans = [span for span in _workflow() if span["spanId"] in ("box", "inv", "graph", "merge")]
    spans.append(_gen("m_answer", "ANSWER", "Question: x", ANSWER, parent="gone", at=4))
    _, trace = _load(tmp_path, spans)

    assert trace.request.source_span_id == "inv"
    assert "inv" not in {s.span_id for s in trace.spans}
