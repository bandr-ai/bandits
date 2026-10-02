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
from bandits.traces import SpanStatus, WorkflowDeclaration
from bandits.verify.turns import WorkflowTurnsError, extract_turns
from tests.ingest.otlp_standard_test import _request, _span, _write

QUESTION = "If my order arrives after the promised date, is the delivery fee refunded?"
DRAFT = "Under the late delivery terms, the delivery fee is refunded once a parcel arrives late."
EXPAND = '{"expanded_queries": ["delivery fee refund rules for parcels that arrive late"]}'
ANSWER = "The delivery fee is refunded automatically when your parcel arrives late."

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
            "handle_request",
            {
                "langfuse.observation.type": "SPAN",
                "input.value": json.dumps(
                    invocation if invocation is not None else {"query": QUESTION}
                ),
                "output.value": json.dumps({"answer": ANSWER, "category": "billing"}),
            },
            parent="box" if container else None,
        ),
        _chain("graph", "LangGraph", parent="inv", at=0),
        _chain(
            "n_draft", "draft", parent="graph", at=1, step=step(2), output={"draft_document": DRAFT}
        ),
        _gen(
            "m_draft",
            "DRAFT",
            f"Customer question: {QUESTION}\n\nWrite the passage:",
            DRAFT,
            parent="n_draft",
            at=1,
        ),
        _chain("n_expand", "expand", parent="graph", at=1, step=step(2)),
        _gen(
            "m_expand",
            "EXPAND",
            f"Original question: {QUESTION}",
            EXPAND,
            parent="n_expand",
            at=1,
        ),
        _chain(
            "merge",
            "join_drafts",
            parent="graph",
            at=3,
            step=step(3),
            input={
                "draft_document": DRAFT,
                "expanded_queries": json.loads(EXPAND)["expanded_queries"],
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
        spans.insert(0, _span("box", "handle_request", {"langfuse.synthetic_span": "trace_record"}))
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
    assert request.raw_output == {"answer": ANSWER, "category": "billing"}


def test_failed_invocation_keeps_its_recorded_status(tmp_path) -> None:
    spans = _workflow()
    invocation = next(span for span in spans if span["spanId"] == "inv")
    invocation["status"] = {"code": 2, "message": "workflow failed"}
    trace = load_otlp_standard(
        _write(tmp_path / "failed.jsonl", _request(spans)), workflow=DECLARED
    ).traces[0]
    assert trace.request is not None
    assert trace.request.status == SpanStatus.ERROR


def test_conversation_mode_is_unchanged_on_the_same_export(tmp_path) -> None:
    path = _write(tmp_path / "wf.jsonl", _request(_workflow()))
    trace = load_otlp_standard(path).traces[0]

    # The existing reading, kept as it was: the first prompt is taken as the task.
    assert trace.interaction == "conversation"
    assert trace.request is None
    assert trace.task.startswith("Customer question:")
    assert trace.user_turns


def test_steps_containing_calls_are_kept_as_structure_not_actions(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())

    nodes = {node.name: node for node in trace.workflow_nodes}
    assert {"LangGraph", "draft", "expand"} <= set(nodes)
    assert nodes["draft"].output == {"draft_document": DRAFT}
    assert nodes["draft"].framework["langgraph_step"] == 2
    # The invocation is the request record, the container is nothing: neither is a node.
    assert "inv" not in {n.span_id for n in trace.workflow_nodes}
    assert "box" not in {n.span_id for n in trace.workflow_nodes}
    assert not {n.span_id for n in trace.workflow_nodes} & {s.span_id for s in trace.spans}


def test_evidence_links_record_their_basis(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())

    enclosing = {
        (link.call_span_id, link.target_span_id) for link in _links(trace, "enclosing_result")
    }
    assert ("m_draft", "n_draft") in enclosing and ("m_expand", "n_expand") in enclosing

    rounds = {(link.call_span_id, link.target_span_id) for link in _links(trace, "same_round")}
    assert rounds == {("m_draft", "m_expand"), ("m_expand", "m_draft")}
    assert all("not proof" in link.basis for link in _links(trace, "same_round"))

    merged = {
        link.call_span_id: link
        for link in _links(trace, "shared_result")
        if link.target_span_id == "merge"
    }
    assert set(merged) == {"m_draft", "m_expand"}
    assert merged["m_draft"].shared_with == ("m_expand",)
    assert not merged["m_draft"].ambiguous

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
            "Respond Flow",
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
            "handle_request",
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
    spans.append(_gen("m_twin", "TWIN", "again", DRAFT, parent="n_twin", at=1))
    _, trace = _load(tmp_path, spans)

    to_merge = {
        link.call_span_id: link for link in trace.evidence if link.target_span_id == "merge"
    }
    assert to_merge["m_draft"].ambiguous and to_merge["m_twin"].ambiguous


def test_short_outputs_do_not_make_a_result_shared(tmp_path) -> None:
    spans = _workflow()
    merge_at = next(i for i, span in enumerate(spans) if span["spanId"] == "merge")
    spans.append(_gen("m_label", "LABEL", "Query: x", "billing", parent="graph", at=0))
    spans[merge_at] = _chain(  # merge also quotes the label
        "merge",
        "join_drafts",
        parent="graph",
        at=3,
        step=3,
        input={"draft_document": DRAFT, "category": "billing"},
    )
    _, trace = _load(tmp_path, spans)

    to_merge = {
        link.call_span_id: link for link in trace.evidence if link.target_span_id == "merge"
    }
    assert to_merge["m_label"].ambiguous
    assert "m_label" not in to_merge["m_draft"].shared_with


def test_a_node_with_several_calls_encloses_each(tmp_path) -> None:
    spans = _workflow()
    spans.append(
        _gen("m_draft2", "DRAFT", "second attempt", DRAFT + " Again.", parent="n_draft", at=2)
    )
    _, trace = _load(tmp_path, spans)

    enclosing = {
        (link.call_span_id, link.target_span_id) for link in _links(trace, "enclosing_result")
    }
    assert {("m_draft", "n_draft"), ("m_draft2", "n_draft")} <= enclosing


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
            "value": {"stringValue": '{"answer": "The delivery fee...[truncated]'},
        }
        for a in spans[1]["attributes"]
    ]
    _, trace = _load(tmp_path, spans)

    assert trace.request.raw_output == '{"answer": "The delivery fee...[truncated]'
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
            "--mode",
            "conversation",
            "--task-field",
            "input.query",
            "--project",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "only apply to workflows" in result.output


def test_workflow_mode_is_otlp_std_only(tmp_path) -> None:
    with pytest.raises(ValueError, match="otlp-std"):
        load_corpus(tmp_path / "x.jsonl", "chat-json", workflow=DECLARED)


def test_a_failed_step_keeps_its_status_and_metadata(tmp_path) -> None:
    spans = _workflow()
    draft = next(span for span in spans if span["spanId"] == "n_draft")
    draft["status"] = {"code": 2, "message": "retrieval timed out"}
    draft["attributes"].append({"key": "metadata.retry_count", "value": {"intValue": "3"}})
    _, trace = _load(tmp_path, spans)

    node = next(n for n in trace.workflow_nodes if n.span_id == "n_draft")
    assert node.status.value == "error"
    assert node.attributes["metadata.retry_count"] == 3
    assert "output.value" not in node.attributes  # already in node.output


def test_code_only_records_inside_a_kept_step_are_preserved(tmp_path) -> None:
    spans = _workflow()
    # A pipeline step with no model call beneath it is kept as a span; its child
    # has an output of its own that nothing else records.
    spans.append(_chain("sort_results", "sort_results", parent="graph", at=5))
    spans.append(
        _chain(
            "scoring", "sort_scoring", parent="sort_results", at=5, output={"scores": [0.91, 0.12]}
        )
    )
    corpus, trace = _load(tmp_path, spans)

    assert "sort_results" in {s.span_id for s in trace.spans}
    scoring = next(n for n in trace.workflow_nodes if n.span_id == "scoring")
    assert scoring.output == {"scores": [0.91, 0.12]}
    assert scoring.parent_span_id == "sort_results"
    assert not any(
        "CHAIN" in issue.detail for issue in corpus.issues if issue.kind == "unrepresented_span"
    )


def test_every_recorded_span_is_kept_somewhere(tmp_path) -> None:
    spans = _workflow()
    spans.append(_chain("sort_results", "sort_results", parent="graph", at=5))
    spans.append(_chain("scoring", "sort_scoring", parent="sort_results", at=5, output={"s": 1}))
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


def _set(span: dict, key: str, value: str | None) -> None:
    attributes = [a for a in span["attributes"] if a["key"] != key]
    if value is not None:
        attributes.append({"key": key, "value": {"stringValue": value}})
    span["attributes"] = attributes


def test_no_stage_system_prompt_becomes_the_workflow_policy(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())
    assert trace.system_prompt is None
    assert any(s.attributes.get("gen_ai.input.messages") for s in trace.spans)


def test_a_cut_off_invocation_input_is_reported(tmp_path) -> None:
    spans = _workflow()
    _set(spans[1], "input.value", '{"query": "If an employee leaves mid')
    corpus, _ = _load(tmp_path, spans)
    assert any(
        issue.kind == "unparsed_value" and "input.value" in issue.detail for issue in corpus.issues
    )


def test_an_invocation_with_no_recorded_input_is_unreadable(tmp_path) -> None:
    spans = _workflow()
    _set(spans[1], "input.value", None)
    _, trace = _load(tmp_path, spans, WorkflowDeclaration())
    assert trace.request.source_span_id == "inv"
    view = build_view(trace, TraceView.REQUEST)
    assert not view.readable
    assert view.unreadable_reason == "the request record is empty"


def test_workflow_policy_and_structured_outputs_stay_on_calls(tmp_path) -> None:
    _, trace = _load(tmp_path, _workflow())
    assert trace.system_prompt is None
    expand = next(s for s in trace.spans if s.span_id == "m_expand")
    messages = expand.attributes["gen_ai.output.messages"]
    assert messages[0]["role"] == "assistant"
    assert json.loads(messages[0]["parts"][0]["content"]) == json.loads(EXPAND)
    assert expand.attributes["gen_ai.input.messages"][0]["role"] == "system"


def test_query_fields_match_independently_and_labels_do_not_match_substrings(tmp_path) -> None:
    queries = [
        "Find the refund terms for parcels delivered after the promised date",
        "Find the delivery fee rules for orders shipped to remote addresses",
    ]
    spans = _workflow()
    spans.extend(
        [
            _gen(
                "outline",
                "OUTLINE",
                QUESTION,
                json.dumps({"queries": queries}),
                parent="graph",
                at=0,
            ),
            _gen("label", "LABEL", QUESTION, "billing", parent="graph", at=0),
            _chain(
                "search_one",
                "lookup",
                parent="graph",
                at=3,
                input={"query": queries[0], "source": "fastbilling"},
            ),
            _chain(
                "search_two",
                "lookup",
                parent="graph",
                at=3,
                input={"query": queries[1], "source": "fastbilling"},
            ),
        ]
    )
    _, trace = _load(tmp_path, spans)
    links = [link for link in trace.evidence if link.target_span_id in {"search_one", "search_two"}]
    assert {(link.call_span_id, link.target_span_id) for link in links} == {
        ("outline", "search_one"),
        ("outline", "search_two"),
    }
    assert all(
        link.match_chars == len(queries[0 if link.target_span_id == "search_one" else 1])
        for link in links
    )


def test_retrieved_documents_link_to_model_input_only_after_result(tmp_path) -> None:
    spans = _workflow()
    document = "A late parcel qualifies for a full delivery fee refund within thirty days."
    spans.extend(
        [
            _chain(
                "gathered_docs", "gather_docs", parent="graph", at=3, output={"docs": [document]}
            ),
            _gen("grounded", "RESPOND", document, ANSWER, parent="inv", at=5),
            _chain("future_docs", "gather_docs", parent="graph", at=7, output={"docs": [document]}),
        ]
    )
    _, trace = _load(tmp_path, spans)
    links = [
        link
        for link in trace.evidence
        if link.kind == "input_context" and link.call_span_id == "grounded"
    ]
    assert [link.target_span_id for link in links] == ["gathered_docs"]
    assert links[0].match_chars == len(document)
