"""Request and answer field discovery, and the hint shown when a task is missing."""

from bandits.ingest.discovery import (
    TASK_KEYS,
    Candidate,
    RequestSummary,
    TraceRequests,
    _hashed,
    answer_options,
    chosen_runs,
    discover,
    pick_path,
    task_options,
)


def _summary(*traces: list[tuple[str, object, object]]) -> RequestSummary:
    """Each trace is a list of ``(name, input, output)`` candidates of kind SPAN."""
    return RequestSummary(
        [
            TraceRequests(
                f"t{i}",
                [
                    Candidate(
                        f"s{i}{j}",
                        ("kind=SPAN", name),
                        {"input": _hashed(inp), "output": _hashed(out)},
                    )
                    for j, (name, inp, out) in enumerate(candidates)
                ],
            )
            for i, candidates in enumerate(traces)
        ]
    )


def test_request_field_must_be_text_in_every_run() -> None:
    runs = [
        [("run", {"query": "a", "email": "x"}, None)],
        [("run", {"query": "b", "email": "y"}, None)],
    ]
    assert discover(_summary(*runs)).task_fields == ("input.query",)
    assert discover(_summary(*runs, [("run", {"email": "z"}, None)])).task_fields == ()
    assert pick_path({"input": _hashed({"query": ""})}, "input", ("query",)) is None
    assert pick_path({"input": _hashed({"query": 3})}, "input", ("query",)) is None


def test_whole_text_input_is_the_request() -> None:
    assert discover(_summary([("run", "ask", None)], [("run", "tell", None)])).task_fields == (
        "input",
    )
    found = discover(_summary([("run", "ask", None)], [("run", {"query": "q"}, None)]))
    assert found.task_fields == ("input", "input.query")  # one identity, two shapes
    assert discover(RequestSummary()).task_fields == ()


def test_priority_order_decides_between_fields() -> None:
    assert (
        pick_path({"input": _hashed({"text": "t", "query": "q"})}, "input", TASK_KEYS)
        == "input.query"
    )


def test_nested_fields_qualify_by_their_last_key_two_levels_deep() -> None:
    record = {"input": _hashed({"payload": {"query": "q"}, "a": {"b": {"query": "deep"}}})}
    assert pick_path(record, "input", TASK_KEYS) == "input.payload.query"


def test_two_identities_that_each_settle_every_run_are_never_chosen_between() -> None:
    trace = [("app", {"payload": {"query": "q"}}, None), ("graph", {"question": "q"}, None)]
    found = discover(_summary(trace, trace))
    assert found.task_fields == ()
    assert [(o.paths, o.identities[0][1], o.covered, o.total) for o in found.task_options] == [
        (("input.payload.query",), "app", 2, 2),
        (("input.question",), "graph", 2, 2),
    ]
    assert found.hints() == [
        "--task-field input.payload.query (selects app(SPAN); 2/2)",
        "--task-field input.question (selects graph(SPAN); 2/2)",
    ]


def test_mixed_shapes_of_one_identity_give_one_set_with_both_paths() -> None:
    old = [("app", {"query": "q"}, {"answer": "a"})]
    new = [
        ("app", {"payload": {"query": "q"}}, {"answer": "a"}),
        ("graph", {"question": "q"}, None),
    ]
    found = discover(_summary(old, new, old, new))
    assert found.task_fields == ("input.query", "input.payload.query")
    assert found.delivered_field == "output.answer"


def test_a_conflict_counts_as_resolving_so_two_keys_can_fail_a_trace() -> None:
    # Stricter than priority on purpose: query and text disagree in one run.
    runs = [
        [("run", {"query": "q"}, None)],
        [("run", {"text": "t"}, None)],
        [("run", {"query": "a", "text": "b"}, None)],
    ]
    (option,) = task_options(_summary(*runs))
    assert option.paths == ("input.query", "input.text") and option.complete
    runs.append([("run", {"query": "a", "text": "b"}, None), ("other", {"query": "a"}, None)])
    (option, _) = task_options(_summary(*runs))
    assert not option.complete


def test_answer_is_read_on_the_chosen_run_only() -> None:
    trace = [("app", {"query": "q"}, {"answer": "a"}), ("graph", {"x": "q"}, {"response": "r"})]
    found = discover(_summary(trace, trace))
    assert found.task_fields == ("input.query",)
    assert found.delivered_field == "output.answer"


def test_different_answer_paths_by_shape_choose_none() -> None:
    summary = _summary(
        [("app", {"query": "q"}, {"answer": "a"})], [("app", {"query": "q"}, {"result": "r"})]
    )
    found = discover(summary)
    assert found.delivered_field is None
    assert [(o.paths, o.covered, o.total) for o in found.answer_options] == [
        (("output.answer",), 1, 2),
        (("output.result",), 1, 2),
    ]


def test_traces_without_candidates_do_not_block_a_choice() -> None:
    found = discover(_summary([("run", {"query": "q"}, None)], []))
    assert found.task_fields == ("input.query",)
    assert chosen_runs(_summary([]), ()) == [None]
    assert answer_options([None]) == []


def test_declared_fields_are_kept_and_select_the_run_for_the_answer() -> None:
    trace = [
        ("app", {"payload": {"query": "q"}}, {"answer": "a"}),
        ("graph", {"question": "q"}, {"state": "s"}),
    ]
    found = discover(_summary(trace), task_fields=("input.question",))
    assert found.task_fields == ("input.question",) and found.task_options == []
    assert found.delivered_field is None  # the graph run records no answer field


def test_one_run_recorded_under_two_names_is_one_option() -> None:
    found = discover(
        _summary(
            [("answer", {"query": "q"}, {"answer": "a"})],
            [("answer_stream", {"query": "q"}, {"answer": "a"})],
        )
    )
    (option,) = found.task_options
    assert [name for _, name in option.identities] == ["answer", "answer_stream"]
    assert found.task_fields == ("input.query",)
    assert found.delivered_field == "output.answer"


def test_a_merged_option_resolving_two_runs_in_one_trace_stays_ambiguous() -> None:
    clean = [("answer", {"query": "q"}, None)]
    both = [("answer", {"query": "q"}, None), ("answer_stream", {"query": "q2"}, None)]
    found = discover(_summary(clean, both))
    (option,) = found.task_options
    assert (option.covered, option.total) == (1, 2)
    assert found.task_fields == ()
