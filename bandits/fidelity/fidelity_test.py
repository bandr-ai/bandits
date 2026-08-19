"""Tests for the fidelity gate.

The centrepiece is :func:`test_golden_corpus_per_tool_rates`, which pins the
*actual* numbers the current pipeline achieves on ``tests/fixtures``. They are
low. They are written down anyway, because the whole point of this stage is that
an environment which cannot reproduce its own source trace gets said so out
loud. Weakening these assertions to make the suite look better would delete the
only signal the project produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bandits.contracts import (
    CallStatus,
    EntitySchema,
    FidelityReport,
    Observation,
    StateSchema,
    ToolClass,
)
from bandits.env import WriteRule
from bandits.fidelity import (
    CallReplay,
    GateCriteria,
    ReplayResult,
    build_report,
    compare_observations,
    compare_values,
    render_str,
    replay_corpus,
    replay_trace,
    to_json,
    tool_classes_from_surface,
)
from bandits.ingest import load_corpus_and_registry
from bandits.state import infer_schema
from bandits.surface import build_surface
from bandits.task import mine_tasks

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


# ---------------------------------------------------------------- the golden pipeline


@pytest.fixture(scope="module")
def golden():
    corpus, registry = load_corpus_and_registry(
        FIXTURES / "traces.otlp.jsonl", "otlp", FIXTURES / "tools.json"
    )
    surface = build_surface(corpus, declared_tools=registry)
    schema = infer_schema(corpus, surface)
    mining = mine_tasks(corpus, schema)
    return {
        "corpus": corpus,
        "surface": surface,
        "schema": schema,
        "tasks": tuple(mining.tasks),
        "tool_classes": tool_classes_from_surface(surface),
    }


@pytest.fixture(scope="module")
def golden_report(golden) -> FidelityReport:
    results = replay_corpus(
        golden["corpus"],
        golden["schema"],
        golden["tasks"],
        golden["tool_classes"],
        surface=golden["surface"],
    )
    return build_report(results)


#: The rates the pipeline actually achieves today, as ``tool -> (matched, replayed)``.
#: A regression moves one of these. Recorded, not aspirational.
GOLDEN_PER_TOOL: dict[str, tuple[int, int]] = {
    "get_customer": (2, 2),
    "get_order": (7, 8),
    "get_product": (2, 2),
    "get_store_policy": (2, 2),
    "refund_order": (3, 4),
    "search_orders": (1, 1),
    "send_email": (3, 3),
    "update_order_status": (1, 1),
}
GOLDEN_MATCHED = 21
GOLDEN_REPLAYED = 23
GOLDEN_OVERALL = 21 / 23  # 91.3%

#: Both remaining mismatches come from ``ep-double-refund``, whose very first
#: call is a write. With no pre-write read there is nothing to reconstruct the
#: row from, so the environment answers ``not_found`` where production answered
#: ``already_refunded``. That is an epistemic limit of the trace, not a defect:
#: an agent that writes before it reads leaves no record of the prior world.
#: The corpus stays REJECTED because ``refund_order`` sits at 75%, below the
#: per-tool floor, even though the 91% overall clears the threshold -- which is
#: the entire reason the floor exists.


def test_golden_pipeline_produces_a_report(golden_report):
    """The end-to-end path: ingest -> surface -> schema -> tasks -> replay -> report."""
    assert golden_report.env_id == "corpus"
    assert golden_report.trace_id == "*"
    assert golden_report.per_tool, "a report with no per-tool rows measures nothing"


def test_golden_corpus_per_tool_rates(golden_report):
    """Pin the real numbers. See the module docstring before touching these."""
    actual = {t.tool: (t.matched, t.replayed) for t in golden_report.per_tool}
    assert actual == GOLDEN_PER_TOOL


def test_golden_overall_rate(golden_report):
    assert sum(t.matched for t in golden_report.per_tool) == GOLDEN_MATCHED
    assert sum(t.replayed for t in golden_report.per_tool) == GOLDEN_REPLAYED
    assert golden_report.overall_rate == pytest.approx(GOLDEN_OVERALL)


def test_golden_covers_every_called_tool(golden):
    """Every tool with a recorded call gets a row; a never-called tool does not."""
    called = {inv.tool for tr in golden["corpus"].traces for inv in tr.invocations}
    results = replay_corpus(
        golden["corpus"],
        golden["schema"],
        golden["tasks"],
        golden["tool_classes"],
        surface=golden["surface"],
    )
    report = build_report(results)
    assert {t.tool for t in report.per_tool} == called
    # escalate_to_human is declared-only and UNKNOWN. It is never probed, so it
    # never appears in a fidelity row (PLAN.md Step 9).
    assert "escalate_to_human" not in {t.tool for t in report.per_tool}


def test_golden_is_rejected_and_says_why(golden_report):
    """91% overall clears the threshold; one tool below the floor still rejects.

    This is the whole argument for a per-tool floor. An aggregate-only gate
    would have accepted this environment and shipped a broken ``refund_order``.
    """
    assert golden_report.accepted is False
    text = "\n".join(golden_report.notes)
    assert "refund_order" in text
    assert "per-tool floor" in text
    # The overall rate is ABOVE the threshold, so that must not be the reason.
    assert "below the 90% threshold" not in text


def test_examples_are_carried_for_every_failing_tool(golden_report):
    for tool in golden_report.per_tool:
        if tool.matched == tool.replayed:
            continue
        assert tool.examples, f"{tool.tool} failed but carries no example divergence"
        assert len(tool.examples) <= 3
        for example in tool.examples:
            assert "trace_id" in example and "step" in example


def test_replay_does_not_stop_at_the_first_divergence(golden):
    """A trace whose second call diverges must still measure calls three onward."""
    trace = next(t for t in golden["corpus"].traces if t.trace_id == "ep-refund-ok")
    task = next(t for t in golden["tasks"] if t.trace_id == "ep-refund-ok")
    result = replay_trace(trace, golden["schema"], task, golden["tool_classes"])
    assert result.replayed == len(trace.invocations)
    assert [c.step for c in result.calls] == sorted(i.step for i in trace.invocations)
    assert result.mismatched > 0
    assert any(c.matched for c in result.calls)


def test_replay_refuses_a_task_from_another_trace(golden):
    trace = next(t for t in golden["corpus"].traces if t.trace_id == "ep-refund-ok")
    other = next(t for t in golden["tasks"] if t.trace_id == "ep-browse")
    with pytest.raises(ValueError, match="was mined from trace"):
        replay_trace(trace, golden["schema"], other, golden["tool_classes"])


# ---------------------------------------------------------------- external tools


def test_external_tool_is_judged_by_the_effect_ledger(golden, golden_report):
    """send_email's recorded body is {"sent": true}; the stub answers {"ok": true}.

    It matches anyway, because what the rebuilt world owes us is the logged
    attempt, not a guess at a vendor's acknowledgement format.
    """
    row = next(t for t in golden_report.per_tool if t.tool == "send_email")
    assert (row.matched, row.replayed) == (3, 3)

    trace = next(t for t in golden["corpus"].traces if t.trace_id == "ep-refund-ok")
    task = next(t for t in golden["tasks"] if t.trace_id == "ep-refund-ok")
    result = replay_trace(trace, golden["schema"], task, golden["tool_classes"])
    email = next(c for c in result.calls if c.tool == "send_email")
    assert email.external is True
    assert email.matched is True


def test_external_tool_that_logs_nothing_is_a_mismatch(golden):
    """If the ledger stays empty the external tool did not model anything."""
    trace = next(t for t in golden["corpus"].traces if t.trace_id == "ep-notfound")
    task = next(t for t in golden["tasks"] if t.trace_id == "ep-notfound")
    # Reclassify send_email as READ: it stops touching the ledger entirely.
    classes = dict(golden["tool_classes"])
    classes["send_email"] = ToolClass.READ
    result = replay_trace(trace, golden["schema"], task, classes)
    email = next(c for c in result.calls if c.tool == "send_email")
    assert email.matched is False


# ---------------------------------------------------------------- unsupported tools


def _unsupported_run(golden, tool: str = "get_order"):
    classes = dict(golden["tool_classes"])
    classes[tool] = ToolClass.UNKNOWN  # "not enough evidence to reimplement it"
    results = replay_corpus(
        golden["corpus"], golden["schema"], golden["tasks"], classes, surface=golden["surface"]
    )
    return build_report(results)


def test_unsupported_is_counted_separately_from_mismatched(golden):
    report = _unsupported_run(golden)
    row = next(t for t in report.per_tool if t.tool == "get_order")
    assert row.unsupported == 8
    assert row.mismatched == 0, "'could not model it' must never be reported as 'modeled it wrong'"
    assert row.matched == 0
    assert any("UNSUPPORTED: get_order" in n for n in report.notes)


def test_unsupported_stays_in_the_denominator(golden):
    """Refusing a tool must never improve the score by measuring less."""
    report = _unsupported_run(golden)
    row = next(t for t in report.per_tool if t.tool == "get_order")
    assert row.replayed == 8
    assert row.rate == 0.0
    assert sum(t.replayed for t in report.per_tool) == GOLDEN_REPLAYED
    # get_order went from 1 matched to 0, and nothing else moved.
    # get_order was forced UNKNOWN, so all 8 of its calls become unsupported
    # and its 7 previously-matched calls leave the matched total.
    assert sum(t.matched for t in report.per_tool) == GOLDEN_MATCHED - GOLDEN_PER_TOOL["get_order"][0]
    assert report.accepted is False


def test_unsupported_reason_is_carried_into_the_examples(golden):
    report = _unsupported_run(golden)
    row = next(t for t in report.per_tool if t.tool == "get_order")
    assert all(e["verdict"] == "unsupported" for e in row.examples)
    assert all("UNKNOWN" in e["reason"] for e in row.examples)


# ---------------------------------------------------------------- the gate


def _calls(tool: str, matched: int, mismatched: int = 0, unsupported: int = 0):
    out = []
    step = 0
    for _ in range(matched):
        out.append(CallReplay(step=step, tool=tool, verdict="matched"))
        step += 1
    for _ in range(mismatched):
        out.append(CallReplay(step=step, tool=tool, verdict="mismatched"))
        step += 1
    for _ in range(unsupported):
        out.append(
            CallReplay(step=step, tool=tool, verdict="unsupported", unsupported_reason="synthetic")
        )
        step += 1
    return out


def _synthetic(**tools) -> ReplayResult:
    result = ReplayResult(trace_id="t-1", task_id="task-1", env_id="env-1")
    for tool, spec in tools.items():
        result.calls.extend(_calls(tool, *spec))
    return result


def test_one_tool_below_the_floor_rejects_a_passing_aggregate():
    """The most important gate test. This is the README's example, exactly.

    overall 79/85 = 93% -- comfortably over the 90% threshold -- while
    update_order_status sits at 3/8 = 38%. An average would ship this.
    """
    result = _synthetic(
        get_order=(48, 0),
        search_orders=(12, 0),
        refund_order=(9, 1),
        send_email=(7, 0),
        update_order_status=(3, 5),
    )
    report = build_report([result])
    assert sum(t.replayed for t in report.per_tool) == 85
    assert sum(t.matched for t in report.per_tool) == 79
    assert report.overall_rate == pytest.approx(79 / 85)
    assert report.overall_rate > report.threshold, "the aggregate on its own would accept"
    assert report.accepted is False
    assert any("update_order_status at 38%" in n for n in report.notes)


def test_the_same_corpus_without_the_broken_tool_is_accepted():
    """Control for the test above: only the one bad tool is doing the rejecting."""
    result = _synthetic(
        get_order=(48, 0), search_orders=(12, 0), refund_order=(9, 1), send_email=(7, 0)
    )
    report = build_report([result])
    assert report.accepted is True
    assert any(n.startswith("ACCEPTED") for n in report.notes)


def test_low_aggregate_rejects_even_when_every_tool_clears_the_floor():
    """The converse: no single line to point at, but nine wrong answers in fifty."""
    result = _synthetic(a=(81, 19), b=(81, 19))
    report = build_report([result])
    assert all(t.rate >= 0.8 for t in report.per_tool)
    assert report.overall_rate < 0.9
    assert report.accepted is False
    assert any("below the 90% threshold" in n for n in report.notes)


def test_a_tool_that_is_entirely_unsupported_trips_the_floor_on_its_own():
    result = _synthetic(get_order=(99, 0), escalate=(0, 0, 1))
    report = build_report([result])
    assert report.overall_rate == pytest.approx(99 / 100)
    assert report.accepted is False
    assert any("escalate at 0%" in n for n in report.notes)


def test_min_calls_for_floor_exempts_but_reports():
    result = _synthetic(get_order=(99, 0), rare=(0, 1))
    lenient = build_report([result], criteria=GateCriteria(0.9, 0.8, min_calls_for_floor=2))
    assert lenient.accepted is True
    assert any("NOT JUDGED against the per-tool floor: rare" in n for n in lenient.notes)
    strict = build_report([result], criteria=GateCriteria(0.9, 0.8, min_calls_for_floor=1))
    assert strict.accepted is False


def test_an_empty_replay_is_never_accepted():
    report = build_report([ReplayResult(trace_id="t", task_id="k", env_id="e")])
    assert report.accepted is False
    assert report.overall_rate == 0.0
    assert any("nothing was replayed" in n for n in report.notes)


def test_gate_criteria_validates_its_thresholds():
    with pytest.raises(ValueError):
        GateCriteria(threshold=1.5)
    with pytest.raises(ValueError):
        GateCriteria(per_tool_floor=-0.1)
    with pytest.raises(ValueError):
        GateCriteria(min_calls_for_floor=0)


# ---------------------------------------------------------------- a broken environment


def _break_orders_primary_key(schema: StateSchema) -> StateSchema:
    """Corrupt the schema the way bad inference would: wrong key column on orders."""
    entities = []
    for entity in schema.entities:
        if entity.name == "orders":
            entity = entity.model_copy(update={"primary_key": "customer_id"})
        entities.append(entity)
    return schema.model_copy(update={"entities": tuple(entities)})


def test_a_broken_schema_is_rejected(golden, golden_report):
    """A wrong column mapping must score strictly worse than the real one."""
    broken = _break_orders_primary_key(golden["schema"])
    results = replay_corpus(
        golden["corpus"], broken, golden["tasks"], golden["tool_classes"], surface=golden["surface"]
    )
    report = build_report(results)
    assert report.accepted is False
    assert report.overall_rate < golden_report.overall_rate
    row = next(t for t in report.per_tool if t.tool == "get_order")
    assert row.rate == 0.0


def test_a_wrong_write_rule_is_rejected(golden):
    """refund_order pointed at the wrong column: amount_cents -> total_cents."""
    rules = {
        "refund_order": WriteRule(
            tool="refund_order",
            entity="orders",
            key_arg="order_id",
            key_column="order_id",
            column_map={"amount_cents": "total_cents"},
        )
    }
    trace = next(t for t in golden["corpus"].traces if t.trace_id == "ep-status-update")
    task = next(t for t in golden["tasks"] if t.trace_id == "ep-status-update")
    clean = build_report(
        [replay_trace(trace, golden["schema"], task, golden["tool_classes"])]
    )
    dirty_trace = next(t for t in golden["corpus"].traces if t.trace_id == "ep-refund-ok-2")
    dirty_task = next(t for t in golden["tasks"] if t.trace_id == "ep-refund-ok-2")
    dirty = build_report(
        [replay_trace(dirty_trace, golden["schema"], dirty_task, golden["tool_classes"], rules=rules)]
    )
    assert clean.per_tool  # the control replay produced something
    assert dirty.accepted is False
    row = next(t for t in dirty.per_tool if t.tool == "refund_order")
    assert row.rate == 0.0


# ---------------------------------------------------------------- tolerance policy


def _obs(response, status=CallStatus.OK, error_kind=None):
    return Observation(response=response, status=status, error_kind=error_kind)


def _blocking(divs):
    return [d for d in divs if not d.tolerated]


def test_timestamp_difference_is_tolerated():
    divs = compare_values(
        {"order_id": 1, "created_at": "2026-07-02T10:00:00Z"},
        {"order_id": 1, "created_at": "2026-08-19T04:31:07Z"},
    )
    assert len(divs) == 1
    assert divs[0].tolerated is True
    assert divs[0].field_class == "timestamp"
    assert _blocking(divs) == []


def test_status_difference_is_never_tolerated():
    divs = compare_observations(
        {"order_id": 1}, CallStatus.OK, None, _obs({"order_id": 1}, CallStatus.ERROR, "not_found")
    )
    blocking = _blocking(divs)
    assert {d.path for d in blocking} == {"$status", "$error_kind"}


def test_a_status_field_value_is_never_tolerated():
    divs = compare_values({"status": "refunded"}, {"status": "delivered"})
    assert _blocking(divs)
    assert divs[0].field_class == "exact"


def test_error_kind_difference_is_never_tolerated():
    divs = compare_observations(
        {"error": "already_refunded"},
        CallStatus.ERROR,
        "already_refunded",
        _obs({"error": "already_refunded"}, CallStatus.ERROR, "not_found"),
    )
    assert [d.path for d in _blocking(divs)] == ["$error_kind"]


def test_generated_id_is_tolerated_but_a_business_id_is_not():
    tolerated = compare_values({"request_id": "req-a"}, {"request_id": "req-b"})
    assert tolerated[0].tolerated is True
    business = compare_values({"order_id": 7741}, {"order_id": 7742})
    assert business[0].tolerated is False
    assert business[0].field_class == "exact"


def test_free_text_is_tolerated_only_when_both_sides_are_strings():
    prose = compare_values({"body": "Refunded 42.00."}, {"body": "We refunded you."})
    assert prose[0].tolerated is True
    typed = compare_values({"body": "42"}, {"body": 42})
    assert typed[0].tolerated is False
    assert "type changed" in typed[0].reason


def test_unordered_collections_compare_as_multisets():
    reordered = compare_values({"order_ids": [7741, 7742]}, {"order_ids": [7742, 7741]})
    assert len(reordered) == 1 and reordered[0].tolerated is True
    dropped = compare_values({"order_ids": [7741, 7742]}, {"order_ids": [7741]})
    assert dropped[0].tolerated is False
    assert "array length differs" in dropped[0].reason
    swapped = compare_values({"order_ids": [7741, 7742]}, {"order_ids": [7741, 9999]})
    assert _blocking(swapped)


def test_structural_shape_is_exact_even_for_null_values():
    """An extra key the real tool never returned changes what an agent observes."""
    extra = compare_values({"order_id": 1}, {"order_id": 1, "refund_amount_cents": None})
    assert extra[0].tolerated is False
    assert extra[0].path == "$.refund_amount_cents"
    missing = compare_values({"order_id": 1, "policy": "x"}, {"order_id": 1})
    assert missing[0].tolerated is False
    assert missing[0].path == "$.policy"


def test_numbers_have_no_epsilon():
    divs = compare_values({"total_cents": 4200}, {"total_cents": 4201})
    assert divs[0].tolerated is False
    assert "exact arithmetic" in divs[0].reason


def test_booleans_and_type_changes_are_exact():
    assert _blocking(compare_values({"paid": True}, {"paid": False}))
    assert _blocking(compare_values({"total_cents": 4200}, {"total_cents": "4200"}))


def test_nested_structures_report_a_usable_path():
    divs = compare_values(
        {"orders": [{"order_id": 1, "status": "delivered"}]},
        {"orders": [{"order_id": 1, "status": "refunded"}]},
    )
    assert [d.path for d in _blocking(divs)] == ["$.orders[0].status"]


def test_divergence_serializes_to_json():
    div = compare_values({"order_id": 1}, {"order_id": 2})[0]
    payload = div.to_json()
    assert json.dumps(payload)
    assert set(payload) == {"path", "expected", "actual", "reason", "tolerated", "field_class"}


# ---------------------------------------------------------------- rendering


def test_render_shows_per_tool_rows_and_the_verdict(golden_report):
    text = render_str(golden_report, width=200)
    assert "REJECTED" in text
    for tool in GOLDEN_PER_TOOL:
        assert tool in text
    assert "overall" in text
    assert "21/23" in text


def test_render_says_accepted_when_it_is():
    report = build_report([_synthetic(get_order=(10, 0))])
    text = render_str(report, width=200)
    assert "ACCEPTED" in text
    assert "REJECTED" not in text


def test_to_json_round_trips_through_the_contract(golden_report):
    payload = to_json(golden_report)
    assert payload["totals"] == {
        "replayed": GOLDEN_REPLAYED,
        "matched": GOLDEN_MATCHED,
        "mismatched": GOLDEN_REPLAYED - GOLDEN_MATCHED,
        "unsupported": 0,
    }
    assert payload["per_tool_rates"]["get_order"] == pytest.approx(7 / 8)
    revived = FidelityReport.model_validate_json(golden_report.model_dump_json())
    assert revived == golden_report


# ---------------------------------------------------------------- merging


def test_per_trace_and_merged_reports_agree(golden):
    results = replay_corpus(
        golden["corpus"],
        golden["schema"],
        golden["tasks"],
        golden["tool_classes"],
        surface=golden["surface"],
    )
    merged = build_report(results)
    per_trace = [build_report([r]) for r in results]
    assert sum(sum(t.replayed for t in r.per_tool) for r in per_trace) == GOLDEN_REPLAYED
    assert sum(sum(t.matched for t in r.per_tool) for r in per_trace) == GOLDEN_MATCHED
    assert merged.trace_id == "*"
    assert all(r.trace_id != "*" for r in per_trace)


def test_replay_corpus_refuses_a_task_whose_trace_is_absent(golden):
    empty = golden["corpus"].model_copy(update={"traces": ()})
    with pytest.raises(KeyError, match="not in this corpus"):
        replay_corpus(empty, golden["schema"], golden["tasks"], golden["tool_classes"])


def test_static_entity_is_still_measured(golden_report):
    """store_policy is a static snapshot. It is measured, not exempted.

    It now replays perfectly, but the point of this test is unchanged: a
    structure we admit we could not infer is still held to the same standard as
    everything else. Exempting it would let an environment score well by
    modelling less.
    """
    row = next(t for t in golden_report.per_tool if t.tool == "get_store_policy")
    assert row.replayed == 2
    assert row.rate == 1.0


def test_schema_entity_helper_is_untouched(golden):
    """Guard: the fidelity stage must not mutate anything it is handed."""
    orders = golden["schema"].entity("orders")
    assert isinstance(orders, EntitySchema)
    assert orders.primary_key == "order_id"
