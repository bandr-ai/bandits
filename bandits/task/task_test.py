"""Tests for pre-state reconstruction, filler generation and task mining."""

from __future__ import annotations

from pathlib import Path

import pytest

from bandits.contracts import EntityRows, TaskCase
from bandits.ingest import load_corpus, load_registry
from bandits.state import infer_schema
from bandits.surface import build_surface
from bandits.task._testdata import (
    POLICY,
    RETAIL_SCHEMA,
    WRITE_TOOLS,
    corpus,
    ep_cross_entity,
    ep_double_refund,
    ep_refund_ok,
    ep_status_update,
)
from bandits.task.filler import FillerError, fill_task, generate_filler
from bandits.task.mine import mine_task, mine_tasks
from bandits.task.prestate import (
    attribute_object,
    reconstruct_final_state,
    reconstruct_pre_state,
)


def _rows(pre_state: tuple[EntityRows, ...], entity: str) -> list[dict]:
    for er in pre_state:
        if er.entity == entity:
            return list(er.rows)
    return []


# --------------------------------------------------------------------------
# THE test: post-state must not leak backward
# --------------------------------------------------------------------------


def test_post_write_read_does_not_leak_into_pre_state():
    """ep-refund-ok reads order 7741 twice: 'delivered' at step 2 (pre-write) and
    'refunded' at step 5 (post-write). The pre-state must be the step-2 value.

    If this test ever fails, every rollout for this task starts in the solved
    state and the reward signal is silently dead."""
    pre = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    orders = _rows(pre.to_entity_rows(), "orders")
    row = next(r for r in orders if r["order_id"] == 7741)
    assert row["status"] == "delivered"
    assert row["status"] != "refunded"
    # no row anywhere in the pre-state carries the value the episode produced
    assert not any(r.get("status") == "refunded" for r in orders)
    # and the rejection is auditable, not silent
    assert any(b["step"] == 5 and b["entity"] == "orders" for b in pre.blocked)
    assert pre.first_write_step == 4


def test_write_response_row_is_not_pre_state_evidence():
    """The refund_order response itself carries status='refunded'. It is a write,
    so it never contributes rows to the pre-state."""
    pre = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    # 7742 is a partial row and carries no status at all; 7741 is observed and
    # must carry the pre-write one.
    observed = [r for r in _rows(pre.to_entity_rows(), "orders") if "status" in r]
    assert observed and all(r["status"] == "delivered" for r in observed)


def test_explicit_write_tool_override_matches_schema_derived_set():
    a = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    b = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA, write_tools=WRITE_TOOLS)
    assert a.to_entity_rows() == b.to_entity_rows()


# --------------------------------------------------------------------------
# per-entity, per-row tracking
# --------------------------------------------------------------------------


def test_read_of_other_entity_after_write_is_still_pre_state():
    """A write to orders says nothing about products, so a later get_product is
    still evidence of the starting world."""
    pre = reconstruct_pre_state(ep_cross_entity(), RETAIL_SCHEMA)
    products = _rows(pre.to_entity_rows(), "products")
    assert [p["sku"] for p in products] == ["SKU-BLU-1"]


def test_read_of_other_row_of_written_entity_after_write_is_still_pre_state():
    """Per-row, not per-entity: refunding 7750 does not invalidate a later read
    of order 7742."""
    pre = reconstruct_pre_state(ep_cross_entity(), RETAIL_SCHEMA)
    ids = {r["order_id"]: r for r in _rows(pre.to_entity_rows(), "orders")}
    assert set(ids) == {7750, 7742}
    assert ids[7750]["status"] == "delivered"  # pre-write value
    assert ids[7742]["status"] == "in_transit"
    assert pre.blocked == []


def test_unidentifiable_write_target_dirties_the_whole_entity():
    """When a write names no row, we cannot tell which rows it touched, so all
    later reads of that entity are refused."""
    trace = ep_cross_entity()
    stripped = trace.model_copy(
        update={
            "invocations": tuple(
                inv.model_copy(update={"arguments": {}, "response": {"ok": True}})
                if inv.tool == "refund_order"
                else inv
                for inv in trace.invocations
            )
        }
    )
    pre = reconstruct_pre_state(stripped, RETAIL_SCHEMA)
    ids = {r["order_id"] for r in _rows(pre.to_entity_rows(), "orders")}
    assert ids == {7750}  # 7742 was read after the un-attributable write
    assert any("whole entity" in n for n in pre.notes)
    # products, a different entity, is untouched by the blanket dirty flag
    assert _rows(pre.to_entity_rows(), "products")


def test_error_response_is_not_pre_state_evidence():
    """{'error': 'not_found', 'order_id': 9999} describes an ABSENT row. Seeding
    it would invert the task."""
    pre = reconstruct_pre_state(ep_double_refund(), RETAIL_SCHEMA)
    assert _rows(pre.to_entity_rows(), "orders") == []


def test_first_observation_wins_and_only_fills_gaps():
    trace = ep_status_update()
    partial = trace.invocations[0].model_copy(
        update={"step": 0, "response": {"order_id": 7742, "status": "in_transit"}}
    )
    later = trace.invocations[0].model_copy(update={"step": 1, "call_id": "c1"})
    t2 = trace.model_copy(update={"invocations": (partial, later)})
    pre = reconstruct_pre_state(t2, RETAIL_SCHEMA)
    row = _rows(pre.to_entity_rows(), "orders")[0]
    assert row["status"] == "in_transit"
    assert row["sku"] == "SKU-YEL-7"  # gap filled by the later pre-write read


# --------------------------------------------------------------------------
# solvability
# --------------------------------------------------------------------------


def test_unsolvable_task_is_flagged():
    """ep-double-refund writes before it reads, so no pre-state row backs the
    ids in its instruction. Unflagged, every rollout fails identically and the
    pass@k gate reads it as 'beyond the model's ability'."""
    task, reason, pre = mine_task(ep_double_refund(), RETAIL_SCHEMA)
    assert reason is None and task is not None and pre is not None
    warnings = task.provenance["solvability_warnings"]
    assert warnings
    assert any("7741" in w for w in warnings)
    assert any("88" in w for w in warnings)


def test_solvable_task_has_no_warning():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None
    assert task.provenance["solvability_warnings"] == ()


# --------------------------------------------------------------------------
# mining
# --------------------------------------------------------------------------


def test_mine_tasks_over_the_corpus():
    result = mine_tasks(corpus(), RETAIL_SCHEMA)
    ids = {t.task_id for t in result.tasks}
    assert ids == {
        "task-ep-refund-ok",
        "task-ep-cross-entity",
        "task-ep-double-refund",
        "task-ep-status-update",
    }
    reasons = {s["trace_id"]: s["reason"] for s in result.skipped}
    assert "no invocations" in reasons["ep-chatter"]
    assert "no user message" in reasons["ep-headless"]
    assert [t.task_id for t in result.warned] == ["task-ep-double-refund"]


def test_task_carries_tools_outcome_and_provenance():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None
    assert task.outcome is True
    assert task.tools == (
        "get_customer",
        "search_orders",
        "get_order",
        "get_store_policy",
        "refund_order",
        "send_email",
    )
    assert task.provenance["source_trace_id"] == "ep-refund-ok"
    assert task.provenance["invocation_count"] == 7


# --------------------------------------------------------------------------
# filler
# --------------------------------------------------------------------------


def _observed_strings() -> set[str]:
    out: set[str] = set()
    for e in RETAIL_SCHEMA.entities:
        for f in e.fields:
            out |= {str(v) for v in f.sample_values}
    return out


def test_filler_never_emits_an_observed_real_value():
    real = _observed_strings()
    for entity in RETAIL_SCHEMA.entities:
        rows = generate_filler(entity, count=6, seed=7)
        for row in rows:
            for name, value in row.items():
                if isinstance(value, bool):
                    continue  # booleans have two values; shape and value coincide
                assert str(value) not in real, f"{entity.name}.{name} leaked {value!r}"


def test_filler_is_deterministic_across_runs():
    e = RETAIL_SCHEMA.entity("orders")
    assert e is not None
    assert generate_filler(e, count=5, seed=42) == generate_filler(e, count=5, seed=42)
    assert generate_filler(e, count=5, seed=42) != generate_filler(e, count=5, seed=43)


def test_filler_copies_shapes():
    e = RETAIL_SCHEMA.entity("orders")
    assert e is not None
    for row in generate_filler(e, count=5, seed=3):
        assert isinstance(row["order_id"], int)
        assert len(str(row["order_id"])) == 4
        assert row["placed_at"][4] == "-" and len(row["placed_at"]) == 10
        assert row["sku"].startswith(("A", "B", "C")) or row["sku"][3] == "-"
        assert row["sku"].count("-") == 2


def test_filler_primary_keys_are_unique_and_avoid_task_rows():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None
    filled = fill_task(task, RETAIL_SCHEMA, seed=11, count_per_entity=5)
    orders = _rows(filled, "orders")
    assert orders[0]["order_id"] == 7741  # the real row is first and unmodified
    assert orders[0]["status"] == "delivered"
    assert len({r["order_id"] for r in orders}) == len(orders)


def test_filler_skips_static_snapshot_entities():
    e = RETAIL_SCHEMA.entity("store_policy")
    assert e is not None
    assert generate_filler(e, count=5, seed=1) == ()


def test_preserve_values_is_opt_in_for_enums():
    e = RETAIL_SCHEMA.entity("orders")
    assert e is not None
    rows = generate_filler(e, count=5, seed=5, preserve_values={"status"})
    assert all(r["status"] in {"delivered", "refunded", "in_transit"} for r in rows)


def test_filler_refuses_when_the_value_space_is_exhausted():
    e = RETAIL_SCHEMA.entity("products")
    assert e is not None
    tiny = e.model_copy(
        update={
            "fields": (
                e.fields[0].model_copy(update={"json_types": ("boolean",)}),
            ),
            "primary_key": "sku",
        }
    )
    with pytest.raises(FillerError):
        generate_filler(tiny, count=5, seed=1)


# --------------------------------------------------------------------------
# final-state reconstruction (consumed by verifier synthesis)
# --------------------------------------------------------------------------


def test_final_state_is_the_post_write_view():
    final = reconstruct_final_state(ep_refund_ok(), RETAIL_SCHEMA)
    row = next(iter(final["orders"].values()))
    assert row["status"] == "refunded"
    # fields the entity does not declare never enter the state
    assert "refund_amount_cents" not in row


def test_mine_task_returns_a_taskcase_contract():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert isinstance(task, TaskCase)


# --------------------------------------------------------------------------
# static-snapshot entities: attributed by content, not by key
# --------------------------------------------------------------------------


def test_static_snapshot_is_seeded_from_a_keyless_body():
    """store_policy has primary_key=None BY DEFINITION - that is what makes it a
    static snapshot. If attribution demands a key, the feature is unreachable at
    runtime and every environment starts with an empty table."""
    pre = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    assert _rows(pre.to_entity_rows(), "store_policy") == [{"policy": POLICY}]


def test_duplicate_static_bodies_are_deduplicated():
    """Two reads of the same snapshot are one row, not two: content is the key."""
    trace = ep_refund_ok()
    again = trace.invocations[3].model_copy(update={"step": 7, "call_id": "again"})
    t2 = trace.model_copy(update={"invocations": (*trace.invocations, again)})
    pre = reconstruct_pre_state(t2, RETAIL_SCHEMA)
    assert _rows(pre.to_entity_rows(), "store_policy") == [{"policy": POLICY}]

    # ...but a genuinely different body is a second row
    other = trace.invocations[3].model_copy(
        update={"step": 8, "call_id": "other", "response": {"policy": "No refunds."}}
    )
    t3 = trace.model_copy(update={"invocations": (*trace.invocations, other)})
    rows = _rows(reconstruct_pre_state(t3, RETAIL_SCHEMA).to_entity_rows(), "store_policy")
    assert rows == [{"policy": POLICY}, {"policy": "No refunds."}]


def test_static_attribution_never_steals_a_keyed_row():
    """Containment, not overlap: an orders body can never land in store_policy."""
    hit = attribute_object({"order_id": 7741, "status": "delivered"}, RETAIL_SCHEMA)
    assert hit == ("orders", 7741)
    assert attribute_object({"sent": True}, RETAIL_SCHEMA) is None


def test_static_snapshot_rows_are_never_partial():
    """A static snapshot is never written, so every observation of it is valid
    pre-state and its rows are fully observed."""
    pre = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    assert "store_policy" not in pre.partial_row_keys()


# --------------------------------------------------------------------------
# partial rows: entities production NAMED but never showed
# --------------------------------------------------------------------------


def test_id_list_only_row_is_seeded_partial_and_marked_in_provenance():
    """search_orders(customer_id=88) -> {"order_ids": [7741, 7742]}. 7742 is
    never read, but production proved it exists; dropping it makes the replayed
    search return one id where production returned two."""
    task, _, pre = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None and pre is not None
    rows = {r["order_id"]: r for r in _rows(task.pre_state, "orders")}
    assert set(rows) == {7741, 7742}

    # the partial row carries ONLY what production stated: the key, plus the
    # equality filter the search itself applied.
    assert rows[7742] == {"order_id": 7742, "customer_id": 88}

    # and it is distinguishable from an observed row, out of band
    assert task.provenance["partial_pre_state_rows"] == {"orders": (7742,)}
    assert pre.partial_row_keys() == {"orders": [7742]}
    assert any("PARTIAL" in n for n in pre.notes)


def test_a_row_that_was_actually_read_is_not_marked_partial():
    """7741 is named by the same id list AND read in full. Once the body arrives
    the row stops being partial - in either order."""
    pre = reconstruct_pre_state(ep_refund_ok(), RETAIL_SCHEMA)
    assert 7741 not in pre.partial_row_keys()["orders"]
    row = next(r for r in _rows(pre.to_entity_rows(), "orders") if r["order_id"] == 7741)
    assert row["status"] == "delivered"

    # reversed order: the id list arrives after the full read
    trace = ep_refund_ok()
    invs = list(trace.invocations)
    invs[1], invs[2] = (
        invs[2].model_copy(update={"step": 1}),
        invs[1].model_copy(update={"step": 2}),
    )
    reordered = reconstruct_pre_state(trace.model_copy(update={"invocations": tuple(invs)}),
                                      RETAIL_SCHEMA)
    assert reordered.partial_row_keys() == {"orders": [7742]}


def test_id_list_read_after_a_write_to_that_row_is_still_refused():
    """Partial rows obey the same pre/post-write discipline as observed ones."""
    trace = ep_status_update()  # get_order 7742, update_order_status 7742, get_order 7742
    late = trace.invocations[0].model_copy(
        update={
            "step": 3,
            "call_id": "late",
            "tool": "search_orders",
            "arguments": {"customer_id": 88},
            "response": {"order_ids": [7742, 7799]},
        }
    )
    pre = reconstruct_pre_state(trace.model_copy(
        update={"invocations": (*trace.invocations, late)}), RETAIL_SCHEMA)
    ids = {r["order_id"] for r in _rows(pre.to_entity_rows(), "orders")}
    assert 7799 in ids  # a clean row named after the write is still fine
    assert pre.partial_row_keys() == {"orders": [7799]}
    assert any(b["row_key"] == 7742 and b["partial"] for b in pre.blocked)
    # 7742's observed pre-write body survives; the post-write naming adds nothing
    row = next(r for r in _rows(pre.to_entity_rows(), "orders") if r["order_id"] == 7742)
    assert row["status"] == "in_transit"


def test_id_lists_from_unrelated_tools_are_ignored():
    """Only a tool the schema already lists in read_by is trusted to be naming
    that entity's ids. A bare integer array from anything else is left alone."""
    trace = ep_refund_ok()
    stray = trace.invocations[1].model_copy(
        update={"tool": "send_email", "step": 8, "call_id": "stray"}
    )
    pre = reconstruct_pre_state(
        trace.model_copy(update={"invocations": (trace.invocations[0], stray)}), RETAIL_SCHEMA
    )
    assert _rows(pre.to_entity_rows(), "orders") == []


def test_partial_row_does_not_defeat_the_filler_or_the_solvability_check():
    task, _, _ = mine_task(ep_refund_ok(), RETAIL_SCHEMA)
    assert task is not None
    assert task.provenance["solvability_warnings"] == ()
    filled = fill_task(task, RETAIL_SCHEMA, seed=11, count_per_entity=5)
    orders = _rows(filled, "orders")
    assert orders[0]["order_id"] == 7741 and orders[0]["status"] == "delivered"
    assert orders[1] == {"order_id": 7742, "customer_id": 88}
    assert len({r["order_id"] for r in orders}) == len(orders)


# --------------------------------------------------------------------------
# against the REAL pipeline, not only hand-built fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_pipeline():
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    corpus = load_corpus(fixtures / "traces.otlp.jsonl", "otlp")
    surface = build_surface(corpus, declared_tools=load_registry(fixtures / "tools.json"))
    schema = infer_schema(corpus, surface)
    return corpus, schema, mine_tasks(corpus, schema)


def test_real_pipeline_seeds_the_static_snapshot_and_the_partial_row(real_pipeline):
    """The hand-written RETAIL_SCHEMA is a model of the inferred one. This test
    uses the inferred one, so a drift in stages 1-3 cannot hide behind it."""
    _corpus, schema, mining = real_pipeline
    policy_entity = next(e for e in schema.entities if e.static_snapshot)
    assert policy_entity.primary_key is None

    task = next(t for t in mining.tasks if t.trace_id == "ep-refund-ok")
    assert _rows(task.pre_state, policy_entity.name) == [{"policy": POLICY}]

    orders = {r["order_id"]: r for r in _rows(task.pre_state, "orders")}
    assert set(orders) == {7741, 7742}
    assert orders[7741]["status"] == "delivered"  # STILL not 'refunded'
    assert orders[7742] == {"order_id": 7742, "customer_id": 88}
    assert task.provenance["partial_pre_state_rows"] == {"orders": (7742,)}


def test_real_pipeline_marks_no_partial_rows_where_there_are_none(real_pipeline):
    _corpus, _schema, mining = real_pipeline
    for task in mining.tasks:
        if task.trace_id == "ep-refund-ok":
            continue
        assert task.provenance["partial_pre_state_rows"] == {}, task.task_id
