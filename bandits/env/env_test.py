"""Tests for the environment runtime.

The schema and task here are hand-written contract objects modelling the same
retail world as ``tests/fixtures/make_corpus.py``. Hand-writing them keeps this
stage independent of stages 1-3: if schema inference regresses, these tests
still say whether the *runtime* is correct.
"""

from __future__ import annotations

import pytest

from bandits.contracts import (
    CallStatus,
    EntityRows,
    EntitySchema,
    FieldProfile,
    ForeignKey,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolProfile,
    ToolSurface,
    WriteEffect,
)
from bandits.env import (
    BanditsSession,
    EffectLedger,
    ReadOnlyEntityError,
    SessionClosedError,
    Store,
    UnsupportedToolError,
    WriteRule,
)
from bandits.env.tools import ReadRule, infer_rule

POLICY = "Refunds accepted within 30 days of delivery on unopened items."


# ---------------------------------------------------------------- fixtures


def f(name: str, *types: str, samples=(), ident: bool = False) -> FieldProfile:
    return FieldProfile(
        name=name,
        json_types=tuple(types),
        occurrences=len(samples) or 1,
        distinct_values=len(set(map(repr, samples))),
        sample_values=tuple(samples),
        looks_like_identifier=ident,
    )


def retail_schema() -> StateSchema:
    customers = EntitySchema(
        name="customers",
        primary_key="customer_id",
        fields=(
            f("customer_id", "integer", samples=[88, 91], ident=True),
            f("name", "string", samples=["Ada Lovelace", "Grace Hopper"]),
            f("email", "string", samples=["ada@example.com"]),
            f("tier", "string", samples=["gold", "silver"]),
        ),
        read_by=("get_customer",),
        evidence_count=2,
    )
    orders = EntitySchema(
        name="orders",
        primary_key="order_id",
        fields=(
            f("order_id", "integer", samples=[7741, 7742, 7750], ident=True),
            f("customer_id", "integer", samples=[88, 91], ident=True),
            f("sku", "string", samples=["SKU-RED-9", "SKU-BLU-1"], ident=True),
            f("status", "string", samples=["delivered", "refunded", "in_transit"]),
            f("total_cents", "integer", samples=[4200, 15900, 999]),
            f("placed_at", "string", samples=["2026-07-02"]),
        ),
        foreign_keys=(
            ForeignKey(field="customer_id", references_entity="customers",
                       references_field="customer_id", confidence=0.9),
            ForeignKey(field="sku", references_entity="products",
                       references_field="sku", confidence=0.9),
        ),
        read_by=("get_order", "search_orders"),
        written_by=("refund_order", "update_order_status"),
        evidence_count=6,
    )
    products = EntitySchema(
        name="products",
        primary_key="sku",
        fields=(
            f("sku", "string", samples=["SKU-RED-9"], ident=True),
            f("title", "string", samples=["Red Widget"]),
            f("price_cents", "integer", samples=[4200]),
            f("in_stock", "boolean", samples=[True]),
        ),
        read_by=("get_product",),
        evidence_count=2,
    )
    # Read-only, never cross-referenced: materialized verbatim, not invented.
    # ``set_store_policy`` is declared as a writer purely so the read-only
    # refusal has something to refuse.
    store_policy = EntitySchema(
        name="store_policy",
        read_by=("get_store_policy",),
        written_by=("set_store_policy",),
        static_snapshot=True,
        evidence_count=2,
    )
    return StateSchema(entities=(customers, orders, products, store_policy))


TOOL_CLASSES = {
    "get_customer": ToolClass.READ,
    "search_orders": ToolClass.READ,
    "get_order": ToolClass.READ,
    "get_product": ToolClass.READ,
    "get_store_policy": ToolClass.READ,
    "refund_order": ToolClass.WRITE,
    "update_order_status": ToolClass.WRITE,
    "set_store_policy": ToolClass.WRITE,
    "send_email": ToolClass.EXTERNAL,
    "escalate_to_human": ToolClass.UNKNOWN,
}


def order_row(oid, cid, sku, status, cents, placed="2026-07-02"):
    return {
        "order_id": oid,
        "customer_id": cid,
        "sku": sku,
        "status": status,
        "total_cents": cents,
        "placed_at": placed,
    }


def retail_task(order_rows=None) -> TaskCase:
    rows = order_rows or (
        order_row(7741, 88, "SKU-RED-9", "delivered", 4200),
        order_row(7742, 88, "SKU-YEL-7", "in_transit", 999, "2026-07-05"),
    )
    return TaskCase(
        task_id="ep-refund-ok",
        trace_id="trace-0",
        instruction="I want a refund for my order, my customer id is 88.",
        pre_state=(
            EntityRows(entity="customers", rows=(
                {"customer_id": 88, "name": "Ada Lovelace",
                 "email": "ada@example.com", "tier": "gold"},
            )),
            EntityRows(entity="orders", rows=tuple(rows)),
            EntityRows(entity="products", rows=(
                {"sku": "SKU-RED-9", "title": "Red Widget",
                 "price_cents": 4200, "in_stock": True},
            )),
            EntityRows(entity="store_policy", rows=({"policy": POLICY},)),
        ),
        tools=tuple(TOOL_CLASSES),
        outcome=True,
    )


@pytest.fixture()
def session():
    with BanditsSession(retail_schema(), retail_task(), TOOL_CLASSES) as s:
        yield s


# ---------------------------------------------------------------- round trip


def test_write_then_read_round_trip(session):
    """The write->read relation, actually working: delivered -> refund -> refunded."""
    before = session.execute("get_order", {"order_id": 7741})
    assert before.status is CallStatus.OK
    assert before.response["status"] == "delivered"

    refund = session.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
    assert refund.status is CallStatus.OK
    assert refund.response["order_id"] == 7741
    assert refund.response["status"] == "refunded"

    after = session.execute("get_order", {"order_id": 7741})
    assert after.status is CallStatus.OK
    assert after.response["status"] == "refunded"
    # The rest of the row is untouched: no collateral damage.
    assert after.response["total_cents"] == 4200
    assert after.response["customer_id"] == 88


def test_update_order_status_maps_argument_onto_column(session):
    obs = session.execute("update_order_status", {"order_id": 7742, "status": "delivered"})
    assert obs.status is CallStatus.OK
    assert session.execute("get_order", {"order_id": 7742}).response["status"] == "delivered"


def test_second_refund_reproduces_the_already_refunded_error(session):
    session.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
    again = session.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
    assert again.status is CallStatus.ERROR
    assert again.error_kind == "already_refunded"
    assert again.response == {"error": "already_refunded", "order_id": 7741}


def test_search_orders_lists_ids_for_a_customer(session):
    obs = session.execute("search_orders", {"customer_id": 88})
    assert obs.status is CallStatus.OK
    assert sorted(obs.response["order_ids"]) == [7741, 7742]


def test_get_customer_and_get_product_round_trip(session):
    cust = session.execute("get_customer", {"customer_id": 88})
    assert cust.response["name"] == "Ada Lovelace"
    prod = session.execute("get_product", {"sku": "SKU-RED-9"})
    assert prod.response["in_stock"] is True  # boolean codec survives SQLite


# ---------------------------------------------------------------- external


def test_send_email_records_an_effect_and_changes_no_state(session):
    before_snap = session.snapshot()
    before_digest = session.digest()

    obs = session.execute("send_email", {
        "to_customer_id": 88, "subject": "Your refund", "body": "Refunded 42.00 for order 7741."
    })
    assert obs.status is CallStatus.OK

    effects = session.effects()
    assert len(effects) == 1
    assert effects[0].tool == "send_email"
    assert effects[0].arguments["to_customer_id"] == 88

    assert session.snapshot() == before_snap
    assert session.digest() == before_digest


def test_ledger_is_ordered_queryable_and_append_only():
    ledger = EffectLedger()
    ledger.append("send_email", {"to_customer_id": 88, "subject": "a"}, step=0)
    ledger.append("send_email", {"to_customer_id": 91, "subject": "b"}, step=1)
    ledger.append("send_email", {"to_customer_id": 88, "subject": "c"}, step=2)
    ledger.append("charge_card", {"to_customer_id": 88}, step=3)

    assert [e.step for e in ledger.all()] == [0, 1, 2, 3]
    assert ledger.count("send_email") == 3
    assert ledger.count_by_argument("to_customer_id", 88, tool="send_email") == 2
    assert ledger.count_by_argument("to_customer_id", "88", tool="send_email") == 2
    assert ledger.count_by_argument("to_customer_id", 91) == 1

    from bandits.env import LedgerFrozenError

    ledger.freeze()
    with pytest.raises(LedgerFrozenError):
        ledger.append("send_email", {}, step=4)


def test_effects_survive_teardown_and_ledger_freezes():
    s = BanditsSession(retail_schema(), retail_task(), TOOL_CLASSES)
    with s:
        s.execute("send_email", {"to_customer_id": 88, "subject": "x", "body": "y"})
    assert len(s.effects()) == 1
    assert s.ledger.frozen
    with pytest.raises(SessionClosedError):
        s.execute("get_order", {"order_id": 7741})


# ---------------------------------------------------------------- errors


def test_missing_order_returns_a_not_found_observation(session):
    obs = session.execute("get_order", {"order_id": 9999})
    assert obs.status is CallStatus.ERROR
    assert obs.error_kind == "not_found"
    assert obs.response == {"error": "not_found", "order_id": 9999}
    assert obs.response != {}


def test_write_to_a_missing_row_is_not_found_not_an_insert(session):
    obs = session.execute("refund_order", {"order_id": 9999, "amount_cents": 1})
    assert obs.status is CallStatus.ERROR
    assert obs.error_kind == "not_found"
    assert len(session.snapshot()["orders"]) == 2


# ---------------------------------------------------------------- unsupported


def test_unknown_class_tool_raises_and_is_listed_unsupported(session):
    manifest = session.manifest()
    assert "escalate_to_human" in manifest.unsupported_tools

    with pytest.raises(UnsupportedToolError) as err:
        session.execute("escalate_to_human", {"reason": "angry customer"})
    assert "UNKNOWN" in str(err.value)

    # It must NOT come back as a success-shaped observation. Faking success here
    # would corrupt every reward computed downstream.
    obs = None
    try:
        obs = session.execute("escalate_to_human", {"reason": "angry customer"})
    except UnsupportedToolError:
        pass
    assert obs is None
    assert session.effects() == ()


def test_tool_outside_the_action_space_raises(session):
    with pytest.raises(UnsupportedToolError):
        session.execute("delete_everything", {})


# ---------------------------------------------------------------- static


def test_store_policy_is_readable_verbatim(session):
    obs = session.execute("get_store_policy", {})
    assert obs.status is CallStatus.OK
    assert obs.response == {"policy": POLICY}
    assert session.snapshot()["store_policy"] == [{"policy": POLICY}]
    assert "store_policy" in session.manifest().static_entities


def test_store_policy_is_not_writable(session):
    assert "set_store_policy" in session.manifest().unsupported_tools
    with pytest.raises(UnsupportedToolError):
        session.execute("set_store_policy", {"policy": "anything goes"})
    # unchanged, verbatim
    assert session.snapshot()["store_policy"] == [{"policy": POLICY}]


def test_store_refuses_direct_writes_to_a_static_entity():
    store = Store(retail_schema()).open()
    store.seed(retail_task())
    with pytest.raises(ReadOnlyEntityError):
        store.update("store_policy", "policy", POLICY, {"policy": "nope"})
    store.close()


# ---------------------------------------------------------------- digest


def _product_task(products) -> TaskCase:
    return TaskCase(
        task_id="ep-refund-ok", trace_id="trace-0", instruction="x",
        pre_state=(
            EntityRows(entity="orders", rows=(
                order_row(7741, 88, "SKU-RED-9", "delivered", 4200),)),
            EntityRows(entity="products", rows=tuple(products)),
        ),
    )


def test_digest_is_stable_across_row_insertion_order():
    """Two stores seeded with the same rows in different order are the same state."""
    products = (
        {"sku": "SKU-RED-9", "title": "Red Widget", "price_cents": 4200, "in_stock": True},
        {"sku": "SKU-BLU-1", "title": "Blue Widget", "price_cents": 15900, "in_stock": False},
    )
    with BanditsSession(retail_schema(), _product_task(products), TOOL_CLASSES) as a, \
            BanditsSession(
                retail_schema(), _product_task(tuple(reversed(products)), ), TOOL_CLASSES) as b:
        # The raw row order genuinely differs...
        assert [r["sku"] for r in a.snapshot()["products"]] != [
            r["sku"] for r in b.snapshot()["products"]
        ]
        # ...and the digest does not care.
        assert a.digest() == b.digest()


def test_digest_changes_on_a_real_write_and_not_on_a_read(session):
    d0 = session.digest()
    session.execute("get_order", {"order_id": 7741})
    assert session.digest() == d0
    session.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
    assert session.digest() != d0


def test_schema_digest_is_deterministic():
    assert Store(retail_schema()).schema_digest() == Store(retail_schema()).schema_digest()


# ---------------------------------------------------------------- rules


def test_write_rule_override_changes_the_argument_to_column_mapping():
    """A human override remaps amount_cents onto total_cents and echoes it back.

    The generic path ignores amount_cents (no such column). Real customers will
    need exactly this kind of override, so it is a first-class input.
    """
    override = WriteRule(
        tool="refund_order",
        entity="orders",
        key_arg="order_id",
        key_column="order_id",
        column_map={"amount_cents": "total_cents"},
        set_values={"status": "refunded"},
        response_echo={"refund_amount_cents": "amount_cents"},
        conflict_error_kind="already_refunded",
        inferred=False,
    )
    with BanditsSession(
        retail_schema(), retail_task(), TOOL_CLASSES, rules={"refund_order": override}
    ) as s:
        obs = s.execute("refund_order", {"order_id": 7741, "amount_cents": 100})
        assert obs.status is CallStatus.OK
        assert obs.response["refund_amount_cents"] == 100
        row = s.execute("get_order", {"order_id": 7741}).response
        assert row["total_cents"] == 100  # remapped, not the default 4200
        assert row["status"] == "refunded"


def test_read_rule_override_selects_a_different_key_argument():
    override = ReadRule(
        tool="get_order", entity="orders", key_arg="id", key_column="order_id", inferred=False
    )
    with BanditsSession(
        retail_schema(), retail_task(), TOOL_CLASSES, rules={"get_order": override}
    ) as s:
        assert s.execute("get_order", {"id": 7741}).response["status"] == "delivered"


def test_external_stub_response_is_overridable():
    with BanditsSession(
        retail_schema(), retail_task(), TOOL_CLASSES,
        external_stubs={"send_email": {"sent": True}},
    ) as s:
        obs = s.execute("send_email", {"to_customer_id": 88, "subject": "x", "body": "y"})
        assert obs.response == {"sent": True}
        assert len(s.effects()) == 1


def test_status_effect_is_only_inferred_from_observed_values():
    schema = retail_schema()
    orders = schema.entity("orders")
    rule = infer_rule(schema, "refund_order", ToolClass.WRITE)
    assert isinstance(rule, WriteRule)
    assert rule.set_values == {"status": "refunded"}

    # A verb whose past tense was never observed in the data must not be invented.
    stripped = EntitySchema(
        **{**orders.model_dump(), "fields": tuple(
            f("status", "string", samples=["delivered", "in_transit"]) if p.name == "status" else p
            for p in orders.fields
        )}
    )
    schema2 = StateSchema(entities=(stripped,))
    rule2 = infer_rule(schema2, "refund_order", ToolClass.WRITE)
    assert isinstance(rule2, WriteRule)
    assert rule2.set_values == {}
    # ... and calling it fails loudly rather than pretending to have done work.
    task = TaskCase(
        task_id="t", trace_id="tr", instruction="x",
        pre_state=(EntityRows(entity="orders", rows=(
            order_row(7741, 88, "SKU-RED-9", "delivered", 4200),)),),
    )
    with BanditsSession(schema2, task, {"refund_order": ToolClass.WRITE}) as s:
        with pytest.raises(UnsupportedToolError):
            s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})


def test_manifest_identity_is_deterministic_and_honest():
    with BanditsSession(retail_schema(), retail_task(), TOOL_CLASSES) as a, \
            BanditsSession(retail_schema(), retail_task(), TOOL_CLASSES) as b:
        ma, mb = a.manifest(), b.manifest()
        assert ma == mb
        assert ma.task_id == "ep-refund-ok"
        assert ma.tool_classes["send_email"] is ToolClass.EXTERNAL
        assert set(ma.unsupported_tools) == {"escalate_to_human", "set_store_policy"}
        assert a.unsupported_reason("escalate_to_human")


def test_pre_state_with_an_unknown_column_fails_loudly():
    from bandits.env import StoreError

    task = TaskCase(
        task_id="t", trace_id="tr", instruction="x",
        pre_state=(EntityRows(entity="orders", rows=({"order_id": 1, "nope": 2},)),),
    )
    with pytest.raises(StoreError):
        BanditsSession(retail_schema(), task, TOOL_CLASSES).open()


def test_snapshot_is_detached_so_mutating_it_cannot_hack_reward(session):
    snap = session.snapshot()
    snap["orders"][0]["status"] = "refunded"
    assert session.execute("get_order", {"order_id": 7741}).response["status"] == "delivered"


def test_file_backed_store_is_deleted_at_close():
    import os

    s = BanditsSession(retail_schema(), retail_task(), TOOL_CLASSES, db_path="")
    with s:
        path = s._db_path
        assert path and os.path.exists(path)
        s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
    assert not os.path.exists(path)
    s.close()  # idempotent


# ---------------------------------------------------------------- projection


def orders_with_refund_column(write_effects=()) -> StateSchema:
    """The retail schema, plus the column ``refund_order`` teaches ``orders``.

    This is the real shape of the defect: ``refund_amount_cents`` is a
    legitimate column of the table, inferred from a *write* tool's response,
    which ``get_order`` never returned.
    """
    schema = retail_schema()
    orders = schema.entity("orders")
    grown = EntitySchema(
        **{
            **orders.model_dump(),
            "fields": (*orders.fields, f("refund_amount_cents", "integer", samples=[4200])),
            "write_effects": tuple(write_effects),
        }
    )
    return StateSchema(
        entities=tuple(grown if e.name == "orders" else e for e in schema.entities)
    )


def retail_surface() -> ToolSurface:
    """A stage-2 surface recording what each read tool was seen to return."""

    def profile(name, klass, response):
        return ToolProfile(
            name=name,
            tool_class=klass,
            call_count=1,
            response_fields=tuple(f(r, "string") for r in response),
        )

    return ToolSurface(
        tools=(
            profile("get_order", ToolClass.READ,
                    ["order_id", "customer_id", "sku", "status", "total_cents", "placed_at"]),
            profile("get_customer", ToolClass.READ,
                    ["customer_id", "name", "email", "tier"]),
            profile("search_orders", ToolClass.READ, ["order_ids", "order_ids[]"]),
        )
    )


def test_read_returns_only_the_fields_the_tool_was_observed_to_return():
    """get_order must not leak refund_amount_cents just because the table has it."""
    schema = orders_with_refund_column([REFUND_EFFECT])
    with BanditsSession(
        schema, retail_task(), TOOL_CLASSES, surface=retail_surface()
    ) as s:
        row = s.execute("get_order", {"order_id": 7741}).response
        assert set(row) == {
            "order_id", "customer_id", "sku", "status", "total_cents", "placed_at"
        }
        # ...and it stays projected after a write populates the extra column.
        s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
        after = s.execute("get_order", {"order_id": 7741}).response
        assert "refund_amount_cents" not in after
        assert after["status"] == "refunded"
        # The column is really there; only the response is narrowed.
        assert s.snapshot()["orders"][0]["refund_amount_cents"] == 4200


def test_projection_narrows_rows_in_a_list_response_too():
    surface = retail_surface()
    rule = ReadRule(
        tool="search_orders", entity="orders", filter_args={"customer_id": "customer_id"},
        projection=("order_id", "status"), envelope="orders", inferred=False,
    )
    with BanditsSession(
        orders_with_refund_column(), retail_task(), TOOL_CLASSES,
        surface=surface, rules={"search_orders": rule},
    ) as s:
        rows = s.execute("search_orders", {"customer_id": 88}).response["orders"]
        assert all(set(r) == {"order_id", "status"} for r in rows)


def test_read_rule_projection_overrides_the_surface():
    """A human override beats the observed evidence, like the write rules do."""
    override = ReadRule(
        tool="get_order", entity="orders", key_column="order_id",
        projection=("order_id", "status"), inferred=False,
    )
    with BanditsSession(
        orders_with_refund_column(), retail_task(), TOOL_CLASSES,
        surface=retail_surface(), rules={"get_order": override},
    ) as s:
        assert s.execute("get_order", {"order_id": 7741}).response == {
            "order_id": 7741, "status": "delivered",
        }


def test_without_a_surface_null_columns_are_omitted_and_that_is_weaker():
    """The documented fallback, including the case where it is wrong.

    No surface means no evidence of the tool's field set, so NULL columns are
    dropped. That is right for a column this tool never returned and never
    populated -- and wrong the moment a write populates it, which is exactly
    what the second half of this test pins down. It is a heuristic, not a fix.
    """
    schema = orders_with_refund_column([REFUND_EFFECT])
    with BanditsSession(schema, retail_task(), TOOL_CLASSES) as s:
        row = s.execute("get_order", {"order_id": 7741}).response
        assert "refund_amount_cents" not in row  # NULL, so omitted
        assert row["status"] == "delivered"

        s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
        after = s.execute("get_order", {"order_id": 7741}).response
        # The heuristic's blind spot, asserted rather than hidden:
        assert after["refund_amount_cents"] == 4200


# ---------------------------------------------------------------- write effects


REFUND_EFFECT = WriteEffect(
    tool="refund_order",
    key_argument="order_id",
    argument_columns={"amount_cents": "refund_amount_cents"},
    sets_constants={"status": "refunded"},
    response_echoes=("order_id", "status", "refund_amount_cents"),
    evidence_count=3,
    confidence=0.8,
)


def test_write_effect_maps_the_argument_onto_a_differently_named_column():
    schema = orders_with_refund_column([REFUND_EFFECT])
    with BanditsSession(schema, retail_task(), TOOL_CLASSES) as s:
        obs = s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
        assert obs.status is CallStatus.OK
        assert s.snapshot()["orders"][0]["refund_amount_cents"] == 4200
        assert s.snapshot()["orders"][0]["status"] == "refunded"


def test_write_effect_response_echoes_shape_the_response():
    schema = orders_with_refund_column([REFUND_EFFECT])
    with BanditsSession(schema, retail_task(), TOOL_CLASSES) as s:
        obs = s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
        assert obs.response == {
            "order_id": 7741, "status": "refunded", "refund_amount_cents": 4200,
        }


def test_write_effect_rule_still_reproduces_the_conflict_error():
    schema = orders_with_refund_column([REFUND_EFFECT])
    with BanditsSession(schema, retail_task(), TOOL_CLASSES) as s:
        s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
        again = s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
        assert again.status is CallStatus.ERROR
        assert again.error_kind == "already_refunded"


def test_write_effect_is_authoritative_over_the_verb_guess():
    """The constant comes from evidence, not from the tool's English name."""
    effect = WriteEffect(
        tool="refund_order",
        key_argument="order_id",
        argument_columns={"amount_cents": "refund_amount_cents"},
        sets_constants={"status": "in_transit"},  # deliberately not "refunded"
        response_echoes=("order_id", "status"),
    )
    schema = orders_with_refund_column([effect])
    rule = infer_rule(schema, "refund_order", ToolClass.WRITE)
    assert isinstance(rule, WriteRule)
    assert rule.set_values == {"status": "in_transit"}
    assert rule.column_map["amount_cents"] == "refund_amount_cents"


def test_empty_write_effects_fall_back_to_the_previous_behaviour():
    """The parallel-stage case: nothing recorded, so the old path still runs."""
    schema = orders_with_refund_column()  # no write_effects at all
    rule = infer_rule(schema, "refund_order", ToolClass.WRITE)
    assert isinstance(rule, WriteRule)
    assert rule.set_values == {"status": "refunded"}  # last-resort verb guess
    assert rule.column_map is None
    assert rule.response_columns is None
    with BanditsSession(schema, retail_task(), TOOL_CLASSES) as s:
        obs = s.execute("refund_order", {"order_id": 7741, "amount_cents": 4200})
        assert obs.response == {"order_id": 7741, "status": "refunded"}


def test_write_effect_identity_arguments_still_map():
    """argument_columns lists only the renames; identical names must still work."""
    effect = WriteEffect(
        tool="update_order_status",
        key_argument="order_id",
        response_echoes=("order_id", "status"),
    )
    schema = orders_with_refund_column([effect])
    with BanditsSession(schema, retail_task(), TOOL_CLASSES) as s:
        obs = s.execute("update_order_status", {"order_id": 7742, "status": "delivered"})
        assert obs.response == {"order_id": 7742, "status": "delivered"}
        assert s.execute("get_order", {"order_id": 7742}).response["status"] == "delivered"
