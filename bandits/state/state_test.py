"""Tests for stage 3 - state schema inference.

The golden corpus in ``tests/fixtures`` is the ground truth: entities, primary
keys and the ``store_policy`` snapshot must come out exactly as
``expected.json`` says. Beyond that, the tests here pin the *honesty*
properties - union-not-intersection fields, no foreign key from one
observation, degrade-to-snapshot rather than invent - because those are the
claims that make a rebuilt environment trustworthy.
"""

from __future__ import annotations

import pytest

from bandits.contracts import StateSchema, ToolClass, ToolProfile, ToolSurface
from bandits.ingest import load_corpus_and_registry
from bandits.state import infer_schema
from bandits.state._fixtures import (
    FIXTURE_DIR,
    corpus_from_calls,
    expected,
    golden_corpus,
    golden_surface,
)
from bandits.state.entities import (
    _anchor_consensus,
    choose_anchor,
    echo_candidates,
    entity_name,
    pluralize,
    singularize,
    snapshot_name,
    tool_noun,
)
from bandits.state.identifiers import find_identifiers, is_identifier_scalar
from bandits.state.relations import infer_foreign_keys
from bandits.state.writes import NO_DIFF_CONFIDENCE
from bandits.surface import build_surface

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return golden_corpus()


@pytest.fixture(scope="module")
def surface():
    return golden_surface()


@pytest.fixture(scope="module")
def schema(corpus, surface) -> StateSchema:
    return infer_schema(corpus, surface)


@pytest.fixture(scope="module")
def schema_no_surface(corpus) -> StateSchema:
    return infer_schema(corpus)


@pytest.fixture(scope="module")
def golden() -> dict:
    return expected()


def _fields(entity) -> set[str]:
    return {f.name for f in entity.fields}


# --------------------------------------------------------------------------
# entities and keys - the expected.json contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("with_surface", [True, False])
def test_entities_match_expected(corpus, surface, golden, with_surface):
    schema = infer_schema(corpus, surface if with_surface else None)
    modelled = [e.name for e in schema.entities if not e.static_snapshot]
    statics = [e.name for e in schema.entities if e.static_snapshot]
    assert modelled == golden["expected_entities"]
    assert statics == golden["expected_static_entities"]


def test_primary_keys_match_expected(schema, golden):
    for name, pk in golden["expected_primary_keys"].items():
        entity = schema.entity(name)
        assert entity is not None, f"missing entity {name}"
        assert entity.primary_key == pk


def test_entity_names_come_from_the_pluralization_rule():
    assert entity_name("order_id", {"get_order", "refund_order"}) == "orders"
    assert entity_name("customer_id", {"get_customer"}) == "customers"
    # sku has no `_id` stem, so the noun comes from the tool that anchors on it
    assert entity_name("sku", {"get_product"}) == "products"
    # no verb we recognize -> fall back to the field name, never a guess
    assert entity_name("sku", {"skuLookup"}) == "skus"


def test_pluralization_rules():
    assert pluralize("order") == "orders"
    assert pluralize("box") == "boxes"
    assert pluralize("policy") == "policies"
    assert pluralize("address") == "addresses"
    assert pluralize("day") == "days"
    assert singularize("orders") == "order"
    assert singularize("policies") == "policy"
    assert singularize("address") == "address"


def test_tool_nouns():
    assert tool_noun("get_product") == "product"
    assert tool_noun("search_orders") == "order"
    assert tool_noun("update_order_status") == "order_status"
    assert tool_noun("frobnicate") is None
    assert snapshot_name("get_store_policy") == "store_policy"


# --------------------------------------------------------------------------
# fields: union across projections, not intersection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("with_surface", [True, False])
def test_orders_fields_are_the_union_of_every_projection(corpus, surface, with_surface):
    schema = infer_schema(corpus, surface if with_surface else None)
    orders = schema.entity("orders")
    assert orders is not None
    required = {
        "order_id",
        "customer_id",
        "sku",
        "status",
        "total_cents",
        "placed_at",
        # only refund_order ever returns this one: proof of union, not intersection
        "refund_amount_cents",
    }
    assert required <= _fields(orders)


def test_refund_amount_is_evidenced_only_by_refund_order(schema):
    orders = schema.entity("orders")
    refund = next(f for f in orders.fields if f.name == "refund_amount_cents")
    order_id = next(f for f in orders.fields if f.name == "order_id")
    # thin evidence is recorded, not hidden: far fewer sightings than the key
    assert 0 < refund.occurrences < order_id.occurrences
    assert refund.json_types == ("integer",)


def test_primary_key_fields_are_flagged_as_identifiers(schema):
    orders = schema.entity("orders")
    flags = {f.name: f.looks_like_identifier for f in orders.fields}
    assert flags["order_id"] is True
    assert flags["customer_id"] is True
    assert flags["sku"] is True
    # `status` round-trips argument->response but never anchors a row
    assert flags["status"] is False
    assert flags["total_cents"] is False


def test_error_responses_never_contribute_fields_or_rows(schema):
    orders = schema.entity("orders")
    assert "error" not in _fields(orders)
    order_id = next(f for f in orders.fields if f.name == "order_id")
    # 9999 only ever appeared in a not_found body
    assert 9999 not in order_id.sample_values


# --------------------------------------------------------------------------
# static snapshot: degrade honestly
# --------------------------------------------------------------------------


@pytest.mark.parametrize("with_surface", [True, False])
def test_store_policy_is_a_static_snapshot(corpus, surface, with_surface):
    schema = infer_schema(corpus, surface if with_surface else None)
    policy = schema.entity("store_policy")
    assert policy is not None
    assert policy.static_snapshot is True
    assert policy.primary_key is None, "a snapshot must not get an invented key"
    assert policy.foreign_keys == ()
    assert _fields(policy) == {"policy"}
    assert policy.read_by == ("get_store_policy",)
    assert policy.written_by == ()
    # the observed row is kept verbatim
    row = next(f for f in policy.fields if f.name == "policy")
    assert row.sample_values == (
        "Refunds accepted within 30 days of delivery on unopened items.",
    )
    assert policy.evidence_count == 2


def test_read_once_never_cross_referenced_entity_degrades_to_snapshot():
    """An entity with one read and no relationships must not become a table."""
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_config", {"config_id": 1}, {"config_id": 1, "retries": 3}),
            ]
        }
    )
    schema = infer_schema(corpus)
    configs = schema.entity("configs")
    assert configs is not None
    assert configs.static_snapshot is True
    assert configs.foreign_keys == ()
    # only what was observed - no invented columns, no invented rows
    assert _fields(configs) == {"config_id", "retries"}
    assert configs.evidence_count == 1


def test_boolean_acknowledgements_are_unresolved_not_snapshots(schema_no_surface):
    """`send_email` returns {"sent": true}: an ack, not a row of state."""
    assert "send_email" in schema_no_surface.unresolved
    assert all(e.name != "send_email" for e in schema_no_surface.entities)


def test_external_tools_are_excluded_when_the_surface_says_so(schema):
    assert schema.unresolved == ()
    assert all("send_email" not in e.read_by + e.written_by for e in schema.entities)


def test_nondeterministic_identifier_less_tool_is_unresolved():
    """Same arguments, different bodies, no id: we cannot pin a snapshot."""
    corpus = corpus_from_calls(
        {
            "t1": [("get_feed", {}, {"headline": "a"})],
            "t2": [("get_feed", {}, {"headline": "b"})],
        }
    )
    schema = infer_schema(corpus)
    assert schema.unresolved == ("get_feed",)
    assert schema.entities == ()


# --------------------------------------------------------------------------
# foreign keys
# --------------------------------------------------------------------------


@pytest.mark.parametrize("with_surface", [True, False])
def test_orders_foreign_keys(corpus, surface, with_surface):
    schema = infer_schema(corpus, surface if with_surface else None)
    orders = schema.entity("orders")
    fks = {fk.field: fk for fk in orders.foreign_keys}
    assert fks["customer_id"].references_entity == "customers"
    assert fks["customer_id"].references_field == "customer_id"
    assert fks["sku"].references_entity == "products"
    assert fks["sku"].references_field == "sku"
    assert all(0.0 < fk.confidence <= 1.0 for fk in orders.foreign_keys)
    # customer_id was seen for every customer we know; sku was not
    assert fks["customer_id"].confidence > fks["sku"].confidence


def test_no_spurious_foreign_keys_on_the_golden_corpus(schema):
    emitted = {
        (e.name, fk.field, fk.references_entity)
        for e in schema.entities
        for fk in e.foreign_keys
    }
    assert emitted == {
        ("orders", "customer_id", "customers"),
        ("orders", "sku", "products"),
    }


def test_no_foreign_key_from_a_single_observation():
    """One co-occurrence is a coincidence, not a relationship."""
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_widget", {"widget_id": 1}, {"widget_id": 1, "gadget_id": 5}),
                ("get_gadget", {"gadget_id": 5}, {"gadget_id": 5, "label": "x"}),
            ]
        }
    )
    schema = infer_schema(corpus)
    assert all(e.foreign_keys == () for e in schema.entities)


def test_the_same_relationship_appears_once_there_are_two_observations():
    """Positive control for the negative test above: evidence is the only difference."""
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_widget", {"widget_id": 1}, {"widget_id": 1, "gadget_id": 5}),
                ("get_gadget", {"gadget_id": 5}, {"gadget_id": 5, "label": "x"}),
            ],
            "t2": [
                ("get_widget", {"widget_id": 2}, {"widget_id": 2, "gadget_id": 6}),
                ("get_gadget", {"gadget_id": 6}, {"gadget_id": 6, "label": "y"}),
            ],
        }
    )
    schema = infer_schema(corpus)
    widgets = schema.entity("widgets")
    assert [(fk.field, fk.references_entity) for fk in widgets.foreign_keys] == [
        ("gadget_id", "gadgets")
    ]
    # and a cross-referenced entity is no longer degraded to a snapshot
    assert widgets.static_snapshot is False
    assert schema.entity("gadgets").static_snapshot is False


def test_value_only_foreign_key_needs_full_containment_and_more_evidence():
    """A renamed key is inferable, but only from values that always land in range."""
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_invoice", {"invoice_id": 1}, {"invoice_id": 1, "buyer": 88}),
                ("get_customer", {"customer_id": 88}, {"customer_id": 88, "tier": "gold"}),
            ],
            "t2": [
                ("get_invoice", {"invoice_id": 2}, {"invoice_id": 2, "buyer": 91}),
                ("get_customer", {"customer_id": 91}, {"customer_id": 91, "tier": "silver"}),
            ],
            "t3": [
                ("get_invoice", {"invoice_id": 3}, {"invoice_id": 3, "buyer": 88}),
            ],
        }
    )
    schema = infer_schema(corpus)
    invoices = schema.entity("invoices")
    fk = next(fk for fk in invoices.foreign_keys if fk.field == "buyer")
    assert fk.references_entity == "customers"
    assert fk.references_field == "customer_id"
    # the weak rule may never claim as much confidence as the name-matched one
    assert fk.confidence <= 0.7


def test_value_only_rule_refuses_when_containment_is_partial():
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_invoice", {"invoice_id": 1}, {"invoice_id": 1, "buyer": 88}),
                ("get_customer", {"customer_id": 88}, {"customer_id": 88, "tier": "gold"}),
            ],
            "t2": [
                ("get_invoice", {"invoice_id": 2}, {"invoice_id": 2, "buyer": 404}),
            ],
            "t3": [
                ("get_invoice", {"invoice_id": 3}, {"invoice_id": 3, "buyer": 405}),
            ],
        }
    )
    schema = infer_schema(corpus)
    invoices = schema.entity("invoices")
    assert [fk.field for fk in invoices.foreign_keys] == []


# --------------------------------------------------------------------------
# reads and writes
# --------------------------------------------------------------------------


def test_orders_reads_and_writes_with_surface(schema):
    orders = schema.entity("orders")
    assert "refund_order" in orders.written_by
    assert "update_order_status" in orders.written_by
    assert "get_order" in orders.read_by
    assert "search_orders" in orders.read_by


def test_orders_reads_and_writes_without_surface(schema_no_surface):
    """Fallback path: writers are the tools whose response changed a known row."""
    orders = schema_no_surface.entity("orders")
    assert set(orders.written_by) == {"refund_order", "update_order_status"}
    assert "get_order" in orders.read_by
    assert "search_orders" in orders.read_by
    assert "refund_order" not in orders.read_by


def test_read_only_entities_have_no_writers(schema):
    for name in ("customers", "products"):
        assert schema.entity(name).written_by == ()


def test_search_orders_is_credited_through_its_id_list(schema):
    """A list-of-ids response is read evidence only - it adds no fields."""
    orders = schema.entity("orders")
    assert "search_orders" in orders.read_by
    assert "order_ids" not in _fields(orders)


def test_unknown_class_tools_are_in_neither_list():
    from bandits.contracts import ToolClass, ToolProfile, ToolSurface

    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_order", {"order_id": 1}, {"order_id": 1, "status": "new"}),
                ("poke_order", {"order_id": 1}, {"order_id": 1, "status": "poked"}),
            ]
        }
    )
    surface = ToolSurface(
        tools=(
            ToolProfile(name="get_order", tool_class=ToolClass.READ),
            ToolProfile(name="poke_order", tool_class=ToolClass.UNKNOWN),
        )
    )
    orders = infer_schema(corpus, surface).entity("orders")
    assert "poke_order" not in orders.written_by
    assert "poke_order" not in orders.read_by
    # without the surface, the observed change is enough to call it a write
    assert "poke_order" in infer_schema(corpus).entity("orders").written_by


# --------------------------------------------------------------------------
# identifiers and anchoring
# --------------------------------------------------------------------------


def test_identifier_index_finds_the_recurring_ids(corpus):
    index = find_identifiers(corpus)
    assert {"order_id", "customer_id", "sku"} <= set(index.names())
    assert index.values("customer_id") >= {88, 91}
    # arguments-only fields are never nominated: no recurrence, no evidence
    assert not index.has("to_customer_id")
    assert not index.has("amount_cents")
    # response-only fields likewise
    assert not index.has("total_cents")


def test_status_is_a_loose_identifier_candidate_but_never_an_anchor(corpus):
    """`update_order_status` echoes `status`; consensus keeps it an attribute."""
    index = find_identifiers(corpus)
    assert index.has("status")
    invocations = [i for t in corpus.traces for i in t.invocations]
    update = next(i for i in invocations if i.tool == "update_order_status")
    assert echo_candidates(update, index) == ["order_id", "status"]
    consensus = _anchor_consensus(invocations, index)
    assert choose_anchor(update, index, consensus) == ("order_id", 7742)


def test_reference_fields_do_not_anchor_the_body(corpus):
    """get_order's body carries customer_id and sku, but the order_id names the row."""
    index = find_identifiers(corpus)
    invocations = [i for t in corpus.traces for i in t.invocations]
    get_order = next(
        i for i in invocations if i.tool == "get_order" and i.arguments.get("order_id") == 7741
    )
    assert echo_candidates(get_order, index) == ["order_id"]


def test_identifier_scalar_shape():
    assert is_identifier_scalar(7741)
    assert is_identifier_scalar("SKU-RED-9")
    assert not is_identifier_scalar(True)
    assert not is_identifier_scalar(42.0)
    assert not is_identifier_scalar(None)
    assert not is_identifier_scalar("x" * 500)


# --------------------------------------------------------------------------
# whole-schema properties
# --------------------------------------------------------------------------


def test_inference_is_deterministic(corpus, surface):
    assert infer_schema(corpus, surface) == infer_schema(corpus, surface)


def test_evidence_counts_are_recorded(schema):
    counts = {e.name: e.evidence_count for e in schema.entities}
    assert counts["orders"] > counts["customers"] > 0
    assert all(v > 0 for v in counts.values())


def test_empty_corpus_yields_an_empty_schema():
    schema = infer_schema(corpus_from_calls({}))
    assert schema == StateSchema()


def test_foreign_key_evidence_is_returned_for_review(corpus):
    from bandits.state.entities import infer_drafts

    drafts, _ = infer_drafts(corpus)
    _fks, evidence = infer_foreign_keys(drafts)
    considered = {(e.source_entity, e.field, e.target_entity) for e in evidence}
    assert ("orders", "customer_id", "customers") in considered
    # every emitted key must have a matching evidence record
    assert any(e.rule == "FK-1" and e.matched_distinct > 0 for e in evidence)


# --------------------------------------------------------------------------
# write effects - what a write tool actually writes
# --------------------------------------------------------------------------


def _surface(**classes) -> ToolSurface:
    """A minimal ToolSurface for synthetic corpora: ``_surface(refund_order="write")``."""
    return ToolSurface(
        tools=tuple(
            ToolProfile(name=name, tool_class=ToolClass(value), class_confidence=1.0)
            for name, value in sorted(classes.items())
        )
    )


def _effect(entity, tool):
    for w in entity.write_effects:
        if w.tool == tool:
            return w
    raise AssertionError(f"no WriteEffect for {tool!r} on {entity.name!r}: {entity.write_effects}")


@pytest.mark.parametrize("with_surface", [True, False])
def test_every_writer_has_a_write_effect(corpus, surface, with_surface):
    """``write_effects`` covers exactly ``written_by``, so a missing one is a bug."""
    schema = infer_schema(corpus, surface if with_surface else None)
    for entity in schema.entities:
        assert {w.tool for w in entity.write_effects} == set(entity.written_by)


@pytest.mark.parametrize("with_surface", [True, False])
def test_refund_order_write_effect_is_fully_inferred(corpus, surface, with_surface):
    """The whole point: ``refunded`` comes from the observed diff, not from the verb."""
    schema = infer_schema(corpus, surface if with_surface else None)
    effect = _effect(schema.entity("orders"), "refund_order")

    assert effect.key_argument == "order_id"
    assert effect.sets_constants == {"status": "refunded"}
    assert effect.argument_columns == {"amount_cents": "refund_amount_cents"}
    assert set(effect.response_echoes) >= {"order_id", "status", "refund_amount_cents"}
    assert effect.evidence_count == 3
    assert 0.0 < effect.confidence < 1.0
    assert effect.evidence


def test_update_order_status_sets_no_constant_because_status_comes_from_an_argument(schema):
    """``update_order_status`` is a writer, but its status is *not* a constant.

    The corpus calls it exactly once, with ``status="delivered"``, and the
    column ends up ``"delivered"``. A count-based rule would happily report
    ``sets_constants == {"status": "delivered"}`` - and the rebuilt environment
    would then mark every order delivered whatever the agent asked for.

    The column's value equalled a *supplied argument* on every call, so it is
    argument-driven, and we publish nothing. ``argument_columns`` is also empty:
    the contract only records mappings where the argument and column names
    differ, and here both are ``status``, which needs no map. The identity
    mapping is recorded in ``evidence`` for the reviewer.
    """
    effect = _effect(schema.entity("orders"), "update_order_status")

    assert effect.key_argument == "order_id"
    assert effect.sets_constants == {}
    assert effect.argument_columns == {}
    assert set(effect.response_echoes) == {"order_id", "status"}
    assert effect.evidence_count == 1
    assert any("argument-driven" in line for line in effect.evidence)
    assert any("'status' -> column 'status'" in line for line in effect.evidence)


def test_a_column_with_a_different_value_per_call_is_not_a_constant():
    """Unanimity or nothing: a varying column is omitted, never reported by majority."""
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_ticket", {"ticket_id": 1}, {"ticket_id": 1, "state": "open", "owner": "ann"}),
                (
                    "assign_ticket",
                    {"ticket_id": 1, "owner": "bob"},
                    {"ticket_id": 1, "state": "assigned", "owner": "bob", "queue": "support"},
                ),
            ],
            "t2": [
                ("get_ticket", {"ticket_id": 2}, {"ticket_id": 2, "state": "open", "owner": "ann"}),
                (
                    "assign_ticket",
                    {"ticket_id": 2, "owner": "cat"},
                    {"ticket_id": 2, "state": "escalated", "owner": "cat", "queue": "support"},
                ),
            ],
        }
    )
    effect = _effect(infer_schema(corpus).entity("tickets"), "assign_ticket")

    # 'state' changed on every call but to a different value each time.
    assert "state" not in effect.sets_constants
    # 'owner' changed on every call, to the value the caller supplied.
    assert "owner" not in effect.sets_constants
    # 'queue' is the real constant, and the rule still finds it.
    assert effect.sets_constants == {"queue": "support"}
    assert any("not a constant" in line for line in effect.evidence)


def test_an_errored_write_contributes_no_evidence():
    """The failed call changed nothing, so it must not diff, count, or echo."""
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_ticket", {"ticket_id": 1}, {"ticket_id": 1, "state": "open"}),
                ("close_ticket", {"ticket_id": 1}, {"ticket_id": 1, "state": "closed"}),
            ],
            "t2": [
                ("get_ticket", {"ticket_id": 2}, {"ticket_id": 2, "state": "open"}),
                (
                    "close_ticket",
                    {"ticket_id": 2},
                    {"error": "forbidden", "ticket_id": 2, "state": "open"},
                    "error",
                ),
            ],
        }
    )
    effect = _effect(infer_schema(corpus).entity("tickets"), "close_ticket")

    assert effect.evidence_count == 1
    # Had the error counted, 'state' would have been open on one call and
    # closed on the other, and the constant would have vanished.
    assert effect.sets_constants == {"state": "closed"}
    assert "error" not in effect.response_echoes


def test_write_effect_without_a_diff_is_capped_and_says_so():
    """A write that is the first sighting of its row rests on its own response only.

    Needs a surface: with no prior observation there is no diff, so
    ``infer_writers`` cannot see the write at all - which is exactly why the
    confidence is capped.
    """
    corpus = corpus_from_calls(
        {
            "t1": [
                (
                    "assign_ticket",
                    {"ticket_id": 1, "owner": "bob"},
                    {"ticket_id": 1, "owner": "bob", "queue": "support"},
                ),
                ("get_ticket", {"ticket_id": 1}, {"ticket_id": 1, "owner": "bob", "state": "open"}),
            ],
        }
    )
    schema = infer_schema(corpus, _surface(assign_ticket="write", get_ticket="read"))
    effect = _effect(schema.entity("tickets"), "assign_ticket")

    assert effect.confidence <= NO_DIFF_CONFIDENCE
    # 'state' came from the following read, not from the write, so it is not
    # claimed as something assign_ticket set.
    assert "state" not in effect.sets_constants


def test_argument_column_mapping_needs_a_value_match_not_a_name_match():
    """Similar names alone prove nothing: no value match, no mapping."""
    corpus = corpus_from_calls(
        {
            "t1": [
                ("get_ticket", {"ticket_id": 1}, {"ticket_id": 1, "owner_name": "ann"}),
                (
                    "assign_ticket",
                    {"ticket_id": 1, "owner": "bob"},
                    {"ticket_id": 1, "owner_name": "robert"},
                ),
            ],
            "t2": [
                ("get_ticket", {"ticket_id": 2}, {"ticket_id": 2, "owner_name": "ann"}),
                (
                    "assign_ticket",
                    {"ticket_id": 2, "owner": "cat"},
                    {"ticket_id": 2, "owner_name": "catherine"},
                ),
            ],
        }
    )
    effect = _effect(infer_schema(corpus).entity("tickets"), "assign_ticket")

    assert effect.argument_columns == {}


# --------------------------------------------------------------------------
# the real pipeline - stage 1 -> stage 2 -> stage 3, no hand-built fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_schema() -> StateSchema:
    """Schema built from the on-disk OTLP export through ingest and surface.

    Everything above this point feeds :mod:`bandits.state` corpora it built
    itself. This fixture is the only one that proves the inference survives real
    upstream output - argument JSON that went through a span attribute, tool
    classes decided by stage 2's classifier rather than asserted by a fixture.
    """
    corpus, registry = load_corpus_and_registry(
        FIXTURE_DIR / "traces.otlp.jsonl",
        source="otlp",
        tools_path=FIXTURE_DIR / "tools.json",
    )
    return infer_schema(corpus, build_surface(corpus, registry))


def test_pipeline_refund_order_write_effect(pipeline_schema):
    orders = pipeline_schema.entity("orders")
    assert orders is not None
    effect = _effect(orders, "refund_order")

    assert effect.key_argument == "order_id"
    assert effect.sets_constants == {"status": "refunded"}
    assert effect.argument_columns == {"amount_cents": "refund_amount_cents"}
    assert set(effect.response_echoes) >= {"order_id", "status", "refund_amount_cents"}
    assert effect.evidence_count == 3


def test_pipeline_update_order_status_write_effect(pipeline_schema):
    effect = _effect(pipeline_schema.entity("orders"), "update_order_status")
    assert effect.key_argument == "order_id"
    assert effect.sets_constants == {}
    assert effect.evidence_count == 1


def test_pipeline_write_effects_cover_written_by(pipeline_schema):
    for entity in pipeline_schema.entities:
        assert {w.tool for w in entity.write_effects} == set(entity.written_by)
