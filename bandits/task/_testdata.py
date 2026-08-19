"""Hand-written contract objects mirroring the golden retail corpus.

Stages 1-3 are being built in parallel, so this stage does not import them. The
schema and traces below are written by hand as contract objects and match
``tests/fixtures/make_corpus.py`` and ``tests/fixtures/expected.json``: the same
tool classes, the same entities, the same primary keys, the same episodes.
Underscore-prefixed so pytest never collects it as a test module.
"""

from __future__ import annotations

from bandits.contracts import (
    CallStatus,
    EntitySchema,
    FieldProfile,
    InvocationPoint,
    JsonObject,
    JsonValue,
    Message,
    StateSchema,
    Trace,
    TraceCorpus,
)

POLICY = "Refunds accepted within 30 days of delivery on unopened items."

WRITE_TOOLS = {"refund_order", "update_order_status"}
EXTERNAL_TOOLS = {"send_email", "escalate_to_human"}


def _f(name: str, jtype: str, samples: tuple[JsonValue, ...], ident: bool = False) -> FieldProfile:
    return FieldProfile(
        name=name,
        json_types=(jtype,),
        occurrences=len(samples),
        distinct_values=len(set(map(str, samples))),
        sample_values=samples,
        looks_like_identifier=ident,
    )


RETAIL_SCHEMA = StateSchema(
    entities=(
        EntitySchema(
            name="customers",
            primary_key="customer_id",
            fields=(
                _f("customer_id", "integer", (88, 91), ident=True),
                _f("name", "string", ("Ada Lovelace", "Grace Hopper")),
                _f("email", "string", ("ada@example.com", "grace@example.com")),
                _f("tier", "string", ("gold", "silver")),
            ),
            read_by=("get_customer",),
            evidence_count=2,
        ),
        EntitySchema(
            name="orders",
            primary_key="order_id",
            fields=(
                _f("order_id", "integer", (7741, 7742, 7750, 7751), ident=True),
                _f("customer_id", "integer", (88, 91), ident=True),
                _f("sku", "string", ("SKU-RED-9", "SKU-BLU-1", "SKU-GRN-3", "SKU-YEL-7")),
                _f("status", "string", ("delivered", "refunded", "in_transit")),
                _f("total_cents", "integer", (4200, 15900, 2500, 999)),
                _f("placed_at", "string", ("2026-07-02", "2026-07-11", "2026-08-01")),
            ),
            read_by=("get_order", "search_orders"),
            written_by=("refund_order", "update_order_status"),
            evidence_count=6,
        ),
        EntitySchema(
            name="products",
            primary_key="sku",
            fields=(
                _f("sku", "string", ("SKU-RED-9", "SKU-BLU-1"), ident=True),
                _f("title", "string", ("Red Widget", "Blue Widget")),
                _f("price_cents", "integer", (4200, 15900)),
                _f("in_stock", "boolean", (True,)),
            ),
            read_by=("get_product",),
            evidence_count=2,
        ),
        EntitySchema(
            name="store_policy",
            fields=(_f("policy", "string", (POLICY,)),),
            read_by=("get_store_policy",),
            static_snapshot=True,
            evidence_count=2,
        ),
    )
)


def order(oid: int, cid: int, sku: str, status: str, cents: int, placed: str) -> JsonObject:
    return {
        "order_id": oid,
        "customer_id": cid,
        "sku": sku,
        "status": status,
        "total_cents": cents,
        "placed_at": placed,
    }


def customer(cid: int, name: str, email: str, tier: str) -> JsonObject:
    return {"customer_id": cid, "name": name, "email": email, "tier": tier}


def product(sku: str, title: str, cents: int, stock: bool) -> JsonObject:
    return {"sku": sku, "title": title, "price_cents": cents, "in_stock": stock}


def make_trace(
    trace_id: str,
    instruction: str,
    calls: list[tuple],
    answer: str,
    outcome: bool | None,
) -> Trace:
    invs = []
    for step, spec in enumerate(calls):
        tool, args, resp = spec[0], spec[1], spec[2]
        status = spec[3] if len(spec) > 3 else "ok"
        err = spec[4] if len(spec) > 4 else None
        invs.append(
            InvocationPoint(
                call_id=f"call_{trace_id}_{step}",
                trace_id=trace_id,
                step=step,
                tool=tool,
                arguments=args,
                response=resp,
                status=CallStatus.ERROR if status == "error" else CallStatus.OK,
                error_kind=err,
            )
        )
    return Trace(
        trace_id=trace_id,
        source="chat",
        source_digest="0" * 64,
        messages=(
            Message(role="system", content="You are a retail support agent."),
            Message(role="user", content=instruction),
            Message(role="assistant", content=answer),
        ),
        invocations=tuple(invs),
        outcome=outcome,
    )


def ep_refund_ok() -> Trace:
    """Golden episode 1: search -> get -> write -> get, plus an external effect."""
    return make_trace(
        "ep-refund-ok",
        "I want a refund for my order, my customer id is 88.",
        [
            ("get_customer", {"customer_id": 88}, customer(88, "Ada Lovelace", "ada@example.com", "gold")),
            ("search_orders", {"customer_id": 88}, {"order_ids": [7741, 7742]}),
            ("get_order", {"order_id": 7741}, order(7741, 88, "SKU-RED-9", "delivered", 4200, "2026-07-02")),
            ("get_store_policy", {}, {"policy": POLICY}),
            (
                "refund_order",
                {"order_id": 7741, "amount_cents": 4200},
                {"order_id": 7741, "status": "refunded", "refund_amount_cents": 4200},
            ),
            # THE LEAK: this read happens after the write. status is post-state.
            ("get_order", {"order_id": 7741}, order(7741, 88, "SKU-RED-9", "refunded", 4200, "2026-07-02")),
            (
                "send_email",
                {"to_customer_id": 88, "subject": "Your refund", "body": "Refunded 42.00 for order 7741."},
                {"sent": True},
            ),
        ],
        "Your order 7741 has been refunded for $42.00 and a confirmation email is on its way.",
        True,
    )


def ep_cross_entity() -> Trace:
    """A read of `products` AFTER a write to `orders`: still valid pre-state."""
    return make_trace(
        "ep-cross-entity",
        "Refund order 7750 for customer 91 and tell me what the item was.",
        [
            ("get_order", {"order_id": 7750}, order(7750, 91, "SKU-BLU-1", "delivered", 15900, "2026-07-11")),
            (
                "refund_order",
                {"order_id": 7750, "amount_cents": 15900},
                {"order_id": 7750, "status": "refunded", "refund_amount_cents": 15900},
            ),
            # after the write, but a different entity - untouched, so it counts
            ("get_product", {"sku": "SKU-BLU-1"}, product("SKU-BLU-1", "Blue Widget", 15900, True)),
            # after the write, different ROW of the written entity - also counts
            ("get_order", {"order_id": 7742}, order(7742, 88, "SKU-YEL-7", "in_transit", 999, "2026-07-05")),
            (
                "send_email",
                {"to_customer_id": 91, "subject": "Refund processed", "body": "Refunded 159.00."},
                {"sent": True},
            ),
        ],
        "Order 7750 (Blue Widget) refunded for $159.00.",
        True,
    )


def ep_double_refund() -> Trace:
    """Golden episode 4: labeled FAIL, and writes before it ever reads."""
    return make_trace(
        "ep-double-refund",
        "Refund order 7741 for customer 88.",
        [
            (
                "refund_order",
                {"order_id": 7741, "amount_cents": 4200},
                {"error": "already_refunded", "order_id": 7741},
                "error",
                "already_refunded",
            ),
            ("get_order", {"order_id": 7741}, order(7741, 88, "SKU-RED-9", "refunded", 4200, "2026-07-02")),
        ],
        "Something went wrong with the refund.",
        False,
    )


def ep_status_update() -> Trace:
    return make_trace(
        "ep-status-update",
        "Mark order 7742 as delivered.",
        [
            ("get_order", {"order_id": 7742}, order(7742, 88, "SKU-YEL-7", "in_transit", 999, "2026-07-05")),
            (
                "update_order_status",
                {"order_id": 7742, "status": "delivered"},
                {"order_id": 7742, "status": "delivered"},
            ),
            ("get_order", {"order_id": 7742}, order(7742, 88, "SKU-YEL-7", "delivered", 999, "2026-07-05")),
        ],
        "Order 7742 is now marked delivered.",
        True,
    )


def ep_no_invocations() -> Trace:
    return Trace(
        trace_id="ep-chatter",
        source="chat",
        source_digest="0" * 64,
        messages=(
            Message(role="user", content="What are your opening hours?"),
            Message(role="assistant", content="Nine to five."),
        ),
        outcome=True,
    )


def ep_no_instruction() -> Trace:
    return Trace(
        trace_id="ep-headless",
        source="chat",
        source_digest="0" * 64,
        messages=(Message(role="assistant", content="Done."),),
        invocations=(
            InvocationPoint(
                call_id="c0", trace_id="ep-headless", step=0, tool="get_order",
                arguments={"order_id": 7741},
                response=order(7741, 88, "SKU-RED-9", "delivered", 4200, "2026-07-02"),
            ),
        ),
        outcome=True,
    )


def corpus() -> TraceCorpus:
    return TraceCorpus(
        source="chat",
        traces=(
            ep_refund_ok(),
            ep_cross_entity(),
            ep_double_refund(),
            ep_status_update(),
            ep_no_invocations(),
            ep_no_instruction(),
        ),
    )
