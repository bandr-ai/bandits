"""Generate the golden fixture corpus every stage develops against.

A small retail customer-service world with deliberate structure:

  * ID recurrence  - search -> get -> write -> get, on the same order_id, so
    schema inference has something real to latch onto.
  * A write/read pair - refund_order writes what get_order later reads.
  * An external tool - send_email, which must never be "performed".
  * A read-only, never-cross-referenced entity - store_policy, which MUST come
    out as a static snapshot rather than an invented table.
  * Error responses - not_found and already_refunded, so the rebuilt env can
    reproduce adversity.
  * Both labeled outcomes - pass and fail episodes.
  * Two encodings of the SAME episodes - OTLP spans and plain chat JSON - so
    the tool-call recovery path can be tested for equivalence.

Run:  python tests/fixtures/make_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

TOOLS = [
    {
        "name": "get_customer",
        "description": "Fetch a customer by id.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "integer"}, "include_orders": {"type": "boolean"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "search_orders",
        "description": "List order ids for a customer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "status": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_order",
        "description": "Fetch one order by id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "integer"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "refund_order",
        "description": "Refund an order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "integer"}, "amount_cents": {"type": "integer"}},
            "required": ["order_id", "amount_cents"],
        },
    },
    {
        "name": "update_order_status",
        "description": "Set an order status.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "integer"}, "status": {"type": "string"}},
            "required": ["order_id", "status"],
        },
    },
    {
        "name": "get_product",
        "description": "Fetch a product by sku.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
    },
    {
        "name": "get_store_policy",
        "description": "Return the current store refund policy text.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "send_email",
        "description": "Send an email to a customer. Irreversible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to_customer_id": {"type": "integer"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to_customer_id", "subject", "body"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand off to a human agent. Never called in this corpus.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]

POLICY = "Refunds accepted within 30 days of delivery on unopened items."


def _cust(cid, name, email, tier):
    return {"customer_id": cid, "name": name, "email": email, "tier": tier}


def _order(oid, cid, sku, status, cents, placed):
    return {
        "order_id": oid,
        "customer_id": cid,
        "sku": sku,
        "status": status,
        "total_cents": cents,
        "placed_at": placed,
    }


def _prod(sku, title, cents, stock):
    return {"sku": sku, "title": title, "price_cents": cents, "in_stock": stock}


# ---------------------------------------------------------------- episodes

def episodes():
    """Each episode: (id, instruction, [(tool, args, response, status, error)], answer, outcome)."""
    eps = []

    # 1. happy path refund, fully exercises search -> get -> write -> get
    eps.append((
        "ep-refund-ok",
        "I want a refund for my order, my customer id is 88.",
        [
            ("get_customer", {"customer_id": 88},
             _cust(88, "Ada Lovelace", "ada@example.com", "gold"), "ok", None),
            ("search_orders", {"customer_id": 88},
             {"order_ids": [7741, 7742]}, "ok", None),
            ("get_order", {"order_id": 7741},
             _order(7741, 88, "SKU-RED-9", "delivered", 4200, "2026-07-02"), "ok", None),
            ("get_store_policy", {}, {"policy": POLICY}, "ok", None),
            ("refund_order", {"order_id": 7741, "amount_cents": 4200},
             {"order_id": 7741, "status": "refunded", "refund_amount_cents": 4200}, "ok", None),
            ("get_order", {"order_id": 7741},
             _order(7741, 88, "SKU-RED-9", "refunded", 4200, "2026-07-02"), "ok", None),
            ("send_email", {"to_customer_id": 88, "subject": "Your refund",
                            "body": "Refunded 42.00 for order 7741."}, {"sent": True}, "ok", None),
        ],
        "Your order 7741 has been refunded for $42.00 and a confirmation email is on its way.",
        True,
    ))

    # 2. second customer, different shape, gives the entities more than one row
    eps.append((
        "ep-refund-ok-2",
        "Customer 91 wants order 7750 refunded.",
        [
            ("get_customer", {"customer_id": 91},
             _cust(91, "Grace Hopper", "grace@example.com", "silver"), "ok", None),
            ("get_order", {"order_id": 7750},
             _order(7750, 91, "SKU-BLU-1", "delivered", 15900, "2026-07-11"), "ok", None),
            ("get_product", {"sku": "SKU-BLU-1"},
             _prod("SKU-BLU-1", "Blue Widget", 15900, True), "ok", None),
            ("refund_order", {"order_id": 7750, "amount_cents": 15900},
             {"order_id": 7750, "status": "refunded", "refund_amount_cents": 15900}, "ok", None),
            ("send_email", {"to_customer_id": 91, "subject": "Refund processed",
                            "body": "Refunded 159.00."}, {"sent": True}, "ok", None),
        ],
        "Order 7750 refunded for $159.00.",
        True,
    ))

    # 3. error path - not_found. The rebuilt env must be able to produce this.
    eps.append((
        "ep-notfound",
        "Refund order 9999 please.",
        [
            ("get_order", {"order_id": 9999}, {"error": "not_found", "order_id": 9999},
             "error", "not_found"),
            ("send_email", {"to_customer_id": 88, "subject": "Order not found",
                            "body": "We could not find order 9999."}, {"sent": True}, "ok", None),
        ],
        "I could not find order 9999.",
        True,
    ))

    # 4. FAILURE - agent refunds without checking status, order was already refunded
    eps.append((
        "ep-double-refund",
        "Refund order 7741 for customer 88.",
        [
            ("refund_order", {"order_id": 7741, "amount_cents": 4200},
             {"error": "already_refunded", "order_id": 7741}, "error", "already_refunded"),
            ("get_order", {"order_id": 7741},
             _order(7741, 88, "SKU-RED-9", "refunded", 4200, "2026-07-02"), "ok", None),
        ],
        "Something went wrong with the refund.",
        False,
    ))

    # 5. FAILURE - skipped precondition, refunded an undelivered order
    eps.append((
        "ep-bad-precondition",
        "Customer 91 wants to cancel and refund order 7751.",
        [
            ("get_order", {"order_id": 7751},
             _order(7751, 91, "SKU-GRN-3", "in_transit", 2500, "2026-08-01"), "ok", None),
            ("refund_order", {"order_id": 7751, "amount_cents": 2500},
             {"order_id": 7751, "status": "refunded", "refund_amount_cents": 2500}, "ok", None),
        ],
        "Refunded.",
        False,
    ))

    # 6. status update, a second write tool so classification is not single-example
    eps.append((
        "ep-status-update",
        "Mark order 7742 as delivered.",
        [
            ("get_order", {"order_id": 7742},
             _order(7742, 88, "SKU-YEL-7", "in_transit", 999, "2026-07-05"), "ok", None),
            ("update_order_status", {"order_id": 7742, "status": "delivered"},
             {"order_id": 7742, "status": "delivered"}, "ok", None),
            ("get_order", {"order_id": 7742},
             _order(7742, 88, "SKU-YEL-7", "delivered", 999, "2026-07-05"), "ok", None),
        ],
        "Order 7742 is now marked delivered.",
        True,
    ))

    # 7. read-only browsing, gives products more evidence
    eps.append((
        "ep-browse",
        "Is SKU-RED-9 in stock and what does it cost?",
        [
            ("get_product", {"sku": "SKU-RED-9"},
             _prod("SKU-RED-9", "Red Widget", 4200, True), "ok", None),
            ("get_store_policy", {}, {"policy": POLICY}, "ok", None),
        ],
        "Red Widget is $42.00 and in stock.",
        True,
    ))

    return eps


# ---------------------------------------------------------------- encoders

def to_otlp(eps):
    """OTLP-shaped spans using the GenAI attribute mapping."""
    out = []
    for ti, (eid, instruction, calls, answer, outcome) in enumerate(eps):
        tid = f"{ti:032x}"
        spans = [{
            "traceId": tid,
            "spanId": f"{ti:08x}0000",
            "name": "chat retail-agent",
            "startTimeUnixNano": str(1_700_000_000_000_000_000 + ti * 10**9),
            "endTimeUnixNano": str(1_700_000_000_000_000_000 + ti * 10**9 + 5 * 10**8),
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "retail-agent",
                "gen_ai.system": "openai",
                "gen_ai.prompt": json.dumps([{"role": "user", "content": instruction}]),
                "gen_ai.completion": json.dumps([{"role": "assistant", "content": answer}]),
                "bandits.episode_id": eid,
                "bandits.outcome": outcome,
            },
        }]
        for si, (tool, args, resp, status, _err) in enumerate(calls):
            spans.append({
                "traceId": tid,
                "spanId": f"{ti:08x}{si + 1:04x}",
                "parentSpanId": f"{ti:08x}0000",
                "name": f"execute_tool {tool}",
                "startTimeUnixNano": str(1_700_000_000_000_000_000 + ti * 10**9 + si * 10**7),
                "endTimeUnixNano": str(1_700_000_000_000_000_000 + ti * 10**9 + si * 10**7 + 10**6),
                "status": {"code": 2 if status == "error" else 1},
                "attributes": {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool,
                    "gen_ai.tool.call.id": f"call_{eid}_{si}",
                    "gen_ai.tool.call.arguments": json.dumps(args),
                    "gen_ai.tool.message": json.dumps(resp),
                },
            })
        out.append({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]})
    return out


def to_chat(eps):
    """The SAME episodes as plain OpenAI-style chat, with no tool spans at all.

    Recovering invocation points from this is the fallback path in PLAN.md Step 6.
    A correct ingest must produce identical InvocationPoints from both encodings.
    """
    out = []
    for eid, instruction, calls, answer, outcome in eps:
        msgs = [
            {"role": "system", "content": "You are a retail support agent."},
            {"role": "user", "content": instruction},
        ]
        for si, (tool, args, resp, _status, _err) in enumerate(calls):
            cid = f"call_{eid}_{si}"
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": cid,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)},
                }],
            })
            msgs.append({"role": "tool", "tool_call_id": cid, "name": tool,
                         "content": json.dumps(resp)})
        msgs.append({"role": "assistant", "content": answer})
        out.append({"conversation_id": eid, "messages": msgs, "outcome": outcome})
    return out


def main():
    eps = episodes()
    (HERE / "tools.json").write_text(json.dumps(TOOLS, indent=2) + "\n")
    with (HERE / "traces.otlp.jsonl").open("w") as f:
        for rec in to_otlp(eps):
            f.write(json.dumps(rec) + "\n")
    with (HERE / "traces.chat.jsonl").open("w") as f:
        for rec in to_chat(eps):
            f.write(json.dumps(rec) + "\n")

    expected = {
        "episode_count": len(eps),
        "invocation_count": sum(len(e[2]) for e in eps),
        "labeled_pass": sum(1 for e in eps if e[4]),
        "labeled_fail": sum(1 for e in eps if not e[4]),
        "declared_tools": [t["name"] for t in TOOLS],
        "never_called_tools": ["escalate_to_human"],
        "expected_tool_classes": {
            "get_customer": "read", "search_orders": "read", "get_order": "read",
            "get_product": "read", "get_store_policy": "read",
            "refund_order": "write", "update_order_status": "write",
            "send_email": "external", "escalate_to_human": "unknown",
        },
        "expected_entities": ["customers", "orders", "products"],
        "expected_static_entities": ["store_policy"],
        "expected_primary_keys": {"customers": "customer_id", "orders": "order_id",
                                  "products": "sku"},
        "expected_error_modes": {"get_order": ["not_found"],
                                 "refund_order": ["already_refunded"]},
    }
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(json.dumps(expected, indent=2))


if __name__ == "__main__":
    main()
