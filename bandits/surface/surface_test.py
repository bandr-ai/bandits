"""Tests for stage 2. Ground truth is tests/fixtures/expected.json."""

from __future__ import annotations

import pytest

from bandits.contracts import (
    CallStatus,
    InvocationPoint,
    Message,
    ToolClass,
    Trace,
    TraceCorpus,
)
from bandits.surface import build_surface
from bandits.surface._fixture import declared_tools, expected, fixture_corpus
from bandits.surface.classify import irreversible_verbs, is_acknowledgement
from bandits.surface.profiling import build_field_profiles, flatten


@pytest.fixture(scope="module")
def surface():
    return build_surface(fixture_corpus(), declared_tools=declared_tools())


@pytest.fixture(scope="module")
def truth():
    return expected()


def _field(profile, name):
    for f in profile:
        if f.name == name:
            return f
    return None


# ------------------------------------------------------------------ the union


def test_surface_unions_declared_and_observed(surface, truth):
    assert [t.name for t in surface.tools] == sorted(truth["declared_tools"])
    assert all(not t.observed_only for t in surface.tools)


def test_observed_only_tool_is_flagged():
    corpus = _corpus_with(
        [
            ("mystery_tool", {"order_id": 1}, {"order_id": 1, "status": "x"}, CallStatus.OK, None),
        ]
    )
    surface = build_surface(corpus, declared_tools=[])
    profile = surface.by_name("mystery_tool")
    assert profile.observed_only is True
    assert profile.declared_only is False


# ------------------------------------------------------------- classification


def test_every_tool_class_matches_expected(surface, truth):
    actual = {t.name: t.tool_class.value for t in surface.tools}
    assert actual == truth["expected_tool_classes"]


def test_every_classification_has_evidence(surface):
    for tool in surface.tools:
        assert tool.class_evidence, f"{tool.name} has no evidence"
        assert all(isinstance(e, str) and e for e in tool.class_evidence)


def test_trusted_registry_class_overrides_incomplete_trace_evidence():
    corpus = _corpus_with([
        ("get_order", {"order_id": 1}, {"order_id": 1, "status": "pending"}, CallStatus.OK, None),
    ])
    surface = build_surface(corpus, declared_tools={
        "get_order": {"tool_class": "read", "input_schema": {"type": "object"}},
    })
    profile = surface.by_name("get_order")
    assert profile.tool_class is ToolClass.READ
    assert profile.class_confidence == 1.0
    assert profile.class_evidence == ("trusted tool registry declares tool_class=read",)


def test_escalate_to_human_is_unknown_declared_only(surface):
    profile = surface.by_name("escalate_to_human")
    assert profile.tool_class is ToolClass.UNKNOWN
    assert profile.declared_only is True
    assert profile.observed_only is False
    assert profile.call_count == 0
    assert profile.class_confidence == 0.0
    joined = " ".join(profile.class_evidence)
    assert "never called" in joined
    # It carries the verb "escalate" and still must not be guessed from the name.
    assert "escalate" in irreversible_verbs("escalate_to_human")
    assert "Step 9" in joined


def test_send_email_is_external_on_ack_shape_plus_verb(surface):
    profile = surface.by_name("send_email")
    assert profile.tool_class is ToolClass.EXTERNAL
    joined = " ".join(profile.class_evidence).lower()
    assert "acknowledgement-shaped" in joined          # the response signal
    assert "send" in joined and "verb" in joined       # the name signal
    assert "name alone is never sufficient" in joined  # and that name alone did not decide it
    assert profile.class_confidence > 0.5


def test_verb_in_name_alone_does_not_make_a_tool_external():
    """A tool named like an external one but returning a real body is a READ.

    This is the guard against name-only classification: `send_report` has the
    verb `send`, no observable state change, and yet must not be stubbed into
    the effect ledger, because it answers with a body a rebuilt store could
    serve.
    """
    corpus = _corpus_with(
        [
            ("send_report", {"report_id": 42},
             {"report_id": 42, "title": "Q3 refunds", "rows": [{"sku": "SKU-RED-9", "n": 3}]},
             CallStatus.OK, None),
            ("send_report", {"report_id": 43},
             {"report_id": 43, "title": "Q4 refunds", "rows": [{"sku": "SKU-BLU-1", "n": 1}]},
             CallStatus.OK, None),
        ]
    )
    profile = build_surface(corpus).by_name("send_report")
    assert "send" in irreversible_verbs("send_report")
    assert profile.tool_class is ToolClass.READ
    joined = " ".join(profile.class_evidence)
    assert "did NOT make this external" in joined


def test_refund_order_is_write_citing_order_7741(surface):
    profile = surface.by_name("refund_order")
    assert profile.tool_class is ToolClass.WRITE
    assert profile.class_confidence >= 0.9
    joined = " ".join(profile.class_evidence)
    assert "7741" in joined
    assert "before" in joined and "after" in joined
    assert "get_order" in joined
    assert '"delivered" -> "refunded"' in joined


def test_update_order_status_is_write(surface):
    profile = surface.by_name("update_order_status")
    assert profile.tool_class is ToolClass.WRITE
    joined = " ".join(profile.class_evidence)
    assert "7742" in joined and "get_order" in joined


def test_reads_are_not_misclassified_as_writes(surface):
    for name in ("get_order", "get_customer", "search_orders", "get_product",
                 "get_store_policy"):
        assert surface.by_name(name).tool_class is ToolClass.READ, name


def test_ack_shapes():
    assert is_acknowledgement({"sent": True})
    assert is_acknowledgement({"ok": True, "status": "queued"})
    assert is_acknowledgement({})
    assert is_acknowledgement(None)
    assert not is_acknowledgement({"policy": "Refunds accepted within 30 days."})
    assert not is_acknowledgement({"order_id": 7741, "status": "refunded"})
    assert not is_acknowledgement({"result": {"nested": 1}})


def test_ack_response_without_a_verb_stays_unknown():
    """No verb, no body, no state change -> ambiguous, never a guess."""
    corpus = _corpus_with([("do_thing", {"x": 1}, {"ok": True}, CallStatus.OK, None)])
    profile = build_surface(corpus).by_name("do_thing")
    assert profile.tool_class is ToolClass.UNKNOWN
    assert "ambiguous" in " ".join(profile.class_evidence)


def test_write_detected_across_a_changed_read():
    corpus = _corpus_with(
        [
            ("fetch_ticket", {"ticket_id": 5}, {"ticket_id": 5, "state": "open"},
             CallStatus.OK, None),
            ("close_ticket", {"ticket_id": 5}, {"ok": True}, CallStatus.OK, None),
            ("fetch_ticket", {"ticket_id": 5}, {"ticket_id": 5, "state": "closed"},
             CallStatus.OK, None),
        ]
    )
    surface = build_surface(corpus)
    assert surface.by_name("close_ticket").tool_class is ToolClass.WRITE
    assert surface.by_name("fetch_ticket").tool_class is ToolClass.READ


# -------------------------------------------------------------- error modes


def test_error_modes_match_expected(surface, truth):
    actual = {
        t.name: sorted(e.error_kind for e in t.error_modes)
        for t in surface.tools
        if t.error_modes
    }
    assert actual == {k: sorted(v) for k, v in truth["expected_error_modes"].items()}


def test_error_mode_keeps_an_example_response(surface):
    mode = surface.by_name("get_order").error_modes[0]
    assert mode.error_kind == "not_found"
    assert mode.occurrences == 1
    assert mode.example_response == {"error": "not_found", "order_id": 9999}


def test_error_responses_do_not_pollute_response_fields(surface):
    names = {f.name for f in surface.by_name("get_order").response_fields}
    assert "error" not in names
    assert "order_id" in names


# ---------------------------------------------------------------- profiling


def test_search_orders_uses_one_of_three_declared_parameters(surface):
    """The '5 of 38 parameters are ever used' signal from PLAN.md."""
    profile = surface.by_name("search_orders")
    declared_params = set(profile.declared_schema["properties"])
    assert declared_params == {"customer_id", "status", "limit"}

    used = {f.name for f in profile.argument_fields}
    assert used == {"customer_id"}

    customer_id = _field(profile.argument_fields, "customer_id")
    assert customer_id.occurrences == profile.call_count
    assert customer_id.null_count == 0
    assert customer_id.looks_like_identifier is True
    assert _field(profile.argument_fields, "status") is None
    assert _field(profile.argument_fields, "limit") is None


def test_declared_schema_is_the_input_schema(surface):
    assert surface.by_name("get_order").declared_schema == {
        "type": "object",
        "properties": {"order_id": {"type": "integer"}},
        "required": ["order_id"],
    }


def test_array_elements_are_profiled(surface):
    profile = surface.by_name("search_orders")
    array = _field(profile.response_fields, "order_ids")
    element = _field(profile.response_fields, "order_ids[]")
    assert array.json_types == ("array",)
    assert element.json_types == ("integer",)
    assert element.occurrences == 2
    assert element.looks_like_identifier is True


def test_nested_objects_flatten_to_dotted_paths():
    pairs = dict(flatten({"order": {"customer_id": 88, "lines": [{"sku": "A"}]}}))
    assert pairs["order.customer_id"] == 88
    assert pairs["order.lines[].sku"] == "A"
    assert pairs["order.lines"] == [{"sku": "A"}]


def test_identifier_by_value_recurrence_without_an_id_name():
    """A field with no id-ish name is still an identifier if its values travel."""
    cross = {"string:\"AB-1\"": frozenset({"other_tool"}),
             "string:\"AB-2\"": frozenset({"other_tool"}),
             "string:\"AB-3\"": frozenset({"other_tool"})}
    profiles = build_field_profiles(
        [{"handle": "AB-1"}, {"handle": "AB-2"}, {"handle": "AB-3"}],
        tool="this_tool",
        cross_tool_arg_values=cross,
    )
    assert _field(profiles, "handle").looks_like_identifier is True

    # Same shape, but the values never appear in any other tool -> not an id.
    profiles = build_field_profiles(
        [{"handle": "AB-1"}, {"handle": "AB-2"}, {"handle": "AB-3"}],
        tool="this_tool",
        cross_tool_arg_values={},
    )
    assert _field(profiles, "handle").looks_like_identifier is False


def test_low_cardinality_shared_value_is_not_an_identifier(surface):
    """`status` is shared between tools but has few distinct values."""
    status = _field(surface.by_name("update_order_status").argument_fields, "status")
    assert status.looks_like_identifier is False


def test_profiles_are_deterministic():
    a = build_surface(fixture_corpus(), declared_tools=declared_tools())
    b = build_surface(fixture_corpus(), declared_tools=declared_tools())
    assert a.model_dump_json() == b.model_dump_json()


def test_call_counts_add_up(surface, truth):
    assert sum(t.call_count for t in surface.tools) == truth["invocation_count"]


# ------------------------------------------------------------------- helpers


def _corpus_with(calls):
    """One synthetic single-trace corpus from ``(tool, args, resp, status, kind)``."""
    invocations = tuple(
        InvocationPoint(
            call_id=f"c{step}",
            trace_id="t-synth",
            step=step,
            tool=tool,
            arguments=args,
            response=response,
            status=status,
            error_kind=kind,
        )
        for step, (tool, args, response, status, kind) in enumerate(calls)
    )
    trace = Trace(
        trace_id="t-synth",
        source="synthetic",
        source_digest="0" * 64,
        messages=(Message(role="user", content="do it"),),
        invocations=invocations,
    )
    return TraceCorpus(source="synthetic", traces=(trace,))
