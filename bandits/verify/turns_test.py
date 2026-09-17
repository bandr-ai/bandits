from __future__ import annotations

from datetime import UTC, datetime

from bandits.traces import Span, SpanKind, SpanStatus, Trace, UserTurn
from bandits.verify.turns import Reaction, Turn, extract_turns, render_action

_T = datetime(2025, 1, 1, tzinfo=UTC)


def _span(span_id: str, kind: SpanKind, name: str, output, status=SpanStatus.OK) -> Span:
    return Span(
        span_id=span_id,
        kind=kind,
        name=name,
        started_at=_T,
        ended_at=_T,
        output=output,
        status=status,
    )


def _trace(spans, user_turns=()) -> Trace:
    return Trace(
        trace_id="t",
        source="test",
        source_digest="d",
        spans=tuple(spans),
        user_turns=tuple(user_turns),
    )


def test_turns_pair_each_action_with_what_followed() -> None:
    trace = _trace(
        [
            _span("m1", SpanKind.MODEL, "gpt", "look at file"),
            _span("t1", SpanKind.TOOL, "read", "contents"),
            _span("t2", SpanKind.TOOL, "run", "boom", SpanStatus.ERROR),
            _span("m2", SpanKind.MODEL, "gpt", "done"),
        ]
    )
    turns = extract_turns(trace)
    assert [t.index for t in turns] == [0, 1]
    assert turns[0].action == "look at file"
    assert [r.name for r in turns[0].reactions] == ["read", "run"]
    assert turns[0].errored
    assert turns[0].observed
    assert "[tool:run] [ERROR] boom" in turns[0].next_state()
    assert not turns[1].observed
    assert turns[1].next_state() is None


def test_user_turns_are_reactions_to_the_action_they_followed() -> None:
    trace = _trace(
        [
            _span("m1", SpanKind.MODEL, "gpt", "I cancelled it"),
            _span("m2", SpanKind.MODEL, "gpt", "ok"),
        ],
        [
            UserTurn(text="start", after_span_id=None),
            UserTurn(text="no, the other order!", after_span_id="m1"),
        ],
    )
    turns = extract_turns(trace)
    assert turns[0].reactions[0].kind == "user"
    assert turns[0].reactions[0].text == "no, the other order!"


def test_tool_results_before_any_action_are_dropped() -> None:
    trace = _trace(
        [_span("t0", SpanKind.TOOL, "init", "x"), _span("m1", SpanKind.MODEL, "gpt", "hi")]
    )
    turns = extract_turns(trace)
    assert len(turns) == 1
    assert turns[0].reactions == ()


def test_action_renders_tool_calls() -> None:
    text = render_action(
        {"content": "searching", "tool_calls": [{"name": "web_search", "arguments": {"q": "x"}}]}
    )
    assert text == 'searching\n→ web_search({"q": "x"})'


def test_as_dict_exposes_what_a_predicate_may_read() -> None:
    trace = _trace([_span("m1", SpanKind.MODEL, "gpt", "a"), _span("t1", SpanKind.TOOL, "x", "y")])
    row = extract_turns(trace)[0].as_dict(task="T")
    assert set(row) == {
        "trace_id",
        "index",
        "task",
        "action",
        "next_state",
        "reactions",
        "observed",
        "errored",
    }
    assert row["reactions"] == [{"kind": "tool", "name": "x", "text": "y", "error": False}]


def test_next_state_enforces_its_limit_as_a_total_not_a_per_reaction_floor() -> None:
    """``per``'s 200-char floor keeps many reactions from being reduced to
    nothing individually, but that alone does not bound the joined total --
    ten 200-char reactions blew a 1200-char limit past 2000 characters."""
    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action="ACT",
        reactions=tuple(
            Reaction(span_id=f"r{i}", kind="tool", name="e", text="x" * 200) for i in range(10)
        ),
    )
    # `_clip`'s own "...[N chars omitted]..." marker adds a little past
    # `limit`, same as everywhere else it's used -- the bug was scaling with
    # reaction count (~2099 chars here before the fix), not that marker.
    assert len(turn.next_state(limit=1200)) < 1300
