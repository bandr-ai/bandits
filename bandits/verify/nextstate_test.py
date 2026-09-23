from __future__ import annotations

from datetime import UTC, datetime

from bandits.store import DerivedStore
from bandits.traces import Span, SpanKind, SpanStatus, Trace
from bandits.verify.nextstate import (
    Archetype,
    TurnJudgeRun,
    judge_turn,
    judge_turns,
    load_turn_judge_run,
    parse_verdict,
    render_turn_prompt,
    save_turn_judge_run,
    signal_for,
)
from bandits.verify.turns import Reaction, Turn, extract_turns

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


def _trace(trace_id: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        source="test",
        source_digest="d",
        task="fix the bug",
        spans=(
            _span("m1", SpanKind.MODEL, "gpt", "open schema.py"),
            _span("t1", SpanKind.TOOL, "execute", "schema.py not found", SpanStatus.ERROR),
            _span("m2", SpanKind.MODEL, "gpt", "run tests"),
            _span("t2", SpanKind.TOOL, "execute", "3 passed"),
            _span("m3", SpanKind.MODEL, "gpt", "done"),
        ),
    )


def test_parse_verdict_reads_last_boxed_score_and_hint() -> None:
    assert parse_verdict("Hint: it assumed a path.\n\\boxed{-1}") == (-1, "it assumed a path.")
    assert parse_verdict("\\boxed{+1}") == (1, "")
    assert parse_verdict("maybe \\boxed{0} no, \\boxed{ -1 }")[0] == -1
    assert parse_verdict("no score here")[0] is None


def test_prompt_shows_only_this_turn_and_its_reaction() -> None:
    turn = Turn(
        trace_id="t",
        index=2,
        action_span_id="m",
        action="ACT",
        reactions=(Reaction(span_id="x", kind="tool", name="execute", text="REACT", error=True),),
    )
    prompt = render_turn_prompt("TASK", turn, Archetype.CODING, previous_action="PREV")
    assert "<action>\nACT\n</action>" in prompt
    assert "[tool:execute] [ERROR] REACT" in prompt
    assert "<previous_action>\nPREV\n</previous_action>" in prompt
    assert "coding" in prompt
    assert "\\boxed{-1}" in prompt


def test_routine_content_rule_applies_to_browsing_not_coding() -> None:
    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action="ACT",
        reactions=(Reaction(span_id="x", kind="tool", name="page_down", text="page 2 of 9"),),
    )
    lenient = render_turn_prompt(None, turn, Archetype.COMPUTER_USE)
    strict = render_turn_prompt(None, turn, Archetype.CODING)
    assert "Not finding the answer on this step" in lenient
    assert "Not finding the answer on this step" not in strict


def test_unobserved_turn_is_not_sent_to_the_judge() -> None:
    calls = []
    turn = Turn(trace_id="t", index=0, action_span_id="m", action="ACT")
    verdict = judge_turn(
        None,
        turn,
        Archetype.GENERIC,
        predict=lambda *a: calls.append(a) or "\\boxed{+1}",
        model="m",
    )
    assert calls == []
    assert verdict.score is None and not verdict.observed


def test_votes_take_majority_and_tie_is_neutral() -> None:
    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action="ACT",
        reactions=(Reaction(span_id="x", kind="tool", name="e", text="r"),),
    )
    replies = iter(["a \\boxed{-1}", "b \\boxed{+1}", "c \\boxed{-1}"])
    verdict = judge_turn(
        None, turn, Archetype.GENERIC, predict=lambda *a: next(replies), model="m", votes=3
    )
    assert verdict.score == -1 and verdict.votes == (-1, 1, -1) and verdict.hint == "a"
    assert verdict.judge_votes == {"-1": 2 / 3, "0": 0.0, "1": 1 / 3}
    replies = iter(["\\boxed{-1}", "\\boxed{+1}"])
    verdict = judge_turn(
        None, turn, Archetype.GENERIC, predict=lambda *a: next(replies), model="m", votes=2
    )
    assert verdict.score == 0
    assert verdict.judge_votes == {"-1": 0.5, "0": 0.0, "1": 0.5}


def test_single_vote_is_dense_with_observed_zeros_on_the_other_labels() -> None:
    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action="ACT",
        reactions=(Reaction(span_id="x", kind="tool", name="e", text="r"),),
    )
    verdict = judge_turn(
        None, turn, Archetype.GENERIC, predict=lambda *a: "\\boxed{+1}", model="m"
    )
    assert verdict.judge_votes == {"-1": 0.0, "0": 0.0, "1": 1.0}


def test_judge_votes_is_none_when_unobserved_or_failed() -> None:
    turn = Turn(trace_id="t", index=0, action_span_id="m", action="ACT")
    verdict = judge_turn(None, turn, Archetype.GENERIC, predict=lambda *a: "\\boxed{+1}", model="m")
    assert verdict.judge_votes is None
    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action="ACT",
        reactions=(Reaction(span_id="x", kind="tool", name="e", text="r"),),
    )
    verdict = judge_turn(None, turn, Archetype.GENERIC, predict=lambda *a: "no box", model="m")
    assert verdict.judge_votes is None


def test_judge_failure_is_recorded_not_raised() -> None:
    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action="ACT",
        reactions=(Reaction(span_id="x", kind="tool", name="e", text="r"),),
    )

    def broken(*args):
        raise RuntimeError("down")

    verdict = judge_turn(None, turn, Archetype.GENERIC, predict=broken, model="m")
    assert verdict.score is None and verdict.failure.startswith("transport")
    verdict = judge_turn(None, turn, Archetype.GENERIC, predict=lambda *a: "no box", model="m")
    assert verdict.failure.startswith("unparseable")


def test_a_transport_failure_is_not_hidden_by_a_later_unparseable_vote() -> None:
    """A single overwritten ``failure`` variable let whichever vote failed
    last decide the recorded reason -- a transport failure followed by an
    unparseable reply reported "unparseable", so ``judge_turns``' retry pass
    (which only fires on a ``transport`` failure) never recovered the vote
    the transport actually lost."""
    replies = iter([RuntimeError("429"), "no box here", "HINT: none\n\\boxed{+1}"])

    def flaky(*args):
        item = next(replies)
        if isinstance(item, Exception):
            raise item
        return item

    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action="ACT",
        reactions=(Reaction(span_id="x", kind="tool", name="e", text="r"),),
    )
    verdict = judge_turn(None, turn, Archetype.GENERIC, predict=flaky, model="m", votes=3)
    assert verdict.score == 1 and verdict.votes == (1,)
    assert verdict.failure.startswith("transport")


def test_judge_turns_counts_signals_and_round_trips(tmp_path) -> None:
    def predict(model, prompt, temperature):
        return "assumed a path\n\\boxed{-1}" if "not found" in prompt else "\\boxed{+1}"

    run = judge_turns(
        [_trace("a"), _trace("b")],
        "corpus-1",
        Archetype.CODING,
        predict=predict,
        model="m",
        workers=2,
    )
    assert len(run.verdicts) == 6
    signal = run.signal_by_trace()["a"]
    assert (signal.turns, signal.scored, signal.negative, signal.positive, signal.unobserved) == (
        3,
        2,
        1,
        1,
        1,
    )
    assert signal.score == 0.5 and not signal.passes and signal.first_negative_index == 0
    assert not signal.final_observed
    store = DerivedStore(tmp_path)
    envelope = save_turn_judge_run(run, store)
    loaded = load_turn_judge_run(envelope.artifact_id, store)
    assert loaded == run
    assert envelope.summary["negative"] == 2


def test_transport_failures_get_a_second_sequential_pass() -> None:
    seen: dict[str, int] = {}

    def flaky(model, prompt, temperature):
        key = prompt.split("<action>\n", 1)[1].split("\n", 1)[0]
        seen[key] = seen.get(key, 0) + 1
        if key == "open schema.py" and seen[key] < 2:
            raise RuntimeError("429")
        return "HINT: none\n\\boxed{+1}"

    run = judge_turns([_trace("a")], "c", Archetype.CODING, predict=flaky, model="m", workers=1)
    by_index = {v.index: v for v in run.verdicts}
    # One failed attempt in the pool, one successful attempt in the second pass.
    assert seen["open schema.py"] == 2
    assert by_index[0].score == 1 and by_index[0].failure is None
    assert run.signal_by_trace()["a"].failed == 0


def test_signal_for_empty_trace() -> None:
    signal = signal_for("x", (), ())
    assert signal.score is None and not signal.passes


def test_judge_run_model_validates() -> None:
    trace = _trace("a")
    turns = extract_turns(trace)
    run = TurnJudgeRun(
        corpus_id="c",
        archetype=Archetype.SUPPORT,
        model="m",
        prompt_digest="p",
        trace_ids=("a",),
        verdicts=(),
        signals=(signal_for("a", turns, ()),),
    )
    assert run.signals[0].turns == 3
