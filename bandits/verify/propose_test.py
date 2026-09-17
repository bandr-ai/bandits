from __future__ import annotations

from types import SimpleNamespace

import pytest

from bandits.store import DerivedStore
from bandits.verify.nextstate import Archetype, TurnJudgeRun, TurnVerdict, signal_for
from bandits.verify.propose import (
    CheckStats,
    FamilyCheck,
    FamilyVerifier,
    ProposalError,
    RejectedCheck,
    TraceScore,
    VerifierScores,
    apply_verifier,
    compile_check,
    decide_check,
    evaluate_check,
    load_family_verifier,
    parse_checks,
    propose_verifier,
    revise_check,
    run_check,
    sample_turns,
    save_family_verifier,
    save_verifier_scores,
    survival,
    turn_payload,
)
from bandits.verify.turns import Reaction, Turn


def _turn(trace_id: str, index: int, reaction: str | None, error: bool = False) -> Turn:
    reactions = (
        ()
        if reaction is None
        else (
            Reaction(
                span_id=f"{trace_id}-r{index}",
                kind="tool",
                name="execute",
                text=reaction,
                error=error,
            ),
        )
    )
    return Turn(
        trace_id=trace_id,
        index=index,
        action_span_id=f"{trace_id}-m{index}",
        action=f"act {index}",
        reactions=reactions,
    )


def _turns() -> list[Turn]:
    out = []
    for trace_id in ("a", "b", "c"):
        out += [
            _turn(trace_id, 0, "file not found", error=True),
            _turn(trace_id, 1, "3 passed"),
            _turn(trace_id, 2, "file not found"),
            _turn(trace_id, 3, None),
        ]
    return out


def _judge_run(turns) -> TurnJudgeRun:
    verdicts = []
    for turn in turns:
        score = None if not turn.observed else (-1 if "not found" in turn.next_state() else 1)
        verdicts.append(
            TurnVerdict(
                trace_id=turn.trace_id,
                index=turn.index,
                action_span_id=turn.action_span_id,
                observed=turn.observed,
                score=score,
                hint="assumed a path" if score == -1 else "",
            )
        )
    by_trace: dict[str, list[Turn]] = {}
    for turn in turns:
        by_trace.setdefault(turn.trace_id, []).append(turn)
    return TurnJudgeRun(
        corpus_id="c",
        archetype=Archetype.CODING,
        model="m",
        prompt_digest="p",
        trace_ids=tuple(by_trace),
        verdicts=tuple(verdicts),
        signals=tuple(signal_for(t, own, verdicts) for t, own in by_trace.items()),
    )


GOOD = 'def check(turn):\n    """log says not found"""\n    s = turn["next_state"]\n    return None if s is None else ("not found" in s)\n'
NOISY = "def check(turn):\n    return True\n"
UNOBSERVED_CHECK = "def check(turn):\n    return not turn['observed']\n"


def test_sandbox_rejects_imports_dunders_and_judge_reads() -> None:
    with pytest.raises(RejectedCheck):
        compile_check("import os\ndef check(turn): return True")
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn): return turn.__class__")
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn): return turn['judge'] == -1")
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn): return open('x')")
    with pytest.raises(RejectedCheck):
        compile_check("def a(turn): return True\ndef b(turn): return False")
    with pytest.raises(RejectedCheck):
        compile_check("x = 1")
    with pytest.raises(RejectedCheck):
        compile_check("while True:\n    pass\ndef check(turn): return False")


def test_a_default_evaluated_at_definition_time_is_rejected_not_run() -> None:
    """A default is an expression exec() evaluates before run_check's alarm
    exists to bound it. Rejection must come from the AST alone -- if this
    test ever reaches the actual exec(), it hangs rather than fails."""
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn, x=sum(range(10**10))): return True")
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn, *, x=sum(range(10**10))): return True")


def test_decorators_and_annotations_are_rejected() -> None:
    with pytest.raises(RejectedCheck):
        compile_check("@staticmethod\ndef check(turn): return True")
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn) -> bool: return True")
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn: dict): return True")


def test_try_except_is_rejected_not_run() -> None:
    """SIGALRM interrupts by raising _Timeout() inside whatever is running;
    a try/except inside the check's own body can catch that before it
    reaches run_check's handler. `try: while True: pass / except: pass`
    swallows the interrupt and loops forever, with the alarm already spent
    -- rejection must come from the AST alone, same as the definition-time
    default case, or this test hangs rather than fails."""
    with pytest.raises(RejectedCheck):
        compile_check(
            "def check(turn):\n"
            "    try:\n"
            "        while True:\n"
            "            pass\n"
            "    except:\n"
            "        pass\n"
            "    return True\n"
        )
    with pytest.raises(RejectedCheck):
        compile_check("def check(turn):\n    try:\n        return True\n    except Exception:\n        return False\n")


def test_sandbox_check_cannot_mutate_the_shared_turn() -> None:
    fn = compile_check('def check(turn):\n    turn["next_state"] = "poisoned"\n    return False')
    turn = {"next_state": "original"}
    assert fn(dict(turn)) is False  # sanity: the function itself does mutate its argument
    results, errors = run_check(fn, [{"trace_id": "t", "index": 0, "next_state": "original"}])
    assert errors == 0
    assert results[("t", 0)] is False


def test_decide_check_targets_code_identity_not_name() -> None:
    verifier = propose_verifier(
        _turns(),
        {},
        _judge_run(_turns()),
        "judge-1",
        family_id="fam",
        rounds=1,
        propose=lambda **kw: SimpleNamespace(
            checks=[
                {"name": "same_name", "hypothesis": "a", "code": GOOD},
                {"name": "same_name", "hypothesis": "b", "code": NOISY},
            ]
        ),
    )
    ids = {c.check_id for c in verifier.checks}
    assert len(ids) == 2  # distinct code -> distinct identity, despite the shared name
    good_id = next(c.check_id for c in verifier.checks if c.code == GOOD)
    decided = decide_check(verifier, good_id, "accepted")
    accepted_ids = {c.check_id for c in decided.checks if c.decision == "accepted"}
    assert accepted_ids == {good_id}


def test_a_single_function_under_any_name_is_the_check() -> None:
    fn = compile_check("def check_not_found(turn):\n    return 'not found' in turn['next_state']")
    assert fn({"next_state": "file not found"}) is True
    fn = compile_check("def helper(t): return t\ndef check(turn): return helper(False)")
    assert fn({}) is False


def test_evaluate_check_scores_against_the_judge() -> None:
    turns = _turns()
    run = _judge_run(turns)
    payload = turn_payload(turns, {}, {}, with_judge=False)
    assert "judge" not in payload[0]
    stats, error = evaluate_check(GOOD, payload, run.verdict_by_key())
    assert error is None
    assert (stats.fired, stats.fired_scored, stats.fired_negative, stats.negatives) == (6, 6, 6, 6)
    assert stats.precision == 1.0 and stats.recall == 1.0
    assert survival(stats, min_fired=3, min_precision=0.6) == (
        True,
        "precision vs judge 1.00 over 6 judged turn(s)",
    )
    stats, _ = evaluate_check(NOISY, payload, run.verdict_by_key())
    assert not survival(stats, min_fired=3, min_precision=0.6)[0]
    stats, error = evaluate_check("def check(turn): return 1/0", payload, run.verdict_by_key())
    assert error is None and stats.errors == len(payload) and stats.fired == 0


def test_parse_checks_tolerates_json_strings_and_junk() -> None:
    parsed = parse_checks(
        SimpleNamespace(
            checks='[{"name":"n","hypothesis":"h","code":"def check(turn): return None"}, {"code": "nope"}, 3]'
        )
    )
    assert [c.name for c in parsed] == ["n"]
    assert parse_checks(SimpleNamespace(checks="garbage")) == []


def test_sample_prefers_negatives_and_stays_within_size() -> None:
    turns = _turns()
    run = _judge_run(turns)
    chosen = sample_turns(turns, run.verdict_by_key(), size=4, seed=1)
    assert len(chosen) == 4
    negatives = sum(1 for t in chosen if run.verdict_by_key()[(t.trace_id, t.index)].score == -1)
    assert negatives >= 2


def test_propose_verifier_reexecutes_and_keeps_survivors(tmp_path) -> None:
    turns = _turns()
    run = _judge_run(turns)
    seen: list[dict] = []

    def propose(*, turns: str, library: str, correction: str):
        seen.append({"library": library, "correction": correction, "has_judge": '"judge"' in turns})
        return SimpleNamespace(
            checks=[
                {"name": "not_found", "hypothesis": "log says not found", "code": GOOD},
                {"name": "always", "hypothesis": "everything", "code": NOISY},
                {
                    "name": "broken",
                    "hypothesis": "x",
                    "code": "import os\ndef check(turn): return True",
                },
            ]
        )

    verifier = propose_verifier(
        turns, {"a": "T"}, run, "judge-1", family_id="fam", propose=propose, rounds=2
    )
    assert seen[0]["has_judge"] and seen[0]["library"] == "" and seen[0]["correction"] == ""
    assert "always" in seen[1]["correction"] and "broken" in seen[1]["correction"]
    assert "not_found" in seen[1]["library"]
    names = {c.name: c for c in verifier.checks}
    assert names["not_found"].survived and names["not_found"].decision == "pending"
    assert not names["always"].survived and not names["broken"].survived
    assert names["broken"].reason.startswith("rejected")
    assert verifier.proposed == 6 and len(verifier.checks) == 3

    store = DerivedStore(tmp_path)
    envelope = save_family_verifier(verifier, store)
    assert envelope.summary == {"proposed": 6, "survived": 1, "accepted": 0, "pending": 3}
    assert load_family_verifier(envelope.artifact_id, store) == verifier


def test_a_dead_repl_round_is_retried_over_another_sample_then_skipped() -> None:
    turns = _turns()
    run = _judge_run(turns)
    payloads: list[str] = []

    def flaky(*, turns: str, library: str, correction: str):
        payloads.append(turns)
        if len(payloads) == 1:
            raise ProposalError("the REPL sandbox failed twice")
        return SimpleNamespace(checks=[{"name": "nf", "hypothesis": "h", "code": GOOD}])

    verifier = propose_verifier(
        turns, {}, run, "j", family_id="f", propose=flaky, rounds=1, sample=4
    )
    assert len(payloads) == 2 and payloads[0] != payloads[1]
    assert verifier.failed_rounds == 0 and [c.name for c in verifier.checks] == ["nf"]
    assert verifier.raw_replies[0].startswith("round 1 attempt 1 failed")

    def dead(**kwargs):
        raise ProposalError("the REPL sandbox failed twice")

    with pytest.raises(ProposalError):
        propose_verifier(turns, {}, run, "j", family_id="f", propose=dead, rounds=2)


def test_revise_check_queues_a_scored_child_and_marks_the_parent() -> None:
    turns = _turns()
    run = _judge_run(turns)
    verifier = propose_verifier(
        turns,
        {},
        run,
        "judge-1",
        family_id="fam",
        rounds=1,
        propose=lambda **kw: SimpleNamespace(
            checks=[{"name": "nf", "hypothesis": "log says not found", "code": NOISY}]
        ),
    )
    original = verifier.checks[0]
    assert not original.survived  # NOISY fires on everything; a real reviewer would revise it

    seen: dict[str, str] = {}

    def reviser(*, feedback: str, turns: str) -> SimpleNamespace:
        seen["feedback"] = feedback
        seen["turns"] = turns
        return SimpleNamespace(checks=[{"name": "nf_v2", "hypothesis": "revised", "code": GOOD}])

    revised = revise_check(
        verifier,
        original.check_id,
        "fires on turns with no error at all",
        [("a", 1)],
        turns,
        {},
        run,
        reviser=reviser,
    )
    assert seen["feedback"] == "fires on turns with no error at all"
    assert '"trace_id": "a"' in seen["turns"] and '"judge"' in seen["turns"]

    parent = next(c for c in revised.checks if c.check_id == original.check_id)
    child = next(c for c in revised.checks if c.parent_check_id == original.check_id)
    assert parent.decision == "revised" and child.check_id in parent.note
    assert child.name == "nf_v2" and child.code == GOOD and child.survived
    assert child.round_number == original.round_number + 1
    assert revised.proposed == verifier.proposed + 1

    with pytest.raises(ValueError):
        revise_check(verifier, "missing", "why", [("a", 1)], turns, {}, run, reviser=reviser)
    with pytest.raises(ValueError):
        revise_check(verifier, original.check_id, "why", [], turns, {}, run, reviser=reviser)
    with pytest.raises(ValueError):
        revise_check(
            verifier, original.check_id, "why", [("zzz", 99)], turns, {}, run, reviser=reviser
        )


def test_revise_check_rejects_unchanged_code() -> None:
    """A failed revision that returns the original code must not silently mint
    a second record with the same identity as the first (the exact bug a
    reviewer reported: check_id was pure content-digest, so parent and an
    unchanged "revision" collided, and accepting one accepted both)."""
    turns = _turns()
    run = _judge_run(turns)
    verifier = propose_verifier(
        turns,
        {},
        run,
        "judge-1",
        family_id="fam",
        rounds=1,
        propose=lambda **kw: SimpleNamespace(
            checks=[{"name": "nf", "hypothesis": "h", "code": NOISY}]
        ),
    )
    original = verifier.checks[0]
    unchanged = lambda **kw: SimpleNamespace(  # noqa: E731
        checks=[{"name": "nf", "hypothesis": "h", "code": NOISY}]
    )
    with pytest.raises(ProposalError, match="identical"):
        revise_check(
            verifier, original.check_id, "why", [("a", 1)], turns, {}, run, reviser=unchanged
        )


def test_revise_check_inherits_the_verifiers_own_acceptance_bar() -> None:
    """A verifier built at a stricter bar must hold its revisions to that same
    bar, not the function's own defaults."""
    turns = _turns()
    run = _judge_run(turns)
    verifier = propose_verifier(
        turns,
        {},
        run,
        "judge-1",
        family_id="fam",
        rounds=1,
        min_fired=3,
        min_precision=0.6,
        propose=lambda **kw: SimpleNamespace(
            checks=[{"name": "nf", "hypothesis": "h", "code": NOISY}]
        ),
    )
    assert verifier.min_fired == 3 and verifier.min_precision == 0.6
    original = verifier.checks[0]
    stricter = verifier.replace(min_precision=0.99)

    revised = revise_check(
        stricter,
        original.check_id,
        "why",
        [("a", 1)],
        turns,
        {},
        run,
        reviser=lambda **kw: SimpleNamespace(
            checks=[{"name": "nf_v2", "hypothesis": "h", "code": GOOD}]
        ),
    )
    child = next(c for c in revised.checks if c.parent_check_id == original.check_id)
    # GOOD scores 1.0 precision, so it survives even the 0.99 bar inherited
    # from `stricter` -- proving the bar was actually read from the verifier.
    assert child.survived
    explicit_bar = revise_check(
        verifier,
        original.check_id,
        "why",
        [("a", 1)],
        turns,
        {},
        run,
        reviser=lambda **kw: SimpleNamespace(
            checks=[{"name": "nf_v3", "hypothesis": "h", "code": GOOD}]
        ),
        min_precision=1.01,
    )
    explicit_child = next(
        c for c in explicit_bar.checks if c.parent_check_id == original.check_id
    )
    assert not explicit_child.survived  # an explicit override still wins over the verifier's own


def test_evaluate_check_reports_missed_negatives() -> None:
    turns = _turns()
    run = _judge_run(turns)
    payload = turn_payload(turns, {}, {}, with_judge=False)
    # A check that only ever returns False fires nowhere, so every negative
    # the judge scored is a miss it should be revised to catch.
    stats, error = evaluate_check(
        "def check(turn): return False", payload, run.verdict_by_key()
    )
    assert error is None
    assert stats.fired == 0
    assert set(stats.missed) == {("a", 0), ("a", 2), ("b", 0), ("b", 2), ("c", 0)}
    assert len(stats.missed) == 5  # capped at the default `examples` sample size


def test_sample_payload_is_clipped_but_execution_payload_is_not() -> None:
    long = "x" * 5000
    turn = Turn(
        trace_id="t",
        index=0,
        action_span_id="m",
        action=long,
        reactions=(Reaction(span_id="r", kind="tool", name="e", text=long),),
    )
    clipped = turn_payload([turn], {}, {}, with_judge=False, clip=700)[0]
    assert len(clipped["action"]) < 800 and "chars omitted" in clipped["next_state"]
    assert len(turn_payload([turn], {}, {}, with_judge=False)[0]["action"]) == 5000


def test_decide_and_apply(tmp_path) -> None:
    turns = _turns()
    run = _judge_run(turns)
    verifier = propose_verifier(
        turns,
        {},
        run,
        "judge-1",
        family_id="fam",
        rounds=1,
        propose=lambda **kw: SimpleNamespace(
            checks=[{"name": "nf", "hypothesis": "h", "code": GOOD}]
        ),
    )
    with pytest.raises(ValueError):
        decide_check(verifier, "missing", "accepted")
    check_id = verifier.checks[0].check_id
    accepted = decide_check(verifier, check_id, "accepted", note="yes")
    assert accepted.accepted()[0].note == "yes"

    scores = apply_verifier(
        accepted, turns, {}, run, verifier_id="v", judge_run_id="j", include_judge=False
    )
    by = {s.trace_id: s for s in scores.scores}
    assert by["a"].turns == 4 and by["a"].observed == 3
    assert [f.index for f in by["a"].flagged] == [0, 2]
    assert by["a"].flagged[0].by == (check_id,)
    assert scores.checks_applied == (check_id,)
    assert by["a"].score == pytest.approx(1 - 2 / 3) and not by["a"].passes

    with_judge = apply_verifier(accepted, turns, {}, run, verifier_id="v", judge_run_id="j")
    assert {s.trace_id: s for s in with_judge.scores}["a"].flagged[0].by == (check_id, "judge")
    envelope = save_verifier_scores(with_judge, DerivedStore(tmp_path))
    assert envelope.summary["passing"] == 0


def test_apply_verifier_never_flags_an_unobserved_turn() -> None:
    """A check that happens to be true of an unobserved turn -- the most
    direct case, ``not turn["observed"]`` -- must not be allowed to flag the
    final, reaction-less action: there is no reaction saying it was wrong."""
    turns = [
        _turn("a", 0, "3 passed"),
        _turn("a", 1, None),  # unobserved: the episode's last action
    ]
    run = _judge_run(turns)
    verifier = propose_verifier(
        turns,
        {},
        run,
        "judge-1",
        family_id="fam",
        rounds=1,
        propose=lambda **kw: SimpleNamespace(
            checks=[{"name": "unobserved", "hypothesis": "h", "code": UNOBSERVED_CHECK}]
        ),
    )
    check_id = verifier.checks[0].check_id
    accepted = decide_check(verifier, check_id, "accepted")

    scores = apply_verifier(
        accepted, turns, {}, run, verifier_id="v", judge_run_id="j", include_judge=False
    )
    by = {s.trace_id: s for s in scores.scores}
    assert by["a"].observed == 1
    assert by["a"].flagged == ()
    assert by["a"].passes


def test_apply_verifier_does_not_pass_a_turn_the_judge_never_scored() -> None:
    """A judge that failed to produce a score (transport failure, or an
    unparseable reply that survived retry) is zero signal, not evidence the
    turn was clean. Reading it as clean is the exact bug: one observed turn,
    no checks, a failed judge, used to read as ``passes=True``."""
    turn = _turn("a", 0, "3 passed")
    run = TurnJudgeRun(
        corpus_id="c",
        archetype=Archetype.CODING,
        model="m",
        prompt_digest="p",
        trace_ids=("a",),
        verdicts=(
            TurnVerdict(
                trace_id="a", index=0, action_span_id="a-m0", observed=True, score=None
            ),
        ),
        signals=(),
    )
    verifier = FamilyVerifier(
        family_id="fam",
        archetype=Archetype.CODING,
        corpus_id="c",
        judge_run_id="j",
        model="m",
        prompt_digest="p",
        checks=(),
    )
    scores = apply_verifier(verifier, [turn], {}, run, verifier_id="v", judge_run_id="j")
    by = {s.trace_id: s for s in scores.scores}
    assert by["a"].unresolved == (0,)
    assert by["a"].flagged == ()
    assert not by["a"].passes


def test_apply_verifier_does_not_pass_a_turn_every_check_raised_on() -> None:
    """A check exception is zero signal too, not a quiet "did not fire". With
    the judge excluded and one check that raises on the only observed turn,
    that turn has no source of truth at all."""
    turn = _turn("a", 0, "3 passed")
    run = _judge_run([turn])
    stats = CheckStats(turns=1, fired=0, fired_scored=0, fired_negative=0, fired_positive=0, negatives=0)
    broken = FamilyCheck(
        check_id="broken-000",
        name="broken",
        hypothesis="h",
        code="def check(turn):\n    return 1 / 0\n",
        code_digest="d",
        stats=stats,
        survived=False,
        reason="raised",
        decision="accepted",
    )
    verifier = FamilyVerifier(
        family_id="fam",
        archetype=Archetype.CODING,
        corpus_id="c",
        judge_run_id="j",
        model="m",
        prompt_digest="p",
        checks=(broken,),
    )
    scores = apply_verifier(
        verifier, [turn], {}, run, verifier_id="v", judge_run_id="j", include_judge=False
    )
    by = {s.trace_id: s for s in scores.scores}
    assert by["a"].unresolved == (0,)
    assert by["a"].flagged == ()
    assert not by["a"].passes


def _bare_check(check_id: str, code: str, *, decision: str = "accepted") -> FamilyCheck:
    stats = CheckStats(turns=1, fired=0, fired_scored=0, fired_negative=0, fired_positive=0, negatives=0)
    return FamilyCheck(
        check_id=check_id,
        name=check_id,
        hypothesis="h",
        code=code,
        code_digest="d",
        stats=stats,
        survived=False,
        reason="n/a",
        decision=decision,
    )


def test_an_abstaining_check_does_not_confirm_a_turn_clean() -> None:
    """A check returning None is declining to have an opinion, not
    confirming nothing was wrong -- one accepted check that always abstains,
    with the judge failing too, must leave the turn unresolved rather than
    reading as a clean pass with score 1.0."""
    turn = _turn("a", 0, "3 passed")
    run = TurnJudgeRun(
        corpus_id="c",
        archetype=Archetype.CODING,
        model="m",
        prompt_digest="p",
        trace_ids=("a",),
        verdicts=(
            TurnVerdict(trace_id="a", index=0, action_span_id="a-m0", observed=True, score=None),
        ),
        signals=(),
    )
    verifier = FamilyVerifier(
        family_id="fam",
        archetype=Archetype.CODING,
        corpus_id="c",
        judge_run_id="j",
        model="m",
        prompt_digest="p",
        checks=(_bare_check("abstains-000", "def check(turn):\n    return None\n"),),
    )
    scores = apply_verifier(verifier, [turn], {}, run, verifier_id="v", judge_run_id="j")
    by = {s.trace_id: s for s in scores.scores}
    assert by["a"].unresolved == (0,)
    assert by["a"].flagged == ()
    assert not by["a"].passes
    assert by["a"].score is None


def test_a_successful_check_does_not_mask_another_checks_failure() -> None:
    """Checks are OR'd: a turn is confirmed clean only if *every* applied
    check resolves to an actual boolean. One check returning False while a
    second raises proves nothing -- the one that raised might have fired."""
    turn = _turn("a", 0, "3 passed")
    run = _judge_run([turn])
    verifier = FamilyVerifier(
        family_id="fam",
        archetype=Archetype.CODING,
        corpus_id="c",
        judge_run_id="j",
        model="m",
        prompt_digest="p",
        checks=(
            _bare_check("clean-000", "def check(turn):\n    return False\n"),
            _bare_check("broken-001", "def check(turn):\n    return 1 / 0\n"),
        ),
    )
    scores = apply_verifier(
        verifier, [turn], {}, run, verifier_id="v", judge_run_id="j", include_judge=False
    )
    by = {s.trace_id: s for s in scores.scores}
    assert by["a"].unresolved == (0,)
    assert by["a"].flagged == ()
    assert not by["a"].passes


def test_score_is_undefined_when_unresolved_turns_remain() -> None:
    """Counting an unresolved turn as clean in the score's denominator would
    report a rate over turns nothing actually rated."""
    score = TraceScore(trace_id="a", turns=1, observed=1, flagged=(), unresolved=(0,))
    assert score.score is None
    assert not score.passes


def test_save_verifier_scores_summary_reports_unresolved_counts(tmp_path) -> None:
    scores = VerifierScores(
        verifier_id="v",
        judge_run_id="j",
        include_judge=True,
        checks_applied=(),
        scores=(
            TraceScore(trace_id="a", turns=1, observed=1, flagged=(), unresolved=(0,)),
            TraceScore(trace_id="b", turns=1, observed=1, flagged=()),
        ),
    )
    envelope = save_verifier_scores(scores, DerivedStore(tmp_path))
    assert envelope.summary["unresolved_traces"] == 1
    assert envelope.summary["unresolved_turns"] == 1
    assert envelope.summary["passing"] == 1
