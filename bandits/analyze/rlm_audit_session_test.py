"""Tests for resumable, observable audit sessions.

Nothing here reaches a model or a filesystem outside ``tmp_path``: every
predictor is injected and every store is rooted under a pytest tmp dir, so the
suite runs without the ``audit`` extra and without credentials.

The scenario every test here is checking against is the one that motivated
this module: ``audit-rlm`` running unobserved, uncapped, and losing every
finding on Ctrl+C because nothing was persisted until the whole run finished.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from bandits.analyze.rlm_audit import ClusteringAuditError, audit_clustering, prompt_digest
from bandits.analyze.rlm_audit_session import (
    AuditSessionRecorder,
    AuditSessionState,
    AuditSessionStore,
    new_audit_session_id,
)
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_models import (
    AuditBudget,
    Budget,
    FamilyContract,
    RLMClusteringRun,
    StopReason,
    TraceView,
)
from bandits.traces import Span, SpanKind, Trace, TraceCorpus, UserTurn


def _trace(trace_id: str, *messages: str) -> Trace:
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    return Trace(
        trace_id=trace_id,
        source="chat-json",
        source_digest="0" * 64,
        task=messages[0] if messages else None,
        user_turns=tuple(UserTurn(text=text) for text in messages),
        spans=(
            Span(
                span_id=f"{trace_id}:span-0",
                kind=SpanKind.MODEL,
                name="model",
                started_at=moment,
                ended_at=moment,
            ),
        ),
    )


def _corpus(*traces: Trace) -> TraceCorpus:
    return TraceCorpus(source="chat-json", traces=traces)


def _contract(contract_id: str, definition: str = "refund an eligible order") -> FamilyContract:
    return FamilyContract(
        contract_id=contract_id,
        name="Refund an order",
        definition=definition,
        required_outcome_shape=("the order is refunded and the balance reflects it",),
    )


def _run(*, contracts: tuple[FamilyContract, ...], model: str = "test-model") -> RLMClusteringRun:
    return RLMClusteringRun(
        analysis_id="analysis-1",
        view=TraceView.USER_MESSAGES,
        seed=42,
        contracts=contracts,
        stop_reason=StopReason.PASSES_COMPLETE,
        completed_passes=2,
        requested_passes=2,
        budget=Budget(),
        model=model,
        prompt_digest="mining-digest",
    )


def _keep_predict(**_):
    return SimpleNamespace(recommendation="keep", rationale="fine")


# --- checkpointing -----------------------------------------------------------


def test_a_finding_is_persisted_immediately_after_each_contract(tmp_path) -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    store = AuditSessionStore(tmp_path)
    seen_after_first: list[int] = []

    def predict(**_):
        # Read the session file mid-loop, from a second store handle, exactly
        # as a watcher in another process would.
        watcher = AuditSessionStore(tmp_path)
        if watcher.exists(recorder.session_id):
            seen_after_first.append(len(watcher.read(recorder.session_id).findings))
        return SimpleNamespace(recommendation="keep", rationale="fine")

    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
    )

    audit_clustering(run, "run-1", corpus, predict=predict, session=recorder)

    # The second contract's predict call must have seen the first contract's
    # finding already on disk — proof the write happens after each contract,
    # not only once at the very end.
    assert seen_after_first == [0, 1]
    on_disk = store.read("rlm-audit-test")
    assert len(on_disk.findings) == 2
    assert on_disk.completed_contract_ids == ("c1", "c2")


def test_raw_reply_is_persisted_with_the_finding(tmp_path) -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))
    run = _run(contracts=(_contract("c1"),))
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
    )

    audit_clustering(run, "run-1", corpus, predict=_keep_predict, session=recorder)

    on_disk = store.read("rlm-audit-test")
    assert on_disk.findings[0].raw_reply
    assert "keep" in on_disk.findings[0].raw_reply


# --- interruption --------------------------------------------------------------


def test_should_stop_halts_before_the_next_contract_and_saves_progress(tmp_path) -> None:
    corpus = ReadOnlyCorpus(
        _corpus(_trace("t1", "refund"), _trace("t2", "cancel"), _trace("t3", "return"))
    )
    run = _run(contracts=(_contract("c1"), _contract("c2"), _contract("c3")))
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
    )

    calls = {"n": 0}

    def predict(**_):
        calls["n"] += 1
        return SimpleNamespace(recommendation="keep", rationale="fine")

    audit = audit_clustering(
        run,
        "run-1",
        corpus,
        predict=predict,
        session=recorder,
        should_stop=lambda: calls["n"] >= 1,
    )

    assert audit.status == "incomplete"
    assert audit.stop_reason == "interrupted"
    assert len(audit.findings) == 1
    on_disk = store.read("rlm-audit-test")
    assert on_disk.status == "interrupted"
    assert len(on_disk.findings) == 1


def test_a_signal_killed_call_is_not_recorded_as_a_finding(tmp_path) -> None:
    """The exact failure mode from a real SIGINT: the in-flight contract's own
    call dies (its subprocess sandbox killed by the propagated signal), which
    looks like an ordinary exception from inside the try/except. Without the
    should_stop check in the except clause, that becomes a permanent
    "uncertain" finding and completed_contract_ids gains an entry for a
    contract that was never actually audited — so a resume would skip it
    forever. The fix: when should_stop() is already true at the moment a
    contract's call raises, treat it as an interruption, not a failure.
    """
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
    )

    interrupted = {"flag": False}

    def predict(**_):
        # Simulate a SIGINT arriving mid-call: the flag is already set by the
        # time this call raises, exactly as it would be if the handler fired
        # while the sandbox subprocess was still working.
        interrupted["flag"] = True
        raise RuntimeError("Deno exited (code -2) during health check")

    audit = audit_clustering(
        run,
        "run-1",
        corpus,
        predict=predict,
        session=recorder,
        should_stop=lambda: interrupted["flag"],
    )

    assert audit.status == "incomplete"
    assert audit.stop_reason == "interrupted"
    # The killed contract must not appear as any kind of finding.
    assert audit.findings == ()

    on_disk = store.read("rlm-audit-test")
    assert on_disk.status == "interrupted"
    assert on_disk.completed_contract_ids == ()
    assert on_disk.findings == ()

    # Resume must audit c1 again, not skip it.
    audited_ids: list[str] = []

    def predict_after_resume(*, contract: str, **_):
        import json

        audited_ids.append(json.loads(contract)["contract_id"])
        return SimpleNamespace(recommendation="keep", rationale="fine")

    resumed = audit_clustering(
        run, "run-1", corpus, predict=predict_after_resume, model="test-model", resume=on_disk
    )
    assert audited_ids[0] == "c1"
    assert {f.contract_id for f in resumed.findings} == {"c1", "c2"}


def test_a_genuine_failure_unrelated_to_interruption_still_becomes_uncertain() -> None:
    """The should_stop guard must not swallow ordinary failures: a contract
    that fails for its own reasons, with no interrupt in flight, still needs
    an uncertain finding so it isn't silently skipped forever."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))
    run = _run(contracts=(_contract("c1"),))

    def predict(**_):
        raise RuntimeError("provider down")

    audit = audit_clustering(
        run, "run-1", corpus, predict=predict, should_stop=lambda: False
    )
    assert audit.status == "complete"
    assert audit.findings[0].recommendation == "uncertain"
    assert "provider down" in audit.findings[0].rationale


def test_interrupted_session_never_reports_untouched_contracts_as_passed(tmp_path) -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
    )
    audit = audit_clustering(
        run, "run-1", corpus, predict=_keep_predict, session=recorder, should_stop=lambda: True
    )
    assert {f.contract_id for f in audit.findings} == set()
    assert any("never checked, not passed" in lim for lim in audit.limitations)


# --- resume --------------------------------------------------------------------


def test_resume_skips_already_completed_contracts(tmp_path) -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))

    digest = prompt_digest("test-model")
    seed_state = AuditSessionState(
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=digest,
        contract_order=("c1", "c2"),
        completed_contract_ids=("c1",),
        findings=(
            __import__("bandits.analyze.rlm_models", fromlist=["AuditFinding"]).AuditFinding(
                contract_id="c1", recommendation="keep", rationale="already checked"
            ),
        ),
        llm_calls=1,
        cost_usd=0.01,
    )

    audited_ids: list[str] = []

    def predict(*, contract: str, **_):
        import json

        audited_ids.append(json.loads(contract)["contract_id"])
        return SimpleNamespace(recommendation="keep", rationale="fine")

    audit = audit_clustering(
        run, "run-1", corpus, predict=predict, model="test-model", resume=seed_state
    )

    # c1 must never be re-audited — it already has a finding.
    assert audited_ids == ["c2"]
    assert {f.contract_id for f in audit.findings} == {"c1", "c2"}
    c1_finding = next(f for f in audit.findings if f.contract_id == "c1")
    assert c1_finding.rationale == "already checked"


def test_resume_carries_cumulative_cost_forward(tmp_path) -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    digest = prompt_digest("test-model")
    from bandits.analyze.rlm_models import AuditFinding

    seed_state = AuditSessionState(
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=digest,
        contract_order=("c1", "c2"),
        completed_contract_ids=("c1",),
        findings=(AuditFinding(contract_id="c1", recommendation="keep", rationale="ok"),),
        llm_calls=5,
        cost_usd=1.23,
    )

    # No predictor.cost() attached in this stub, so usd stays at the seeded
    # value — the point under test is that it starts from 1.23, not 0.
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=digest,
        seed_state=seed_state,
        resumed_from="rlm-audit-test",
    )
    audit = audit_clustering(
        run, "run-1", corpus, predict=_keep_predict, model="test-model",
        resume=seed_state, session=recorder,
    )
    assert recorder.state.cost_usd >= 1.23
    assert audit.findings  # sanity: still produced findings


def test_resume_rejects_a_different_run_id() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))
    run = _run(contracts=(_contract("c1"),))
    seed_state = AuditSessionState(
        session_id="s1",
        run_id="run-OTHER",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
        contract_order=("c1",),
    )
    with pytest.raises(ClusteringAuditError, match="run-OTHER"):
        audit_clustering(run, "run-1", corpus, predict=_keep_predict, resume=seed_state)


def test_resume_rejects_a_different_model() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))
    run = _run(contracts=(_contract("c1"),))
    seed_state = AuditSessionState(
        session_id="s1",
        run_id="run-1",
        model="other-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("other-model"),
        contract_order=("c1",),
    )
    with pytest.raises(ClusteringAuditError, match="other-model"):
        audit_clustering(
            run, "run-1", corpus, predict=_keep_predict, model="test-model", resume=seed_state
        )


def test_resume_rejects_a_different_view() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")), view=TraceView.FIRST_USER_MESSAGE)
    run = _run(contracts=(_contract("c1"),))
    seed_state = AuditSessionState(
        session_id="s1",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
        contract_order=("c1",),
    )
    with pytest.raises(ClusteringAuditError, match="view"):
        audit_clustering(
            run, "run-1", corpus, predict=_keep_predict, model="test-model", resume=seed_state
        )


def test_resume_rejects_a_changed_prompt_digest() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))
    run = _run(contracts=(_contract("c1"),))
    seed_state = AuditSessionState(
        session_id="s1",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest="a-stale-digest",
        contract_order=("c1",),
    )
    with pytest.raises(ClusteringAuditError, match="prompt"):
        audit_clustering(
            run, "run-1", corpus, predict=_keep_predict, model="test-model", resume=seed_state
        )


# --- budgets ---------------------------------------------------------------


def test_max_contracts_budget_stops_the_session_early(tmp_path) -> None:
    corpus = ReadOnlyCorpus(
        _corpus(_trace("t1", "refund"), _trace("t2", "cancel"), _trace("t3", "return"))
    )
    run = _run(contracts=(_contract("c1"), _contract("c2"), _contract("c3")))
    audit = audit_clustering(
        run,
        "run-1",
        corpus,
        predict=_keep_predict,
        budget=AuditBudget(max_contracts=2),
    )
    assert audit.status == "incomplete"
    assert audit.stop_reason == "max_contracts"
    assert len(audit.findings) == 2


def test_max_llm_calls_budget_stops_the_session_early() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    audit = audit_clustering(
        run, "run-1", corpus, predict=_keep_predict, budget=AuditBudget(max_llm_calls=1)
    )
    assert audit.status == "incomplete"
    assert audit.stop_reason == "max_llm_calls"
    assert len(audit.findings) == 1


def test_a_contract_that_makes_several_dspy_calls_counts_all_of_them(tmp_path) -> None:
    """A contract's audit is a whole dspy.RLM sub-loop, not one physical call.

    Real predictors expose the calls a single invocation made via
    ``predict.spend.entries`` (set by ``scoped_to_history``). A flat +1 per
    contract would silently misreport --max-llm-calls and the progress line's
    call count as "contracts audited" rather than "calls spent" — this proves
    a contract that made several calls is counted as several.
    """
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
    )

    class _Spend:
        def __init__(self, entries):
            self.entries = entries

    def make_predict(n_entries):
        def predict(**_):
            result = SimpleNamespace(recommendation="keep", rationale="fine")
            return result

        predict.spend = _Spend([{"cost": 0.001}] * n_entries)
        predict.cost = lambda: 0.001 * n_entries
        return predict

    calls_seen = []

    def predict(**_):
        # Attach a fresh spend per call, mimicking one contract making 12
        # real DSPy calls and the next making only 1.
        n = 12 if not calls_seen else 1
        calls_seen.append(n)
        predict.spend = _Spend([{"cost": 0.001}] * n)
        predict.cost = lambda: 0.001 * n
        return SimpleNamespace(recommendation="keep", rationale="fine")

    predict.spend = _Spend([])
    predict.cost = lambda: None

    audit = audit_clustering(run, "run-1", corpus, predict=predict, session=recorder)
    assert audit.status == "complete"
    assert len(audit.findings) == 2
    # First contract made 12 calls, second made 1: total must be 13, not 2.
    assert recorder.state.llm_calls == 13


def test_max_llm_calls_stops_mid_run_once_a_contracts_many_calls_cross_it() -> None:
    """--max-llm-calls must be checked against real call volume: a budget of 5
    must stop after one contract that alone made 12 calls, not let a second
    contract start because only "1 contract" had been counted."""
    corpus = ReadOnlyCorpus(
        _corpus(_trace("t1", "refund"), _trace("t2", "cancel"), _trace("t3", "return"))
    )
    run = _run(contracts=(_contract("c1"), _contract("c2"), _contract("c3")))

    class _Spend:
        def __init__(self, entries):
            self.entries = entries

    def predict(**_):
        predict.spend = _Spend([{"cost": 0.001}] * 12)
        predict.cost = lambda: 0.012
        return SimpleNamespace(recommendation="keep", rationale="fine")

    predict.spend = _Spend([])
    predict.cost = lambda: None

    audit = audit_clustering(
        run, "run-1", corpus, predict=predict, budget=AuditBudget(max_llm_calls=5)
    )
    assert audit.status == "incomplete"
    assert audit.stop_reason == "max_llm_calls"
    # Budget of 5 must stop after the first contract's 12 calls blew past it,
    # not after "1 contract" — so only one finding, not two or three.
    assert len(audit.findings) == 1


def test_resumed_elapsed_time_counts_toward_max_seconds() -> None:
    """A session resumed after already running close to --max-seconds must
    stop quickly, not get a fresh clock that ignores what it already spent."""
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    digest = prompt_digest("test-model")

    seed_state = AuditSessionState(
        session_id="s1",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=digest,
        contract_order=("c1", "c2"),
        completed_contract_ids=(),
        findings=(),
        llm_calls=0,
        cost_usd=0.0,
        # Already at, in fact just past, a 60-second budget before this
        # process even starts — the budget check must fire on the very first
        # contract, with no real-time race against how fast the stub returns.
        elapsed_seconds=60.5,
    )

    audit = audit_clustering(
        run,
        "run-1",
        corpus,
        predict=_keep_predict,
        model="test-model",
        resume=seed_state,
        budget=AuditBudget(max_seconds=60.0),
    )
    assert audit.status == "incomplete"
    assert audit.stop_reason == "max_seconds"
    # The baseline alone already exceeds the ceiling, so the check must fire
    # before the first contract is even attempted.
    assert len(audit.findings) == 0
    assert audit.findings == seed_state.findings


def test_a_complete_audit_still_reports_the_advisory_limitation() -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund")))
    run = _run(contracts=(_contract("c1"),))
    audit = audit_clustering(run, "run-1", corpus, predict=_keep_predict)
    assert audit.status == "complete"
    assert any("advisory" in lim for lim in audit.limitations)


# --- no duplicate audits -----------------------------------------------------


def test_every_contract_is_audited_at_most_once_even_across_a_resume() -> None:
    corpus = ReadOnlyCorpus(
        _corpus(_trace("t1", "refund"), _trace("t2", "cancel"), _trace("t3", "return"))
    )
    run = _run(contracts=(_contract("c1"), _contract("c2"), _contract("c3")))
    test_model = "test-model"
    digest = prompt_digest(test_model)

    calls = {"n": 0}

    def predict(**_):
        calls["n"] += 1
        return SimpleNamespace(recommendation="keep", rationale="fine")

    first = audit_clustering(
        run,
        "run-1",
        corpus,
        predict=predict,
        model=test_model,
        budget=AuditBudget(max_contracts=1),
    )
    assert len(first.findings) == 1

    resumed_state = AuditSessionState(
        session_id="s1",
        run_id="run-1",
        model=test_model,
        view=TraceView.USER_MESSAGES,
        prompt_digest=digest,
        contract_order=tuple(c.contract_id for c in run.contracts),
        completed_contract_ids=(first.findings[0].contract_id,),
        findings=first.findings,
    )
    second = audit_clustering(
        run, "run-1", corpus, predict=predict, model=test_model, resume=resumed_state
    )

    all_ids = [f.contract_id for f in second.findings]
    assert sorted(all_ids) == ["c1", "c2", "c3"]
    assert len(all_ids) == len(set(all_ids))
    # c1 was audited once total across both calls, not twice.
    assert calls["n"] == 3  # 1 in `first` + 2 for c2, c3 in `second`


# --- per-contract failures continue the session -----------------------------


def test_a_failing_contract_becomes_uncertain_and_the_session_continues(tmp_path) -> None:
    corpus = ReadOnlyCorpus(_corpus(_trace("t1", "refund"), _trace("t2", "cancel")))
    run = _run(contracts=(_contract("c1"), _contract("c2")))
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="rlm-audit-test",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest=prompt_digest("test-model"),
    )

    def predict(*, contract: str, **_):
        import json

        if json.loads(contract)["contract_id"] == "c1":
            raise RuntimeError("provider down")
        return SimpleNamespace(recommendation="keep", rationale="fine")

    audit = audit_clustering(run, "run-1", corpus, predict=predict, session=recorder)

    assert audit.status == "complete"
    c1 = next(f for f in audit.findings if f.contract_id == "c1")
    assert c1.recommendation == "uncertain"
    c2 = next(f for f in audit.findings if f.contract_id == "c2")
    assert c2.recommendation == "keep"


# --- session store atomicity / listing --------------------------------------


def test_session_store_write_is_atomic_no_partial_file_ever_readable(tmp_path) -> None:
    store = AuditSessionStore(tmp_path)
    state = AuditSessionState(
        session_id="s1",
        run_id="run-1",
        model="test-model",
        view=TraceView.USER_MESSAGES,
        prompt_digest="d",
        contract_order=("c1",),
    )
    store.write(state)
    # No .tmp file left behind after a successful write.
    assert not (store.path("s1").with_suffix(".json.tmp")).exists()
    assert store.read("s1").session_id == "s1"


def test_session_store_lists_newest_first(tmp_path) -> None:
    store = AuditSessionStore(tmp_path)
    early = AuditSessionState(
        session_id="s-early",
        run_id="run-1",
        model="m",
        view=TraceView.USER_MESSAGES,
        prompt_digest="d",
        updated_at="2024-01-01T00:00:00+00:00",
    )
    late = AuditSessionState(
        session_id="s-late",
        run_id="run-1",
        model="m",
        view=TraceView.USER_MESSAGES,
        prompt_digest="d",
        updated_at="2024-01-02T00:00:00+00:00",
    )
    store.write(early)
    store.write(late)
    listed = store.list()
    assert [s.session_id for s in listed] == ["s-late", "s-early"]


def test_interrupt_marks_the_session_interrupted_not_running(tmp_path) -> None:
    store = AuditSessionStore(tmp_path)
    recorder = AuditSessionRecorder(
        store,
        session_id="s1",
        run_id="run-1",
        model="m",
        view=TraceView.USER_MESSAGES,
        prompt_digest="d",
    )
    recorder.begin(contract_order=("c1",))
    recorder.interrupt()
    assert store.read("s1").status == "interrupted"


# --- backward compatibility with mining sessions ----------------------------


def test_audit_sessions_and_mining_sessions_live_in_separate_stores(tmp_path) -> None:
    from bandits.analyze.rlm_session import SessionStore

    mining_store = SessionStore(tmp_path)
    audit_store = AuditSessionStore(tmp_path)
    assert mining_store._root != audit_store._root
    assert audit_store._root.is_relative_to(mining_store._root)


def test_new_audit_session_id_is_distinguishable_from_a_mining_session_id() -> None:
    from bandits.analyze.rlm_session import new_session_id

    audit_id = new_audit_session_id("run-abcd1234")
    mining_id = new_session_id("analysis-abcd1234", TraceView.USER_MESSAGES, 42)
    assert audit_id.startswith("rlm-audit-")
    assert not mining_id.startswith("rlm-audit-")
