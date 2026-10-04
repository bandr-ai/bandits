"""Per-prediction attribution of a shared DSPy call history."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from bandits.analyze.rlm_history import scoped_to_history, summarize_history


def test_default_recording_reaches_worker_threads_and_restores_environment(tmp_path, monkeypatch):
    from bandits import ledger

    monkeypatch.delenv("BANDITS_LEDGER", raising=False)
    monkeypatch.delenv("BANDITS_LEDGER_STRICT", raising=False)

    @ledger.project_recording
    def run(*, project):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(ledger.record, {"event_type": "worker", "text": "full response"}).result()
        raise RuntimeError("failed chunk")

    with pytest.raises(RuntimeError, match="failed chunk"):
        run(project=tmp_path)
    rows = [
        json.loads(row) for row in (tmp_path / ".bandits/ledger.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["text"] == "full response"
    assert "BANDITS_LEDGER" not in os.environ
    assert "BANDITS_LEDGER_STRICT" not in os.environ


def test_full_provider_reasoning_is_saved_immediately_without_duplicates(tmp_path, monkeypatch):
    from bandits import providers
    from bandits.analyze.rlm_history import record_history

    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    text = "reasoning " * 5000

    class FakeLM:
        def __init__(self, *args, **kwargs):
            self.history = []
            self.kwargs = kwargs

        def __call__(self):
            self.history.append(
                {
                    "outputs": [{"text": "", "reasoning_content": text}],
                    "messages": [{"role": "user", "content": "prompt"}],
                    "response": SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                finish_reason="length",
                                message=SimpleNamespace(content="", reasoning_content=text),
                            )
                        ]
                    ),
                }
            )
            return ""

    monkeypatch.setitem(sys.modules, "dspy", SimpleNamespace(LM=FakeLM))
    monkeypatch.setattr(providers, "resolve", lambda _: SimpleNamespace(litellm_id="test/model"))
    monkeypatch.setattr(providers, "credentials", lambda *args, **kwargs: {})
    lm = providers.dspy_lm("test/model")
    lm()
    # Already durable before the enclosing prediction finishes or fails.
    from bandits.ledger import read_events

    rows = [row for row in read_events(path) if row["event_type"] == "model_call"]
    assert rows[0]["response"]["raw"]["choices"][0]["message"]["reasoning_content"] == text
    assert rows[0]["response"]["finish_reason"] == "length"
    record_history(lm.history, language_model=lm)
    # This fake never reaches ``forward``, where starts and call ids are now
    # written; its history-fallback row is the only one, and is not repeated.
    assert len(path.read_text().splitlines()) == 1


def test_final_repl_output_is_recorded_without_prompt_truncation(tmp_path, monkeypatch):
    from bandits.analyze.rlm_history import record_repl

    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    output = "last output " * 3000
    rlm = SimpleNamespace(_execute_code=lambda repl, code, inputs: output)
    record_repl(rlm)
    assert rlm._execute_code(None, "print(data)", {"chunk": "original input"}) == output
    from bandits.ledger import read_events

    rows = read_events(path)
    assert rows[0]["variables"] == {"chunk": "original input"}
    assert rows[1]["output"] == output


def test_default_recording_refuses_to_run_without_a_writable_ledger(tmp_path, monkeypatch):
    from pathlib import Path

    from bandits import ledger

    monkeypatch.delenv("BANDITS_LEDGER", raising=False)
    ran = []

    @ledger.project_recording
    def run(*, project):
        ran.append(True)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(ledger.LedgerWriteError, match="disk full"):
        run(project=tmp_path)
    assert ran == []


def test_tokens_are_summed_only_over_the_calls_that_reported_them():
    """A missing usage block contributes nothing, never a zero.

    Summing a zero in for an unreported call produces a total that looks
    authoritative and understates the bill.
    """
    calls, tokens = summarize_history(
        [
            {"usage": {"prompt_tokens": 10, "total_tokens": 12}},
            {"usage": None},
            {},
            {"usage": {"prompt_tokens": 5, "total_tokens": 6}},
        ]
    )

    assert calls == 4, "every entry is a physical call, reported usage or not"
    assert tokens == {"prompt_tokens": 15, "total_tokens": 18}


def test_no_reported_usage_stays_empty_rather_than_becoming_zero():
    calls, tokens = summarize_history([{}, {}])

    assert calls == 2
    assert tokens == {}, "empty reads as unknown; a zero would read as free"


class _SharedHistoryModel:
    """Stands in for a ``dspy.LM``: one history list, appended to forever."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def call(self, count: int, tokens: int) -> None:
        for _ in range(count):
            self.history.append({"usage": {"total_tokens": tokens}})


def test_one_family_is_never_charged_for_the_calls_of_another():
    """`lm.history` is shared for the life of the process, not per prediction.

    Reading it whole would attribute every earlier family's calls to the family
    running now, which is the failure that makes per-family cost meaningless.
    """
    language_model = _SharedHistoryModel()

    def predict(*, members: str, question: str):
        language_model.call(int(members), tokens=10)
        return SimpleNamespace()

    wrapped = scoped_to_history(predict, language_model)

    wrapped(members="3", question="q")
    assert wrapped.spend() == (3, {"total_tokens": 30})

    wrapped(members="2", question="q")
    assert wrapped.spend() == (2, {"total_tokens": 20}), (
        "the second family spent two calls, not the five in the shared history"
    )
    assert len(language_model.history) == 5, "the underlying history still accumulates"


def test_a_failed_prediction_keeps_the_calls_it_completed():
    """The subcalls made before the failure are what explain it."""
    language_model = _SharedHistoryModel()

    def predict(*, members: str, question: str):
        language_model.call(4, tokens=25)
        raise RuntimeError("the sandbox died")

    wrapped = scoped_to_history(predict, language_model)

    with pytest.raises(RuntimeError, match="the sandbox died"):
        wrapped(members="x", question="q")

    assert wrapped.spend() == (4, {"total_tokens": 100})


def test_the_recorded_slice_is_copied_rather_than_referenced():
    """A reference into a growing list would describe later calls too."""
    language_model = _SharedHistoryModel()

    def predict(*, members: str, question: str):
        language_model.call(1, tokens=5)
        return SimpleNamespace()

    wrapped = scoped_to_history(predict, language_model)
    wrapped(members="x", question="q")

    language_model.call(9, tokens=5)

    assert wrapped.spend() == (1, {"total_tokens": 5})


def test_contract_repair_propagates_recording_failure():
    from bandits import ledger
    from bandits.analyze.rlm_mine import _repair_contracts

    def predict(**kwargs):
        raise ledger.LedgerWriteError("disk full")

    with pytest.raises(ledger.LedgerWriteError, match="disk full"):
        _repair_contracts(
            ["bad contract"], chunk_json="[]", taxonomy_json="[]", predict=predict, known=set()
        )


def test_shared_history_is_recorded_once_across_worker_threads(tmp_path, monkeypatch):
    from bandits.analyze.rlm_history import record_history

    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    entries = [{"outputs": ["result"], "messages": [{"role": "user", "content": "prompt"}]}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: record_history(entries), range(20)))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["response"]["text"] == "result"


# --- real DSPy, controlled provider responses, real sandbox --------------------
#
# These run only where the audit extra is installed. Every provider response is
# scripted at ``dspy.LM.forward``; no network or paid call is made. The account
# tests use DSPy's own Deno/Pyodide interpreter, so the evidence helpers are
# exercised through the actual tool bridge rather than a mock interpreter.


def _real_dspy():
    dspy = pytest.importorskip("dspy")
    pytest.importorskip("litellm")
    return dspy


def _sandbox_available() -> bool:
    try:
        from dspy.primitives.python_interpreter import _find_deno_executable, _get_deno_version
    except ImportError:
        return False
    return _get_deno_version(_find_deno_executable()) is not None


def _reply(text: str, finish: str = "stop"):
    from litellm import ModelResponse

    return ModelResponse(
        model="openai/test",
        choices=[
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _action(code: str) -> str:
    return (
        "[[ ## reasoning ## ]]\nnext step\n[[ ## code ## ]]\n```python\n"
        + code
        + "\n```\n[[ ## completed ## ]]"
    )


class _ScriptedProvider:
    """Replays root actions in order; answers extraction calls with ``extract``."""

    def __init__(self, actions, extract=None):
        self.actions = list(actions)
        self.extract = extract
        self.calls: list[str] = []

    def __call__(self, lm, prompt=None, messages=None, **kwargs):
        system = (messages or [{}])[0].get("content", "") if messages else ""
        if "extract the final outputs now" in system:
            self.calls.append("extract")
            return _reply(self.extract)
        self.calls.append("root")
        return _reply(_action(self.actions.pop(0)))


def _workflow_corpus():
    from tests.analyze.rlm_test import _catalog, _workflow_trace

    return _catalog(_workflow_trace("w1"))


def _valid_account_code() -> str:
    from tests.analyze.rlm_test import _account

    account = _account("w1")
    return f"SUBMIT(account={json.dumps(account)})"


def _predictor(dspy, monkeypatch, provider, *, catalog, guard=None, iterations=6):
    from bandits import providers
    from bandits.analyze.rlm_mine import GenerationSettings, build_account_predictor

    def forward(lm, prompt=None, messages=None, **kwargs):
        return provider(lm, prompt=prompt, messages=messages, **kwargs)

    monkeypatch.setattr(dspy.LM, "forward", forward)
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    return build_account_predictor(
        catalog=catalog,
        model="openai/test-model",
        settings=GenerationSettings(root_max_iterations=iterations, max_output_chars=2000),
        guard=guard,
    )


def test_submit_type_feedback_in_real_dspy_through_the_real_sandbox(monkeypatch):
    """A malformed SUBMIT is refused inside the loop with DSPy's own type error,
    the corrected SUBMIT is accepted, and the helpers ran in the real sandbox."""
    dspy = _real_dspy()
    if not _sandbox_available():
        pytest.skip("Deno sandbox not available")
    from bandits.analyze.rlm_mine import _run_account
    from tests.analyze.rlm_test import _identity

    corpus, catalog = _workflow_corpus()
    provider = _ScriptedProvider(
        [
            'page = inspect_run("w1")\n'
            'ev = get_evidence("w1", "clue1.first_model_prompt.json", 0, 50)\n'
            'print(page["total_events"], ev["range_end"], ev["origin"])',
            'SUBMIT(account={"run_id": "w1", "intent": {"status": "maybe"}})',
            _valid_account_code(),
        ]
    )
    predict = _predictor(dspy, monkeypatch, provider, catalog=catalog)
    account = _run_account(corpus, "w1", predict=predict, identity=_identity("w1"), attempt=1)

    assert account.status == "accepted", account.validation_errors or account.completion.error
    assert account.completion.mode == "submit"
    assert account.completion.iterations_to_submit == 3
    assert len(account.completion.submit_rejections) == 1
    assert "account" in account.completion.submit_rejections[0]
    assert account.completion.inspected_ranges == ("clue1.first_model_prompt.json[0:50]",)
    assert provider.calls == ["root", "root", "root"]


def test_iteration_cap_extraction_is_quarantined_in_real_dspy(monkeypatch):
    """Provenance is set before extraction runs; its answer is kept, never accepted."""
    dspy = _real_dspy()
    if not _sandbox_available():
        pytest.skip("Deno sandbox not available")
    from bandits.analyze.rlm_mine import _run_account
    from tests.analyze.rlm_test import _account, _identity

    corpus, catalog = _workflow_corpus()
    extracted = "[[ ## account ## ]]\n" + json.dumps(_account("w1")) + "\n[[ ## completed ## ]]"
    provider = _ScriptedProvider(['print("still looking")'], extract=extracted)
    predict = _predictor(dspy, monkeypatch, provider, catalog=catalog, iterations=1)
    account = _run_account(corpus, "w1", predict=predict, identity=_identity("w1"), attempt=1)

    assert provider.calls == ["root", "extract"]
    assert account.status == "quarantined" and account.failure_kind == "no_submit"
    assert account.completion.mode == "extract"
    assert account.completion.iterations_to_submit is None
    assert account.account is not None and '"run_id": "w1"' in account.candidate
    assert not account.eligible_for_families


def test_session_guard_refuses_calls_before_dispatch_in_real_dspy(monkeypatch):
    """Roots, adapter retries and extraction all pass ``forward``; the third is refused."""
    dspy = _real_dspy()
    if not _sandbox_available():
        pytest.skip("Deno sandbox not available")
    from bandits.analyze.rlm_budget import SessionBudgetGuard
    from bandits.analyze.rlm_mine import _run_account
    from bandits.analyze.rlm_models import StopReason
    from tests.analyze.rlm_test import _identity

    corpus, catalog = _workflow_corpus()
    provider = _ScriptedProvider(['print("a")', 'print("b")', 'print("c")'])
    guard = SessionBudgetGuard(max_calls=2, max_seconds=600)
    predict = _predictor(dspy, monkeypatch, provider, catalog=catalog, guard=guard)
    account = _run_account(corpus, "w1", predict=predict, identity=_identity("w1"), attempt=1)

    assert provider.calls == ["root", "root"], "the refused call never reached the provider"
    assert guard.calls_admitted == 2
    assert account.status == "failed" and account.failure_kind == "budget:max_llm_calls"
    assert guard.refusals and "call ceiling" in guard.refusals[0]
    assert StopReason.MAX_LLM_CALLS.value == "max_llm_calls"


def test_provider_effective_settings_come_from_the_request_body():
    """LiteLLM keeps chat_template_kwargs through its parameter mapping and then
    drops it from the Fireworks body; only the body is evidence."""
    _real_dspy()
    from bandits import providers

    fireworks = "accounts/fireworks/models/test-model"
    checked = providers.preflight_settings(
        fireworks,
        {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 4096,
            "extra_body": {"chat_template_kwargs": {"force_nonempty_content": True}},
        },
    )
    assert checked["verified"] is True
    assert checked["effective"]["temperature"] == 1.0 and checked["effective"]["top_p"] == 0.95
    assert checked["dropped"] == ["extra_body"]

    # LiteLLM allows reasoning_effort per model; the Nemotron id is known to it.
    nemotron = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
    budget = providers.preflight_settings(nemotron, {"reasoning_effort": 1024})
    assert budget["dropped"] == [] and budget["effective"]["reasoning_effort"] == 1024
    with pytest.raises(providers.ProviderError, match="reasoning_effort"):
        providers.preflight_settings(fireworks, {"reasoning_effort": 1024})

    vllm = providers.preflight_settings(
        "hosted_vllm/test-model",
        {"chat_template_kwargs": {"force_nonempty_content": True}},
    )
    assert vllm["dropped"] == []
    assert vllm["effective"]["chat_template_kwargs"] == {"force_nonempty_content": True}

    with pytest.raises(providers.ProviderError, match="does not support"):
        # A model LiteLLM knows refuses top_p alongside temperature.
        providers.preflight_settings(
            "anthropic/claude-sonnet-5", {"temperature": 1.0, "top_p": 0.9}
        )


def test_full_trajectory_accounts_then_families_end_to_end_in_real_dspy(monkeypatch):
    """Both real RLM stages, one shared guard, controlled responses: an accepted
    account becomes the only input family formation sees."""
    dspy = _real_dspy()
    if not _sandbox_available():
        pytest.skip("Deno sandbox not available")
    from bandits import providers
    from bandits.analyze.rlm_budget import SessionBudgetGuard
    from bandits.analyze.rlm_mine import (
        GenerationSettings,
        build_account_predictor,
        build_predictor,
        mine_taxonomy,
    )
    from bandits.analyze.rlm_models import TraceView
    from tests.analyze.rlm_test import _identity

    corpus, catalog = _workflow_corpus()
    family = {
        "contracts": [
            {
                "contract_id": "diagnose-step",
                "name": "Diagnose a failed test step",
                "definition": "explain why the requested test step failed",
                "required_outcome_shape": ["a supported diagnosis of the failed step"],
            }
        ],
        "operations": [
            {
                "operation": "CREATE",
                "contract_ids": ["diagnose-step"],
                "trace_ids": ["w1"],
                "rationale": "first run",
            }
        ],
        "assignments": {"w1": "diagnose-step"},
        "ambiguous_trace_ids": [],
        "uncovered_trace_ids": [],
    }
    seen_chunks: list[dict] = []
    calls: list[str] = []

    def forward(lm, prompt=None, messages=None, **kwargs):
        system = messages[0]["content"]
        if "ACCOUNT of one recorded run" in system:
            calls.append("account")
            return _reply(_action(_valid_account_code()))
        calls.append("family")
        code = f"import json\nrows = json.loads(chunk)\nSUBMIT(**{json.dumps(family)})"
        return _reply(_action(code))

    monkeypatch.setattr(dspy.LM, "forward", forward)
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    guard = SessionBudgetGuard(max_calls=10, max_seconds=600)
    settings = GenerationSettings(root_max_iterations=3)
    account_predict = build_account_predictor(
        catalog=catalog, model="openai/test-model", settings=settings, guard=guard
    )
    family_predict = build_predictor(
        model="openai/test-model",
        view=TraceView.FULL_TRAJECTORY,
        settings=settings,
        guard=guard,
        catalog=corpus.evidence("grouping"),
        accounts_mode=True,
    )

    def spy(**inputs):
        seen_chunks.extend(json.loads(inputs["chunk"]))
        return family_predict(**inputs)

    spy.completion = family_predict.completion
    spy.spend = family_predict.spend
    run = mine_taxonomy(
        corpus,
        "analysis-1",
        predict=spy,
        account_predict=account_predict,
        identity_for=_identity,
        guard=guard,
        contract_repairs=0,
    )
    assert calls == ["account", "family"]
    assert run.assignments == {"w1": "diagnose-step"}
    assert run.chunks[0].completion_mode == "submit"
    assert run.chunks[0].iterations_to_submit == 1
    assert run.accounts[0].status == "accepted"
    assert set(seen_chunks[0]) == {"trace_id", "status", "intent", "milestones", "limitations"}
    assert guard.calls_admitted == 2 and run.complete


# --- recording gate: the synchronous mining path, verified against real DSPy ---


def _rows(path):
    from bandits.ledger import read_events

    return read_events(path)


def _paired(rows):
    """Every start has exactly one terminal row with its call id, and vice versa."""
    starts = [r["call_id"] for r in rows if r["event_type"] == "model_call_start"]
    ends = [r["call_id"] for r in rows if r["event_type"] in ("model_call", "model_call_error")]
    assert len(starts) == len(set(starts))
    assert sorted(starts) == sorted(ends)
    return starts


def test_every_call_in_an_account_invocation_is_recorded_with_lineage(tmp_path, monkeypatch):
    """Roots, a reasoning-only reply and its JSON-adapter fallback, batched and
    single subcalls, REPL steps, helper retrievals and SUBMIT — each recorded
    once, under the invocation and iteration that caused it."""
    dspy = _real_dspy()
    if not _sandbox_available():
        pytest.skip("Deno sandbox not available")
    from bandits import providers
    from bandits.analyze.rlm_mine import GenerationSettings, _run_account, build_account_predictor
    from tests.analyze.rlm_test import _identity

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    monkeypatch.setenv("BANDITS_LEDGER_STRICT", "1")
    corpus, catalog = _workflow_corpus()
    reasoning = "long returned reasoning " * 400
    roots = iter(
        [
            "reasoning-only",
            'ev = get_evidence("w1", "clue0.step_input")\n'
            'print(llm_query_batched(["sub-a", "sub-b"]), llm_query("sub-c"), ev["content"])',
            _valid_account_code(),
        ]
    )

    def forward(lm, prompt=None, messages=None, **kwargs):
        if prompt and prompt.startswith("sub-"):
            return _reply(f"answer to {prompt}", finish="length" if prompt == "sub-b" else "stop")
        if "response_format" in kwargs:  # JSONAdapter fallback after a failed parse
            code = 'print("recovered")'
            return _reply(json.dumps({"reasoning": "retry", "code": code}))
        step = next(roots)
        if step == "reasoning-only":
            from litellm import ModelResponse

            return ModelResponse(
                model="openai/test",
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": reasoning,
                        },
                    }
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 99, "total_tokens": 109},
            )
        return _reply(_action(step))

    monkeypatch.setattr(dspy.LM, "forward", forward)
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    predict = build_account_predictor(
        catalog=catalog,
        model="openai/test-model",
        settings=GenerationSettings(root_max_iterations=6, subcall_workers=2),
    )
    account = _run_account(corpus, "w1", predict=predict, identity=_identity("w1"), attempt=1)
    assert account.status == "accepted"

    rows = _rows(path)
    calls = _paired(rows)
    completed = [r for r in rows if r["event_type"] == "model_call"]
    assert len(calls) == len(completed) == account.llm_calls
    # The reasoning-only, length-limited reply is kept whole.
    assert any(r["response"]["reasoning"] == reasoning for r in completed)
    assert sum(r["response"]["finish_reason"] == "length" for r in completed) == 2
    # Every call — including batched workers — sits under this invocation.
    for row in completed:
        assert row["run_id"] == "w1" and row["attempt"] == 1
    assert len({r["invocation_id"] for r in completed}) == 1, "one invocation"
    subcalls = [r for r in completed if (r["request"]["prompt"] or "").startswith("sub-")]
    roots = [r for r in completed if r not in subcalls]
    assert {r["stage"] for r in roots} == {"rlm_iteration"}
    assert sorted(r["request"]["prompt"] for r in subcalls) == ["sub-a", "sub-b", "sub-c"]
    # Every subcall, batched workers included, sits under the REPL step whose
    # code issued it, inside that step's iteration.
    issuing = next(r for r in rows if r["event_type"] == "repl_start" and "llm_query" in r["code"])
    assert {r["stage"] for r in subcalls} == {"rlm_repl"}
    assert {r["repl_id"] for r in subcalls} == {issuing["repl_id"]}
    assert {r["iteration"] for r in subcalls} == {issuing["iteration"]}
    assert any("response_format" in r["request"]["settings"] for r in completed)
    kinds = [r["event_type"] for r in rows]
    for kind in ("repl_start", "repl_end", "evidence_access", "submit_accepted"):
        assert kind in kinds, kind
    access = next(r for r in rows if r["event_type"] == "evidence_access")
    assert access["returned"]["content"] == "diagnose_step"
    assert not any(r.get("_bandits_recorded") for r in rows)


@pytest.mark.parametrize("history", [{"max_history_size": 1}, {"disable_history": True}])
def test_completions_survive_history_eviction_and_disabled_history(tmp_path, monkeypatch, history):
    dspy = _real_dspy()
    from bandits import providers

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    monkeypatch.setattr(
        dspy.LM, "forward", lambda lm, prompt=None, messages=None, **kw: _reply(f"re: {prompt}")
    )
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    lm = providers.dspy_lm("openai/test-model", cache=False)
    with dspy.context(**history):
        for prompt in ("one", "two", "three"):
            lm(prompt=prompt)
    rows = _rows(path)
    assert len(_paired(rows)) == 3
    assert [r["response"]["text"] for r in rows if r["event_type"] == "model_call"] == [
        "re: one",
        "re: two",
        "re: three",
    ]


def test_a_provider_error_closes_its_own_call(tmp_path, monkeypatch):
    dspy = _real_dspy()
    from bandits import providers

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))

    def forward(lm, prompt=None, messages=None, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(dspy.LM, "forward", forward)
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    lm = providers.dspy_lm("openai/test-model")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        lm(prompt="q")
    rows = _rows(path)
    _paired(rows)
    error = next(r for r in rows if r["event_type"] == "model_call_error")
    assert error["error"] == "provider unavailable" and error["request"]["prompt"] == "q"


def test_the_recorded_request_is_the_body_litellm_built(tmp_path, monkeypatch):
    """Requested settings are not evidence; the body is — including from a
    batched worker thread. Uses real LiteLLM against an in-process transport."""
    _real_dspy()
    import httpx

    from bandits import providers

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    sent: list[dict] = []

    def send(self, request, **kwargs):
        sent.append(json.loads(request.content))
        body = {
            "id": "r",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    lm = providers.dspy_lm(
        "accounts/fireworks/models/test-model",
        api_key="test",
        cache=False,
        num_retries=0,
        temperature=1.0,
        top_p=0.95,
        max_tokens=64,
        extra_body={"chat_template_kwargs": {"force_nonempty_content": True}},
    )
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    lm(prompt="main")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda p: contextvars.copy_context().run(lm, prompt=p), ["w-a", "w-b"]))
    rows = [r for r in _rows(path) if r["event_type"] == "model_call"]
    assert len(rows) == 3 == len(sent)
    for row in rows:
        body = row["effective_request"]
        assert body["temperature"] == 1.0 and body["top_p"] == 0.95
        assert "chat_template_kwargs" not in json.dumps(body), "dropped on the wire, so absent"
        assert "extra_body" in row["request"]["settings"], "the request still shows what was asked"
        prompt = row["request"]["prompt"]
        assert body["messages"][-1]["content"] == prompt


def test_async_calls_are_recorded_too(tmp_path, monkeypatch):
    import asyncio

    dspy = _real_dspy()
    from bandits import providers

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))

    async def aforward(lm, prompt=None, messages=None, **kwargs):
        return _reply("async reply")

    monkeypatch.setattr(dspy.LM, "aforward", aforward)
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    lm = providers.dspy_lm("openai/test-model", cache=False)
    asyncio.run(lm.acall(prompt="q"))
    rows = _rows(path)
    assert len(_paired(rows)) == 1
    assert rows[-1]["response"]["text"] == "async reply"


def test_a_write_failure_inside_a_sandbox_tool_stops_the_invocation(tmp_path, monkeypatch):
    """DSPy turns a tool exception into REPL output; the failure must still stop the run."""
    dspy = _real_dspy()
    if not _sandbox_available():
        pytest.skip("Deno sandbox not available")
    from pathlib import Path

    from bandits import ledger
    from bandits.analyze.rlm_mine import _run_account
    from tests.analyze.rlm_test import _identity

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    monkeypatch.setenv("BANDITS_LEDGER_STRICT", "1")
    ledger.clear_failure()
    corpus, catalog = _workflow_corpus()
    provider = _ScriptedProvider(['get_evidence("w1", "clue0.step_input")', _valid_account_code()])
    predict = _predictor(dspy, monkeypatch, provider, catalog=catalog)
    real_open = Path.open
    import contextvars

    inside_tool = contextvars.ContextVar("inside_tool", default=False)

    def failing_open(self, *args, **kwargs):
        if self == path and inside_tool.get():
            raise OSError("disk full")
        return real_open(self, *args, **kwargs)

    original = catalog.get_evidence

    def flagged(*args, **kwargs):
        token = inside_tool.set(True)
        try:
            return original(*args, **kwargs)
        finally:
            inside_tool.reset(token)

    monkeypatch.setattr(catalog, "get_evidence", flagged)
    monkeypatch.setattr(Path, "open", failing_open)
    try:
        with pytest.raises(ledger.LedgerWriteError, match="disk full"):
            _run_account(corpus, "w1", predict=predict, identity=_identity("w1"), attempt=1)
        assert provider.calls == ["root"], "no further model call after the lost record"
    finally:
        ledger.clear_failure()


def test_mine_rlm_cli_end_to_end_with_real_dspy_and_sandbox(tmp_path, monkeypatch):
    """The real command, real DSPy, real sandbox; only provider replies are scripted.

    One run's account submits, another's never does (quarantined), a third is
    accepted with unknown intent. Inspection commands then show each truthfully,
    and the project ledger reconstructs every call.
    """
    dspy = _real_dspy()
    if not _sandbox_available():
        pytest.skip("Deno sandbox not available")
    import re

    from typer.testing import CliRunner

    from bandits import providers
    from bandits.analyze import analyze_corpus, save_analysis
    from bandits.cli import app
    from bandits.store import ArtifactStore, DerivedStore
    from bandits.traces import TraceCorpus
    from tests.analyze.rlm_test import _account, _workflow_trace

    corpus = TraceCorpus(
        source="otlp",
        traces=(_workflow_trace("w1"), _workflow_trace("w2"), _workflow_trace("w3")),
    )
    ArtifactStore(tmp_path / ".bandits").write(corpus, source_path="synthetic")
    analysis_id = save_analysis(
        analyze_corpus(corpus), DerivedStore(tmp_path / ".bandits")
    ).artifact_id
    family = {
        "contracts": [
            {
                "contract_id": "diagnose-step",
                "name": "Diagnose a failed test step",
                "definition": "explain why the requested test step failed",
                "required_outcome_shape": ["a supported diagnosis of the failed step"],
            }
        ],
        "operations": [],
        "assignments": {"w1": "diagnose-step"},
        "ambiguous_trace_ids": [],
        "uncovered_trace_ids": [],
    }

    def forward(lm, prompt=None, messages=None, **kwargs):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "extract the final outputs now" in system:
            extracted = json.dumps(_account("w2"))
            return _reply(f"[[ ## account ## ]]\n{extracted}\n[[ ## completed ## ]]")
        if "ACCOUNT of one recorded run" in system:
            run_id = re.search(r'"run_id": "(w\d)"', user).group(1)
            if run_id == "w2":
                return _reply(_action('print(inspect_run("w2")["total_events"])'))
            account = _account(run_id, status="unknown") if run_id == "w3" else _account(run_id)
            code = (
                f'ev = get_evidence("{run_id}", "clue1.first_model_prompt.json", 0, 64)\n'
                f"SUBMIT(account={json.dumps(account)})"
            )
            return _reply(_action(code))
        return _reply(_action(f"SUBMIT(**{json.dumps(family)})"))

    monkeypatch.setattr(dspy.LM, "forward", forward)
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    monkeypatch.delenv("BANDITS_LEDGER", raising=False)
    runner = CliRunner()
    mined = runner.invoke(
        app,
        [
            "mine-rlm",
            analysis_id,
            "--view",
            "full-trajectory",
            "--model",
            "openai/test-model",
            "--root-max-iterations",
            "2",
            "--max-attempts",
            "1",
            "--contract-repairs",
            "0",
            "--provider-retries",
            "0",
            "--max-llm-calls",
            "24",
            "--chunk-size",
            "3",
            "--project",
            str(tmp_path),
        ],
    )
    out = " ".join(mined.stdout.split())
    assert mined.exit_code == 0, out
    session_id = re.search(r"session: (\S+)", out).group(1)
    draft_id = re.search(r"draft_id: (\S+)", out).group(1)
    assert "account_complete 2" in out and "quarantined 1" in out and "eligible 1" in out
    assert "1 trace(s) quarantined_extract" in out and "1 trace(s) missing_intent" in out

    shown = runner.invoke(
        app, ["rlm-session", session_id, "--accounts", "--project", str(tmp_path)]
    )
    shown_out = " ".join(shown.stdout.split())
    assert shown.exit_code == 0, shown_out
    assert "quarantined w2 attempt 1 · extract" in shown_out
    assert "accepted w1 attempt 1 · submit · iterations 1 (submitted at 1)" in shown_out

    families = runner.invoke(app, ["rlm-families", draft_id, "--project", str(tmp_path)])
    families_out = " ".join(families.stdout.split())
    assert families.exit_code == 0, families_out
    assert "Diagnose a failed test step" in families_out
    assert "unassigned: w2 quarantined_extract" in families_out
    assert "unassigned: w3 missing_intent" in families_out

    rows = _rows(tmp_path / ".bandits" / "ledger.jsonl")
    calls = _paired(rows)
    completed = [r for r in rows if r["event_type"] == "model_call"]
    # w1 one root; w2 two roots then extraction; w3 one root; one family root.
    assert len(calls) == len(completed) == 6
    by_run = {}
    for row in completed:
        by_run.setdefault(row.get("run_id", "family"), []).append(row["stage"])
    assert by_run == {
        "w1": ["rlm_iteration"],
        "w2": ["rlm_iteration", "rlm_iteration", "rlm_extract"],
        "w3": ["rlm_iteration"],
        "family": ["rlm_iteration"],
    }
    assert sum(r["event_type"] == "extract_start" for r in rows) == 1
    assert sum(r["event_type"] == "submit_accepted" for r in rows) == 3  # w1, w3, family
    # Three account invocations and one family invocation, each bracketed and
    # each with its own sandbox; the run itself is bracketed too.
    assert sum(r["event_type"] == "invocation_start" for r in rows) == 4
    assert sum(r["event_type"] == "invocation_end" for r in rows) == 4
    assert sum(r["event_type"] == "sandbox_start" for r in rows) == 4
    assert sum(r["event_type"] == "sandbox_end" for r in rows) == 4
    assert [r["event_type"] for r in rows if r["event_type"].startswith("run_")] == [
        "run_started",
        "run_finished",
    ]
    assert {r.get("session_id") for r in completed} == {session_id}

    # The ledger reconstructs and reconciles the session with no model call.
    checked = runner.invoke(app, ["rlm-ledger", session_id, "--project", str(tmp_path)])
    checked_out = " ".join(checked.stdout.split())
    assert checked.exit_code == 0, checked_out
    assert "calls: 6 started, 6 completed, 0 failed" in checked_out
    assert "consistent" in checked_out
    one = completed[0]["call_id"]
    shown_call = runner.invoke(
        app, ["rlm-ledger", session_id, "--call", one, "--project", str(tmp_path)]
    )
    call_rows = json.loads(shown_call.stdout)
    assert [r["event_type"] for r in call_rows] == ["model_call_start", "model_call"]

    # A call that started and never ended (a killed process) is reported.
    with (tmp_path / ".bandits" / "ledger.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {"event_type": "model_call_start", "call_id": "dangling", "session_id": session_id}
            )
            + "\n"
        )
    broken = runner.invoke(app, ["rlm-ledger", session_id, "--project", str(tmp_path)])
    assert broken.exit_code == 1
    assert "started and never ended" in " ".join(broken.stdout.split())


def _fireworks_transport(monkeypatch, *, content="ok", cost_model=None):
    """Real LiteLLM against an in-process transport; returns the completion bodies sent."""
    import httpx

    sent: list[dict] = []

    def send(self, request, **kwargs):
        if not str(request.url).endswith("/chat/completions"):
            return httpx.Response(404, request=request)
        sent.append(json.loads(request.content))
        body = {
            "id": "r",
            "object": "chat.completion",
            "created": 0,
            "model": cost_model or "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return sent


def test_cache_hits_are_recorded_as_such_and_never_billed(tmp_path, monkeypatch):
    dspy = _real_dspy()
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    sent = _fireworks_transport(monkeypatch)
    # Memory only, so the test writes no disk cache; DSPy's defaults are restored.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    try:
        _check_cache_hits(dspy, path, sent, tmp_path)
    finally:
        dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)


def _check_cache_hits(dspy, path, sent, tmp_path):
    from bandits import providers
    from bandits.analyze.rlm_budget import SessionBudgetGuard

    guard = SessionBudgetGuard(max_calls=10, max_seconds=600)
    lm = providers.dspy_lm(
        "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
        api_key="test",
        cache=True,
        num_retries=0,
        call_guard=guard,
        temperature=0.3,
    )
    prompt = f"cache probe {tmp_path.name}"
    lm(prompt=prompt)
    lm(prompt=prompt)  # same request; only the per-call timeout differs
    assert len(sent) == 1, "the second call never reached the provider"
    rows = [r for r in _rows(path) if r["event_type"] == "model_call"]
    assert [r["cache_hit"] for r in rows] == [False, True]
    assert rows[1]["cost_usd"] == 0.0
    assert guard.cache_hits == 1 and guard.calls_admitted == 2
    assert lm.call_log[1]["cost"] == 0.0 and lm.call_log[1]["cache_hit"] is True


def test_the_typed_dspy_call_path_is_recorded(tmp_path, monkeypatch):
    dspy = _real_dspy()
    from bandits import providers

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    monkeypatch.setattr(
        dspy.LM, "forward", lambda lm, prompt=None, messages=None, **kw: _reply("typed reply")
    )
    monkeypatch.setattr(providers, "credentials", lambda *a, **k: {})
    lm = providers.dspy_lm("openai/test-model", cache=False)
    with dspy.context(experimental=True):
        lm(prompt="typed request")
    rows = _rows(path)
    assert len(_paired(rows)) == 1
    assert rows[-1]["response"]["text"] == "typed reply"


def test_no_credential_reaches_the_ledger_or_its_blobs(tmp_path, monkeypatch):
    dspy = _real_dspy()
    from bandits import providers

    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    monkeypatch.setenv("BANDITS_LEDGER_BLOB_MIN", "10")
    _fireworks_transport(monkeypatch, content="z" * 50)
    lm = providers.dspy_lm(
        "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
        api_key="sk-never-recorded-123",
        cache=False,
        num_retries=0,
    )
    with dspy.context(lm=lm):
        lm(prompt="prompt " * 10)
    written = path.read_text() + "".join(
        blob.read_text() for blob in (tmp_path / "ledger.jsonl.blobs").iterdir()
    )
    assert "sk-never-recorded-123" not in written
    assert any(r.get("effective_request") for r in _rows(path))
