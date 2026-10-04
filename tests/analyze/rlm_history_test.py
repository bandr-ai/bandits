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
    rows = [
        json.loads(row)
        for row in path.read_text().splitlines()
        if json.loads(row)["event_type"] == "model_call"
    ]
    assert rows[0]["response"]["raw"]["choices"][0]["message"]["reasoning_content"] == text
    assert rows[0]["response"]["finish_reason"] == "length"
    record_history(lm.history, language_model=lm)
    assert len(path.read_text().splitlines()) == 2


def test_final_repl_output_is_recorded_without_prompt_truncation(tmp_path, monkeypatch):
    from bandits.analyze.rlm_history import record_repl

    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    output = "last output " * 3000
    rlm = SimpleNamespace(_execute_code=lambda repl, code, inputs: output)
    record_repl(rlm)
    assert rlm._execute_code(None, "print(data)", {"chunk": "original input"}) == output
    rows = [json.loads(row) for row in path.read_text().splitlines()]
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
    assert account.status == "failed" and account.failure_kind == "budget"
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
        catalog=catalog,
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
