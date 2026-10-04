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
