"""The ledger is what makes a paid run inspectable, so its failure modes matter.

Two of them would be silent: a call attributed to the wrong stage, and a secret
written to a file that ships beside the corpus. Both are asserted here.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from bandits import ledger


@pytest.fixture
def ledger_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "nested" / "run.jsonl"
    monkeypatch.setenv("BANDITS_LEDGER", str(path))
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_nothing_is_written_unless_the_ledger_is_asked_for(tmp_path, monkeypatch):
    """Opt-in: prompts carry customer data, so recording is never the default."""
    monkeypatch.delenv("BANDITS_LEDGER", raising=False)

    with ledger.model_call(provider="fireworks", model="m", request={"prompt": "hi"}) as call:
        call["response"] = {"usage": {"total_tokens": 5}}

    assert not ledger.enabled()
    assert list(tmp_path.iterdir()) == []


def test_a_successful_call_records_what_it_asked_and_what_it_cost(ledger_path):
    with ledger.model_call(provider="fireworks", model="m", request={"prompt": "hi"}) as call:
        call["response"] = {"usage": {"total_tokens": 5}, "choices": []}

    (row,) = _rows(ledger_path)
    assert row["event_type"] == "model_call"
    assert row["status"] == "success"
    assert row["request"]["prompt"] == "hi"
    assert row["usage"] == {"total_tokens": 5}
    assert row["duration_seconds"] >= 0
    assert row["started_at"] and row["finished_at"]


def test_a_failed_call_is_recorded_and_the_error_still_propagates(ledger_path):
    """The failure is the record most worth having: it still spent budget."""
    with pytest.raises(RuntimeError, match="429"):
        with ledger.model_call(provider="fireworks", model="m", request={}):
            raise RuntimeError("429 rate limited")

    (row,) = _rows(ledger_path)
    assert row["status"] == "error"
    assert row["error"]["type"] == "RuntimeError"
    assert "429" in row["error"]["message"]
    assert row["duration_seconds"] >= 0


def test_a_stage_names_the_calls_made_inside_it(ledger_path):
    with ledger.stage("family_audit", family_id="family-1"):
        with ledger.model_call(provider="fireworks", model="m", request={}):
            pass

    (row,) = _rows(ledger_path)
    assert row["stage"] == "family_audit"
    assert row["family_id"] == "family-1"


def test_a_nested_stage_points_at_the_one_that_contains_it(ledger_path):
    with ledger.stage("audit_run"):
        with ledger.stage("family_audit", family_id="family-1"):
            ledger.record({"event_type": "marker"})

    (row,) = _rows(ledger_path)
    assert row["parent_event_id"] and row["event_id"] != row["parent_event_id"]


def test_a_stage_does_not_leak_into_what_follows_it(ledger_path):
    with ledger.stage("family_audit", family_id="family-1"):
        ledger.record({"event_type": "inside"})
    ledger.record({"event_type": "outside"})

    inside, outside = _rows(ledger_path)
    assert inside["family_id"] == "family-1"
    assert "family_id" not in outside, "a family id outliving its family misattributes the call"


def test_concurrent_stages_do_not_cross_contaminate(ledger_path):
    """Families audited in parallel must not be charged for each other."""
    barrier = threading.Barrier(4)

    def run(name: str) -> None:
        with ledger.stage("family_audit", family_id=name):
            # Every thread sits inside its stage while the others enter theirs,
            # so a shared context would be observed rather than merely possible.
            barrier.wait(timeout=5)
            ledger.record({"event_type": "marker", "wrote": name})

    threads = [threading.Thread(target=run, args=(f"family-{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    for row in _rows(ledger_path):
        assert row["family_id"] == row["wrote"]


def test_a_retry_attempt_records_the_delay_it_was_about_to_wait(ledger_path):
    error = RuntimeError("throttled")
    error.code = 429
    error.headers = {"Retry-After": "8"}

    ledger.record_attempt(attempt=2, error=error, delay=8.0)

    (row,) = _rows(ledger_path)
    assert row["event_type"] == "retry"
    assert row["attempt"] == 2
    assert row["http_status"] == 429
    assert row["retry_after"] == "8"
    assert row["computed_delay_seconds"] == 8.0


def test_no_authorization_header_reaches_the_file(ledger_path):
    """Headers are dropped wholesale rather than filtered.

    An allowlist that misses a field a provider adds in its next release leaks
    it, and nothing recorded here needs a header to be interpretable.
    """
    error = RuntimeError("throttled")
    error.code = 429
    error.headers = {"Retry-After": "3", "Authorization": "Bearer sk-secret-token"}

    with ledger.stage("judge"):
        ledger.record_attempt(attempt=1, error=error, delay=3.0)
        with ledger.model_call(provider="fireworks", model="m", request={"prompt": "hi"}) as call:
            call["response"] = {"usage": {"total_tokens": 1}}

    text = ledger_path.read_text()
    assert "sk-secret-token" not in text
    assert "Authorization" not in text


def test_a_ledger_that_cannot_be_written_never_fails_the_run(tmp_path, monkeypatch):
    """Trading the work for the record of the work is always the wrong trade."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("BANDITS_LEDGER", str(blocker / "run.jsonl"))

    ledger.record({"event_type": "marker"})  # must not raise


def test_completed_events_survive_a_crash_mid_run(ledger_path):
    """Appended line by line, so a run that dies keeps what it already wrote."""
    ledger.record({"event_type": "first"})
    with pytest.raises(RuntimeError):
        with ledger.model_call(provider="fireworks", model="m", request={}):
            raise RuntimeError("killed")

    rows = _rows(ledger_path)
    assert [row["event_type"] for row in rows] == ["first", "model_call"]
