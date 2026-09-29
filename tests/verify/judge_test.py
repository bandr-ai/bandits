"""The judge's model call, end to end through LiteLLM against a local server.

No key and no network: a stub OpenAI-compatible endpoint answers, and what it
receives is the request a provider would have received.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("litellm")

from bandits import providers  # noqa: E402
from bandits.verify.judge import JudgeError, complete  # noqa: E402

MODEL = "accounts/fireworks/models/nemotron"
USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
    "prompt_tokens_details": {"cached_tokens": 4},
}


class _Stub:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.statuses: list[int] = []
        self.base = ""


@pytest.fixture
def stub(monkeypatch, tmp_path) -> Iterator[_Stub]:
    state = _Stub()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
            state.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status = state.statuses.pop(0) if state.statuses else 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status != 200:
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "nope"}}')
                return
            self.end_headers()
            reply = {
                "id": "r",
                "object": "chat.completion",
                "created": 1,
                "model": state.bodies[-1]["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "<think>x</think> \\boxed{1}"},
                    }
                ],
                "usage": USAGE,
            }
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.base = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setattr(
        providers, "credentials", lambda ref: {"api_key": "k", "api_base": state.base}
    )
    monkeypatch.setenv("BANDITS_LEDGER", str(tmp_path / "ledger.jsonl"))
    yield state
    server.shutdown()


def _ledger(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]


def test_fireworks_gets_the_request_the_urllib_client_sent(stub) -> None:
    reply = complete(
        MODEL,
        "judge this",
        0.0,
        system_prompt="policy",
        max_tokens=6000,
        extra={"repetition_penalty": 1.1, "reasoning_effort": "low"},
    )

    # Reasoning is left in the text: the judge's parser has always read it there.
    assert reply == "<think>x</think> \\boxed{1}"
    assert stub.bodies == [
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "judge this"},
            ],
            "temperature": 0.0,
            "max_tokens": 6000,
            # Not mapped by LiteLLM for Fireworks, and sent anyway.
            "repetition_penalty": 1.1,
            "reasoning_effort": "low",
        }
    ]


def test_the_ledger_records_the_provider_and_what_the_user_typed(stub, tmp_path) -> None:
    complete(MODEL, "p", 0.0, extra={"frequency_penalty": 0.3})

    (row,) = _ledger(tmp_path)
    assert row["provider"] == "fireworks_ai"
    # The Jev cost report matches this against the model the dataset records.
    assert row["model"] == MODEL
    assert row["request"]["frequency_penalty"] == 0.3
    assert row["usage"]["prompt_tokens_details"] == {"cached_tokens": 4}


def test_another_provider_is_one_model_string_away(stub) -> None:
    assert complete("hosted_vllm/my-judge", "p", 0.0).endswith("\\boxed{1}")
    assert stub.bodies[0]["model"] == "my-judge"


def test_a_429_is_retried_by_us_and_recorded(stub, tmp_path) -> None:
    stub.statuses = [429]

    complete(MODEL, "p", 0.0)

    # Two requests on the wire, not more: LiteLLM's own retries are off.
    assert len(stub.bodies) == 2
    retry, call = _ledger(tmp_path)
    assert retry["event_type"] == "retry"
    assert retry["http_status"] == 429
    assert retry["retry_after"] == "0"
    assert call["status"] == "success"
    assert call["logical_call_id"] == retry["logical_call_id"]


def test_a_400_fails_once_as_a_judge_error(stub) -> None:
    stub.statuses = [400]

    with pytest.raises(JudgeError, match="judge request failed"):
        complete(MODEL, "p", 0.0)
    assert len(stub.bodies) == 1


def test_a_bare_model_name_is_a_judge_error_before_any_request(stub) -> None:
    with pytest.raises(JudgeError, match="<provider>/<model>"):
        complete("nemotron", "p", 0.0)
    assert stub.bodies == []
