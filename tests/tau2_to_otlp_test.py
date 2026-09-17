from __future__ import annotations

import json

from bandits.ingest.otlp import load_otlp
from bandits.verify.turns import extract_turns
from scripts.tau2_to_otlp import convert

_TASK_ID = "task-1"


def _corpus(*, trailing_user: bool) -> dict:
    messages = [
        {"role": "user", "content": "I need a refund"},
        {"role": "assistant", "content": "Sure, let me look that up."},
    ]
    if trailing_user:
        messages.append({"role": "user", "content": "Thanks, that's all I needed"})
    return {
        "traces": [
            {
                "trace_id": "t1",
                "metadata": {
                    "tau2_model": "assistant",
                    "tau2_reward": 1.0,
                    "tau2_success": True,
                    "tau2_task_id": _TASK_ID,
                },
                "messages": messages,
                "invocations": [],
            }
        ]
    }


def _load(tmp_path, corpus: dict):
    records, _labels = convert(corpus, flatten=False)
    path = tmp_path / "out.otlp.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return load_otlp(path).traces[0]


def test_a_trailing_customer_message_is_not_dropped(tmp_path) -> None:
    trace = _load(tmp_path, _corpus(trailing_user=True))

    assert [t.text for t in trace.user_turns] == [
        "I need a refund",
        "Thanks, that's all I needed",
    ]
    # Anchored to the real last action, not floating or attached nowhere.
    last_real_span = trace.spans[-2]
    assert trace.user_turns[-1].after_span_id == last_real_span.span_id


def test_a_trailing_customer_message_reaches_the_last_real_turn_as_a_reaction(tmp_path) -> None:
    trace = _load(tmp_path, _corpus(trailing_user=True))

    turns = extract_turns(trace)
    # The synthetic carrier span opens one more, empty, unobserved turn --
    # excluded from every judge/check computation by design -- so the real
    # payoff lands one turn before the end, not on the last one.
    assert turns[-1].observed is False
    assert turns[-2].observed
    assert "Thanks, that's all I needed" in turns[-2].next_state()


def test_a_conversation_that_does_not_end_on_a_customer_message_is_unaffected(tmp_path) -> None:
    trace = _load(tmp_path, _corpus(trailing_user=False))

    assert [t.text for t in trace.user_turns] == ["I need a refund"]
