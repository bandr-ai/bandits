"""τ² trajectories preserve exact tool-call/result pairing and outcome labels."""

from bandits.ingest import load_corpus


def test_tau2_adapter_recovers_tool_response_by_tau2_id(tmp_path) -> None:
    path = tmp_path / "tau2.jsonl"
    path.write_text('''{"task_id": 7, "success": true, "reward": 1, "model": "demo", "messages": [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Where is my order?"}, {"role": "assistant", "content": null, "tool_calls": [{"id": "call-1", "name": "get_order", "arguments": {"order_id": "o-1"}}]}, {"role": "tool", "id": "call-1", "content": "{\\"order_id\\": \\"o-1\\", \\"status\\": \\"shipped\\"}"}]}\n''')

    corpus = load_corpus(path, "tau2")

    assert not corpus.issues
    trace = corpus.traces[0]
    assert trace.source == "tau2"
    assert trace.trace_id == "tau2:7"
    assert trace.outcome is True
    assert trace.instruction == "Where is my order?"
    assert trace.invocations[0].call_id == "call-1"
    assert trace.invocations[0].response == {"order_id": "o-1", "status": "shipped"}
