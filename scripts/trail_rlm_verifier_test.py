from types import SimpleNamespace

from trail_rlm_verifier import ProposedSignal, _binary_labels, _load_labels, evaluate_family

from bandits.analyze.models import TaskFamily


def _trace(trace_id: str, reliable: bool) -> dict:
    return {
        "trace_id": trace_id,
        "task": "research the answer",
        "spans": [
            {
                "kind": "tool",
                "name": "search",
                "status": "ok" if reliable else "error",
                "arguments": {},
                "output": "evidence" if reliable else None,
            }
        ],
    }


def test_binary_labels_keep_only_outer_tertiles() -> None:
    labels, cuts = _binary_labels({str(i): float(i) for i in range(1, 7)})
    assert cuts == {"low": 3.0, "high": 4.0}
    assert labels == {"1": False, "2": False, "3": False, "4": True, "5": True, "6": True}


def test_loads_tau2_sealed_labels(tmp_path) -> None:
    path = tmp_path / "labels.json"
    path.write_text('{"a": {"success": true}, "b": {"success": false}}')
    labels, provenance = _load_labels(path, "tau2")
    assert labels == {"a": True, "b": False}
    assert provenance["format"] == "tau2 sealed success"


def test_family_discovery_selects_on_fit_and_reports_held_out() -> None:
    traces = {
        trace_id: _trace(trace_id, reliable)
        for trace_id, reliable in {
            "p1": True,
            "p2": True,
            "n1": False,
            "n2": False,
            "hp": True,
            "hn": False,
        }.items()
    }
    labels = {trace_id: trace["spans"][0]["status"] == "ok" for trace_id, trace in traces.items()}
    family = TaskFamily(
        family_id="family-test",
        descriptor="research tasks",
        trace_ids=tuple(traces),
        medoid_trace_id="p1",
        workload_mass=len(traces),
        fit_trace_ids=("p1", "p2", "n1", "n2"),
        held_out_trace_ids=("hp", "hn"),
        proposed_by="model",
    )

    def predict(**kwargs):
        assert "hp" not in kwargs["fit_examples"]
        assert "hn" not in kwargs["fit_examples"]
        return SimpleNamespace(
            signals=[
                ProposedSignal(
                    name="tool_completed",
                    hypothesis="A successful external lookup completed.",
                    code=(
                        "def signal(trace):\n"
                        "    tools = tool_spans(trace)\n"
                        "    if not tools:\n"
                        "        return None\n"
                        "    return 1.0 if all(s.get('status') == 'ok' for s in tools) else 0.0"
                    ),
                )
            ]
        )

    result = evaluate_family(family, traces, labels, predict, keep_auc=0.62)

    assert result["signals"][0]["status"] == "kept"
    assert result["signals"][0]["fit"]["auc"] == 1.0
    assert result["signals"][0]["held_out"]["auc"] == 1.0
    assert result["baselines"]["no_span_error"]["held_out"]["auc"] == 1.0


def test_rejected_signal_keeps_the_model_output_for_audit() -> None:
    traces = {
        trace_id: _trace(trace_id, reliable)
        for trace_id, reliable in {
            "p1": True,
            "p2": True,
            "n1": False,
            "n2": False,
            "hp": True,
            "hn": False,
        }.items()
    }
    labels = {trace_id: trace["spans"][0]["status"] == "ok" for trace_id, trace in traces.items()}
    family = TaskFamily(
        family_id="family-test",
        descriptor="research tasks",
        trace_ids=tuple(traces),
        medoid_trace_id="p1",
        workload_mass=len(traces),
        fit_trace_ids=("p1", "p2", "n1", "n2"),
        held_out_trace_ids=("hp", "hn"),
    )
    proposal = ProposedSignal(
        name="wrong_entrypoint",
        hypothesis="Malformed on purpose.",
        code="def other(trace):\n    return 1.0",
    )

    result = evaluate_family(
        family,
        traces,
        labels,
        lambda **_: SimpleNamespace(signals=[proposal]),
        keep_auc=0.62,
    )

    rejected = result["signals"][0]
    assert rejected["status"] == "rejected"
    assert rejected["code"] == proposal.code
    assert rejected["fit"] is None
    assert rejected["held_out"] is None
