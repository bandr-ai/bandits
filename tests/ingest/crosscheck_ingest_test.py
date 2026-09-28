from pathlib import Path

from scripts.crosscheck_ingest import bandits_spans, source_facts, source_losses

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "upstream" / "interlingua"


def test_captured_litellm_call_recovers_tool_call_from_alternate_recording() -> None:
    source = FIXTURES / "litellm.json"
    losses = source_losses(bandits_spans(source), source_facts(source))
    assert losses == []
    assert bandits_spans(source)["ba48e2429fb1cf09"]["output.tool_calls"][0][0] == "lookup_order"


def test_standard_genai_capture_has_no_source_loss() -> None:
    source = FIXTURES / "openllmetry.otlp.json"
    facts = source_facts(source)
    assert len(facts) == 1
    assert source_losses(bandits_spans(source), facts) == []
