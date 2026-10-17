"""Proposal calls are cached per seed and prompt, so a stopped run never pays twice."""
import json

from test_e2e_pipeline import make

from bandits_sft.llm import set_llm_factory


def test_proposals_cached_per_seed(tmp_path):
    from bandits_sft.analyzer import propose_for_seed, system_labels
    from bandits_sft.config import load_config
    config, _ = make(tmp_path)
    cfg = load_config(config)
    seed = {"id": "s/1", "system": "sys", "tools": [], "messages": [
        {"role": "user", "content": "check item"}, {"role": "assistant", "content": "{}"}]}
    calls = []

    class Counting:
        def chat(self, messages, **kw):
            calls.append(1)
            return {"content": json.dumps({"variations": []}), "tool_calls": [], "cost": 0}
    set_llm_factory(lambda cfg, role: Counting())
    try:
        labels = system_labels([seed])
        for _ in range(2):
            propose_for_seed(cfg, seed, "all", labels, [seed], "contracts", [], [], 3, tmp_path / "cache")
    finally:
        set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real API in tests")))
    assert len(calls) == 1 and len(list((tmp_path / "cache").iterdir())) == 1


