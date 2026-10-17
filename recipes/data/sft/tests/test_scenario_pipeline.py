"""Scenario recipe through the same loop as the platform worker (preflight, steps, run_step, report); fake LLMs only."""
import json

import pytest
from test_e2e_pipeline import E2EFake, make

from bandits_sft import cli
from bandits_sft.config import load_config
from bandits_sft.io import read_jsonl
from bandits_sft.llm import Budget, configure_call_log, set_budget, set_llm_factory
from bandits_sft.pipeline import Stop
from bandits_sft.scenario_pipeline import SCENARIO_STEPS, ScenarioPipeline, coverage_seeds, stop_rate, stop_text


class ScenarioFake(E2EFake):
    def _answer(self, messages):
        system, o = messages[0]["content"], self.opts
        if system.startswith("Derive a coverage catalogue"):
            data = json.JSONDecoder().raw_decode(messages[-1]["content"])[0]  # a retry appends feedback text
            result = {"capabilities": [{"name": "scope", "capability": "handle an alternate evidence condition",
                                        "basis": "seed", "seed_ids": [data["seeds"][0]["id"]]}]}
        elif system.startswith("Design ONE candidate"):
            result = {k: "new warehouse transfer workflow" for k in
                      ("situation", "workflow", "decisive_evidence", "dependencies", "novelty_reason")}
            result["variation"] = {k: "seed" if k == "basis" else "changed condition" for k in
                                   ("name", "change", "preserve", "dependencies", "correct_when", "teaches", "basis")}
            result["episode_state"] = {"context": {}, "facts": ["transfer pending"]}
        elif system.startswith("Write a new candidate training"):
            if o.get("generator_down"):
                raise RuntimeError("generator unavailable")
            data = json.loads(messages[-1]["content"])
            msgs = data["references"][0]["messages"]
            n = o["n"] = o.get("n", 0) + 1
            msgs[0]["content"] = f"Warehouse inventory transfer inspection for item {n}-{'xyz' * n}."
            msgs[-1]["content"] = '{"status":"pending"}'
            result = {"traces": [{"messages": msgs}]}
        elif system.startswith("Check whether these candidate"):
            result = {"supported": True, "novelty_ok": True, "dimensions_ok": True, "pair_ok": True, "reason": "ok"}
        else:
            text = messages[-1]["content"]
            if o.get("judge_down") and self.role == "judge" and "Warehouse inventory transfer" in text and "internal consistency" not in system:
                raise RuntimeError("judge unavailable")
            return super()._answer(messages)
        return {"content": json.dumps(result), "tool_calls": []}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    opts = {}
    set_llm_factory(lambda cfg, role: ScenarioFake(role, opts))
    yield opts
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real API in tests")))
    set_budget(None)
    configure_call_log(None)


def drive(tmp_path, budget=5, target=4, on_progress=None, progress_every=1, **extra):
    """The platform worker's loop."""
    config, out = make(tmp_path, per_seed_variations=False, prompt_source="analyzer", **extra)
    cfg = load_config(config)
    configure_call_log(out / "llm_calls.jsonl")
    p = ScenarioPipeline(cfg, budget, target, on_progress, progress_every)
    set_budget(Budget(budget, p.spend.total))
    stop = None
    try:
        p.preflight()
        for step in SCENARIO_STEPS:
            p.run_step(step, getattr(p, step))
            if step == "judge_setup":
                p.state["rubric_path"] = cfg.rubric_path
                p.save()
    except Stop as e:
        stop = e
    finally:
        set_budget(None)
        configure_call_log(None)
    return p, p.report(stop), stop, out


def n_calls(out):
    return len(read_jsonl(out / "llm_calls.jsonl"))


def test_runs_like_the_worker_and_writes_all_outputs(tmp_path, fake):
    _p, rep, stop, out = drive(tmp_path)
    assert stop is None and rep["final"] >= 1 and rep["status"] in ("complete", "short_of_target")
    assert set(rep["steps"]) == set(SCENARIO_STEPS) and all(s["status"] == "done" for s in rep["steps"].values())
    for name in ("synthetic.jsonl", "synthetic_sharegpt.jsonl", "seeds_plus_synthetic_sharegpt.jsonl", "report.json", "report.md"):
        assert (out / "final" / name).exists()
    assert len(read_jsonl(out / "final/synthetic.jsonl")) == rep["final"]
    assert not {t["id"] for t in read_jsonl(out / "final/synthetic.jsonl")} & {f"s{i}" for i in range(4)}
    assert rep["cost_usd"] > 0 and rep["by_role"] and rep["scenario"]["jobs"] == 8
    domain = json.loads((out / "setup/domain.json").read_text())
    assert domain["no_op"] is True and domain["label_fields"] == [] and rep["setup"]["no_op"] is True  # free-text finals
    assert "no-op setup" in (out / "final/report.md").read_text()
    assert (out / "analyzer/chosen.json").exists() and not (out / "analyzer/prod").exists()  # no controlled steps ran


def test_resume_pays_nothing(tmp_path, fake):
    _, _, _, out = drive(tmp_path)
    n = n_calls(out)
    _, rep, stop, _ = drive(tmp_path)
    assert stop is None and n_calls(out) == n and rep["final"] >= 1


def test_kill_mid_scenarios_then_resume_pays_nothing_twice(tmp_path, fake):
    ref = tmp_path / "ref"
    ref.mkdir()
    _, _, _, ref_out = drive(ref, target=6)

    class Killed(Exception):
        pass

    def kill(done):
        raise Killed

    work = tmp_path / "run"
    work.mkdir()
    _, _, stop, out = drive(work, target=6, on_progress=kill)
    assert stop and "scenarios failed" in str(stop) and 0 < len(read_jsonl(out / "scenarios/results.jsonl"))
    _, rep, stop, _ = drive(work, target=6)
    assert stop is None and rep["final"] >= 1
    assert n_calls(out) == n_calls(ref_out)  # a card, generation or judge verdict is never bought twice


def test_judge_outage_is_not_a_reject_and_resumes(tmp_path, fake):
    fake["judge_down"] = True
    _p, rep, stop, out = drive(tmp_path)
    assert stop and not stop.ok and ("circuit breaker" in str(stop) or "incomplete" in str(stop))
    assert rep["final"] == 0 and "quality_gate" not in rep["scenario"]["rejected_by_category"]
    fake["judge_down"] = False
    _, rep, stop, out = drive(tmp_path)
    assert stop is None and rep["final"] >= 1
    assert fake["n"] == len(list((out / "scenarios/jobs").rglob("generated_raw.json")))  # each generation bought once


def test_broken_generator_trips_the_breaker_not_the_run_loop(tmp_path, fake):
    fake["generator_down"] = True
    _p, rep, stop, _out = drive(tmp_path)
    assert stop and "circuit breaker" in str(stop) and not stop.ok
    assert rep["scenario"]["rejected_by_category"].get("error", 0) >= 1 and rep["final"] == 0


def test_budget_stop_is_ok_and_reported(tmp_path, fake):
    _p, rep, stop, out = drive(tmp_path, budget=0.02)
    assert stop and stop.ok and rep["status"].startswith("stopped")
    assert (out / "final/report.json").exists()


def test_preflight_rejects_supplied_rubric_and_controlled_settings(tmp_path, fake):
    with pytest.raises(Stop, match="rubric_path"):
        ScenarioPipeline(load_config(make(tmp_path, per_seed_variations=False, prompt_source="analyzer",
                                          rubric_path="r.md")[0]), 5, 4).preflight()
    config, _ = make(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, prompt_source="analyzer")
    with pytest.raises(Stop, match="per_seed_variations"):
        ScenarioPipeline(load_config(config), 5, 4).preflight()


def test_larger_target_on_resume_appends_jobs(tmp_path, fake):
    _, _, _, out = drive(tmp_path, target=2)
    first = read_jsonl(out / "scenarios/jobs.jsonl")
    p2, _, _, _ = drive(tmp_path, target=6)
    p3 = ScenarioPipeline(p2.cfg, 5, 6)
    jobs = p3._planned_jobs(p3._seeds(), json.loads((out / "scenarios/coverage.json").read_text()),
                            json.loads((out / "setup/domain.json").read_text()))
    assert jobs[:len(first)] == first and len(jobs) == 12


def seed_with(messages):
    return {"id": "x", "system": "s", "tools": [], "messages": messages}


def test_stop_text_needs_a_shared_wrap_up_before_the_final_answer():
    def chat(i):  # a chat that always ends with a thanks turn before the answer
        return seed_with([{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"},
                          {"role": "user", "content": "wrap up now"}, {"role": "assistant", "content": "done"}])
    seeds = [chat(i) for i in range(10)]
    assert stop_text(seeds) == "wrap up now" and stop_rate(seeds) == 1.0
    mixed = [chat(0), chat(1)] + [seed_with([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
                                  for _ in range(10)]
    assert stop_text(mixed) == "" and stop_rate(mixed) == 0.0  # 2 of 12 is below max(3, 20%)
    early = [seed_with([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                        {"role": "user", "content": "thanks"}, {"role": "assistant", "content": "b"},
                        {"role": "user", "content": f"more {i}"}, {"role": "assistant", "content": "c"}]) for i in range(5)]
    assert stop_text(early) == ""  # "thanks" is not the turn right before the final answer


def test_coverage_input_is_bounded_and_spreads_over_agents():
    def agent(name, i):
        return {"id": f"{name}{i}", "system": name, "tools": [], "messages": [
            {"role": "user", "content": "u" * 4000}, {"role": "assistant", "content": "a" * 4000}]}
    seeds = [agent("A", i) for i in range(20)] + [agent("B", i) for i in range(3)]
    shown = coverage_seeds(seeds, 40_000)
    assert len(json.dumps(shown)) < 60_000 and {s["system"] for s in shown} == {"A", "B"}
    assert coverage_seeds(seeds[:2], 150_000) == seeds[:2]  # small sets pass whole


def test_cli_dispatches_dataset(monkeypatch):
    from bandits_sft import scenario_pipeline as sp
    seen = []
    monkeypatch.setattr(sp, "main", lambda argv: seen.extend(argv) or 0)
    with pytest.raises(SystemExit) as e:
        cli.main(["dataset", "--help"])
    assert e.value.code == 0 and seen == ["--help"]
