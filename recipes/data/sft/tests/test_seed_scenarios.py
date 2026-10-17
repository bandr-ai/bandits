"""Seeds-only integration through real existing setup/production; fake LLMs, no API calls."""
import json

import pytest
from test_e2e_pipeline import E2EFake, make

from bandits_sft import seed_scenarios as ss
from bandits_sft.io import read_jsonl
from bandits_sft.llm import set_llm_factory


class SeedOnlyFake(E2EFake):
    def _answer(self, messages):
        system = messages[0]["content"]
        if system.startswith("Derive a coverage catalogue"):
            data = json.JSONDecoder().raw_decode(messages[-1]["content"])[0]  # a retry appends feedback text
            result = {"capabilities": [{"name": "scope", "capability": "handle an alternate evidence condition",
                       "basis": "seed", "seed_ids": [data["seeds"][0]["id"]]}]}
            if self.opts.get("catalogue_bad"):
                result["capabilities"][0]["seed_ids"] = ["invented_external_seed"]
        elif system.startswith("Design ONE candidate"):
            result = {k: "new warehouse transfer workflow" for k in
                      ("situation", "workflow", "decisive_evidence", "dependencies", "novelty_reason")}
            result["variation"] = {k: "seed" if k == "basis" else "changed condition" for k in
                                   ("name", "change", "preserve", "dependencies", "correct_when", "teaches", "basis")}
            result["episode_state"] = {"context": {}, "facts": ["transfer pending"]}
            if self.opts.get("card_bad"):
                result["episode_state"] = {"context": {}, "facts": []}
            if self.opts.get("basis_unquoted"):
                result["variation"]["basis"] = "zebra quantum marmalade policy nowhere stated"
        elif system.startswith("Write a new candidate training"):
            if self.opts.get("scenario_down"):
                raise RuntimeError("generator unavailable")
            data = json.loads(messages[-1]["content"])
            msgs = data["references"][0]["messages"]
            msgs[0]["content"] = "Warehouse inventory transfer inspection for item B."  # novel fake input
            msgs[-1]["content"] = '{"status":"pending"}'
            result = {"traces": [{"messages": msgs}]}
        elif system.startswith("Check whether these candidate"):
            result = {"supported": True, "novelty_ok": not self.opts.get("novelty_bad"), "dimensions_ok": True,
                      "pair_ok": True, "reason": "alternate evidence route"}
        else:
            return super()._answer(messages)
        return {"content": json.dumps(result), "tool_calls": []}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    opts = {}
    set_llm_factory(lambda cfg, role: SeedOnlyFake(role, opts))
    yield opts
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real API in tests")))


def args(tmp_path, **kwargs):
    config, out = make(tmp_path, **kwargs)
    return ["--seeds", str(tmp_path / "seeds.jsonl"), "--out", str(out), "--target", "2", "--budget", "2",
            "--config", config], out


def test_seeds_only_derives_setup_and_exports_both_lanes(tmp_path, fake):
    argv, out = args(tmp_path)
    assert ss.main(argv) == 0
    rows = read_jsonl(out / "final/synthetic.jsonl")
    assert len(rows) == 2 and {t["meta"]["lane"] for t in rows} == {"controlled", "scenario"}
    assert (out / "judge/loop/rubric_final.md").exists()
    assert (out / "analyzer/chosen.json").exists()
    assert (out / "scenarios/coverage.json").exists()
    assert (out / "final/HANDOFF.md").exists()
    manifest = json.loads((out / "final/manifest.json").read_text())
    assert manifest["synthetic_only"] and not manifest.get("failure_records")
    assert not ({t["id"] for t in rows} & {f"seed{i}" for i in range(4)})
    assert json.loads((out / "final/report.json").read_text())["cost_usd"] > 0


def test_resume_reuses_every_setup_and_generation_call(tmp_path, fake):
    argv, out = args(tmp_path)
    assert ss.main(argv) == 0
    log = (out / "llm_calls.jsonl").read_text()
    assert ss.main(argv) == 0
    assert (out / "llm_calls.jsonl").read_text() == log


def test_only_seed_data_required_no_protected_eval(tmp_path, fake):
    argv, out = args(tmp_path)
    assert ss.main(argv) == 0
    assert json.loads((out / "final/report.json").read_text())["external_eval_decontamination"] is False


def test_bad_scenario_generator_keeps_controlled_pool_and_resumes(tmp_path, fake):
    argv, out = args(tmp_path)
    fake["scenario_down"] = True
    assert ss.main(argv) == 1
    assert {t["meta"]["lane"] for t in read_jsonl(out / "final/synthetic.jsonl")} == {"controlled"}
    assert (out / "scenarios/jobs/scenarios/0000/card.json").exists()
    fake["scenario_down"] = False
    assert ss.main(argv) == 0
    assert len(read_jsonl(out / "final/synthetic.jsonl")) == 2


def test_invalid_derived_reference_is_dropped_not_used(tmp_path, fake):
    argv, out = args(tmp_path)
    fake["catalogue_bad"] = True
    assert ss.main(argv) == 0
    assert json.loads((out / "scenarios/coverage.json").read_text()) == []
    rejected = json.loads((out / "scenarios/coverage_rejected.json").read_text())
    assert rejected and "seed_ids" in rejected[0]["errors"][0]
    assert (out / "scenarios/coverage_raw_0.json").exists() and (out / "scenarios/coverage_raw_1.json").exists()
    jobs = read_jsonl(out / "scenarios/jobs.jsonl")
    assert all(j["capability"] is None for j in jobs)  # no gap job targets an unsupported capability


def test_budget_stop_still_writes_handoff(tmp_path, fake):
    argv, out = args(tmp_path)
    argv[argv.index("--budget") + 1] = "0.000000001"
    assert ss.main(argv) == 0
    report = json.loads((out / "final/report.json").read_text())
    assert report["status"] == "budget_stop" and report["cost_usd"] == 0
    assert (out / "final/HANDOFF.md").exists()


def test_supplied_task_rubric_is_not_required_or_accepted(tmp_path, fake):
    argv, _ = args(tmp_path, rubric_path="external.md")
    with pytest.raises(ValueError, match="derives its rubric"):
        ss.main(argv)


def test_changed_seeds_refused_on_resume(tmp_path, fake):
    argv, _out = args(tmp_path)
    assert ss.main(argv) == 0
    path = tmp_path / "seeds.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="inputs changed"):
        ss.main(argv)


def test_code_change_still_resumes(tmp_path, fake):
    argv, out = args(tmp_path)
    assert ss.main(argv) == 0
    path = out / "seed_only_inputs.json"
    saved = json.loads(path.read_text())
    saved["code_versions"][-1]["analyzer.py"] = "older"  # a fix landed between launches
    path.write_text(json.dumps(saved))
    assert ss.main(argv) == 0
    assert len(json.loads(path.read_text())["code_versions"]) == 2


def test_novelty_rejections_report_shortfall(tmp_path, fake):
    argv, out = args(tmp_path)
    fake["novelty_bad"] = True
    assert ss.main(argv) == 0
    report = json.loads((out / "final/report.json").read_text())
    assert report["status"] == "short_of_target" and report["by_lane"] == {"controlled": 1}


def test_dry_run_needs_no_key_and_makes_no_api_calls(tmp_path, monkeypatch):
    argv, out = args(tmp_path)
    monkeypatch.delenv("FAKE_KEY", raising=False)
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(AssertionError("dry run called LLM")))
    assert ss.main(argv + ["--dry-run"]) == 0
    report = json.loads((out / "dry_run.json").read_text())
    assert report["seeds"] == 4 and report["paid_calls"] == 0


def test_existing_cli_exposes_dataset(monkeypatch):
    from bandits_sft import cli
    seen = []
    monkeypatch.setattr(ss, "main", lambda argv: seen.extend(argv) or 0)
    with pytest.raises(SystemExit) as e:
        cli.main(["dataset", "--help"])
    assert e.value.code == 0 and seen == ["--help"]


def test_scenario_only_skips_controlled_lane(tmp_path, fake):
    argv, out = args(tmp_path)
    argv[argv.index("--target") + 1] = "1"
    assert ss.main([*argv, "--scenario-share", "1"]) == 0
    rows = read_jsonl(out / "final/synthetic.jsonl")
    assert rows and {r["meta"]["lane"] for r in rows} == {"scenario"}
    assert not (out / "analyzer/prod/jobs.jsonl").exists()


def test_bad_card_rejects_job_not_run(tmp_path, fake):
    argv, out = args(tmp_path)
    fake["card_bad"] = True
    assert ss.main(argv) == 0
    report = json.loads((out / "final/report.json").read_text())
    assert report["status"] == "short_of_target"
    assert all(k.startswith("bad card") for k in report["scenario"]["rejected_jobs"])
    assert report["scenario"]["jobs_finished"] == len(read_jsonl(out / "scenarios/jobs.jsonl"))
    assert list((out / "scenarios/jobs").rglob("card_raw.json"))  # paid answers kept for inspection


def test_stop_rate_measured_from_seeds():
    nudge = {"role": "user", "content": "wrap up now"}
    base = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    seeds = [{"messages": base + [nudge]}, {"messages": base + [nudge]}, {"messages": base}, {"messages": base}]
    assert ss.stop_rate(seeds) == 0.5
    assert ss.stop_rate([{"messages": base}]) == 0.0


def test_unquoted_basis_is_a_warning_not_a_reject(tmp_path, fake):
    argv, out = args(tmp_path)
    fake["basis_unquoted"] = True
    assert ss.main(argv) == 0
    cards = [json.loads(p.read_text()) for p in (out / "scenarios/jobs").rglob("card.json")]
    assert cards and all("basis_warning" in c for c in cards)


def test_stop_text_is_the_repeated_later_user_turn():
    nudge = {"role": "user", "content": "wrap up now"}
    base = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert ss.stop_text([{"messages": base + [nudge]}, {"messages": base + [nudge]}]) == "wrap up now"
    assert ss.stop_text([{"messages": base}]) == ""


def test_control_jobs_keep_seed_outcomes():
    def seed(i, label):
        ans = {"conclusion": label, "confidence": 0.9, "summary": "x" * 80}
        return {"id": f"s{i}", "messages": [{"role": "user", "content": "q"},
                                             {"role": "assistant", "content": json.dumps(ans)}]}
    seeds = [seed(i, "common") for i in range(9)] + [seed(9, "rare")]
    jobs = ss.scenario_jobs(seeds, [{"capability": "c", "seed_ids": ["s9"]}], 400, 0)
    control = [j for j in jobs if j["allocation"] == "broad"]
    assert all(set(j["outcome"]) == {"conclusion"} for j in control)  # labels only: no scores or prose
    common = sum(j["outcome"]["conclusion"] == "common" for j in control) / len(control)
    assert 0.8 < common < 1.0  # seed-proportional, not drifting to rare classes
    assert all(j["outcome"] is None for j in jobs if j["allocation"] == "coverage_gap")
    free = {"id": "f", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "free text"}]}
    assert ss.scenario_jobs([free], [], 2, 0)[0]["outcome"] is None
