"""End-to-end command (simia-plus pipeline): steps, resume, budget, circuit breaker, fallbacks; offline fakes only."""
from __future__ import annotations

import itertools
import json
import re

import pytest
from test_pipeline import ANALYZER_OUT, FakeLLM, seed

from simia_plus import pipeline as pl
from simia_plus.llm import log_call, set_llm_factory
from simia_plus.rubric_loop import SLOTS_V2
from simia_plus.schema import from_sharegpt

SLOTS = "\n".join(f"<{k}>\n{k} text\n</{k}>" for k in SLOTS_V2)
CLEAN = {"facts": [], "contradictions": [], "impossible_results": [], "unsupported_claims": []}
_n = itertools.count()


class E2EFake(FakeLLM):
    """FakeLLM plus the roles the pipeline adds; every call is logged with a fixed cost so the budget sees it."""

    def __init__(self, role, opts):
        super().__init__(role)
        self.opts = opts

    def chat(self, messages, tools=None, json_mode=False, tool_choice=None):
        out = self._answer(messages)
        prod = "prod/" in messages[-1]["content"] and "prod_cost" in self.opts
        cost = self.opts["prod_cost"] if prod else self.opts.get("cost", 0.001)
        log_call({"role": self.role, "ok": True, "cost": cost})
        return {**out, "cost": cost}  # like the real client: the budget settles each call to its actual cost

    def _answer(self, messages):
        text, sys_ = messages[-1]["content"], messages[0]["content"]
        o = self.opts
        if self.role == "rubric_induce":
            if o.get("induce_bad"):
                return {"content": "no slots here", "tool_calls": []}
            return {"content": SLOTS + ("\n<changelog>x</changelog>" if "revise" in sys_ else ""), "tool_calls": []}
        if self.role == "selftest":
            return {"content": json.dumps({"edits": [{"message": 0, "find": "please", "replace": "pls"}],
                                           "declared_change": "c", "defect": "d"}), "tool_calls": []}
        if self.role == "judge" and "internal consistency" in sys_:
            return {"content": json.dumps(CLEAN), "tool_calls": []}
        if self.role == "judge" and "data curator" in sys_:
            return {"content": json.dumps({"duplicate_groups": []}), "tool_calls": []}
        if self.role == "judge":  # rubric judge
            if o.get("judge_down") and "prod/" in text:
                raise RuntimeError("provider down")
            return {"content": json.dumps({"integrity": {"I1_consistent": "pass", "I2_possible": "pass", "I3_decisive_kept": "n/a", "I4_same_situation": "n/a"}, "answer_correct": True}), "tool_calls": []}
        if self.role == "analyzer" and text.startswith("<observed_contracts>"):  # per-seed proposals (old and wide)
            n = int(text.rsplit("Propose ", 1)[-1].split()[0]) if "Propose " in text else 3
            vs = [{"name": f"p{next(_n)}", "kind": ("task", "evidence")[k % 2], "requires_tools": [], "basis": "seed",
                   "decision_type": f"d{k}", "answer_transition": "changes", "teaches": "t", "change": "c", "preserve": "p",
                   "dependencies": "d", "correct_when": "w", "signals": ["lookup"]} for k in range(max(n, 3))]
            return {"content": json.dumps({"variations": vs}), "tool_calls": []}
        if self.role == "generator" and "Output format (fixed)" in text:
            m = re.search(r"((?:v\d+|prod)/[^\s\"]+)/\d+", text)
            gid = m.group(1) if m else "?"
            if o.get("gen_fail") and o["gen_fail"] in gid:
                raise RuntimeError("generator broke")
            out = json.loads(ANALYZER_OUT)
            out["messages"][0]["content"] = f"cancel #W2222 please ({gid})"  # distinct traces, so dedup keeps them
            out["messages"][-1]["content"] = json.dumps({"status": "pending", "n": next(_n)})
            return {"content": json.dumps(out), "tool_calls": []}
        return super().chat(messages)


@pytest.fixture
def fake(monkeypatch):
    opts: dict = {}
    monkeypatch.setenv("FAKE_KEY", "x")
    set_llm_factory(lambda cfg, role: E2EFake(role, opts))
    yield opts
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


def make(tmp_path, n_seeds=4, **extra):
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text("\n".join(json.dumps(seed(i)) for i in range(n_seeds)) + "\n")
    m = {"model": "fake", "api_key_env": "FAKE_KEY", "price_input_per_mtok": 1, "price_output_per_mtok": 1}
    cfg = {"seeds_path": str(seeds), "out_dir": str(tmp_path / "out"), "workers": 2, "pilot_jobs": 6,
           "selftest_seeds": 2, "case_block": False, "features": {"rubric_judge": True, "near_dedup": False},
           "breaker_window": 4, "models": {r: m for r in ("default", "judge", "generator", "analyzer", "diagnose",
                                                         "rubric_induce", "selftest")}, **extra}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p), tmp_path / "out"


def run(cfg, budget=5, target=20):
    return pl.main(["--config", cfg, "--budget", str(budget), "--target", str(target)])


def calls(out):
    return sum(1 for _ in (out / "llm_calls.jsonl").open())


def test_end_to_end_then_resume_does_nothing(tmp_path, fake):
    cfg, out = make(tmp_path)
    assert run(cfg) == 0
    rep = json.loads((out / "final" / "report.json").read_text())
    assert rep["status"] == "complete" and rep["final"] == 20 and rep["from_production"] > 0
    assert all(s["status"] == "done" for s in rep["steps"].values()) and set(rep["steps"]) == set(pl.STEPS)
    assert rep["rubric"].endswith("judge/loop/rubric_final.md") and (out / "judge" / "loop" / "report.json").exists()
    assert len(json.loads((out / "judge" / "loop" / "report.json").read_text())["history"]) == 2  # r0 + 1 REVISE round
    assert set(json.loads((out / "analyzer" / "chosen.json").read_text())["dev_usable_rates"]) == {"v1"}  # kept >= 50%
    jobs = [json.loads(x) for x in (out / "analyzer" / "prod" / "jobs.jsonl").read_text().splitlines()]
    arms = [j["arm"] for j in jobs]
    assert arms.count("old") >= max(arms.count("task"), arms.count("evidence"), arms.count("stack")) and len(set(arms)) >= 3
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert (out / "final" / "synthetic_sharegpt.jsonl").exists() and "Rejected, and why" in (out / "final" / "report.md").read_text()
    n = calls(out)
    assert run(cfg) == 0 and calls(out) == n  # resume: every step skipped, no new calls


def test_plan_is_sized_to_the_budget(tmp_path, fake):
    cfg, out = make(tmp_path)
    fake["cost"] = 0.01
    assert run(cfg, budget=2.0, target=500) == 0
    rep = json.loads((out / "final" / "report.json").read_text())
    assert any("planned to the budget" in f["what"] for f in rep["fallbacks"])
    assert rep["cost_usd"] <= 2.0 and rep["short_of_target"] > 0 and rep["final"] > 0


def test_budget_stops_waves_when_production_costs_more_than_the_pilot(tmp_path, fake):
    cfg, out = make(tmp_path)
    fake["cost"], fake["prod_cost"] = 0.01, 0.1  # the plan is sized on pilot cost; real jobs cost 10x
    assert run(cfg, budget=2.0, target=500) == 0  # a budget stop is a normal end
    rep = json.loads((out / "final" / "report.json").read_text())
    assert rep["status"].startswith("stopped: budget") and "not started" in rep["status"]
    assert rep["cost_usd"] <= 2.0 + 0.1 * 3 * 4  # at most about one wave (4 jobs x 3 calls) over
    assert rep["final"] > 0 and (out / "final" / "synthetic.jsonl").exists()


def test_circuit_breaker_on_generation_errors(tmp_path, fake):
    cfg, out = make(tmp_path)
    fake["gen_fail"] = "prod/"  # pilot works, every production job raises
    assert run(cfg, target=200) == 1
    rep = json.loads((out / "final" / "report.json").read_text())
    assert "circuit breaker" in rep["status"] and rep["steps"]["production"]["status"] == "stopped"
    assert rep["final"] > 0  # pilot traces still delivered


def test_circuit_breaker_counts_judge_errors(tmp_path, fake):
    cfg, out = make(tmp_path)
    fake["judge_down"] = True  # fail closed would silently reject everything; the breaker stops it
    assert run(cfg, target=200) == 1
    assert "judge call error" in json.loads((out / "final" / "report.json").read_text())["status"]
    gen_calls = sum('"generator"' in x for x in (out / "llm_calls.jsonl").open())
    fake["judge_down"] = False  # provider back: resume re-judges the failed traces without regenerating them
    assert run(cfg, target=200) == 0
    rep = json.loads((out / "final" / "report.json").read_text())
    assert rep["status"] == "complete" and "judge/check call error (fail closed)" not in rep["rejected_why"]
    first = [json.loads(x)["id"] for x in (out / "analyzer" / "prod" / "generated.jsonl").open()]
    assert len(first) == len(set(first)) and gen_calls > 0


def test_induction_retried_once_then_stops(tmp_path, fake):
    cfg, out = make(tmp_path)
    fake["induce_bad"] = True
    assert run(cfg) == 1
    rep = json.loads((out / "final" / "report.json").read_text())
    assert "judge_setup failed" in rep["status"] and any("retrying once" in f["what"] for f in rep["fallbacks"])
    assert not (out / "analyzer").exists()  # nothing spent past the failed step


def test_low_pilot_yield_stops_before_production(tmp_path, fake):
    cfg, out = make(tmp_path)
    fake["gen_fail"] = "v"  # every pilot job (ids v1/..., v2/...) raises
    assert run(cfg) == 1
    rep = json.loads((out / "final" / "report.json").read_text())
    assert "may not suit synthesis" in rep["status"] and not (out / "analyzer" / "prod" / "jobs.jsonl").exists()
    assert any("revision round" in f["what"] for f in rep["fallbacks"])  # round 1 < 50% -> round 2 ran


def test_preflight_catches_missing_key_and_prices(tmp_path, fake, monkeypatch):
    cfg, out = make(tmp_path)
    from pathlib import Path
    raw = json.loads(Path(cfg).read_text())
    raw["models"]["judge"].pop("price_output_per_mtok")
    Path(cfg).write_text(json.dumps(raw))
    monkeypatch.delenv("FAKE_KEY")
    assert run(cfg) == 1
    status = json.loads((out / "final" / "report.json").read_text())["status"]
    assert "FAKE_KEY is not set" in status and "role judge: price" in status and not (out / "llm_calls.jsonl").exists()


def test_mix_helpers():
    assert pl._split_counts(6, {"old": .5, "task": 1 / 6, "evidence": 1 / 6, "stack": 1 / 6}) == {"old": 3, "task": 1, "evidence": 1, "stack": 1}
    assert sum(pl._split_counts(7, {"old": .5, "task": .5}).values()) == 7
    vs = [{"_type": t, "n": i} for i, t in enumerate(["old"] * 4 + ["task"] * 2 + ["stack"] * 2)]
    order = [v["_type"] for v in pl._interleave(vs, {"old": .5, "task": .25, "stack": .25})]
    assert order[:4].count("old") == 2 and len(order) == 8


def test_job_failure_kinds():
    t = lambda **v: {"messages": [{"role": "user", "content": "x"}], "meta": {"verify": v}}
    assert pl._job_failure(None) == "execution error"
    assert pl._job_failure(t(rubric={"label": "error"})) == "judge call error"
    assert pl._job_failure(t(consistency={"label": "error"})) == "consistency call error"
    assert pl._job_failure(t(rubric={"label": "bad"})) is None  # a rejection is normal
    assert pl._job_failure({"messages": [], "meta": {"refusal": "no", "verify": {}}}) is None


def test_spend_reads_only_complete_rows(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"ok": True, "cost": 0.5, "item": "a"}) + "\n" + '{"ok": true, "co')
    s = pl.Spend(p)
    assert s.update() == 0.5
    with p.open("a") as f:
        f.write('st": 0.25}\n' + json.dumps({"ok": True, "cost": None}) + "\n")
    assert s.update() == 0.75 and s.unpriced == 1


def test_lesson_dedup_is_chunked(tmp_path, monkeypatch):
    from simia_plus import analyzer
    from simia_plus.config import load_config
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "models": {"default": {"model": "fake"}}}))
    cfg = load_config(cp)
    sent = []
    monkeypatch.setattr(analyzer, "LESSON_CHUNK", 4)
    monkeypatch.setattr(analyzer, "chat_json", lambda llm, s, u: sent.append(u.count("<conversation")) or {"duplicate_groups": []})
    monkeypatch.setattr(analyzer, "get_llm", lambda cfg, role: None)
    ver = [{"id": f"{s}{k}", "messages": [{"role": "assistant", "content": "{}"}],
            "meta": {"seed_id": s, "verify": {"kept": True}}} for s in "abcd" for k in range(2)]
    analyzer.mark_lesson_duplicates(cfg, ver)
    assert sent == [4, 4]


def test_input_json_flag_is_soft_and_reaches_both_judges(tmp_path):
    import copy

    from simia_plus.config import load_config
    from simia_plus.verify import input_json_flags, verify
    s = from_sharegpt(seed(0), 0)
    s["messages"][0]["content"] = 'Failure context:\n{"a": {"b": 1}, "c": 2}'
    ok = copy.deepcopy(s)
    ok["messages"][0]["content"] = 'Failure context:\n{"a": {"b": 2}, "c": 3}'
    broken = copy.deepcopy(s)
    broken["messages"][0]["content"] = 'Failure context:\n{"a": {"b": 1, "c": 2}'
    assert input_json_flags(ok, s) == [] and input_json_flags(s, None) == []
    assert "no longer parses" in input_json_flags(broken, s)[0]
    plain = copy.deepcopy(s)
    plain["messages"][0]["content"] = "no json here"
    assert "missing" in input_json_flags(plain, s)[0]
    nojson_seed = from_sharegpt(seed(0), 0)  # seed without valid JSON in the user turn: never flagged
    assert input_json_flags(broken, nojson_seed) == []
    cp = tmp_path / "c.json"
    rub = tmp_path / "r.md"
    rub.write_text("# rubric\n## Output\nx")
    cp.write_text(json.dumps({"seeds_path": "x", "out_dir": str(tmp_path), "rubric_path": str(rub),
                              "features": {"rubric_judge": True, "change_gate": False, "input_json_gate": False},
                              "models": {"default": {"model": "fake"}}}))
    cfg = load_config(cp)
    seen = []

    def factory(c, role):
        def chat(self, messages, tools=None, json_mode=False, tool_choice=None):
            seen.append(messages[-1]["content"])
            sys_ = messages[0]["content"]
            out = CLEAN if "internal consistency" in sys_ else {"integrity": {"I1_consistent": "pass", "I2_possible": "pass", "I3_decisive_kept": "n/a", "I4_same_situation": "n/a"}, "answer_correct": True}
            return {"content": json.dumps(out), "tool_calls": []}
        return type("F", (), {"chat": chat})()
    set_llm_factory(factory)
    t = verify(cfg, {**broken, "id": "b", "meta": {}}, set(), {"seed": s, "variation": {"name": "v", "change": "c"}})
    assert any("no longer parses" in f for f in t["meta"]["verify"]["soft_flags"])
    assert t["meta"]["verify"]["kept"]  # soft: the judges decide, code does not reject
    assert len(seen) == 2 and all("<code_notes" in u and "no longer parses" in u for u in seen)
    seen.clear()
    verify(cfg, {**ok, "id": "o", "meta": {}}, set(), {"seed": s, "variation": {"name": "v", "change": "c"}})
    assert seen and not any("<code_notes" in u for u in seen)
    cfg.features.input_json_gate = True  # default: code rejects before any judge call
    seen.clear()
    t = verify(cfg, {**broken, "id": "b2", "meta": {}}, set(), {"seed": s, "variation": {"name": "v", "change": "c"}})
    v = t["meta"]["verify"]
    assert not v["kept"] and v["rule_issues"][0].startswith("input JSON broken: message 0") and not seen
    assert not any("no longer parses" in f for f in v["soft_flags"])
    assert pl._why(t) == "code check: input JSON broken"
    set_llm_factory(lambda cfg, role: (_ for _ in ()).throw(RuntimeError("no real LLM in tests")))


def test_flag_kind_groups_specifics():
    assert pl._flag_kind("message 3: JSON block no longer parses (Expecting ',' at char 40); the seed's does") == "message N: JSON block no longer parses"
    assert pl._flag_kind("answer keys differ from seed: +['a'] -[]") == "answer keys differ from seed: +[..] -[..]"


def test_budget_smaller_than_setup_is_a_hard_cap(tmp_path, fake):
    """Review repro: $0.01 per call and a budget below setup cost used to spend $0.42 in judge setup."""
    cfg, out = make(tmp_path)
    fake["cost"] = 0.01
    assert run(cfg, budget=0.05) == 0  # a budget stop is a normal end
    rep = json.loads((out / "final" / "report.json").read_text())
    assert rep["status"].startswith("stopped: budget") and "judge_setup" in rep["status"]
    assert rep["cost_usd"] <= 0.05 and rep["steps"]["judge_setup"]["status"] == "stopped"
    assert not (out / "judge" / "selftest" / "dev.jsonl").exists()  # nothing partial saved as finished


def test_budget_reservations_are_concurrent_safe():
    import threading

    from simia_plus.llm import Budget, BudgetExhausted
    b, got, refused = Budget(1.0), [], []

    def take():
        try:
            got.append(b.reserve(0.3))
        except BudgetExhausted:
            refused.append(1)
    ts = [threading.Thread(target=take) for _ in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(got) == 3 and len(refused) == 7 and b.hit
    with pytest.raises(BudgetExhausted):  # stays exhausted: a later, cheaper call is refused too
        b.reserve(0.01)


def test_permanent_client_errors_are_not_retried(monkeypatch):
    from simia_plus import llm
    from simia_plus.config import ModelCfg
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def client(status):
        def create(**kw):
            calls.append(status)
            e = RuntimeError(f"http {status}")
            e.status_code = status
            raise e
        raw = type("R", (), {"create": staticmethod(create)})
        return type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"with_raw_response": raw})})})

    for status, attempts in ((400, 1), (401, 1), (429, 3), (503, 3)):
        calls.clear()
        c = llm.OpenAILLM(ModelCfg(model="m", base_url="http://localhost:1", api_key_env="NO_SUCH_KEY"), "r", retries=3)
        c.client = client(status)
        with pytest.raises(RuntimeError):
            c.chat([{"role": "user", "content": "x"}])
        assert len(calls) == attempts, (status, calls)


@pytest.mark.parametrize("breaker", [{"breaker_window": 100}, {"breaker_max_bad": 1.0}])
def test_failed_production_is_never_complete_and_resumes(tmp_path, fake, breaker):
    """Review repro: a failed batch smaller than the breaker window reported `complete` with 0 jobs verified and was
    skipped on resume."""
    cfg, out = make(tmp_path, **breaker)
    fake["gen_fail"] = "prod/"
    assert run(cfg, target=40) == 1
    rep = json.loads((out / "final" / "report.json").read_text())
    assert rep["status"] != "complete" and rep["steps"]["production"]["status"] == "stopped"
    assert ("incomplete" in rep["status"]) or ("circuit breaker" in rep["status"])
    fake.pop("gen_fail")  # provider recovers
    assert run(cfg, target=40) == 0
    rep = json.loads((out / "final" / "report.json").read_text())
    assert rep["status"] == "complete" and rep["from_production"] > 0


def _sdk_client(replies, sent):
    """An OpenAI SDK stand-in: each create() returns the next (content, completion_tokens) as a raw response."""
    from types import SimpleNamespace as NS

    def create(**kw):
        content, out_tokens = replies[len(sent)]
        sent.append(kw)
        msg = NS(content=content, tool_calls=None, model_extra={})
        resp = NS(choices=[NS(message=msg, finish_reason="stop")], model="m", model_extra={},
                  usage=NS(model_dump=lambda t=out_tokens: {"prompt_tokens": 0, "completion_tokens": t}))
        return NS(parse=lambda r=resp: r, headers={}, status_code=200)
    raw = NS(create=create)
    return NS(chat=NS(completions=NS(with_raw_response=raw)))


def test_every_billed_retry_is_reserved_before_it_is_sent(monkeypatch):
    """Review repro: one reservation covered all retries, so a billed empty reply plus a retry passed the cap."""
    from simia_plus import llm
    from simia_plus.config import ModelCfg
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    cfg = ModelCfg(model="m", base_url="http://localhost:1", api_key_env="NO_SUCH_KEY", max_tokens=10,
                   price_input_per_mtok=1, price_output_per_mtok=1)
    msgs = [{"role": "user", "content": "x"}]
    seen = []
    real = llm.max_call_cost
    monkeypatch.setattr(llm, "max_call_cost", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    probe = llm.OpenAILLM(cfg, "r", retries=1)
    probe.client = _sdk_client([("ok", 0)], [])
    llm.set_budget(llm.Budget(1.0))
    probe.chat(msgs)
    bound = seen[0]  # exactly what the client reserves per attempt for this request
    assert bound > 10 / 1e6
    try:
        for limit, expect_requests in ((0.000011, 0), (bound + 0.000001, 1), (2 * bound + 0.00002, 2)):
            sent = []
            c = llm.OpenAILLM(cfg, "r", retries=3)
            c.client = _sdk_client([("", 10), ("ok", 10)], sent)  # empty reply billed 10 tokens, then a real one
            b = llm.Budget(limit)
            llm.set_budget(b)
            try:
                c.chat(msgs)
            except llm.BudgetExhausted:
                pass
            assert len(sent) == expect_requests, (limit, len(sent))
            assert b.spent <= limit and b.reserved == 0  # never past the cap, nothing left held
            assert b.spent == pytest.approx(len(sent) * 10 / 1e6)  # each attempt settled to what it was billed
    finally:
        llm.set_budget(None)


def test_call_cost_bound_is_a_real_upper_bound():
    from simia_plus.config import ModelCfg
    from simia_plus.llm import max_call_cost, token_cost
    cfg = ModelCfg(model="m", max_tokens=100, price_input_per_mtok=1, price_output_per_mtok=1)
    text = "日本語のテキスト" * 200  # 3 bytes per character: more tokens than characters/3 could allow for
    msgs = [{"role": "user", "content": text}]
    assert max_call_cost(cfg, msgs) >= token_cost(cfg, {"prompt_tokens": len(text.encode()), "completion_tokens": 100})
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * 5000, "parameters": {}}}]
    assert max_call_cost(cfg, msgs, tools=tools) - max_call_cost(cfg, msgs) >= 5000 / 1e6  # tool schemas count
    over = ModelCfg(model="m", max_tokens=100, price_input_per_mtok=1, price_output_per_mtok=1,
                    extra={"max_completion_tokens": 50000})
    assert max_call_cost(over, msgs) - max_call_cost(cfg, msgs) >= (50000 - 100) / 1e6  # the larger output limit wins
