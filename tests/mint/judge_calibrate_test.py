"""mint judge-calibrate: agreement of the judge with known verdicts, recorded per type."""

import json
import re

import pytest

from .judge_test import setup


@pytest.fixture(autouse=True)
def _model(monkeypatch, judge):
    monkeypatch.setenv("MINT_JUDGE_MODEL", "calib-model")


def exact_sum_rule(ans, user, n):
    """A perfect judge for the synthetic sum tasks: supported iff the answer is the sum."""
    ev = user.split("--- nums.txt ---\n")[1].split("\n\n")[0]
    total = sum(int(x) for x in re.findall(r"\d+", ev))
    try:
        ok = float(ans) == total
    except ValueError:
        ok = False
    return ["supported" if ok else "contradicted"] * n


def decisions(env):
    p = env.ws / "decisions.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def calib(env, *args):
    return env.run("judge-calibrate", "--json", *args)


def test_calibrate_on_a_deterministic_type_records_agreement(env, judge):
    env.build_pipeline()
    env.run("run", "generate")
    judge.rule = exact_sum_rule
    code, _, _, res = calib(env, "--type", "addnums", "--n", "3")
    assert code == 0, res
    e = res["data"]["types"]["addnums"]
    # per task: reference 42-style, its alternative "<n>.0" and the wrong answer: 3 answers each
    assert e["total"] == 9 and e["agreement"] == 1.0 and e["false_accept"] == 0
    assert judge.calls == 9
    user = judge.requests[0]["messages"][1]["content"]
    assert "REFERENCE" not in user  # the deterministic answer is never given to the judge
    assert "1. The answer directly" in user and "Add up the numbers" in user
    (d,) = decisions(env)
    assert d["kind"] == "judge_calibration" and d["type"] == "addnums" and d["agreement"] == 1.0
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "calibration on addnums: judge agrees with the deterministic result 100% (9/9)" in md
    assert calib(env, "--type", "addnums", "--n", "3")[3]["ok"]
    assert len(decisions(env)) == 1  # identical result is not recorded twice


def test_disagreement_is_counted_by_kind(env, judge):
    env.build_pipeline()
    env.run("run", "generate")
    judge.mode = "no"  # rejects everything: every correct answer is a false reject
    _, _, _, res = calib(env, "--type", "addnums", "--n", "3")
    e = res["data"]["types"]["addnums"]
    assert e["agreed"] == 3 and e["false_reject"] == 6 and e["false_accept"] == 0
    assert round(e["agreement"], 2) == 0.33


def test_dry_run_records_nothing(env, judge):
    env.build_pipeline()
    env.run("run", "generate")
    judge.rule = exact_sum_rule
    assert calib(env, "--type", "addnums", "--n", "1", "--dry-run")[0] == 0
    assert decisions(env) == []


def write_answers(env, lines):
    p = env.tmp / "answers.jsonl"
    p.write_text("\n".join(json.dumps(x) if isinstance(x, dict) else x for x in lines) + "\n")
    return str(p)


def test_external_answers_on_a_deterministic_task(env, judge):
    env.build_pipeline()
    env.run("run", "generate")
    judge.rule = exact_sum_rule
    f = write_answers(
        env,
        [
            {"task_id": "t_q1", "answer": "42", "true_reward": 1},
            {"task_id": "t_q1", "answer": "41", "true_reward": 0},
            {"task_id": "t_q2", "answer": "7", "true_reward": 1.0},
        ],
    )
    _, _, _, res = calib(env, "--answers", f)
    assert res["data"]["types"]["addnums"]["total"] == 3
    assert res["data"]["types"]["addnums"]["agreement"] == 1.0


def test_external_answers_on_a_judge_task_use_its_rubric(env, judge):
    setup(env)
    env.run("run", "generate")
    f = write_answers(
        env,
        [
            {"task_id": "j_q1", "answer": "the total is 42", "true_reward": 1},
            {"task_id": "j_q1", "answer": "WRONG 1000", "true_reward": 0},
        ],
    )
    _, _, _, res = calib(env, "--answers", f, "--type", "judge")
    assert res["data"]["types"]["judge"]["agreement"] == 1.0
    assert "REFERENCE (may be wrong)" in judge.requests[0]["messages"][1]["content"]
    assert "1. States a total" in judge.requests[0]["messages"][1]["content"]


def test_needs_endpoint_and_model(env, judge, no_judge_env, monkeypatch):
    env.build_pipeline()
    env.run("run", "generate")
    assert "MINT_JUDGE_BASE_URL" in env.run("judge-calibrate", "--type", "addnums")[2]
    monkeypatch.setenv("MINT_JUDGE_BASE_URL", judge.url)
    monkeypatch.setenv("MINT_JUDGE_API_KEY", "k")
    assert "needs a judge model" in env.run("judge-calibrate", "--type", "addnums")[2]
    assert judge.calls == 0


def test_bad_inputs_with_config(env, judge):
    setup(env)
    env.run("run", "generate")
    code, _, err, _ = calib(env, "--type", "judge")
    assert code == 1 and "no deterministic verifier" in err
    assert calib(env)[0] == 2  # neither --type nor --answers
    bad = write_answers(env, [{"task_id": "nope", "answer": "x", "true_reward": 1}])
    assert "unknown task id" in calib(env, "--answers", bad)[2]
    bad = write_answers(env, ["{not json"])
    assert calib(env, "--answers", bad)[0] == 2
    judge.mode = "500"
    f = write_answers(env, [{"task_id": "j_q1", "answer": "x", "true_reward": 1}])
    code, _, err, _ = calib(env, "--answers", f)
    assert code == 1 and "every judge call failed" in err and decisions(env) == []
