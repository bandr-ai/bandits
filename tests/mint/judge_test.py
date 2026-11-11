"""Judge tier: contract, verifier against a fake endpoint, statuses, infra errors.

No real model is ever called: the judge endpoint is a local fake HTTP server.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mint.models import Answer, BuildResult, ContractError
from mint.results import derive_status

from .validate_test import rows

KEY = "fake-key-not-real"

JUDGE_PY = """
from mint import Answer, BuildResult

SPEC = {"unit": "episode", "env": "static", "job": "main", "check": "judge", "gold": "imitation",
        "hidden": [], "applies_when": "any", "judge_reason": "JUDGE_REASON"}
VERSION = 1
KW = {}

def applies(t, findings):
    return t["meta"]["nums"] != "BLOCK" and bool(t["meta"]["answer"])

def group_key(t):
    return t["id"]

def build(group, out):
    t = group[0]
    kw = dict(
        rubric=["States a total for the numbers", "The total equals the sum of the numbers",
                "Does not use numbers absent from the evidence", "Is a single short statement"],
        answer=Answer(expected="the total is " + t["meta"]["answer"], check="judge"),
        wrong_answers=["WRONG the total is 1000"],
        alternatives=["it adds up to " + t["meta"]["answer"]],
        judge_model="fake-judge-1",
        evidence={"nums.txt": t["meta"]["nums"]},
    )
    kw.update(KW)
    return BuildResult(
        task_id="j_" + t["id"],
        instruction="Report the total of the numbers in /app/nums.txt in one sentence.",
        **kw,
    )
"""


class FakeJudge:
    """OpenAI-compatible fake. Verdicts by answer text: WRONG -> all contradicted,
    HALF -> half supported, FLAKY -> alternates per call, else all supported."""

    def __init__(self):
        self.requests, self.mode, self.calls = [], "ok", 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(
                    {"path": self.path, "auth": self.headers["Authorization"], **body}
                )
                outer.calls += 1
                if outer.mode == "500":
                    self.send_response(500)
                    self.end_headers()
                    return
                user = body["messages"][1]["content"]
                n = len(re.findall(r"^\d+\. ", user.split("CRITERIA:")[1], re.M))
                ans = re.search(r"<<<ANSWER\n(.*)\nANSWER>>>", user, re.S).group(1)
                if "WRONG" in ans:
                    v = ["contradicted"] * n
                elif "HALF" in ans:
                    v = ["supported"] * (n // 2) + ["insufficient"] * (n - n // 2)
                elif "FLAKY" in ans and outer.calls % 2:
                    v = ["supported"] * (n - 1) + ["insufficient"]
                else:
                    v = ["supported"] * n
                content = json.dumps(
                    {"criteria": [{"id": i + 1, "verdict": x} for i, x in enumerate(v)]}
                )
                if outer.mode == "garbage":
                    content = "I think it is fine."
                out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def judge(monkeypatch):
    fj = FakeJudge()
    monkeypatch.setenv("MINT_JUDGE_BASE_URL", fj.url)
    monkeypatch.setenv("MINT_JUDGE_API_KEY", KEY)
    monkeypatch.delenv("MINT_JUDGE_MODEL", raising=False)
    yield fj
    fj.close()


@pytest.fixture
def no_judge_env(monkeypatch):
    for v in ("MINT_JUDGE_BASE_URL", "MINT_JUDGE_API_KEY", "MINT_JUDGE_MODEL"):
        monkeypatch.delenv(v, raising=False)


def setup(env, py=JUDGE_PY, reason="free text answer; no exact value or test can check it"):
    env.build_pipeline()
    (env.ws / "tasktypes/addnums.py").unlink()
    env.write("tasktypes/judge.py", py.replace("JUDGE_REASON", reason))


def try_(env, answer=None, tid="j_q1"):
    args = ["task", "try", tid, "--json"] + (["--answer", answer] if answer is not None else [])
    return env.run(*args)


def test_judge_tasks_are_judged_never_validated(env, judge):
    setup(env)
    code, _, _, res = env.run("run", "generate", "--json")
    assert res.get("data", {}).get("by_status") == {"judged": 3}, res
    r = rows(env)["j_q1"]
    assert (
        r["check"] == "judge" and r["gates"]["structure"] == "pass" and r["gates"]["leak"] == "pass"
    )
    d = env.ws / "out/tasks/judge/j_q1"
    toml = (d / "task.toml").read_text()
    for frag in (
        "verifiable = false",
        'judge_model = "fake-judge-1"',
        "judge_temperature = 0",
        "judge_rubric_version =",
        "judge_reason = ",
        '"${MINT_JUDGE_API_KEY}"',
        '"${MINT_JUDGE_MODEL:-fake-judge-1}"',
        'network_mode = "public"',
    ):
        assert frag in toml, frag
    rub = json.loads((d / "tests/rubric.json").read_text())
    assert len(rub["criteria"]) == 4 and rub["model"] == "fake-judge-1" and rub["task"]
    assert (d / "tests/evidence/nums.txt").read_text() == "17\n25"
    # the key never lands in the package
    assert not any(KEY in p.read_text() for p in d.rglob("*") if p.is_file())
    # level 1 + default level 2 keep it judged; level 2 skips judged tasks (they cost calls)
    code, out, _, res = env.run("validate", "--sample-size", "3", "--json")
    assert code == 0 and judge.calls == 0, res
    assert {x["status"] for x in rows(env).values()} == {"judged"}
    assert "skipped" in env.run("validate", "--sample-size", "3")[1]


def test_derive_status_never_validates_a_judge_row():
    row = {
        "task_id": "x",
        "check": "judge",
        "gates": {
            "structure": "pass",
            "leak": "pass",
            "secrets": "pass",
            "exec": "pass",
            "runner": "harbor",
            "trials": ["t"],
        },
    }
    assert derive_status(row) == ("judged", "")
    row["gates"]["exec"] = "fail"
    assert derive_status(row)[0] == "exec_failed"


def test_verifier_scores_fraction_of_supported_criteria(env, judge):
    setup(env)
    env.run("run", "generate")
    assert try_(env, "the total is 42")[3]["data"]["reward"] == 1.0
    assert try_(env, "WRONG 1000")[3]["data"]["reward"] == 0.0
    assert try_(env, "HALF right")[3]["data"]["reward"] == 0.5
    n = judge.calls
    assert try_(env)[3]["data"]["reward"] == 0.0  # no answer
    assert try_(env, "   ")[3]["data"]["reward"] == 0.0  # blank answer
    assert judge.calls == n  # empty answers never reach the judge
    req = judge.requests[0]
    assert req["temperature"] == 0 and req["model"] == "fake-judge-1"  # pinned model by default
    assert req["path"] == "/v1/chat/completions" and req["auth"] == f"Bearer {KEY}"
    user = req["messages"][1]["content"]
    assert "17\n25" in user and "Report the total" in user and "1. States a total" in user
    assert "REFERENCE (may be wrong)" in user and "the total is 42" in user


def test_model_env_overrides_the_pin(env, judge, monkeypatch):
    setup(env)
    env.run("run", "generate")
    monkeypatch.setenv("MINT_JUDGE_MODEL", "other-model")
    try_(env, "the total is 42")
    assert judge.requests[-1]["model"] == "other-model"


def test_config_and_endpoint_problems_are_infra_errors_not_scores(env, judge, no_judge_env):
    setup(env)
    env.run("run", "generate")
    code, _, err, _ = try_(env, "the total is 42")  # nothing configured
    assert code == 1 and "infrastructure error" in err and "missing MINT_JUDGE_BASE_URL" in err
    assert env.run("task", "try", "j_q1")[0] == 0  # empty answer needs no judge
    import os

    os.environ["MINT_JUDGE_BASE_URL"], os.environ["MINT_JUDGE_API_KEY"] = judge.url, KEY
    try:
        judge.mode = "garbage"
        code, _, err, _ = try_(env, "the total is 42")
        assert code == 1 and "infrastructure error" in err
        judge.mode = "500"
        code, _, err, _ = try_(env, "the total is 42")
        assert code == 1 and "infrastructure error" in err
    finally:
        del os.environ["MINT_JUDGE_BASE_URL"], os.environ["MINT_JUDGE_API_KEY"]


def test_pin_comes_from_config_and_is_required(env, judge):
    setup(env, JUDGE_PY.replace('        judge_model="fake-judge-1",\n', ""))
    env.run("run", "generate")
    assert {r["status"] for r in rows(env).values()} == {"exec_failed"}  # nothing pinned
    assert "not pinned" in (env.ws / "out/results.jsonl").read_text()
    cfgp = env.ws / "mint.json"
    cfg = json.loads(cfgp.read_text())
    cfg["judge_model"] = "cfg-model"
    cfgp.write_text(json.dumps(cfg))
    env.run("run", "generate")
    assert {r["status"] for r in rows(env).values()} == {"judged"}
    assert 'judge_model = "cfg-model"' in (env.ws / "out/tasks/judge/j_q1/task.toml").read_text()


@pytest.mark.parametrize("reason", ["", "   "])
def test_judge_task_without_judge_reason_fails_structure(env, judge, reason):
    setup(env, reason=reason)
    env.run("run", "generate")
    assert {r["status"] for r in rows(env).values()} == {"exec_failed"}
    code, out, _, res = env.run("validate", "--json")
    assert code == 1 and res["data"]["level1"]["failed"] == 3
    assert "judge_reason" in json.dumps(res["data"]["level1"]["failures"])


def test_missing_judge_reason_key_fails_structure(env, judge):
    setup(env, JUDGE_PY.replace(', "judge_reason": "JUDGE_REASON"', ""))
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--json")
    assert code == 1 and "judge_reason" in json.dumps(res["data"]["level1"]["failures"])


@pytest.mark.parametrize(
    "kw, msg",
    [
        ("{'rubric': ['a', 'b']}", "3-6"),
        ("{'rubric': ['a'] * 7}", "3-6"),
        ("{'rubric': ['a', 'b', 'b']}", "distinct"),
        ("{'wrong_answers': []}", "wrong_answers"),
        ("{'answer': Answer(expected='', check='judge')}", "non-empty reference"),
        ("{'evidence': {'img.png': b'\\x00'}}", "text evidence"),
    ],
)
def test_judge_contract_errors(env, judge, kw, msg):
    setup(env, JUDGE_PY.replace("KW = {}", f"KW = {kw}"))
    env.run("run", "generate")
    assert msg in (env.ws / "out/queue.jsonl").read_text()


def test_materialize_requires_a_rubric(tmp_path):
    from mint.harbor import materialize

    b = BuildResult(
        task_id="x",
        instruction="Do it.",
        evidence={"a.txt": "a"},
        answer=Answer("ref", "judge"),
        wrong_answers=["w"],
    )
    with pytest.raises(ContractError, match="rubric"):
        materialize(b, tmp_path / "x", "t", "t@1", [])


def test_status_and_report_show_judged_separately_with_warning(env, judge):
    setup(env)
    env.run("run", "generate")
    out = env.run("status")[1]
    assert "judged (imitation)=3" in out and "model opinion, not ground truth" in out
    env.run("report")
    md = (env.ws / "report.md").read_text()
    assert "| type | tasks | validated | judged |" in md
    assert "| judge | 3 | 0 | 3 | 0 | 0 |" in md
    assert "model opinion, not ground truth" in md and "## Judge" in md
    assert "why no deterministic check: free text answer" in md
