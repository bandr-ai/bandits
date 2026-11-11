"""Gates on a tiny fixture task, and the status rules."""

import stat

import pytest

from l2h.gates import LocalRunner, Runner, RunOutcome, run_exec_gates, verdict
from l2h.harbor import materialize, structure_check
from l2h.models import Answer, BuildResult, ContractError
from l2h.results import check_invariants, derive_status, finalize, new_row


def make_task(tmp, **kw):
    b = BuildResult(
        task_id="t1",
        instruction="Add the numbers in /app/n.txt.",
        evidence={"n.txt": "2\n3"},
        answer=Answer(expected="5", check="numeric"),
        wrong_answers=["6"],
        alternatives=["5.0", " 5 "],
        **kw,
    )
    d = tmp / "t1"
    materialize(b, d, "add", "add@1", ["a.jsonl#0"])
    return d


def test_fixture_task_is_structurally_valid(tmp_path):
    assert structure_check(make_task(tmp_path)) == []


def test_structure_catches_breakage(tmp_path):
    d = make_task(tmp_path)
    (d / "solution/solve.sh").unlink()
    (d / "tests/test.sh").write_text("#!/bin/bash\necho hi\n")
    (d / "environment/Dockerfile").write_text("FROM ubuntu\nCOPY missing/ /app/\n")
    probs = " | ".join(structure_check(d))
    assert "solution/solve.sh" in probs and "reward" in probs and "missing/" in probs


def test_all_gates_pass_on_good_task(tmp_path):
    d = make_task(tmp_path)
    res = run_exec_gates(LocalRunner(), d, "answer.txt", ["6"], ["5.0", " 5 "], False)
    assert res["exec"] == "pass", res
    assert {k: v["gate"] for k, v in res["detail"].items()} == {
        "oracle": "pass",
        "noop": "pass",
        "wrong": "pass",
        "alternatives": "pass",
    }
    assert res["runner"] == "local"


def test_noop_that_passes_is_a_failure(tmp_path):
    d = make_task(tmp_path)
    (d / "tests/test.sh").write_text(
        '#!/bin/bash\nmkdir -p "$L2H_REWARD_DIR"\necho 1 > "$L2H_REWARD_DIR/reward.txt"\n'
    )  # always passes
    res = run_exec_gates(LocalRunner(), d, "answer.txt", ["6"], [], False)
    assert res["exec"] == "fail"
    assert res["detail"]["noop"]["gate"] == "fail" and res["detail"]["wrong"]["gate"] == "fail"
    assert res["detail"]["oracle"]["gate"] == "pass"


def test_wrong_oracle_fails_the_oracle_gate(tmp_path):
    d = make_task(tmp_path)
    (d / "solution/solve.sh").write_text(
        '#!/bin/bash\nmkdir -p "$L2H_ANSWER_DIR"\necho 99 > "$L2H_ANSWER_DIR/answer.txt"\n'
    )
    res = run_exec_gates(LocalRunner(), d, "answer.txt", ["6"], [], False)
    assert res["exec"] == "fail" and res["detail"]["oracle"]["gate"] == "fail"


def test_too_strict_verifier_fails_alternative_gate(tmp_path):
    d = make_task(tmp_path)
    res = run_exec_gates(LocalRunner(), d, "answer.txt", ["6"], ["five"], False)
    assert res["exec"] == "fail" and res["detail"]["alternatives"]["gate"] == "fail"


def test_crashing_verifier_is_an_error_not_a_failure(tmp_path):
    d = make_task(tmp_path)
    (d / "tests/test.sh").write_text("#!/bin/bash\nexit 3\n")  # never writes a reward
    res = run_exec_gates(LocalRunner(), d, "answer.txt", ["6"], [], False)
    assert res["exec"] == "error"
    assert res["detail"]["oracle"]["state"] == "infra_error"
    assert "without writing" in res["detail"]["oracle"]["detail"]


def test_missing_expected_json_is_an_infra_error(tmp_path):
    d = make_task(tmp_path)
    (d / "tests/expected.json").unlink()
    assert run_exec_gates(LocalRunner(), d, "answer.txt", ["6"], [], False)["exec"] == "error"


def test_timeout_and_spawn_failure_are_errors(tmp_path):
    d = make_task(tmp_path)
    (d / "tests/test.sh").write_text("#!/bin/bash\nsleep 5\n")
    assert LocalRunner(timeout=0.3).run_verifier(d, {}).state == "infra_error"
    assert LocalRunner(bash="/nonexistent/bash").run_verifier(d, {}).state == "infra_error"
    assert verdict(RunOutcome("infra_error"), True) == "error"


def test_injected_runner_errors_are_recorded(tmp_path):
    class Broken(Runner):
        name = "broken"

        def run_solution(self, task_dir):
            return RunOutcome("infra_error", detail="docker daemon not running")

        def run_verifier(self, task_dir, answer):
            return RunOutcome("infra_error", detail="docker daemon not running")

    res = run_exec_gates(Broken(), make_task(tmp_path), "answer.txt", ["6"], [], False)
    assert res["exec"] == "error" and res["runner"] == "broken"


# ------------------------------------------------------------------ status


def row(structure="pass", leak="pass", exec_="not_run", runner=None):
    r = new_row("t", "ty", ["f#0"], "ty@1", "imitation")
    r["gates"].update(structure=structure, leak=leak, exec=exec_)
    if runner:
        r["gates"]["runner"] = runner
    return r


def test_never_validated_without_exec_gates():
    assert finalize(row())["status"] == "generated"
    assert finalize(row(exec_="error", runner="local"))["status"] == "generated"
    assert finalize(row(exec_="pass", runner="harbor"))["status"] == "validated"
    assert finalize(row(exec_="pass", runner="local"))["status"] == "generated"
    assert finalize(row(exec_="fail", runner="local"))["status"] == "blocked"
    assert finalize(row(leak="fail"))["status"] == "blocked"
    assert finalize(row(structure="fail", exec_="pass", runner="local"))["status"] == "blocked"


def test_invariant_rejects_forged_validated():
    forged = row()
    forged["status"] = "validated"
    with pytest.raises(ContractError, match="validated"):
        check_invariants([forged])
    forged2 = row(exec_="pass")  # exec says pass but no runner recorded
    forged2["status"] = "validated"
    with pytest.raises(ContractError):
        check_invariants([forged2])


def test_blocked_reason_wins():
    r = row(exec_="pass", runner="local")
    r["blocked_reason"] = "batch audit failed"
    assert derive_status(r) == ("blocked", "batch audit failed")


def test_materialize_rejects_bad_input(tmp_path):
    with pytest.raises(ContractError):
        materialize(BuildResult(task_id="../x", answer=Answer("1")), tmp_path / "x", "t", "t@1", [])
    with pytest.raises(ContractError):
        materialize(BuildResult(task_id="ok"), tmp_path / "y", "t", "t@1", [])
    with pytest.raises(ContractError):
        Answer(expected=["a"], check="exact").validate()
    with pytest.raises(ContractError):
        materialize(
            BuildResult(task_id="ok", answer=Answer("1"), files={"../evil": "x"}),
            tmp_path / "z",
            "t",
            "t@1",
            [],
        )


def test_grade_ladder_normalizers(tmp_path):
    import importlib.util
    from importlib import resources

    spec = importlib.util.spec_from_file_location(
        "grade", resources.files("l2h") / "templates/grade.py"
    )
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    assert g.matches(
        {"check": "exact", "expected": "Yes", "normalize": ["strip", "lower"]}, "  yes\n"
    )
    assert not g.matches({"check": "exact", "expected": "Yes"}, "yes")
    assert g.matches(
        {
            "check": "numeric",
            "expected": "1284.5",
            "normalize": ["strip", "thousands"],
            "tolerance": 0.01,
        },
        "1,284.50",
    )
    assert g.matches(
        {"check": "numeric", "expected": "12", "normalize": ["percent"], "rel_tolerance": 0.1},
        "12.5%",
    )
    assert not g.matches({"check": "numeric", "expected": "12"}, "abc")
    assert g.matches({"check": "exact", "expected": "a,b", "normalize": ["sorted_list"]}, "b\na")
    assert g.matches({"check": "allowed_set", "expected": ["x", "y"]}, "y")
    assert stat.S_IMODE(0o755) & 0o111
