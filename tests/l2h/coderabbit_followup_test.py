"""Remaining first-round review findings: matching sample size, answer file names,
exec evidence tied to wrong answers/alternatives, and deleted task packages."""

import shutil

import pytest

from l2h import Answer
from l2h.models import ContractError

from .conftest import TASKTYPE_PY
from .validate_test import rows


def test_matching_sample_honours_n(env):
    env.build_pipeline()
    for n in (1, 3):
        items = env.run("sample", "--matching", "has_answer", "-n", str(n), "--json")[3]["data"][
            "items"
        ]
        assert len(items) == n


def test_empty_answer_file_is_rejected():
    with pytest.raises(ContractError):
        Answer(expected="x", file="").validate()


def test_answer_file_with_spaces_is_quoted_in_the_oracle(env):
    env.build_pipeline()
    env.write(
        "tasktypes/addnums.py",
        TASKTYPE_PY.replace(
            'check="numeric", tolerance=0.01)',
            'check="numeric", tolerance=0.01, file="my answer.txt")',
        ),
    )
    env.run("run", "generate")
    code, _, _, res = env.run("validate", "--sample-size", "3", "--json")
    assert code == 0, res
    assert rows(env)["t_q1"]["status"] == "validated"


def test_changed_wrong_answers_invalidate_exec_evidence(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    assert rows(env)["t_q1"]["status"] == "validated"
    env.write(
        "tasktypes/addnums.py",
        TASKTYPE_PY.replace(
            'wrong_answers=[str(float(t["meta"]["answer"]) + 1)]',
            'wrong_answers=[str(float(t["meta"]["answer"]) + 2)]',
        ),
    )
    env.run("run", "generate")
    assert rows(env)["t_q1"]["status"] != "validated"


def test_deleted_package_loses_its_validated_status(env):
    env.build_pipeline()
    env.run("run", "generate")
    env.run("validate", "--sample-size", "3")
    shutil.rmtree(env.ws / "out/tasks/addnums/t_q1")
    env.run("validate", "--level", "2")
    assert rows(env)["t_q1"]["status"] != "validated"
