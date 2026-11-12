"""Generated task.toml parses with Harbor's own model (skipped when harbor is not installed)."""

import tomllib

import pytest

from mint.harbor import materialize
from mint.models import Answer, BuildResult


def test_task_toml_validates_against_harbor_model(tmp_path):
    config = pytest.importorskip("harbor.models.task.config")
    b = BuildResult(
        task_id="t1", instruction="Add.", evidence={"n.txt": "2\n3"}, answer=Answer("5", "numeric")
    )
    materialize(b, tmp_path / "t1", "add", "add@1", ["a.jsonl#0"])
    cfg = config.TaskConfig.model_validate(tomllib.loads((tmp_path / "t1/task.toml").read_text()))
    assert cfg.task.name == "mint/add-t1"
    assert cfg.verifier.environment_mode.value == "separate"
    assert [str(a) for a in cfg.artifacts] == ["/app/answer"]


def test_judge_task_toml_validates_against_harbor_model(tmp_path):
    config = pytest.importorskip("harbor.models.task.config")
    b = BuildResult(
        task_id="j1",
        instruction="Explain.",
        evidence={"n.txt": "2\n3"},
        answer=Answer("ref", "judge"),
        rubric=["a", "b", "c"],
        wrong_answers=["w"],
        judge_model="m-1",
        judge_reason="free text",
    )
    materialize(b, tmp_path / "j1", "ex", "ex@1", ["a.jsonl#0"])
    cfg = config.TaskConfig.model_validate(tomllib.loads((tmp_path / "j1/task.toml").read_text()))
    assert cfg.verifier.env["MINT_JUDGE_MODEL"] == "${MINT_JUDGE_MODEL:-m-1}"
    assert cfg.verifier.environment.network_mode.value == "public"
    assert cfg.metadata["mint"]["verifiable"] is False
