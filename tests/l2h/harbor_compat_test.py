"""Generated task.toml parses with Harbor's own model (skipped when harbor is not installed)."""

import tomllib

import pytest

from l2h.harbor import materialize
from l2h.models import Answer, BuildResult


def test_task_toml_validates_against_harbor_model(tmp_path):
    config = pytest.importorskip("harbor.models.task.config")
    b = BuildResult(
        task_id="t1", instruction="Add.", evidence={"n.txt": "2\n3"}, answer=Answer("5", "numeric")
    )
    materialize(b, tmp_path / "t1", "add", "add@1", ["a.jsonl#0"])
    cfg = config.TaskConfig.model_validate(tomllib.loads((tmp_path / "t1/task.toml").read_text()))
    assert cfg.task.name == "l2h/add-t1"
    assert cfg.verifier.environment_mode.value == "separate"
    assert [str(a) for a in cfg.artifacts] == ["/app/answer"]
