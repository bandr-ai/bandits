from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from bandits_jev.cli import app

runner = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip ANSI codes: CI renders output with color, which can split a
    plain-text match across color codes."""
    return _ANSI_ESCAPE.sub("", text)


def _score(tmp_path, split: str):
    return runner.invoke(
        app,
        [
            "score",
            "decision-dataset-does-not-matter",
            "--model",
            "irrelevant/model",
            "--revision",
            "abc123",
            "--split",
            split,
            "--project",
            str(tmp_path),
        ],
    )


def test_score_rejects_an_invalid_split(tmp_path) -> None:
    result = _score(tmp_path, "bogus")

    assert result.exit_code == 1
    assert "--split must be one of" in plain(result.stdout)


def test_score_refuses_the_test_split_without_allow_test(tmp_path) -> None:
    result = _score(tmp_path, "test")

    assert result.exit_code == 1
    assert "--allow-test" in plain(result.stdout)


def test_import_predictions_refuses_test_without_allow_test(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "import-predictions",
            str(tmp_path / "predictions.jsonl"),
            "--dataset",
            "decision-dataset-does-not-matter",
            "--name",
            "jev",
            "--model",
            "jev",
            "--revision",
            "today",
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "--allow-test" in plain(result.stdout)


@pytest.mark.parametrize("command", ["dataset", "import", "score", "train", "calibrate", "import-predictions", "report"])
def test_every_command_is_on_the_jev_cli(command) -> None:
    assert runner.invoke(app, [command, "--help"]).exit_code == 0
