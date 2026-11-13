from importlib import resources
from pathlib import Path

from mint import __version__
from mint.cli import main


def test_user_install_without_workspace_is_exact_and_idempotent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["skill", "install"]) == 0
    target = tmp_path / ".claude/skills/mint/SKILL.md"
    assert target.read_bytes() == resources.files("mint").joinpath("SKILL.md").read_bytes()
    assert main(["skill", "install"]) == 0
    assert "already installed" in capsys.readouterr().out


def test_project_dry_run_conflict_and_force(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / ".claude/skills/mint/SKILL.md"
    assert main(["skill", "install", "--project", "--dry-run"]) == 0
    assert not target.exists()
    target.parent.mkdir(parents=True)
    target.write_text("my custom skill")
    assert main(["skill", "install", "--project"]) == 2
    assert target.read_text() == "my custom skill"
    assert main(["skill", "install", "--project", "--force", "--json"]) == 0
    assert target.read_text().startswith("---\nname: mint\n")


def test_install_refuses_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / ".claude").symlink_to(elsewhere, target_is_directory=True)
    assert main(["skill", "install", "--project"]) == 2
    assert not list(elsewhere.iterdir())


def test_release_versions_match():
    import tomllib

    import bandits

    root = Path(__file__).resolve().parents[2]
    recipe = tomllib.loads((root / "recipes/mint/pyproject.toml").read_text())
    assert __version__ == bandits.__version__ == recipe["project"]["version"]
