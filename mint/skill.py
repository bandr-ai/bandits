"""Install the packaged Mint skill for Claude Code without needing a workspace."""

from importlib import resources
from pathlib import Path

from .output import Result, UsageError
from .workspace import atomic_write


def install(*, project: bool = False, force: bool = False, dry_run: bool = False) -> Result:
    root = Path.cwd() if project else Path.home()
    folder = root / ".claude" / "skills" / "mint"
    target = folder / "SKILL.md"
    # Never follow an existing symlink when writing instructions.
    for path in (root / ".claude", root / ".claude" / "skills", folder, target):
        if path.is_symlink():
            raise UsageError(f"skill path is a symlink: {path}", "choose a regular skill directory")
    content = resources.files("mint").joinpath("SKILL.md").read_text(encoding="utf-8")
    existing = target.read_text(encoding="utf-8") if target.is_file() else None
    if existing is not None and existing != content and not force:
        raise UsageError(
            f"a different skill already exists at {target}",
            "review it, then use `mint skill install --force` to replace SKILL.md",
        )
    if target.exists() and not target.is_file():
        raise UsageError(
            f"skill target is not a file: {target}", "choose a regular skill directory"
        )
    changed = existing != content
    if changed and not dry_run:
        try:
            atomic_write(target, content)
        except OSError as error:
            raise UsageError(
                f"cannot install skill: {error}", "check directory permissions"
            ) from error
    verb = (
        "would install" if dry_run and changed else "installed" if changed else "already installed"
    )
    return Result(
        {"path": str(target), "changed": changed, "dry_run": dry_run, "agent": "claude"},
        f"{verb}: {target}",
        ["Start a new Claude Code session and ask it to use the Mint skill with your logs."],
    )
