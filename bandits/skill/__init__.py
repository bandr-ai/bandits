"""The product surface: an installable skill plus the alignment workspace behind it.

``SKILL.md`` is the procedure an agent follows. ``templates/`` holds the blank
alignment artifacts. :mod:`bandits.skill.scaffold` is the code that populates
them from real pipeline output and parses the human's edits back out.

The design premise, borrowed from LangChain's ``eval-engineering`` and then
pushed further: agents are poor one-shot environment generators because they are
misaligned with human goals, so good environments come from infusing human
feedback into the generation process. What we add on top is that the generation
is deterministic and checkable -- reconstruction from invocation points,
code-based verifiers, and a per-tool fidelity gate -- so the human's attention
lands on the handful of judgements that genuinely require it instead of on
auditing a model's prose.
"""

from __future__ import annotations

from pathlib import Path

from bandits.skill.scaffold import (
    TODO,
    AppliedWorkspace,
    OpenQuestion,
    WorkspaceOverrides,
    WorkspacePaths,
    apply_overrides,
    environment_md,
    is_undecided,
    read_back,
    scaffold_workspace,
    tasks_md,
    verifier_md,
)

SKILL_PATH = Path(__file__).resolve().parent / "SKILL.md"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

__all__ = [
    "SKILL_PATH",
    "TEMPLATE_DIR",
    "TODO",
    "AppliedWorkspace",
    "OpenQuestion",
    "WorkspaceOverrides",
    "WorkspacePaths",
    "apply_overrides",
    "environment_md",
    "is_undecided",
    "read_back",
    "scaffold_workspace",
    "tasks_md",
    "verifier_md",
]
