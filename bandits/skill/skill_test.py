"""Tests for the alignment workspace.

Everything here runs against the **real** pipeline on the golden fixture -- no
hand-built stub surfaces -- because the scaffold's whole job is to render
artifacts the pipeline actually produces, and a stub would let a field drift
without anyone noticing.

The single most important test in this module is
``test_unedited_scaffold_is_not_reviewed``. Silence must never count as
approval; if that one ever passes for the wrong reason, an unreviewed reward
function can reach a training run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bandits.contracts import StateSchema, TaskCase, ToolClass, ToolSurface, Verifier
from bandits.ingest import load_corpus_and_registry
from bandits.skill import SKILL_PATH, TEMPLATE_DIR
from bandits.skill.scaffold import (
    TODO,
    apply_overrides,
    is_undecided,
    read_back,
    scaffold_workspace,
)
from bandits.state import infer_schema
from bandits.surface import build_surface
from bandits.task import mine_tasks
from bandits.verify import UnlabeledTraceError, synthesize_verifier

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# real pipeline artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifacts():
    corpus, registry = load_corpus_and_registry(
        FIXTURES / "traces.otlp.jsonl", "otlp", tools_path=FIXTURES / "tools.json"
    )
    surface = build_surface(corpus, declared_tools=registry or None)
    schema = infer_schema(corpus, surface)
    mining = mine_tasks(corpus, schema)
    by_trace = {t.trace_id: t for t in corpus.traces}
    verifiers: list[Verifier] = []
    skipped: list[dict] = []
    for task in mining.tasks:
        try:
            verifiers.append(synthesize_verifier(task, by_trace[task.trace_id], schema))
        except (UnlabeledTraceError, ValueError) as exc:
            skipped.append({"task_id": task.task_id, "reason": str(exc)})
    return {
        "surface": surface,
        "schema": schema,
        "tasks": tuple(mining.tasks),
        "verifiers": tuple(verifiers),
        "skipped_tasks": tuple(mining.skipped),
        "skipped_verifiers": tuple(skipped),
    }


@pytest.fixture
def workspace(tmp_path, artifacts) -> Path:
    scaffold_workspace(
        tmp_path / "ws",
        surface=artifacts["surface"],
        schema=artifacts["schema"],
        tasks=artifacts["tasks"],
        verifiers=artifacts["verifiers"],
        skipped_tasks=artifacts["skipped_tasks"],
        skipped_verifiers=artifacts["skipped_verifiers"],
        name="golden",
    )
    return tmp_path / "ws"


def _sub(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"pattern not present in {path.name}: {old!r}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# the skill itself
# ---------------------------------------------------------------------------


def test_skill_frontmatter_has_name_and_triggering_description():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = text.split("---", 2)[1]
    assert re.search(r"^name:\s*bandits\s*$", front, re.M)
    description = re.search(r"^description:\s*(.+)$", front, re.M)
    assert description
    body = description.group(1).lower()
    for trigger in ("rl environment from", "agent traces", "training environment"):
        assert trigger in body


def test_skill_body_walks_every_step():
    text = SKILL_PATH.read_text(encoding="utf-8")
    for step in range(8):
        assert f"## Step {step} —" in text, f"missing Step {step}"
    # the honest early exit must be present and unambiguous
    assert "**Stop.**" in text
    assert "LLM-judge environment as a consolation" in text or "consolation" in text


def test_templates_exist_and_carry_todo_markers():
    for name in ("ENVIRONMENT.md", "TASKS.md", "VERIFIER.md"):
        path = TEMPLATE_DIR / name
        assert path.exists(), name
        assert TODO in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------


def test_scaffold_produces_all_three_files_from_real_artifacts(workspace):
    for name in ("ENVIRONMENT.md", "TASKS.md", "VERIFIER.md"):
        assert (workspace / name).exists()
        assert (workspace / name).stat().st_size > 0


def test_environment_md_carries_real_surface_and_schema_content(workspace):
    text = (workspace / "ENVIRONMENT.md").read_text(encoding="utf-8")
    # tools, with their classes and their evidence
    assert "| refund_order | write |" in text
    assert "| get_order | read |" in text
    assert "| send_email | external |" in text
    assert "observed state change: refund_order(order_id=7741)" in text
    # entities
    for entity in ("orders", "customers", "products", "store_policy"):
        assert f"### {entity}" in text
    # the honest degradations
    assert "Static snapshot:** yes" in text
    assert "escalate_to_human" in text


def test_tasks_md_states_the_label_problem_and_lists_real_tasks(workspace, artifacts):
    text = (workspace / "TASKS.md").read_text(encoding="utf-8")
    assert '"Reproduce what production did" is not a reward function' in text
    for task in artifacts["tasks"]:
        assert task.task_id in text
    # solvability warnings survive into the review surface
    assert "no pre-state row carries that value" in text


def test_verifier_md_renders_assertions_and_refusals(workspace):
    text = (workspace / "VERIFIER.md").read_text(encoding="utf-8")
    assert "state_equals" in text
    assert "state_unchanged" in text
    assert "effect_count" in text
    assert "orders[7741].status" in text
    # a refusal is reported as a refusal
    assert "refusing to synthesize a verifier" in text


def test_scaffold_refuses_to_clobber_human_edits(tmp_path, artifacts):
    root = tmp_path / "ws"
    kwargs = dict(surface=artifacts["surface"], schema=artifacts["schema"])
    scaffold_workspace(root, **kwargs)
    (root / "ENVIRONMENT.md").write_text("my careful edits", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scaffold_workspace(root, **kwargs)
    assert (root / "ENVIRONMENT.md").read_text(encoding="utf-8") == "my careful edits"
    scaffold_workspace(root, overwrite=True, **kwargs)
    assert "ENVIRONMENT" in (root / "ENVIRONMENT.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# TODO(human) coverage: every human decision is marked
# ---------------------------------------------------------------------------


def test_todo_marks_every_place_a_human_must_decide(workspace, artifacts):
    env = (workspace / "ENVIRONMENT.md").read_text(encoding="utf-8")
    tasks = (workspace / "TASKS.md").read_text(encoding="utf-8")
    ver = (workspace / "VERIFIER.md").read_text(encoding="utf-8")

    # a class we could not infer is never defaulted
    assert "| escalate_to_human | unknown | " + TODO in env
    # the blind-write question is asked for every read tool, by name
    reads = [t.name for t in artifacts["surface"].tools if t.tool_class is ToolClass.READ]
    assert reads
    for name in reads:
        assert f"### tool:{name}" in env
    assert env.count(f"- **Confirmed read-only:** {TODO}") == len(reads)
    # snapshots and write semantics need acknowledgement
    assert f"- **Acceptable as a snapshot:** {TODO}" in env
    assert f"- **Write semantics correct:** {TODO}" in env

    # every task must name a real downstream label
    n_tasks = len(artifacts["tasks"])
    assert tasks.count(f"- **Downstream signal for this task:** {TODO}") == n_tasks
    assert f"- **Downstream signal available in your systems:** {TODO}" in tasks
    # warned tasks are undecided, not silently included
    assert "| task-ep-notfound | " + TODO in tasks

    # every verifier needs a signature
    n_ver = len(artifacts["verifiers"])
    assert ver.count(f"- **Reviewed by:** {TODO}") == n_ver
    assert ver.count(f"- **Missing invariants:** {TODO}") == n_ver


# ---------------------------------------------------------------------------
# THE test: silence is not approval
# ---------------------------------------------------------------------------


def test_unedited_scaffold_is_not_reviewed(workspace, artifacts):
    """An untouched workspace must read back as approving absolutely nothing."""
    overrides = read_back(workspace)

    assert overrides.reviewed_by == {}
    assert overrides.is_reviewed is False
    assert overrides.ready_for_training is False
    assert set(overrides.unreviewed_verifiers) == {
        v.verifier_id for v in artifacts["verifiers"]
    }
    assert overrides.open_questions, "an unedited scaffold must report open questions"

    # and applying it hands the trainer nothing that could grade a rollout
    applied = apply_overrides(
        overrides,
        surface=artifacts["surface"],
        tasks=artifacts["tasks"],
        verifiers=artifacts["verifiers"],
    )
    assert applied.verifiers == ()
    assert set(applied.dropped_verifiers) == {v.verifier_id for v in artifacts["verifiers"]}
    assert all(v.reviewed_by is None for v in artifacts["verifiers"])


@pytest.mark.parametrize("value", ["", "-", "?", "TBD", "n/a", TODO, f"{TODO} (yes / no)"])
def test_placeholders_never_count_as_an_answer(value):
    assert is_undecided(value) is True


@pytest.mark.parametrize("value", ["yes", "no", "Ada Lovelace", "ada@example.com"])
def test_real_answers_are_recognized(value):
    assert is_undecided(value) is False


# ---------------------------------------------------------------------------
# round-tripping human edits
# ---------------------------------------------------------------------------


def test_read_back_round_trips_a_tool_class_override(workspace, artifacts):
    env = workspace / "ENVIRONMENT.md"
    # the human catches a blind write: get_customer silently stamps a "last seen" field
    _sub(env, "| get_customer | read | read |", "| get_customer | read | write |")
    _sub(
        env,
        "### tool:get_customer\n\n- **Inferred:** read",
        "### tool:get_customer\n\n- **Inferred:** read",
    )
    text = env.read_text(encoding="utf-8")
    marker = "### tool:get_customer"
    head, tail = text.split(marker, 1)
    tail = tail.replace(f"- **Confirmed read-only:** {TODO} (yes / no)", "- **Confirmed read-only:** no", 1)
    tail = tail.replace(
        f"- **If no, what does it change:** {TODO}",
        "- **If no, what does it change:** customers.last_seen_at, set to now",
        1,
    )
    env.write_text(head + marker + tail, encoding="utf-8")

    overrides = read_back(workspace)
    assert overrides.tool_classes == {"get_customer": ToolClass.WRITE}
    assert overrides.declared_blind_writes == ("get_customer",)
    assert overrides.issues == ()

    applied = apply_overrides(overrides, surface=artifacts["surface"])
    profile = applied.surface.by_name("get_customer")
    assert profile.tool_class is ToolClass.WRITE
    assert profile.class_confidence == 1.0
    assert any("human override" in e for e in profile.class_evidence)
    assert any("blind write" in e for e in profile.class_evidence)
    # untouched tools are passed through unchanged
    assert applied.surface.by_name("get_order") == artifacts["surface"].by_name("get_order")


def test_a_blind_write_left_out_of_the_table_is_reported_as_a_contradiction(workspace):
    env = workspace / "ENVIRONMENT.md"
    text = env.read_text(encoding="utf-8")
    marker = "### tool:get_product"
    head, tail = text.split(marker, 1)
    tail = tail.replace(f"- **Confirmed read-only:** {TODO} (yes / no)", "- **Confirmed read-only:** no", 1)
    env.write_text(head + marker + tail, encoding="utf-8")

    overrides = read_back(workspace)
    assert overrides.declared_blind_writes == ("get_product",)
    assert any("get_product" in issue and "still says 'read'" in issue for issue in overrides.issues)


def test_read_back_rejects_a_class_that_is_not_a_tool_class(workspace):
    _sub(workspace / "ENVIRONMENT.md", "| get_order | read | read |", "| get_order | read | maybe |")
    overrides = read_back(workspace)
    assert overrides.tool_classes == {}
    assert any("get_order" in issue and "maybe" in issue for issue in overrides.issues)


def test_read_back_round_trips_a_reviewer_sign_off(workspace, artifacts):
    ver = workspace / "VERIFIER.md"
    target = artifacts["verifiers"][0]
    text = ver.read_text(encoding="utf-8")
    head, tail = text.split(f"### {target.verifier_id}", 1)
    tail = tail.replace(
        f"- **Reviewed by:** {TODO}",
        "- **Reviewed by:** Ada Lovelace <ada@example.com>",
        1,
    )
    ver.write_text(head + f"### {target.verifier_id}" + tail, encoding="utf-8")

    overrides = read_back(workspace)
    assert overrides.reviewed_by == {target.verifier_id: "Ada Lovelace <ada@example.com>"}
    assert target.verifier_id not in overrides.unreviewed_verifiers
    # one signature is not a blanket approval
    assert overrides.is_reviewed is False
    assert overrides.unreviewed_verifiers


def test_read_back_round_trips_a_task_exclusion(workspace, artifacts):
    tasks_file = workspace / "TASKS.md"
    _sub(tasks_file, "| task-ep-refund-ok | yes |", "| task-ep-refund-ok | no |")
    _sub(tasks_file, "| task-ep-refund-ok-2 | yes |", "| task-ep-refund-ok-2 | yes |")

    overrides = read_back(workspace)
    assert "task-ep-refund-ok" in overrides.excluded_tasks
    assert "task-ep-refund-ok" not in overrides.included_tasks
    assert "task-ep-refund-ok-2" in overrides.included_tasks
    # a warned task left undecided is excluded, never included
    assert "task-ep-notfound" in overrides.undecided_tasks
    assert "task-ep-notfound" not in overrides.included_tasks

    applied = apply_overrides(
        overrides, surface=artifacts["surface"], tasks=artifacts["tasks"]
    )
    kept = {t.task_id for t in applied.tasks}
    assert "task-ep-refund-ok" not in kept
    assert "task-ep-refund-ok-2" in kept
    assert "task-ep-notfound" not in kept


def test_a_signed_verifier_for_an_excluded_task_is_still_withheld(workspace, artifacts):
    """Sign-off is necessary, not sufficient: the task has to be included too."""
    target = next(v for v in artifacts["verifiers"] if v.task_id == "task-ep-refund-ok")
    ver = workspace / "VERIFIER.md"
    text = ver.read_text(encoding="utf-8")
    head, tail = text.split(f"### {target.verifier_id}", 1)
    tail = tail.replace(f"- **Reviewed by:** {TODO}", "- **Reviewed by:** Ada", 1)
    ver.write_text(head + f"### {target.verifier_id}" + tail, encoding="utf-8")
    _sub(workspace / "TASKS.md", "| task-ep-refund-ok | yes |", "| task-ep-refund-ok | no |")

    applied = apply_overrides(
        read_back(workspace),
        surface=artifacts["surface"],
        tasks=artifacts["tasks"],
        verifiers=artifacts["verifiers"],
    )
    assert target.verifier_id in applied.dropped_verifiers


def test_full_sign_off_produces_a_graded_verifier(workspace, artifacts):
    """The happy path: everything answered, and reviewed_by lands on the contract."""
    tasks_file = workspace / "TASKS.md"
    text = tasks_file.read_text(encoding="utf-8")
    text = text.replace(f"| {TODO} |", "| yes |")
    text = text.replace(f"- **Downstream signal for this task:** {TODO}", "- **Downstream signal for this task:** zendesk.ticket_reopened_7d == false")
    tasks_file.write_text(text, encoding="utf-8")

    ver = workspace / "VERIFIER.md"
    ver.write_text(
        ver.read_text(encoding="utf-8").replace(f"- **Reviewed by:** {TODO}", "- **Reviewed by:** Ada"),
        encoding="utf-8",
    )

    overrides = read_back(workspace)
    assert overrides.is_reviewed is True
    assert overrides.label_sources
    assert all(v.startswith("zendesk.") for v in overrides.label_sources.values())

    applied = apply_overrides(
        overrides,
        surface=artifacts["surface"],
        tasks=artifacts["tasks"],
        verifiers=artifacts["verifiers"],
    )
    assert applied.verifiers
    assert all(v.reviewed_by == "Ada" for v in applied.verifiers)
    # the source contracts were not mutated
    assert all(v.reviewed_by is None for v in artifacts["verifiers"])


def test_read_back_tolerates_a_missing_file(tmp_path):
    (tmp_path / "empty").mkdir()
    overrides = read_back(tmp_path / "empty")
    assert overrides.reviewed_by == {}
    assert overrides.included_tasks == ()
    assert overrides.is_reviewed is False


def test_read_back_survives_reordered_columns_and_extra_prose(workspace):
    """The parser addresses cells by column name, so a reviewer may restructure."""
    env = workspace / "ENVIRONMENT.md"
    lines = env.read_text(encoding="utf-8").splitlines()
    out = []
    in_table = False
    for line in lines:
        if line.startswith("| tool | inferred | decision |"):
            in_table = True
        elif in_table and not line.startswith("|"):
            in_table = False
        if in_table and line.startswith("|") and not set(line) <= set("|- :"):
            cells = line.strip().strip("|").split("|")
            cells = [cells[-1], *cells[:-1], " reviewer notes "]
            line = "|" + "|".join(cells) + "|"
        out.append(line)
    text = "\n".join(out).replace("## 6. Open questions", "Some notes I typed.\n\n## 6. Open questions")
    env.write_text(text, encoding="utf-8")

    overrides = read_back(workspace)
    assert overrides.issues == ()
    assert overrides.tool_classes == {}
    assert any(q.subject == "escalate_to_human" for q in overrides.open_questions)


def test_a_pipe_in_an_instruction_does_not_break_the_table(artifacts, tmp_path):
    from bandits.skill.scaffold import tasks_md

    weird = artifacts["tasks"][0].model_copy(
        update={"instruction": "refund order 7741 | and email | the customer"}
    )
    body = tasks_md([weird])
    path = tmp_path / "TASKS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    overrides = read_back(tmp_path)
    assert weird.task_id in overrides.included_tasks + overrides.undecided_tasks + overrides.excluded_tasks


def test_renderers_are_deterministic(artifacts):
    from bandits.skill.scaffold import environment_md, tasks_md, verifier_md

    surface: ToolSurface = artifacts["surface"]
    schema: StateSchema = artifacts["schema"]
    tasks: tuple[TaskCase, ...] = artifacts["tasks"]
    assert environment_md(surface, schema) == environment_md(surface, schema)
    assert tasks_md(tasks) == tasks_md(tasks)
    assert verifier_md(artifacts["verifiers"]) == verifier_md(artifacts["verifiers"])
