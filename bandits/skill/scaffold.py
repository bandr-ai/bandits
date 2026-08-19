"""The alignment workspace: pipeline artifacts out to markdown, human edits back in.

An agent cannot one-shot an environment that matches what a team actually wants.
The pipeline (ingest -> fidelity) is deterministic and produces facts; the parts
it *cannot* settle -- is this tool really read-only, is this task worth training
on, where does the outcome label come from, is this reward function right --
have to be decided by a person. This module is the surface those decisions
happen on.

Two directions:

``scaffold_workspace``
    Real ``ToolSurface`` / ``StateSchema`` / ``TaskCase`` / ``Verifier`` objects
    in, three markdown files out, every undecided thing stamped
    ``**TODO(human)**``.

``read_back``
    The edited markdown in, a :class:`WorkspaceOverrides` out: tool-class
    overrides, task inclusion/exclusion, reviewer sign-off, declared blind
    writes, outcome-label sources, and every ``**TODO(human)**`` still standing.

**Silence is never approval.** A scaffold nobody edited reads back with zero
reviewers, zero included tasks and a list of open questions. ``apply_overrides``
will therefore hand the trainer nothing. That is the intended behaviour, not a
degenerate case.

---------------------------------------------------------------------------
The markdown format, specified
---------------------------------------------------------------------------

Deliberately small, so round-tripping is reliable and a diff in a pull request
is readable. Only three constructs are parsed; everything else in the file is
prose for humans and is ignored.

1. **Section heading** -- ``### <id>``. Binds the fields under it to that id
   until the next ``##``/``###``.

2. **Field line** -- ``- **Key:** value``. Case-insensitive on ``Key``. The
   value runs to end of line.

3. **Decision table** -- a GitHub-style pipe table. The header row names the
   columns; the parser addresses cells by column name, so columns may be
   reordered or added without breaking it. Cells use ``\\|`` for a literal pipe
   and ``<br>`` for a line break.

A value counts as **undecided** when it is empty, or contains ``TODO(human)``,
or is one of ``-``, ``?``, ``tbd``, ``n/a``. Undecided never means yes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandits.contracts import (
    Assertion,
    EntitySchema,
    FidelityReport,
    JsonObject,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolProfile,
    ToolSurface,
    Verifier,
)

__all__ = [
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

#: The marker an agent writes wherever it refuses to decide for the human.
TODO = "**TODO(human)**"

_UNDECIDED = {"", "-", "--", "?", "tbd", "n/a", "na", "none", "unset", "_unset_", "todo"}

_FIELD_RE = re.compile(r"^\s*[-*]\s+\*\*(?P<key>[^*:]+):\*\*\s*(?P<value>.*?)\s*$")
_HEADING_RE = re.compile(r"^\s*(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")
_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

_YES = {"yes", "y", "true", "include", "included", "keep", "confirmed", "ok"}
_NO = {"no", "n", "false", "exclude", "excluded", "drop", "reject"}


# ---------------------------------------------------------------------------
# small text helpers
# ---------------------------------------------------------------------------


def is_undecided(value: str | None) -> bool:
    """True when a human has not actually answered.

    Empty, a placeholder dash, or anything still carrying ``TODO(human)``.
    """
    if value is None:
        return True
    text = value.strip()
    if "TODO(human)" in text:
        return True
    return text.strip("*_` ").lower() in _UNDECIDED


def _yes_no(value: str) -> bool | None:
    """``True`` / ``False`` / ``None`` for undecided. Never guesses."""
    if is_undecided(value):
        return None
    token = value.strip().strip("*_`. ").lower().split()[0] if value.strip() else ""
    if token in _YES:
        return True
    if token in _NO:
        return False
    return None


def _cell(value: Any) -> str:
    """Escape a value for a markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _uncell(text: str) -> str:
    return text.replace("\\|", "|").replace("<br>", " ").strip()


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _bullets(lines: Iterable[str], indent: str = "  ") -> str:
    out = [f"{indent}- {line}" for line in lines]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# parsed representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenQuestion:
    """One decision the scaffold asked for and the markdown does not answer."""

    file: str
    subject: str
    """The tool name, task id, verifier id or entity the question is about."""

    question: str


@dataclass(frozen=True)
class WorkspaceOverrides:
    """Everything a human's edits added on top of what the pipeline inferred."""

    tool_classes: dict[str, ToolClass] = field(default_factory=dict)
    """Tool -> class, only where the decision differs from the inferred class."""

    declared_blind_writes: tuple[str, ...] = ()
    """Tools a human says mutate state despite being classified READ.

    The classifier cannot detect these: a mutating tool that is never followed
    by a read of the same row anywhere in the corpus leaves no evidence of a
    write. This field is how that knowledge gets into the pipeline.
    """

    included_tasks: tuple[str, ...] = ()
    excluded_tasks: tuple[str, ...] = ()
    undecided_tasks: tuple[str, ...] = ()
    """Tasks whose ``Include`` cell is still undecided. Treated as excluded."""

    label_sources: dict[str, str] = field(default_factory=dict)
    """Task -> the downstream signal a human named as the outcome label."""

    reviewed_by: dict[str, str] = field(default_factory=dict)
    """Verifier id -> reviewer. Only verifiers a person actually signed."""

    unreviewed_verifiers: tuple[str, ...] = ()
    accepted_static_entities: tuple[str, ...] = ()
    accepted_unsupported_tools: tuple[str, ...] = ()
    open_questions: tuple[OpenQuestion, ...] = ()
    issues: tuple[str, ...] = ()
    """Contradictions and unparseable values. Never silently ignored."""

    @property
    def is_reviewed(self) -> bool:
        """True only when at least one verifier is signed and none is unsigned."""
        return bool(self.reviewed_by) and not self.unreviewed_verifiers

    @property
    def ready_for_training(self) -> bool:
        """The gate ``apply_overrides`` enforces: signed verifiers, no open questions."""
        return self.is_reviewed and bool(self.included_tasks) and not self.open_questions


@dataclass(frozen=True)
class WorkspacePaths:
    environment: Path
    tasks: Path
    verifier: Path

    def __iter__(self):  # pragma: no cover - convenience
        return iter((self.environment, self.tasks, self.verifier))


@dataclass(frozen=True)
class AppliedWorkspace:
    """Pipeline artifacts with the human's decisions folded in."""

    surface: ToolSurface
    tasks: tuple[TaskCase, ...]
    verifiers: tuple[Verifier, ...]
    dropped_tasks: tuple[str, ...] = ()
    dropped_verifiers: tuple[str, ...] = ()
    """Verifiers withheld because nobody signed them."""


# ---------------------------------------------------------------------------
# rendering: ENVIRONMENT.md
# ---------------------------------------------------------------------------


def _evidence_block(profile: ToolProfile) -> str:
    if not profile.class_evidence:
        return "  - _no evidence recorded_"
    return _bullets(profile.class_evidence)


def _unfalsified(profile: ToolProfile) -> bool:
    """True when nothing in the corpus could have disproved 'this tool only reads'."""
    return any("unfalsified" in e or "no before/after" in e for e in profile.class_evidence)


def _entity_block(entity: EntitySchema) -> str:
    lines = [f"### {entity.name}", ""]
    lines.append(f"- **Primary key:** {entity.primary_key or TODO + ' (no key was recoverable)'}")
    lines.append(f"- **Fields:** {', '.join(f.name for f in entity.fields) or '_none observed_'}")
    lines.append(f"- **Written by:** {', '.join(entity.written_by) or '_nothing_'}")
    lines.append(f"- **Read by:** {', '.join(entity.read_by) or '_nothing_'}")
    lines.append(f"- **Rows of evidence:** {entity.evidence_count}")
    if entity.foreign_keys:
        fks = ", ".join(
            f"{fk.field} -> {fk.references_entity}.{fk.references_field} ({fk.confidence:.2f})"
            for fk in entity.foreign_keys
        )
        lines.append(f"- **Foreign keys:** {fks}")
    else:
        lines.append("- **Foreign keys:** _none inferred_")
    if entity.static_snapshot:
        lines += [
            "- **Static snapshot:** yes — the corpus never wrote this entity and nothing",
            "  cross-references it, so no structure could be inferred. Rows are materialized",
            "  verbatim. We refuse to invent a table here.",
            f"- **Acceptable as a snapshot:** {TODO} (yes / no — if agents must *write* this,",
            "  say so and the pipeline needs write evidence, not a snapshot)",
        ]
    if entity.write_effects:
        lines.append("- **Observed write semantics:**")
        for we in entity.write_effects:
            lines.append(
                f"  - `{we.tool}` key=`{we.key_argument}` "
                f"sets={we.sets_constants or '{}'} "
                f"arg_columns={we.argument_columns or '{}'} "
                f"(confidence {we.confidence:.2f}, {we.evidence_count} observation(s))"
            )
            lines.append(_bullets(we.evidence, indent="    "))
        lines.append(f"- **Write semantics correct:** {TODO} (yes / no + what is wrong)")
    lines.append("")
    return "\n".join(lines)


def environment_md(
    surface: ToolSurface,
    schema: StateSchema,
    *,
    name: str = "environment",
    fidelity: FidelityReport | None = None,
) -> str:
    """Render ENVIRONMENT.md from a real surface and schema."""
    unknown = [t for t in surface.tools if t.tool_class is ToolClass.UNKNOWN]
    reads = [t for t in surface.tools if t.tool_class is ToolClass.READ]
    externals = [t for t in surface.tools if t.tool_class is ToolClass.EXTERNAL]

    out: list[str] = []
    out.append(f"# ENVIRONMENT — {name}")
    out.append("")
    out.append(
        "Generated by `bandits` from the corpus. Everything above the line in each section "
        "is a fact recovered from the traces; everything marked "
        f"{TODO} is a decision only you can make."
    )
    out.append("")
    out.append("**Review order:** tool classes first (they decide everything downstream), "
               "then the blind-write check, then the entities.")
    out.append("")

    # -- tool table ---------------------------------------------------------
    out.append("## 1. Tool classification")
    out.append("")
    out.append(
        "`read` is answered from the rebuilt store, `write` mutates it and is what verifiers "
        "assert on, `external` is stubbed and recorded to the effect ledger and never performed, "
        "`unknown` means the tool is not reimplemented and calling it raises."
    )
    out.append("")
    out.append("Edit the **decision** column. Leave it equal to *inferred* to accept.")
    out.append("")
    out.append("| tool | inferred | decision | calls | confidence | flags |")
    out.append("|---|---|---|---|---|---|")
    for profile in surface.tools:
        flags = []
        if profile.observed_only:
            flags.append("observed-only")
        if profile.declared_only:
            flags.append("declared, never called")
        if profile.error_modes:
            flags.append(f"{len(profile.error_modes)} error mode(s)")
        decision = TODO if profile.tool_class is ToolClass.UNKNOWN else profile.tool_class.value
        out.append(
            "| {} | {} | {} | {} | {:.2f} | {} |".format(
                _cell(profile.name),
                _cell(profile.tool_class.value),
                _cell(decision),
                profile.call_count,
                profile.class_confidence,
                _cell(", ".join(flags)),
            )
        )
    out.append("")

    out.append("Evidence for each classification, so the decision above is reviewable:")
    out.append("")
    for profile in surface.tools:
        out.append(f"- `{profile.name}` -> {profile.tool_class.value}")
        out.append(_evidence_block(profile))
    out.append("")

    if unknown:
        out.append(
            f"**{len(unknown)} tool(s) could not be classified.** They are left out of the "
            "rebuilt environment and raise if called — never a fabricated success. Give each "
            "one a class, or confirm it stays out:"
        )
        out.append("")
        for profile in unknown:
            out.append(f"- `{profile.name}` — see its evidence above")
        out.append("")

    # -- blind write check --------------------------------------------------
    out.append("## 2. Blind-write check — the thing our classifier cannot see")
    out.append("")
    out.append(
        "A tool is classified `write` only when the corpus shows a before/after difference on "
        "the same row. A tool that mutates state and is **never followed by a read of what it "
        "changed** leaves no such evidence anywhere in the corpus, so it is classified `read`. "
        "The heuristic is not weak here, it is blind: the information is not in the data."
    )
    out.append("")
    out.append(
        "If a blind write ships as `read`, the rebuilt tool returns a row instead of changing "
        "one, every verifier assertion about that change fails identically, and the task looks "
        "impossible rather than mismodelled."
    )
    out.append("")
    out.append("Answer for each tool below. This is the highest-value five minutes in the "
               "whole procedure.")
    out.append("")
    if reads:
        for profile in reads:
            risk = "no write evidence was falsifiable" if _unfalsified(profile) else "diffed reads available"
            out.append(f"### tool:{profile.name}")
            out.append("")
            out.append(f"- **Inferred:** read ({risk}, {profile.call_count} call(s))")
            out.append(f"- **Confirmed read-only:** {TODO} (yes / no)")
            out.append(
                "- **If no, what does it change:** " + TODO + " (entity, field, and the value it sets)"
            )
            out.append("")
    else:
        out.append("_No tool was classified `read`._")
        out.append("")

    # -- external -----------------------------------------------------------
    out.append("## 3. External tools — recorded, never performed")
    out.append("")
    if externals:
        for profile in externals:
            out.append(
                f"- `{profile.name}` — {profile.call_count} call(s). Stubbed; each attempt lands "
                "in the effect ledger so a verifier can assert it *would* have fired."
            )
        out.append("")
        out.append(f"- **All irreversible tools are listed above:** {TODO} (yes / no — name any "
                   "tool that touches money, messaging, or a third party and is not here)")
    else:
        out.append(
            f"_None detected._ **This is worth a second look:** {TODO} — does this agent really "
            "never send, charge, or call anything outside your own database?"
        )
    out.append("")

    # -- entities -----------------------------------------------------------
    out.append("## 4. Reconstructed state")
    out.append("")
    out.append(
        f"{len(schema.entities)} entity/entities inferred from repeated identifiers across "
        "tool responses. Columns are the union of every field any response ever showed."
    )
    out.append("")
    for entity in schema.entities:
        out.append(_entity_block(entity))
    if schema.unresolved:
        out.append("### Unattributed tools")
        out.append("")
        out.append(
            "These tools returned successful bodies that could not be attributed to any entity: "
            + ", ".join(f"`{t}`" for t in schema.unresolved)
        )
        out.append("")
        out.append(f"- **What state do these read or write:** {TODO}")
        out.append("")

    # -- fidelity -----------------------------------------------------------
    out.append("## 5. Fidelity")
    out.append("")
    if fidelity is None:
        out.append("_Not run yet._ Run `bandits fidelity` and paste the per-tool table here.")
    else:
        out.append(f"`{fidelity.env_id}` — overall {fidelity.overall_rate:.0%}, "
                   f"threshold {fidelity.threshold:.0%}, "
                   f"**{'ACCEPTED' if fidelity.accepted else 'REJECTED'}**")
        out.append("")
        out.append("| tool | matched | rate | mismatched | unsupported |")
        out.append("|---|---|---|---|---|")
        for tf in fidelity.per_tool:
            out.append(
                f"| {_cell(tf.tool)} | {tf.matched}/{tf.replayed} | {tf.rate:.0%} | "
                f"{tf.mismatched} | {tf.unsupported} |"
            )
        out.append("")
        for note in fidelity.notes:
            out.append(f"> {note}")
        if not fidelity.accepted:
            out.append("")
            out.append(
                f"- **Remaining gaps consciously accepted:** {TODO} (name each failing tool and "
                "why the gap is tolerable, or send it back to schema inference)"
            )
    out.append("")

    # -- open questions -----------------------------------------------------
    out.append("## 6. Open questions")
    out.append("")
    out.append(
        "Add anything the reconstruction got wrong that has no box above. One bullet each; "
        "delete a line only when it is genuinely resolved."
    )
    out.append("")
    out.append(f"- **Anything the traces could not tell us:** {TODO}")
    out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# rendering: TASKS.md
# ---------------------------------------------------------------------------


def _default_include(task: TaskCase) -> str:
    warnings = task.provenance.get("solvability_warnings") or ()
    if task.outcome is None:
        return TODO
    if task.outcome is False:
        return "no"
    if warnings:
        return TODO
    return "yes"


def tasks_md(
    tasks: Sequence[TaskCase],
    *,
    name: str = "environment",
    skipped: Sequence[JsonObject] = (),
) -> str:
    """Render TASKS.md from mined TaskCases."""
    out: list[str] = []
    out.append(f"# TASKS — {name}")
    out.append("")
    out.append(
        f"{len(tasks)} task(s) mined from the corpus, {len(skipped)} refused. Each has an "
        "instruction taken from the real user turn and a starting state reconstructed from the "
        "reads that happened *before* the first write."
    )
    out.append("")

    # -- the label problem, stated up front ---------------------------------
    out.append("## 0. Where the outcome labels come from — read this before the table")
    out.append("")
    out.append(
        "**\"Reproduce what production did\" is not a reward function.** It trains the model to "
        "copy the old system's mistakes, and it looks like it is working, because agreement is "
        "highest on the easy traces. A trace tells you what the agent *did*. It does not tell "
        "you whether that was right."
    )
    out.append("")
    out.append("The label has to come from outside the trace. Best to worst:")
    out.append("")
    out.append(
        "1. **A downstream signal you already have.** Ticket closed and not reopened in 7 days. "
        "Refund not reversed. No human escalation. Payment settled. CSAT thumbs-up. This almost "
        "always exists somewhere in your stack and has almost never been joined to your traces. "
        "It usually lives with a different team than the traces do — start asking now."
    )
    out.append("2. **Human review of a sample.** Expensive, real, and sometimes the only option.")
    out.append(
        "3. **Throw the trace away.** Not every trace becomes a task. Low yield is a correct "
        "outcome, not a bug."
    )
    out.append("")
    out.append(
        f"- **Downstream signal available in your systems:** {TODO} (name the table/field, or "
        "say none)"
    )
    out.append(f"- **Join key from that system to a trace_id:** {TODO}")
    out.append(f"- **Lag before the signal is trustworthy:** {TODO} (e.g. 7 days for a reopen)")
    out.append("")

    # -- table --------------------------------------------------------------
    out.append("## 1. Task index")
    out.append("")
    out.append(
        "Edit the **include** column: `yes` trains on this task, `no` drops it. Anything left "
        f"as {TODO} is treated as **excluded** — silence never counts as approval."
    )
    out.append("")
    out.append("| task_id | include | trace | label in export | warnings | instruction |")
    out.append("|---|---|---|---|---|---|")
    for task in tasks:
        warnings = task.provenance.get("solvability_warnings") or ()
        label = {True: "pass", False: "fail", None: "unlabeled"}[task.outcome]
        out.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                _cell(task.task_id),
                _cell(_default_include(task)),
                _cell(task.trace_id),
                _cell(label),
                _cell("; ".join(warnings) or "-"),
                _cell(task.instruction[:110]),
            )
        )
    out.append("")

    if skipped:
        out.append("### Refused by mining")
        out.append("")
        for item in skipped:
            out.append(f"- `{item.get('trace_id') or item.get('task_id')}` — {item.get('reason')}")
        out.append("")

    # -- per task -----------------------------------------------------------
    out.append("## 2. Task detail")
    out.append("")
    for task in tasks:
        prov = task.provenance
        warnings = prov.get("solvability_warnings") or ()
        blocked = prov.get("blocked_post_state_reads") or ()
        out.append(f"### {task.task_id}")
        out.append("")
        out.append(f"- **Instruction:** {task.instruction}")
        out.append(f"- **Source trace:** `{task.trace_id}` (digest `{prov.get('source_digest', '?')}`)")
        out.append(f"- **Tools used:** {', '.join(task.tools) or '_none_'}")
        out.append(
            "- **Label in export:** "
            + {True: "pass", False: "fail", None: "unlabeled"}[task.outcome]
            + " — this is what the export claimed, not a downstream outcome"
        )
        out.append(
            f"- **Downstream signal for this task:** {TODO} (the real label: which system, "
            "which field, what value means success)"
        )
        starting = ", ".join(f"{r.entity}×{len(r.rows)}" for r in task.pre_state) or "_empty_"
        out.append(f"- **Starting state:** {starting}")
        if prov.get("first_write_step") is None:
            out.append("- **First write:** none — this episode is read-only")
        else:
            out.append(f"- **First write:** step {prov.get('first_write_step')}")
        if blocked:
            out.append(
                f"- **Post-write reads excluded from the starting state:** {len(blocked)} "
                "(they describe the world *after* the change and must not leak backward)"
            )
        if prov.get("partial_pre_state_rows"):
            out.append(
                f"- **Partial rows:** {prov['partial_pre_state_rows']} — production named these "
                "ids in a list but never read them, so only some fields are known"
            )
        for warning in warnings:
            out.append(f"- **Solvability warning:** {warning}")
            out.append(
                f"- **Warning resolved:** {TODO} (an unsolvable task makes every rollout fail "
                "identically, which at the pass@k gate is indistinguishable from 'too hard')"
            )
        out.append("")

    out.append("## 3. Tasks the traces do not contain")
    out.append("")
    out.append(
        "Production is thin exactly where the training signal is: failure paths. Now that the "
        "environment executes, you are no longer limited to what happened. Make the order "
        "non-refundable, the customer unverified, the API rate-limited — situations that never "
        "occurred but are entirely reachable in the real system."
    )
    out.append("")
    out.append(f"- **Situations worth generating that production never produced:** {TODO}")
    out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# rendering: VERIFIER.md
# ---------------------------------------------------------------------------


def _assertion_line(assertion: Assertion) -> str:
    kind = assertion.kind.value
    if assertion.tool:
        target = f"effects[{assertion.tool}]"
    elif assertion.entity and assertion.field:
        target = f"{assertion.entity}[{_fmt(assertion.row_key)}].{assertion.field}"
    elif assertion.entity:
        target = f"{assertion.entity}[{_fmt(assertion.row_key)}]" if assertion.row_key else assertion.entity
    else:
        target = "?"
    return f"`{kind}` · `{target}` == `{_fmt(assertion.expected)}`"


def verifier_md(
    verifiers: Sequence[Verifier],
    *,
    name: str = "environment",
    skipped: Sequence[JsonObject] = (),
) -> str:
    """Render VERIFIER.md from synthesized Verifiers."""
    out: list[str] = []
    out.append(f"# VERIFIER — {name}")
    out.append("")
    out.append(
        "Reward is code: assertions over the final state and the effect ledger. Never a judge. "
        "Each verifier below was synthesized from the state diff of one trace labeled correct."
    )
    out.append("")
    out.append(
        "**A generated reward function nobody has read is an unexamined reward function.** "
        "`reviewed_by` is unset on every verifier here, and `bandits.verify.evaluate` raises "
        "`UnreviewedVerifierError` rather than grading with it. Signing is not a formality — it "
        "is the only thing standing between a wrong assertion and a model trained on it."
    )
    out.append("")
    out.append("## 0. What to look for")
    out.append("")
    out.append(
        "- **Does it assert what must stay the same?** If a verifier only checks the field that "
        "changed, the agent learns that collateral damage is free."
    )
    out.append(
        "- **Is the expected value right, or just what happened?** These came from one episode. "
        "If production refunded the wrong amount, the assertion now encodes that."
    )
    out.append(
        "- **Are effects asserted?** For many agents success lives entirely in effects — an "
        "email sent, a charge attempted — not in stored data."
    )
    out.append(
        "- **Is anything unverifiable?** If a real success condition cannot be written as an "
        "assertion, say so here. We do not reach for a judge to cover the gap; the task either "
        "gets a narrower reward or gets dropped."
    )
    out.append("")

    out.append("## 1. Verifiers")
    out.append("")
    if not verifiers:
        out.append("_None synthesized._")
        out.append("")
    for verifier in verifiers:
        out.append(f"### {verifier.verifier_id}")
        out.append("")
        out.append(f"- **Task:** {verifier.task_id}")
        out.append(f"- **Assertions:** {len(verifier.assertions)}")
        out.append("")
        for assertion in verifier.assertions:
            out.append(f"  - {_assertion_line(assertion)}")
            if assertion.description:
                out.append(f"    - {assertion.description}")
        out.append("")
        out.append(f"- **Assertions correct:** {TODO} (yes / no + which one is wrong)")
        out.append(f"- **Missing invariants:** {TODO} (what must stay the same and is not asserted)")
        out.append(f"- **Unverifiable in code:** {TODO} (say so rather than adding a judge)")
        out.append(f"- **Reviewed by:** {TODO}")
        out.append("")
        out.append(
            "  Replace the marker with your name. Until you do, this verifier grades nothing."
        )
        out.append("")

    if skipped:
        out.append("## 2. Tasks we refused to write a verifier for")
        out.append("")
        for item in skipped:
            out.append(f"- `{item.get('task_id')}` — {item.get('reason')}")
        out.append("")
        out.append(
            "These are refusals, not failures. A verifier synthesized from a failed or unlabeled "
            "trajectory would make that trajectory the training target."
        )
        out.append("")

    out.append("## 3. Anti-cheat")
    out.append("")
    out.append(
        "A rebuilt world is simpler than reality, so it admits strategies reality does not. "
        "`bandits.verify.check_rollout` fails a rollout that writes the store directly, reads "
        "the verifier or the effect ledger from inside the episode, or touches the network."
    )
    out.append("")
    out.append(f"- **Exploits specific to this domain:** {TODO} (a shortcut in your world that "
               "a generic check would miss)")
    out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# scaffolding
# ---------------------------------------------------------------------------


def scaffold_workspace(
    workspace: str | Path,
    *,
    surface: ToolSurface,
    schema: StateSchema,
    tasks: Sequence[TaskCase] = (),
    verifiers: Sequence[Verifier] = (),
    fidelity: FidelityReport | None = None,
    skipped_tasks: Sequence[JsonObject] = (),
    skipped_verifiers: Sequence[JsonObject] = (),
    name: str = "environment",
    overwrite: bool = False,
) -> WorkspacePaths:
    """Write ENVIRONMENT.md, TASKS.md and VERIFIER.md into ``workspace``.

    Existing files are never clobbered unless ``overwrite=True`` — the whole
    point of the workspace is that a human edits it, and a re-scaffold that
    silently threw those edits away would be the worst possible failure mode.
    """
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    paths = WorkspacePaths(
        environment=root / "ENVIRONMENT.md",
        tasks=root / "TASKS.md",
        verifier=root / "VERIFIER.md",
    )
    bodies = {
        paths.environment: environment_md(surface, schema, name=name, fidelity=fidelity),
        paths.tasks: tasks_md(tasks, name=name, skipped=skipped_tasks),
        paths.verifier: verifier_md(verifiers, name=name, skipped=skipped_verifiers),
    }
    for path, body in bodies.items():
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists and may carry human edits. "
                "Pass overwrite=True only if you are certain."
            )
        path.write_text(body, encoding="utf-8")
    return paths


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


@dataclass
class _Doc:
    """A parsed markdown file: ordered tables plus fields bound to sections."""

    tables: list[list[dict[str, str]]] = field(default_factory=list)
    sections: dict[str, dict[str, str]] = field(default_factory=dict)
    section_order: list[str] = field(default_factory=list)
    toplevel: dict[str, str] = field(default_factory=dict)


def _parse_markdown(text: str) -> _Doc:
    doc = _Doc()
    section: str | None = None
    header: list[str] | None = None
    rows: list[dict[str, str]] = []

    def flush() -> None:
        nonlocal header, rows
        if header is not None and rows:
            doc.tables.append(rows)
        header, rows = None, []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [_uncell(c) for c in stripped.strip("|").split("|")]
            if _SEP_RE.match(line):
                continue
            if header is None:
                header = [c.strip().lower() for c in cells]
            else:
                padded = list(cells) + [""] * (len(header) - len(cells))
                rows.append(dict(zip(header, padded, strict=False)))
            continue
        flush()
        heading = _HEADING_RE.match(line)
        if heading:
            if len(heading["hashes"]) >= 3:
                section = heading["text"].strip().strip("`")
                doc.sections.setdefault(section, {})
                if section not in doc.section_order:
                    doc.section_order.append(section)
            else:
                section = None
            continue
        field_line = _FIELD_RE.match(line)
        if field_line:
            key = field_line["key"].strip().lower()
            value = field_line["value"].strip()
            if section is None:
                doc.toplevel.setdefault(key, value)
            else:
                doc.sections[section].setdefault(key, value)
    flush()
    return doc


def _find_table(doc: _Doc, *required: str) -> list[dict[str, str]]:
    for rows in doc.tables:
        if rows and all(col in rows[0] for col in required):
            return rows
    return []


def read_back(workspace: str | Path) -> WorkspaceOverrides:
    """Parse the edited workspace back into structured overrides.

    Missing files are tolerated (you may only have written ENVIRONMENT.md yet);
    unanswered questions are collected rather than defaulted.
    """
    root = Path(workspace)
    env_doc = _read_doc(root / "ENVIRONMENT.md")
    task_doc = _read_doc(root / "TASKS.md")
    ver_doc = _read_doc(root / "VERIFIER.md")

    open_questions: list[OpenQuestion] = []
    issues: list[str] = []

    # -- tool classes -------------------------------------------------------
    tool_classes: dict[str, ToolClass] = {}
    inferred: dict[str, str] = {}
    valid = {c.value for c in ToolClass}
    for row in _find_table(env_doc, "tool", "decision"):
        tool = row.get("tool", "").strip().strip("`")
        if not tool:
            continue
        was = row.get("inferred", "").strip().lower()
        inferred[tool] = was
        decision = row.get("decision", "")
        if is_undecided(decision):
            open_questions.append(
                OpenQuestion("ENVIRONMENT.md", tool, "tool class is undecided")
            )
            continue
        chosen = decision.strip().strip("`*_ ").lower()
        if chosen not in valid:
            issues.append(
                f"ENVIRONMENT.md: tool {tool!r} has decision {decision!r}, "
                f"which is not one of {sorted(valid)}"
            )
            continue
        if chosen != was:
            tool_classes[tool] = ToolClass(chosen)

    # -- blind writes -------------------------------------------------------
    blind: list[str] = []
    for section, fields in env_doc.sections.items():
        if not section.startswith("tool:"):
            continue
        tool = section[len("tool:"):].strip().strip("`")
        answer = fields.get("confirmed read-only")
        verdict = _yes_no(answer or "")
        if verdict is None:
            open_questions.append(
                OpenQuestion("ENVIRONMENT.md", tool, "blind-write check unanswered")
            )
        elif verdict is False:
            blind.append(tool)
            effective = tool_classes.get(tool, ToolClass(inferred.get(tool, "read")))
            if effective is ToolClass.READ:
                issues.append(
                    f"ENVIRONMENT.md: {tool!r} is declared a blind write but its decision cell "
                    "still says 'read'. Change the table too — the pipeline reads the table."
                )

    # -- entity / environment acknowledgements ------------------------------
    accepted_static: list[str] = []
    accepted_unsupported: list[str] = []
    for section, fields in env_doc.sections.items():
        if section.startswith("tool:"):
            continue
        if "acceptable as a snapshot" in fields:
            verdict = _yes_no(fields["acceptable as a snapshot"])
            if verdict is None:
                open_questions.append(
                    OpenQuestion("ENVIRONMENT.md", section, "static snapshot not accepted or rejected")
                )
            elif verdict:
                accepted_static.append(section)
        if "write semantics correct" in fields and _yes_no(fields["write semantics correct"]) is None:
            open_questions.append(
                OpenQuestion("ENVIRONMENT.md", section, "write semantics not confirmed")
            )
        unattributed = fields.get("what state do these read or write")
        if unattributed is not None and is_undecided(unattributed):
            open_questions.append(
                OpenQuestion("ENVIRONMENT.md", section, "unattributed tools not explained")
            )
    for key, value in env_doc.toplevel.items():
        if is_undecided(value):
            open_questions.append(OpenQuestion("ENVIRONMENT.md", "environment", key))
        elif key.startswith("remaining gaps"):
            accepted_unsupported.append(value)

    # -- tasks --------------------------------------------------------------
    included: list[str] = []
    excluded: list[str] = []
    undecided: list[str] = []
    for row in _find_table(task_doc, "task_id", "include"):
        task_id = row.get("task_id", "").strip().strip("`")
        if not task_id:
            continue
        verdict = _yes_no(row.get("include", ""))
        if verdict is None:
            undecided.append(task_id)
            open_questions.append(OpenQuestion("TASKS.md", task_id, "inclusion undecided"))
        elif verdict:
            included.append(task_id)
        else:
            excluded.append(task_id)

    label_sources: dict[str, str] = {}
    for section, fields in task_doc.sections.items():
        value = fields.get("downstream signal for this task")
        if value is None:
            continue
        if is_undecided(value):
            if section in included:
                open_questions.append(
                    OpenQuestion("TASKS.md", section, "no downstream outcome label named")
                )
        else:
            label_sources[section] = value
        warning = fields.get("warning resolved")
        if warning is not None and _yes_no(warning) is not True and section in included:
            open_questions.append(
                OpenQuestion("TASKS.md", section, "solvability warning unresolved but included")
            )
    for key, value in task_doc.toplevel.items():
        if is_undecided(value):
            open_questions.append(OpenQuestion("TASKS.md", "labels", key))

    # -- verifiers ----------------------------------------------------------
    reviewed: dict[str, str] = {}
    unreviewed: list[str] = []
    for section, fields in ver_doc.sections.items():
        if "reviewed by" not in fields:
            continue
        value = fields["reviewed by"]
        if is_undecided(value):
            unreviewed.append(section)
            open_questions.append(OpenQuestion("VERIFIER.md", section, "no reviewer signed off"))
        else:
            reviewed[section] = value
        for key in ("assertions correct", "missing invariants", "unverifiable in code"):
            if key in fields and is_undecided(fields[key]) and section in reviewed:
                open_questions.append(OpenQuestion("VERIFIER.md", section, key))
    for key, value in ver_doc.toplevel.items():
        if is_undecided(value):
            open_questions.append(OpenQuestion("VERIFIER.md", "anti-cheat", key))

    return WorkspaceOverrides(
        tool_classes=tool_classes,
        declared_blind_writes=tuple(sorted(blind)),
        included_tasks=tuple(included),
        excluded_tasks=tuple(excluded),
        undecided_tasks=tuple(undecided),
        label_sources=label_sources,
        reviewed_by=reviewed,
        unreviewed_verifiers=tuple(unreviewed),
        accepted_static_entities=tuple(accepted_static),
        accepted_unsupported_tools=tuple(accepted_unsupported),
        open_questions=tuple(open_questions),
        issues=tuple(issues),
    )


def _read_doc(path: Path) -> _Doc:
    if not path.exists():
        return _Doc()
    return _parse_markdown(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------


def apply_overrides(
    overrides: WorkspaceOverrides,
    *,
    surface: ToolSurface,
    tasks: Sequence[TaskCase] = (),
    verifiers: Sequence[Verifier] = (),
    reviewer_evidence: Mapping[str, str] | None = None,
) -> AppliedWorkspace:
    """Fold a human's decisions back into the pipeline artifacts.

    Three rules, all of them refusals:

    * a task is kept only when its ``Include`` cell says yes;
    * a verifier gets ``reviewed_by`` only when a person signed it, and is
      withheld entirely otherwise;
    * a human-overridden tool class replaces the inferred one and the evidence
      records who changed it, so the audit trail survives.
    """
    del reviewer_evidence  # reserved; the evidence line below is generated
    profiles: list[ToolProfile] = []
    for profile in surface.tools:
        override = overrides.tool_classes.get(profile.name)
        if override is None:
            profiles.append(profile)
            continue
        note = (
            f"human override in ENVIRONMENT.md: {profile.tool_class.value} -> {override.value}"
        )
        if profile.name in overrides.declared_blind_writes:
            note += " (declared a blind write: mutates without any follow-up read in the corpus)"
        profiles.append(
            profile.model_copy(
                update={
                    "tool_class": override,
                    "class_confidence": 1.0,
                    "class_evidence": profile.class_evidence + (note,),
                }
            )
        )
    new_surface = ToolSurface(tools=tuple(profiles))

    keep = set(overrides.included_tasks)
    kept_tasks = tuple(t for t in tasks if t.task_id in keep)
    dropped_tasks = tuple(t.task_id for t in tasks if t.task_id not in keep)

    kept_verifiers: list[Verifier] = []
    dropped_verifiers: list[str] = []
    for verifier in verifiers:
        reviewer = overrides.reviewed_by.get(verifier.verifier_id)
        if reviewer is None or verifier.task_id not in keep:
            dropped_verifiers.append(verifier.verifier_id)
            continue
        kept_verifiers.append(verifier.model_copy(update={"reviewed_by": reviewer}))

    return AppliedWorkspace(
        surface=new_surface,
        tasks=kept_tasks,
        verifiers=tuple(kept_verifiers),
        dropped_tasks=dropped_tasks,
        dropped_verifiers=tuple(dropped_verifiers),
    )
