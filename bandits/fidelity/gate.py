"""The accept/reject decision.

One number decides whether a rebuilt environment is allowed near training. This
module is where that number is turned into a verdict, and the verdict is
deliberately harder to pass than a single average.

The rule
--------
**Accept requires both:**

1. ``overall_rate >= threshold`` (default 0.90), and
2. **every tool that was actually replayed** is at or above ``per_tool_floor``
   (default 0.80).

Both, because either alone is unsafe. An aggregate hides a single broken tool:
in the README's example the overall rate is 93% while ``update_order_status``
sits at 38%, and an agent trained in that environment learns a status update is
a coin flip. PLAN.md Step 12 is explicit that per-tool is what gets reported --
"one tool at 40% divergence tells you exactly what to fix, an average tells you
nothing" -- so per-tool is also what gets enforced. Conversely a per-tool floor
alone would accept an environment where every tool sits at 81%, which is nine
wrong answers in fifty and no single line to point at.

``min_calls_for_floor`` exists because a floor over one observation is noise,
not evidence. It defaults to 1 (every tool is judged); raise it when a corpus
has a long tail of tools called once. Tools below it are still reported, still
counted in the overall rate, and named in ``notes`` so the exemption is visible
rather than silent.

What happens to unsupported calls
---------------------------------
**They stay in the denominator and never count as matched.** A tool the
environment refuses to reimplement is a hole in the model of the world, and the
one thing this gate must never do is get a better score by measuring less. If
unsupported calls were excluded, an environment that supports one tool perfectly
and refuses the other eight would report 100%.

They are nevertheless reported on their own column and called out in ``notes``,
because the fix is different: unsupported sends you back to schema inference or
to writing an explicit rule (PLAN.md Step 7), mismatched sends you to the tool
implementation. ``ToolFidelity.rate`` is ``matched / replayed``, so a tool that
is entirely unsupported reports 0% and trips the floor on its own.

Nothing here is a partial credit scheme. A call either reproduced the recorded
observation or it did not.
"""

from __future__ import annotations

from collections.abc import Sequence

from bandits.contracts import FidelityReport, ToolFidelity

from .replay import DEFAULT_MAX_EXAMPLES, ReplayResult, per_tool_fidelity

__all__ = [
    "DEFAULT_PER_TOOL_FLOOR",
    "DEFAULT_THRESHOLD",
    "GateCriteria",
    "build_report",
    "gate",
]

DEFAULT_THRESHOLD = 0.9
"""Overall agreement below which an environment is not a model of anything."""

DEFAULT_PER_TOOL_FLOOR = 0.8
"""No single tool may fall below this, however good the aggregate looks."""


class GateCriteria:
    """The thresholds, bundled so a CLI and a test can pass the same object."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        per_tool_floor: float = DEFAULT_PER_TOOL_FLOOR,
        min_calls_for_floor: int = 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if not 0.0 <= per_tool_floor <= 1.0:
            raise ValueError(f"per_tool_floor must be in [0, 1], got {per_tool_floor}")
        if min_calls_for_floor < 1:
            raise ValueError(f"min_calls_for_floor must be >= 1, got {min_calls_for_floor}")
        self.threshold = threshold
        self.per_tool_floor = per_tool_floor
        self.min_calls_for_floor = min_calls_for_floor


def build_report(
    results: Sequence[ReplayResult],
    *,
    criteria: GateCriteria | None = None,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    env_id: str | None = None,
    trace_id: str | None = None,
) -> FidelityReport:
    """Turn one or many replays into a :class:`FidelityReport` with a verdict.

    A single replay produces a report identified by its own env and trace. Many
    replays merge into a corpus-level report; ``FidelityReport`` holds one
    ``env_id``/``trace_id`` pair, so the merged report uses ``"corpus"`` and
    ``"*"`` and names the traces it covers in ``notes``. That substitution is
    stated here rather than hidden -- the contract has no field for a set of
    environments and it is frozen.
    """
    criteria = criteria or GateCriteria()
    per_tool = per_tool_fidelity(results, max_examples=max_examples)
    replayed = sum(t.replayed for t in per_tool)
    matched = sum(t.matched for t in per_tool)
    overall = matched / replayed if replayed else 0.0

    if len(results) == 1:
        report_env = env_id or results[0].env_id
        report_trace = trace_id or results[0].trace_id
    else:
        report_env = env_id or "corpus"
        report_trace = trace_id or "*"

    notes = _notes(results, per_tool, criteria, overall, replayed, matched)
    accepted = _accepted(per_tool, criteria, overall, replayed)
    return FidelityReport(
        env_id=report_env,
        trace_id=report_trace,
        per_tool=per_tool,
        overall_rate=overall,
        accepted=accepted,
        threshold=criteria.threshold,
        notes=tuple(notes),
    )


#: Alias. ``gate(results)`` reads better at a call site than ``build_report``.
gate = build_report


def _judged(per_tool: Sequence[ToolFidelity], criteria: GateCriteria) -> list[ToolFidelity]:
    return [t for t in per_tool if t.replayed >= criteria.min_calls_for_floor]


def _accepted(
    per_tool: Sequence[ToolFidelity],
    criteria: GateCriteria,
    overall: float,
    replayed: int,
) -> bool:
    if replayed == 0:
        return False
    if overall < criteria.threshold:
        return False
    return all(t.rate >= criteria.per_tool_floor for t in _judged(per_tool, criteria))


def _notes(
    results: Sequence[ReplayResult],
    per_tool: Sequence[ToolFidelity],
    criteria: GateCriteria,
    overall: float,
    replayed: int,
    matched: int,
) -> list[str]:
    """Human-readable reasons. A rejection must always say what to fix."""
    notes: list[str] = []
    if len(results) > 1:
        notes.append(
            f"merged {len(results)} replays covering traces: "
            + ", ".join(sorted(r.trace_id for r in results))
        )
    notes.append(
        "unsupported calls are counted in the denominator and never as matched; "
        "excluding them would flatter the score by measuring less"
    )

    if replayed == 0:
        notes.append("REJECTED: nothing was replayed, so there is no evidence of fidelity at all")
        return notes

    if overall < criteria.threshold:
        notes.append(
            f"REJECTED: overall {matched}/{replayed} = {overall:.1%} is below the "
            f"{criteria.threshold:.0%} threshold"
        )

    below = [t for t in _judged(per_tool, criteria) if t.rate < criteria.per_tool_floor]
    for tool in sorted(below, key=lambda t: (t.rate, t.tool)):
        notes.append(
            f"REJECTED: {tool.tool} at {tool.rate:.0%} ({tool.matched}/{tool.replayed}) is below "
            f"the {criteria.per_tool_floor:.0%} per-tool floor "
            f"[{tool.mismatched} mismatched, {tool.unsupported} unsupported]"
        )

    exempt = [t for t in per_tool if t.replayed < criteria.min_calls_for_floor]
    for tool in exempt:
        notes.append(
            f"NOT JUDGED against the per-tool floor: {tool.tool} was replayed only "
            f"{tool.replayed} time(s), below min_calls_for_floor="
            f"{criteria.min_calls_for_floor}. It still counts toward the overall rate."
        )

    unsupported = [t for t in per_tool if t.unsupported]
    for tool in unsupported:
        notes.append(
            f"UNSUPPORTED: {tool.tool} refused {tool.unsupported}/{tool.replayed} call(s) - "
            f"the environment could not model it, which is a different fix from modeling it wrong"
        )

    for result in results:
        for note in result.notes:
            if note not in notes:
                notes.append(f"{result.trace_id}: {note}")

    if not any(n.startswith("REJECTED") for n in notes):
        notes.append(
            f"ACCEPTED: overall {matched}/{replayed} = {overall:.1%} >= "
            f"{criteria.threshold:.0%} and every judged tool is at or above the "
            f"{criteria.per_tool_floor:.0%} floor"
        )
    return notes
