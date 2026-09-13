"""Render a mining session so a person can judge it, not just count it.

The problem this solves. A taxonomy printed as "14 contracts, 143 assigned" is
unreviewable: it says a grouping happened and nothing about whether the grouping
is any good. The only way to know that is to read what a family claims, see the
requests it claims, and see the requests it nearly claimed and rejected — which
is exactly what a list of ids cannot show.

So a family renders as a card: its definition, the outcome a verifier would have
to establish, real member requests, and its borderline cases. A reviewer reading
one should be able to say "that is one task" or "those are two tasks" without
opening the corpus.

Everything here is presentation over artifacts that already exist. Nothing in
this module decides anything, and a rendering bug can mislead a reader but can
never change a taxonomy.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bandits.analyze.rlm_audit_session import AuditSessionState
from bandits.analyze.rlm_corpus import ReadOnlyCorpus
from bandits.analyze.rlm_models import FamilyContract, RLMClusteringAudit, RLMClusteringRun
from bandits.analyze.rlm_session import SessionState

_EXCERPT = 100
"""How much of a request a card shows. Enough to recognize the task, short
enough that a dozen fit on a screen — a card nobody scrolls is a card nobody
reads."""


def _excerpt(text: str, width: int = _EXCERPT) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _requests(corpus: ReadOnlyCorpus | None, trace_ids: Sequence[str]) -> list[tuple[str, str]]:
    """The opening request of each trace, for display beside its id.

    Always the *first* message, even under the full-trajectory view: a card is
    for recognizing what was asked, and the agent's subsequent actions are not
    that. Falls back to the bare id when no corpus is supplied, so a card still
    renders from an artifact alone.
    """
    if corpus is None:
        return [(trace_id, "") for trace_id in trace_ids]
    rows = []
    for trace_id in trace_ids:
        try:
            view = corpus.get_user_messages(trace_id)
        except KeyError:
            rows.append((trace_id, "[unknown trace]"))
            continue
        first = next((m for m in view.messages if not m.startswith("[tool")), "")
        rows.append((trace_id, _excerpt(first.removeprefix("[user] "))))
    return rows


def family_card(
    contract: FamilyContract,
    *,
    members: Sequence[str] = (),
    corpus: ReadOnlyCorpus | None = None,
    audit: RLMClusteringAudit | None = None,
    max_examples: int = 4,
) -> Panel:
    """One family, rendered as something a reviewer can accept or reject.

    The required outcome is given its own line rather than folded in with the
    definition, because it is the field that decides membership: two families
    with similar definitions and different outcomes are two families, and a card
    that buried the outcome would make that distinction invisible.
    """
    body: list[object] = []

    body.append(Text("Definition", style="bold"))
    body.append(Text(f"  {contract.definition}"))
    body.append(Text(""))
    body.append(Text("Required outcome", style="bold"))
    for line in contract.required_outcome_shape:
        body.append(Text(f"  • {line}"))

    if contract.inclusion_rules:
        body.append(Text(""))
        body.append(Text("Includes", style="bold"))
        for rule in contract.inclusion_rules:
            body.append(Text(f"  • {rule}", style="green"))
    if contract.exclusion_rules:
        body.append(Text(""))
        body.append(Text("Excludes", style="bold"))
        for rule in contract.exclusion_rules:
            body.append(Text(f"  • {rule}", style="red"))

    if members:
        body.append(Text(""))
        body.append(Text(f"Members ({len(members)})", style="bold"))
        for trace_id, request in _requests(corpus, list(members)[:max_examples]):
            body.append(Text(f"  {trace_id}  {request}", style="dim"))
        if len(members) > max_examples:
            body.append(Text(f"  … {len(members) - max_examples} more", style="dim"))

    # The counterexamples are the most informative half of a contract: a
    # boundary is only legible from the near misses it excludes.
    if contract.counterexample_trace_ids:
        body.append(Text(""))
        body.append(Text("Counterexamples", style="bold"))
        for trace_id, request in _requests(corpus, contract.counterexample_trace_ids[:3]):
            body.append(Text(f"  {trace_id}  {request}", style="yellow"))

    status = "needs review"
    border = "cyan"
    finding = None
    if audit is not None:
        finding = next((f for f in audit.findings if f.contract_id == contract.contract_id), None)
    if finding is not None:
        status = f"audit: {finding.recommendation}"
        border = {"keep": "green", "revise": "yellow", "split": "red"}.get(
            finding.recommendation, "yellow"
        )
        body.append(Text(""))
        body.append(Text("Audit", style="bold"))
        body.append(Text(f"  {finding.rationale}", style=border))
        if finding.topical_only:
            body.append(
                Text(
                    "  flagged as a topical grouping: members may need different verifiers",
                    style="red",
                )
            )
        if finding.least_compatible_pair:
            left, right = finding.least_compatible_pair
            body.append(Text(f"  least compatible: {left} vs {right}", style="dim"))

    return Panel(
        Group(*body),
        title=f"[bold]{contract.name}[/bold]  [dim]{contract.contract_id}[/dim]",
        subtitle=f"[dim]{status} · revision {contract.revision}[/dim]",
        border_style=border,
    )


def taxonomy_overview(
    contracts: Sequence[FamilyContract],
    *,
    members: dict[str, tuple[str, ...]] | None = None,
    audit: RLMClusteringAudit | None = None,
) -> Table:
    """Every family at a glance, biggest first, with what the audit said."""
    table = Table("family", "name", "members", "outcome", "audit")
    findings = {f.contract_id: f for f in (audit.findings if audit else ())}
    ordered = sorted(
        contracts,
        key=lambda c: -len((members or {}).get(c.contract_id, ())),
    )
    for contract in ordered:
        finding = findings.get(contract.contract_id)
        verdict = "—"
        if finding is not None:
            colour = {"keep": "green", "revise": "yellow", "split": "red"}.get(
                finding.recommendation, "yellow"
            )
            verdict = f"[{colour}]{finding.recommendation}[/{colour}]"
            if finding.topical_only:
                verdict += " [red]topical[/red]"
        table.add_row(
            contract.contract_id,
            contract.name,
            str(len((members or {}).get(contract.contract_id, ()))),
            _excerpt(contract.required_outcome_shape[0], 46),
            verdict,
        )
    return table


def live_panel(state: SessionState, *, recent: Sequence[dict] = ()) -> Panel:
    """The whole run in one screen, for watching it happen.

    Progress, spend and the taxonomy as it currently stands, together: a
    progress bar with no taxonomy beside it tells you a run is working without
    telling you whether it is working *well*, and the second question is the one
    worth watching for.
    """
    status_colour = {
        "running": "cyan",
        "awaiting_review": "green",
        "failed": "red",
        "incomplete": "yellow",
    }.get(state.status, "white")

    header = Table.grid(padding=(0, 2))
    header.add_column(style="dim")
    header.add_column()
    header.add_row("status", f"[{status_colour}]{state.status}[/{status_colour}]")
    header.add_row(
        "pass",
        f"{state.pass_index + 1} of {state.requested_passes} ({state.completed_passes} complete)",
    )
    header.add_row("traces", _bar(state.traces_seen_this_pass, state.traces_total))
    header.add_row("chunks", str(state.chunk_index))
    header.add_row("assigned", str(len(state.assignments)))
    header.add_row(
        "unresolved",
        f"{len(state.ambiguous_trace_ids)} ambiguous, {len(state.uncovered_trace_ids)} uncovered",
    )
    header.add_row("spend", f"{state.llm_calls} calls · ${state.cost_usd:.4f}")

    body: list[object] = [header]

    if state.contracts:
        body.append(Text(""))
        table = Table("family", "name", "members", box=None, pad_edge=False)
        counts: dict[str, int] = {}
        for contract_id in state.assignments.values():
            counts[contract_id] = counts.get(contract_id, 0) + 1
        for contract in sorted(state.contracts, key=lambda c: -counts.get(c.contract_id, 0)):
            table.add_row(
                contract.contract_id,
                _excerpt(contract.name, 34),
                str(counts.get(contract.contract_id, 0)),
            )
        body.append(table)
    else:
        body.append(Text("\nno contracts proposed yet", style="dim"))

    if recent:
        body.append(Text(""))
        for event in recent:
            body.append(Text(f"  {_event_line(event)}", style="dim"))

    if state.last_error:
        body.append(Text(f"\nlast error: {state.last_error}", style="red"))

    return Panel(
        Group(*body),
        title=f"[bold]{state.session_id}[/bold]",
        subtitle=f"[dim]{state.view.value} · seed {state.seed}[/dim]",
        border_style=status_colour,
    )


def audit_live_panel(state: AuditSessionState, *, recent: Sequence[dict] = ()) -> Panel:
    """The whole audit session in one screen, for watching it happen.

    Deliberately its own function rather than a branch inside ``live_panel``:
    an audit session has no passes, chunks or taxonomy-under-construction to
    show, only a fixed contract list and the findings reached against it so
    far, so most of ``live_panel``'s fields would be meaningless here.
    """
    status_colour = {
        "running": "cyan",
        "awaiting_review": "green",
        "interrupted": "yellow",
        "failed": "red",
        "incomplete": "yellow",
    }.get(state.status, "white")

    header = Table.grid(padding=(0, 2))
    header.add_column(style="dim")
    header.add_column()
    header.add_row("status", f"[{status_colour}]{state.status}[/{status_colour}]")
    header.add_row(
        "contracts", _bar(len(state.completed_contract_ids), len(state.contract_order))
    )
    header.add_row("spend", f"{state.llm_calls} calls · ${state.cost_usd:.4f}")
    if state.resumed_from:
        header.add_row("resumed from", state.resumed_from)

    body: list[object] = [header]

    if state.findings:
        body.append(Text(""))
        table = Table("contract", "verdict", box=None, pad_edge=False)
        for finding in state.findings[-12:]:
            colour = "yellow" if finding.demands_action else "dim"
            table.add_row(
                finding.contract_id, f"[{colour}]{finding.recommendation}[/{colour}]"
            )
        body.append(table)
    else:
        body.append(Text("\nno findings yet", style="dim"))

    if recent:
        body.append(Text(""))
        for event in recent:
            body.append(Text(f"  {_audit_event_line(event)}", style="dim"))

    if state.last_error:
        body.append(Text(f"\nlast error: {state.last_error}", style="red"))

    return Panel(
        Group(*body),
        title=f"[bold]{state.session_id}[/bold]",
        subtitle=f"[dim]auditing {state.run_id}[/dim]",
        border_style=status_colour,
    )


def _audit_event_line(event: dict) -> str:
    name = event.get("event", "?")
    if name == "contract_audited":
        return (
            f"{event.get('contract_id')}: {event.get('recommendation')} "
            f"({event.get('done')}/{event.get('of')})"
        )
    if name == "session_started":
        return f"started · {event.get('contracts')} contract(s) to audit"
    if name == "session_finished":
        return f"finished · {event.get('status')} ({event.get('stop_reason')})"
    if name == "session_failed":
        return f"failed: {event.get('error', '')[:60]}"
    return name


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "—"
    filled = round(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def _event_line(event: dict) -> str:
    name = event.get("event", "?")
    if name == "chunk_complete":
        operations = ", ".join(event.get("operations") or []) or "no change"
        return (
            f"pass {event.get('pass', 0) + 1} chunk {event.get('chunk')}: "
            f"{event.get('traces')} traces, {operations}"
            + (
                f" [failed: {event.get('error', '')[:40]}]"
                if event.get("status") == "error"
                else ""
            )
        )
    if name == "session_finished":
        return f"finished: {event.get('stop_reason')} ({event.get('completed_passes')} passes)"
    if name == "session_failed":
        return f"failed: {event.get('error', '')[:60]}"
    if name == "session_started":
        return f"started: {event.get('traces')} traces, {event.get('passes')} passes"
    return name


def print_pass_history(run: RLMClusteringRun, console: Console) -> None:
    """What each pass changed, so a reviewer can see the taxonomy settling.

    The question at a pause is whether the second look agreed with the first,
    and that is a per-pass fact: a run whose second pass moved a third of its
    traces has not settled, however tidy the final list of families looks.
    """
    if not run.passes:
        return
    table = Table("pass", "traces", "operations", "moved", "families after")
    for result in run.passes:
        table.add_row(
            str(result.pass_index + 1) + ("" if result.complete else " [yellow](partial)[/yellow]"),
            str(len(result.trace_ids)),
            str(len(result.operations)) or "—",
            f"{len(result.reassigned_trace_ids)} ({result.churn:.0%})",
            str(len(result.contracts_after)),
        )
    console.print(table)
