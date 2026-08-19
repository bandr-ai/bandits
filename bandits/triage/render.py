"""Rendering a :class:`TriageReport` for a human.

Same discipline as the fidelity table: signals and tools listed individually,
the verdict last. This page is often the first artifact a customer sees, and it
has one job -- tell them what their telemetry can and cannot support, in terms
they can act on without taking our word for anything.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from bandits.triage.assess import TriageReport, Verdict

__all__ = ["render_report"]

_VERDICT_STYLE = {
    Verdict.GO: "bold green",
    Verdict.PARTIAL: "bold yellow",
    Verdict.NO_GO: "bold red",
}


def render_report(report: TriageReport, console: Console | None = None) -> None:
    """Print the signal table, the per-tool table, and the verdict."""
    console = console or Console()

    signals = Table(
        title=f"triage · {report.source} · {report.traces} traces · {report.invocations} invocation points",
        title_justify="left",
        show_edge=False,
        pad_edge=False,
    )
    signals.add_column("signal", style="cyan", no_wrap=True)
    signals.add_column("", justify="right", no_wrap=True)
    signals.add_column("observed", justify="right")
    signals.add_column("what it buys", style="dim")

    for signal in report.signals:
        if signal.present:
            mark, style = "yes", "green"
        elif signal.blocking:
            mark, style = "NO", "bold red"
        else:
            mark, style = "no", "yellow"
        signals.add_row(
            signal.name,
            Text(mark, style=style),
            f"{signal.observed}/{signal.population}",
            signal.detail,
        )
    console.print(signals)

    if report.tools:
        console.print()
        tools = Table(show_edge=False, pad_edge=False)
        tools.add_column("tool", style="cyan", no_wrap=True)
        tools.add_column("calls", justify="right")
        tools.add_column("args", justify="right")
        tools.add_column("responses", justify="right")
        tools.add_column("errors", justify="right")
        tools.add_column("ids", style="dim")
        tools.add_column("note", style="dim")
        for tool in report.tools:
            tools.add_row(
                tool.tool,
                str(tool.calls),
                str(tool.with_arguments),
                str(tool.with_object_response),
                str(tool.error_calls) if tool.error_calls else "-",
                ", ".join(tool.identifier_fields) or "-",
                Text(tool.note, style="green" if tool.reconstructible else "yellow"),
            )
        console.print(tools)

    if report.issue_counts:
        console.print()
        listed = ", ".join(f"{kind} x{count}" for kind, count in sorted(report.issue_counts.items()))
        console.print(f"[dim]ingest issues:[/dim] {listed}")

    console.print()
    console.print(Text(report.verdict.value, style=_VERDICT_STYLE[report.verdict]))
    for reason in report.reasons:
        console.print(f"  {reason}")
