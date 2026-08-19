"""Rendering a :class:`FidelityReport` for a human and for a machine.

The terminal table is the artifact a customer's engineer looks at, so it is
per-tool first and the aggregate last -- the opposite of the usual dashboard,
and on purpose. The single overall number is the least useful line on the page;
it is there so a CI job has something to print, not so anyone reads it instead
of the rows above it.
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table
from rich.text import Text

from bandits.contracts import FidelityReport, JsonObject

__all__ = ["render", "render_str", "to_json", "to_json_str"]


def _rate_style(rate: float, floor: float) -> str:
    if rate >= 0.99:
        return "green"
    if rate < floor:
        return "bold red"
    return "yellow"


def render(report: FidelityReport, console: Console | None = None, *, floor: float = 0.8) -> None:
    """Print the per-tool table, the overall line, and the verdict."""
    console = console or Console()

    table = Table(
        title=f"fidelity · {report.env_id} · {report.trace_id}",
        title_justify="left",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("tool", style="cyan", no_wrap=True)
    table.add_column("matched", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("mismatched", justify="right")
    table.add_column("unsupported", justify="right")
    table.add_column("note", style="dim")

    for tool in report.per_tool:
        note = ""
        if tool.unsupported:
            note = "unsupported: could not be modeled"
        elif tool.mismatched:
            note = _first_reason(tool.examples)
        table.add_row(
            tool.tool,
            f"{tool.matched}/{tool.replayed}",
            Text(f"{tool.rate:6.0%}", style=_rate_style(tool.rate, floor)),
            str(tool.mismatched) if tool.mismatched else "-",
            str(tool.unsupported) if tool.unsupported else "-",
            note,
        )

    replayed = sum(t.replayed for t in report.per_tool)
    matched = sum(t.matched for t in report.per_tool)
    mismatched = sum(t.mismatched for t in report.per_tool)
    unsupported = sum(t.unsupported for t in report.per_tool)
    table.add_section()
    table.add_row(
        Text("overall", style="bold"),
        Text(f"{matched}/{replayed}", style="bold"),
        Text(f"{report.overall_rate:6.0%}", style=_rate_style(report.overall_rate, report.threshold)),
        str(mismatched) if mismatched else "-",
        str(unsupported) if unsupported else "-",
        "",
    )

    console.print(table)
    verdict = (
        Text("ACCEPTED", style="bold green")
        if report.accepted
        else Text("REJECTED", style="bold red")
    )
    console.print(verdict)
    for note in report.notes:
        style = "red" if note.startswith("REJECTED") else "dim"
        console.print(Text(f"  {note}", style=style))


def render_str(report: FidelityReport, *, width: int = 110, floor: float = 0.8) -> str:
    """The same table as a plain string, for tests and for piping to a file."""
    console = Console(record=True, width=width, no_color=True, force_terminal=False)
    render(report, console, floor=floor)
    return console.export_text()


def to_json(report: FidelityReport) -> JsonObject:
    """Machine-consumable form. Round-trips through ``FidelityReport`` losslessly."""
    payload: JsonObject = json.loads(report.model_dump_json())
    payload["per_tool_rates"] = {t.tool: t.rate for t in report.per_tool}
    payload["totals"] = {
        "replayed": sum(t.replayed for t in report.per_tool),
        "matched": sum(t.matched for t in report.per_tool),
        "mismatched": sum(t.mismatched for t in report.per_tool),
        "unsupported": sum(t.unsupported for t in report.per_tool),
    }
    return payload


def to_json_str(report: FidelityReport, *, indent: int = 2) -> str:
    return json.dumps(to_json(report), indent=indent, sort_keys=True)


def _first_reason(examples: tuple[JsonObject, ...]) -> str:
    for example in examples:
        for divergence in example.get("divergences") or ():
            path = divergence.get("path", "?")
            return f"{path}: {divergence.get('reason', '')}"
        if example.get("reason"):
            return str(example["reason"])
    return ""
