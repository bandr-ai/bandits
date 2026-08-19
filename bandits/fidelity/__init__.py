"""Stage 6 -- the fidelity gate. Replay a trace against the world rebuilt from it.

    from bandits.fidelity import replay_corpus, build_report, render
    results = replay_corpus(corpus, schema, tasks, tool_classes)
    report = build_report(results)
    render(report)
    assert report.accepted

Accept requires an overall rate at or above the threshold **and** no individual
tool below the per-tool floor. See :mod:`bandits.fidelity.gate` for why both,
:mod:`bandits.fidelity.compare` for exactly which differences are tolerated,
and :mod:`bandits.fidelity.replay` for how unsupported tools and external
tools are treated.
"""

from .compare import Divergence, classify_field, compare_observations, compare_values
from .gate import (
    DEFAULT_PER_TOOL_FLOOR,
    DEFAULT_THRESHOLD,
    GateCriteria,
    build_report,
    gate,
)
from .replay import (
    CallReplay,
    ReplayResult,
    per_tool_fidelity,
    replay_corpus,
    replay_trace,
    tool_classes_from_surface,
)
from .report import render, render_str, to_json, to_json_str

__all__ = [
    "DEFAULT_PER_TOOL_FLOOR",
    "DEFAULT_THRESHOLD",
    "CallReplay",
    "Divergence",
    "GateCriteria",
    "ReplayResult",
    "build_report",
    "classify_field",
    "compare_observations",
    "compare_values",
    "gate",
    "per_tool_fidelity",
    "render",
    "render_str",
    "replay_corpus",
    "replay_trace",
    "to_json",
    "to_json_str",
    "tool_classes_from_surface",
]
