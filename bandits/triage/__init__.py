"""Stage 0 -- triage. Can an environment be built from this telemetry at all?

Every other stage assumes the answer is yes. This one asks it, on a customer's
real export, before anything is promised.

docs/PRODUCT.md puts it as the disqualifying question: *did you keep the tool
calls?* No invocation points -> no action space -> no state -> no verifier ->
judge. That chain is not recoverable later by trying harder; it is decided by
what the customer's logging retained months ago. Finding it out in week six of a
deployment is the expensive way to find out.

So triage runs on the raw export, produces one page, and takes a position:

``GO``
    Arguments, responses and repeated identifiers are all present. An environment
    can be reconstructed and graded by state assertion.
``PARTIAL``
    Invocation points exist, but something the reward path needs is thin -- no
    write tools, no error responses, no identifier recurrence. Some tools will
    reconstruct and some will not, and the report says which.
``NO_GO``
    No usable invocation points. Say so plainly and do not sell an environment.

What triage is not
------------------
It is not a fidelity number. Fidelity is measured by rebuilding the world and
replaying against it (:mod:`bandits.fidelity`), and it is the only claim a
customer can verify without trusting us. Triage is the cheap upstream check on
whether that measurement is even reachable. A GO here is a statement about the
*data*, never a promise about the eventual gate.

Deterministic, like everything before the gate: no model calls, no network.
"""

from __future__ import annotations

from bandits.triage.assess import (
    Signal,
    ToolReadiness,
    TriageReport,
    Verdict,
    triage_corpus,
)
from bandits.triage.render import render_report

__all__ = [
    "Signal",
    "ToolReadiness",
    "TriageReport",
    "Verdict",
    "render_report",
    "triage_corpus",
]
