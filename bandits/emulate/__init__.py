"""Compile historical traces into verifiable, trace-grounded environments.

The emulate plane is read-only with respect to everything beneath it. It
consumes corpora, analyses, task sets and reviewed verifiers, and it produces
scenarios, grounding transitions and rollouts as new derived artifacts. Nothing
here writes a ``Span``, a ``Trace`` or a ``TraceCorpus``: a simulated
observation must never become indistinguishable from a recorded one.
"""

from __future__ import annotations
