"""Turn the golden fixture episodes into contract objects, without stage 1.

Stage 1 (`bandits.ingest`) is built in parallel and this module must not depend
on it, so we load ``tests/fixtures/make_corpus.py`` by path and convert its
``episodes()`` into `Trace` / `InvocationPoint` objects directly. When ingest
lands, only the tests use this; `build_surface` never touches it.

Test-support only. Nothing in the shipping path imports it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from bandits.contracts import (
    CallStatus,
    InvocationPoint,
    Message,
    Trace,
    TraceCorpus,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _load_make_corpus() -> Any:
    path = FIXTURE_DIR / "make_corpus.py"
    spec = importlib.util.spec_from_file_location("_bandits_fixture_make_corpus", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load fixture generator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def declared_tools() -> list[dict[str, Any]]:
    """The fixture tool registry, as ``tools.json`` holds it."""
    return json.loads((FIXTURE_DIR / "tools.json").read_text())


def expected() -> dict[str, Any]:
    """``expected.json`` -- the ground truth this stage must reproduce."""
    return json.loads((FIXTURE_DIR / "expected.json").read_text())


def fixture_corpus() -> TraceCorpus:
    """The golden corpus as `TraceCorpus`, with the episode id as the trace id."""
    module = _load_make_corpus()
    digest = hashlib.sha256((FIXTURE_DIR / "make_corpus.py").read_bytes()).hexdigest()

    traces: list[Trace] = []
    for episode_id, instruction, calls, answer, outcome in module.episodes():
        messages = (
            Message(role="system", content="You are a retail support agent."),
            Message(role="user", content=instruction),
            Message(role="assistant", content=answer),
        )
        invocations = tuple(
            InvocationPoint(
                call_id=f"call_{episode_id}_{step}",
                trace_id=episode_id,
                step=step,
                tool=tool,
                arguments=args,
                response=response,
                status=CallStatus(status),
                error_kind=error_kind,
            )
            for step, (tool, args, response, status, error_kind) in enumerate(calls)
        )
        traces.append(
            Trace(
                trace_id=episode_id,
                source="fixture",
                source_digest=digest,
                messages=messages,
                invocations=invocations,
                outcome=outcome,
            )
        )
    return TraceCorpus(source="fixture", traces=tuple(traces))
