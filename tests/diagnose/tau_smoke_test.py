"""Real-artifact smoke test against the stored tau2 corpus.

D73/I34. Every other diagnose test runs on fixtures. D36 established that a
fixture is a claim about the data and an unverified claim is exactly as wrong
as unverified code -- but harder to notice, because it makes the tests agree
with it. The numbers pinned here are the ones v2 measured and that D39 and E26
were decided from; leaving them unpinned means the next contract change moves
them silently, which is the failure E27 demonstrated on the export side.

Skips cleanly when ``work/tau/`` is absent, so CI needs no fixture.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from bandits.diagnose.compile import extract_transitions
from bandits.diagnose.retrieve import build_index
from bandits.store import ArtifactStore

TAU_ROOT = Path("work/tau/run/proj/.bandits")
CORPUS_ID = "corpus-ee3b33086ef177d7"
TASK_SET_ID = "taskset-37654fb6a6bd578f"
FAMILY_ID = "family-451ae91f975c"
MARKERS = ("###TRANSFER###",)

pytestmark = pytest.mark.skipif(
    not (TAU_ROOT / "artifacts" / CORPUS_ID).exists(),
    reason="the tau2 artifact graph is not present in this checkout",
)


@pytest.fixture(scope="module")
def family_traces():
    corpus = ArtifactStore(TAU_ROOT).read(CORPUS_ID)
    by_id = {trace.trace_id: trace for trace in corpus.traces}
    payload = json.loads((TAU_ROOT / "derived" / TASK_SET_ID / "payload.json").read_bytes())
    family = next(f for f in payload["families"] if f["family_id"] == FAMILY_ID)
    return corpus, family, [by_id[tid] for tid in family["trace_ids"] if tid in by_id]


@pytest.fixture(scope="module")
def transitions(family_traces):
    _corpus, _family, traces = family_traces
    rows = []
    for trace in traces:
        rows.extend(extract_transitions(trace, family_id=FAMILY_ID, markers=MARKERS))
    return tuple(rows)


def test_the_stored_corpus_still_declares_no_control_markers(family_traces) -> None:
    """E24/D38. The marker is in the text and undeclared on the artifact, so a
    caller must pass it explicitly until Gate 0 re-ingests. If this ever starts
    failing, the migration happened and compile's callers can stop supplying it.
    """
    corpus, _family, traces = family_traces
    assert corpus.control_markers == ()
    assert len(traces) == 36


def test_transition_extraction_reproduces_the_measured_counts(transitions) -> None:
    """E22/E25/D39. The batch-aware counts, from the real artifact.

    371 rather than the pre-D39 393: fragmented batches merged back into single
    actions, which is the correction, visible in the number.
    """
    assert len(transitions) == 371
    assert sum(1 for row in transitions if row.observations) == 371
    assert dict(sorted(Counter(len(r.action_calls) for r in transitions).items())) == {
        0: 189,
        1: 168,
        2: 10,
        4: 4,
    }


def test_batched_actions_survive_as_batches(transitions) -> None:
    """D39. An action carrying several calls must not read as one call, and
    ``action_tool`` must refuse to name a single tool for it.
    """
    batched = [row for row in transitions if len(row.action_calls) > 1]
    assert len(batched) == 14
    assert all(row.action_tool is None for row in batched)
    for row in batched:
        assert len({call.call_id for call in row.action_calls}) == len(row.action_calls)


def test_over_half_the_family_is_the_user_policys_responsibility(transitions) -> None:
    """The count behind D23's gate being primary rather than secondary work."""
    user = sum(1 for row in transitions if row.reaction_role in ("user", "mixed"))
    assert user == 189


def test_the_marker_is_stripped_and_recorded_not_dropped(transitions) -> None:
    """D29. Filtering marker-carrying content would discard a large share of
    this family's user turns; stripping keeps the transition and records it.
    """
    stripped = [row for row in transitions if row.stripped_markers]
    assert stripped, "no marker was found; the corpus or the marker list changed"
    for row in stripped:
        for observation in row.observations:
            assert "###TRANSFER###" not in str(observation.content)


def test_the_fit_index_excludes_held_out_lineages(family_traces, transitions) -> None:
    """D28. Held-out traces can never be retrieval sources."""
    _corpus, family, _traces = family_traces
    held_out = set(family["held_out_trace_ids"])
    assert held_out, "the family records no held-out split"
    # Every transition goes in, so this exercises the index's own exclusion
    # rather than a filter applied by the test.
    index = build_index(transitions, fit_trace_ids=family["fit_trace_ids"])
    assert index
    assert not {row.trace_id for row in index} & held_out
    assert all(row.observed for row in index)


def test_every_tool_in_this_family_still_has_zero_error_evidence(transitions) -> None:
    """E26. Not 'sparse' -- zero. This is why D27 abstains rather than inventing
    an error, and the assertion exists so the claim is rechecked rather than
    remembered.
    """
    errored = [
        row
        for row in transitions
        if any(observation.error for observation in row.observations)
    ]
    assert errored == []
