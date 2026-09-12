from pathlib import Path

from prepare_trail_corpus import prepare


def test_prepares_real_trail_gaia_without_copying_labels_into_traces() -> None:
    trail = Path("/tmp/trail-benchmark/benchmarking")
    if not trail.exists():
        return

    corpus, scores = prepare(trail, "gaia")

    assert len(corpus.traces) == 116
    assert set(scores) == {trace.trace_id for trace in corpus.traces}
    assert all(trace.user_turns[0].text == trace.task for trace in corpus.traces)
    serialized = corpus.model_dump_json()
    assert "reliability_score" not in serialized
    assert '"overall"' not in serialized
