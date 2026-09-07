"""Embedding backend tests. No network: the embedder callable is injected.

The paraphrase cases here pin the semantic geometry expected from the embedding
backend without retaining a second production clustering implementation.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from bandits.analyze.embed import DEFAULT_SIMILARITY as EMBEDDING_SIMILARITY
from bandits.analyze.embed import (
    EmbeddingCache,
    EmbeddingError,
    build_cache,
    compute_cache_id,
    cosine_distance,
    descriptors,
    embedding_distance,
    load_cache,
    save_cache,
)
from bandits.analyze.families import mine_task_set, normalize_instruction
from bandits.analyze.models import CorpusAnalysis, TaskCandidate
from bandits.store import DerivedStore

# Two requests per topic, sharing almost no vocabulary with their own pair.
LOGIN = ("log into the dev box", "ssh into dev for me")
REVIEW = ("review this PR", "can you take a look at my pull request")

# Vectors chosen so the measured geometry matches the real corpus the backend was
# tuned against: within-family pairs land around 0.65 cosine and cross-family
# pairs near 0.45. Each text gets its own vector, so a pair is never trivially
# identical and cosine_distance does real arithmetic.
_VECTORS: dict[str, list[float]] = {
    LOGIN[0]: [1.0, 0.0, 0.0, 0.0],
    LOGIN[1]: [0.66, 0.75, 0.0, 0.0],
    REVIEW[0]: [0.0, 0.0, 1.0, 0.0],
    REVIEW[1]: [0.0, 0.0, 0.66, 0.75],
}
_BY_DESCRIPTOR = {normalize_instruction(text): vector for text, vector in _VECTORS.items()}


def _stub_embedder(model: str, texts):
    """Stands in for the Fireworks call, returning fixed vectors for known text."""
    return [list(_BY_DESCRIPTOR.get(text, [0.0, 0.0, 0.0, 1.0])) for text in texts]


def _analysis(*instructions: str) -> CorpusAnalysis:
    return CorpusAnalysis(
        corpus_id="corpus-test",
        source="otlp",
        tasks=tuple(
            TaskCandidate(task_id=f"task-{i}", trace_id=f"trace-{i}", instruction=text)
            for i, text in enumerate(instructions)
        ),
        evidence=(),
    )


def _cache(*instructions: str) -> EmbeddingCache:
    return build_cache(
        [normalize_instruction(text) for text in instructions],
        model="stub",
        embed=_stub_embedder,
    )


def _mine(analysis, cache, **overrides):
    return mine_task_set(
        analysis,
        "analysis-x",
        budget=10,
        distance=embedding_distance(cache),
        backend="embedding",
        embedding_model=cache.model,
        proposed_by="model",
        **overrides,
    )


def test_paraphrases_land_in_one_embedding_family() -> None:
    analysis = _analysis(*LOGIN, *REVIEW)
    cache = _cache(*LOGIN, *REVIEW)

    embedded = _mine(analysis, cache, similarity=EMBEDDING_SIMILARITY)

    assert len(embedded.families) == 2
    grouped = sorted(sorted(f.trace_ids) for f in embedded.families)
    assert grouped == [["trace-0", "trace-1"], ["trace-2", "trace-3"]]


def test_an_overly_strict_threshold_rejects_pairs_the_backend_identifies() -> None:
    """The embedding cutoff is calibrated and materially affects grouping."""
    analysis = _analysis(*LOGIN, *REVIEW)
    cache = _cache(*LOGIN, *REVIEW)

    assert len(_mine(analysis, cache, similarity=0.7).families) == 4
    assert len(_mine(analysis, cache, similarity=EMBEDDING_SIMILARITY).families) == 2


def test_embedding_grouping_is_recorded_as_model_proposed() -> None:
    analysis = _analysis(*LOGIN)
    task_set = _mine(analysis, _cache(*LOGIN), similarity=EMBEDDING_SIMILARITY)

    assert {f.proposed_by for f in task_set.families} == {"model"}


def test_an_overly_strict_cosine_threshold_rejects_a_real_paraphrase_pair() -> None:
    """0.7 rejects a measured same-family pair that reaches 0.62 similarity."""
    cache = _cache(*LOGIN)
    distance = embedding_distance(cache)
    pair = distance(*(normalize_instruction(text) for text in LOGIN))

    assert pair <= 1.0 - 0.6, "should group under the embedding default"
    assert pair > 1.0 - 0.7, "and be rejected by an overly strict cutoff"


def test_descriptors_are_the_strings_clustering_compares() -> None:
    """Two order numbers are one descriptor, so they are embedded once."""
    analysis = _analysis("Refund order 7741", "Refund order 8820", "")

    assert descriptors(analysis) == ["refund order <order_id>"]


def test_a_status_code_is_embedded_as_itself_rather_than_as_an_identifier() -> None:
    """Masking feeds the embedder, so collapsing these would be one family."""
    analysis = _analysis("Handle the HTTP 404", "Handle the HTTP 500", "")

    assert descriptors(analysis) == ["handle the http 404", "handle the http 500"]


def test_a_missing_vector_reads_as_far_rather_than_reaching_the_network() -> None:
    distance = embedding_distance(EmbeddingCache(model="stub", vectors={"known": (1.0, 0.0)}))

    assert distance("known", "absent") == 1.0
    assert distance("absent", "absent") == 0.0


def test_cosine_distance_floors_at_zero_for_opposed_vectors() -> None:
    assert cosine_distance((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert cosine_distance((1.0, 0.0), (-1.0, 0.0)) == 1.0
    assert cosine_distance((0.0, 0.0), (1.0, 0.0)) == 1.0


def test_building_over_a_cache_only_embeds_what_is_missing() -> None:
    calls: list[list[str]] = []

    def counting(model: str, texts):
        calls.append(list(texts))
        return _stub_embedder(model, texts)

    first = build_cache(["alpha"], model="stub", embed=counting)
    second = build_cache(["alpha", "beta"], model="stub", embed=counting, existing=first)

    assert calls == [["alpha"], ["beta"]]
    assert set(second.vectors) == {"alpha", "beta"}


def test_vectors_from_two_models_are_refused() -> None:
    first = build_cache(["alpha"], model="stub", embed=_stub_embedder)

    with pytest.raises(EmbeddingError, match="not comparable"):
        build_cache(["beta"], model="other", embed=_stub_embedder, existing=first)


def test_a_saved_cache_round_trips(tmp_path) -> None:
    store = DerivedStore(tmp_path)
    cache = _cache(*LOGIN)

    envelope = save_cache(cache, store, "corpus-test")

    assert envelope.artifact_id == compute_cache_id(cache, "corpus-test")
    assert envelope.parent_artifact_id == "corpus-test"
    assert load_cache(envelope.artifact_id, store) == cache


def test_identical_descriptors_in_two_corpora_get_separate_cache_artifacts(tmp_path) -> None:
    store = DerivedStore(tmp_path)
    cache = _cache(*LOGIN)

    first = save_cache(cache, store, "corpus-a")
    second = save_cache(cache, store, "corpus-b")

    assert first.artifact_id != second.artifact_id
    assert first.parent_artifact_id == "corpus-a"
    assert second.parent_artifact_id == "corpus-b"


def test_cache_identity_includes_vector_contents() -> None:
    first = EmbeddingCache(model="stub", vectors={"task": (1.0, 0.0)})
    second = EmbeddingCache(model="stub", vectors={"task": (0.0, 1.0)})

    assert compute_cache_id(first, "corpus-test") != compute_cache_id(second, "corpus-test")


def test_a_success_response_with_an_error_body_is_still_a_failure() -> None:
    """A 200 carrying a quota message must not surface as a traceback."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"error": "quota exceeded"}'

    with (
        mock.patch("urllib.request.urlopen", return_value=_Response()),
        mock.patch.dict(os.environ, {"FIREWORKS_API_KEY": "x"}),
        pytest.raises(EmbeddingError, match="not usable"),
    ):
        build_cache(["alpha"], model="stub")


def test_an_unusable_response_is_described_without_quoting_its_contents() -> None:
    """A body can carry partial vectors; an error that pastes them is unreadable."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"error": "quota exceeded", "data": None, "junk": [0.123] * 5000}
            ).encode()

    with (
        mock.patch("urllib.request.urlopen", return_value=_Response()),
        mock.patch.dict(os.environ, {"FIREWORKS_API_KEY": "x"}),
        pytest.raises(EmbeddingError) as raised,
    ):
        build_cache(["alpha"], model="stub")

    message = str(raised.value)
    assert "quota exceeded" in message, "the stated reason has to survive"
    assert "0.123" not in message, "the payload's contents must not"
    assert len(message) < 300


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (b"not-json", "not valid JSON"),
        (b'{"data":[{"index":0}]}', "not usable"),
        (b'{"data":[{"index":0,"embedding":"wrong"}]}', "not usable"),
        (b'{"data":[{"index":1,"embedding":[1.0]}]}', "not usable"),
        (b'{"data":[{"index":0,"embedding":[1.0,null]}]}', "not usable"),
    ],
)
def test_every_malformed_success_response_becomes_an_embedding_error(body, detail) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body

    with (
        mock.patch("urllib.request.urlopen", return_value=_Response()),
        mock.patch.dict(os.environ, {"FIREWORKS_API_KEY": "x"}),
        pytest.raises(EmbeddingError, match=detail),
    ):
        build_cache(["alpha"], model="stub")


def test_embedding_cache_refuses_incomparable_vectors() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        EmbeddingCache(model="stub", vectors={"a": (1.0,), "b": (1.0, 2.0)})

    with pytest.raises(ValueError, match="finite"):
        EmbeddingCache(model="stub", vectors={"a": (float("nan"),)})
