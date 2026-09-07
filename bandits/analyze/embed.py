"""Group instructions by meaning with embeddings.

Measured on 61 real Claude Code sessions, related pairs embed at 0.62-0.90
cosine, against 0.41-0.49 across families.

Embedding is a model-derived judgement, so a grouping that used one is recorded
as ``proposed_by="model"`` rather than ``"rule"``, and the vectors are cached
into an artifact: re-running an analysis must not depend on a network call, and
two runs of the same corpus must produce the same families.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pydantic import model_validator

from bandits.analyze.families import normalize_instruction, normalize_request
from bandits.store import DerivedEnvelope, DerivedStore
from bandits.traces import Contract

if TYPE_CHECKING:
    from bandits.analyze.models import CorpusAnalysis

DEFAULT_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
DEFAULT_SIMILARITY = 0.6
"""Cosine-similarity cutoff chosen from the measured gap between within-family
and cross-family pairs on a real corpus."""

_BATCH = 32


class EmbeddingError(RuntimeError):
    """The embedding backend could not be reached or returned something unusable."""


class EmbeddingCache(Contract):
    """Vectors keyed by the text that produced them, pinned to a model."""

    schema_version: int = 1
    model: str
    vectors: dict[str, tuple[float, ...]]

    @model_validator(mode="after")
    def vectors_are_comparable(self) -> EmbeddingCache:
        """Every cached vector must be finite, non-empty, and the same size."""
        dimensions = {len(vector) for vector in self.vectors.values()}
        if 0 in dimensions:
            raise ValueError("embedding vectors must not be empty")
        if len(dimensions) > 1:
            raise ValueError("embedding vectors must all have the same dimension")
        if any(not math.isfinite(value) for vector in self.vectors.values() for value in vector):
            raise ValueError("embedding vectors must contain only finite numbers")
        return self

    def digest(self, corpus_id: str) -> str:
        """Identify the corpus and exact vectors, not merely their descriptor keys."""
        payload = json.dumps(
            {"corpus_id": corpus_id, "model": self.model, "vectors": self.vectors},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


Embedder = Callable[[str, Sequence[str]], list[list[float]]]
"""(model, texts) -> one vector per text, in order."""


def fireworks_embedder(model: str, texts: Sequence[str]) -> list[list[float]]:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise EmbeddingError("FIREWORKS_API_KEY is not set")

    request = urllib.request.Request(
        "https://api.fireworks.ai/inference/v1/embeddings",
        data=json.dumps({"model": model, "input": list(texts)}).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    payload: object = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EmbeddingError(f"embedding request failed: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EmbeddingError("embedding response was not valid JSON") from exc

    try:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise TypeError("response data is not a list")
        data = payload["data"]
        if any(not isinstance(item, dict) for item in data):
            raise TypeError("response data contains a non-object item")
        ordered = sorted(data, key=lambda item: item["index"])
        indices = [item["index"] for item in ordered]
        if indices != list(range(len(texts))):
            raise ValueError("embedding indexes are missing, duplicated, or out of range")
        vectors = [item["embedding"] for item in ordered]
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise TypeError("an embedding is not a non-empty list")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for vector in vectors
            for value in vector
        ):
            raise TypeError("an embedding contains a non-numeric value")
        if len({len(vector) for vector in vectors}) > 1:
            raise ValueError("embedding dimensions are inconsistent")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("an embedding contains a non-finite value")
    except (KeyError, TypeError, ValueError) as exc:
        # A 200 carrying an error body — a quota message, a changed schema —
        # is still a failure, and must surface as one rather than a traceback.
        # Only the shape and any stated reason: the body may carry partial
        # vectors, and pasting those into an error makes it unreadable.
        raise EmbeddingError(f"embedding response was not usable: {_describe(payload)}") from exc
    return vectors


def _describe(payload: object) -> str:
    """What came back, without its contents. Vectors are large and never help here."""
    if not isinstance(payload, dict):
        return f"{type(payload).__name__}, expected an object with a 'data' list"
    stated = next(
        (str(payload[key])[:200] for key in ("error", "message", "detail") if key in payload),
        None,
    )
    shape = f"keys {sorted(payload)}"
    return f"{shape}; {stated}" if stated else shape


def build_cache(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    embed: Embedder = fireworks_embedder,
    existing: EmbeddingCache | None = None,
) -> EmbeddingCache:
    """Embed whatever is not cached already, in batches."""
    if existing is not None and existing.model != model:
        raise EmbeddingError(
            f"cache was built with {existing.model!r}; vectors from two models are "
            "not comparable and must not be mixed"
        )

    vectors = dict(existing.vectors) if existing else {}
    missing = [text for text in dict.fromkeys(texts) if text not in vectors]
    for start in range(0, len(missing), _BATCH):
        batch = missing[start : start + _BATCH]
        for text, vector in zip(batch, embed(model, batch), strict=True):
            vectors[text] = tuple(vector)
    return EmbeddingCache(model=model, vectors=vectors)


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    if not norm:
        return 1.0
    similarity = sum(x * y for x, y in zip(left, right, strict=True)) / norm
    # Cosine runs to -1; a negative similarity is no more "opposite" than
    # unrelated for this purpose, so distance stops at 1.
    return min(1.0, max(0.0, 1.0 - similarity))


def embedding_distance(cache: EmbeddingCache) -> Callable[[str, str], float]:
    """A :data:`~bandits.analyze.families.Distance` backed by cached vectors.

    Anything absent from the cache is treated as maximally far rather than
    embedded on demand: a distance function that reaches the network mid-cluster
    would make the result depend on when it ran.
    """

    def distance(left: str, right: str) -> float:
        if left == right:
            return 0.0
        first, second = cache.vectors.get(left), cache.vectors.get(right)
        if first is None or second is None:
            return 1.0
        return cosine_distance(first, second)

    return distance


def compute_cache_id(cache: EmbeddingCache, corpus_id: str) -> str:
    return f"embeddings-{cache.digest(corpus_id)}"


def save_cache(cache: EmbeddingCache, store: DerivedStore, corpus_id: str) -> DerivedEnvelope:
    return store.write(
        compute_cache_id(cache, corpus_id),
        kind="embeddings",
        parent_artifact_id=corpus_id,
        payload=cache.model_dump_json().encode(),
        summary={"vectors": len(cache.vectors)},
    )


def load_cache(cache_id: str, store: DerivedStore) -> EmbeddingCache:
    return EmbeddingCache.model_validate_json(store.read_payload(cache_id))


def requests(analysis: CorpusAnalysis) -> list[str]:
    """The exact strings duplicate detection will compare, so the cache covers them.

    Not the same strings as :func:`descriptors`. Grouping compares instructions
    with their identifiers masked out, and two runs against different orders
    reduce to one descriptor there — which is the point of it, and the reason
    sameness has to be measured on text that kept its identifiers.
    """
    return sorted({normalize_request(t.instruction) for t in analysis.tasks if t.instruction})


def descriptors(analysis: CorpusAnalysis) -> list[str]:
    """The exact strings clustering will compare, so the cache covers all of them.

    Grouping runs over normalized descriptors rather than raw instructions, and
    :func:`embedding_distance` treats anything absent as maximally far. Deriving
    this list anywhere but here would let the two drift apart and silently push
    every uncovered pair to distance 1.0.
    """
    return sorted({normalize_instruction(t.instruction) for t in analysis.tasks if t.instruction})
