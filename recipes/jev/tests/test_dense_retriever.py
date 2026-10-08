"""The in-process retriever on a tiny random model and a synthetic index (CPU).

Needs torch, transformers, pyarrow and the Qwen3-Embedding tokenizer files
(downloaded once, about 11 MB); skipped without them.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from bandits_jev.dense_retriever import (  # noqa: E402
    MODEL,
    SNIPPET_CHARS,
    TASK_PREFIX,
    DenseRetriever,
    load_index,
    load_texts,
)
from bandits_jev.rollout import run_rollout  # noqa: E402

DOCS = {
    "d1": "The Golden Gate Bridge opened in 1937 in San Francisco. " * 20,
    "d2": "The Brooklyn Bridge opened in 1883 and links two boroughs.",
    "d3": "Pasta recipes from northern Italy.",
    "d4": "A history of bridge engineering. The Tacoma bridge collapsed in 1940.",
    "d5": "Unrelated notes about gardening.",
}


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def retriever(tokenizer, tmp_path_factory):
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=4096,
    )
    model = transformers.Qwen3Model(config).eval()
    staging = DenseRetriever(model, tokenizer, torch.zeros(1, 16), ["x"], {}, "cpu")
    ids = list(DOCS)
    vectors = staging.embed([DOCS[i] for i in ids]).numpy()
    folder = tmp_path_factory.mktemp("index")
    for n, part in enumerate([(0, 3), (3, 5)]):  # two shards, like the real index
        with open(folder / f"corpus.shard{n + 1}_of_2.pkl", "wb") as handle:
            pickle.dump((vectors[part[0] : part[1]], ids[part[0] : part[1]]), handle)
    pq.write_table(pa.table({"docid": ids, "text": [DOCS[i] for i in ids], "url": [""] * 5}), folder / "c.parquet")
    index, docids = load_index(str(folder / "corpus.shard*.pkl"), "cpu", torch.float32)
    assert docids == ids
    return DenseRetriever(model, tokenizer, index, docids, load_texts(str(folder / "*.parquet")), "cpu")


def test_hits_are_ranked_by_inner_product_with_the_prefixed_query(retriever):
    hits = retriever.search("bridge opened", k=5)
    query = retriever.embed([TASK_PREFIX + "bridge opened"])[0]
    expected = sorted(((float(query @ retriever.vectors[i]), d) for i, d in enumerate(retriever.docids)), reverse=True)
    assert [h["docid"] for h in hits] == [d for _, d in expected]
    assert [h["score"] for h in hits] == pytest.approx([s for s, _ in expected], abs=1e-5)


def test_snippets_documents_and_k(retriever):
    hits = retriever.search("anything", k=2)
    assert len(hits) == 2
    assert hits[0]["snippet"] == DOCS[hits[0]["docid"]][:SNIPPET_CHARS]
    assert retriever.get_document("d3") == {"docid": "d3", "text": DOCS["d3"]}
    assert retriever.get_document("missing") is None
    assert len(retriever.search("anything", k=50)) == 5


def test_concurrent_searches_match_serial_ones(retriever):
    queries = [f"query {i} about bridges" for i in range(16)]
    serial = [retriever.search(q, 3) for q in queries]
    with ThreadPoolExecutor(8) as pool:
        parallel = list(pool.map(lambda q: retriever.search(q, 3), queries))
    assert [[h["docid"] for h in hs] for hs in parallel] == [[h["docid"] for h in hs] for hs in serial]


def test_a_rollout_runs_end_to_end_on_the_in_process_retriever(retriever):
    first = retriever.search("Golden Gate", 10)[0]["docid"]
    script = iter([
        {"content": None, "tool_calls": [{"id": "a", "name": "search", "arguments": json.dumps({"query": "Golden Gate"})}]},
        {"content": None, "tool_calls": [{"id": "b", "name": "open", "arguments": json.dumps({"id": first})}]},
        {"content": "Answer: 1937", "tool_calls": []},
    ])

    async def chat(messages, tool_choice):
        return next(script)

    task = {"query_id": "q", "query": "When did it open?", "answer": "1937", "evidence_ids": ["d1"]}
    result = asyncio.run(run_rollout(task, chat, retriever))
    assert result["events"][1]["observation"].startswith(f"Opened [{first}]")
    assert result["correct"] and result["evidence_seen_recall"] == 1.0
