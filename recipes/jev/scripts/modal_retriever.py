"""Serve the BrowseComp-Plus Qwen3-Embedding-8B index on one Modal L4.

Queries are embedded the way the upstream FAISS searcher does (task prefix,
last-token pooling, fp16, max 8192 tokens) and scored by inner product against
the prebuilt normalized document vectors, held on the GPU. The model is frozen;
nothing here is trained. Files live on the `jev-browsecomp` volume:
/index/corpus.shard*.pkl and /corpus/*.parquet.

Check it (measures evidence recall on the first 100 benchmark queries, batched and
single-call latency, and that batched and single results agree). The encoding path
was separately checked once: fresh document embeddings matched the stored vectors
with cosine 0.99998 or better:

    uvx modal run recipes/jev/scripts/modal_retriever.py

Deploy so rollouts can call it by name:

    uvx modal deploy recipes/jev/scripts/modal_retriever.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-Embedding-8B"
TASK_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
MAX_LENGTH = 8192
SNIPPET_CHARS = 400

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers>=4.51", "numpy", "pyarrow"
)
app = modal.App("jev-retriever", image=image)
volume = modal.Volume.from_name("jev-browsecomp")
hf_cache = modal.Volume.from_name("jev-hf-cache", create_if_missing=True)


@app.cls(
    gpu="L4",
    memory=16384,
    timeout=3600,
    scaledown_window=300,
    max_containers=1,
    volumes={"/data": volume, "/root/.cache/huggingface": hf_cache},
)
class Retriever:
    @modal.enter()
    def load(self) -> None:
        import glob
        import pickle

        import numpy as np
        import pyarrow.parquet as pq
        import torch
        from transformers import AutoModel, AutoTokenizer

        shards, self.docids = [], []
        for path in sorted(glob.glob("/data/index/corpus.shard*.pkl")):
            with open(path, "rb") as handle:
                vectors, lookup = pickle.load(handle)
            shards.append(torch.from_numpy(np.asarray(vectors)).to("cuda", torch.float16))
            self.docids += lookup
        self.index = torch.cat(shards)
        self.texts: dict[str, str] = {}
        for path in sorted(glob.glob("/data/corpus/*.parquet")):
            table = pq.read_table(path, columns=["docid", "text"])
            self.texts.update(zip(table["docid"].to_pylist(), table["text"].to_pylist(), strict=True))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
        self.torch = torch

    def _embed(self, texts: list[str]):
        torch = self.torch
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state[:, -1]
        return torch.nn.functional.normalize(hidden.float(), dim=-1)

    @modal.batched(max_batch_size=32, wait_ms=40)
    def search_one(self, queries: list[str], ks: list[int]) -> list[list[dict]]:
        """Call with one query and one k; concurrent calls run as one GPU batch.
        Top-k as {"docid", "score", "snippet"}."""
        torch = self.torch
        scores = self._embed([TASK_PREFIX + q for q in queries]).to(torch.float16) @ self.index.T
        top = scores.float().topk(max(ks), dim=-1)
        return [
            [
                {
                    "docid": self.docids[i],
                    "score": float(s),
                    "snippet": self.texts[self.docids[i]][:SNIPPET_CHARS],
                }
                for s, i in zip(row_scores.tolist()[:k], row_ids.tolist()[:k], strict=True)
            ]
            for row_scores, row_ids, k in zip(top.values, top.indices, ks, strict=True)
        ]


@app.cls(memory=8192, timeout=3600, scaledown_window=120, max_containers=2, volumes={"/data": volume})
@modal.concurrent(max_inputs=64)
class Documents:
    """Full document text by id; CPU only. (A Modal class with a batched method
    cannot have other methods, so this lives apart from Retriever.)"""

    @modal.enter()
    def load(self) -> None:
        import glob

        import pyarrow.parquet as pq

        self.texts: dict[str, str] = {}
        for path in sorted(glob.glob("/data/corpus/*.parquet")):
            table = pq.read_table(path, columns=["docid", "text"])
            self.texts.update(zip(table["docid"].to_pylist(), table["text"].to_pylist(), strict=True))

    @modal.method()
    def get(self, docids: list[str]) -> list[str | None]:
        return [self.texts.get(docid) for docid in docids]


@app.local_entrypoint()
def main() -> None:
    work = Path(__file__).resolve().parents[3] / "work"
    queries = [
        row for row in csv.reader((work / "step-rl/browsecomp-plus-queries.tsv").open(), delimiter="\t")
    ][:100]
    evidence: dict[str, set[str]] = {}
    for line in (work / "browsecomp-plus-upstream/topics-qrels/qrel_evidence.txt").read_text().splitlines():
        qid, _, docid, _ = line.split()
        evidence.setdefault(qid, set()).add(docid)

    retriever = Retriever()
    texts = [text for _, text in queries]
    started = time.perf_counter()
    first = retriever.search_one.remote(texts[0], 10)
    cold = time.perf_counter() - started

    started = time.perf_counter()
    hits_100 = list(retriever.search_one.map(texts, [100] * len(texts)))
    batched_seconds = time.perf_counter() - started
    recall = {10: [], 100: []}
    for (qid, _), hits in zip(queries, hits_100, strict=True):
        got = [h["docid"] for h in hits]
        for k in recall:
            recall[k].append(len(evidence[qid] & set(got[:k])) / len(evidence[qid]))

    latencies = []
    singles = []
    for text in texts[:20]:
        t0 = time.perf_counter()
        singles.append(retriever.search_one.remote(text, 10))
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    overlap = [
        len({h["docid"] for h in a} & {h["docid"] for h in b[:10]}) / 10
        for a, b in zip(singles, hits_100[:20], strict=True)
    ]
    document = Documents().get.remote([first[0]["docid"]])[0]
    print(
        json.dumps(
            {
                "queries": len(queries),
                "evidence_recall@10": sum(recall[10]) / len(queries),
                "evidence_recall@100": sum(recall[100]) / len(queries),
                "first_call_seconds_including_load": round(cold, 1),
                "concurrent_100_wall_seconds": round(batched_seconds, 2),
                "single_call_median_s": round(latencies[len(latencies) // 2], 3),
                "single_vs_batched_top10_overlap_min": min(overlap),
                "documents_get_ok": bool(document and document.startswith(first[0]["snippet"][:50])),
            },
            indent=2,
        )
    )
