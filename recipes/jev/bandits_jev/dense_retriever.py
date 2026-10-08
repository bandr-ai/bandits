"""Dense retrieval over the BrowseComp-Plus index, inside whatever process needs it.

Queries are embedded the way the upstream FAISS searcher embeds them (task
prefix, left padding, last-token pooling, max 8192 tokens), normalised, and
scored by inner product against the prebuilt normalised document vectors.
Before training, the trainer checks this module against 20 saved reference
searches (work/step-rl/retrieval_reference.json). torch, transformers and
pyarrow are imported lazily.
"""

from __future__ import annotations

import glob
import pickle
import threading
from typing import Any

MODEL = "Qwen/Qwen3-Embedding-8B"
TASK_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
MAX_LENGTH = 8192
SNIPPET_CHARS = 400


def load_texts(corpus_glob: str) -> dict[str, str]:
    import pyarrow.parquet as pq

    texts: dict[str, str] = {}
    for path in sorted(glob.glob(corpus_glob)):
        table = pq.read_table(path, columns=["docid", "text"])
        texts.update(zip(table["docid"].to_pylist(), table["text"].to_pylist(), strict=True))
    return texts


def load_index(index_glob: str, device: str, dtype: Any) -> tuple[Any, list[str]]:
    """Concatenate pickled (vectors, docids) shards in sorted path order."""
    import numpy as np
    import torch

    shards, docids = [], []
    for path in sorted(glob.glob(index_glob)):
        with open(path, "rb") as handle:
            vectors, lookup = pickle.load(handle)
        shards.append(torch.from_numpy(np.asarray(vectors)).to(device, dtype))
        docids += lookup
    if not shards:
        raise FileNotFoundError(f"no index shards match {index_glob}")
    return torch.cat(shards), docids


class DenseRetriever:
    def __init__(self, model: Any, tokenizer: Any, vectors: Any, docids: list[str], texts: dict[str, str], device: str) -> None:
        self.model, self.tokenizer, self.vectors = model, tokenizer, vectors
        self.docids, self.texts, self.device = docids, texts, device
        self._lock = threading.Lock()

    @classmethod
    def load(cls, index_glob: str, corpus_glob: str, *, model_name: str = MODEL, device: str = "cuda") -> DenseRetriever:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = torch.float16 if device != "cpu" else torch.float32
        vectors, docids = load_index(index_glob, device, dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
        return cls(model, tokenizer, vectors, docids, load_texts(corpus_glob), device)

    def embed(self, texts: list[str]) -> Any:
        import torch

        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state[:, -1]
        return torch.nn.functional.normalize(hidden.float(), dim=-1)

    def search_many(self, queries: list[str], k: int) -> list[list[dict[str, Any]]]:
        """Top-k per query as {"docid", "score", "snippet"}. Safe to call from several threads."""
        with self._lock:
            scores = self.embed([TASK_PREFIX + q for q in queries]).to(self.vectors.dtype) @ self.vectors.T
            top = scores.float().topk(min(k, len(self.docids)), dim=-1)
            values, indices = top.values.tolist(), top.indices.tolist()
        return [
            [
                {"docid": self.docids[i], "score": float(s), "snippet": self.texts[self.docids[i]][:SNIPPET_CHARS]}
                for s, i in zip(row_scores, row_ids, strict=True)
            ]
            for row_scores, row_ids in zip(values, indices, strict=True)
        ]

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        return self.search_many([query], k)[0]

    def get_document(self, docid: str) -> dict[str, Any] | None:
        text = self.texts.get(docid)
        return None if text is None else {"docid": docid, "text": text}
