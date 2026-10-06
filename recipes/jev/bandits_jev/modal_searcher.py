"""A ``Searcher`` backed by the deployed ``jev-retriever`` Modal app (scripts/modal_retriever.py)."""

from __future__ import annotations

from typing import Any


class ModalSearcher:
    """Blocking client. Callers that must not block run it in a thread; concurrent
    searches are batched on the GPU by the server."""

    def __init__(self, retriever: Any = None, documents: Any = None, *, app_name: str = "jev-retriever") -> None:
        self._retriever, self._documents, self.app_name = retriever, documents, app_name

    def _remote(self, class_name: str) -> Any:
        import modal

        return modal.Cls.from_name(self.app_name, class_name)()

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if self._retriever is None:
            self._retriever = self._remote("Retriever")
        return self._retriever.search_one.remote(query, k)

    def get_document(self, docid: str) -> dict[str, Any] | None:
        if self._documents is None:
            self._documents = self._remote("Documents")
        text = self._documents.get.remote([docid])[0]
        return None if text is None else {"docid": docid, "text": text}


def searcher() -> ModalSearcher:
    return ModalSearcher()
