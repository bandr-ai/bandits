from __future__ import annotations

from bandits_jev.modal_searcher import ModalSearcher


class Remote:
    def __init__(self, fn):
        self.remote = fn


class FakeRetriever:
    def __init__(self):
        self.calls = []
        self.search_one = Remote(lambda q, k: self.calls.append((q, k)) or [{"docid": "d1"}])


class FakeDocuments:
    def __init__(self, texts):
        self.get = Remote(lambda ids: [texts.get(i) for i in ids])


def test_search_sends_one_query_and_k_to_the_batched_method():
    retriever = FakeRetriever()
    assert ModalSearcher(retriever).search("bridge", 5) == [{"docid": "d1"}]
    assert retriever.calls == [("bridge", 5)]


def test_get_document_wraps_the_text_and_maps_unknown_ids_to_none():
    searcher = ModalSearcher(documents=FakeDocuments({"d1": "full text"}))
    assert searcher.get_document("d1") == {"docid": "d1", "text": "full text"}
    assert searcher.get_document("nope") is None
