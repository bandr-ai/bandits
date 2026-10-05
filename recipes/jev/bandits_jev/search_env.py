"""Browser-style search tools for one rollout, over any retriever.

``SearchSession`` runs ``browser.search``, ``browser.open`` and ``browser.find``
against a ``Searcher`` and returns the real reply text for each call. Replies
to misuse (an id never shown, nothing open) are ordinary replies, so the
judge can mark them down. The agent loop pairs each call with the policy
token that produced it; this module knows nothing about tokens or veRL.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

SEARCH, OPEN, FIND = "browser.search", "browser.open", "browser.find"
TOOLS = (SEARCH, OPEN, FIND)
SNIPPET_CHARS = 400
PAGE_CHARS = 4000
MAX_FIND_MATCHES = 8
FIND_CONTEXT_CHARS = 150


class Searcher(Protocol):
    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Hits as {"docid": str, "score": float, "snippet": str}."""
        ...

    def get_document(self, docid: str) -> dict[str, Any] | None:
        """{"docid": str, "text": str}, or None when the id is unknown."""
        ...


class InMemorySearcher:
    """Word-overlap retriever over a dict of docid -> text, for tests and smoke runs."""

    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for docid, text in self.documents.items():
            score = len(words & set(re.findall(r"\w+", text.lower())))
            if score:
                scored.append({"docid": docid, "score": float(score), "snippet": text[:SNIPPET_CHARS]})
        scored.sort(key=lambda hit: (-hit["score"], hit["docid"]))
        return scored[:k]

    def get_document(self, docid: str) -> dict[str, Any] | None:
        text = self.documents.get(docid)
        return None if text is None else {"docid": docid, "text": text}


class SearchSession:
    """State for one trajectory: which ids the agent has seen and which page is open."""

    def __init__(self, searcher: Searcher, *, top_k: int = 10) -> None:
        self.searcher = searcher
        self.top_k = top_k
        self.seen: set[str] = set()
        self.open_text: str | None = None
        self.open_id: str | None = None

    def execute(self, tool: str, arguments: dict[str, Any]) -> str:
        """The reply to one tool call. Never raises on bad arguments."""
        try:
            if tool == SEARCH:
                return self._search(str(arguments["query"]))
            if tool == OPEN:
                return self._open(str(arguments["id"]))
            if tool == FIND:
                return self._find(str(arguments["pattern"]))
        except KeyError as exc:
            return f"Error: missing argument {exc}"
        return f"Error: unknown tool {tool!r}"

    @staticmethod
    def action_text(arguments: dict[str, Any]) -> str:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    def _search(self, query: str) -> str:
        hits = self.searcher.search(query, self.top_k)
        if not hits:
            return f"No results for {query!r}."
        self.seen.update(hit["docid"] for hit in hits)
        lines = [f"Search results for {query!r}:"]
        lines += [f"[{hit['docid']}] {hit['snippet'][:SNIPPET_CHARS]}" for hit in hits]
        return "\n".join(lines)

    def _open(self, docid: str) -> str:
        if docid not in self.seen:
            return f"Error: id {docid!r} was not in any search result so far."
        document = self.searcher.get_document(docid)
        if document is None:
            return f"Error: no document with id {docid!r}."
        self.open_id, self.open_text = docid, document["text"]
        text = self.open_text
        suffix = f"\n[truncated; {len(text)} characters in all, use browser.find]" if len(text) > PAGE_CHARS else ""
        return f"Opened [{docid}]:\n{text[:PAGE_CHARS]}{suffix}"

    def _find(self, pattern: str) -> str:
        if self.open_text is None:
            return "Error: no page is open; use browser.open first."
        try:
            matches = list(re.finditer(pattern, self.open_text, flags=re.IGNORECASE))
        except re.error as exc:
            return f"Error: invalid pattern ({exc})."
        if not matches:
            return f"No matches for {pattern!r} in [{self.open_id}]."
        lines = [f"{len(matches)} match(es) for {pattern!r} in [{self.open_id}]:"]
        for match in matches[:MAX_FIND_MATCHES]:
            start = max(0, match.start() - FIND_CONTEXT_CHARS)
            lines.append("..." + self.open_text[start : match.end() + FIND_CONTEXT_CHARS].replace("\n", " ") + "...")
        return "\n".join(lines)

