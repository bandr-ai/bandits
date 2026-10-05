"""A five-document retriever for tests and plumbing smoke runs (not the real corpus)."""

from bandits_jev.search_env import InMemorySearcher

DOCUMENTS = {
    "d1": "The Golden Gate Bridge opened in 1937 in San Francisco.",
    "d2": "The Brooklyn Bridge opened in 1883 and links two boroughs.",
    "d3": "Pasta recipes from northern Italy.",
    "d4": "A history of bridge engineering. The Tacoma bridge collapsed in 1940.",
    "d5": "Unrelated notes about gardening.",
}


def searcher() -> InMemorySearcher:
    return InMemorySearcher(DOCUMENTS)
