"""The search tools on a five-document corpus whose replies can be checked by hand."""

from __future__ import annotations

import json

from bandits_jev.search_env import (
    FIND,
    OPEN,
    PAGE_CHARS,
    SEARCH,
    InMemorySearcher,
    SearchSession,
)

DOCS = {
    "d1": "The Golden Gate Bridge opened in 1937 in San Francisco.",
    "d2": "The Brooklyn Bridge opened in 1883 and links two boroughs.",
    "d3": "Pasta recipes from northern Italy.",
    "d4": "A history of bridge engineering. " + "filler " * 800 + "The Tacoma bridge collapsed in 1940.",
    "d5": "Unrelated notes about gardening.",
}


def session():
    return SearchSession(InMemorySearcher(DOCS), top_k=3)


def test_search_returns_ranked_hits_with_ids_and_remembers_them():
    s = session()
    reply = s.execute(SEARCH, {"query": "bridge opened 1937"})
    lines = reply.splitlines()
    assert lines[0] == "Search results for 'bridge opened 1937':"
    assert lines[1].startswith("[d1]")
    assert s.seen == {"d1", "d2", "d4"}


def test_a_search_with_no_hits_says_so_and_shows_nothing():
    s = session()
    assert s.execute(SEARCH, {"query": "quantum chromodynamics"}) == "No results for 'quantum chromodynamics'."
    assert s.seen == set()


def test_open_only_works_on_ids_already_shown():
    s = session()
    assert s.execute(OPEN, {"id": "d1"}).startswith("Error: id 'd1' was not in any search result")
    s.execute(SEARCH, {"query": "Golden Gate"})
    assert s.opened == set()
    assert s.execute(OPEN, {"id": "d1"}) == f"Opened [d1]:\n{DOCS['d1']}"
    assert s.opened == {"d1"}


def test_a_long_page_is_cut_and_find_still_reaches_the_rest():
    s = session()
    s.execute(SEARCH, {"query": "bridge engineering history"})
    opened = s.execute(OPEN, {"id": "d4"})
    assert "[truncated;" in opened and "Tacoma" not in opened
    assert len(DOCS["d4"]) > PAGE_CHARS
    found = s.execute(FIND, {"pattern": "tacoma"})
    assert found.startswith("1 match(es) for 'tacoma' in [d4]:") and "collapsed in 1940" in found


def test_find_needs_an_open_page_and_a_valid_pattern():
    s = session()
    assert s.execute(FIND, {"pattern": "x"}).startswith("Error: no page is open")
    s.execute(SEARCH, {"query": "Golden Gate"})
    s.execute(OPEN, {"id": "d1"})
    assert s.execute(FIND, {"pattern": "("}).startswith("Error: invalid pattern")
    assert s.execute(FIND, {"pattern": "zebra"}) == "No matches for 'zebra' in [d1]."


def test_misuse_replies_are_ordinary_text_not_exceptions():
    s = session()
    assert s.execute(SEARCH, {}) == "Error: missing argument 'query'"
    assert s.execute("browser.fly", {}).startswith("Error: unknown tool")


def test_action_text_is_stable_so_repeats_can_be_detected():
    a = SearchSession.action_text({"query": "x", "topn": 3})
    b = SearchSession.action_text({"topn": 3, "query": "x"})
    assert a == b and json.loads(a) == {"query": "x", "topn": 3}


def test_a_catastrophic_pattern_times_out_into_an_error_reply():
    import time

    long_page = {"long": "terra sancta college established students teachers " * 2000}
    s = SearchSession(InMemorySearcher(long_page), top_k=3)
    s.execute(SEARCH, {"query": "terra sancta"})
    s.execute(OPEN, {"id": "long"})
    started = time.time()
    reply = s.execute(FIND, {"pattern": "terra.*sancta.*college.*established.*students.*teachers.*zzz"})
    assert reply.startswith("Error: the pattern took longer than")
    assert time.time() - started < 10
