from __future__ import annotations

import pytest

from bandits_jev.answer_match import answer_matches, extract_answer


def test_the_last_answer_line_is_used():
    assert extract_answer("thinking...\nAnswer: Paris\nmore\nFinal answer: Lyon") == "Lyon"
    assert extract_answer("just Rome") == "just Rome"


@pytest.mark.parametrize(
    ("reply", "gold"),
    [
        ("Answer: Queen Arwa University", "Queen Arwa University"),
        ("Answer: the queen arwa university.", "Queen Arwa University"),
        ("Answer: Queen Arwa University of Yemen", "Queen Arwa University"),
        ("Georgia Hirst", "Georgia Hirst"),
    ],
)
def test_matches(reply, gold):
    assert answer_matches(reply, gold)


@pytest.mark.parametrize(
    ("reply", "gold"),
    [
        ("Answer: Arwa University", "Queen Arwa University"),
        ("Answer: Boston is a big city in the northeast of the United States of America", "Boston"),
        ("Answer: Red", "Blue"),
        ("", "Red"),
        ("Answer: Red", ""),
    ],
)
def test_misses(reply, gold):
    assert not answer_matches(reply, gold)
