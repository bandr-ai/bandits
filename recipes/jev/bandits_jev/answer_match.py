"""A cheap stand-in for the benchmark's official LLM answer grader.

Strict on purpose: the extracted answer must equal the gold answer after
normalisation, or contain it as whole words while being no more than twice as
long. It will miss some correct paraphrases and is a proxy, not the official
BrowseComp-Plus grade; report it as such.
"""

from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}
_ANSWER = re.compile(r"(?:final\s+)?(?:exact\s+)?answer\s*[:\-]\s*(.+)", re.IGNORECASE)


def extract_answer(text: str) -> str:
    """The text after the last 'Answer:' line, else the whole reply."""
    matches = _ANSWER.findall(text)
    return (matches[-1] if matches else text).strip()


def _tokens(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in _ARTICLES]


def answer_matches(reply: str, gold: str) -> bool:
    predicted, target = _tokens(extract_answer(reply)), _tokens(gold)
    if not predicted or not target:
        return False
    if predicted == target:
        return True
    if len(predicted) > 2 * len(target):
        return False
    return any(predicted[i : i + len(target)] == target for i in range(len(predicted) - len(target) + 1))
