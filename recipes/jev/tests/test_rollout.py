"""The rollout loop driven by a scripted policy over the five-document corpus."""

from __future__ import annotations

import asyncio
import json

from bandits_jev.rollout import ContextOverflow, run_rollout
from bandits_jev.search_env import InMemorySearcher

DOCS = {
    "d1": "The Golden Gate Bridge opened in 1937 in San Francisco.",
    "d2": "The Brooklyn Bridge opened in 1883 and links two boroughs.",
    "d3": "Pasta recipes from northern Italy.",
}
TASK = {"query_id": "q1", "query": "When did the Golden Gate Bridge open?", "answer": "1937", "evidence_ids": ["d1", "d2"]}


def call(name, id_="c1", **arguments):
    return {"id": id_, "name": name, "arguments": json.dumps(arguments)}


def policy(script):
    """Yields the scripted messages in order; records the tool_choice and message count seen."""
    seen = []
    steps = iter(script)

    async def chat(messages, tool_choice):
        seen.append((tool_choice, len(messages)))
        step = next(steps)
        if isinstance(step, Exception):
            raise step
        return step

    chat.seen = seen
    return chat


def run(script, **kwargs):
    chat = policy(script)
    return asyncio.run(run_rollout(TASK, chat, InMemorySearcher(DOCS), **kwargs)), chat


def test_a_search_open_answer_rollout_is_scored_against_ground_truth():
    result, chat = run(
        [
            {"content": "Let me search.", "tool_calls": [call("search", query="Golden Gate Bridge opened")]},
            {"content": None, "tool_calls": [call("open", id="d1")]},
            {"content": "Answer: 1937", "tool_calls": []},
        ]
    )
    assert [e["tool"] for e in result["events"]] == ["browser.search", "browser.open"]
    assert result["events"][1]["observation"].startswith("Opened [d1]")
    assert result["ended"] == "answer" and result["correct"] and result["turns"] == 3
    # both evidence docs (d1, d2) appear in the results, but only d1 was opened
    assert result["evidence_seen_recall"] == 1.0 and result["evidence_opened_recall"] == 0.5
    assert result["assistant_texts"] == ["Let me search.", "Answer: 1937"]
    search, opened = result["events"]
    assert search["new_evidence_seen"] == 2 and not search["evidence_opened"]
    assert opened["new_evidence_seen"] == 0 and opened["evidence_opened"]
    assert chat.seen == [("auto", 2), ("auto", 4), ("auto", 6)]  # system + user, then +2 per tool turn


def test_only_the_first_call_in_a_turn_runs_and_the_rest_are_counted():
    result, _ = run(
        [
            {"content": None, "tool_calls": [call("search", "a", query="bridge"), call("search", "b", query="pasta")]},
            {"content": "Answer: 1937", "tool_calls": []},
        ]
    )
    assert result["tool_calls"] == 1 and result["dropped_extra_calls"] == 1


def test_the_last_turn_forbids_tools_and_whatever_it_says_is_the_answer():
    result, chat = run(
        [
            {"content": None, "tool_calls": [call("search", query="bridge")]},
            {"content": "Answer: 1937", "tool_calls": [call("search", query="ignored")]},
        ],
        max_turns=2,
    )
    assert chat.seen[-1][0] == "none"
    assert result["final"] == "Answer: 1937" and result["correct"] and result["tool_calls"] == 1


def test_bad_arguments_and_unknown_tools_become_replies_not_crashes():
    result, _ = run(
        [
            {"content": None, "tool_calls": [{"id": "c", "name": "search", "arguments": "{not json"}]},
            {"content": None, "tool_calls": [call("fly", x="1")]},
            {"content": "Answer: 1937", "tool_calls": []},
        ]
    )
    assert result["events"][0]["observation"].startswith("Error: invalid arguments")
    assert result["events"][1]["observation"].startswith("Error: unknown tool")
    assert result["correct"]


def test_a_context_overflow_ends_the_rollout_with_an_error_and_no_answer():
    result, _ = run([{"content": None, "tool_calls": [call("search", query="bridge")]}, ContextOverflow("too long")])
    assert result["ended"] == "error" and result["final"] is None and not result["correct"]
    assert result["error"] == "context overflow: too long"


def test_a_wrong_answer_is_not_correct_and_unfound_evidence_scores_zero():
    result, _ = run([{"content": "Answer: 1883", "tool_calls": []}])
    assert not result["correct"] and result["evidence_seen_recall"] == 0.0


def test_summary_compares_judge_scores_on_evidence_steps_with_the_rest():
    from bandits_jev.rollout import summarize

    def event(score, new=0, opened=False, action="a"):
        probabilities = {"positive": max(score, 0), "neutral": 0.1, "negative": max(-score, 0)}
        return {"action": action, "new_evidence_seen": new, "evidence_opened": opened,
                "judge": {"score": score, "probabilities": probabilities}}

    done = {"ended": "answer", "correct": True, "tool_calls": 3, "evidence_seen_recall": 1.0,
            "evidence_opened_recall": 0.5, "dropped_extra_calls": 0,
            "events": [event(0.8, new=1, action="x"), event(0.8, action="x"), event(-0.5, action="y")]}
    out = summarize([done, {**done, "ended": "max_turns", "correct": False, "events": []}])
    assert out["rollouts"] == 2 and out["answered"] == 0.5 and out["accuracy_proxy"] == 0.5
    assert out["judge_mean_score_on_evidence_steps"] == 0.8 and out["evidence_steps"] == 1
    assert out["judge_mean_score_on_other_steps"] == (0.8 - 0.5) / 2
    assert out["rollouts_with_repeated_action"] == 0.5
    assert out["judge_class_share"]["positive"] == 2 / 3


def test_sampled_token_ids_are_kept_per_model_call_when_the_server_returns_them():
    result, _ = run(
        [
            {"content": None, "tool_calls": [call("search", query="bridge")], "prompt_token_ids": [1, 2], "token_ids": [7, 8], "logprobs": [-0.1, -0.2]},
            {"content": "Answer: 1937", "tool_calls": [], "prompt_token_ids": [1, 2, 7, 8, 3], "token_ids": [9], "logprobs": [-0.3]},
        ]
    )
    assert result["model_calls"] == [
        {"prompt_token_ids": [1, 2], "token_ids": [7, 8], "logprobs": [-0.1, -0.2]},
        {"prompt_token_ids": [1, 2, 7, 8, 3], "token_ids": [9], "logprobs": [-0.3]},
    ]


def test_no_token_ids_means_no_model_calls():
    result, _ = run([{"content": "Answer: 1937", "tool_calls": []}])
    assert result["model_calls"] == []
