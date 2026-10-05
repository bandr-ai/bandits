"""Step recording and the shaped reward, driven through veRL's own tool loop.

Needs veRL, which the recipe environment does not install, so it is skipped
there. Run with an interpreter that has veRL, with the recipe on PYTHONPATH.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("verl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop  # noqa: E402
from verl.experimental.agent_loop.tool_parser import FunctionCall  # noqa: E402
from verl.tools.schemas import OpenAIFunctionToolSchema  # noqa: E402

from bandits_jev.verl_glue.reward_manager import JevStepRewardManager  # noqa: E402
from bandits_jev.verl_glue.tools import EVENTS_KEY, BrowserTool  # noqa: E402

CONFIG = {"type": "native", "searcher_factory": "bandits_jev.verl_glue.fake_corpus:searcher"}


def schema(name: str, argument: str) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {
                    "type": "object",
                    "properties": {argument: {"type": "string", "description": argument}},
                    "required": [argument],
                },
            },
        }
    )


def tools():
    return {
        "search": BrowserTool(CONFIG, schema("search", "query")),
        "open": BrowserTool(CONFIG, schema("open", "id")),
    }


def agent_data(mask):
    data = AgentData([], None, None, None, None, {}, "r", {})
    data.response_mask = list(mask)
    data.prompt_ids = list(range(len(mask)))
    return data


def loop(tool_map):
    """A ToolAgentLoop with generation and tokenization stubbed; the tool path is veRL's own."""
    stub = object.__new__(ToolAgentLoop)
    stub.tools = tool_map
    stub.tool_schemas = []
    stub.max_parallel_calls = 1
    stub.max_tool_response_length = 10_000
    stub.tool_response_truncate_side = "right"
    stub.response_length = 1000
    stub.processor = None
    stub.mask_seen_by_tool_reply = []

    def no_multimodal(_):
        return None

    async def merge_context(previous_messages, messages, prompt_ids, mask, logprobs, tools):
        # The reply tokens arrive after the tools ran: 4 tool tokens, mask 0.
        stub.mask_seen_by_tool_reply.append(list(mask))
        return SimpleNamespace(token_ids=prompt_ids + [0] * 4), list(mask) + [0] * 4, None

    stub._assert_mm_supported = no_multimodal
    stub.ct_merge_context_msg = merge_context
    return stub


def call(agent, name, **arguments):
    agent.tool_calls = [FunctionCall(name=name, arguments=json.dumps(arguments))]


def test_the_real_tool_loop_records_the_last_policy_token_before_the_reply_is_appended():
    stub, agent = loop(tools()), agent_data([1, 1, 1, 1, 1])  # 5 policy tokens: action 1
    call(agent, "search", query="bridge opened")
    asyncio.run(stub._handle_processing_tools_state(agent))
    assert agent.response_mask == [1] * 5 + [0] * 4  # reply tokens are tool tokens
    agent.response_mask += [1, 1, 1]  # the policy writes action 2
    call(agent, "open", id="d1")
    asyncio.run(stub._handle_processing_tools_state(agent))

    first, second = agent.extra_fields[EVENTS_KEY]
    assert first["position"] == 4 and second["position"] == 11  # last token of each action
    assert stub.mask_seen_by_tool_reply[0] == [1] * 5  # reply not yet in the stream
    assert first["tool"] == "browser.search" and first["observation"].startswith("Search results")
    assert second["observation"].startswith("Opened [d1]")  # state carried across calls


def test_each_trajectory_starts_with_a_fresh_session():
    stub = loop(tools())
    agent = agent_data([1, 1])
    call(agent, "open", id="d1")
    asyncio.run(stub._handle_processing_tools_state(agent))
    assert agent.extra_fields[EVENTS_KEY][0]["observation"].startswith("Error: id 'd1' was not in any search result")


class FakeJudge:
    def __init__(self, scores):
        self.scores, self.requests = scores, []

    async def score(self, steps):
        self.requests.append(steps)
        results = [{"score": self.scores[s["action"]], "reason": None} for s in steps]
        return {"judge": {"adapter_digest": "abc"}, "results": results}


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "ANSWER"


def manager(judge, step_weight=0.3):
    config = OmegaConf.create({"reward": {"reward_kwargs": {"step_weight": step_weight, "step_cap": 1.0}}})
    compute_score = lambda **kw: 1.0  # noqa: E731
    return JevStepRewardManager(config, FakeTokenizer(), compute_score, judge_client=judge)


def run(reward_manager, data):
    # veRL binds a manager to the event loop it was built on.
    return reward_manager.loop.run_until_complete(reward_manager.run_single(data))


def rollout(events, response_len=14):
    return DataProto.from_dict(
        tensors={
            "responses": torch.zeros(1, response_len, dtype=torch.long),
            "attention_mask": torch.ones(1, 3 + response_len, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["t"], dtype=object),
            "reward_model": np.array([{"ground_truth": "x"}], dtype=object),
            "extra_info": np.array([{"question": "Which bridge opened in 1937?"}], dtype=object),
            "tool_extra_fields": np.array([{EVENTS_KEY: events}], dtype=object),
            "response_len": np.array([response_len]),
            "__num_turns__": np.array([5]),
        },
    )


def event(position, action, observation="a real reply", tool="browser.search"):
    return {"position": position, "tool": tool, "action": action, "observation": observation}


def test_total_is_outcome_plus_weighted_judge_scores_and_the_judge_sees_prior_steps():
    judge = FakeJudge({"a": 0.8, "b": -0.5})
    result = run(manager(judge), rollout([event(4, "a"), event(9, "b")]))
    assert result["reward_score"] == pytest.approx(1.0 + 0.3 * 0.8 - 0.3 * 0.5)
    info = result["reward_extra_info"]
    assert info["outcome"] == 1.0 and info["n_steps"] == 2 and info["judge_failed"] == 0
    assert info["judge_adapter_digest"] == "abc"
    first, second = judge.requests[0]
    assert first["previous"] == [] and first["question"] == "Which bridge opened in 1937?"
    assert second["previous"] == [["browser.search", "a", "a real reply"]]


def test_a_failed_judge_call_costs_the_step_term_not_the_run():
    class Broken:
        async def score(self, steps):
            raise RuntimeError("modal down")

    result = run(manager(Broken()), rollout([event(4, "a")]))
    assert result["reward_score"] == 1.0
    assert result["reward_extra_info"]["judge_failed"] == 1


def test_no_steps_means_no_judge_call_and_the_plain_outcome():
    judge = FakeJudge({})
    result = run(manager(judge), rollout([]))
    assert result["reward_score"] == 1.0 and judge.requests == []


def test_unobserved_steps_are_never_sent_to_the_judge_or_rewarded():
    judge = FakeJudge({"a": 0.9})
    result = run(manager(judge), rollout([event(4, "a"), event(9, "b", observation="")]))
    assert [s["action"] for s in judge.requests[0]] == ["a"]
    assert result["reward_extra_info"]["rejected"] == '{"no_provenance": 1}'
