"""Split generation's agent rollout: the agent model continues a trace from the real system prompt and tools, its
tool calls answered by a simulator (TOOL_SIM_TRACE). Used only when features.split_answer is on."""
from __future__ import annotations

import json

from .llm import chat_json, get_llm
from .schema import to_openai_messages, to_openai_tools


FINAL_NOW = "Write your final reply now, based on what you have observed so far."


def agent_rollout(cfg, trace: dict, situation: str, obs_examples, max_rounds: int = 3,
                  format_result=lambda content, n: content) -> tuple[list[dict] | None, dict]:
    """Split generation: the agent continues a trace that ends before its next step, from the real system prompt and
    tools. Each tool call it makes is answered by a simulator that sees the conversation and the situation, never the
    intended answer (TOOL_SIM_TRACE). After max_rounds tool rounds the agent is asked for its final reply (twice at
    most). obs_examples(tool_name) -> formatted real results of that tool; format_result(content, n) applies harness
    formatting to result number n of the whole trace. Returns (new messages appended, info)."""
    from .prompts import TOOL_SIM_TRACE
    from .schema import render
    agent, sim = get_llm(cfg, "agent"), get_llm(cfg, "tool_sim")
    tools = trace.get("tools") or []
    specs = {t["name"]: t for t in tools}
    oa_tools = to_openai_tools(tools) or None
    msgs, added, calls = list(trace["messages"]), [], 0
    for rnd in range(max_rounds + 2):  # the last two rounds ask for the final reply; a tool call there is retried once
        last = rnd >= max_rounds
        conv = to_openai_messages(trace.get("system", ""), msgs) + ([{"role": "system", "content": FINAL_NOW}] if last else [])
        r = agent.chat(conv, tools=oa_tools)
        if not r.get("tool_calls"):
            if not (r.get("content") or "").strip():
                return None, {"issue": "agent_empty_reply", "rounds": rnd, "simulated_calls": calls}
            m = {"role": "assistant", "content": r["content"]}
            return added + [m], {"rounds": rnd, "simulated_calls": calls}
        if last:
            if rnd == max_rounds:
                continue  # ask once more for the final reply, without the tool calls it just tried
            return None, {"issue": "agent_needs_more_evidence: still calling tools after the last round", "rounds": rnd,
                          "simulated_calls": calls}
        m = {"role": "assistant", "content": r.get("content") or "", "tool_calls": r["tool_calls"]}
        msgs.append(m)
        added.append(m)
        for c in r["tool_calls"]:
            calls += 1
            out = chat_json(sim, "You simulate a software backend precisely.", TOOL_SIM_TRACE.format(
                situation=situation, tool_spec=json.dumps(specs.get(c["name"], {"name": c["name"], "unknown": True}), ensure_ascii=False),
                obs_examples=obs_examples(c["name"]), conversation=render({"messages": msgs}, max_obs_chars=4000),
                call_name=c["name"], call_args=json.dumps(c["arguments"], ensure_ascii=False)))
            res = out.get("result", "") if isinstance(out, dict) else out
            n_result = sum(1 for x in msgs if x["role"] == "tool") + 1
            t = {"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                 "content": format_result(res if isinstance(res, str) else json.dumps(res, ensure_ascii=False), n_result)}
            msgs.append(t)
            added.append(t)
    return None, {"issue": "unreachable"}
