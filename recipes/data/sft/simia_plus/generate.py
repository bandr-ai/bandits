"""Generate one trajectory per job.

mode "simia":     Simia-SFT. One LLM call writes the whole trajectory (user, assistant, calls, results)
                 from the seed as example, then Simia's fix_arguments post-processing (simia.py).
                 Optional blocks add strategy, spec, persona, failure, retrieval.
mode "simia_env": Simia-RL's environment. A teacher agent acts; one simulator model (Simia's prompt,
                 verbatim) plays both the user and the tools with the seed as reference data.
mode "split":    Separate calls per step: user-sim (persona), agent (native tool calling), and a tool-sim
                 that answers from an explicit state (StateGen) and returns ops applied in code. Splitting
                 the roles stops the simulator from inventing a helpful user to rescue a failing agent.
"""
from __future__ import annotations

import json
import random

from .banks import ObsBank, UserBank, format_obs_examples
from .config import Config
from .llm import chat_json, get_llm
from .personas import PERSONAS, UNIVERSAL_RULES
from .prompts import (
    BLOCK_FAILURE,
    BLOCK_FORMAT,
    BLOCK_PERSONA,
    BLOCK_RETRIEVAL,
    BLOCK_SPEC,
    FAILURE_RULE,
    SIMIA_ENV,
    SIMIA_GEN,
    STRATEGIES,
    TOOL_SIM,
    USER_SIM,
)
from .schema import from_sharegpt, to_openai_messages, to_openai_tools
from .simia import process_conversation, should_delete_conversation
from .simia_text import build_sample_text, parse_simia_text, reference_text, tools_text
from .state import apply_ops

DONE = "[DONE]"


def _rng(cfg: Config, job: dict) -> random.Random:
    return random.Random(f"{cfg.random_seed}:{job['job_id']}")


def _examples(lines: list[str]) -> str:
    return "\n".join(f"- {u[:300]}" for u in lines) or "(none)"


def simia_prompt(cfg: Config, job: dict, seed: dict, seeds_by_id: dict, obs: ObsBank, users: UserBank) -> str:
    rng = _rng(cfg, job)
    blocks = [BLOCK_FORMAT] if cfg.simia_prompt == "fixed" else []
    if cfg.features.strategies and not job.get("spec"):
        blocks.append(f"\n## Generation Strategy (overrides requirement 1):\n{STRATEGIES[job['strategy']]}\n")
        if job.get("second_seed_id"):
            blocks.append("\n## Second Example Trajectory:\n"
                          f"{build_sample_text(seeds_by_id[job['second_seed_id']], include_system=False)}\n")
    if job.get("spec"):
        blocks.append(BLOCK_SPEC.format(spec_json=json.dumps(job["spec"], ensure_ascii=False, indent=1)))
    if job.get("persona"):
        blocks.append(BLOCK_PERSONA.format(persona=job["persona"], persona_desc=PERSONAS[job["persona"]],
                                           universal_rules=UNIVERSAL_RULES,
                                           user_examples=_examples(users.sample(cfg.real_user_examples, rng, seed["id"]))))
    if job.get("failure"):
        blocks.append(BLOCK_FAILURE.format(at_call=job["failure"]["at_call"], failure_type=job["failure"]["type"]))
    if cfg.features.retrieval:
        ex = obs.examples_for_tools(seed["tools"], cfg.retrieved_obs_per_tool, rng)
        if ex:
            blocks.append(BLOCK_RETRIEVAL.format(obs_examples=format_obs_examples(ex)))
    return SIMIA_GEN.format(sample_text=build_sample_text(seed), available_tools=tools_text(seed["tools"]),
                            extra_blocks="".join(blocks))


def _unusable(conv: list[dict], deleted: bool) -> str | None:
    if deleted:
        return "simia_markup_filter"
    if not conv or conv[0]["from"] != "human":
        return "no_leading_user_turn"
    if conv[-1]["from"] != "gpt":
        return "no_final_reply"
    return None


def generate_simia(cfg: Config, job: dict, seed: dict, seeds_by_id: dict, obs: ObsBank, users: UserBank) -> dict:
    prompt = simia_prompt(cfg, job, seed, seeds_by_id, obs, users)
    discarded = []
    for attempt in range(1, max(1, cfg.generation_attempts) + 1):
        # a unique tag per job and attempt: identical prompts must not collapse into one cached answer
        tagged = f"{prompt}\n(Generation id: {job['job_id']}/{attempt})"
        out = get_llm(cfg, "generator").chat([{"role": "user", "content": tagged}])
        conv = parse_simia_text(out["content"])
        deleted = should_delete_conversation(conv)  # Simia drops conversations with leaked tool markup
        conv = process_conversation(conv)            # Simia's argument repair (string -> dict, quotes, empty)
        problem = _unusable(conv, deleted)
        if problem is None or attempt == max(1, cfg.generation_attempts):
            break
        discarded.append({"attempt": attempt, "problem": problem, "raw_output": out["content"]})
    trace = trace_from_raw(job, seed, out["content"])
    trace["meta"].update(attempts=attempt, discarded_attempts=discarded)
    return trace


def trace_from_raw(job: dict, seed: dict, raw: str) -> dict:
    """Simia text output -> canonical trace (also used to re-parse saved outputs after parser fixes)."""
    conv = parse_simia_text(raw)
    deleted = should_delete_conversation(conv)
    conv = process_conversation(conv)
    trace = from_sharegpt({"id": job["job_id"], "system": seed.get("system", ""), "tools": seed["tools"],
                           "conversations": conv}, 0)
    trace["meta"] = {"raw_turns": len(conv), "simia_deleted": deleted, "raw_output": raw}
    return trace


def _hermes(content: str, call: dict | None) -> str:
    """Agent turn as Simia-RL's simulator expects it: text plus a <tool_call> block."""
    if call is None:
        return content
    body = json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)
    return f"{content}\n<tool_call>\n{body}\n</tool_call>".strip()


def generate_simia_env(cfg: Config, job: dict, seed: dict) -> dict:
    """Simia-RL's simulated environment (one model plays user and tools, seed as reference) with a teacher agent."""
    agent, env = get_llm(cfg, "agent"), get_llm(cfg, "tool_sim")
    first = next((m["content"] for m in seed["messages"] if m["role"] == "user"), None)
    if first is None:
        raise ValueError(f"seed {seed['id']} has no user message")
    messages: list[dict] = [{"role": "user", "content": first}]  # Simia-RL starts from the seed's first message
    history = [f"User/Tool response: {first}"]
    ref = reference_text(seed)
    ended = "max_steps"

    def env_reply(agent_message: str) -> tuple[str, bool]:
        text = env.chat([{"role": "user", "content": SIMIA_ENV.format(
            system_prompt=seed.get("system", ""), ref_conv_text=ref, history_text="\n\n".join(history),
            agent_message=agent_message)}])["content"]
        return text.replace("[TERMINATE]", "").strip(), "[TERMINATE]" in text

    for _ in range(cfg.max_user_turns * cfg.max_tool_rounds):
        out = agent.chat(to_openai_messages(seed.get("system", ""), messages), tools=to_openai_tools(seed["tools"]))
        msg = {"role": "assistant", "content": out["content"]}
        if out["tool_calls"]:
            msg["tool_calls"] = out["tool_calls"]
        messages.append(msg)
        terminate = False
        for tc in out["tool_calls"] or [None]:
            agent_message = _hermes(out["content"], tc)
            history.append(f"Assistant: {agent_message}")
            reply, terminate = env_reply(agent_message)
            if tc is not None:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": reply})
                history.append(f"User/Tool response: {reply}")
            elif reply and not terminate:
                messages.append({"role": "user", "content": reply})
                history.append(f"User/Tool response: {reply}")
            if terminate:
                break
        if terminate:
            ended = "terminate"
            break
    return {"id": job["job_id"], "system": seed.get("system", ""), "tools": seed["tools"], "messages": messages,
            "meta": {"ended": ended}}


def _user_view(messages: list[dict]) -> list[dict]:
    """What the simulated user sees: their own turns as 'assistant', the agent's visible replies as 'user'."""
    view = []
    for m in messages:
        if m["role"] == "user":
            view.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("content"):
            view.append({"role": "user", "content": m["content"]})
    return view


def generate_loop(cfg: Config, job: dict, seed: dict, obs: ObsBank, users: UserBank) -> dict:
    rng = _rng(cfg, job)
    spec = job["spec"]
    persona = job.get("persona", "cooperative")
    agent, user_llm, tool_llm = get_llm(cfg, "agent"), get_llm(cfg, "user_sim"), get_llm(cfg, "tool_sim")
    tools = {t["name"]: t for t in seed["tools"]}
    user_system = USER_SIM.format(goal=spec.get("goal", ""), user_facts=json.dumps(spec.get("user_facts", {}), ensure_ascii=False),
                                  persona=persona, persona_desc=PERSONAS[persona], universal_rules=UNIVERSAL_RULES,
                                  user_examples=_examples(users.sample(cfg.real_user_examples, rng, seed["id"])))
    state = json.loads(json.dumps(spec.get("initial_state") or {}))
    messages: list[dict] = []
    n_calls, state_errors, ended = 0, [], "max_user_turns"

    def user_turn() -> str | None:
        view = _user_view(messages) or [{"role": "user", "content": "(The assistant is waiting. Write your first message.)"}]
        if view[-1]["role"] != "user":
            view.append({"role": "user", "content": "(continue)"})
        text = user_llm.chat([{"role": "system", "content": user_system}, *view])["content"].strip()
        return None if not text or text.startswith(DONE) else text.replace(DONE, "").strip()

    for _ in range(cfg.max_user_turns):
        text = user_turn()
        if text is None:
            ended = "user_done"
            break
        messages.append({"role": "user", "content": text})
        for _ in range(cfg.max_tool_rounds):
            out = agent.chat(to_openai_messages(seed.get("system", ""), messages), tools=to_openai_tools(seed["tools"]))
            msg = {"role": "assistant", "content": out["content"]}
            if out["tool_calls"]:
                msg["tool_calls"] = out["tool_calls"]
            messages.append(msg)
            if not out["tool_calls"]:
                break
            for tc in out["tool_calls"]:
                n_calls += 1
                if tc["name"] not in tools:
                    result = json.dumps({"error": f"unknown tool: {tc['name']}"})
                else:
                    fail = job.get("failure") if job.get("failure", {}).get("at_call") == n_calls else None
                    sim = chat_json(tool_llm, "You simulate a software backend precisely.", TOOL_SIM.format(
                        tool_spec=json.dumps(tools[tc["name"]], ensure_ascii=False),
                        call_name=tc["name"], call_args=json.dumps(tc["arguments"], ensure_ascii=False),
                        state_json=json.dumps(state, ensure_ascii=False),
                        obs_examples=format_obs_examples(obs.retrieve(tc["name"], tc["arguments"], cfg.retrieved_obs_per_tool, rng)),
                        failure_rule=FAILURE_RULE.format(failure_type=fail["type"]) if fail else ""))
                    res = sim.get("result", "") if isinstance(sim, dict) else sim
                    result = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
                    if not fail and isinstance(sim, dict):
                        state, errs = apply_ops(state, sim.get("ops") or [])
                        state_errors += errs
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": result})
        else:
            ended = "max_tool_rounds"
            break
    return {"id": job["job_id"], "system": seed.get("system", ""), "tools": seed["tools"], "messages": messages,
            "meta": {"final_state": state, "state_errors": state_errors, "ended": ended}}


def generate(cfg: Config, job: dict, seeds_by_id: dict, obs: ObsBank, users: UserBank) -> dict:
    seed = seeds_by_id[job["seed_id"]]
    if job["mode"] == "split":
        trace = generate_loop(cfg, job, seed, obs, users)
    elif job["mode"] == "simia_env":
        trace = generate_simia_env(cfg, job, seed)
    else:
        trace = generate_simia(cfg, job, seed, seeds_by_id, obs, users)
    trace["meta"].update({k: job.get(k) for k in ("job_id", "seed_id", "second_seed_id", "strategy", "mode",
                                                  "persona", "failure", "spec")})
    return trace
