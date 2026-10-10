"""Canonical trace format and converters (Simia/ShareGPT <-> canonical <-> OpenAI chat)."""
from __future__ import annotations

import json
import re
from typing import Any

from .llm import parse_args

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def normalize_tools(tools: Any, defs: dict[str, dict] | None = None) -> list[dict]:
    """Accepts a list (or JSON string) of tool schemas, OpenAI {"type":"function","function":...} entries,
    a {name: schema} dict, or bare tool names resolved through `defs`."""
    if isinstance(tools, str):
        tools = json.loads(tools) if tools.strip() else []
    if isinstance(tools, dict):
        tools = list(tools.values())
    out = []
    for t in tools or []:
        if isinstance(t, str):
            if not defs or t not in defs:
                raise ValueError(f"tool {t!r} is only a name; pass tool_defs_path with its schema")
            t = defs[t]
        if "function" in t and isinstance(t["function"], dict):
            t = t["function"]
        out.append({"name": t["name"], "description": t.get("description", ""),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}}})
    return out


_ROLE_ALIASES = {"human": "user", "ai": "assistant", "gpt": "assistant", "function": "tool"}


def _text_and_reasoning(content: Any) -> tuple[str, str]:
    """Plain string, or a list of content blocks (OpenAI / Anthropic / LangChain style)."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return content, ""
    texts, thoughts = [], []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict):
            if block.get("type") in ("thinking", "reasoning"):
                thoughts.append(block.get("thinking") or block.get("reasoning") or block.get("text") or "")
            elif "text" in block:
                texts.append(block["text"])
    return "\n".join(t for t in texts if t), "\n".join(t for t in thoughts if t)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Coerce roles/arguments, assign missing tool-call ids, link tool results to calls in order."""
    out, pending, n = [], [], 0
    for m in messages:
        role = _ROLE_ALIASES.get(m.get("role"), m.get("role"))
        if role != "tool":
            text, thought = _text_and_reasoning(m.get("content"))
            m = {**m, "content": text, "reasoning": m.get("reasoning") or thought}
        if role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                if "function" in tc:  # OpenAI shape
                    tc = {"id": tc.get("id"), "name": tc["function"]["name"],
                          "arguments": tc["function"].get("arguments")}
                elif "args" in tc and "arguments" not in tc:  # LangChain shape
                    tc = {**tc, "arguments": tc["args"]}
                n += 1
                calls.append({"id": tc.get("id") or f"call_{n}", "name": tc["name"],
                              "arguments": parse_args(tc.get("arguments"))})
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning"):
                msg["reasoning"] = m["reasoning"]
            if calls:
                msg["tool_calls"] = calls
            pending = [c for c in calls]
            out.append(msg)
        elif role == "tool":
            call = None
            if m.get("tool_call_id"):
                call = next((c for c in pending if c["id"] == m["tool_call_id"]), None)
            if call is None and pending:
                call = pending[0]
            if call is not None:
                pending.remove(call)
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({"role": "tool", "tool_call_id": call["id"] if call else m.get("tool_call_id", ""),
                        "name": call["name"] if call else m.get("name", ""), "content": content})
        elif role in ("user", "system"):
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def from_sharegpt(rec: dict, idx: int) -> dict:
    msgs: list[dict] = []
    for turn in rec.get("conversations", []):
        src, val = turn.get("from"), turn.get("value", "")
        if src == "human":
            msgs.append({"role": "user", "content": val})
        elif src == "gpt":
            msgs.append({"role": "assistant", "content": val})
        elif src == "function_call":
            reasoning = " ".join(s.strip() for s in _THINK.findall(val))
            body = _THINK.sub("", val).strip()
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = []
            calls = parsed if isinstance(parsed, list) else [parsed]
            msg = {"role": "assistant", "content": "",
                   "tool_calls": [{"name": c.get("name", ""), "arguments": c.get("arguments", {})}
                                  for c in calls if isinstance(c, dict)]}
            if reasoning:
                msg["reasoning"] = reasoning
            msgs.append(msg)
        elif src == "observation":
            pending = msgs[-1].get("tool_calls", []) if msgs and msgs[-1]["role"] == "assistant" else []
            try:
                parsed = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if len(pending) > 1 and isinstance(parsed, list) and len(parsed) == len(pending):
                for p in parsed:
                    msgs.append({"role": "tool", "content": p})
            else:
                msgs.append({"role": "tool", "content": val})
        elif src == "system":
            continue
    return {"id": str(rec.get("id", f"seed_{idx}")), "system": rec.get("system", ""),
            "tools": normalize_tools(rec.get("tools", [])), "messages": normalize_messages(msgs),
            "meta": rec.get("meta", {})}


def from_canonical(rec: dict, idx: int, defs: dict[str, dict] | None = None) -> dict:
    raw = list(rec.get("messages") or rec.get("final_messages") or [])
    turns = rec.get("turns")
    if "final_messages" in rec and isinstance(turns, list) and turns and isinstance(turns[-1], dict):
        # LLM-gateway log: final_messages is the last request's input; the last turn holds its response
        last = turns[-1]
        if last.get("content") or last.get("tool_calls"):
            raw.append({"role": "assistant", "content": last.get("content"), "tool_calls": last.get("tool_calls") or []})
    system = rec.get("system", "")
    sys_msgs = [m for m in raw if m.get("role") == "system"]
    if sys_msgs and not system:
        system = _text_and_reasoning(sys_msgs[0].get("content"))[0]
    rid = rec.get("id") or rec.get("session_id") or f"seed_{idx}"
    return {"id": str(rid), "system": system, "tools": normalize_tools(rec.get("tools", []), defs),
            "messages": [m for m in normalize_messages(raw) if m["role"] != "system"], "meta": rec.get("meta", {})}


def load_trace(rec: dict, idx: int, fmt: str = "auto", defs: dict[str, dict] | None = None) -> dict:
    if fmt == "auto":
        fmt = "sharegpt" if "conversations" in rec else "canonical"
    return from_sharegpt(rec, idx) if fmt == "sharegpt" else from_canonical(rec, idx, defs)


def to_sharegpt(trace: dict) -> dict:
    conv = []
    for m in trace["messages"]:
        if m["role"] == "user":
            conv.append({"from": "human", "value": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{"name": c["name"], "arguments": c["arguments"]} for c in m["tool_calls"]]
            body = json.dumps(calls[0] if len(calls) == 1 else calls, ensure_ascii=False)
            if m.get("reasoning"):
                body = f"<think>\n{m['reasoning']}\n</think>\n{body}"
            conv.append({"from": "function_call", "value": body})
        elif m["role"] == "assistant":
            conv.append({"from": "gpt", "value": m["content"]})
        elif m["role"] == "tool":
            conv.append({"from": "observation", "value": m["content"]})
    return {"id": trace["id"], "system": trace.get("system", ""),
            "tools": json.dumps(trace.get("tools", []), ensure_ascii=False), "conversations": conv}


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"],
                                                   "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                     for c in m["tool_calls"]]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": t} for t in tools]


def render(trace_or_messages: dict | list, numbered: bool = True, max_obs_chars: int = 4000) -> str:
    """Human-readable transcript for prompts."""
    msgs = trace_or_messages["messages"] if isinstance(trace_or_messages, dict) else trace_or_messages
    lines = []
    for i, m in enumerate(msgs):
        tag = f"[{i}] " if numbered else ""
        if m["role"] == "assistant":
            if m.get("reasoning"):
                lines.append(f"{tag}ASSISTANT (thinking): {m['reasoning']}")
            if m.get("content"):
                lines.append(f"{tag}ASSISTANT: {m['content']}")
            for c in m.get("tool_calls", []):
                lines.append(f"{tag}TOOL_CALL {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")
        elif m["role"] == "tool":
            lines.append(f"{tag}TOOL_RESULT {m.get('name', '')}: {m['content'][:max_obs_chars]}")
        else:
            lines.append(f"{tag}{m['role'].upper()}: {m['content']}")
    return "\n".join(lines)


def tool_calls(trace: dict) -> list[dict]:
    return [c for m in trace["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]
