"""Thin OpenAI-compatible chat client with retries, JSON extraction and a swappable factory (for tests)."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from .config import Config, ModelCfg


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> dict: ...


class OpenAILLM:
    """Returns {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}."""

    def __init__(self, cfg: ModelCfg, retries: int = 4):
        from openai import OpenAI

        self.cfg = cfg
        self.retries = retries
        self.client = OpenAI(base_url=cfg.base_url, api_key=os.environ.get(cfg.api_key_env, "EMPTY"))

    def chat(self, messages, tools=None, json_mode=False):
        kwargs: dict[str, Any] = {"model": self.cfg.model, "messages": messages,
                                  self.cfg.max_tokens_param: self.cfg.max_tokens, **self.cfg.extra}
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if tools:
            kwargs["tools"] = tools
        if json_mode and self.cfg.json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                msg = self.client.chat.completions.create(**kwargs).choices[0].message
                calls = []
                for tc in msg.tool_calls or []:
                    calls.append({"id": tc.id, "name": tc.function.name,
                                  "arguments": parse_args(tc.function.arguments)})
                return {"content": msg.content or "", "tool_calls": calls}
            except Exception as e:  # noqa: BLE001 - provider SDKs raise many types; retried, then re-raised
                last = e
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")


_factory: Callable[[ModelCfg, str], ChatLLM] = lambda cfg, role: OpenAILLM(cfg)
_cache: dict[str, ChatLLM] = {}
_lock = threading.Lock()


def set_llm_factory(factory: Callable[[ModelCfg, str], ChatLLM]) -> None:
    """Replace how clients are built (tests inject a fake)."""
    global _factory
    with _lock:
        _factory = factory
        _cache.clear()


def get_llm(cfg: Config, role: str) -> ChatLLM:
    with _lock:
        if role not in _cache:
            _cache[role] = _factory(cfg.model_for(role), role)
        return _cache[role]


def parse_args(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        val = json.loads(arguments)
        return val if isinstance(val, dict) else {"_value": val}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": str(arguments)}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in model output (handles fences and leading prose)."""
    text = text.strip()
    for candidate in [text, *_FENCE.findall(text)]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return dec.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in model output: {text[:200]!r}")


def chat_json(llm: ChatLLM, system: str, user: str) -> Any:
    out = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], json_mode=True)
    return extract_json(out["content"])
