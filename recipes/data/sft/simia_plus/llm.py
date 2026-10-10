"""Thin OpenAI-compatible chat client with retries, per-call logging, JSON extraction and a swappable
factory (for tests).

Every attempt (success or failure) is appended to the call log configured with configure_call_log():
role, model, work item, attempt, latency, HTTP status, finish reason, token usage, gateway cost header,
the provider that actually served it, the full request messages and the full response.
"""
from __future__ import annotations

import contextvars
import datetime
import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

current_item: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_item", default=None)
_log_path: Path | None = None
_log_lock = threading.Lock()


def configure_call_log(path: str | Path | None) -> None:
    global _log_path
    _log_path = Path(path) if path else None
    if _log_path:
        _log_path.parent.mkdir(parents=True, exist_ok=True)


def log_call(row: dict) -> None:
    if _log_path is None:
        return
    row = {"ts": datetime.datetime.now(datetime.UTC).isoformat(), "item": current_item.get(), **row}
    with _log_lock, _log_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class EmptyResponse(RuntimeError):
    """The model returned neither text nor tool calls (e.g. reasoning ran out of tokens)."""

from .config import Config, ModelCfg


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> dict: ...


class OpenAILLM:
    """Returns {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}."""

    def __init__(self, cfg: ModelCfg, role: str = "", retries: int = 4):
        from openai import OpenAI

        self.cfg = cfg
        self.role = role
        self.retries = retries
        key = os.environ.get(cfg.api_key_env)
        if not key and cfg.base_url is None:
            raise RuntimeError(f"environment variable {cfg.api_key_env} is not set")
        self.client = OpenAI(base_url=cfg.base_url, api_key=key or "EMPTY", timeout=600, max_retries=0)

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
        for attempt in range(1, self.retries + 1):
            t0 = time.monotonic()
            row: dict[str, Any] = {"role": self.role, "model": self.cfg.model, "attempt": attempt,
                                   "request": {"messages": messages, "tools": [t["function"]["name"] for t in tools or []],
                                               "json_mode": bool(kwargs.get("response_format"))}}
            try:
                raw = self.client.chat.completions.with_raw_response.create(**kwargs)
                resp = raw.parse()
                choice = resp.choices[0]
                msg = choice.message
                calls = [{"id": tc.id, "name": tc.function.name, "arguments": parse_args(tc.function.arguments)}
                         for tc in msg.tool_calls or []]
                usage = resp.usage.model_dump() if resp.usage else None
                extra = resp.model_extra or {}
                row.update(ok=True, status=raw.status_code, latency_s=round(time.monotonic() - t0, 3),
                           finish_reason=choice.finish_reason, usage=usage,
                           cost=_float(raw.headers.get("x-litellm-response-cost")),
                           call_id=raw.headers.get("x-litellm-call-id"), provider=extra.get("provider"),
                           served_model=resp.model, response={"content": msg.content, "tool_calls": calls})
                if not (msg.content or "").strip() and not calls:
                    raise EmptyResponse(f"empty response (finish_reason={choice.finish_reason})")
                log_call(row)
                return {"content": msg.content or "", "tool_calls": calls}
            except Exception as e:  # noqa: BLE001 - provider SDKs raise many types; logged, retried, then re-raised
                last = e
                row.update(ok=False, latency_s=round(time.monotonic() - t0, 3), error=f"{type(e).__name__}: {e}"[:2000],
                           status=row.get("status") or getattr(e, "status_code", None))
                log_call(row)
                if attempt < self.retries:
                    time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")


def _float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except ValueError:
        return None


_factory: Callable[[ModelCfg, str], ChatLLM] = lambda cfg, role: OpenAILLM(cfg, role)
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
