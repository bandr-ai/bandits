"""Thin OpenAI-compatible chat client with retries, per-call logging, JSON extraction and a swappable
factory (for tests).

Every attempt (success or failure) is appended to the call log configured with configure_call_log():
role, model, work item, attempt, latency, HTTP status, finish reason, token usage, gateway cost header,
the provider that actually served it, the full request messages and the full response.
"""
from __future__ import annotations

import contextvars
import dataclasses
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


class Truncated(RuntimeError):
    """The response hit the token limit (finish_reason=length); an identical retry would hit it again, so not retried."""


class BudgetExhausted(RuntimeError):
    """The run's budget cannot cover this call. Never retried; callers re-raise it so the run stops cleanly."""


class Budget:
    """Hard spend cap for every model call in a process. Each call reserves its maximum possible cost before it starts
    (prompt + max_tokens at configured prices) and settles to the actual cost after, so concurrent calls together can
    never pass the limit. Once a call is refused the budget stays exhausted: later, cheaper calls are refused too, so a
    run stops instead of trickling on."""

    def __init__(self, limit_usd: float, spent_usd: float = 0.0):
        self.limit, self.spent, self.reserved, self.hit = limit_usd, spent_usd, 0.0, False
        self._lock = threading.Lock()

    def reserve(self, amount: float) -> float:
        with self._lock:
            if self.hit or self.spent + self.reserved + amount > self.limit:
                self.hit = True
                raise BudgetExhausted(f"budget ${self.limit:.4f} reached (spent ${self.spent:.4f}, "
                                      f"in flight ${self.reserved:.4f}, next call up to ${amount:.4f})")
            self.reserved += amount
            return amount

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved -= reserved
            self.spent += actual


_budget: Budget | None = None


def set_budget(budget: Budget | None) -> None:
    """Cap every later model call in this process (None: no cap)."""
    global _budget
    _budget = budget


def get_budget() -> Budget | None:
    return _budget


NON_RETRYABLE = {400, 401, 403, 404, 422}  # a retry of these cannot succeed; 408, 409, 429 and 5xx are retried

from .config import Config, ModelCfg


class ChatLLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, tool_choice: str | None = None) -> dict: ...


class OpenAILLM:
    """Returns {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}. Meters the process budget itself,
    per provider attempt (each retry reserves before it starts and settles to what it was billed)."""

    meters_itself = True

    def __init__(self, cfg: ModelCfg, role: str = "", retries: int = 10):
        from openai import OpenAI

        self.cfg = cfg
        self.role = role
        self.retries = retries
        key = os.environ.get(cfg.api_key_env)
        if not key and cfg.base_url is None:
            raise RuntimeError(f"environment variable {cfg.api_key_env} is not set")
        self.client = OpenAI(base_url=cfg.base_url, api_key=key or "EMPTY", timeout=600, max_retries=0)

    def chat(self, messages, tools=None, json_mode=False, tool_choice=None):
        kwargs: dict[str, Any] = {"model": self.cfg.model, "messages": messages,
                                  self.cfg.max_tokens_param: self.cfg.max_tokens, **self.cfg.extra}
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if json_mode and self.cfg.json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        if self.cfg.no_cache:
            body = dict(kwargs.get("extra_body") or {})
            body["cache"] = {**body.get("cache", {}), "no-cache": True, "no-store": True}
            kwargs["extra_body"] = body
        last: Exception | None = None
        billed = 0.0  # what the provider charged across attempts (failed calls that returned usage are billed)
        budget = _budget
        bound = max_call_cost(self.cfg, messages, tools=tools, extra_params=kwargs) if budget is not None else 0.0
        for attempt in range(1, self.retries + 1):
            held = budget.reserve(bound) if budget is not None else 0.0  # BudgetExhausted: no request is sent
            t0 = time.monotonic()
            row: dict[str, Any] = {"role": self.role, "model": self.cfg.model, "attempt": attempt,
                                   "request": {"messages": messages, "tools": [t["function"]["name"] for t in tools or []],
                                               "json_mode": bool(kwargs.get("response_format")),
                                               # every sampling setting sent (temperature, max tokens, reasoning effort, ...)
                                               "params": {k: v for k, v in kwargs.items() if k not in ("messages", "tools", "model")}}}
            try:
                if self.cfg.stream:
                    raw, content, calls, finish, usage, extra, model, reasoning = self._stream(kwargs)
                else:
                    raw = self.client.chat.completions.with_raw_response.create(**kwargs)
                    resp = raw.parse()
                    choice = resp.choices[0]
                    msg = choice.message
                    content, finish, model = msg.content, choice.finish_reason, resp.model
                    reasoning = (msg.model_extra or {}).get("reasoning_content")
                    calls = [{"id": tc.id, "name": tc.function.name, "arguments": parse_args(tc.function.arguments)}
                             for tc in msg.tool_calls or []]
                    usage = resp.usage.model_dump() if resp.usage else None
                    extra = resp.model_extra or {}
                cost = _float(raw.headers.get("x-litellm-response-cost"))
                if cost is None and isinstance(usage, dict):
                    cost = _float(usage.get("cost"))  # gateways that stream often report cost in usage only
                if cost is None:
                    cost = token_cost(self.cfg, usage)  # direct providers: price x tokens from the config
                row.update(ok=True, status=raw.status_code, latency_s=round(time.monotonic() - t0, 3),
                           finish_reason=finish, usage=usage, cost=cost,
                           call_id=raw.headers.get("x-litellm-call-id"), provider=extra.get("provider"),
                           cache_hit=raw.headers.get("x-litellm-cache-hit"),
                           served_model=model, response={"content": content, "tool_calls": calls, "reasoning": reasoning})
                billed += cost or 0.0
                if budget is not None:
                    budget.settle(held, cost or 0.0)
                    held = 0.0
                if finish == "length":
                    raise Truncated(f"response hit the token limit ({self.cfg.max_tokens})")
                if not (content or "").strip() and not calls:
                    raise EmptyResponse(f"empty response (finish_reason={finish})")
                log_call(row)
                return {"content": content or "", "tool_calls": calls, "cost": billed}
            except Exception as e:  # provider SDKs raise many types; logged, retried, then re-raised
                if budget is not None and held:  # no response came back: nothing billed for this attempt
                    budget.settle(held, 0.0)
                last = e
                row.update(ok=False, latency_s=round(time.monotonic() - t0, 3), error=f"{type(e).__name__}: {e}"[:2000],
                           status=row.get("status") or getattr(e, "status_code", None))
                log_call(row)
                if isinstance(e, Truncated) or row.get("status") in NON_RETRYABLE:
                    e.billed = billed
                    raise
                if attempt < self.retries:
                    time.sleep(min(60, 2 ** attempt))
        err = RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")
        err.billed = billed
        raise err

    def _stream(self, kwargs: dict):
        """One streamed completion folded into the same pieces a plain response gives."""
        raw = self.client.chat.completions.with_raw_response.create(**kwargs, stream=True, stream_options={"include_usage": True})
        parts: list[str] = []
        thinking: list[str] = []
        tcs: dict[int, dict] = {}
        finish = usage = model = None
        extra: dict = {}
        for chunk in raw.parse():
            model = chunk.model or model
            if chunk.usage:
                usage = chunk.usage.model_dump()
            extra = {**extra, **(chunk.model_extra or {})}
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            finish = ch.finish_reason or finish
            if ch.delta.content:
                parts.append(ch.delta.content)
            if (ch.delta.model_extra or {}).get("reasoning_content"):
                thinking.append(ch.delta.model_extra["reasoning_content"])
            for d in ch.delta.tool_calls or []:
                slot = tcs.setdefault(d.index, {"id": None, "name": "", "args": ""})
                slot["id"] = d.id or slot["id"]
                if d.function:
                    slot["name"] += d.function.name or ""
                    slot["args"] += d.function.arguments or ""
        calls = [{"id": s["id"], "name": s["name"], "arguments": parse_args(s["args"])} for _, s in sorted(tcs.items())]
        return raw, "".join(parts), calls, finish, usage, extra, model, "".join(thinking) or None


def token_cost(cfg: ModelCfg, usage: Any) -> float | None:
    """Cost from token counts and the configured prices; None when prices or usage are missing."""
    if not isinstance(usage, dict) or cfg.price_input_per_mtok is None or cfg.price_output_per_mtok is None:
        return None
    prompt = usage.get("prompt_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    cached_price = cfg.price_cached_input_per_mtok if cfg.price_cached_input_per_mtok is not None else cfg.price_input_per_mtok
    out = usage.get("completion_tokens") or 0  # includes reasoning tokens on OpenAI-style APIs
    return round(((prompt - cached) * cfg.price_input_per_mtok + cached * cached_price + out * cfg.price_output_per_mtok) / 1e6, 8)


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
    """One client per (role, model settings): two configs in one process (A/B arms) never share a client whose
    settings differ."""
    m = cfg.model_for(role)
    key = (role, json.dumps(dataclasses.asdict(m), sort_keys=True, default=str)) if dataclasses.is_dataclass(m) else (role, repr(m))
    with _lock:
        if key not in _cache:
            _cache[key] = _factory(m, role)
        return _Metered(_cache[key], m)


OUTPUT_LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def max_call_cost(cfg: ModelCfg, messages: list[dict], tools: list | None = None, extra_params: dict | None = None) -> float:
    """Upper bound of one request's cost. Prompt: the UTF-8 byte length of everything sent (messages, tools, extra
    params); a token is never shorter than one byte, so bytes bound tokens from above. Output: the largest output
    limit the request carries (the config's, or an override in `extra`). 0 when the model has no configured prices
    (preflight refuses a budgeted run without them)."""
    if not isinstance(cfg, ModelCfg):
        return 0.0
    sent = {"messages": messages, "tools": tools,
            "params": {k: v for k, v in (extra_params or {}).items() if k != "messages"}}
    prompt = len(json.dumps(sent, ensure_ascii=False, default=str).encode("utf-8"))
    limits = [cfg.max_tokens, *(v for src in (cfg.extra, extra_params or {}) for k, v in src.items()
                                if k in OUTPUT_LIMIT_KEYS and isinstance(v, int))]
    return token_cost(cfg, {"prompt_tokens": prompt, "completion_tokens": max(limits)}) or 0.0


class _Metered:
    """Wraps any client so the process budget (set_budget) gates and accounts every call, fakes included."""

    def __init__(self, inner: ChatLLM, cfg: ModelCfg):
        self.inner, self.cfg = inner, cfg

    def chat(self, messages, *args, **kwargs):
        budget = _budget
        if budget is None or getattr(self.inner, "meters_itself", False):  # the real client meters each attempt
            return self.inner.chat(messages, *args, **kwargs)
        tools = kwargs.get("tools") if "tools" in kwargs else (args[0] if args else None)
        held = budget.reserve(max_call_cost(self.cfg, messages, tools=tools))
        try:
            out = self.inner.chat(messages, *args, **kwargs)
        except Exception as e:
            budget.settle(held, getattr(e, "billed", 0.0) or 0.0)
            raise
        cost = out.get("cost") if isinstance(out, dict) else None
        budget.settle(held, held if cost is None else cost)  # a client that reports no cost is charged its maximum
        return out


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
