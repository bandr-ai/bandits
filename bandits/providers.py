"""Which provider a model string names, and the credentials to reach it.

Every model call in this package goes through LiteLLM, directly for the judge
and through DSPy for the RLM and emulation stages. This is the one place a
model string is turned into what LiteLLM needs, so a new provider is a
different ``--model`` rather than a code change.

A model string is ``<provider>/<model>`` in LiteLLM's own naming:
``anthropic/claude-sonnet-5``, ``openai/gpt-5``, ``hosted_vllm/my-judge``,
``litellm_proxy/judge``. Fireworks' own ``accounts/fireworks/models/...`` form
is kept as it is, because that string is already recorded in saved artifacts,
compared on ``--resume``, and hashed into prompt digests. It is resolved at call
time only: an artifact stores what the user typed, never the wire id.

A bare name is refused rather than guessed. LiteLLM would route ``gpt-4o`` to
OpenAI on its own, and would also route a name it half-recognises to whichever
provider it matched first; a judge silently answered by the wrong backend is
worse than an error that names the fix.

LiteLLM is an optional extra and is imported only when a model is actually
called, so the core install and the Jev recipe's environment never need it.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIREWORKS_DEFAULT = "accounts/fireworks/models/deepseek-v4p1-flash"
"""The judge's default: Nemotron loops at temperature 0 until it runs out of
tokens, and this does not."""

RLM_FIREWORKS_DEFAULT = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"
"""RLM mining and its audit stay on Nemotron: the judge's move to DeepSeek
did not include them."""


def default_model(fallback: str = FIREWORKS_DEFAULT) -> str:
    """``BANDITS_MODEL`` when set, which moves every stage at once so a
    different provider does not mean passing ``--model`` to every command;
    otherwise the stage's own ``fallback``."""
    return os.environ.get("BANDITS_MODEL") or fallback


DEFAULT_MODEL = default_model()

_REQUIRED: dict[str, dict[str, str]] = {
    "fireworks_ai": {"FIREWORKS_API_KEY": "api_key"},
    "hosted_vllm": {"HOSTED_VLLM_API_BASE": "api_base"},
    "litellm_proxy": {"LITELLM_PROXY_API_BASE": "api_base"},
}
"""What these providers need, which LiteLLM does not report as missing: it says
all three need nothing even with nothing set, which would surface as a 401 or a
connection error per call instead of one clear error before the first."""

_OPTIONAL: dict[str, dict[str, str]] = {
    "hosted_vllm": {"HOSTED_VLLM_API_KEY": "api_key"},
    "litellm_proxy": {"LITELLM_PROXY_API_KEY": "api_key"},
}
"""Passed when set, for a server that asks for a key."""


class ProviderError(RuntimeError):
    """A model string names no provider, or its credentials are missing."""


@dataclass(frozen=True)
class ModelRef:
    model: str
    """What the user typed, and what artifacts and the ledger record."""
    litellm_id: str
    """What LiteLLM is called with."""
    provider: str
    """LiteLLM's provider name: ``fireworks_ai``, ``anthropic``, ``hosted_vllm``..."""


def load_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise ProviderError("calling a model needs the 'llm' extra: uv sync --extra llm") from exc
    return litellm


def resolve(model: str) -> ModelRef:
    """Name the provider behind ``model``, or refuse it."""
    if model.startswith("accounts/"):
        return ModelRef(model, f"fireworks_ai/{model}", "fireworks_ai")
    prefix, separator, _ = model.partition("/")
    known = {str(getattr(p, "value", p)) for p in load_litellm().provider_list}
    if not separator or prefix not in known:
        raise ProviderError(
            f"model {model!r} names no provider; write it as <provider>/<model>, "
            f"e.g. openai/{model} or anthropic/{model} (LiteLLM's provider names)"
        )
    return ModelRef(model, model, prefix)


def env_value(name: str) -> str | None:
    """``name`` from the environment, else from ``./.env``.

    Read per call so a key is never held in an artifact. Only the one line is
    parsed out of ``.env``; nothing in it is executed.
    """
    value = os.environ.get(name)
    if value:
        return value
    env_path = Path(".env")
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition("=")
            if separator and key.strip() == name:
                return raw.strip().strip("'\"") or None
    return None


def _argument(name: str) -> str | None:
    """The LiteLLM keyword an environment variable is passed as, if any."""
    if name.endswith(("_API_BASE", "_BASE_URL")):
        return "api_base"
    if "KEY" in name or "TOKEN" in name:
        return "api_key"
    return None


def credentials(ref: ModelRef, *, api_key: str | None = None) -> dict[str, str]:
    """Keyword arguments carrying whatever this provider needs and the
    environment does not already supply to LiteLLM.

    The providers in ``_REQUIRED`` always get theirs passed explicitly. For the
    rest, a variable already exported is left for LiteLLM to read itself. One that
    is only in ``.env`` is passed explicitly: an ``*_API_BASE`` as
    ``api_base``, a key or token as ``api_key``. Anything else (a cloud
    provider's several credentials, say) must be exported.

    An explicit ``api_key`` replaces the key lookup only. A vLLM server's base
    URL is still needed, and still read from ``.env``, when the caller brought
    its own key.
    """
    if ref.provider in _REQUIRED:
        needed = dict(_REQUIRED[ref.provider])
        optional = dict(_OPTIONAL.get(ref.provider, {}))
    else:
        missing = load_litellm().validate_environment(ref.litellm_id).get("missing_keys") or []
        needed = {name: _argument(name) for name in missing}
        optional = {}

    found: dict[str, str] = {}
    for name, argument in needed.items():
        if argument == "api_key" and api_key:
            continue
        value = env_value(name)
        if not value:
            raise ProviderError(f"{name} is not set and was not found in .env ({ref.model})")
        if argument is None:
            raise ProviderError(f"{name} was found in .env, but {ref.provider} needs it exported")
        found[argument] = value
    for name, argument in optional.items():
        value = env_value(name)
        if value:
            found[argument] = value
    if api_key:
        found["api_key"] = api_key
    return found


_request_body: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "bandits_request_body", default=None
)
"""Where LiteLLM's pre-call hook leaves the body it is about to send, for the
call that set the slot. A context variable, so concurrent calls never cross."""


def _install_body_capture(litellm: Any) -> None:
    """Register (once) a LiteLLM hook that hands each request body to its call.

    ``complete_input_dict`` is the provider request body LiteLLM built — the
    same body the HTTP layer sends, verified against a captured request — so a
    setting LiteLLM accepted and then dropped shows up as absent here.
    """
    from litellm.integrations.custom_logger import CustomLogger

    class _BodyCapture(CustomLogger):
        bandits_body_capture = True

        def _keep(self, kwargs: Any) -> None:
            slot = _request_body.get()
            if slot is None:
                return
            body = (kwargs.get("additional_args") or {}).get("complete_input_dict")
            if body is not None:
                slot["body"] = json.loads(json.dumps(body, default=str))

        def log_pre_api_call(self, model, messages, kwargs):
            self._keep(kwargs)

        async def async_log_pre_api_call(self, model, messages, kwargs):
            self._keep(kwargs)

    if not any(getattr(cb, "bandits_body_capture", False) for cb in litellm.callbacks):
        litellm.callbacks.append(_BodyCapture())


def dspy_lm(
    model: str, *, api_key: str | None = None, call_guard: Any = None, **kwargs: Any
) -> Any:
    """A ``dspy.LM`` for ``model``. The caller has already imported DSPy.

    Every request is recorded at ``forward`` — the one method each root,
    subcall, adapter fallback and extraction passes through — with a call id
    shared by its start and its single terminal row, the settings DSPy passed,
    the body LiteLLM actually built, and the whole returned response. Capture
    there does not depend on DSPy's retained history, which can be disabled or
    evict entries before they are read.

    ``call_guard``, when given, admits each request before dispatch, bounds it
    by the remaining wall time, and settles what it cost afterwards.
    """
    import dspy

    from bandits import ledger
    from bandits.analyze.rlm_history import record_history

    try:
        import litellm

        _install_body_capture(litellm)
    except ImportError:  # pragma: no cover - DSPy depends on LiteLLM
        pass

    class RecordedLM(dspy.LM):
        def __init__(self, *args, **init_kwargs):
            super().__init__(*args, **init_kwargs)
            self.call_log: list[dict[str, Any]] = []
            """One entry per request that reached ``forward``, failed or not.
            Append-only and independent of DSPy's history settings, so
            per-prediction spend never depends on retained history."""

        def __call__(self, *args, **call_kwargs):
            before = len(self.history)
            try:
                return super().__call__(*args, **call_kwargs)
            finally:
                # Only entries ``forward`` did not already record: a backend
                # whose call never reached ``forward`` still leaves a row.
                record_history(self.history[before:], language_model=self)

        def _begin(self, prompt, messages, call_kwargs):
            ledger.raise_if_failed()
            ticket = None
            if call_guard is not None:
                ticket = call_guard.admit(
                    prompt=prompt,
                    messages=messages,
                    max_tokens=call_kwargs.get("max_tokens", self.kwargs.get("max_tokens")),
                )
                if "timeout" not in call_kwargs:
                    call_kwargs["timeout"] = max(1.0, call_guard.remaining_seconds())
            call_id = uuid.uuid4().hex
            settings = {
                key: value
                for key, value in {**self.kwargs, **call_kwargs}.items()
                if not key.startswith("api_")
            }
            ledger.record(
                {
                    "event_type": "model_call_start",
                    "call_id": call_id,
                    "model": model,
                    "request": {"prompt": prompt, "messages": messages, "settings": settings},
                }
            )
            slot: dict[str, Any] = {}
            return call_id, ticket, slot, _request_body.set(slot), time.monotonic(), settings

        def _fail(self, state, prompt, messages, exc):
            call_id, ticket, slot, token, started, settings = state
            _request_body.reset(token)
            if call_guard is not None and ticket is not None:
                call_guard.settle(ticket, None)
            self.call_log.append(
                {"call_id": call_id, "usage": None, "cost": None, "error": str(exc)}
            )
            ledger.record(
                {
                    "event_type": "model_call_error",
                    "call_id": call_id,
                    "model": model,
                    "request": {"prompt": prompt, "messages": messages, "settings": settings},
                    "effective_request": slot.get("body"),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "duration_seconds": round(time.monotonic() - started, 4),
                }
            )

        def _finish(self, state, prompt, messages, response):
            from bandits.analyze.rlm_history import response_record

            call_id, ticket, slot, token, started, settings = state
            _request_body.reset(token)
            if call_guard is not None and ticket is not None:
                call_guard.settle(ticket, response)
            ledger.record(
                {
                    "event_type": "model_call",
                    "provider": "dspy",
                    "call_id": call_id,
                    "model": model,
                    "request": {"prompt": prompt, "messages": messages, "settings": settings},
                    "effective_request": slot.get("body"),
                    **response_record(response),
                    "duration_seconds": round(time.monotonic() - started, 4),
                    "status": "success",
                }
            )
            usage = getattr(response, "usage", None)
            self.call_log.append(
                {
                    "call_id": call_id,
                    "usage": dict(usage) if usage is not None else None,
                    "cost": (getattr(response, "_hidden_params", None) or {}).get("response_cost"),
                    "response": response,
                }
            )
            try:
                response._bandits_recorded = True
            except (AttributeError, TypeError, ValueError):  # pragma: no cover
                pass
            return response

        def forward(self, prompt=None, messages=None, **call_kwargs):
            state = self._begin(prompt, messages, call_kwargs)
            try:
                response = super().forward(prompt=prompt, messages=messages, **call_kwargs)
            except BaseException as exc:
                self._fail(state, prompt, messages, exc)
                raise
            return self._finish(state, prompt, messages, response)

        async def aforward(self, prompt=None, messages=None, **call_kwargs):
            state = self._begin(prompt, messages, call_kwargs)
            try:
                response = await super().aforward(prompt=prompt, messages=messages, **call_kwargs)
            except BaseException as exc:
                self._fail(state, prompt, messages, exc)
                raise
            return self._finish(state, prompt, messages, response)

    ref = resolve(model)
    return RecordedLM(ref.litellm_id, **credentials(ref, api_key=api_key), **kwargs)


_SETTING_ALIASES: dict[str, tuple[str, ...]] = {
    "max_tokens": ("max_tokens", "max_completion_tokens", "max_output_tokens"),
    "reasoning_effort": ("reasoning_effort", "reasoning", "thinking"),
}


def preflight_settings(model: str, settings: dict[str, Any]) -> dict[str, Any]:
    """What a request with these settings would actually send, without sending it.

    Requested keyword arguments are not evidence: LiteLLM can accept a setting,
    keep it through its own parameter mapping, and still leave it out of the
    request body (its Fireworks route drops ``chat_template_kwargs``). So the
    request is built by LiteLLM itself, against an in-process HTTP transport
    that records the body and returns a canned reply — no network, no cost.

    Returns ``requested``, ``effective`` (the sent body without messages),
    ``dropped`` (requested and absent from the body) and ``verified`` (False
    when this provider's route does not use the injectable HTTP client, so the
    body could not be observed). Raises :class:`ProviderError` when LiteLLM
    itself refuses a setting for this model.
    """
    requested = {key: value for key, value in settings.items() if value is not None}
    try:
        litellm = load_litellm()
    except ProviderError:
        # Without LiteLLM nothing can be sent at all; the body is unobservable,
        # and saying so is the only honest record.
        return {
            "requested": requested,
            "effective": {},
            "dropped": [],
            "verified": False,
            "note": "LiteLLM is not installed, so the request body could not be built",
        }
    import httpx
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    ref = resolve(model)
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        try:
            sent.append(json.loads(request.content or b"{}"))
        except ValueError:
            sent.append({})
        return httpx.Response(
            200,
            json={
                "id": "preflight",
                "object": "chat.completion",
                "created": 0,
                "model": ref.litellm_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        litellm.completion(
            model=ref.litellm_id,
            messages=[{"role": "user", "content": "preflight"}],
            api_key="preflight-no-network",
            api_base="http://preflight.invalid/v1"
            if ref.provider in _REQUIRED
            and any(arg == "api_base" for arg in _REQUIRED[ref.provider].values())
            else None,
            client=client,
            num_retries=0,
            **requested,
        )
    except Exception as exc:  # noqa: BLE001 - LiteLLM's own refusal is the finding
        # A failure after the body was sent (parsing the canned reply) does not
        # matter: the body is what this checks.
        if not sent and (
            "UnsupportedParams" in type(exc).__name__ or "does not support" in str(exc)
        ):
            raise ProviderError(f"{model}: {exc}".splitlines()[0]) from exc
        if not sent:
            return {
                "requested": requested,
                "effective": {},
                "dropped": [],
                "verified": False,
                "note": f"request body not observable offline ({type(exc).__name__})",
            }
    if not sent:
        return {
            "requested": requested,
            "effective": {},
            "dropped": [],
            "verified": False,
            "note": "this provider route does not use the injectable HTTP client",
        }
    body = {k: v for k, v in sent[-1].items() if k not in ("messages", "model")}
    dropped = [
        key
        for key in requested
        if not any(alias in body for alias in _SETTING_ALIASES.get(key, (key,)))
    ]
    return {"requested": requested, "effective": body, "dropped": dropped, "verified": True}
