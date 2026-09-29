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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIREWORKS_DEFAULT = "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"

DEFAULT_MODEL = os.environ.get("BANDITS_MODEL") or FIREWORKS_DEFAULT
"""Every stage's default model. ``BANDITS_MODEL`` moves them all at once, so a
different provider does not mean passing ``--model`` to every command."""

_KEY_OVERRIDES: dict[str, str] = {"fireworks_ai": "FIREWORKS_API_KEY"}
"""Providers whose key LiteLLM does not report as missing. It says Fireworks
needs nothing even with no key set, which would surface as a 401 per call
instead of one clear error before the first."""


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


def credentials(ref: ModelRef) -> dict[str, str]:
    """Keyword arguments carrying whatever this provider needs and the
    environment does not already supply to LiteLLM.

    A variable already exported is left for LiteLLM to read itself. One that
    is only in ``.env`` is passed explicitly: an ``*_API_BASE`` as
    ``api_base``, a key or token as ``api_key``. Anything else (a cloud
    provider's several credentials, say) must be exported.
    """
    override = _KEY_OVERRIDES.get(ref.provider)
    if override is not None:
        value = env_value(override)
        if not value:
            raise ProviderError(f"{override} is not set and was not found in .env")
        return {"api_key": value}

    missing = load_litellm().validate_environment(ref.litellm_id).get("missing_keys") or []
    found: dict[str, str] = {}
    for name in missing:
        value = env_value(name)
        if not value:
            raise ProviderError(f"{name} is not set and was not found in .env ({ref.model})")
        if name.endswith(("_API_BASE", "_BASE_URL")):
            found["api_base"] = value
        elif "KEY" in name or "TOKEN" in name:
            found["api_key"] = value
        else:
            raise ProviderError(f"{name} was found in .env, but {ref.provider} needs it exported")
    return found


def dspy_lm(model: str, *, api_key: str | None = None, **kwargs: Any) -> Any:
    """A ``dspy.LM`` for ``model``. The caller has already imported DSPy."""
    import dspy

    ref = resolve(model)
    reach = {"api_key": api_key} if api_key else credentials(ref)
    return dspy.LM(ref.litellm_id, **reach, **kwargs)
