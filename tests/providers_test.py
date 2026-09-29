"""A model string resolves to one provider and the credentials it needs."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("litellm")

from bandits import providers  # noqa: E402
from bandits.providers import ProviderError, credentials, resolve  # noqa: E402

_KEYS = (
    "FIREWORKS_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_API_BASE",
    "HOSTED_VLLM_API_BASE",
    "HOSTED_VLLM_API_KEY",
    "BANDITS_MODEL",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in _KEYS:
        monkeypatch.delenv(name, raising=False)
    # .env is read from the working directory; never the repository's own.
    monkeypatch.chdir(tmp_path)


def test_a_fireworks_account_path_keeps_what_the_user_typed() -> None:
    ref = resolve("accounts/fireworks/models/nemotron")

    assert ref.model == "accounts/fireworks/models/nemotron"
    assert ref.litellm_id == "fireworks_ai/accounts/fireworks/models/nemotron"
    assert ref.provider == "fireworks_ai"


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("anthropic/claude-sonnet-5", "anthropic"),
        ("openai/gpt-5", "openai"),
        ("hosted_vllm/my-judge", "hosted_vllm"),
        ("fireworks_ai/accounts/fireworks/models/x", "fireworks_ai"),
        ("openrouter/anthropic/claude-sonnet-5", "openrouter"),
    ],
)
def test_a_provider_prefix_passes_through(model: str, provider: str) -> None:
    ref = resolve(model)

    assert (ref.model, ref.litellm_id, ref.provider) == (model, model, provider)


@pytest.mark.parametrize("model", ["gpt-4o", "nosuchprovider/model"])
def test_a_name_without_a_known_provider_is_refused(model: str) -> None:
    with pytest.raises(ProviderError, match="<provider>/<model>"):
        resolve(model)


def test_the_fireworks_key_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-env")

    assert credentials(resolve("accounts/fireworks/models/x")) == {"api_key": "fw-env"}


def test_the_fireworks_key_falls_back_to_dotenv(tmp_path) -> None:
    (tmp_path / ".env").write_text("OTHER=1\nFIREWORKS_API_KEY='fw-file'\n")

    assert credentials(resolve("accounts/fireworks/models/x")) == {"api_key": "fw-file"}


def test_a_missing_fireworks_key_is_one_clear_error() -> None:
    # LiteLLM reports nothing missing for Fireworks; the override is what
    # turns a 401 on every call into this.
    with pytest.raises(ProviderError, match="FIREWORKS_API_KEY"):
        credentials(resolve("accounts/fireworks/models/x"))


def test_a_missing_provider_key_is_named() -> None:
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        credentials(resolve("anthropic/claude-sonnet-5"))


def test_an_exported_key_is_left_for_litellm(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

    assert credentials(resolve("anthropic/claude-sonnet-5")) == {}


def test_a_key_only_in_dotenv_is_passed_explicitly(tmp_path) -> None:
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-file\n")

    assert credentials(resolve("anthropic/claude-sonnet-5")) == {"api_key": "sk-ant-file"}


def test_a_base_url_only_in_dotenv_is_passed_as_api_base(tmp_path) -> None:
    (tmp_path / ".env").write_text("OLLAMA_API_BASE=http://gpu:11434\n")

    assert credentials(resolve("ollama/llama3")) == {"api_base": "http://gpu:11434"}


def test_a_vllm_server_without_a_base_is_one_clear_error() -> None:
    with pytest.raises(ProviderError, match="HOSTED_VLLM_API_BASE"):
        credentials(resolve("hosted_vllm/my-judge"))


def test_a_vllm_base_and_optional_key_come_from_dotenv(tmp_path) -> None:
    (tmp_path / ".env").write_text("HOSTED_VLLM_API_BASE=http://gpu:8000/v1\nHOSTED_VLLM_API_KEY=t\n")

    assert credentials(resolve("hosted_vllm/my-judge")) == {
        "api_base": "http://gpu:8000/v1",
        "api_key": "t",
    }


def test_bandits_model_moves_the_default(monkeypatch) -> None:
    monkeypatch.setenv("BANDITS_MODEL", "anthropic/claude-sonnet-5")
    try:
        assert importlib.reload(providers).DEFAULT_MODEL == "anthropic/claude-sonnet-5"
    finally:
        monkeypatch.delenv("BANDITS_MODEL")
        importlib.reload(providers)
    assert providers.DEFAULT_MODEL == providers.FIREWORKS_DEFAULT


def test_dspy_lm_calls_the_resolved_id() -> None:
    pytest.importorskip("dspy")

    lm = providers.dspy_lm("accounts/fireworks/models/x", api_key="k", temperature=0.0)

    assert lm.model == "fireworks_ai/accounts/fireworks/models/x"


def test_a_stage_keeps_its_own_fallback_unless_bandits_model_is_set(monkeypatch) -> None:
    assert providers.default_model("rlm/fallback") == "rlm/fallback"
    monkeypatch.setenv("BANDITS_MODEL", "anthropic/claude-sonnet-5")
    assert providers.default_model("rlm/fallback") == "anthropic/claude-sonnet-5"
