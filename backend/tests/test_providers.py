"""Providers: which endpoints exist, which models they offer, and how an agent's choice
becomes a client. No network: the HTTP calls are replaced by a fake `fetch`."""

from types import SimpleNamespace

import pytest

from app.services import providers
from app.services.providers import (
    all_models,
    configured_providers,
    get_provider,
    kind_of,
    list_models,
    list_ollama,
    list_openrouter,
)


def make_settings(**overrides):
    values = dict(
        llm_base_url="https://openrouter.ai/api/v1",
        llm_api_key="",
        llm_model="openrouter/free",
        ollama_base_url="",
        openrouter_api_key="",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def empty_cache():
    providers._cache.clear()
    yield
    providers._cache.clear()


# --- Which providers exist ----------------------------------------------------------------


def test_kind_is_inferred_from_the_url():
    assert kind_of("https://openrouter.ai/api/v1") == "openrouter"
    assert kind_of("http://host.docker.internal:11434/v1") == "ollama"
    assert kind_of("https://gateway.company.internal/v1") == "openai"


def test_mock_mode_without_extra_providers_has_none():
    assert configured_providers(make_settings()) == []
    assert get_provider(None, make_settings()) is None


def test_extra_providers_work_even_in_mock_mode():
    s = make_settings(ollama_base_url="http://jetson:11434/v1", openrouter_api_key="k")
    assert [p.id for p in configured_providers(s)] == ["ollama", "openrouter"]
    assert get_provider(None, s) is None  # no default: agents without provider use the mock
    assert get_provider("ollama", s).base_url == "http://jetson:11434/v1"


def test_the_default_provider_comes_first_and_is_not_listed_twice():
    s = make_settings(
        llm_api_key="or-key",  # default = OpenRouter
        openrouter_api_key="other-key",
        ollama_base_url="http://jetson:11434/v1",
    )
    ids = [p.id for p in configured_providers(s)]
    assert ids == ["openrouter", "ollama"]
    assert get_provider(None, s).id == "openrouter"
    assert get_provider("openrouter", s).api_key == "or-key"  # the default's settings win


def test_a_company_gateway_is_the_default_provider():
    s = make_settings(llm_base_url="https://gw.internal/v1", llm_api_key="k")
    assert get_provider(None, s).id == "default"
    assert get_provider("ollama", s) is None  # not configured


# --- Listing models -------------------------------------------------------------------------

OLLAMA = providers.Provider("ollama", "ollama", "Ollama", "http://jetson:11434/v1", "ollama")
OPENROUTER = providers.Provider(
    "openrouter", "openrouter", "OpenRouter", providers.OPENROUTER_URL, "k"
)


def fake_ollama(method, url, payload, key):
    if url == "http://jetson:11434/api/tags":  # /v1 removed: Ollama's own API
        return {"models": [{"name": "qwen2.5:7b"}, {"name": "deepseek-r1:7b"}]}
    if url == "http://jetson:11434/api/show":
        caps = {"qwen2.5:7b": ["completion", "tools"], "deepseek-r1:7b": ["completion"]}
        return {"capabilities": caps[payload["model"]]}
    raise AssertionError(url)


def test_ollama_models_say_whether_they_support_tools():
    models = {m.id: m.tools for m in list_ollama(OLLAMA, fake_ollama)}
    assert models == {"deepseek-r1:7b": False, "qwen2.5:7b": True}


def test_an_old_ollama_without_capabilities_gives_unknown_not_an_error():
    def fetch(method, url, payload, key):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "llama3.1:8b"}]}
        raise RuntimeError("404")

    assert list_ollama(OLLAMA, fetch)[0].tools is None


def test_openrouter_keeps_only_free_models_with_tools():
    data = {
        "data": [
            {"id": "z/free-tools:free", "name": "Z free", "pricing": {"prompt": "0", "completion": "0"},
             "supported_parameters": ["tools", "temperature"]},
            {"id": "a/free-no-tools:free", "pricing": {"prompt": "0", "completion": "0"},
             "supported_parameters": ["temperature"]},
            {"id": "b/paid-tools", "pricing": {"prompt": "0.000001", "completion": "0.000002"},
             "supported_parameters": ["tools"]},
            {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"},
             "supported_parameters": ["tools"]},
        ]
    }
    models = list_openrouter(OPENROUTER, lambda *args: data)
    assert [m.id for m in models] == ["openrouter/free", "z/free-tools:free"]
    assert all(m.tools and m.free for m in models)


def test_model_lists_are_cached():
    calls = []

    def fetch(*args):
        calls.append(args)
        return fake_ollama(*args)

    list_models(OLLAMA, fetch)
    list_models(OLLAMA, fetch)
    assert len(calls) == 3  # tags + 2 x show, once only


def test_one_provider_down_does_not_hide_the_others(monkeypatch):
    s = make_settings(llm_api_key="k", ollama_base_url="http://jetson:11434/v1")

    def fetch(method, url, payload, key):
        if "openrouter" in url:
            raise ConnectionError("down")
        return fake_ollama(method, url, payload, key)

    entries = {e.provider.id: e for e in all_models(s, fetch)}
    assert entries["openrouter"].error and entries["openrouter"].is_default
    assert entries["ollama"].error is None and len(entries["ollama"].models) == 2


# --- From an agent's choice to a client -----------------------------------------------------


def test_create_chat_model_uses_the_chosen_provider(monkeypatch):
    from app.core import config
    from app.services import llm as llm_module
    from app.services.llm import create_chat_model

    clients = []
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: clients.append(kwargs))
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    monkeypatch.setattr(config.settings, "ollama_base_url", "http://jetson:11434/v1")

    assert create_chat_model().model == "mock"  # default provider in mock mode
    llm = create_chat_model("qwen2.5:7b", "ollama")
    assert (llm.model, llm.provider) == ("qwen2.5:7b", "ollama")
    assert clients[-1]["base_url"] == "http://jetson:11434/v1"
    assert clients[-1]["api_key"] == "ollama"  # Ollama ignores it, the SDK requires one


def test_a_removed_provider_fails_clearly(monkeypatch):
    from app.core import config
    from app.services.llm import ProviderNotConfigured, create_chat_model

    monkeypatch.setattr(config.settings, "openrouter_api_key", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    with pytest.raises(ProviderNotConfigured, match="openrouter"):
        create_chat_model("x", "openrouter")


def test_agent_with_a_provider_needs_a_model():
    from pydantic import ValidationError

    from app.schemas.agents import AgentWrite

    with pytest.raises(ValidationError, match="Choose a model"):
        AgentWrite(name="a", provider="ollama")
    assert AgentWrite(name="a", provider=" ", model="").provider is None
    assert AgentWrite(name="a", provider="ollama", model="qwen2.5:7b").provider == "ollama"
