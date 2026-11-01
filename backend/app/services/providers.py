"""LLM providers: which endpoints the platform can use, and which models each offers.

Every provider speaks the OpenAI format, so one client serves them all; what differs
is where they are, whether they need a key, and how to list their models.

Configured from the environment:
- the default provider: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (as before);
- Ollama, optional: OLLAMA_BASE_URL (e.g. http://host.docker.internal:11434/v1);
- OpenRouter, optional: OPENROUTER_API_KEY.
A provider is listed once, even if the default already points at it.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

Kind = Literal["ollama", "openrouter", "openai"]
OPENROUTER_URL = "https://openrouter.ai/api/v1"
CACHE_SECONDS = 300  # model lists change rarely; don't query providers on every page load
TIMEOUT_SECONDS = 5.0

LABELS = {"ollama": "Ollama (local)", "openrouter": "OpenRouter", "openai": "OpenAI-compatible"}


@dataclass(frozen=True)
class Provider:
    id: str  # stored on agents and runs: "ollama", "openrouter", "default"
    kind: Kind
    label: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class ModelInfo:
    id: str  # the name to send to the provider, e.g. "qwen2.5:7b"
    label: str
    tools: bool | None  # supports tool calling? None = the provider doesn't say
    free: bool | None = None


@dataclass
class ProviderModels:
    provider: Provider
    is_default: bool
    models: list[ModelInfo] = field(default_factory=list)
    error: str | None = None  # the provider couldn't be listed (down, bad key...)


def kind_of(base_url: str) -> Kind:
    if "openrouter.ai" in base_url:
        return "openrouter"
    if ":11434" in base_url or "ollama" in base_url:
        return "ollama"
    return "openai"


def _settings(settings: Any | None) -> Any:
    if settings is None:  # local import: keeps this module test-friendly
        from app.core.config import settings as app_settings

        return app_settings
    return settings


def configured_providers(settings: Any | None = None) -> list[Provider]:
    """The providers this platform can call, the default one first.

    Without LLM_API_KEY there is no default provider (mock mode), but Ollama or
    OpenRouter configured separately are still usable by agents that choose them.
    """
    settings = _settings(settings)
    providers: list[Provider] = []
    if settings.llm_api_key:
        kind = kind_of(settings.llm_base_url)
        providers.append(
            Provider(
                id=kind if kind != "openai" else "default",
                kind=kind,
                label=LABELS[kind],
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )
        )
    known = {p.id for p in providers}
    if settings.ollama_base_url and "ollama" not in known:
        providers.append(
            Provider("ollama", "ollama", LABELS["ollama"], settings.ollama_base_url, "ollama")
        )
    if settings.openrouter_api_key and "openrouter" not in known:
        providers.append(
            Provider(
                "openrouter", "openrouter", LABELS["openrouter"], OPENROUTER_URL,
                settings.openrouter_api_key,
            )
        )
    return providers


def default_provider(settings: Any | None = None) -> Provider | None:
    """The provider of LLM_* variables; None in mock mode (no LLM_API_KEY)."""
    settings = _settings(settings)
    providers = configured_providers(settings)
    return providers[0] if providers and settings.llm_api_key else None


def get_provider(provider_id: str | None, settings: Any | None = None) -> Provider | None:
    """The provider with this id, or the default one for None. None if not configured."""
    if provider_id is None:
        return default_provider(settings)
    return next((p for p in configured_providers(settings) if p.id == provider_id), None)


# --- Listing models ---------------------------------------------------------------

Fetch = Callable[[str, str, dict | None, str | None], Any]  # (method, url, json, key) -> JSON


def http_fetch(method: str, url: str, payload: dict | None, api_key: str | None) -> Any:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.request(method, url, json=payload, headers=headers, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def list_ollama(provider: Provider, fetch: Fetch) -> list[ModelInfo]:
    """Installed models, and whether each supports tools (Ollama's own API knows)."""
    root = provider.base_url.rstrip("/").removesuffix("/v1")
    names = [m["name"] for m in fetch("GET", f"{root}/api/tags", None, None).get("models", [])]
    models = []
    for name in sorted(names):
        try:
            details = fetch("POST", f"{root}/api/show", {"model": name}, None)
            tools: bool | None = "tools" in details.get("capabilities", [])
        except Exception:  # an old Ollama without capabilities: say we don't know
            tools = None
        models.append(ModelInfo(id=name, label=name, tools=tools))
    return models


def list_openrouter(provider: Provider, fetch: Fetch) -> list[ModelInfo]:
    """Free models that support tools: the only ones usable at no cost for agents.
    (Paid models are left out of the list; they can still be typed by an API call.)"""
    data = fetch("GET", f"{provider.base_url.rstrip('/')}/models", None, None).get("data", [])
    models = [ModelInfo("openrouter/free", "Free router (picks a free model with tools)", True, True)]
    for item in data:
        pricing = item.get("pricing", {})
        free = str(pricing.get("prompt", "1")) in ("0", "0.0") and str(
            pricing.get("completion", "1")
        ) in ("0", "0.0")
        if free and "tools" in item.get("supported_parameters", []) and item["id"] != "openrouter/free":
            models.append(ModelInfo(item["id"], item.get("name") or item["id"], True, True))
    return models[:1] + sorted(models[1:], key=lambda m: m.label.lower())


def list_generic(provider: Provider, fetch: Fetch) -> list[ModelInfo]:
    data = fetch("GET", f"{provider.base_url.rstrip('/')}/models", None, provider.api_key)
    return [ModelInfo(m["id"], m["id"], None) for m in data.get("data", [])]


_cache: dict[str, tuple[float, list[ModelInfo]]] = {}


def list_models(provider: Provider, fetch: Fetch = http_fetch, use_cache: bool = True) -> list[ModelInfo]:
    cached = _cache.get(provider.id)
    if use_cache and cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    lister = {"ollama": list_ollama, "openrouter": list_openrouter}.get(provider.kind, list_generic)
    models = lister(provider, fetch)
    _cache[provider.id] = (time.monotonic(), models)
    return models


def all_models(settings: Any | None = None, fetch: Fetch = http_fetch) -> list[ProviderModels]:
    """Every configured provider with its models. A provider that can't be reached is
    reported with its error, and doesn't prevent listing the others."""
    providers = configured_providers(settings)
    default = default_provider(settings)
    default_id = default.id if default else None
    result = []
    for provider in providers:
        entry = ProviderModels(provider, is_default=provider.id == default_id)
        try:
            entry.models = list_models(provider, fetch)
        except Exception as error:
            logger.warning("Can't list models of %s: %s", provider.id, error)
            entry.error = f"Can't reach {provider.label}: {type(error).__name__}"
        result.append(entry)
    return result
