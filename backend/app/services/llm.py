"""The LLM, seen from the agent loop: send messages and tool schemas, get a reply.

The loop depends only on the small `ChatModel` interface below, not on a provider
SDK. In production it's `OpenAIChatModel` (OpenRouter, Ollama, vLLM or a company
gateway: they all speak the OpenAI format); in tests, a scripted fake.
"""

from dataclasses import dataclass, field
from typing import Protocol

from openai import OpenAI


@dataclass(frozen=True)
class ToolCall:
    """The model asking us to run a tool. It never runs anything itself."""

    id: str  # links the result we send back to this exact request
    name: str
    arguments: str  # raw JSON text written by the model: may be invalid!


@dataclass(frozen=True)
class LLMResponse:
    content: str | None  # text: the final answer, or reasoning shown next to tool calls
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatModel(Protocol):
    """Anything with this attribute and method can drive the agent loop."""

    model: str  # the model name, saved on each run

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAIChatModel:
    """Talks to any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        provider: str | None = None,
    ):
        self.model = model
        self.provider = provider  # saved on each run, next to the model
        # The SDK already retries 429, 5xx and connection errors, with exponential
        # backoff and the provider's Retry-After header. We only choose how many times.
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:  # some providers reject an empty tools list
            kwargs["tools"] = tools
        response = self._client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        usage = response.usage
        return LLMResponse(
            content=message.content,
            tool_calls=[
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in (message.tool_calls or [])
            ],
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


def llm_mode() -> str:
    """ "api" when a key is configured, "mock" otherwise (the app must start without one)."""
    from app.core.config import (
        settings,  # local import: keeps this module test-friendly
    )

    return "api" if settings.llm_api_key else "mock"


class ProviderNotConfigured(ValueError):
    """An agent chose a provider this platform no longer has (removed from .env)."""


def create_chat_model(model: str | None = None, provider_id: str | None = None) -> ChatModel:
    """The LLM an agent chose: a provider (None = the default one) and a model name
    (None = LLM_MODEL).

    Default provider without LLM_API_KEY: the mock. The platform stays usable for a
    demo or a review, and every answer says it's a mock, so nobody mistakes it.
    """
    from app.core.config import settings
    from app.services.providers import get_provider

    if provider_id is None and llm_mode() == "mock":
        from app.services.mock_llm import MockChatModel

        return MockChatModel()
    provider = get_provider(provider_id)
    if provider is None:
        raise ProviderNotConfigured(
            f"The provider '{provider_id}' isn't configured on this platform. "
            "Pick another model in the agent's settings."
        )
    if provider_id is not None and not model:
        raise ProviderNotConfigured(f"No model chosen for the provider '{provider_id}'.")
    return OpenAIChatModel(
        base_url=provider.base_url,
        api_key=provider.api_key,
        model=model or settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        provider=provider.id,
    )


def describe_llm_error(error: Exception) -> str:
    """A message for the user, from an exception raised while calling the LLM.

    Relies on the HTTP status rather than on exception classes, so it keeps working
    across SDK versions and OpenAI-compatible providers.
    """
    status = getattr(error, "status_code", None)
    text = str(error).lower()
    # OpenRouter: 404 "No endpoints found that support tool use"; Ollama: 400 "does not support tools"
    if status in (400, 404) and ("tool use" in text or "support tools" in text):
        return (
            "This model doesn't support tool calling, which agents need. Pick another model "
            "(on OpenRouter, filter models by 'tools'), or remove the agent's tools."
        )
    if status == 401:
        return "The LLM provider rejected the API key (401). Check the key in .env."
    if status == 402:
        return "The LLM provider requires credits for this model (402)."
    if status == 404:
        return "The LLM provider doesn't know this model (404). Check the model name."
    if status == 429:
        return (
            "Rate limit reached at the LLM provider (429), even after automatic retries: "
            "free models are limited. Retry in a minute, or use another model."
        )
    if status == 400:
        return (
            "The LLM provider refused the request (400). The model may not support tool "
            f"calling. Details: {error}"
        )
    if status is not None and status >= 500:
        return f"The LLM provider is having trouble ({status}). Retry later."
    name = type(error).__name__
    if "Timeout" in name:
        return "The LLM provider didn't answer in time (timeout)."
    if "Connection" in name:
        return "Can't reach the LLM provider (connection error). Check LLM_BASE_URL and the network."
    return f"Unexpected error while calling the LLM: {error}"
