"""Web search tool, tool availability, mock mode and LLM error messages. No network."""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import llm
from app.services.mock_llm import MockChatModel
from app.services.tools import catalog, web_search
from app.services.tools.registry import build_tools
from app.services.tools.web_search import WebSearchError, format_results, search_web

# --- Web search -----------------------------------------------------------------------

PAYLOAD = {
    "results": [
        {
            "title": "ECB keeps rates unchanged",
            "url": "https://example.org/ecb",
            "content": "The ECB   left its key\nrates unchanged on Thursday.",
            "score": 0.91,
            "published_date": "2026-10-29",
        },
        {"title": "Second", "url": "https://example.org/2", "content": "Other text", "score": 0.5},
    ]
}


def test_web_results_are_numbered_framed_as_untrusted_and_cited():
    output = format_results("ecb rates", PAYLOAD)

    assert output.content.startswith("Web results (untrusted content")
    assert "[1] ECB keeps rates unchanged (https://example.org/ecb, published 2026-10-29)" in (
        output.content
    )
    assert "The ECB left its key" in output.content  # whitespace collapsed
    assert [s["url"] for s in output.sources] == ["https://example.org/ecb", "https://example.org/2"]
    assert output.sources[0]["type"] == "web"


def test_no_web_results_is_said_plainly():
    assert format_results("zzz", {"results": []}).content == "No web results for: zzz"


class FakeClient:
    def __init__(self, status: int, payload: dict | None = None):
        self.status, self.payload, self.request = status, payload or {}, None

    def post(self, url, headers, json):
        self.request = {"url": url, "headers": headers, "json": json}
        return SimpleNamespace(
            status_code=self.status,
            json=lambda: self.payload,
            raise_for_status=lambda: None,
        )


def test_search_sends_the_key_in_a_header_never_in_the_url():
    client = FakeClient(200, PAYLOAD)
    search_web("ecb rates", "tvly-secret", client=client)

    assert client.request["headers"] == {"Authorization": "Bearer tvly-secret"}
    assert "tvly-secret" not in client.request["url"]
    assert client.request["json"]["max_results"] == web_search.MAX_RESULTS


@pytest.mark.parametrize(
    ("status", "message"), [(401, "rejected the API key"), (429, "quota"), (432, "quota")]
)
def test_search_provider_errors_are_explained(status, message):
    with pytest.raises(WebSearchError, match=message):
        search_web("q", "key", client=FakeClient(status))


def test_web_search_requires_a_query():
    with pytest.raises(ValueError, match="query"):
        web_search.web_search_tool.run({"query": " "})


# --- Availability ---------------------------------------------------------------------


@pytest.fixture
def no_tavily_key(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "")


@pytest.fixture
def tavily_key(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test")


def test_without_key_web_search_is_listed_as_unavailable_with_a_reason(no_tavily_key):
    info = {i.name: i for i in catalog.list_tools()}["web_search"]
    assert info.available is False and "TAVILY_API_KEY" in info.unavailable_reason


def test_without_key_web_search_is_never_offered_to_the_model(no_tavily_key):
    agent = SimpleNamespace(tools=["calculator", "web_search"], knowledge_bases=[])
    assert list(build_tools(agent, db=None)) == ["calculator"]


def test_with_key_web_search_is_available_and_offered(tavily_key):
    info = {i.name: i for i in catalog.list_tools()}["web_search"]
    assert info.available is True and info.unavailable_reason is None
    agent = SimpleNamespace(tools=["web_search"], knowledge_bases=[])
    assert list(build_tools(agent, db=None)) == ["web_search"]


# --- Mock mode ------------------------------------------------------------------------


def test_no_api_key_means_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "")
    assert llm.llm_mode() == "mock"
    assert isinstance(llm.create_chat_model(), MockChatModel)


def test_api_key_means_the_real_client_with_the_agent_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "sk-test")
    model = llm.create_chat_model("qwen/qwen3-8b")
    assert llm.llm_mode() == "api"
    assert isinstance(model, llm.OpenAIChatModel) and model.model == "qwen/qwen3-8b"


# --- LLM error messages ---------------------------------------------------------------


class ProviderError(Exception):
    def __init__(self, status_code: int, text: str = ""):
        super().__init__(text)
        self.status_code = status_code


@pytest.mark.parametrize(
    "error",
    [
        ProviderError(404, "No endpoints found that support tool use."),  # OpenRouter
        ProviderError(400, "registry.ollama.ai/library/gemma2 does not support tools"),  # Ollama
    ],
)
def test_model_without_tool_calling_gets_a_specific_message(error):
    assert "doesn't support tool calling" in llm.describe_llm_error(error)


def test_plain_404_is_still_an_unknown_model():
    assert "doesn't know this model" in llm.describe_llm_error(ProviderError(404, "not found"))


def test_429_says_retries_already_happened():
    assert "after automatic retries" in llm.describe_llm_error(ProviderError(429))
