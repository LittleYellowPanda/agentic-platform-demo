"""Specification of the mock LLM. Run: uv run pytest tests/test_mock_llm.py -v

The helpers (find_city, find_expression, estimate_tokens) are given; the tests
named after a TODO check your code in `MockChatModel.complete`.
"""

import json

from app.services.agent_loop import run_agent
from app.services.llm import ChatModel, LLMResponse
from app.services.mock_llm import (
    MOCK_PREFIX,
    MockChatModel,
    find_city,
    find_expression,
)
from app.services.tools import Tool, ToolOutput
from app.services.tools.calculator import calculator_tool


def schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


ALL_TOOLS = [schema("weather"), schema("calculator"), schema("search_knowledge_base")]


def conversation(question: str) -> list[dict]:
    return [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": question}]


def ask(question: str, tools: list[dict] = ALL_TOOLS) -> LLMResponse:
    return MockChatModel().complete(conversation(question), tools)


def calls_of(response: LLMResponse) -> dict[str, dict]:
    return {call.name: json.loads(call.arguments) for call in response.tool_calls}


# --- Given helpers (should pass before you start) -----------------------------------------


def test_helper_find_city():
    assert find_city("What's the weather in Lyon tomorrow?") == "Lyon"
    assert find_city("Quel temps fait-il à Saint Malo ?") == "Saint Malo"
    assert find_city("Is it raining?") == "Paris"


def test_helper_find_expression():
    assert find_expression("What is 15% of 2340?") == "2340 * 15 / 100"
    assert find_expression("compute (1250 - 1000) / 1000 please") == "(1250 - 1000) / 1000"
    assert find_expression("How many funds in 2024?") is None


def test_it_is_a_chat_model():
    model: ChatModel = MockChatModel()  # the type checker checks the Protocol here
    assert model.model == "mock"


# --- TODO 1: results came back -> final answer ------------------------------------------------


def test_todo1_answers_from_the_tool_results_of_this_turn():
    messages = conversation("weather in Lyon and 2 + 2?") + [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "a", "content": "Weather in Lyon, France\nmore lines"},
        {"role": "tool", "tool_call_id": "b", "content": "2 + 2 = 4"},
    ]
    response = MockChatModel().complete(messages, ALL_TOOLS)

    assert response.tool_calls == []
    assert response.content.startswith(MOCK_PREFIX)
    # first line of each result, in the order the tools ran
    assert response.content.index("Weather in Lyon, France") < response.content.index("2 + 2 = 4")
    assert "more lines" not in response.content


def test_todo1_ignores_tool_results_of_earlier_turns():
    messages = conversation("first question") + [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "a", "content": "OLD RESULT"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "b", "content": "NEW RESULT"},
    ]
    content = MockChatModel().complete(messages, ALL_TOOLS).content
    assert "NEW RESULT" in content and "OLD RESULT" not in content


# --- TODO 2 + 3: deciding the tool calls --------------------------------------------------------


def test_todo3_weather_question_calls_weather_with_the_city():
    assert calls_of(ask("What's the weather in Lyon?", [schema("weather")])) == {
        "weather": {"city": "Lyon"}
    }


def test_todo3_calculation_calls_the_calculator():
    assert calls_of(ask("What is 15% of 2340?", [schema("calculator")])) == {
        "calculator": {"expression": "2340 * 15 / 100"}
    }


def test_todo3_linked_kb_is_always_searched_with_the_question():
    question = "What does the prospectus say about fees?"
    assert calls_of(ask(question, [schema("search_knowledge_base")])) == {
        "search_knowledge_base": {"query": question}
    }


def test_todo2_never_calls_a_tool_the_agent_does_not_have():
    # a weather question, but this agent only has the calculator
    response = ask("What's the weather in Lyon?", [schema("calculator")])
    assert response.tool_calls == []


def test_todo3_several_rules_give_parallel_calls_in_one_reply():
    response = ask("Weather in Lyon, and what is 15% of 2340?", ALL_TOOLS)
    assert [call.name for call in response.tool_calls] == [
        "weather",
        "calculator",
        "search_knowledge_base",
    ]


def test_todo3_arguments_are_json_text_like_a_real_model():
    call = ask("What is 2 * 21?", [schema("calculator")]).tool_calls[0]
    assert isinstance(call.arguments, str)
    assert json.loads(call.arguments) == {"expression": "2 * 21"}


def test_todo4_call_ids_are_unique_within_the_reply():
    ids = [call.id for call in ask("Weather in Lyon, 15% of 2340?", ALL_TOOLS).tool_calls]
    assert len(set(ids)) == len(ids) == 3


# --- TODO 4: answering directly, and the token estimates ---------------------------------------


def test_todo4_no_matching_rule_answers_directly_and_says_it_is_a_mock():
    response = ask("Tell me a joke", [schema("calculator")])
    assert response.tool_calls == []
    assert response.content.startswith(MOCK_PREFIX)
    assert "Tell me a joke" in response.content
    assert "LLM_API_KEY" in response.content  # tells the user how to get real answers


def test_todo4_tokens_are_estimated_not_zero():
    for response in (ask("Tell me a joke", []), ask("What is 2 * 21?", [schema("calculator")])):
        assert response.prompt_tokens > 0 and response.completion_tokens > 0


# --- The real test: your loop + the mock + a real tool ------------------------------------------


def test_full_run_with_your_loop_terminates_with_a_mock_answer():
    tools = {"calculator": calculator_tool}
    result = run_agent(MockChatModel(), "You are helpful.", "What is 15% of 2340?", tools)

    assert result.stop_reason == "answer"
    assert [step.kind for step in result.steps] == ["llm_call", "tool_call", "llm_call"]
    assert "351" in result.answer and result.answer.startswith(MOCK_PREFIX)


def test_full_run_with_a_failing_tool_still_answers():
    def broken(arguments: dict) -> ToolOutput:
        raise RuntimeError("Open-Meteo is down")

    weather = Tool(name="weather", description="", parameters={}, run=broken)
    result = run_agent(MockChatModel(), "sys", "Weather in Lyon?", {"weather": weather})

    assert result.stop_reason == "answer"
    assert "Open-Meteo is down" in result.answer  # the error came back as an observation
