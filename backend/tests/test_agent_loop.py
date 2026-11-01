"""Specification of the agent loop. Your implementation is done when all of these pass.

    uv run pytest tests/test_agent_loop.py -v

No network, no API key: the LLM is a scripted fake that replays prepared answers
and records every conversation it receives, so we can check what the loop sent.
"""

import copy
import json

import pytest

from app.services.agent_loop import AgentResult, execute_tool_call, run_agent
from app.services.llm import LLMResponse, ToolCall
from app.services.tools import Tool, ToolOutput

# --- Test helpers -------------------------------------------------------------------


class ScriptedLLM:
    """Replays prepared responses, one per call, and records what it received."""

    def __init__(self, *responses: LLMResponse):
        self.responses = list(responses)
        self.calls: list[dict] = []  # each: {"messages": [...], "tools": [...]}
        self.model = "scripted"

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        # deepcopy: snapshot the conversation AS IT WAS at this call
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self.responses:
            raise AssertionError("The loop called the LLM more times than expected.")
        return self.responses.pop(0)


def answer(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        content=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


def calls(*tool_calls: ToolCall, content: str | None = None) -> LLMResponse:
    return LLMResponse(
        content=content, tool_calls=list(tool_calls), prompt_tokens=20, completion_tokens=8
    )


def tool_call(call_id: str, name: str, **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


def make_tool(name: str, reply=lambda args: ToolOutput(content="ok")) -> Tool:
    return Tool(
        name=name,
        description=f"The {name} tool.",
        parameters={"type": "object", "properties": {}},
        run=reply,
    )


def add_tool() -> Tool:
    return make_tool("add", lambda args: ToolOutput(content=str(args["a"] + args["b"])))


SYSTEM = "You are a helpful assistant."


# --- 1. Direct answer -----------------------------------------------------------------


def test_direct_answer_stops_after_one_llm_call():
    llm = ScriptedLLM(answer("Paris."))
    result = run_agent(llm, SYSTEM, "Capital of France?", tools={})

    assert isinstance(result, AgentResult)
    assert result.answer == "Paris."
    assert result.stop_reason == "answer"
    assert len(llm.calls) == 1


def test_first_call_sends_system_prompt_then_user_message():
    llm = ScriptedLLM(answer("Paris."))
    run_agent(llm, SYSTEM, "Capital of France?", tools={})

    assert llm.calls[0]["messages"] == [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Capital of France?"},
    ]


def test_tool_schemas_are_sent_to_the_llm():
    llm = ScriptedLLM(answer("Hello."))
    run_agent(llm, SYSTEM, "Hi", tools={"add": add_tool()})

    assert llm.calls[0]["tools"] == [add_tool().schema()]


# --- 2. One tool call, then an answer ---------------------------------------------------


def test_tool_result_is_sent_back_and_the_model_answers():
    llm = ScriptedLLM(
        calls(tool_call("call_1", "add", a=2, b=40)),
        answer("2 + 40 = 42."),
    )
    result = run_agent(llm, SYSTEM, "What is 2 + 40?", tools={"add": add_tool()})

    assert result.answer == "2 + 40 = 42."
    assert len(llm.calls) == 2


def test_second_call_contains_the_assistant_tool_request_in_openai_format():
    llm = ScriptedLLM(
        calls(tool_call("call_1", "add", a=2, b=40), content="Let me compute that."),
        answer("42."),
    )
    run_agent(llm, SYSTEM, "What is 2 + 40?", tools={"add": add_tool()})

    assistant = llm.calls[1]["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Let me compute that."
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "add", "arguments": json.dumps({"a": 2, "b": 40})},
        }
    ]


def test_second_call_contains_the_tool_result_linked_by_its_id():
    llm = ScriptedLLM(calls(tool_call("call_1", "add", a=2, b=40)), answer("42."))
    run_agent(llm, SYSTEM, "What is 2 + 40?", tools={"add": add_tool()})

    tool_message = llm.calls[1]["messages"][3]
    assert tool_message == {"role": "tool", "tool_call_id": "call_1", "content": "42"}
    assert len(llm.calls[1]["messages"]) == 4  # system, user, assistant, tool


def test_several_tool_calls_in_one_turn_are_all_executed_in_order():
    llm = ScriptedLLM(
        calls(tool_call("call_a", "add", a=1, b=1), tool_call("call_b", "add", a=2, b=2)),
        answer("2 and 4."),
    )
    run_agent(llm, SYSTEM, "1+1 and 2+2?", tools={"add": add_tool()})

    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in second[3:]] == [
        ("call_a", "2"),
        ("call_b", "4"),
    ]


# --- 3. Failures become messages for the model, never crashes ----------------------------


def test_unknown_tool_is_reported_to_the_model_and_the_loop_continues():
    llm = ScriptedLLM(calls(tool_call("call_1", "weather", city="Paris")), answer("Sorry."))
    result = run_agent(llm, SYSTEM, "Weather?", tools={"add": add_tool()})

    tool_message = llm.calls[1]["messages"][3]
    assert tool_message["role"] == "tool"
    assert "unknown tool" in tool_message["content"].lower()
    assert "add" in tool_message["content"]  # tells the model what does exist
    assert result.answer == "Sorry."


def test_invalid_json_arguments_are_reported_to_the_model():
    broken = ToolCall(id="call_1", name="add", arguments='{"a": 2, "b": ')  # cut off
    llm = ScriptedLLM(calls(broken), answer("Retrying failed."))
    run_agent(llm, SYSTEM, "2+2?", tools={"add": add_tool()})

    assert llm.calls[1]["messages"][3]["content"].startswith("Error")


def test_arguments_that_are_not_an_object_are_reported_to_the_model():
    not_an_object = ToolCall(id="call_1", name="add", arguments="[1, 2]")
    llm = ScriptedLLM(calls(not_an_object), answer("Hm."))
    run_agent(llm, SYSTEM, "1+2?", tools={"add": add_tool()})

    assert llm.calls[1]["messages"][3]["content"].startswith("Error")


def test_a_tool_that_raises_is_reported_to_the_model():
    def explode(args):
        raise RuntimeError("weather service down")

    llm = ScriptedLLM(calls(tool_call("call_1", "weather")), answer("Unavailable."))
    result = run_agent(llm, SYSTEM, "Weather?", tools={"weather": make_tool("weather", explode)})

    content = llm.calls[1]["messages"][3]["content"]
    assert content.startswith("Error")
    assert "weather service down" in content  # the model can explain what happened
    assert result.answer == "Unavailable."


# --- 4. Stopping ---------------------------------------------------------------------------


def test_stops_after_max_iterations_without_answer():
    # A model that keeps calling tools forever
    llm = ScriptedLLM(*[calls(tool_call(f"call_{i}", "add", a=i, b=i)) for i in range(10)])
    result = run_agent(llm, SYSTEM, "Loop!", tools={"add": add_tool()}, max_iterations=3)

    assert len(llm.calls) == 3
    assert result.stop_reason == "max_iterations"
    assert result.answer is None


# --- 5. Trace: what the UI and the database will show ---------------------------------------


def test_steps_record_llm_calls_and_tool_calls_in_order():
    llm = ScriptedLLM(
        calls(tool_call("call_1", "add", a=2, b=3), content="Computing."), answer("5.")
    )
    result = run_agent(llm, SYSTEM, "2+3?", tools={"add": add_tool()})

    assert [step.kind for step in result.steps] == ["llm_call", "tool_call", "llm_call"]
    first, tool, last = result.steps
    assert first.content == "Computing."
    assert (tool.tool_name, tool.arguments, tool.output, tool.is_error) == (
        "add",
        {"a": 2, "b": 3},
        "5",
        False,
    )
    assert last.content == "5."
    assert all(step.duration_ms >= 0 for step in result.steps)


def test_failed_tool_calls_are_marked_as_errors_in_the_steps():
    llm = ScriptedLLM(calls(tool_call("call_1", "nope")), answer("Ok."))
    result = run_agent(llm, SYSTEM, "?", tools={})

    assert result.steps[1].is_error is True


def test_tokens_are_summed_over_all_llm_calls():
    llm = ScriptedLLM(calls(tool_call("call_1", "add", a=1, b=1)), answer("2.", 30, 7))
    result = run_agent(llm, SYSTEM, "1+1?", tools={"add": add_tool()})

    assert result.prompt_tokens == 20 + 30
    assert result.completion_tokens == 8 + 7


def test_sources_returned_by_tools_are_collected():
    chunk = {"filename": "prospectus.pdf", "chunk_index": 3, "score": 0.87}
    search = make_tool(
        "search_knowledge_base", lambda args: ToolOutput(content="...", sources=[chunk])
    )
    llm = ScriptedLLM(
        calls(tool_call("call_1", "search_knowledge_base", query="limits")), answer("10%.")
    )
    result = run_agent(llm, SYSTEM, "Exposure limit?", tools={"search_knowledge_base": search})

    assert result.sources == [chunk]


def test_on_step_is_called_for_each_step_as_it_happens():
    seen = []
    llm = ScriptedLLM(calls(tool_call("call_1", "add", a=1, b=1)), answer("2."))
    result = run_agent(llm, SYSTEM, "1+1?", tools={"add": add_tool()}, on_step=seen.append)

    assert len(seen) == 3  # llm_call, tool_call, llm_call
    assert seen == result.steps  # tomorrow, this callback streams steps to the browser


# --- 6. execute_tool_call on its own ----------------------------------------------------------


def test_execute_tool_call_returns_a_tool_step():
    step = execute_tool_call(tool_call("call_1", "add", a=4, b=5), {"add": add_tool()})
    assert (step.kind, step.tool_name, step.output, step.is_error) == (
        "tool_call",
        "add",
        "9",
        False,
    )


@pytest.mark.parametrize("arguments", ["", "{}"])
def test_empty_arguments_mean_no_arguments(arguments):
    seen = {}
    tool = make_tool("ping", lambda args: (seen.update(args=args), ToolOutput(content="pong"))[1])
    step = execute_tool_call(ToolCall(id="c", name="ping", arguments=arguments), {"ping": tool})
    assert step.output == "pong" and seen["args"] == {}
