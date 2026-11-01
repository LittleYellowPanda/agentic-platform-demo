"""YOUR TURN: fill in TODO 1-4 in `complete`. Spec: uv run pytest tests/test_mock_llm.py -v

A fake LLM that follows simple rules, so the platform runs without any API key.

It speaks the same `ChatModel` interface as the real client: the agent loop can't
tell the difference. Deterministic, so it also makes demos and tests reproducible.

The rules, like a (very) small model would apply them:
- the last message is a tool result  -> write the final answer from those results
- the question mentions the weather  -> call `weather` with the city
- the question contains a calculation -> call `calculator` with the expression
- a knowledge base is linked          -> call `search_knowledge_base` with the question
- nothing matches                     -> answer directly, saying it's a mock
Several matching rules -> several tool calls in the same reply (parallel calls).
"""

import json
import re

from app.services.llm import LLMResponse, ToolCall

MOCK_PREFIX = "[mock] "
WEATHER_WORDS = ("weather", "météo", "meteo", "temperature", "température", "forecast")
KB_TOOL = "search_knowledge_base"

_PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:of|de)\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
_EXPRESSION = re.compile(r"[\d(][\d\s.()]*[-+*/^][\d\s.()+\-*/^]*\d\)?")
_CITY = re.compile(r"\b(?:in|à|a|at)\s+([A-Z][\w'-]+(?:\s[A-Z][\w'-]+)?)")


def find_city(text: str) -> str:
    """The capitalized place after "in" / "à", or Paris by default."""
    match = _CITY.search(text)
    return match.group(1) if match else "Paris"


def find_expression(text: str) -> str | None:
    """An arithmetic expression for the calculator, or None. "15% of 2340" -> "2340 * 15 / 100"."""
    percent = _PERCENT.search(text)
    if percent:
        rate, base = (n.replace(",", ".") for n in percent.groups())
        return f"{base} * {rate} / 100"
    expression = _EXPRESSION.search(text)
    return expression.group(0).strip() if expression else None


def estimate_tokens(text: str) -> int:
    """Rough count (about 4 characters per token), so the metrics aren't all zeros."""
    return max(1, len(text) // 4)


class MockChatModel:
    model = "mock"

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Same signature as the real client: read the conversation, reply.

        Reminder of what `messages` looks like (OpenAI format):
            {"role": "system", "content": "..."}
            {"role": "user", "content": "What is 15% of 2340?"}
            {"role": "assistant", "content": None, "tool_calls": [...]}
            {"role": "tool", "tool_call_id": "...", "content": "2340 * 15 / 100 = 351"}
        and each element of `tools` looks like:
            {"type": "function", "function": {"name": "calculator", ...}}
        """
        prompt_tokens = estimate_tokens(" ".join(str(m.get("content") or "") for m in messages))

        # TODO 1: if the LAST message is a tool result, the tools have answered:
        #   write the final answer. Collect the first line of each tool message of
        #   THIS turn only: walk the messages backwards, stop at the first message
        #   that isn't a tool result. Keep them in the order the tools ran.
        #   The answer starts with MOCK_PREFIX, then one "- <line>" per result.
        #   Return an LLMResponse with that content, no tool calls, and the token
        #   estimates (completion_tokens = estimate_tokens(answer)).
        if messages and messages[-1]["role"] == "tool":
            tool_result = []
            for message in reversed(messages):
                if message["role"] != "tool":
                    break
                content = message.get("content")
                if content:
                    first_line = content.splitlines()[0]
                    tool_result.append(f"- {first_line}")
            answer = MOCK_PREFIX + "\n".join(reversed(tool_result))
            return LLMResponse(
                content=answer,
                tool_calls=[],
                prompt_tokens=prompt_tokens,
                completion_tokens=estimate_tokens(answer),
            )

        # TODO 2: the names of the tools this agent has (a set), from `tools`.
        #   The question is the content of the last message.

        tool_names = {tool["function"]["name"] for tool in tools}
        question = messages[-1]["content"] if messages else ""  

        # TODO 3: decide which tools to call, as a list of (name, arguments dict),
        #   in this order, each only if the tool is available:
        #   - "weather" if the question contains one of WEATHER_WORDS (case-insensitive)
        #     -> {"city": find_city(question)}
        #   - "calculator" if find_expression(question) finds something
        #     -> {"expression": ...}
        #   - KB_TOOL always (when available) -> {"query": question}
        tool_calls = []
        if "weather" in tool_names and any(word in question.lower() for word in WEATHER_WORDS):
            tool_calls.append(("weather", {"city": find_city(question)}))
        if "calculator" in tool_names:
            expression = find_expression(question)
            if expression:
                tool_calls.append(("calculator", {"expression": expression}))
        if KB_TOOL in tool_names:
            tool_calls.append((KB_TOOL, {"query": question}))   

        # TODO 4: if there are calls, return them all in ONE LLMResponse:
        #   - ToolCall(id=..., name=..., arguments=...): arguments is JSON TEXT,
        #     like a real model writes it (json.dumps), not a dict.
        #   - ids must be unique; f"mock_{len(messages)}_{i}" stays unique across turns.
        #   - content=None, completion_tokens = sum of estimate_tokens(arguments).
        #   Otherwise answer directly: MOCK_PREFIX + a sentence saying no LLM is
        #   configured, repeating the question, and mentioning LLM_API_KEY.
        if tool_calls:
            calls = [
                ToolCall(
                    id=f"mock_{len(messages)}_{i}",
                    name=name,
                    arguments=json.dumps(args),
                )
                for i, (name, args) in enumerate(tool_calls)
            ]
            completion_tokens = sum(estimate_tokens(call.arguments) for call in calls)
            return LLMResponse(
                content=None,
                tool_calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        else:
            answer = (
                MOCK_PREFIX
                + f"No LLM is configured. I can't answer '{question}'. "
                + "Set the LLM_API_KEY environment variable to use a real model."
            )
            return LLMResponse(
                content=answer,
                tool_calls=[],
                prompt_tokens=prompt_tokens,
                completion_tokens=estimate_tokens(answer),
            )



