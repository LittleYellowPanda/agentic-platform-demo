"""The agent loop: let the LLM call tools until it can answer.

    messages = [system prompt, user question]
    repeat at most max_iterations times:
        ask the LLM (messages + tool schemas)
        no tool call?  -> that's the final answer: stop
        otherwise      -> run each requested tool, append its result, loop again

"""
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from app.services.llm import ChatModel, LLMResponse, ToolCall
from app.services.tools import Tool


@dataclass
class Step:
    """One visible step of a run: an LLM call, or a tool call."""

    kind: Literal["llm_call", "tool_call"]
    duration_ms: int
    # llm_call
    content: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # tool_call
    tool_name: str | None = None
    arguments: dict | None = None
    output: str | None = None
    is_error: bool = False
    sources: list[dict] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str | None  # None if the loop stopped before the model answered
    stop_reason: Literal["answer", "max_iterations"]
    steps: list[Step]
    messages: list[dict]  # the full conversation sent to the model, for debugging

    @property
    def prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(step.completion_tokens for step in self.steps)

    @property
    def sources(self) -> list[dict]:
        return [source for step in self.steps for source in step.sources]


def _elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def execute_tool_call(call: ToolCall, tools: dict[str, Tool]) -> Step:
    """Run one tool call. Never raises: every failure becomes a message for the model.

    Why not raise? The model made the mistake, so the model should see it and get a
    chance to correct itself (another tool, fixed arguments, or an honest answer).
    """
    start = time.perf_counter()
    if call.name not in tools:
        return Step(
            kind="tool_call",
            tool_name=call.name,
            is_error=True,
            duration_ms=_elapsed_ms(start),
            output=f"Error: Unknown tool '{call.name}'. Available tools: {', '.join(tools.keys())}"
        )

    arguments = call.arguments
    try:
        json_args = json.loads(arguments) if arguments else {}
        if not isinstance(json_args, dict):
            raise ValueError("Arguments must be a JSON object.")
    except (ValueError) as e:
        return Step(
            kind="tool_call",
            tool_name=call.name,
            is_error=True,
            duration_ms=_elapsed_ms(start),
            output=f"Error: Invalid arguments for tool '{call.name}': {str(e)}"
        )
    
    try:
        output_tool = tools.get(call.name).run(json_args)
    except Exception as e:
        return Step(
            kind="tool_call",
            tool_name=call.name,
            is_error=True,
            duration_ms=_elapsed_ms(start),
            output=f"Error: Failed to run tool '{call.name}': {str(e)}",
            arguments=json_args
        )

    return Step(
                kind="tool_call",
                tool_name=call.name,
                is_error=False,
                duration_ms=_elapsed_ms(start),
                output=output_tool.content,
                sources= output_tool.sources, 
                arguments=json_args
            )



def _assistant_message(response: LLMResponse) -> dict:
    """The model's turn, in the OpenAI format, so the next call sees what it asked for."""

    if not response.tool_calls:
        return {
            "role": "assistant",
            "content": response.content,
        }

    list_of_tool_calls = []
    for call in response.tool_calls:
        list_of_tool_calls.append({
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments
            }
        })

    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": list_of_tool_calls
    }


def run_agent(
    llm: ChatModel,
    system_prompt: str,
    user_message: str,
    tools: dict[str, Tool],
    history: list[dict] | None = None,
    max_iterations: int = 5,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    """Run the agent loop: let the LLM call tools until it can answer."""
    # The earlier turns go BETWEEN the instructions and the new question: the model
    # reads the conversation in order, and the last message is what it answers.
    # `*history` copies the messages into a new list: the caller's list is untouched.
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *(history or []),
        {"role": "user", "content": user_message},
    ]

    tool_schemas = [tool.schema() for tool in tools.values()]
    steps: list[Step] = []

    def record(step: Step) -> None:
        """Keep the step, and announce it (tomorrow: streamed live to the browser)."""
        steps.append(step)
        if on_step is not None:
            on_step(step)

    for _ in range(max_iterations):
        start = time.perf_counter()
        response = llm.complete(messages=messages, tools=tool_schemas)
        duration_ms = _elapsed_ms(start)
        record(Step(
            kind="llm_call",
            duration_ms=duration_ms,
            content=response.content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens
        ))

        if not response.tool_calls:
            messages.append(_assistant_message(response))
            return AgentResult(answer=response.content, stop_reason="answer", steps=steps, messages=messages)

        messages.append(_assistant_message(response))
        for call in response.tool_calls:
            step = execute_tool_call(call, tools)
            record(step)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": step.output
            })

    # The loop ran out of iterations without an answer (a model stuck calling tools).
    # This limit is the agent's safety net: without it, it could loop and spend forever.
    return AgentResult(answer=None, stop_reason="max_iterations", steps=steps, messages=messages)
