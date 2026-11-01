"""Try the agent loop against a real LLM, from the command line.

From backend/:
    uv run python -m scripts.try_agent "What is 15% of 2,340 euros?"
    uv run python -m scripts.try_agent --mock "Weather in Lyon and 15% of 2340?"

Uses LLM_API_KEY from the .env file (and optionally LLM_MODEL, LLM_BASE_URL);
without a key, or with --mock, the rule-based mock answers instead.
"""

import sys

from app.core.config import settings
from app.services.agent_loop import Step, run_agent
from app.services.llm import create_chat_model
from app.services.mock_llm import MockChatModel
from app.services.tools.calculator import calculator_tool
from app.services.tools.weather import weather_tool

SYSTEM_PROMPT = (
    "You are a precise assistant. Use the calculator tool for any arithmetic, "
    "even simple. Answer in the language of the question."
)


def print_step(step: Step) -> None:
    """Called by the loop at each step: the same hook will stream steps to the UI."""
    if step.kind == "llm_call":
        tokens = f"{step.prompt_tokens}+{step.completion_tokens} tokens"
        print(f"\n[LLM, {step.duration_ms} ms, {tokens}]")
        if step.content:
            print(f"  says: {step.content}")
    else:
        marker = "ERROR" if step.is_error else "tool"
        print(f"  [{marker}] {step.tool_name}({step.arguments}) -> {step.output}")


def main() -> int:
    args = sys.argv[1:]
    use_mock = "--mock" in args
    question = (
        " ".join(a for a in args if a != "--mock") or "What is 15% of 2,340 euros?"
    )
    llm = MockChatModel() if use_mock else create_chat_model()
    print(f"Model: {llm.model}\nQuestion: {question}")

    result = run_agent(
        llm,
        SYSTEM_PROMPT,
        question,
        tools={calculator_tool.name: calculator_tool, weather_tool.name: weather_tool},
        max_iterations=settings.agent_max_iterations,
        on_step=print_step,
    )

    print("\n" + "-" * 60)
    print(f"Stop reason: {result.stop_reason}")
    print(f"Tokens: {result.prompt_tokens} in, {result.completion_tokens} out")
    print(f"Answer: {result.answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
