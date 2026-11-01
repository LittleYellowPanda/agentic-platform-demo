"""The tools an agent can switch on. No database here: safe to import anywhere."""

from collections.abc import Callable
from dataclasses import dataclass

from app.services.tools import Tool, web_search
from app.services.tools.calculator import calculator_tool
from app.services.tools.weather import weather_tool

KB_TOOL_NAME = "search_knowledge_base"

# The KB search tool isn't selectable: it's added automatically when the agent has
# at least one knowledge base, since it needs to know which ones.
SELECTABLE_TOOLS: dict[str, Tool] = {
    calculator_tool.name: calculator_tool,
    weather_tool.name: weather_tool,
    web_search.web_search_tool.name: web_search.web_search_tool,
}

# Tools that need configuration: name -> function giving why it can't run (or None).
# Wrapped in lambdas so tests can replace the module function.
_REQUIREMENTS: dict[str, Callable[[], str | None]] = {
    web_search.NAME: lambda: web_search.unavailable_reason(),
}


def unavailable_reason(name: str) -> str | None:
    """None if the tool can run on this platform now, else a message for the user."""
    check = _REQUIREMENTS.get(name)
    return check() if check else None


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    automatic: bool  # True: enabled by linking a KB, not by a switch
    available: bool = True
    unavailable_reason: str | None = None


def list_tools() -> list[ToolInfo]:
    infos = []
    for tool in SELECTABLE_TOOLS.values():
        reason = unavailable_reason(tool.name)
        infos.append(
            ToolInfo(tool.name, tool.description, False, reason is None, reason)
        )
    infos.append(
        ToolInfo(
            KB_TOOL_NAME,
            "Searches the knowledge bases linked to the agent. Enabled automatically "
            "when at least one knowledge base is linked.",
            True,
        )
    )
    return infos
