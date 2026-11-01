"""Assemble the tool set of an agent for one run."""

from typing import TYPE_CHECKING

from app.services.tools import Tool
from app.services.tools.catalog import SELECTABLE_TOOLS, unavailable_reason
from app.services.tools.knowledge_base import SearchFunction, make_knowledge_base_tool

if TYPE_CHECKING:  # type hints only: no database import at runtime
    from sqlalchemy.orm import Session

    from app.db.models import Agent


def build_tools(
    agent: "Agent", db: "Session", search: SearchFunction | None = None
) -> dict[str, Tool]:
    """The tools this agent may use in a run, by name.

    Skipped: tools removed from the code since the agent was saved, and tools that
    can't run here (missing API key). Offering the model a tool that can only fail
    wastes a turn; the run's snapshot of its tools shows what was really available.
    """
    tools = {
        name: SELECTABLE_TOOLS[name]
        for name in agent.tools
        if name in SELECTABLE_TOOLS and unavailable_reason(name) is None
    }
    if agent.knowledge_bases:
        kb_tool = make_knowledge_base_tool(db, agent.knowledge_bases, search=search)
        tools[kb_tool.name] = kb_tool
    return tools
