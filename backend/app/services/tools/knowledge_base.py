"""Knowledge-base search tool: the bridge between Module A (RAG) and the agents.

Built per run, for the KBs linked to the agent. The search function is injected,
so tests can replace it without a database.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from app.services.tools import Tool, ToolOutput
from app.services.tools.catalog import KB_TOOL_NAME

if TYPE_CHECKING:  # type hints only: no database import at runtime
    from sqlalchemy.orm import Session

    from app.db.models import KnowledgeBase
    from app.schemas.knowledge_bases import SearchHit

TOOL_NAME = KB_TOOL_NAME
DEFAULT_TOP_K = 5
EXCERPT_CHARS = 300  # stored in sources for the UI; the model gets the full chunk

SearchFunction = Callable[
    ["Session", list["KnowledgeBase"], str, int], list["SearchHit"]
]


def make_knowledge_base_tool(
    db: "Session",
    knowledge_bases: Sequence["KnowledgeBase"],
    top_k: int = DEFAULT_TOP_K,
    search: SearchFunction | None = None,
) -> Tool:
    if search is None:
        from app.services.search import search_knowledge_bases as search
    kbs = list(knowledge_bases)
    names = "; ".join(
        f"{kb.name} ({kb.description})" if kb.description else kb.name for kb in kbs
    )

    def run(arguments: dict) -> ToolOutput:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("the 'query' argument is required")

        hits = search(db, kbs, query, top_k)
        if not hits:
            return ToolOutput(content="No relevant passage found in the documents.")

        # Numbered passages, each with its origin, so the model can cite them
        passages = [
            f"[{rank}] {hit.filename}, chunk {hit.chunk_index + 1} "
            f"(similarity {hit.score:.2f})\n{hit.content}"
            for rank, hit in enumerate(hits, start=1)
        ]
        sources = [
            {
                "type": "document",
                "rank": rank,
                "filename": hit.filename,
                "chunk_index": hit.chunk_index,
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "knowledge_base_id": hit.knowledge_base_id,
                "score": hit.score,
                "excerpt": hit.content[:EXCERPT_CHARS],
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        return ToolOutput(content="\n\n".join(passages), sources=sources)

    return Tool(
        name=TOOL_NAME,
        description=(
            f"Search the internal documents for passages relevant to a question. "
            f"Available knowledge bases: {names}. Use it before answering any question "
            f"about these documents, and cite the file names of the passages you use."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, as a question or keywords",
                }
            },
            "required": ["query"],
        },
        run=run,
    )
