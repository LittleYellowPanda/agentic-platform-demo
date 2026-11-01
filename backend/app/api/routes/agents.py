"""HTTP routes for tools, agents and agent runs."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Agent, AgentRun, Conversation, KnowledgeBase
from app.db.session import get_db
from app.schemas.agents import (
    AgentRead,
    AgentWrite,
    ConversationRead,
    ConversationSummary,
    LLMStatusRead,
    ProviderModelsRead,
    RunCreate,
    RunRead,
    RunSummary,
    ToolInfoRead,
)
from app.services.agent_runs import execute_run
from app.services.llm import llm_mode
from app.services.providers import all_models, get_provider
from app.services.run_stream import stream_run
from app.services.tools.catalog import list_tools

router = APIRouter(tags=["agents"])

DbSession = Annotated[Session, Depends(get_db)]


def _to_read(agent: Agent) -> AgentRead:
    return AgentRead(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        system_prompt=agent.system_prompt,
        model=agent.model,
        provider=agent.provider,
        tools=agent.tools,
        knowledge_base_ids=sorted(kb.id for kb in agent.knowledge_bases),
        max_iterations=agent.max_iterations,
        created_at=agent.created_at,
    )


def _get_agent_or_404(db: Session, agent_id: int) -> Agent:
    agent = db.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")
    return agent


def _check_conversation(db: Session, agent: Agent, conversation_id: int | None) -> None:
    """A conversation to continue must exist AND belong to this agent: otherwise one
    agent could read another agent's thread through its history."""
    if conversation_id is None:
        return
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.agent_id != agent.id:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found for this agent.",
        )


def _check_provider(provider_id: str | None) -> None:
    """422 for a provider this platform doesn't have: saving the agent, or running one
    whose provider was removed from .env, must fail clearly, not at the first LLM call."""
    if provider_id is not None and get_provider(provider_id) is None:
        raise HTTPException(
            status_code=422,
            detail=f"The LLM provider '{provider_id}' isn't configured on this platform. "
            "Choose another model in the agent's settings.",
        )


def _load_knowledge_bases(db: Session, ids: list[int]) -> list[KnowledgeBase]:
    kbs = (
        list(db.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(ids))))
        if ids
        else []
    )
    missing = sorted(set(ids) - {kb.id for kb in kbs})
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Knowledge base(s) not found: {', '.join(map(str, missing))}.",
        )
    return kbs


def _apply(agent: Agent, payload: AgentWrite, kbs: list[KnowledgeBase]) -> None:
    agent.name = payload.name
    agent.description = payload.description
    agent.system_prompt = payload.system_prompt
    agent.model = payload.model
    agent.provider = payload.provider
    agent.tools = payload.tools
    agent.max_iterations = payload.max_iterations
    agent.knowledge_bases = kbs


# --- Tools --------------------------------------------------------------------------


@router.get("/tools", response_model=list[ToolInfoRead])
def get_tools() -> list[ToolInfoRead]:
    """Tools an agent can use (to build the switches in the Agent Builder)."""
    return [ToolInfoRead(**vars(info)) for info in list_tools()]


@router.get("/llm", response_model=LLMStatusRead)
def get_llm_status() -> LLMStatusRead:
    """Which LLM the platform uses (the UI shows a banner in mock mode). Never the key."""
    from app.core.config import settings

    mode = llm_mode()
    return LLMStatusRead(
        mode=mode,
        default_model="mock" if mode == "mock" else settings.llm_model,
        base_url=None if mode == "mock" else settings.llm_base_url,
    )


@router.get("/llm/models", response_model=list[ProviderModelsRead])
def get_llm_models() -> list[ProviderModelsRead]:
    """Every configured provider and the models it offers, for the model dropdown.
    Asked live (cached 5 min): a model pulled into Ollama shows up without a restart."""
    return [
        ProviderModelsRead(
            provider=entry.provider.id,
            label=entry.provider.label,
            default=entry.is_default,
            error=entry.error,
            models=[vars(m) for m in entry.models],
        )
        for entry in all_models()
    ]


# --- Agents -------------------------------------------------------------------------


@router.post("/agents", response_model=AgentRead, status_code=201)
def create_agent(payload: AgentWrite, db: DbSession) -> AgentRead:
    _check_provider(payload.provider)
    agent = Agent()
    _apply(agent, payload, _load_knowledge_bases(db, payload.knowledge_base_ids))
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return _to_read(agent)


@router.get("/agents", response_model=list[AgentRead])
def list_agents(db: DbSession) -> list[AgentRead]:
    # selectinload: the linked KBs of all agents in ONE extra query, not one per agent
    agents = db.scalars(
        select(Agent)
        .options(selectinload(Agent.knowledge_bases))
        .order_by(Agent.created_at.desc())
    )
    return [_to_read(agent) for agent in agents]


@router.get("/agents/{agent_id}", response_model=AgentRead)
def get_agent(agent_id: int, db: DbSession) -> AgentRead:
    return _to_read(_get_agent_or_404(db, agent_id))


@router.put("/agents/{agent_id}", response_model=AgentRead)
def update_agent(agent_id: int, payload: AgentWrite, db: DbSession) -> AgentRead:
    agent = _get_agent_or_404(db, agent_id)
    _check_provider(payload.provider)
    _apply(agent, payload, _load_knowledge_bases(db, payload.knowledge_base_ids))
    db.commit()
    db.refresh(agent)
    return _to_read(agent)


@router.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: int, db: DbSession) -> Response:
    _get_agent_or_404(db, agent_id)
    db.execute(
        delete(Agent).where(Agent.id == agent_id)
    )  # runs and steps cascade in SQL
    db.commit()
    return Response(status_code=204)


# --- Runs ---------------------------------------------------------------------------


@router.post("/agents/{agent_id}/runs", response_model=RunRead, status_code=201)
def create_run(agent_id: int, payload: RunCreate, db: DbSession) -> AgentRun:
    """Ask the agent a question and wait for the full run (or stream it: .../runs/stream).

    Failures of the LLM provider don't make this request fail: they're part of the
    run's story, stored with status "error" and a message for the user.
    """
    agent = _get_agent_or_404(db, agent_id)
    _check_conversation(db, agent, payload.conversation_id)
    _check_provider(agent.provider)
    return execute_run(
        db, agent, payload.message, conversation_id=payload.conversation_id
    )


@router.post("/agents/{agent_id}/runs/stream")
def create_run_streamed(
    agent_id: int, payload: RunCreate, db: DbSession
) -> StreamingResponse:
    """Same run, but each step is sent to the browser as soon as it happens (SSE).

    Checks happen BEFORE the stream starts, so a missing agent or an invalid message
    is a normal 404 / 422. Once streaming starts, the 200 status is already sent:
    problems can only be reported as events.
    """
    agent = _get_agent_or_404(db, agent_id)
    _check_conversation(db, agent, payload.conversation_id)
    _check_provider(agent.provider)
    return StreamingResponse(
        stream_run(agent_id, payload.message, conversation_id=payload.conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # never cache a live stream
            "X-Accel-Buffering": "no",  # tell nginx-like proxies not to buffer it
        },
    )


@router.get("/agents/{agent_id}/runs", response_model=list[RunSummary])
def list_runs(agent_id: int, db: DbSession, limit: int = 20) -> list[AgentRun]:
    _get_agent_or_404(db, agent_id)
    return list(
        db.scalars(
            select(AgentRun)
            .where(AgentRun.agent_id == agent_id)
            .order_by(AgentRun.created_at.desc())
            .limit(min(max(limit, 1), 100))
        )
    )


@router.get("/runs/{run_id}", response_model=RunRead)
def get_run(run_id: int, db: DbSession) -> AgentRun:
    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return run


# --- Conversations ------------------------------------------------------------------


@router.get(
    "/agents/{agent_id}/conversations", response_model=list[ConversationSummary]
)
def list_conversations(
    agent_id: int, db: DbSession, limit: int = 30
) -> list[ConversationSummary]:
    """The agent's threads, most recently active first, with their number of turns."""
    _get_agent_or_404(db, agent_id)
    turn_count = func.count(AgentRun.id).label("turn_count")
    last_activity = func.coalesce(
        func.max(AgentRun.created_at), Conversation.created_at
    ).label("last_activity")
    rows = db.execute(
        select(Conversation, turn_count, last_activity)
        .outerjoin(AgentRun, AgentRun.conversation_id == Conversation.id)
        .where(Conversation.agent_id == agent_id)
        .group_by(Conversation.id)
        .order_by(last_activity.desc())
        .limit(min(max(limit, 1), 100))
    )
    return [
        ConversationSummary(
            id=conversation.id,
            agent_id=conversation.agent_id,
            title=conversation.title,
            created_at=conversation.created_at,
            turn_count=count,
            last_activity=activity,
        )
        for conversation, count, activity in rows
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
def get_conversation(conversation_id: int, db: DbSession) -> Conversation:
    conversation = db.scalars(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        # all runs and all their steps in 2 extra queries, not one per run
        .options(selectinload(Conversation.runs).selectinload(AgentRun.steps))
    ).first()
    if conversation is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation {conversation_id} not found."
        )
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int, db: DbSession) -> Response:
    if db.get(Conversation, conversation_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation {conversation_id} not found."
        )
    db.execute(
        delete(Conversation).where(Conversation.id == conversation_id)
    )  # runs cascade
    db.commit()
    return Response(status_code=204)
