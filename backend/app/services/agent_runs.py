"""Run an agent on one question, and record the whole trace in the database."""

import logging
import time
from collections.abc import Callable

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.models import Agent, AgentRun, Conversation, RunStep
from app.services.agent_loop import Step, run_agent
from app.services.conversation import build_history, conversation_title
from app.services.llm import ChatModel, create_chat_model, describe_llm_error
from app.services.prompts import build_system_prompt
from app.services.tools.registry import build_tools

logger = logging.getLogger(__name__)


def _to_row(step: Step, run_id: int, position: int) -> RunStep:
    return RunStep(
        run_id=run_id,
        position=position,
        kind=step.kind,
        duration_ms=step.duration_ms,
        content=step.content,
        prompt_tokens=step.prompt_tokens,
        completion_tokens=step.completion_tokens,
        tool_name=step.tool_name,
        arguments=step.arguments,
        output=step.output,
        is_error=step.is_error,
        sources=step.sources,
    )


def _fail(db: Session, run: AgentRun, message: str, start: float) -> AgentRun:
    db.rollback()  # discard anything half-written; saved steps are already committed
    run.status = "error"
    run.error_message = message
    run.duration_ms = round((time.perf_counter() - start) * 1000)
    db.commit()
    db.refresh(run)
    return run


def execute_run(
    db: Session,
    agent: Agent,
    user_message: str,
    llm: ChatModel | None = None,
    on_run_created: Callable[[AgentRun], None] | None = None,
    on_step_saved: Callable[[RunStep], None] | None = None,
    conversation_id: int | None = None,
) -> AgentRun:
    """Run the agent and return the finished run, with its steps.

    The run row is created first, and each step is saved as soon as the loop
    reports it: a run that fails halfway still shows how far it got.
    The two optional hooks let a caller follow the run live (SSE streaming).

    Without `conversation_id`, the question starts a new conversation. With it
    (already checked by the route: it exists and belongs to this agent), the
    previous turns of that conversation are sent to the model as history.
    """
    from app.core.config import settings

    model = llm or create_chat_model(agent.model, agent.provider)
    system_prompt = build_system_prompt(agent.system_prompt)
    tools = build_tools(agent, db)

    if conversation_id is None:
        conversation = Conversation(
            agent_id=agent.id, title=conversation_title(user_message)
        )
        db.add(conversation)
        db.flush()  # get its id now, in the same transaction as the run
        history: list[dict] = []
    else:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:  # the route checks this first; never trust it blindly
            raise ValueError(f"Conversation {conversation_id} not found")
        # Read BEFORE creating the new run: the history is the earlier turns only
        previous = db.scalars(
            select(AgentRun)
            .where(AgentRun.conversation_id == conversation.id)
            .order_by(AgentRun.id)
        ).all()
        history = build_history(previous, settings.conversation_max_turns)

    run = AgentRun(
        agent_id=agent.id,
        conversation_id=conversation.id,
        user_message=user_message,
        status="running",
        model=model.model,  # what really runs: the agent's choice, the default, or "mock"
        provider=getattr(model, "provider", None),  # the mock and test fakes have none
        system_prompt=system_prompt,
        tools=sorted(tools),
    )
    db.add(run)
    db.commit()
    if on_run_created:
        on_run_created(run)

    position = 0

    def save_step(step: Step) -> None:
        nonlocal position
        row = _to_row(step, run.id, position)
        db.add(row)
        db.commit()
        position += 1
        if on_step_saved:
            on_step_saved(row)

    start = time.perf_counter()
    try:
        result = run_agent(
            model,
            system_prompt,
            user_message,
            tools,
            max_iterations=agent.max_iterations,
            on_step=save_step,
            history=history,
        )
    except Exception as error:
        # Your loop never raises for tool problems. What lands here is either the LLM
        # provider failing (bad key, rate limit, outage), or a bug in our own code,
        # which must not be disguised as a provider problem.
        if type(error).__module__.startswith("openai"):
            logger.warning("Run %s: LLM call failed: %s", run.id, error)
            return _fail(db, run, describe_llm_error(error), start)
        logger.exception("Run %s: unexpected error", run.id)
        return _fail(
            db, run, "Unexpected error in the platform. See the server logs.", start
        )

    run.status = "completed" if result.stop_reason == "answer" else "max_iterations"
    # Some models start their answer with blank lines: clean the edges, never the inside
    run.answer = result.answer.strip() if result.answer else result.answer
    run.prompt_tokens = result.prompt_tokens
    run.completion_tokens = result.completion_tokens
    run.duration_ms = round((time.perf_counter() - start) * 1000)
    if result.stop_reason == "max_iterations":
        run.error_message = f"The agent stopped after {agent.max_iterations} LLM calls without answering."
    db.commit()
    db.refresh(run)
    return run


INTERRUPTED_RUN_MESSAGE = "Interrupted: the server stopped during this run."


def recover_interrupted_runs(db: Session) -> int:
    """Mark runs left 'running' by a previous server process as failed.

    A run lives in the memory of the process that executes it: if the server stops
    mid-run, nothing will ever finish it, and it would show "running" forever.
    Called once at startup, like the document recovery.
    Limit: only correct with ONE server process. With several, a starting instance
    would fail the runs that another instance is still executing (production: a
    heartbeat column, or runs executed by a task queue that tracks them).
    """
    result = db.execute(
        update(AgentRun)
        .where(AgentRun.status == "running")
        .values(status="error", error_message=INTERRUPTED_RUN_MESSAGE)
    )
    db.commit()
    return result.rowcount
