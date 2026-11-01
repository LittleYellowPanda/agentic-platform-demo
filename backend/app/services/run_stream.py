"""Stream a run's steps to the browser as they happen (Server-Sent Events).

Your agent loop is synchronous and reports steps through a callback; an HTTP
response is something we iterate over. The bridge between the two is a queue:

    worker thread                      response (iterates the generator)
    -------------                      ---------------------------------
    execute_run(on_step=...)  --put--> queue --get--> "event: step\\ndata: {...}\\n\\n"

Why a separate worker thread instead of running the loop inside the generator?
- the loop pushes steps through a callback; a generator can't yield from inside it;
- if the browser disconnects, the generator stops but the worker finishes the run:
  it's still saved, and GET /runs/{id} shows it. A run never dies half-written.

Nothing here imports FastAPI or the database at module level: the tests inject
fakes for the session factory and the run function.
"""

import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

KEEPALIVE_SECONDS = 15.0  # proxies close connections that stay silent too long
_END = object()  # sentinel: the worker is done, close the stream


def format_event(event: str, data: dict) -> str:
    """One SSE message: a name, one line of JSON, and a blank line that ends it."""
    return (
        f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
    )


def stream_run(
    agent_id: int,
    user_message: str,
    conversation_id: int | None = None,
    session_factory: Callable[[], Any] | None = None,
    run_function: Callable[..., Any] | None = None,
    load_agent: Callable[[Any, int], Any] | None = None,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> Iterator[str]:
    """Start the run in a worker thread and yield SSE messages until it ends.

    Events, in order:
        run_started  {"run_id", "conversation_id", "model", "provider", "tools"}
        step         one per step, the same fields as GET /runs/{id} steps
        done         the run summary: status, answer, error_message, tokens, duration
        error        only if the platform itself crashed (never for LLM errors,
                     which are a normal run with status "error")
    """
    if session_factory is None:
        from app.db.session import SessionLocal as session_factory
    if run_function is None:
        from app.services.agent_runs import execute_run as run_function
    if load_agent is None:

        def load_agent(db: Any, agent_id: int) -> Any:
            from app.db.models import Agent

            return db.get(Agent, agent_id)

    events: queue.Queue = queue.Queue()

    def send(event: str, data: dict) -> None:
        events.put(format_event(event, data))

    def worker() -> None:
        # Its own session: the request's session belongs to another thread, and
        # SQLAlchemy sessions must never be shared between threads.
        from app.schemas.agents import RunStepRead, RunSummary

        db = session_factory()
        try:
            agent = load_agent(db, agent_id)
            run = run_function(
                db,
                agent,
                user_message,
                conversation_id=conversation_id,
                on_run_created=lambda run: send(
                    "run_started",
                    {
                        "run_id": run.id,
                        # new conversation: the browser learns its id here
                        "conversation_id": run.conversation_id,
                        "model": run.model,
                        "provider": run.provider,
                        "tools": run.tools,
                    },
                ),
                # serialized HERE, in the worker thread: no ORM object crosses threads
                on_step_saved=lambda row: send(
                    "step", RunStepRead.model_validate(row).model_dump(mode="json")
                ),
            )
            send("done", RunSummary.model_validate(run).model_dump(mode="json"))
        except Exception:
            logger.exception("Streaming run for agent %s crashed", agent_id)
            send(
                "error",
                {"message": "Unexpected error in the platform. See the server logs."},
            )
        finally:
            db.close()
            events.put(_END)

    threading.Thread(target=worker, name=f"run-agent-{agent_id}", daemon=True).start()

    while True:
        try:
            item = events.get(timeout=keepalive_seconds)
        except queue.Empty:
            yield ": keep-alive\n\n"  # a comment line: ignored by the browser
            continue
        if item is _END:
            return
        yield item
