"""SSE streaming of a run: event format, order, keep-alive, crashes. No database, no HTTP."""

import json
import threading
from types import SimpleNamespace

from app.services.run_stream import format_event, stream_run


class FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def step_row(position: int, kind: str = "llm_call") -> SimpleNamespace:
    return SimpleNamespace(
        position=position,
        kind=kind,
        duration_ms=10,
        content=None,
        prompt_tokens=0,
        completion_tokens=0,
        tool_name=None,
        arguments=None,
        output=None,
        is_error=False,
        sources=[],
    )


FINISHED_RUN = SimpleNamespace(
    id=42,
    agent_id=1,
    conversation_id=7,
    user_message="hi",
    status="completed",
    answer="Hello",
    error_message=None,
    model="mock",
    provider=None,
    prompt_tokens=5,
    completion_tokens=3,
    duration_ms=20,
    created_at="2026-10-30T10:00:00",
    tools=["calculator"],
)


def parse(messages: list[str]) -> list[tuple[str, dict]]:
    events = []
    for message in messages:
        if message.startswith(":"):
            events.append(("comment", {}))
            continue
        name_line, data_line = message.strip().split("\n")
        events.append((name_line.removeprefix("event: "), json.loads(data_line[6:])))
    return events


def run_stream(run_function, session=None, **kwargs):
    session = session or FakeSession()
    return list(
        stream_run(
            1,
            "hi",
            session_factory=lambda: session,
            run_function=run_function,
            load_agent=lambda db, agent_id: SimpleNamespace(id=agent_id),
            **kwargs,
        )
    )


def test_event_format_is_name_data_and_blank_line():
    assert format_event("step", {"a": "é"}) == 'event: step\ndata: {"a": "é"}\n\n'


def test_events_arrive_in_order_started_steps_done():
    def fake_run(
        db, agent, message, on_run_created, on_step_saved, conversation_id=None
    ):
        on_run_created(FINISHED_RUN)
        on_step_saved(step_row(0))
        on_step_saved(step_row(1, "tool_call"))
        return FINISHED_RUN

    events = parse(run_stream(fake_run))

    assert [name for name, _ in events] == ["run_started", "step", "step", "done"]
    assert events[0][1] == {
        "run_id": 42,
        "conversation_id": 7,
        "model": "mock",
        "tools": ["calculator"],
    }
    assert events[2][1]["kind"] == "tool_call"
    assert events[3][1]["status"] == "completed" and events[3][1]["answer"] == "Hello"


def test_the_run_happens_in_another_thread_with_its_own_session():
    seen = {}
    session = FakeSession()

    def fake_run(
        db, agent, message, on_run_created, on_step_saved, conversation_id=None
    ):
        seen["thread"] = threading.current_thread().name
        seen["db"] = db
        return FINISHED_RUN

    run_stream(fake_run, session=session)

    assert seen["thread"] != threading.current_thread().name
    assert seen["db"] is session and session.closed  # closed even on success


def test_a_crash_in_the_platform_becomes_an_error_event_and_the_stream_ends():
    session = FakeSession()

    def crashing_run(
        db, agent, message, on_run_created, on_step_saved, conversation_id=None
    ):
        on_run_created(FINISHED_RUN)
        raise RuntimeError("bug in our code")

    events = parse(run_stream(crashing_run, session=session))

    assert [name for name, _ in events] == ["run_started", "error"]
    assert "bug in our code" not in events[1][1]["message"]  # no internals leaked
    assert session.closed


def test_keepalive_comments_while_the_llm_thinks():
    release = threading.Event()

    def slow_run(
        db, agent, message, on_run_created, on_step_saved, conversation_id=None
    ):
        release.wait(timeout=2)
        return FINISHED_RUN

    stream = stream_run(
        1,
        "hi",
        session_factory=FakeSession,
        run_function=slow_run,
        load_agent=lambda db, agent_id: SimpleNamespace(id=agent_id),
        keepalive_seconds=0.05,
    )
    first = next(stream)
    release.set()
    rest = list(stream)

    assert first == ": keep-alive\n\n"
    assert parse(rest)[-1][0] == "done"


def test_a_client_that_disconnects_does_not_stop_the_run():
    finished = threading.Event()
    release = threading.Event()

    def run(db, agent, message, on_run_created, on_step_saved, conversation_id=None):
        on_run_created(FINISHED_RUN)
        release.wait(timeout=2)
        on_step_saved(step_row(0))
        finished.set()
        return FINISHED_RUN

    stream = stream_run(
        1,
        "hi",
        session_factory=FakeSession,
        run_function=run,
        load_agent=lambda db, agent_id: SimpleNamespace(id=agent_id),
    )
    next(stream)  # run_started received...
    stream.close()  # ...then the browser tab is closed
    release.set()

    assert finished.wait(timeout=2)  # the run still completed (and was saved)


def test_the_conversation_to_continue_is_passed_to_the_run():
    received = {}

    def fake_run(
        db, agent, message, on_run_created, on_step_saved, conversation_id=None
    ):
        received["conversation_id"] = conversation_id
        return FINISHED_RUN

    list(
        stream_run(
            1,
            "and in Paris?",
            conversation_id=7,
            session_factory=FakeSession,
            run_function=fake_run,
            load_agent=lambda db, agent_id: SimpleNamespace(id=agent_id),
        )
    )
    assert received == {"conversation_id": 7}
