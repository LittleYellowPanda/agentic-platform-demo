"""Specification of conversation memory: build_history, and the history in your loop.

Run: uv run pytest tests/test_conversation.py -v
"""

from types import SimpleNamespace

from app.services.agent_loop import run_agent
from app.services.conversation import TITLE_LENGTH, build_history, conversation_title
from app.services.llm import LLMResponse


def run(question: str, answer: str | None) -> SimpleNamespace:
    return SimpleNamespace(user_message=question, answer=answer)


# --- build_history ---------------------------------------------------------------


def test_no_previous_run_means_no_history():
    assert build_history([]) == []


def test_each_turn_becomes_a_user_then_an_assistant_message():
    history = build_history([run("Weather in Lyon?", "14 °C, overcast.")])
    assert history == [
        {"role": "user", "content": "Weather in Lyon?"},
        {"role": "assistant", "content": "14 °C, overcast."},
    ]


def test_turns_stay_in_chronological_order():
    history = build_history([run("first", "A1"), run("second", "A2")])
    assert [m["content"] for m in history] == ["first", "A1", "second", "A2"]


def test_todo1_runs_without_an_answer_are_skipped():
    runs = [
        run("ok", "fine"),
        run("failed", None),
        run("limit", ""),
        run("blank", "   "),
    ]
    assert [m["content"] for m in build_history(runs)] == ["ok", "fine"]


def test_todo2_only_the_last_turns_are_kept():
    runs = [run(f"q{i}", f"a{i}") for i in range(1, 6)]
    history = build_history(runs, max_turns=2)
    assert [m["content"] for m in history] == ["q4", "a4", "q5", "a5"]


def test_todo2_the_limit_counts_answered_turns_not_raw_runs():
    # 2 answered turns wanted: the failed run in between must not use up a slot
    runs = [run("q1", "a1"), run("q2", "a2"), run("q3", None)]
    assert len(build_history(runs, max_turns=2)) == 4


def test_todo2_zero_turns_means_no_history():
    assert build_history([run("q", "a")], max_turns=0) == []


def test_messages_are_plain_dicts_in_openai_format():
    message = build_history([run("q", "a")])[0]
    assert set(message) == {"role", "content"}


# --- The history in YOUR loop ------------------------------------------------------


class RecordingLLM:
    """Answers at once, and keeps a copy of the messages it received."""

    model = "recording"

    def __init__(self):
        self.received: list[dict] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.received = [dict(m) for m in messages]
        return LLMResponse(content="Paris: 12 °C.")


HISTORY = [
    {"role": "user", "content": "Weather in Lyon?"},
    {"role": "assistant", "content": "14 °C, overcast."},
]


def test_loop_history_goes_between_the_system_prompt_and_the_new_question():
    llm = RecordingLLM()
    run_agent(llm, "You are helpful.", "And in Paris?", {}, history=HISTORY)

    assert llm.received == [
        {"role": "system", "content": "You are helpful."},
        *HISTORY,
        {"role": "user", "content": "And in Paris?"},
    ]


def test_loop_without_history_works_as_before():
    llm = RecordingLLM()
    run_agent(llm, "sys", "Hello", {})
    assert [m["role"] for m in llm.received] == ["system", "user"]


def test_loop_does_not_modify_the_history_it_was_given():
    history = [dict(m) for m in HISTORY]
    run_agent(RecordingLLM(), "sys", "And in Paris?", {}, history=history)
    assert history == HISTORY  # the caller's list is untouched (no append into it)


def test_loop_result_contains_the_whole_conversation():
    result = run_agent(RecordingLLM(), "sys", "And in Paris?", {}, history=HISTORY)
    assert [m["role"] for m in result.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


# --- Small pieces around it -----------------------------------------------------------


def test_a_question_without_conversation_starts_a_new_one():
    from app.schemas.agents import RunCreate

    assert RunCreate(message="Hello").conversation_id is None
    assert RunCreate(message="And in Paris?", conversation_id=3).conversation_id == 3


def test_title_is_the_first_question_on_one_line():
    assert (
        conversation_title("  What is\nthe weather   in Lyon?  ")
        == "What is the weather in Lyon?"
    )


def test_long_titles_are_shortened_with_an_ellipsis():
    title = conversation_title("word " * 50)
    assert len(title) == TITLE_LENGTH and title.endswith("…")
