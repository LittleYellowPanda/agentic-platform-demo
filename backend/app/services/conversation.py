"""The memory of a conversation: what the model sees of the previous turns.

The model has no memory between calls: each run starts from scratch. To make a
conversation, each new question is sent with the previous ones and their answers:

    system prompt
    user: "What is the weather in Lyon?"           <- turn 1, from the database
    assistant: "14 °C and overcast."                <- turn 1's FINAL answer
    user: "And in Paris?"                          <- the new question

"And in Paris?" only makes sense because turn 1 is there.

Only the final answers are resent, not the tool calls and results of earlier turns:
much cheaper in tokens, and usually enough. The trade-off: the model no longer has
the exact passages it read before (it must search again if it needs them).
"""

from collections.abc import Sequence
from typing import Protocol

DEFAULT_MAX_TURNS = 10


TITLE_LENGTH = 80


def conversation_title(question: str) -> str:
    """The first question, on one line and shortened: the thread's name in the list."""
    one_line = " ".join(question.split())
    if len(one_line) <= TITLE_LENGTH:
        return one_line
    return one_line[: TITLE_LENGTH - 1].rstrip() + "…"


class PastRun(Protocol):
    """What build_history needs from a run (an AgentRun row, or a test double)."""

    user_message: str
    answer: str | None


def build_history(
    runs: Sequence[PastRun], max_turns: int = DEFAULT_MAX_TURNS
) -> list[dict]:
    """Messages for the previous turns, oldest first, ready to put before the question.

    `runs` are the earlier runs of the conversation, oldest first.
    """
    # Step 1: keep only the runs that produced an answer. A run that failed or hit
    #   the iteration limit has no answer (None or empty): resending its question
    #   alone would leave the model with an unanswered question in the middle.
    answered = [run for run in runs if run.answer and run.answer.strip()]

    # Step 2: keep only the LAST max_turns of them (the most recent context matters
    #   most, and the cost of each call grows with every turn kept). max_turns <= 0
    #   means no history at all.
    if max_turns <= 0:
        return []
    recent = answered[-max_turns:]

    # Step 3: one user message + one assistant message per turn, in order
    history: list[dict] = []
    for run in recent:
        history.append({"role": "user", "content": run.user_message})
        history.append({"role": "assistant", "content": run.answer})
    return history
