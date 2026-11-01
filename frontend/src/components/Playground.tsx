import { useEffect, useState, type FormEvent } from "react";

import { deleteConversation, errorMessage, getConversation } from "../api/client";
import type { Agent, Conversation, Run } from "../api/types";
import { useRunStream, type RunStreamState } from "../hooks/useRunStream";
import ConversationList from "./ConversationList";
import RunTrace, { type TraceView } from "./RunTrace";

type Props = { agent: Agent };

// A chat with the agent: earlier turns (from the database) + the live turn (SSE).
// Each new question is sent with the conversation's id; the backend adds the
// previous questions and answers as history, so the agent remembers the thread.
export default function Playground({ agent }: Props) {
  const { state, start, stop, reset } = useRunStream();
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [thread, setThread] = useState<Conversation | null>(null);
  const [threadError, setThreadError] = useState<string | null>(null);
  const [listVersion, setListVersion] = useState(0);

  const isRunning = state.status === "running";
  const now = useNow(isRunning);
  const elapsed = state.startedAt ? Math.max(0, Math.floor((now - state.startedAt) / 1000)) : 0;

  // (Re)load the thread when another conversation is opened, and when a run ends
  // (it's then saved, with its final status, tokens and duration)
  const finishedRunId = state.status === "done" || state.status === "failed" ? state.runId : null;
  useEffect(() => {
    if (conversationId === null) return;
    let ignore = false;
    getConversation(conversationId).then(
      (loaded) => {
        if (!ignore) {
          setThread(loaded);
          setThreadError(null);
        }
      },
      (err: unknown) => {
        if (!ignore) setThreadError(errorMessage(err));
      },
    );
    return () => {
      ignore = true;
    };
  }, [conversationId, finishedRunId]);

  // Derived, not stored: a thread loaded for another conversation is never shown
  const pastRuns: Run[] = thread && thread.id === conversationId ? thread.runs : [];

  // The live turn stays until the reloaded thread has the same run FINISHED (after
  // "Stop watching", the saved copy may still say "running": keep the live one)
  const liveBelongsHere =
    state.conversationId === null || state.conversationId === conversationId;
  const showLive =
    state.status !== "idle" &&
    liveBelongsHere &&
    !pastRuns.some((r) => r.id === state.runId && r.status !== "running");
  const shownRuns = showLive ? pastRuns.filter((r) => r.id !== state.runId) : pastRuns;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    if (!text || isRunning) return;
    setQuestion("");
    void start(agent.id, text, conversationId, (info) => {
      // First question of a new conversation: from now on, we're in it
      setConversationId(info.conversationId);
    });
  }

  function newConversation() {
    reset();
    setConversationId(null);
    setThreadError(null);
  }

  function openConversation(id: number) {
    if (id === conversationId) return;
    reset();
    setConversationId(id);
  }

  async function handleDelete() {
    if (conversationId === null) return;
    try {
      await deleteConversation(conversationId);
      newConversation();
      setListVersion((v) => v + 1);
    } catch (err) {
      setThreadError(errorMessage(err));
    }
  }

  const title = thread && thread.id === conversationId ? thread.title : "New conversation";
  const isEmpty = shownRuns.length === 0 && !showLive;

  return (
    <>
      <section className="panel playground">
        <div className="panel-header">
          <h2 className="thread-title">{title}</h2>
          <div className="thread-actions">
            {conversationId !== null && (
              <button type="button" className="small-button" onClick={handleDelete} disabled={isRunning}>
                Delete
              </button>
            )}
            <button type="button" className="small-button" onClick={newConversation}>
              New conversation
            </button>
          </div>
        </div>

        {threadError && <p className="error-text small">{threadError}</p>}

        <div className="thread">
          {isEmpty ? (
            <p className="muted small">
              Ask a question: each step the agent takes appears as it happens. Follow-up
              questions continue the conversation: the agent remembers the earlier
              questions and answers.
            </p>
          ) : (
            <>
              {shownRuns.map((run, index) => (
                <RunTrace
                  key={run.id}
                  trace={fromRun(run)}
                  // only the latest turn shows its steps; older ones fold away
                  stepsCollapsed={showLive || index < shownRuns.length - 1}
                />
              ))}
              {showLive && <RunTrace trace={fromStream(state, elapsed)} />}
            </>
          )}
        </div>

        <form className="ask-form" onSubmit={handleSubmit}>
          <label className="visually-hidden" htmlFor="question">
            Question
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter adds a line (like chat apps)
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
            rows={2}
            maxLength={4000}
            placeholder={
              pastRuns.length > 0
                ? "Ask a follow-up question…"
                : "What is the weather in Lyon, and what is 15% of 2,340?"
            }
          />
          {/* Different keys = two different DOM buttons. Without them, React reuses
              the same <button> and only flips its type: clicking "Stop" re-renders
              it as type="submit" during the click, and the browser then submits
              the form, starting a new run. */}
          {isRunning ? (
            <button key="stop" type="button" onClick={stop}>
              Stop watching
            </button>
          ) : (
            <button key="run" type="submit" className="primary" disabled={!question.trim()}>
              Send
            </button>
          )}
        </form>
      </section>

      <ConversationList
        // reloads when a run starts or ends, or after a delete: derived, no effect
        key={`${state.runId ?? "none"}:${state.status}:${listVersion}`}
        agentId={agent.id}
        selectedId={conversationId}
        onOpen={openConversation}
      />
    </>
  );
}

function fromStream(state: RunStreamState, elapsedSeconds: number): TraceView {
  const summary = state.summary;
  return {
    question: state.question,
    model: summary?.model ?? state.model,
    provider: summary?.provider ?? state.provider,
    steps: state.steps,
    status:
      state.status === "running" ? "streaming" : state.status === "failed" ? "failed" : summary!.status,
    answer: summary?.answer ?? null,
    error: state.error ?? summary?.error_message ?? null,
    promptTokens: summary?.prompt_tokens ?? sum(state, "prompt_tokens"),
    completionTokens: summary?.completion_tokens ?? sum(state, "completion_tokens"),
    durationMs: summary?.duration_ms ?? null,
    elapsedSeconds,
  };
}

function fromRun(run: Run): TraceView {
  return {
    question: run.user_message,
    model: run.model,
    provider: run.provider,
    steps: run.steps,
    status: run.status,
    answer: run.answer,
    error: run.error_message,
    promptTokens: run.prompt_tokens,
    completionTokens: run.completion_tokens,
    durationMs: run.duration_ms,
  };
}

function sum(state: RunStreamState, field: "prompt_tokens" | "completion_tokens"): number {
  return state.steps.reduce((total, step) => total + step[field], 0);
}

// The current time, refreshed every second while `active` (for the elapsed timer)
function useNow(active: boolean): number {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer); // stop the timer when the run ends or we leave
  }, [active]);
  return now;
}
