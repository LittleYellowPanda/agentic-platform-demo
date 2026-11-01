import { useCallback, useEffect, useRef, useState } from "react";

import { API_URL, readErrorMessage } from "../api/client";
import { parseSSE } from "../api/sse";
import type { RunStep, RunSummary } from "../api/types";

export type StreamStatus = "idle" | "running" | "done" | "failed";

export type RunStreamState = {
  status: StreamStatus;
  question: string;
  startedAt: number | null; // Date.now() when the question was sent, for the timer
  runId: number | null;
  conversationId: number | null; // from "run_started": a new conversation's id
  model: string | null;
  provider: string | null;
  steps: RunStep[]; // grows one step at a time while the run streams
  summary: RunSummary | null; // the "done" event: status, answer, tokens, duration
  error: string | null; // the stream itself failed (network, HTTP error, crash)
};

const IDLE: RunStreamState = {
  status: "idle",
  question: "",
  startedAt: null,
  runId: null,
  conversationId: null,
  model: null,
  provider: null,
  steps: [],
  summary: null,
  error: null,
};

/**
 * Start an agent run and follow it live.
 *
 * Why fetch and not the browser's EventSource? EventSource only does GET, and a
 * run is started with a POST carrying the question. So we read the response body
 * ourselves: a ReadableStream of bytes, decoded to text, cut into events by parseSSE.
 */
export function useRunStream() {
  const [state, setState] = useState<RunStreamState>(IDLE);
  const controllerRef = useRef<AbortController | null>(null);

  // Stop READING. The run itself continues on the server and is saved.
  const stop = useCallback(() => {
    if (!controllerRef.current) return;
    controllerRef.current.abort();
    controllerRef.current = null;
    setState((s) =>
      s.status === "running"
        ? {
            ...s,
            status: "failed",
            error: "Stopped watching. The run continues on the server: see the history.",
          }
        : s,
    );
  }, []);

  // Leaving the page stops reading (cleanup function of the effect)
  useEffect(() => () => controllerRef.current?.abort(), []);

  const start = useCallback(
    async (
      agentId: number,
      question: string,
      conversationId: number | null = null,
      // Called as soon as the server knows the run (and the conversation, if new)
      onStarted?: (info: { runId: number; conversationId: number }) => void,
    ) => {
      stop(); // one run at a time
      const controller = new AbortController();
      controllerRef.current = controller;
      setState({ ...IDLE, status: "running", question, startedAt: Date.now() });

      // Only the CURRENT run may write the state: a previous run's late events
      // (or its abort error) must not overwrite the new one
      const update = (change: (s: RunStreamState) => RunStreamState) => {
        if (controllerRef.current === controller) setState(change);
      };

      try {
        // Step 1: POST the question, with the controller's signal
        const response = await fetch(`${API_URL}/agents/${agentId}/runs/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: question, conversation_id: conversationId }),
          signal: controller.signal,
        });
        // Step 2: before the stream starts, errors are normal HTTP errors (404, 422, 429)
        if (!response.ok || !response.body) {
          const message = await readErrorMessage(response);
          update((s) => ({ ...s, status: "failed", error: message }));
          return;
        }

        // Step 3: read chunks until the stream ends
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let rest = "";
        let finished = false;
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          // stream: true -> a UTF-8 character cut between two chunks ("é") is kept
          // for the next call instead of becoming garbage
          const parsed = parseSSE(rest, decoder.decode(value, { stream: true }));
          rest = parsed.rest;

          // Step 4: each event updates the state; never mutate: build new objects
          for (const { event, data } of parsed.events) {
            const payload = JSON.parse(data);
            if (event === "run_started") {
              update((s) => ({
                ...s,
                runId: payload.run_id,
                conversationId: payload.conversation_id,
                model: payload.model,
                provider: payload.provider ?? null,
              }));
              if (controllerRef.current === controller) {
                onStarted?.({ runId: payload.run_id, conversationId: payload.conversation_id });
              }
            } else if (event === "step") {
              update((s) => ({ ...s, steps: [...s.steps, payload as RunStep] }));
            } else if (event === "done") {
              finished = true;
              update((s) => ({ ...s, status: "done", summary: payload as RunSummary }));
            } else if (event === "error") {
              finished = true;
              update((s) => ({ ...s, status: "failed", error: payload.message }));
            }
          }
        }
        // Step 5: the connection closed without "done" or "error": say so
        if (!finished) {
          update((s) => ({
            ...s,
            status: "failed",
            error: "The connection closed before the end of the run. It may still finish: check the history.",
          }));
        }
      } catch (err) {
        if (controller.signal.aborted) return; // we stopped it on purpose: not an error
        update((s) => ({
          ...s,
          status: "failed",
          error: err instanceof Error ? err.message : String(err),
        }));
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
      }
    },
    [stop],
  );

  const reset = useCallback(() => {
    stop();
    setState(IDLE);
  }, [stop]);

  return { state, start, stop, reset };
}
