import type { RunStatus, RunStep, Source } from "../api/types";
import { formatDuration } from "../format";
import AnswerText from "./AnswerText";

// One view for a live run (steps arriving over SSE) and a past run (from the history)
export type TraceView = {
  question: string;
  model: string | null;
  provider?: string | null; // where the model ran: "ollama", "openrouter"...
  steps: RunStep[];
  status: RunStatus | "streaming" | "failed"; // failed = the stream itself broke
  answer: string | null;
  error: string | null;
  promptTokens: number;
  completionTokens: number;
  durationMs: number | null;
  elapsedSeconds?: number; // live only: time since the question was sent
};

const STATUS_LABELS: Record<TraceView["status"], string> = {
  streaming: "Running",
  running: "Running",
  completed: "Completed",
  max_iterations: "Stopped: limit reached",
  error: "Error",
  failed: "Interrupted",
};

const STATUS_BADGE: Record<TraceView["status"], string> = {
  streaming: "processing",
  running: "processing",
  completed: "ready",
  max_iterations: "pending",
  error: "error",
  failed: "error",
};

// The heart of "transparent reasoning": every LLM call and tool call, in order,
// with what went in, what came out, how long it took and what it cost.
type Props = {
  trace: TraceView;
  // Past turns of a conversation: the steps fold away behind a summary line, so
  // the thread reads as questions and answers. The live turn keeps them open.
  stepsCollapsed?: boolean;
};

export default function RunTrace({ trace, stepsCollapsed = false }: Props) {
  const isLive = trace.status === "streaming";
  const sources = uniqueSources(trace.steps);

  function renderSteps() {
    let llmCallNumber = 0;
    return (
      <ol className="steps">
        {trace.steps.map((step) => {
          if (step.kind === "llm_call") llmCallNumber += 1;
          return (
            <StepItem
              key={step.position}
              step={step}
              llmCallNumber={llmCallNumber}
              isFinal={
                step.kind === "llm_call" &&
                step.position === trace.steps.length - 1 &&
                trace.answer !== null
              }
            />
          );
        })}
        {isLive && (
          <li className="step step-pending">
            <span className="step-marker" aria-hidden="true" />
            <span className="muted">
              Working… {trace.elapsedSeconds !== undefined && `${trace.elapsedSeconds} s`}
            </span>
          </li>
        )}
      </ol>
    );
  }

  return (
    <section className="trace" aria-label="Run trace" aria-busy={isLive}>
      <div className="trace-header">
        <p className="trace-question">{trace.question}</p>
        <div className="trace-meta">
          <span className="status-badge" data-status={STATUS_BADGE[trace.status]}>
            {STATUS_LABELS[trace.status]}
          </span>
          {trace.model && (
            <span className="muted small">
              {trace.provider ? `${trace.model} · ${trace.provider}` : trace.model}
            </span>
          )}
          {!isLive && (
            <span className="muted small">
              {formatDuration(trace.durationMs)} · {trace.promptTokens} in /{" "}
              {trace.completionTokens} out tokens
            </span>
          )}
        </div>
      </div>

      {stepsCollapsed ? (
        <details className="steps-fold">
          <summary>
            {trace.steps.length} steps · {formatDuration(trace.durationMs)} ·{" "}
            {trace.promptTokens + trace.completionTokens} tokens
          </summary>
          {renderSteps()}
        </details>
      ) : (
        renderSteps()
      )}

      {trace.answer !== null && (
        <div className="answer">
          <h3>Answer</h3>
          <AnswerText text={trace.answer} />
        </div>
      )}

      {trace.error && (
        <p className="error-list" role="alert">
          {trace.error}
        </p>
      )}

      {sources.length > 0 && (
        <div className="sources">
          <h3>Sources</h3>
          <ul>
            {sources.map((source) => (
              <SourceItem key={sourceKey(source)} source={source} />
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function StepItem({
  step,
  llmCallNumber,
  isFinal,
}: {
  step: RunStep;
  llmCallNumber: number;
  isFinal: boolean;
}) {
  if (step.kind === "llm_call") {
    return (
      <li className="step" data-kind="llm">
        <span className="step-marker" aria-hidden="true" />
        <div className="step-body">
          <div className="step-head">
            <span className="step-title">LLM call {llmCallNumber}</span>
            <span className="step-stats">
              {formatDuration(step.duration_ms)} · {step.prompt_tokens} → {step.completion_tokens}{" "}
              tokens
            </span>
          </div>
          <p className="step-note">
            {isFinal
              ? "Wrote the final answer."
              : step.content
                ? step.content
                : "Asked for tools."}
          </p>
        </div>
      </li>
    );
  }

  return (
    <li className="step" data-kind="tool" data-error={step.is_error ? "true" : undefined}>
      <span className="step-marker" aria-hidden="true" />
      <div className="step-body">
        <div className="step-head">
          <span className="step-title">
            <code>{step.tool_name}</code>
            {step.is_error && <span className="step-error-label">failed</span>}
          </span>
          <span className="step-stats">{formatDuration(step.duration_ms)}</span>
        </div>
        {step.arguments && <code className="step-args">{formatArguments(step.arguments)}</code>}
        {step.output && (
          // Errors open by default: they're what you want to read
          <details className="step-output" open={step.is_error}>
            <summary>
              {step.is_error ? "Error sent back to the model" : "Result sent back to the model"}
              {step.sources.length > 0 && ` · ${step.sources.length} sources`}
            </summary>
            <pre>{step.output}</pre>
          </details>
        )}
      </div>
    </li>
  );
}

function SourceItem({ source }: { source: Source }) {
  if (source.type === "web") {
    return (
      <li className="source">
        <a href={source.url} target="_blank" rel="noreferrer">
          {source.title}
        </a>
        <span className="muted small"> · {hostname(source.url)}</span>
        {source.excerpt && <p className="source-excerpt">{source.excerpt}</p>}
      </li>
    );
  }
  return (
    <li className="source">
      <span className="filename">{source.filename}</span>
      <span className="muted small">
        {" "}
        · chunk {source.chunk_index + 1} · similarity {source.score.toFixed(2)}
      </span>
      {source.excerpt && <p className="source-excerpt">{source.excerpt}</p>}
    </li>
  );
}

// {"city": "Lyon"} -> city: "Lyon"   (shorter than JSON, still unambiguous)
function formatArguments(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(", ");
}

function sourceKey(source: Source): string {
  return source.type === "web" ? `web:${source.url}` : `doc:${source.chunk_id}`;
}

// The same chunk or page can come back from several searches in one run
function uniqueSources(steps: RunStep[]): Source[] {
  const seen = new Map<string, Source>();
  for (const step of steps) {
    for (const source of step.sources) {
      if (!seen.has(sourceKey(source))) seen.set(sourceKey(source), source);
    }
  }
  return [...seen.values()];
}

function hostname(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}
