import { useState, type FormEvent } from "react";

import {
  createAgent,
  errorMessage,
  getLlmStatus,
  listKnowledgeBases,
  listModels,
  listTools,
  updateAgent,
} from "../api/client";
import type { Agent, AgentWrite, ModelOption, ProviderModels } from "../api/types";
import { pluralize } from "../format";
import { useAsyncData } from "../hooks/useAsyncData";

const DEFAULT_PROMPT = "You are a helpful assistant. Answer concisely and precisely.";

// A <select> holds one string: the provider and the model are joined as "provider|model".
// "" = the platform default; "|name" = a model of the default provider (agents saved
// before providers existed). Model names contain ":" and "/", never "|".
function toChoice(provider: string | null | undefined, model: string | null | undefined) {
  return model ? `${provider ?? ""}|${model}` : "";
}

function fromChoice(choice: string): { provider: string | null; model: string | null } {
  if (!choice) return { provider: null, model: null };
  const at = choice.indexOf("|");
  return { provider: choice.slice(0, at) || null, model: choice.slice(at + 1) };
}

function optionLabel(model: ModelOption): string {
  if (model.tools === false) return `${model.label} (no tool calling)`;
  if (model.tools === null) return `${model.label} (tool support unknown)`;
  return model.label;
}

// Is the agent's saved choice still in the lists? (a model removed from Ollama, say)
function isListed(groups: ProviderModels[], choice: string): boolean {
  const { provider, model } = fromChoice(choice);
  return groups.some(
    (g) => g.provider === provider && g.models.some((m) => m.id === model),
  );
}

type Props = {
  agent?: Agent; // given: edit this agent; absent: create a new one
  onSaved: (agent: Agent) => void;
  onCancel: () => void;
};

// The Agent Builder. Everything the backend validates is validated there (unknown
// tools, missing KBs, iteration bounds): the form only guides, the API decides.
export default function AgentForm({ agent, onSaved, onCancel }: Props) {
  const tools = useAsyncData(listTools);
  const knowledgeBases = useAsyncData(listKnowledgeBases);
  const llm = useAsyncData(getLlmStatus);
  const models = useAsyncData(listModels);

  const [name, setName] = useState(agent?.name ?? "");
  const [description, setDescription] = useState(agent?.description ?? "");
  const [systemPrompt, setSystemPrompt] = useState(agent?.system_prompt ?? DEFAULT_PROMPT);
  const [modelChoice, setModelChoice] = useState(toChoice(agent?.provider, agent?.model));
  const [selectedTools, setSelectedTools] = useState<string[]>(agent?.tools ?? []);
  const [kbIds, setKbIds] = useState<number[]>(agent?.knowledge_base_ids ?? []);
  const [maxIterations, setMaxIterations] = useState(String(agent?.max_iterations ?? 6));
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectableTools = tools.data?.filter((t) => !t.automatic) ?? [];
  const kbTool = tools.data?.find((t) => t.automatic);

  // Toggle a value in a list without mutating it: React compares by reference
  function toggle<T>(list: T[], value: T): T[] {
    return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);

    const payload: AgentWrite = {
      name,
      description: description.trim() || null,
      system_prompt: systemPrompt,
      ...fromChoice(modelChoice), // model + provider
      tools: selectedTools,
      knowledge_base_ids: kbIds,
      max_iterations: Number(maxIterations),
    };

    try {
      const saved = agent ? await updateAgent(agent.id, payload) : await createAgent(payload);
      onSaved(saved);
    } catch (err) {
      setError(errorMessage(err));
      setIsSubmitting(false);
    }
  }

  return (
    <form className="panel create-form agent-form" onSubmit={handleSubmit}>
      <h2>{agent ? `Edit ${agent.name}` : "New agent"}</h2>

      <label className="field">
        <span>Name</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Fund analyst"
          maxLength={255}
          required
        />
      </label>

      <label className="field">
        <span>Description</span>
        <input
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Answers questions about our fund prospectuses"
        />
      </label>

      <label className="field">
        <span>Instructions (system prompt)</span>
        <textarea
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          rows={5}
          required
        />
        <span className="hint">
          The platform adds its own rules after yours: tool results are data, not
          instructions; cite your sources; don't guess.
        </span>
      </label>

      <div className="field-row">
        <label className="field">
          <span>Model</span>
          <select
            value={modelChoice}
            onChange={(e) => setModelChoice(e.target.value)}
            disabled={models.isLoading}
          >
            <option value="">
              Platform default ({llm.data?.mode === "mock" ? "mock" : llm.data?.default_model ?? "…"})
            </option>
            {modelChoice && models.data && !isListed(models.data, modelChoice) && (
              // Keep the saved choice visible, even if its provider no longer lists it
              <option value={modelChoice}>
                {fromChoice(modelChoice).model} (current, not listed)
              </option>
            )}
            {models.data?.map((group) => (
              <optgroup
                key={group.provider}
                label={group.default ? `${group.label} (default provider)` : group.label}
              >
                {group.error && <option disabled>{group.error}</option>}
                {group.models.map((m) => (
                  <option
                    key={m.id}
                    value={toChoice(group.provider, m.id)}
                    // Agents need tools: a model that can't call them would fail every run
                    disabled={m.tools === false}
                  >
                    {optionLabel(m)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
          <span className="hint">
            {models.isLoading
              ? "Asking the providers for their models…"
              : models.error
                ? `Can't list models: ${models.error}`
                : models.data?.length === 0
                  ? "No provider configured: set LLM_API_KEY, OLLAMA_BASE_URL or OPENROUTER_API_KEY in .env."
                  : "Models that can't call tools are greyed out: agents need them."}
          </span>
        </label>
        <label className="field">
          <span>Max LLM calls per run</span>
          <input
            type="number"
            min={1}
            max={15}
            value={maxIterations}
            onChange={(e) => setMaxIterations(e.target.value)}
          />
          <span className="hint">Safety limit against endless tool loops (1 to 15).</span>
        </label>
      </div>

      <fieldset className="choice-group">
        <legend>Tools</legend>
        {tools.error && <p className="error-text small">{tools.error}</p>}
        {selectableTools.map((tool) => {
          const checked = selectedTools.includes(tool.name);
          return (
            <label
              key={tool.name}
              className="choice"
              data-unavailable={tool.available ? undefined : "true"}
            >
              <input
                type="checkbox"
                checked={checked}
                // An unavailable tool can still be UNchecked (an agent saved earlier)
                disabled={!tool.available && !checked}
                onChange={() => setSelectedTools((list) => toggle(list, tool.name))}
              />
              <span className="choice-text">
                <span className="choice-name">{tool.name}</span>
                <span className="choice-description">{tool.description}</span>
                {!tool.available && (
                  <span className="choice-warning">Unavailable: {tool.unavailable_reason}</span>
                )}
              </span>
            </label>
          );
        })}
      </fieldset>

      <fieldset className="choice-group">
        <legend>Knowledge bases</legend>
        {knowledgeBases.data?.length === 0 && (
          <p className="muted small">
            No knowledge base yet. Create one in the Knowledge bases page to give the agent
            documents to search.
          </p>
        )}
        {knowledgeBases.data?.map((kb) => (
          <label key={kb.id} className="choice">
            <input
              type="checkbox"
              checked={kbIds.includes(kb.id)}
              onChange={() => setKbIds((list) => toggle(list, kb.id))}
            />
            <span className="choice-text">
              <span className="choice-name" data-plain="true">{kb.name}</span>
              <span className="choice-description">
                {kb.description ?? "No description"} · {pluralize(kb.document_count, "document")}
              </span>
            </span>
          </label>
        ))}
        {kbTool && (
          <p className="hint">
            Linking a knowledge base gives the agent the <code>{kbTool.name}</code> tool
            automatically.
          </p>
        )}
      </fieldset>

      {error && (
        <p className="error-list" role="alert">
          {error}
        </p>
      )}

      <div className="form-actions">
        <button type="submit" className="primary" disabled={isSubmitting}>
          {isSubmitting ? "Saving…" : agent ? "Save changes" : "Create agent"}
        </button>
        <button type="button" onClick={onCancel} disabled={isSubmitting}>
          Cancel
        </button>
      </div>
    </form>
  );
}
