import { useState } from "react";

import { deleteAgent, errorMessage, listKnowledgeBases } from "../api/client";
import type { Agent } from "../api/types";
import { useAsyncData } from "../hooks/useAsyncData";
import Playground from "./Playground";

type Props = {
  agent: Agent;
  onEdit: () => void;
  onDeleted: () => void;
};

export default function AgentDetail({ agent, onEdit, onDeleted }: Props) {
  const knowledgeBases = useAsyncData(listKnowledgeBases);
  const [isConfirming, setIsConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const kbNames = agent.knowledge_base_ids.map(
    (id) => knowledgeBases.data?.find((kb) => kb.id === id)?.name ?? `#${id}`,
  );

  async function handleDelete() {
    try {
      await deleteAgent(agent.id);
      onDeleted();
    } catch (err) {
      setError(errorMessage(err));
      setIsConfirming(false);
    }
  }

  return (
    <div className="kb-detail">
      <header className="kb-header">
        <div>
          <h1>{agent.name}</h1>
          {agent.description && <p className="kb-description">{agent.description}</p>}
          <dl className="kb-settings">
            <div>
              <dt>Model</dt>
              <dd>
                {agent.model ?? "platform default"}
                {agent.provider && <span className="muted"> · {agent.provider}</span>}
              </dd>
            </div>
            <div>
              <dt>Tools</dt>
              <dd>
                {agent.tools.length === 0 && kbNames.length === 0 ? (
                  "none"
                ) : (
                  <span className="chips">
                    {agent.tools.map((tool) => (
                      <code key={tool} className="chip">
                        {tool}
                      </code>
                    ))}
                    {kbNames.length > 0 && <code className="chip">search_knowledge_base</code>}
                  </span>
                )}
              </dd>
            </div>
            <div>
              <dt>Knowledge bases</dt>
              <dd>{kbNames.length > 0 ? kbNames.join(", ") : "none"}</dd>
            </div>
            <div>
              <dt>Max LLM calls</dt>
              <dd>{agent.max_iterations}</dd>
            </div>
          </dl>
        </div>
        <div className="kb-header-actions">
          <button type="button" className="small-button" onClick={onEdit}>
            Edit
          </button>
          {isConfirming ? (
            <>
              <span className="small">Delete this agent and its history?</span>
              <button type="button" className="danger small-button" onClick={handleDelete}>
                Delete
              </button>
              <button
                type="button"
                className="small-button"
                onClick={() => setIsConfirming(false)}
              >
                Cancel
              </button>
            </>
          ) : (
            <button type="button" className="small-button" onClick={() => setIsConfirming(true)}>
              Delete
            </button>
          )}
        </div>
      </header>
      {error && <p className="error-text">{error}</p>}

      <Playground agent={agent} />
    </div>
  );
}
