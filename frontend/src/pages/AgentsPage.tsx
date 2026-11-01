import { useState } from "react";

import { listAgents } from "../api/client";
import type { Agent } from "../api/types";
import AgentDetail from "../components/AgentDetail";
import AgentForm from "../components/AgentForm";
import AgentList from "../components/AgentList";
import { useAsyncData } from "../hooks/useAsyncData";

type Mode = { kind: "view" } | { kind: "create" } | { kind: "edit"; agent: Agent };

// Same structure as the KB page: the page owns the list and the selection.
// The content area shows either the builder (create / edit) or the selected agent.
export default function AgentsPage() {
  const agents = useAsyncData(listAgents);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>({ kind: "view" });

  const selected = agents.data?.find((a) => a.id === selectedId) ?? null;

  function handleSelect(id: number) {
    setSelectedId(id);
    setMode({ kind: "view" });
  }

  function handleSaved(agent: Agent) {
    setSelectedId(agent.id);
    setMode({ kind: "view" });
    agents.reload();
  }

  function handleDeleted() {
    setSelectedId(null);
    agents.reload();
  }

  return (
    <div className="kb-page">
      <aside className="sidebar" aria-label="Agents">
        <div className="sidebar-header">
          <h2>Agents</h2>
          <button
            type="button"
            className="primary small-button"
            onClick={() => setMode({ kind: "create" })}
          >
            New
          </button>
        </div>
        {agents.error && <p className="error-text small">{agents.error}</p>}
        {agents.data && (
          <AgentList
            agents={agents.data}
            selectedId={mode.kind === "create" ? null : selectedId}
            onSelect={handleSelect}
          />
        )}
      </aside>

      <main className="content">
        {mode.kind === "create" ? (
          <AgentForm onSaved={handleSaved} onCancel={() => setMode({ kind: "view" })} />
        ) : mode.kind === "edit" ? (
          <AgentForm
            agent={mode.agent}
            onSaved={handleSaved}
            onCancel={() => setMode({ kind: "view" })}
          />
        ) : selected ? (
          // key: another agent = a fresh playground (no previous answer shown under it)
          <AgentDetail
            key={selected.id}
            agent={selected}
            onEdit={() => setMode({ kind: "edit", agent: selected })}
            onDeleted={handleDeleted}
          />
        ) : (
          <div className="empty-state">
            <h1>Agents</h1>
            <p className="muted">
              An agent is a model with instructions and tools. It decides on its own when to
              calculate, check the weather, search the web or your knowledge bases, and you
              can watch each step it takes.
            </p>
            <button type="button" className="primary" onClick={() => setMode({ kind: "create" })}>
              Create an agent
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
