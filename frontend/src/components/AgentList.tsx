import type { Agent } from "../api/types";
import { pluralize } from "../format";

type Props = {
  agents: Agent[];
  selectedId: number | null;
  onSelect: (id: number) => void;
};

// Same pattern as KbList: no state of its own, data down, clicks up
export default function AgentList({ agents, selectedId, onSelect }: Props) {
  if (agents.length === 0) {
    return <p className="muted small">No agent yet.</p>;
  }

  return (
    <ul className="kb-list">
      {agents.map((agent) => {
        const toolCount = agent.tools.length + (agent.knowledge_base_ids.length > 0 ? 1 : 0);
        return (
          <li key={agent.id}>
            <button
              type="button"
              className="kb-list-item"
              aria-current={agent.id === selectedId ? "true" : undefined}
              onClick={() => onSelect(agent.id)}
            >
              <span className="kb-list-name">{agent.name}</span>
              <span className="kb-list-meta">
                {pluralize(toolCount, "tool")}, {pluralize(agent.knowledge_base_ids.length, "KB")}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
