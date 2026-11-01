import { listConversations } from "../api/client";
import { formatTime, pluralize } from "../format";
import { useAsyncData } from "../hooks/useAsyncData";

type Props = {
  agentId: number;
  selectedId: number | null;
  onOpen: (conversationId: number) => void;
};

// The agent's threads, most recently active first. Reloaded by the parent (key).
export default function ConversationList({ agentId, selectedId, onOpen }: Props) {
  const conversations = useAsyncData(() => listConversations(agentId));

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Conversations</h2>
        <span className="muted small">Every run is saved with its full trace</span>
      </div>
      {conversations.error && <p className="error-text small">{conversations.error}</p>}
      {conversations.data?.length === 0 && <p className="muted small">No conversation yet.</p>}
      {conversations.data && conversations.data.length > 0 && (
        <ul className="history">
          {conversations.data.map((conversation) => (
            <li key={conversation.id}>
              <button
                type="button"
                className="history-item conversation-item"
                aria-current={conversation.id === selectedId ? "true" : undefined}
                onClick={() => onOpen(conversation.id)}
              >
                <span className="history-question">{conversation.title}</span>
                <span className="history-meta">
                  {pluralize(conversation.turn_count, "turn")} ·{" "}
                  {formatTime(conversation.last_activity)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
