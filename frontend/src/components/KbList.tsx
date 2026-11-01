import type { KnowledgeBase } from "../api/types";
import { pluralize, shortModelName } from "../format";

type Props = {
  knowledgeBases: KnowledgeBase[];
  selectedId: number | null;
  onSelect: (id: number) => void;
};

// The list in the sidebar. It owns no state: the page tells it which KB is
// selected, and it reports clicks back up (data down, events up).
export default function KbList({ knowledgeBases, selectedId, onSelect }: Props) {
  if (knowledgeBases.length === 0) {
    return <p className="muted small">No knowledge base yet.</p>;
  }

  return (
    <ul className="kb-list">
      {knowledgeBases.map((kb) => (
        <li key={kb.id}>
          <button
            type="button"
            className="kb-list-item"
            aria-current={kb.id === selectedId ? "true" : undefined}
            onClick={() => onSelect(kb.id)}
          >
            <span className="kb-list-name">{kb.name}</span>
            <span className="kb-list-meta">
              {pluralize(kb.document_count, "document")}, {shortModelName(kb.embedding_model)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
