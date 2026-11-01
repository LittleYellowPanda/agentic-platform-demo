import { useState } from "react";

import { deleteKnowledgeBase, errorMessage } from "../api/client";
import type { KnowledgeBase } from "../api/types";
import { shortModelName } from "../format";
import DocumentsPanel from "./DocumentsPanel";
import SearchPanel from "./SearchPanel";

type Props = {
  kb: KnowledgeBase;
  onChanged: () => void;
  onDeleted: () => void;
};

export default function KbDetail({ kb, onChanged, onDeleted }: Props) {
  // Two-step delete: the first click asks for confirmation, the second deletes.
  // Deleting a KB removes all its documents and chunks, so no accidental click.
  const [isConfirming, setIsConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleDelete() {
    try {
      await deleteKnowledgeBase(kb.id);
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
          <h1>{kb.name}</h1>
          {kb.description && <p className="kb-description">{kb.description}</p>}
          <dl className="kb-settings">
            <div>
              <dt>Model</dt>
              <dd title={kb.embedding_model}>{shortModelName(kb.embedding_model)}</dd>
            </div>
            <div>
              <dt>Chunk size</dt>
              <dd>{kb.chunk_size} tokens</dd>
            </div>
            <div>
              <dt>Overlap</dt>
              <dd>{kb.chunk_overlap} tokens</dd>
            </div>
          </dl>
        </div>

        <div className="kb-header-actions">
          {isConfirming ? (
            <>
              <span className="small">Delete this knowledge base and all its documents?</span>
              <button type="button" className="danger" onClick={handleDelete}>
                Delete
              </button>
              <button type="button" onClick={() => setIsConfirming(false)}>
                Keep
              </button>
            </>
          ) : (
            <button type="button" onClick={() => setIsConfirming(true)}>
              Delete knowledge base
            </button>
          )}
        </div>
      </header>

      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}

      <DocumentsPanel kbId={kb.id} onChanged={onChanged} />
      <SearchPanel kbId={kb.id} />
    </div>
  );
}
