import { useState } from "react";

import { listKnowledgeBases } from "../api/client";
import type { KnowledgeBase } from "../api/types";
import CreateKbForm from "../components/CreateKbForm";
import KbDetail from "../components/KbDetail";
import KbList from "../components/KbList";
import { useAsyncData } from "../hooks/useAsyncData";

// The page owns the shared state: the list of KBs and which one is selected.
// Children receive data as props and report events through callbacks.
export default function KnowledgeBasesPage() {
  const knowledgeBases = useAsyncData(listKnowledgeBases);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  // Derived from the list, not stored: always in sync with the latest data
  const selected = knowledgeBases.data?.find((kb) => kb.id === selectedId) ?? null;

  function handleSelect(id: number) {
    setSelectedId(id);
    setIsCreating(false);
  }

  function handleCreated(kb: KnowledgeBase) {
    setIsCreating(false);
    setSelectedId(kb.id);
    knowledgeBases.reload();
  }

  function handleDeleted() {
    setSelectedId(null);
    knowledgeBases.reload();
  }

  return (
    <div className="kb-page">
      <aside className="sidebar" aria-label="Knowledge bases">
        <div className="sidebar-header">
          <h2>Knowledge bases</h2>
          <button type="button" className="primary small-button" onClick={() => setIsCreating(true)}>
            New
          </button>
        </div>
        {knowledgeBases.error && <p className="error-text small">{knowledgeBases.error}</p>}
        {knowledgeBases.data && (
          <KbList
            knowledgeBases={knowledgeBases.data}
            selectedId={isCreating ? null : selectedId}
            onSelect={handleSelect}
          />
        )}
      </aside>

      <main className="content">
        {isCreating ? (
          <CreateKbForm onCreated={handleCreated} onCancel={() => setIsCreating(false)} />
        ) : selected ? (
          // key: switching KB remounts the detail, so its panels start fresh
          // instead of briefly showing the previous KB's documents and results
          <KbDetail
            key={selected.id}
            kb={selected}
            onChanged={knowledgeBases.reload}
            onDeleted={handleDeleted}
          />
        ) : (
          <div className="empty-state">
            <h1>Knowledge bases</h1>
            <p className="muted">
              A knowledge base holds documents an agent can search. Create one, upload text,
              Markdown or PDF files, then test which passages come back for a question.
            </p>
            <button type="button" className="primary" onClick={() => setIsCreating(true)}>
              Create a knowledge base
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
