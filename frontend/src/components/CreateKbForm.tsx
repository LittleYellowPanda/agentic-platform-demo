import { useState, type FormEvent } from "react";

import { createKnowledgeBase, errorMessage, listEmbeddingModels } from "../api/client";
import type { KnowledgeBase, KnowledgeBaseCreate } from "../api/types";
import { useAsyncData } from "../hooks/useAsyncData";

type Props = {
  onCreated: (kb: KnowledgeBase) => void;
  onCancel: () => void;
};

export default function CreateKbForm({ onCreated, onCancel }: Props) {
  const models = useAsyncData(listEmbeddingModels);

  // Controlled inputs: each field displays a piece of state, and every keystroke
  // updates it. Chunk settings are strings: "" means "use the model's default".
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [modelName, setModelName] = useState(""); // "" until the user picks one
  const [chunkSize, setChunkSize] = useState("");
  const [chunkOverlap, setChunkOverlap] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Derived values, recomputed at each render rather than stored in state
  const defaultModel = models.data?.find((m) => m.is_default);
  const selectedModel = models.data?.find((m) => m.name === modelName) ?? defaultModel;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); // stop the browser from reloading the page
    setError(null);
    setIsSubmitting(true);

    const payload: KnowledgeBaseCreate = { name };
    if (description.trim()) payload.description = description.trim();
    if (selectedModel) payload.embedding_model = selectedModel.name;
    if (chunkSize) payload.chunk_size = Number(chunkSize);
    if (chunkOverlap) payload.chunk_overlap = Number(chunkOverlap);

    try {
      const kb = await createKnowledgeBase(payload);
      onCreated(kb);
    } catch (err) {
      // The backend is the judge: its validation messages are shown as is
      setError(errorMessage(err));
      setIsSubmitting(false);
    }
  }

  return (
    <form className="panel create-form" onSubmit={handleSubmit}>
      <h2>New knowledge base</h2>

      <label className="field">
        <span>Name</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Fund prospectuses"
          maxLength={255}
          required
          autoFocus
        />
      </label>

      <label className="field">
        <span>Description</span>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What these documents are about"
          rows={2}
        />
      </label>

      <label className="field">
        <span>Embedding model</span>
        <select
          value={selectedModel?.name ?? ""}
          onChange={(e) => setModelName(e.target.value)}
          disabled={!models.data}
        >
          {models.data?.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name}
            </option>
          ))}
        </select>
        {selectedModel && <small className="muted">{selectedModel.description}</small>}
        {models.error && <small className="error-text">{models.error}</small>}
      </label>

      <fieldset className="field-row">
        <legend>Chunking</legend>
        <label className="field">
          <span>Chunk size (tokens)</span>
          <input
            type="number"
            inputMode="numeric"
            value={chunkSize}
            onChange={(e) => setChunkSize(e.target.value)}
            placeholder={selectedModel ? `${selectedModel.max_chunk_size} (default)` : ""}
          />
        </label>
        <label className="field">
          <span>Overlap (tokens)</span>
          <input
            type="number"
            inputMode="numeric"
            value={chunkOverlap}
            onChange={(e) => setChunkOverlap(e.target.value)}
            placeholder="10% (default)"
          />
        </label>
      </fieldset>
      <p className="muted small">
        These settings can't be changed once documents are indexed with them.
        {selectedModel && ` This model accepts chunks of up to ${selectedModel.max_chunk_size} tokens.`}
      </p>

      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}

      <div className="form-actions">
        <button type="submit" className="primary" disabled={isSubmitting || !name.trim()}>
          {isSubmitting ? "Creating…" : "Create knowledge base"}
        </button>
        <button type="button" onClick={onCancel} disabled={isSubmitting}>
          Cancel
        </button>
      </div>
    </form>
  );
}
