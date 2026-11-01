import { useState, type FormEvent } from "react";

import { errorMessage, searchKnowledgeBase } from "../api/client";
import type { SearchResponse } from "../api/types";

const TOP_K_OPTIONS = [3, 5, 10, 20];

// Lets you type a question and inspect which chunks the retrieval returns,
// with their cosine similarity: what an agent would receive as context.
export default function SearchPanel({ kbId }: { kbId: number }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSearching(true);
    setError(null);
    try {
      setResponse(await searchKnowledgeBase(kbId, query, topK));
    } catch (err) {
      setResponse(null); // don't leave old results under a new error
      setError(errorMessage(err));
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="search-title">
      <div className="panel-header">
        <h2 id="search-title">Test the retrieval</h2>
      </div>

      <form className="search-form" onSubmit={handleSubmit}>
        <label className="visually-hidden" htmlFor="search-query">
          Question
        </label>
        <input
          id="search-query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about these documents"
          maxLength={2000}
        />
        <label className="visually-hidden" htmlFor="search-top-k">
          Number of chunks
        </label>
        <select id="search-top-k" value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
          {TOP_K_OPTIONS.map((k) => (
            <option key={k} value={k}>
              Top {k}
            </option>
          ))}
        </select>
        <button type="submit" className="primary" disabled={isSearching || !query.trim()}>
          {isSearching ? "Searching…" : "Search"}
        </button>
      </form>

      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}

      {response && response.results.length === 0 && (
        <p className="muted">
          No chunks found. Upload a document and wait until it's ready, then search again.
        </p>
      )}

      {response && response.results.length > 0 && (
        <>
          <ol className="results">
            {response.results.map((hit, position) => (
              <li key={hit.chunk_id} className="result">
                <div className="result-rank" aria-hidden="true">
                  {position + 1}
                </div>
                <div className="result-body">
                  <div className="result-head">
                    <span className="filename">{hit.filename}</span>
                    <span className="muted small">chunk {hit.chunk_index + 1}</span>
                  </div>
                  <div
                    className="meter"
                    role="meter"
                    aria-valuemin={0}
                    aria-valuemax={1}
                    aria-valuenow={hit.score}
                    aria-label="Cosine similarity"
                  >
                    <span className="meter-fill" style={{ width: `${Math.max(0, hit.score) * 100}%` }} />
                  </div>
                  <span className="score">{hit.score.toFixed(3)}</span>
                  <p className="result-content">{hit.content}</p>
                </div>
              </li>
            ))}
          </ol>
          <p className="muted small">
            Cosine similarity between the question and each chunk: 1 means the same meaning.
            Compare scores within one search; their absolute level depends on the model.
          </p>
        </>
      )}
    </section>
  );
}
