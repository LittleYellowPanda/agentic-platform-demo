import { useEffect, useState, type DragEvent } from "react";

import { deleteDocument, errorMessage, listDocuments, uploadDocument } from "../api/client";
import type { KbDocument } from "../api/types";
import { useAsyncData } from "../hooks/useAsyncData";
import { pluralize } from "../format";
import StatusBadge from "./StatusBadge";

const POLL_INTERVAL_MS = 1500;
const ACCEPTED_TYPES = ".txt,.md,.pdf";

type Props = {
  kbId: number;
  onChanged: () => void; // tells the page the document count changed
};

function isInProgress(doc: KbDocument): boolean {
  return doc.status === "pending" || doc.status === "processing";
}

export default function DocumentsPanel({ kbId, onChanged }: Props) {
  const documents = useAsyncData(() => listDocuments(kbId));
  const [uploadErrors, setUploadErrors] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // Polling: while at least one document is being indexed, reload the list
  // regularly. The server indexes in the background; it can't notify us, so we ask.
  const isIndexing = documents.data?.some(isInProgress) ?? false;
  const { reload } = documents;

  useEffect(() => {
    if (!isIndexing) return; // nothing in progress: no timer at all
    const timer = setInterval(reload, POLL_INTERVAL_MS);
    // Cleanup: runs when indexing ends or the component disappears. Without it,
    // timers would keep running, and pile up at each new upload.
    return () => clearInterval(timer);
  }, [isIndexing, reload]);

  async function uploadFiles(files: File[]) {
    if (files.length === 0) return;
    setIsUploading(true);
    setUploadErrors([]);

    const errors: string[] = [];
    // One by one: simple, and each file gets its own error message if it fails
    for (const file of files) {
      try {
        await uploadDocument(kbId, file);
      } catch (err) {
        errors.push(`${file.name}: ${errorMessage(err)}`);
      }
    }

    setUploadErrors(errors);
    setIsUploading(false);
    reload(); // the new documents appear as "Waiting", which starts the polling
    onChanged();
  }

  async function handleDelete(doc: KbDocument) {
    try {
      await deleteDocument(kbId, doc.id);
    } catch (err) {
      setUploadErrors([`${doc.filename}: ${errorMessage(err)}`]);
    }
    reload();
    onChanged();
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault(); // otherwise the browser opens the file
    setIsDragging(false);
    uploadFiles(Array.from(event.dataTransfer.files));
  }

  const readyCount = documents.data?.filter((d) => d.status === "ready").length ?? 0;

  return (
    <section className="panel" aria-labelledby="documents-title">
      <div className="panel-header">
        <h2 id="documents-title">Documents</h2>
        {documents.data && documents.data.length > 0 && (
          <span className="muted small">
            {readyCount} of {pluralize(documents.data.length, "document")} ready
          </span>
        )}
      </div>

      <label
        className="dropzone"
        data-dragging={isDragging || undefined}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept={ACCEPTED_TYPES}
          multiple
          className="visually-hidden"
          disabled={isUploading}
          onChange={(e) => {
            uploadFiles(Array.from(e.target.files ?? []));
            e.target.value = ""; // allow choosing the same file again
          }}
        />
        <span className="dropzone-title">
          {isUploading ? "Uploading…" : "Drop files here, or click to choose"}
        </span>
        <span className="muted small">Text, Markdown or PDF with selectable text, up to 20 MB</span>
      </label>

      {uploadErrors.length > 0 && (
        <ul className="error-list" role="alert">
          {uploadErrors.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      )}

      {documents.error && <p className="error-text">{documents.error}</p>}
      {documents.isLoading && <p className="muted small">Loading documents…</p>}

      {documents.data && documents.data.length > 0 && (
        <table className="documents-table">
          <thead>
            <tr>
              <th scope="col">File</th>
              <th scope="col">Status</th>
              <th scope="col" className="numeric">
                Chunks
              </th>
              <th scope="col">
                <span className="visually-hidden">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {documents.data.map((doc) => (
              <tr key={doc.id}>
                <td>
                  <div className="file-cell">
                    <span className="filename">{doc.filename}</span>
                    {doc.error_message && (
                      <span className="error-text small">{doc.error_message}</span>
                    )}
                  </div>
                </td>
                <td>
                  <StatusBadge status={doc.status} />
                </td>
                <td className="numeric">{doc.status === "ready" ? doc.chunk_count : ""}</td>
                <td className="row-actions">
                  <button
                    type="button"
                    className="link-button"
                    onClick={() => handleDelete(doc)}
                    aria-label={`Delete ${doc.filename}`}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
