import type { DocumentStatus } from "../api/types";

const LABELS: Record<DocumentStatus, string> = {
  pending: "Waiting",
  processing: "Indexing",
  ready: "Ready",
  error: "Failed",
};

export default function StatusBadge({ status }: { status: DocumentStatus }) {
  return (
    <span className="status-badge" data-status={status}>
      {LABELS[status]}
    </span>
  );
}
