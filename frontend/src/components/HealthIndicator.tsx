import { getHealth } from "../api/client";
import { useAsyncData } from "../hooks/useAsyncData";

// Small "API online / unreachable" indicator in the header
export default function HealthIndicator() {
  const health = useAsyncData(getHealth);

  const isUp = health.data?.status === "ok" && health.data.database === "ok";
  const label = health.isLoading ? "Checking API" : isUp ? "API online" : "API unreachable";
  const state = health.isLoading ? "checking" : isUp ? "up" : "down";

  return (
    <div className="health" data-state={state}>
      <span className="health-dot" aria-hidden="true" />
      <span>{label}</span>
      {!health.isLoading && !isUp && (
        <button type="button" className="link-button" onClick={health.reload}>
          Retry
        </button>
      )}
    </div>
  );
}
