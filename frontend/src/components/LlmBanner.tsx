import { getLlmStatus } from "../api/client";
import { useAsyncData } from "../hooks/useAsyncData";

// Shown only in mock mode: nobody should mistake rule-based answers for a real model
export default function LlmBanner() {
  const status = useAsyncData(getLlmStatus);

  if (status.data?.mode !== "mock") return null;
  return (
    <div className="banner" role="status">
      <strong>Mock mode.</strong> No LLM key is configured: agents answer with simple rules
      and real tools. Set <code>LLM_API_KEY</code> in <code>.env</code> for a real model.
    </div>
  );
}
