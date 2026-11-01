import { useEffect, useState } from "react";

import HealthIndicator from "./components/HealthIndicator";
import LlmBanner from "./components/LlmBanner";
import AgentsPage from "./pages/AgentsPage";
import KnowledgeBasesPage from "./pages/KnowledgeBasesPage";

type Page = "knowledge-bases" | "agents";

// The page lives in the URL hash (#/agents): a refresh or a shared link keeps it,
// without a router library for two pages.
function pageFromHash(): Page {
  return window.location.hash === "#/agents" ? "agents" : "knowledge-bases";
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash);

  useEffect(() => {
    const onHashChange = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="brand">Agent Platform</span>
          <nav className="app-nav" aria-label="Main">
            <a href="#/knowledge-bases" aria-current={page === "knowledge-bases" ? "page" : undefined}>
              Knowledge bases
            </a>
            <a href="#/agents" aria-current={page === "agents" ? "page" : undefined}>
              Agents
            </a>
          </nav>
        </div>
        <HealthIndicator />
      </header>
      <LlmBanner />
      {page === "agents" ? <AgentsPage /> : <KnowledgeBasesPage />}
    </div>
  );
}
