"""Retrieval benchmark: does the search return the passage that answers the question?

It goes through the REAL platform, over HTTP: create a knowledge base, upload the
corpus, wait for indexing, then ask each question to the search endpoint. So it
measures the whole chain: extraction, chunking, embeddings, pgvector (HNSW) search.

From backend/, with the stack running (docker compose up):
    uv run python -m eval.retrieval                             # all models, 2 chunk sizes
    uv run python -m eval.retrieval --models e5 --chunk-sizes default
    uv run python -m eval.retrieval --api https://other-host:8000 --keep

Each configuration creates a temporary KB named "eval …", deleted at the end
(--keep to inspect it in the UI).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from eval.metrics import first_relevant_rank, percentile, retrieval_summary

HERE = Path(__file__).parent
CASES_FILE = HERE / "retrieval_cases.json"
CORPUS_DIR = HERE / "corpus"
RESULTS_DIR = HERE / "results"
TOP_K = 5


class PlatformAPI:
    """The few endpoints the benchmark needs. Replaced by a fake in the tests."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        import httpx

        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _json(self, response):
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        return response.json() if response.content else None

    def embedding_models(self) -> list[dict]:
        return self._json(self._http.get("/embedding-models"))

    def create_kb(self, payload: dict) -> dict:
        return self._json(self._http.post("/knowledge-bases", json=payload))

    def upload(self, kb_id: int, path: Path) -> dict:
        with path.open("rb") as file:
            files = {"file": (path.name, file, "text/markdown")}
            return self._json(
                self._http.post(f"/knowledge-bases/{kb_id}/documents", files=files)
            )

    def documents(self, kb_id: int) -> list[dict]:
        return self._json(self._http.get(f"/knowledge-bases/{kb_id}/documents"))

    def search(self, kb_id: int, query: str, top_k: int) -> dict:
        return self._json(
            self._http.post(
                f"/knowledge-bases/{kb_id}/search",
                json={"query": query, "top_k": top_k},
            )
        )

    def delete_kb(self, kb_id: int) -> None:
        self._json(self._http.delete(f"/knowledge-bases/{kb_id}"))


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def wait_until_indexed(
    api, kb_id: int, expected: int, timeout: float, poll: float
) -> list[dict]:
    """Poll the documents until all are 'ready'. A document in 'error' stops the run:
    a benchmark on a half-indexed corpus would give meaningless numbers."""
    deadline = time.monotonic() + timeout
    while True:
        docs = api.documents(kb_id)
        failed = [d for d in docs if d["status"] == "error"]
        if failed:
            raise RuntimeError(
                f"Indexing failed: {failed[0]['filename']}: {failed[0]['error_message']}"
            )
        if len(docs) == expected and all(d["status"] == "ready" for d in docs):
            return docs
        if time.monotonic() > deadline:
            raise TimeoutError(f"Indexing not finished after {timeout:.0f} s")
        time.sleep(poll)


def evaluate_config(
    api,
    model: dict,
    chunk_size: int | None,
    cases: list[dict],
    corpus: list[Path],
    keep: bool = False,
    timeout: float = 600.0,
    poll: float = 1.0,
    log=print,
) -> dict:
    """One KB for one (model, chunk size): index the corpus, ask every question."""
    payload = {
        "name": f"eval {short_name(model['name'])} {chunk_size or 'default'} "
        f"{datetime.now():%H%M%S}",
        "description": "Temporary KB created by eval/retrieval.py",
        "embedding_model": model["name"],
    }
    if chunk_size is not None:
        payload["chunk_size"] = chunk_size
        payload["chunk_overlap"] = chunk_size // 10
    kb = api.create_kb(payload)
    try:
        start = time.perf_counter()
        for path in corpus:
            api.upload(kb["id"], path)
        docs = wait_until_indexed(api, kb["id"], len(corpus), timeout, poll)
        index_seconds = time.perf_counter() - start

        per_case, latencies = [], []
        for case in cases:
            t0 = time.perf_counter()
            hits = api.search(kb["id"], case["question"], TOP_K)["results"]
            latencies.append(round((time.perf_counter() - t0) * 1000))
            rank = first_relevant_rank(hits, case)
            per_case.append(
                {
                    "id": case["id"],
                    "lang": case["lang"],
                    "rank": rank,
                    "top_hit": f"{hits[0]['filename']}: {hits[0]['content'][:90]}"
                    if hits
                    else None,
                }
            )
        ranks = [c["rank"] for c in per_case]
        result = {
            "model": model["name"],
            "chunk_size": kb["chunk_size"],
            "chunk_overlap": kb["chunk_overlap"],
            "chunks": sum(d["chunk_count"] for d in docs),
            "index_seconds": round(index_seconds, 1),
            "search_p50_ms": percentile(latencies, 50),
            "search_p95_ms": percentile(latencies, 95),
            "overall": retrieval_summary(ranks),
            "by_lang": {
                lang: retrieval_summary(
                    [c["rank"] for c in per_case if c["lang"] == lang]
                )
                for lang in sorted({c["lang"] for c in per_case})
            },
            "cases": per_case,
        }
        s = result["overall"]
        log(
            f"  {short_name(model['name'])} / chunks {kb['chunk_size']}: "
            f"R@1 {s['recall@1']:.0%}  R@5 {s['recall@5']:.0%}  MRR {s['mrr']:.2f}"
        )
        return result
    finally:
        if not keep:
            api.delete_kb(kb["id"])


def configurations(
    models: list[dict], chunk_sizes: list[str]
) -> list[tuple[dict, int | None]]:
    """(model, chunk size) pairs; sizes above a model's maximum are skipped."""
    configs = []
    for model in models:
        for size in chunk_sizes:
            if size == "default":
                configs.append((model, None))
            elif int(size) <= model["max_chunk_size"]:
                configs.append((model, int(size)))
    return configs


def markdown_report(results: list[dict], cases: list[dict]) -> str:
    lines = [
        f"# Retrieval benchmark — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{len(cases)} questions ({sum(c['lang'] == 'fr' for c in cases)} in French) on "
        f"{len(list(CORPUS_DIR.glob('*.md')))} documents; top {TOP_K}; through the API.",
        "",
        "| Model | Chunk size | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | "
        "MRR (en) | MRR (fr) | Search p50 | Indexing |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s, by = r["overall"], r["by_lang"]
        lines.append(
            f"| {short_name(r['model'])} | {r['chunk_size']} / {r['chunk_overlap']} | "
            f"{r['chunks']} | {s['recall@1']:.0%} | {s['recall@3']:.0%} | "
            f"{s['recall@5']:.0%} | {s['mrr']:.2f} | "
            f"{by.get('en', {}).get('mrr', 0):.2f} | {by.get('fr', {}).get('mrr', 0):.2f} | "
            f"{r['search_p50_ms']:.0f} ms | {r['index_seconds']:.0f} s |"
        )
    lines += ["", "## Questions not answered in the top 5", ""]
    questions = {c["id"]: c for c in cases}
    for r in results:
        missed = [c for c in r["cases"] if c["rank"] is None]
        if missed:
            lines.append(f"**{short_name(r['model'])}, chunks {r['chunk_size']}**")
            lines.append("")
            for c in missed:
                q = questions[c["id"]]
                lines.append(f"- `{c['id']}` ({q['tests']}): *{q['question']}*")
                lines.append(f"  top hit: {c['top_hit']}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://localhost:8000", help="backend URL")
    parser.add_argument(
        "--models", default="all", help="'all', or comma-separated parts of model names"
    )
    parser.add_argument(
        "--chunk-sizes",
        default="default,150",
        help="comma-separated: 'default' (the model's own) and/or token counts",
    )
    parser.add_argument("--keep", action="store_true", help="don't delete the eval KBs")
    args = parser.parse_args()

    api = PlatformAPI(args.api)
    models = api.embedding_models()
    if args.models != "all":
        wanted = [w.strip().lower() for w in args.models.split(",")]
        models = [m for m in models if any(w in m["name"].lower() for w in wanted)]
    configs = configurations(models, [s.strip() for s in args.chunk_sizes.split(",")])
    if not configs:
        raise SystemExit("No configuration to run (check --models and --chunk-sizes).")

    cases = load_cases()
    corpus = sorted(CORPUS_DIR.glob("*.md"))
    print(
        f"{len(configs)} configurations, {len(cases)} questions, {len(corpus)} documents"
    )
    results = [
        evaluate_config(api, m, size, cases, corpus, keep=args.keep)
        for m, size in configs
    ]

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    report = markdown_report(results, cases)
    (RESULTS_DIR / f"retrieval-{stamp}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"retrieval-{stamp}.json").write_text(json.dumps(results, indent=2))
    print("\n" + report)
    print(f"Saved: eval/results/retrieval-{stamp}.md")


if __name__ == "__main__":
    main()
