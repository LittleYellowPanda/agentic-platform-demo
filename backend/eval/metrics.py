"""Scoring for both benchmarks. Pure functions: no network, no database."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import quantiles

# --- Retrieval ------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, Markdown bold removed, whitespace collapsed: '**Ongoing charges:**
    1.65%' and 'ongoing charges: 1.65%' compare equal, however the chunk was cut."""
    return " ".join(text.replace("*", "").lower().split())


def is_relevant(filename: str, content: str, case: dict) -> bool:
    """A hit answers the case if it comes from an expected document AND contains the
    expected phrase. The phrase makes the check independent of chunk boundaries."""
    return filename in case["docs"] and normalize(case["phrase"]) in normalize(content)


def first_relevant_rank(hits: Sequence[dict], case: dict) -> int | None:
    """1-based rank of the first relevant hit, or None if none of them is relevant."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit["filename"], hit["content"], case):
            return rank
    return None


def retrieval_summary(
    ranks: Sequence[int | None], ks: Iterable[int] = (1, 3, 5)
) -> dict:
    """Recall@k (share of questions whose answer is in the top k) and MRR (mean of
    1/rank, 0 when not found): MRR rewards putting the right passage FIRST."""
    n = len(ranks)
    if n == 0:
        return {"questions": 0, "mrr": 0.0, **{f"recall@{k}": 0.0 for k in ks}}
    summary: dict = {"questions": n}
    for k in ks:
        summary[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
    summary["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return summary


# --- Routing ----------------------------------------------------------------------

CORRECT = "correct"
OVER_CALL = "over_call"  # a tool when the answer should be direct
UNDER_CALL = "under_call"  # no tool when one was needed
WRONG_TOOL = "wrong_tool"  # tools, but none of the expected ones
PARTIAL = "partial"  # some expected tools, not all (e.g. multi-tool question)
EXTRA_TOOL = "extra_tool"  # all expected tools, plus one that wasn't needed

VERDICTS = (CORRECT, OVER_CALL, UNDER_CALL, WRONG_TOOL, PARTIAL, EXTRA_TOOL)


def routing_verdict(called: Iterable[str], case: dict) -> str:
    """Judge the model's FIRST decision: the set of tools it asked for (empty = it
    answered directly) against the expected set, plus tools that are allowed."""
    decision = set(called)
    expected = set(case.get("expected", []))
    allowed = set(case.get("allowed", []))

    if not expected:
        return CORRECT if decision <= allowed else OVER_CALL
    if not decision:
        return UNDER_CALL
    if not decision & expected:
        return WRONG_TOOL
    if not expected <= decision:
        return PARTIAL
    if not decision <= expected | allowed:
        return EXTRA_TOOL
    return CORRECT


@dataclass(frozen=True)
class RoutingRecord:
    case_id: str
    category: str
    repeat: int
    called: tuple[str, ...]
    verdict: str  # one of VERDICTS, or "error"
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def percentile(values: Sequence[float], p: int) -> float | None:
    """p-th percentile (p50, p95...). None without values; the value itself for one."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return quantiles(values, n=100, method="inclusive")[p - 1]


def routing_summary(records: Sequence[RoutingRecord]) -> dict:
    """Accuracy, error types, consistency across repeats, latency and tokens."""
    scored = [r for r in records if r.verdict != "error"]
    n = len(scored)
    summary: dict = {"answered": n, "errors": len(records) - n}
    for verdict in VERDICTS:
        summary[verdict] = sum(1 for r in scored if r.verdict == verdict)
    summary["accuracy"] = summary[CORRECT] / n if n else 0.0

    # Same question, several repeats: did the model take the same decision each time?
    decisions: dict[str, set[tuple[str, ...]]] = {}
    for r in scored:
        decisions.setdefault(r.case_id, set()).add(tuple(sorted(r.called)))
    repeated = [d for d in decisions.values()]
    summary["consistency"] = (
        sum(1 for d in repeated if len(d) == 1) / len(repeated) if repeated else 0.0
    )

    latencies = [r.latency_ms for r in scored]
    summary["latency_p50_ms"] = percentile(latencies, 50)
    summary["latency_p95_ms"] = percentile(latencies, 95)
    summary["avg_prompt_tokens"] = sum(r.prompt_tokens for r in scored) / n if n else 0
    return summary


def accuracy_by_category(records: Sequence[RoutingRecord]) -> dict[str, float]:
    by_category: dict[str, list[bool]] = {}
    for r in records:
        if r.verdict != "error":
            by_category.setdefault(r.category, []).append(r.verdict == CORRECT)
    return {cat: sum(oks) / len(oks) for cat, oks in sorted(by_category.items())}
