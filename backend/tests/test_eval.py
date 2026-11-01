"""The benchmarks themselves are tested: wrong scoring would give wrong conclusions."""

import json
import re
from pathlib import Path

import pytest

from eval import metrics
from eval.metrics import RoutingRecord, routing_summary, routing_verdict
from eval.retrieval import CORPUS_DIR, configurations, evaluate_config, load_cases
from eval.routing import load_cases as load_routing_cases
from eval.routing import markdown_report, run_model

# --- The datasets are consistent -------------------------------------------------------


def test_every_retrieval_phrase_is_in_its_documents_and_nowhere_else():
    corpus = {
        p.name: metrics.normalize(p.read_text(encoding="utf-8"))
        for p in CORPUS_DIR.glob("*.md")
    }
    for case in load_cases():
        phrase = metrics.normalize(case["phrase"])
        for doc in case["docs"]:
            assert phrase in corpus[doc], f"{case['id']}: phrase missing from {doc}"
        for name, text in corpus.items():
            if name not in case["docs"]:
                assert phrase not in text, f"{case['id']}: phrase also in {name}"


def test_routing_cases_only_name_existing_tools():
    from app.services.tools.catalog import KB_TOOL_NAME, SELECTABLE_TOOLS

    known = set(SELECTABLE_TOOLS) | {KB_TOOL_NAME}
    for case in load_routing_cases():
        assert set(case["expected"]) | set(case.get("allowed", [])) <= known, case["id"]


# --- Retrieval scoring -----------------------------------------------------------------

CASE = {"docs": ["equity_kid.md"], "phrase": "Ongoing charges: 1.65%"}


def test_relevance_ignores_markdown_and_spacing():
    assert metrics.is_relevant(
        "equity_kid.md", "- **Ongoing charges:**\n  1.65% of the value", CASE
    )


def test_the_right_phrase_in_the_wrong_document_is_not_relevant():
    assert not metrics.is_relevant("bond_kid.md", "Ongoing charges: 1.65%", CASE)


def test_first_relevant_rank_is_one_based():
    hits = [
        {"filename": "bond_kid.md", "content": "Ongoing charges: 0.75%"},
        {"filename": "equity_kid.md", "content": "Ongoing charges: 1.65% per year"},
    ]
    assert metrics.first_relevant_rank(hits, CASE) == 2
    assert metrics.first_relevant_rank(hits[:1], CASE) is None


def test_recall_and_mrr():
    summary = metrics.retrieval_summary([1, 2, None, 5])
    assert summary["recall@1"] == 0.25
    assert summary["recall@3"] == 0.5
    assert summary["recall@5"] == 0.75
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)


# --- Routing scoring ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("called", "case", "verdict"),
    [
        ([], {"expected": []}, "correct"),
        (["weather"], {"expected": []}, "over_call"),
        (["calculator"], {"expected": [], "allowed": ["calculator"]}, "correct"),
        ([], {"expected": ["weather"]}, "under_call"),
        (["calculator"], {"expected": ["weather"]}, "wrong_tool"),
        (["weather"], {"expected": ["weather", "calculator"]}, "partial"),
        (["weather", "web_search"], {"expected": ["weather"]}, "extra_tool"),
        (
            ["search_knowledge_base", "calculator"],
            {"expected": ["search_knowledge_base"], "allowed": ["calculator"]},
            "correct",
        ),
        (
            ["weather", "weather"],
            {"expected": ["weather"]},
            "correct",
        ),  # two cities: fine
    ],
)
def test_routing_verdicts(called, case, verdict):
    assert routing_verdict(called, case) == verdict


def record(case_id, called, verdict, repeat=1, latency=100):
    return RoutingRecord(case_id, "x", repeat, tuple(called), verdict, latency, 50, 5)


def test_routing_summary_counts_and_consistency():
    records = [
        record("a", [], "correct", 1),
        record("a", [], "correct", 2),
        record("b", ["weather"], "correct", 1),
        record("b", [], "under_call", 2),  # changed its mind: inconsistent
        RoutingRecord("c", "x", 1, (), "error", 30, error="429"),
    ]
    s = routing_summary(records)
    assert s["answered"] == 4 and s["errors"] == 1
    assert s["accuracy"] == 0.75
    assert s["under_call"] == 1
    assert s["consistency"] == 0.5


def test_percentiles():
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([7], 95) == 7.0
    assert metrics.percentile([100, 200, 300], 50) == 200


# --- The routing benchmark runs end to end (with the mock as the model) -------------------


def test_routing_benchmark_with_the_mock():
    from app.services.mock_llm import MockChatModel

    cases = load_routing_cases()
    records = run_model(MockChatModel(), cases, log=lambda *_: None)

    assert len(records) == len(cases)
    by_id = {r.case_id: r for r in records}
    # The rules always search a linked KB: the calculator is found, plus a useless search
    assert "calculator" in by_id["t01"].called and by_id["t01"].verdict == "extra_tool"
    assert by_id["t16"].verdict == "over_call"  # the 'weather' word trap catches the rules
    assert by_id["t09"].verdict == "correct"  # a documents question: the rules get it
    report = markdown_report({"mock": records}, cases)
    assert "| mock |" in report and "## Mistakes" in report


def test_a_failing_model_is_recorded_not_fatal():
    class Down:
        model = "down"

        def complete(self, messages, tools):
            raise ConnectionError("provider down")

    records = run_model(
        Down(), load_routing_cases()[:2], schemas=[], prompt="p", log=lambda *_: None
    )
    assert [r.verdict for r in records] == ["error", "error"]
    assert "provider down" in records[0].error


# --- The retrieval benchmark, against a fake platform ----------------------------------------


class FakeAPI:
    """In memory: documents become paragraphs, search ranks them by shared words."""

    def __init__(self, fail_indexing: bool = False):
        self.fail_indexing = fail_indexing
        self.kbs: dict[int, dict] = {}
        self.deleted: list[int] = []

    def create_kb(self, payload):
        kb_id = len(self.kbs) + 1
        self.kbs[kb_id] = {
            "id": kb_id,
            "chunk_size": payload.get("chunk_size", 500),
            "chunk_overlap": payload.get("chunk_overlap", 50),
            "chunks": [],
        }
        return self.kbs[kb_id]

    def upload(self, kb_id, path: Path):
        for paragraph in path.read_text(encoding="utf-8").split("\n\n"):
            self.kbs[kb_id]["chunks"].append(
                {"filename": path.name, "content": paragraph}
            )

    def documents(self, kb_id):
        names = sorted({c["filename"] for c in self.kbs[kb_id]["chunks"]})
        status = "error" if self.fail_indexing else "ready"
        return [
            {"filename": n, "status": status, "error_message": "boom", "chunk_count": 3}
            for n in names
        ]

    def search(self, kb_id, query, top_k):
        words = set(re.findall(r"\w+", query.lower()))
        chunks = self.kbs[kb_id]["chunks"]
        scored = sorted(
            chunks,
            key=lambda c: -len(words & set(re.findall(r"\w+", c["content"].lower()))),
        )
        return {"results": scored[:top_k]}

    def delete_kb(self, kb_id):
        self.deleted.append(kb_id)


MODEL = {"name": "intfloat/multilingual-e5-small", "max_chunk_size": 500}


def test_retrieval_benchmark_end_to_end_with_a_fake_platform():
    api = FakeAPI()
    cases = load_cases()
    corpus = sorted(CORPUS_DIR.glob("*.md"))

    result = evaluate_config(
        api, MODEL, 150, cases, corpus, poll=0, log=lambda *_: None
    )

    assert result["chunk_size"] == 150 and result["chunk_overlap"] == 15
    assert result["overall"]["questions"] == len(cases)
    assert 0 < result["overall"]["recall@5"] <= 1
    assert set(result["by_lang"]) == {"en", "fr"}
    assert api.deleted == [1]  # the temporary KB is cleaned up
    json.dumps(result)  # the result can be saved as JSON


def test_the_kb_is_deleted_even_when_indexing_fails():
    api = FakeAPI(fail_indexing=True)
    with pytest.raises(RuntimeError, match="Indexing failed"):
        evaluate_config(
            api,
            MODEL,
            None,
            load_cases()[:1],
            sorted(CORPUS_DIR.glob("*.md"))[:1],
            poll=0,
        )
    assert api.deleted == [1]


def test_chunk_sizes_above_a_model_maximum_are_skipped():
    small = {"name": "minilm", "max_chunk_size": 120}
    big = {"name": "e5", "max_chunk_size": 500}
    configs = configurations([small, big], ["default", "150"])
    assert [(m["name"], size) for m, size in configs] == [
        ("minilm", None),
        ("e5", None),
        ("e5", 150),
    ]


def test_keep_leaves_the_kb_in_place():
    api = FakeAPI()
    evaluate_config(
        api,
        MODEL,
        None,
        load_cases()[:1],
        sorted(CORPUS_DIR.glob("*.md")),
        keep=True,
        poll=0,
        log=lambda *_: None,
    )
    assert api.deleted == []
