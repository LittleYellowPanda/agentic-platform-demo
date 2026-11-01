"""Routing benchmark: does the model call the right tool, or rightly answer directly?

Only the model's FIRST decision is measured: one LLM call per question, no tool is
executed. It isolates routing (the criterion "tool use vs direct answer") from tool
quality, and it's cheap enough for free models.

From backend/:
    uv run python -m eval.routing --mock                       # rule-based baseline
    uv run python -m eval.routing --model openrouter/free --repeats 2 --delay 3
    uv run python -m eval.routing --model qwen2.5:7b \\
        --base-url http://JETSON_IP:11434/v1 --api-key ollama  # local model

Free OpenRouter models: 20 requests/minute, 50/day without credits. 21 questions x
2 repeats = 42 calls: use --delay 3, and expect to hit the daily cap with several
models (the script records 429s as errors and keeps going).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from eval.metrics import (
    RoutingRecord,
    accuracy_by_category,
    routing_summary,
    routing_verdict,
)

HERE = Path(__file__).parent
CASES_FILE = HERE / "routing_cases.json"
RESULTS_DIR = HERE / "results"

# What the agent sees: the same as a real agent with every tool and one linked KB
DEMO_KB = SimpleNamespace(
    id=1,
    name="Aurora fund documentation",
    description=(
        "prospectuses, key information documents, annual report and sustainability "
        "policy of the Aurora funds managed by Northwind Asset Management"
    ),
)


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def tool_schemas() -> list[dict]:
    """Schemas of all tools, KB tool included. Only the schemas: nothing runs."""
    from app.services.tools.catalog import SELECTABLE_TOOLS
    from app.services.tools.knowledge_base import make_knowledge_base_tool

    kb_tool = make_knowledge_base_tool(None, [DEMO_KB], search=lambda *args: [])
    return [tool.schema() for tool in SELECTABLE_TOOLS.values()] + [kb_tool.schema()]


def system_prompt() -> str:
    """The platform's default agent prompt plus its rules, as in a real run."""
    from app.schemas.agents import DEFAULT_SYSTEM_PROMPT
    from app.services.prompts import build_system_prompt

    return build_system_prompt(DEFAULT_SYSTEM_PROMPT)


def run_model(
    llm,
    cases: list[dict],
    repeats: int = 1,
    delay: float = 0.0,
    schemas: list[dict] | None = None,
    prompt: str | None = None,
    log=print,
) -> list[RoutingRecord]:
    """Ask every question `repeats` times; record the first decision and its verdict."""
    schemas = schemas if schemas is not None else tool_schemas()
    prompt = prompt if prompt is not None else system_prompt()
    records: list[RoutingRecord] = []

    for repeat in range(1, repeats + 1):
        for case in cases:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": case["question"]},
            ]
            start = time.perf_counter()
            try:
                response = llm.complete(messages=messages, tools=schemas)
            except Exception as error:  # a benchmark must survive a 429 and keep going
                records.append(
                    RoutingRecord(
                        case_id=case["id"],
                        category=case["category"],
                        repeat=repeat,
                        called=(),
                        verdict="error",
                        latency_ms=round((time.perf_counter() - start) * 1000),
                        error=f"{type(error).__name__}: {error}"[:300],
                    )
                )
                log(f"  {case['id']} #{repeat}: ERROR {type(error).__name__}")
            else:
                called = tuple(call.name for call in response.tool_calls)
                verdict = routing_verdict(called, case)
                records.append(
                    RoutingRecord(
                        case_id=case["id"],
                        category=case["category"],
                        repeat=repeat,
                        called=called,
                        verdict=verdict,
                        latency_ms=round((time.perf_counter() - start) * 1000),
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                    )
                )
                shown = ", ".join(called) or "direct answer"
                log(f"  {case['id']} #{repeat}: {verdict:<10} ({shown})")
            if delay:
                time.sleep(delay)
    return records


def markdown_report(results: dict[str, list[RoutingRecord]], cases: list[dict]) -> str:
    """One table comparing models, one table of failures per question."""
    lines = [
        f"# Routing benchmark — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{len(cases)} questions; first decision only (no tool executed).",
        "",
        "| Model | Accuracy | Over-call | Under-call | Wrong tool | Partial | "
        "Consistency | p50 latency | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, records in results.items():
        s = routing_summary(records)
        p50 = (
            f"{s['latency_p50_ms'] / 1000:.1f} s"
            if s["latency_p50_ms"] is not None
            else "–"
        )
        lines.append(
            f"| {model} | {s['accuracy']:.0%} | {s['over_call']} | {s['under_call']} | "
            f"{s['wrong_tool']} | {s['partial']} | {s['consistency']:.0%} | {p50} | "
            f"{s['errors']} |"
        )

    lines += ["", "## Accuracy by category", ""]
    categories = sorted({c["category"] for c in cases})
    lines.append("| Model | " + " | ".join(categories) + " |")
    lines.append("|---|" + "---|" * len(categories))
    for model, records in results.items():
        by_cat = accuracy_by_category(records)
        cells = [f"{by_cat[c]:.0%}" if c in by_cat else "–" for c in categories]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines += ["", "## Mistakes", ""]
    questions = {c["id"]: c for c in cases}
    for model, records in results.items():
        wrong = [r for r in records if r.verdict not in ("correct", "error")]
        if not wrong:
            continue
        lines.append(f"**{model}**")
        lines.append("")
        for r in wrong:
            case = questions[r.case_id]
            expected = ", ".join(case["expected"]) or "direct answer"
            got = ", ".join(r.called) or "direct answer"
            lines.append(
                f"- `{r.case_id}` {r.verdict}: *{case['question']}* expected {expected}, "
                f"got {got}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_models(args) -> dict[str, object]:
    models: dict[str, object] = {}
    if args.mock:
        from app.services.mock_llm import MockChatModel

        models["mock (rules)"] = MockChatModel()
    if args.model:
        from app.core.config import settings
        from app.services.llm import OpenAIChatModel

        base_url = args.base_url or settings.llm_base_url
        api_key = args.api_key or settings.llm_api_key
        if not api_key:
            raise SystemExit("No API key: set LLM_API_KEY in .env, or pass --api-key.")
        for name in args.model:
            # no retries: a benchmark should see the provider as it is
            models[name] = OpenAIChatModel(base_url, api_key, name, max_retries=0)
    if not models:
        raise SystemExit("Nothing to evaluate: pass --mock and/or --model NAME.")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", help="model name (repeatable)")
    parser.add_argument(
        "--mock", action="store_true", help="include the rule-based mock"
    )
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (default: .env)")
    parser.add_argument("--api-key", help="API key (default: LLM_API_KEY from .env)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--delay", type=float, default=0.0, help="seconds between calls"
    )
    args = parser.parse_args()

    cases = load_cases()
    models = build_models(args)
    schemas, prompt = tool_schemas(), system_prompt()

    results: dict[str, list[RoutingRecord]] = {}
    for name, llm in models.items():
        print(f"\n== {name}")
        results[name] = run_model(llm, cases, args.repeats, args.delay, schemas, prompt)
        s = routing_summary(results[name])
        print(f"   accuracy {s['accuracy']:.0%}, errors {s['errors']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    report = markdown_report(results, cases)
    (RESULTS_DIR / f"routing-{stamp}.md").write_text(report, encoding="utf-8")
    raw = {m: [r.__dict__ for r in recs] for m, recs in results.items()}
    (RESULTS_DIR / f"routing-{stamp}.json").write_text(json.dumps(raw, indent=2))
    print("\n" + report)
    print(f"Saved: eval/results/routing-{stamp}.md")


if __name__ == "__main__":
    main()
