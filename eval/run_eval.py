import argparse
import json
import time
from pathlib import Path

from app.generation.citation_verifier import verify_citations
from app.generation.generator import generate_answer
from app.retrieval.retriever import retrieve
from eval.metrics.citation_accuracy import score_citation_accuracy
from eval.metrics.correctness import score_correctness
from eval.metrics.faithfulness import score_faithfulness
from eval.metrics.retrieval_relevance import score_retrieval_relevance

# Catch whatever rate-limit exception your provider SDK raises.
# Anthropic: anthropic.RateLimitError. OpenAI: openai.RateLimitError.
try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = ()
try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = ()

RATE_LIMIT_ERRORS = tuple(
    e for e in (AnthropicRateLimitError, OpenAIRateLimitError) if e != ()
)


def load_golden_dataset(path="eval/golden_dataset.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_checkpoint(out_path: Path) -> dict:
    """Load already-completed results so we don't re-pay for them."""
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            return {r["id"]: r for r in data.get("cases", [])}
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


def call_with_retry(fn, *args, max_retries=5, base_delay=2, **kwargs):
    """Retry with exponential backoff on rate-limit errors only.
    Other exceptions propagate immediately — we don't want to mask real bugs."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RATE_LIMIT_ERRORS as e:
            wait = base_delay * (2 ** attempt)
            print(f"    Rate limited, backing off {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"Exhausted {max_retries} retries on rate limit for {fn.__name__}")


def run_single_case(case: dict) -> dict:
    question = case["question"]
    golden_answer = case["golden_answer"]
    expected_source_docs = case.get("source_docs", [])

    chunks = call_with_retry(retrieve, question, mode="hybrid")

    if not chunks:
        generated_answer = "I don't have enough information to answer this question."
        verified_citations = []
    else:
        generation_result = call_with_retry(generate_answer, question, chunks)
        generated_answer = generation_result["answer"]
        verified_citations = call_with_retry(
            verify_citations, generation_result["raw_citations"], chunks
        )

    retrieval_score = score_retrieval_relevance(chunks, expected_source_docs)

    if retrieval_score < 1.0:
        print(f"[{case['id']}] Retrieval Debug")
        print("Question :", question)
        print("Expected :", expected_source_docs)
        retrieved = sorted(
            {c["metadata"].get("source_document", "UNKNOWN") for c in chunks}
        )
        print("Retrieved:")
        for doc in retrieved:
            print("   ", doc)

    scores = {
        "correctness": call_with_retry(
            score_correctness, generated_answer, golden_answer, question
        ),
        "faithfulness": call_with_retry(score_faithfulness, generated_answer, chunks),
        "retrieval_relevance": retrieval_score,
        "citation_accuracy": call_with_retry(score_citation_accuracy, verified_citations),
    }

    return {
        "id": case["id"],
        "type": case.get("type", "unknown"),
        "question": question,
        "golden_answer": golden_answer,
        "generated_answer": generated_answer,
        "retrieved_chunk_ids": [c["id"] for c in chunks],
        "num_citations": len(verified_citations),
        "scores": scores,
    }


def run_eval_suite(
    dataset_path: str = "eval/golden_dataset.jsonl",
    out_path: str = "eval/results/latest.jsonl",
    delay: float = 1.0,
    limit: int | None = None,
) -> dict:
    dataset = load_golden_dataset(dataset_path)
    if not dataset:
        raise ValueError(f"No test cases found in {dataset_path}. Expand the golden dataset first.")

    if limit:
        dataset = dataset[:limit]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(out)

    results = list(checkpoint.values())
    failed_ids = []

    remaining = [c for c in dataset if c["id"] not in checkpoint]
    print(f"\n{len(checkpoint)} cases already completed (loaded from checkpoint).")
    print(f"Running evaluation on {len(remaining)} remaining questions...\n")

    for idx, case in enumerate(remaining, start=1):
        print(f"[{idx}/{len(remaining)}] {case['id']} -> {case['question']}")

        try:
            result = run_single_case(case)
            results.append(result)
        except Exception as e:
            print(f"ERROR (non-rate-limit, case skipped): {case['id']}")
            print(e)
            failed_ids.append(case["id"])
            print()

        _write_partial(out, results, failed_ids)

        if idx < len(remaining):
            time.sleep(delay)

    print("\nEvaluation complete.\n")

    if not results:
        raise RuntimeError("No cases completed successfully — nothing to aggregate.")

    metric_names = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]
    aggregate_scores = {
        metric: round(sum(r["scores"][metric] for r in results) / len(results), 3)
        for metric in metric_names
    }

    aggregate_by_type: dict[str, dict[str, float]] = {}
    types = {r["type"] for r in results}
    for t in types:
        type_results = [r for r in results if r["type"] == t]
        aggregate_by_type[t] = {
            metric: round(sum(r["scores"][metric] for r in type_results) / len(type_results), 3)
            for metric in metric_names
        }
        aggregate_by_type[t]["num_cases"] = len(type_results)

    return {
        "num_cases": len(results),
        "num_failed": len(failed_ids),
        "failed_ids": failed_ids,
        "aggregate_scores": aggregate_scores,
        "aggregate_by_type": aggregate_by_type,
        "cases": results,
    }


def _write_partial(out_path: Path, results: list, failed_ids: list) -> None:
    partial_report = {
        "num_cases": len(results),
        "num_failed": len(failed_ids),
        "failed_ids": failed_ids,
        "cases": results,
    }
    out_path.write_text(
        json.dumps(partial_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_summary(report: dict) -> None:
    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY — {report['num_cases']} test cases ({report.get('num_failed', 0)} failed)")
    print(f"{'='*50}")
    print("\nOverall scores:")
    for metric, score in report["aggregate_scores"].items():
        print(f"  {metric:24s} {score:.3f}")

    print("\nBy question type:")
    for qtype, scores in report["aggregate_by_type"].items():
        n = scores.pop("num_cases")
        print(f"  {qtype} (n={n}):")
        for metric, score in scores.items():
            print(f"    {metric:22s} {score:.3f}")

    print("\nLowest Scoring Cases")
    print("-" * 60)

    low_scoring = sorted(report["cases"], key=lambda r: sum(r["scores"].values()))[:10]

    for case in low_scoring:
        avg = sum(case["scores"].values()) / len(case["scores"])
        print(f"\n[{case['id']}]")
        print(case["question"])
        print(f"Average Score : {avg:.2f}")
        for metric, score in case["scores"].items():   # now correctly inside the loop
            print(f"   {metric:22s}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/golden_dataset.jsonl")
    parser.add_argument("--out", default="eval/results/latest.jsonl")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between cases")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N cases (for cheap iteration)")
    args = parser.parse_args()

    report = run_eval_suite(args.dataset, args.out, delay=args.delay, limit=args.limit)
    print_summary(report)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report written to {out_path}")