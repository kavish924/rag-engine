
import argparse
import json
from pathlib import Path

from app.generation.citation_verifier import verify_citations
from app.generation.generator import generate_answer
from app.retrieval.retriever import retrieve
from eval.metrics.citation_accuracy import score_citation_accuracy
from eval.metrics.correctness import score_correctness
from eval.metrics.faithfulness import score_faithfulness
from eval.metrics.retrieval_relevance import score_retrieval_relevance


def load_golden_dataset(path="eval/golden_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_case(case: dict) -> dict:
    question = case["question"]
    golden_answer = case["golden_answer"]
    expected_source_docs = case.get("source_docs", [])

    chunks = retrieve(question, mode="hybrid")

    if not chunks:
        generated_answer = "I don't have enough information to answer this question."
        verified_citations = []
    else:
        generation_result = generate_answer(question, chunks)
        generated_answer = generation_result["answer"]
        verified_citations = verify_citations(
            generation_result["raw_citations"],
            chunks,
        )

    retrieval_score = score_retrieval_relevance(
        chunks,
        expected_source_docs,
    )

    if retrieval_score < 1.0:
        print("\n----------------------------------------")
        print(f"[{case['id']}] Retrieval Debug")
        print("Question :", question)
        print("Expected :", expected_source_docs)

        retrieved = sorted(
            {
                c["metadata"].get("source_document", "UNKNOWN")
                for c in chunks
            }
        )

        print("Retrieved:")
        for doc in retrieved:
            print("   ", doc)

        print("----------------------------------------")

    scores = {
        "correctness": score_correctness(
            generated_answer,
            golden_answer,
            question,
        ),
        "faithfulness": score_faithfulness(
            generated_answer,
            chunks,
        ),
        "retrieval_relevance": retrieval_score,
        "citation_accuracy": score_citation_accuracy(
            verified_citations,
        ),
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


def run_eval_suite(dataset_path: str = "eval/golden_dataset.json") -> dict:
    """
    Returns: {
      "num_cases": int,
      "aggregate_scores": {metric_name: average_across_all_cases},
      "aggregate_by_type": {question_type: {metric_name: average}},
      "cases": [per-case detail dicts, useful for spotting specific failures],
    }
    """
    dataset = load_golden_dataset(dataset_path)
    if not dataset:
        raise ValueError(f"No test cases found in {dataset_path}. Expand the golden dataset first.")

    results = []

    print(f"\nRunning evaluation on {len(dataset)} questions...\n")

    for idx, case in enumerate(dataset, start=1):

        print(
        f"[{idx}/{len(dataset)}] {case['id']} -> {case['question']}"
    )

        try:
            result = run_single_case(case)
            results.append(result)

        except Exception as e:

            print(f"ERROR: {case['id']}")
            print(e)
            print()

    print("\nEvaluation complete.\n")

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
        "aggregate_scores": aggregate_scores,
        "aggregate_by_type": aggregate_by_type,
        "cases": results,
    }


def print_summary(report: dict) -> None:
    print(f"\n{'='*50}")
    print(f"EVAL SUMMARY — {report['num_cases']} test cases")
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

    low_scoring = sorted(
        report["cases"],
        key=lambda r: sum(r["scores"].values()),
    )[:10]

    for case in low_scoring:

        avg = sum(case["scores"].values()) / len(case["scores"])

        print(f"\n[{case['id']}]")
        print(case["question"])
        print(f"Average Score : {avg:.2f}")

    for metric, score in case["scores"].items():
        print(f"   {metric:22s}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/golden_dataset.json")
    parser.add_argument("--out", default="eval/results/latest.json")
    args = parser.parse_args()

    report = run_eval_suite(args.dataset)
    print_summary(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
    print(f"\nFull report written to {out_path}")