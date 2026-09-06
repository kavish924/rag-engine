
import argparse
import glob
import json
import shutil
from pathlib import Path

from app.generation.citation_verifier import verify_citations
from app.generation.generator import generate_answer
from app.ingestion.pipeline import ChunkingStrategy, run_ingestion
from app.retrieval.retriever import retrieve
from app.storage.bm25_store import BM25Store
from app.storage.vector_store import VectorStore
from eval.metrics.citation_accuracy import score_citation_accuracy
from eval.metrics.correctness import score_correctness
from eval.metrics.faithfulness import score_faithfulness
from eval.metrics.retrieval_relevance import score_retrieval_relevance
from eval.run_eval import load_golden_dataset

STRATEGIES: list[ChunkingStrategy] = ["fixed_size", "recursive_structure", "semantic"]
_COMPARISON_DATA_ROOT = "./eval/results/chunking_comparison_data"


def _get_corpus_files(corpus_dir: str) -> list[str]:
    return [
        f for f in glob.glob(f"{corpus_dir}/**/*", recursive=True)
        if f.lower().endswith((".md", ".markdown", ".txt", ".html", ".htm", ".pdf"))
    ]


def _build_isolated_stores(strategy: str) -> tuple[VectorStore, BM25Store]:
    """Each strategy gets its own persistent dir so runs don't leak into each other."""
    persist_dir = f"{_COMPARISON_DATA_ROOT}/{strategy}"
    shutil.rmtree(persist_dir, ignore_errors=True)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    vector_store = VectorStore(
        collection_name=f"rag_chunks_{strategy}",
        persist_directory=persist_dir,
    )
    bm25_store = BM25Store(persist_path=f"{persist_dir}/bm25_index.pkl")
    return vector_store, bm25_store


def _run_case_against_stores(case: dict, vector_store: VectorStore, bm25_store: BM25Store) -> dict:
    """
    Same logic as eval.run_eval.run_single_case, but retrieves against a
    specific (strategy-isolated) pair of stores instead of the global default.
    """
    question = case["question"]
    golden_answer = case["golden_answer"]
    expected_source_docs = case.get("source_docs", [])

    from app.retrieval import dense as dense_module
    from app.retrieval import sparse_bm25 as sparse_module

    dense_module._vector_store = vector_store
    sparse_module._bm25_store = bm25_store

    chunks = retrieve(question, mode="hybrid")

    if not chunks:
        generated_answer = "I don't have enough information to answer this question."
        verified_citations = []
    else:
        generation_result = generate_answer(question, chunks)
        generated_answer = generation_result["answer"]
        verified_citations = verify_citations(generation_result["raw_citations"], chunks)

    return {
        "correctness": score_correctness(generated_answer, golden_answer, question),
        "faithfulness": score_faithfulness(generated_answer, chunks),
        "retrieval_relevance": score_retrieval_relevance(chunks, expected_source_docs),
        "citation_accuracy": score_citation_accuracy(verified_citations),
    }


def compare_chunking_strategies(
    corpus_dir: str = "scripts/sample_corpus",
    dataset_path: str = "eval/golden_dataset.jsonl",
) -> dict:
    corpus_files = _get_corpus_files(corpus_dir)
    if not corpus_files:
        raise ValueError(f"No documents found in {corpus_dir}. Add sample documents first.")

    golden_dataset = load_golden_dataset(dataset_path)
    if not golden_dataset:
        raise ValueError(f"No test cases found in {dataset_path}. Expand the golden dataset first.")

    metric_names = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]
    report: dict[str, dict] = {}

    for strategy in STRATEGIES:
        print(f"\n--- Ingesting corpus with strategy='{strategy}' ---")
        vector_store, bm25_store = _build_isolated_stores(strategy)
        ingestion_result = run_ingestion(
            corpus_files, strategy=strategy, vector_store=vector_store, bm25_store=bm25_store
        )
        print(f"  chunks created: {ingestion_result['chunks_created']}")

        print(f"--- Running eval suite for strategy='{strategy}' ---")
        case_scores = [
            _run_case_against_stores(case, vector_store, bm25_store)
            for case in golden_dataset
        ]

        report[strategy] = {
            metric: round(sum(cs[metric] for cs in case_scores) / len(case_scores), 3)
            for metric in metric_names
        }
        report[strategy]["num_chunks_created"] = ingestion_result["chunks_created"]

    report["winner_by_metric"] = {
        metric: max(STRATEGIES, key=lambda s: report[s][metric])
        for metric in metric_names
    }

    return report


def print_comparison_table(report: dict) -> None:
    metric_names = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]

    print(f"\n{'='*70}")
    print("CHUNKING STRATEGY COMPARISON")
    print(f"{'='*70}")
    header = f"{'Metric':<24}" + "".join(f"{s:<22}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for metric in metric_names:
        row = f"{metric:<24}"
        for strategy in STRATEGIES:
            row += f"{report[strategy][metric]:<22}"
        print(row)

    print(f"\n{'Chunks created':<24}" + "".join(f"{report[s]['num_chunks_created']:<22}" for s in STRATEGIES))

    print("\nWinner by metric:")
    for metric, winner in report["winner_by_metric"].items():
        print(f"  {metric:24s} -> {winner}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="scripts/sample_corpus")
    parser.add_argument("--dataset", default="eval/golden_dataset.jsonl")
    parser.add_argument("--out", default="eval/results/chunking_comparison.json")
    args = parser.parse_args()

    report = compare_chunking_strategies(corpus_dir=args.corpus_dir, dataset_path=args.dataset)
    print_comparison_table(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report written to {out_path}")