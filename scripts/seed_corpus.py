
import argparse
import glob

from app.ingestion.pipeline import run_ingestion


def seed(strategy: str = "recursive_structure"):
    sample_files = [
        f for f in glob.glob("scripts/sample_corpus/**/*", recursive=True)
        if f.lower().endswith((".md", ".markdown", ".txt", ".html", ".htm", ".pdf"))
    ]
    if not sample_files:
        print("No sample documents found in scripts/sample_corpus/. Add some and re-run.")
        return

    print(f"Indexing {len(sample_files)} file(s) with strategy='{strategy}'...")
    result = run_ingestion(sample_files, strategy=strategy)
    print(f"Done. Chunks created: {result['chunks_created']}, "
          f"duplicates skipped: {result['duplicates_skipped']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="recursive_structure",
                         choices=["fixed_size", "recursive_structure", "semantic"])
    args = parser.parse_args()
    seed(strategy=args.strategy)
