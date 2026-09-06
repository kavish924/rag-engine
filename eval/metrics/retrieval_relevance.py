from pathlib import Path


def _normalize(doc: str | None) -> str:
    
    if not doc:
        return ""

    return Path(doc).name.lower().strip()


def score_retrieval_relevance(
    retrieved_chunks: list[dict],
    expected_source_docs: list[str],
) -> float:
   
    if not expected_source_docs:
        return 1.0

    
    retrieved_docs = {
        _normalize(
            chunk.get("metadata", {}).get("source_document")
        )
        for chunk in retrieved_chunks
    }


    expected_docs = {
        _normalize(doc)
        for doc in expected_source_docs
    }

    if not expected_docs:
        return 1.0

    hits = retrieved_docs.intersection(expected_docs)

    return round(len(hits) / len(expected_docs), 3)