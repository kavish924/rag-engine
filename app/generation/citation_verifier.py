import os
import re
from difflib import SequenceMatcher

CITATION_MATCH_THRESHOLD = 0.90
_DEBUG = os.getenv("CITATION_DEBUG", "false").lower() == "true"

def verify_citations(raw_citations: list[dict], chunks: list[dict]) -> list[dict]:
    if not raw_citations or not chunks:
        return []

    verified_citations = []

    for citation in raw_citations:
        marker = citation.get("marker", "")
        excerpt = citation.get("excerpt", "")

        chunk_index = _extract_chunk_index(marker)

        if chunk_index is None or chunk_index >= len(chunks):
            verified_citations.append(_build_invalid_citation(marker, excerpt))
            continue

        chunk = chunks[chunk_index]
        chunk_text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        supported = _is_excerpt_supported(excerpt, chunk_text)

        if _DEBUG:
            print(f"[{marker}] supported={supported} chunk_id={chunk.get('id')}")

        verified_citations.append({
            "marker": marker,
            "chunk_id": chunk.get("id", ""),
            "source_document": metadata.get("source_document", "unknown"),
            "section_heading": metadata.get("section_heading"),
            "supported": supported,
            "excerpt": excerpt,
        })

    return verified_citations


def _extract_chunk_index(marker: str) -> int | None:
    match = re.fullmatch(r"\[(\d+)\]", marker.strip())
    if not match:
        return None
    n = int(match.group(1))
    return n - 1 if n > 0 else None


def _is_excerpt_supported(excerpt: str, chunk_text: str) -> bool:
    if not excerpt or not chunk_text:
        return False

    normalized_excerpt = _normalize(excerpt)
    normalized_chunk = _normalize(chunk_text)

    if not normalized_excerpt or not normalized_chunk:
        return False

    # Exact substring — fast path, covers the common case.
    if normalized_excerpt in normalized_chunk:
        return True

    # Near-verbatim — catches OCR artifacts, whitespace/hyphenation drift,
    # minor paraphrase, without falling back to loose token overlap.
    ratio = SequenceMatcher(None, normalized_excerpt, normalized_chunk).ratio()
    return ratio >= CITATION_MATCH_THRESHOLD


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _build_invalid_citation(marker: str, excerpt: str) -> dict:
    return {
        "marker": marker,
        "chunk_id": "",
        "source_document": "unknown",
        "section_heading": None,
        "supported": False,
        "excerpt": excerpt,
    }