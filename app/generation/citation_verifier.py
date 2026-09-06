
import re

CITATION_OVERLAP_THRESHOLD = 0.35


def verify_citations(
    raw_citations: list[dict],
    chunks: list[dict],
) -> list[dict]:
    if not raw_citations:
        return []

    if not chunks:
        return []

    verified_citations = []

    for citation in raw_citations:

        marker = citation.get("marker", "")
        excerpt = citation.get("excerpt", "")

        chunk_index = _extract_chunk_index(marker)

        # Invalid citation marker
        if chunk_index is None:
            verified_citations.append(
                _build_invalid_citation(
                    marker=marker,
                    excerpt=excerpt,
                )
            )
            continue

        # Citation points outside retrieved chunk list
        if chunk_index >= len(chunks):
            verified_citations.append(
                _build_invalid_citation(
                    marker=marker,
                    excerpt=excerpt,
                )
            )
            continue

        chunk = chunks[chunk_index]

        print("\n==============================")
        print("Marker :", marker)
        print("Excerpt:")
        print(excerpt)

        print("\nChunk:")
        print(chunk["text"][:500])

        print("==============================")

        chunk_text = chunk.get("text", "")

        metadata = chunk.get("metadata", {})

        supported = _is_excerpt_supported(
            excerpt=excerpt,
            chunk_text=chunk_text,
        )
        print("\n" + "-" * 50)
        print(f"Marker     : {marker}")
        print(f"Excerpt    : {excerpt}")
        print(f"Supported  : {supported}")
        print(f"Chunk ID   : {chunk.get('id')}")
        print(f"Source Doc : {metadata.get('source_document')}")
        print("-" * 50)

        verified_citations.append(
            {
                "marker": marker,
                "chunk_id": chunk.get("id", ""),
                "source_document": metadata.get(
                    "source_document",
                    "unknown",
                ),
                "section_heading": metadata.get(
                    "section_heading"
                ),
                "supported": supported,
                "excerpt": excerpt,
            }
        )

    return verified_citations


def _extract_chunk_index(marker: str) -> int | None:

    match = re.fullmatch(
        r"\[(\d+)\]",
        marker.strip(),
    )

    if not match:
        return None

    citation_number = int(match.group(1))

    if citation_number <= 0:
        return None

    return citation_number - 1


def _is_excerpt_supported(
    excerpt: str,
    chunk_text: str,
) -> bool:

    if not excerpt:
        return False

    if not chunk_text:
        return False

    normalized_excerpt = _normalize_text(excerpt)

    normalized_chunk = _normalize_text(chunk_text)

    if not normalized_excerpt:
        return False

    if not normalized_chunk:
        return False
    
    if normalized_excerpt in normalized_chunk:
        return True

    excerpt_tokens = set(
        _tokenize(normalized_excerpt)
    )

    chunk_tokens = set(
        _tokenize(normalized_chunk)
    )

    if not excerpt_tokens:
        return False

    overlap = excerpt_tokens & chunk_tokens

    overlap_score = (
        len(overlap)
        / len(excerpt_tokens)
    )

    return (
        overlap_score
        >= CITATION_OVERLAP_THRESHOLD
    )


def _normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    return text.strip()


def _tokenize(text: str) -> list[str]:


    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "and",
        "or",
        "that",
        "this",
        "it",
        "as",
        "by",
        "from",
    }

    tokens = text.split()

    return [
        token
        for token in tokens
        if token not in stop_words
        and len(token) > 1
    ]


def _build_invalid_citation(
    marker: str,
    excerpt: str,
) -> dict:

    return {
        "marker": marker,
        "chunk_id": "",
        "source_document": "unknown",
        "section_heading": None,
        "supported": False,
        "excerpt": excerpt,
    }