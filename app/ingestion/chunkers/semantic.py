import re
from typing import Callable

import numpy as np

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def chunk_semantic(
    text: str,
    similarity_threshold: float = 0.75,
    max_chunk_size: int = 1500,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    if embed_fn is None:
        from app.ingestion.embeddings import embed_texts as embed_fn  

    embeddings = embed_fn(sentences)

    chunks: list[str] = []
    current_sentences = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = _cosine_similarity(embeddings[i - 1], embeddings[i])
        candidate_len = len(" ".join(current_sentences)) + len(sentences[i])

        topic_shifted = similarity < similarity_threshold
        too_long = candidate_len > max_chunk_size

        if topic_shifted or too_long:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [c for c in chunks if c]
