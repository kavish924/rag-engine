
from sentence_transformers import CrossEncoder
import time

_MODEL = None


def _get_model():
    global _MODEL

    if _MODEL is None:
        print("\nLoading CrossEncoder model...")

        start = time.perf_counter()

        _MODEL = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        print(
            f"Loaded in {time.perf_counter() - start:.2f} sec\n"
        )

    return _MODEL


def rerank(
    query: str,
    candidates: list[dict],
    keep_top_n: int = 5,
) -> list[dict]:

    if not candidates:
        return []

    model = _get_model()

    sentence_pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]
    start = time.perf_counter()

    scores = model.predict(sentence_pairs)

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 50)
    print("CROSS-ENCODER RERANKER")
    print("=" * 50)
    print(f"Candidates      : {len(candidates)}")
    print(f"Time            : {elapsed:.2f} sec")
    print("=" * 50 + "\n")

    for candidate, score in zip(candidates, scores):
        candidate["relevance_score"] = float(score)

    candidates.sort(
        key=lambda x: x["relevance_score"],
        reverse=True,
    )

    return candidates[:keep_top_n]