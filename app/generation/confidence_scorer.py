

_RETRIEVAL_WEIGHT = 0.4
_CITATION_WEIGHT = 0.4
_COMPLETENESS_WEIGHT = 0.2


def score_confidence(
    chunks: list[dict],
    verified_citations: list[dict],
    answer: str,
    question: str,
) -> dict:

    retrieval = _score_retrieval_confidence(chunks)
    citation = _score_citation_coverage(verified_citations)
    completeness = _score_answer_completeness(question, answer)

    composite = (
        _RETRIEVAL_WEIGHT * retrieval
        + _CITATION_WEIGHT * citation
        + _COMPLETENESS_WEIGHT * completeness
    )

    return {
        "retrieval_confidence": round(retrieval, 3),
        "citation_coverage": round(citation, 3),
        "answer_completeness": round(completeness, 3),
        "composite": round(composite, 3),
    }


def _score_retrieval_confidence(chunks: list[dict]) -> float:

    if not chunks:
        return 0.0

    scores = []

    for chunk in chunks:

        if "relevance_score" in chunk:

            score = chunk["relevance_score"]

            # CrossEncoder scores are approximately in [0,10]
            score = max(0.0, min(score, 10.0))

            scores.append(score / 10.0)

        elif "score" in chunk:

            scores.append(
                max(
                    0.0,
                    min(
                        1.0,
                        chunk["score"],
                    ),
                )
            )

        elif "fused_score" in chunk:
            # RRF does not produce normalized scores.
            # Give a neutral confidence.
            scores.append(0.5)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _score_citation_coverage(citations: list[dict]) -> float:

    if not citations:
        return 0.0

    supported = sum(
        1
        for citation in citations
        if citation.get("supported", False)
    )

    return supported / len(citations)


def _score_answer_completeness(
    question: str,
    answer: str,
) -> float:

    if not answer:
        return 0.0

    answer = answer.strip()

    if answer.lower().startswith(
        "i don't have enough"
    ):
        return 0.0

    words = len(answer.split())

    if words >= 80:
        return 1.0

    if words >= 50:
        return 0.9

    if words >= 30:
        return 0.8

    if words >= 15:
        return 0.6

    if words >= 8:
        return 0.4

    return 0.2