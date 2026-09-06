def score_citation_accuracy(verified_citations: list[dict]) -> float:
    if not verified_citations:
        return 0.0
    supported = sum(1 for c in verified_citations if c.get("supported"))
    return supported / len(verified_citations)