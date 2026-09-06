

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[dict]:
    fused: dict[str, dict] = {}

    for rank, result in enumerate(dense_results, start=1):
        rid = result["id"]
        fused.setdefault(rid, {
            "id": rid, "text": result["text"], "metadata": result["metadata"],
            "fused_score": 0.0, "dense_rank": None, "sparse_rank": None,
        })
        fused[rid]["dense_rank"] = rank
        fused[rid]["fused_score"] += dense_weight / (rrf_k + rank)

    for rank, result in enumerate(sparse_results, start=1):
        rid = result["id"]
        fused.setdefault(rid, {
            "id": rid, "text": result["text"], "metadata": result["metadata"],
            "fused_score": 0.0, "dense_rank": None, "sparse_rank": None,
        })
        fused[rid]["sparse_rank"] = rank
        fused[rid]["fused_score"] += sparse_weight / (rrf_k + rank)

    return sorted(fused.values(), key=lambda r: r["fused_score"], reverse=True)
