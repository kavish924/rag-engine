

from typing import Literal
import time

from app.config import settings
from app.retrieval.dense import dense_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.sparse_bm25 import bm25_search


RetrievalMode = Literal["hybrid", "dense_only"]


def retrieve(
    query: str,
    mode: RetrievalMode = "hybrid",
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    rerank_top_n: int | None = None,
) -> list[dict]:

    dense_top_k = dense_top_k or settings.dense_top_k
    sparse_top_k = sparse_top_k or settings.sparse_top_k
    rerank_top_n = rerank_top_n or settings.rerank_top_n

    total_start = time.perf_counter()


    dense_start = time.perf_counter()

    dense_results = dense_search(
        query,
        top_k=dense_top_k,
    )

    dense_time = time.perf_counter() - dense_start

    print(f"Dense Search          : {dense_time:.3f}s")
    print(f"Dense Results         : {len(dense_results)}")


    if mode == "dense_only":

        rerank_start = time.perf_counter()

        results = rerank(
            query,
            dense_results,
            keep_top_n=rerank_top_n,
        )

        rerank_time = time.perf_counter() - rerank_start

        total_time = time.perf_counter() - total_start

        print(f"CrossEncoder Rerank  : {rerank_time:.3f}s")
        print(f"TOTAL Retrieval      : {total_time:.3f}s\n")

        return results


    sparse_start = time.perf_counter()

    sparse_results = bm25_search(
        query,
        top_k=sparse_top_k,
    )

    sparse_time = time.perf_counter() - sparse_start

    print(f"BM25 Search          : {sparse_time:.3f}s")
    print(f"BM25 Results         : {len(sparse_results)}")



    fusion_start = time.perf_counter()

    candidates = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=settings.rrf_dense_weight,
        sparse_weight=settings.rrf_sparse_weight,
    )

    fusion_time = time.perf_counter() - fusion_start

    print(f"Fusion               : {fusion_time:.3f}s")
    print(f"Fusion Candidates    : {len(candidates)}")

 

    rerank_start = time.perf_counter()

    results = rerank(
        query,
        candidates,
        keep_top_n=rerank_top_n,
    )

    rerank_time = time.perf_counter() - rerank_start


    total_time = time.perf_counter() - total_start

    print("\n========== RETRIEVAL BREAKDOWN ==========")
    print(f"Dense Search         : {dense_time:.3f}s")
    print(f"BM25 Search          : {sparse_time:.3f}s")
    print(f"Fusion               : {fusion_time:.3f}s")
    print(f"CrossEncoder Rerank  : {rerank_time:.3f}s")
    print("-----------------------------------------")
    print(f"TOTAL Retrieval      : {total_time:.3f}s")
    print("=========================================\n")

    return results