
from unittest.mock import MagicMock, patch

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank


def _make_result(chunk_id: str, text: str = "some text") -> dict:
    return {"id": chunk_id, "text": text, "metadata": {"source_document": "doc.md"}}




def test_rrf_favors_items_ranked_high_in_both_lists():
    dense = [_make_result("a"), _make_result("b"), _make_result("c")]
    sparse = [_make_result("b"), _make_result("a"), _make_result("c")]

    fused = reciprocal_rank_fusion(dense, sparse)

  
    fused_ids = [r["id"] for r in fused]
    assert fused_ids[-1] == "c"
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_includes_items_only_in_one_list():
    dense = [_make_result("a"), _make_result("b")]
    sparse = [_make_result("c")]

    fused = reciprocal_rank_fusion(dense, sparse)
    fused_ids = {r["id"] for r in fused}
    assert fused_ids == {"a", "b", "c"}


def test_rrf_weighting_shifts_ranking():
    dense = [_make_result("a"), _make_result("b")]
    sparse = [_make_result("b"), _make_result("a")]

    fused = reciprocal_rank_fusion(dense, sparse, dense_weight=0.0, sparse_weight=1.0)
    assert fused[0]["id"] == "b"

    fused = reciprocal_rank_fusion(dense, sparse, dense_weight=1.0, sparse_weight=0.0)
    assert fused[0]["id"] == "a"


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([], []) == []




def test_rerank_keeps_only_top_n():
    candidates = [_make_result(str(i)) for i in range(10)]

    with patch("app.retrieval.reranker._score_candidates_llm_judge") as mock_score:
        mock_score.return_value = [float(i) for i in range(10)]  # id "9" scores highest
        result = rerank("some query", candidates, keep_top_n=3)

    assert len(result) == 3
    assert result[0]["id"] == "9"
    assert result[1]["id"] == "8"
    assert result[2]["id"] == "7"


def test_rerank_empty_candidates_returns_empty():
    assert rerank("query", [], keep_top_n=5) == []


def test_parse_score_array_handles_malformed_response():
    from app.retrieval.reranker import _parse_score_array


    assert _parse_score_array("[1, 2, 3]", expected_len=3) == [1.0, 2.0, 3.0]

    
    result = _parse_score_array("[1, 2]", expected_len=3)
    assert len(result) == 3


    assert _parse_score_array("I refuse to answer", expected_len=3) == [0.0, 0.0, 0.0]



def test_retrieve_dense_only_skips_bm25_and_fusion():
    from app.retrieval.retriever import retrieve

    with patch("app.retrieval.retriever.dense_search") as mock_dense, \
         patch("app.retrieval.retriever.bm25_search") as mock_bm25, \
         patch("app.retrieval.retriever.rerank") as mock_rerank:

        mock_dense.return_value = [_make_result("a")]
        mock_rerank.side_effect = lambda query, candidates, keep_top_n: candidates[:keep_top_n]

        retrieve("some query", mode="dense_only")

        mock_dense.assert_called_once()
        mock_bm25.assert_not_called()


def test_retrieve_hybrid_calls_both_and_fuses():
    from app.retrieval.retriever import retrieve

    with patch("app.retrieval.retriever.dense_search") as mock_dense, \
         patch("app.retrieval.retriever.bm25_search") as mock_bm25, \
         patch("app.retrieval.retriever.rerank") as mock_rerank:

        mock_dense.return_value = [_make_result("a")]
        mock_bm25.return_value = [_make_result("b")]
        mock_rerank.side_effect = lambda query, candidates, keep_top_n: candidates[:keep_top_n]

        retrieve("some query", mode="hybrid")

        mock_dense.assert_called_once()
        mock_bm25.assert_called_once()