
from unittest.mock import patch

from app.generation.confidence_scorer import score_confidence
from app.generation.generator import _parse_citations
from app.generation.prompts import build_context_blocks




def test_build_context_blocks_numbers_chunks_in_order():
    chunks = [
        {"text": "First chunk text.", "metadata": {"source_document": "a.md", "section_heading": "Intro"}},
        {"text": "Second chunk text.", "metadata": {"source_document": "b.md"}},
    ]
    result = build_context_blocks(chunks)

    assert "[1]" in result
    assert "[2]" in result
    assert "First chunk text." in result
    assert "a.md" in result
    assert "Intro" in result
 
    assert "None" not in result


def test_build_context_blocks_empty_list():
    assert build_context_blocks([]) == "(no context retrieved)"




def test_parse_citations_extracts_valid_markers():
    chunks = [{"id": "c1", "text": "chunk one"}, {"id": "c2", "text": "chunk two"}]
    answer = "The rate limit is 100 requests per minute [1]. Errors return a 401 code [2]."

    citations = _parse_citations(answer, chunks)

    assert len(citations) == 2
    assert citations[0]["marker"] == "[1]"
    assert citations[0]["chunk_index"] == 1
    assert citations[0]["chunk"]["id"] == "c1"
    assert "rate limit" in citations[0]["claim_text"].lower()


def test_parse_citations_drops_out_of_range_references():
    chunks = [{"id": "c1", "text": "chunk one"}]
    answer = "This claims something that doesn't exist [5]."

    citations = _parse_citations(answer, chunks)

    assert citations == []  


def test_parse_citations_handles_multiple_markers_in_one_sentence():
    chunks = [{"id": "c1", "text": "a"}, {"id": "c2", "text": "b"}]
    answer = "This is supported by two sources [1][2]."

    citations = _parse_citations(answer, chunks)

    chunk_indices = {c["chunk_index"] for c in citations}
    assert chunk_indices == {1, 2}




def test_verify_citations_maps_judge_verdicts_correctly():
    from app.generation.citation_verifier import verify_citations

    raw_citations = [
        {
            "marker": "[1]",
            "chunk_index": 1,
            "chunk": {"id": "c1", "text": "The rate limit is 100 req/min.", "metadata": {"source_document": "a.md"}},
            "claim_text": "The rate limit is 100 requests per minute.",
        },
        {
            "marker": "[2]",
            "chunk_index": 2,
            "chunk": {"id": "c2", "text": "Unrelated text about pricing.", "metadata": {"source_document": "b.md"}},
            "claim_text": "The API costs $10/month.",
        },
    ]

    with patch("app.generation.citation_verifier._judge_citations_batch") as mock_judge:
        mock_judge.return_value = [True, False]
        result = verify_citations(raw_citations, chunks=[])

    assert result[0]["supported"] is True
    assert result[0]["source_document"] == "a.md"
    assert result[1]["supported"] is False


def test_verify_citations_empty_input():
    from app.generation.citation_verifier import verify_citations
    assert verify_citations([], chunks=[]) == []


def test_parse_verdict_array_fails_closed_on_malformed_response():
    from app.generation.citation_verifier import _parse_verdict_array

    assert _parse_verdict_array("[true, false]", expected_len=2) == [True, False]
  
    assert _parse_verdict_array("I cannot determine this", expected_len=2) == [False, False]




def test_score_confidence_all_high_signals_gives_high_composite():
    chunks = [{"relevance_score": 9.0}, {"relevance_score": 8.5}]
    verified_citations = [{"supported": True}, {"supported": True}]

    with patch("app.generation.confidence_scorer._score_answer_completeness", return_value=0.9):
        result = score_confidence(chunks, verified_citations, answer="full answer", question="a question?")

    assert result["composite"] > 0.7
    assert result["citation_coverage"] == 1.0


def test_score_confidence_no_citations_scores_zero_coverage():
    chunks = [{"relevance_score": 9.0}]

    with patch("app.generation.confidence_scorer._score_answer_completeness", return_value=0.5):
        result = score_confidence(chunks, verified_citations=[], answer="answer", question="question?")

    assert result["citation_coverage"] == 0.0


def test_score_confidence_no_chunks_scores_zero_retrieval():
    with patch("app.generation.confidence_scorer._score_answer_completeness", return_value=0.5):
        result = score_confidence(chunks=[], verified_citations=[], answer="answer", question="question?")

    assert result["retrieval_confidence"] == 0.0
    assert result["composite"] == 0.0  