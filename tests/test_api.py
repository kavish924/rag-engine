
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}




def test_ask_returns_answer_when_confidence_is_high():
    fake_chunks = [{
        "id": "chunk-1",
        "text": "The rate limit is 100 requests per minute.",
        "metadata": {"source_document": "api_guide.md", "section_heading": "Rate Limits"},
        "relevance_score": 9.0,
    }]

    with patch("app.api.routes.ask.retrieve", return_value=fake_chunks), \
         patch("app.api.routes.ask.generate_answer") as mock_generate, \
         patch("app.api.routes.ask.verify_citations") as mock_verify, \
         patch("app.api.routes.ask.score_confidence") as mock_score:

        mock_generate.return_value = {
            "answer": "The rate limit is 100 requests per minute [1].",
            "raw_citations": [{"marker": "[1]", "chunk_index": 1, "chunk": fake_chunks[0], "claim_text": "..."}],
        }
        mock_verify.return_value = [{
            "marker": "[1]", "chunk_id": "chunk-1", "source_document": "api_guide.md",
            "section_heading": "Rate Limits", "supported": True, "excerpt": "The rate limit is...",
        }]
        mock_score.return_value = {
            "retrieval_confidence": 0.9, "citation_coverage": 1.0,
            "answer_completeness": 0.9, "composite": 0.93,
        }

        response = client.post("/v1/ask", json={"question": "What is the rate limit?"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_fallback"] is False
    assert body["confidence"]["composite"] == 0.93
    assert len(body["citations"]) == 1


def test_ask_returns_fallback_when_confidence_is_low():
    fake_chunks = [{
        "id": "chunk-1", "text": "unrelated text",
        "metadata": {"source_document": "misc.md"}, "relevance_score": 1.0,
    }]

    with patch("app.api.routes.ask.retrieve", return_value=fake_chunks), \
         patch("app.api.routes.ask.generate_answer") as mock_generate, \
         patch("app.api.routes.ask.verify_citations", return_value=[]), \
         patch("app.api.routes.ask.score_confidence") as mock_score, \
         patch("app.api.routes.ask.settings") as mock_settings:

        mock_settings.confidence_threshold = 0.45
        mock_generate.return_value = {"answer": "I'm not sure.", "raw_citations": []}
        mock_score.return_value = {
            "retrieval_confidence": 0.1, "citation_coverage": 0.0,
            "answer_completeness": 0.2, "composite": 0.1,
        }

        response = client.post("/v1/ask", json={"question": "Some obscure question"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_fallback"] is True
    assert body["confidence"]["composite"] == 0.0  


def test_ask_returns_fallback_when_nothing_retrieved():
    with patch("app.api.routes.ask.retrieve", return_value=[]):
        response = client.post("/v1/ask", json={"question": "Anything at all?"})

    assert response.status_code == 200
    body = response.json()
    assert body["is_fallback"] is True
    assert body["retrieved_chunk_ids"] == []


def test_ask_rejects_missing_question_field():
    response = client.post("/v1/ask", json={})
    assert response.status_code == 422 



def test_list_documents_returns_summaries():
    fake_docs = [{
        "source_file": "api_guide.md",
        "num_chunks": 4,
        "chunking_strategies_used": ["recursive_structure"],
    }]

    with patch("app.api.routes.documents._get_vector_store") as mock_get_store:
        mock_get_store.return_value.list_documents.return_value = fake_docs
        response = client.get("/v1/documents")

    assert response.status_code == 200
    body = response.json()
    assert len(body["documents"]) == 1
    assert body["documents"][0]["source_file"] == "api_guide.md"
    assert body["documents"][0]["num_chunks"] == 4


def test_list_documents_empty_store():
    with patch("app.api.routes.documents._get_vector_store") as mock_get_store:
        mock_get_store.return_value.list_documents.return_value = []
        response = client.get("/v1/documents")

    assert response.status_code == 200
    assert response.json()["documents"] == []


#