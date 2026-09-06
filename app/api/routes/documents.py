
from fastapi import APIRouter

from app.api.schemas import DocumentListResponse, DocumentSummary
from app.retrieval.dense import _get_vector_store

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    vector_store = _get_vector_store()
    documents = vector_store.list_documents()

    return DocumentListResponse(
        documents=[
            DocumentSummary(
                source_file=doc["source_file"],
                num_chunks=doc["num_chunks"],
                chunking_strategies_used=doc["chunking_strategies_used"],
                last_indexed_at="unknown", 
            )
            for doc in documents
        ]
    )