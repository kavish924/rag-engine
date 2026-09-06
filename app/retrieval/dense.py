
from app.config import settings
from app.ingestion.embeddings import embed_query
from app.storage.vector_store import VectorStore

_vector_store: VectorStore | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(
            collection_name=settings.chroma_collection,
            persist_directory="./chroma_data",
        )
    return _vector_store


def dense_search(query: str, top_k: int = 10) -> list[dict]:
    query_embedding = embed_query(query)
    store = _get_vector_store()
    return store.query(query_embedding, top_k=top_k)