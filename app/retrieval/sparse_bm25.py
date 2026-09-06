
from app.storage.bm25_store import BM25Store

_bm25_store: BM25Store | None = None


def _get_bm25_store() -> BM25Store:
    global _bm25_store
    if _bm25_store is None:
        _bm25_store = BM25Store(persist_path="./chroma_data/bm25_index.pkl")
    return _bm25_store


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns a list of dicts: {"id", "text", "metadata", "score"}
    where score is the raw BM25 score (higher = more relevant).
    """
    store = _get_bm25_store()
    return store.query(query, top_k=top_k)