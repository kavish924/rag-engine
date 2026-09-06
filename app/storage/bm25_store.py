
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Store:
    def __init__(self, persist_path: str = "./chroma_data/bm25_index.pkl"):
        self.persist_path = Path(persist_path)
        self.chunk_ids: list[str] = []
        self.chunk_texts: list[str] = []
        self.chunk_metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None

        if self.persist_path.exists():
            self._load()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Each chunk dict must have: id, text, metadata."""
        if not chunks:
            return
        for c in chunks:
            self.chunk_ids.append(c["id"])
            self.chunk_texts.append(c["text"])
            self.chunk_metadatas.append(c["metadata"])
        self._rebuild_index()
        self._save()

    def query(self, query_text: str, top_k: int = 10) -> list[dict]:
        if self._bm25 is None or not self.chunk_ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "id": self.chunk_ids[i],
                "text": self.chunk_texts[i],
                "metadata": self.chunk_metadatas[i],
                "score": float(scores[i]),
            }
            for i in ranked_indices
            if scores[i] > 0
        ]

    def _rebuild_index(self) -> None:
        tokenized_corpus = [_tokenize(t) for t in self.chunk_texts]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "chunk_texts": self.chunk_texts, "chunk_metadatas": self.chunk_metadatas},
                f,
            )

    def _load(self) -> None:
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.chunk_texts = data["chunk_texts"]
        self.chunk_metadatas = data["chunk_metadatas"]
        self._rebuild_index()
