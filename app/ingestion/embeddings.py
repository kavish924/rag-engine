from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if settings.embedding_provider == "local":
        model = _get_local_model()

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    client = _get_openai_client()

    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )

    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]