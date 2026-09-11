from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):

    llm_provider: Literal["gemini"] = "gemini"

    # No default — missing key must fail loudly at startup, not silently at query time.
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")

    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        alias="GEMINI_MODEL",
    )
    generation_temperature: float = 0.1
    generation_max_tokens: int = 1024

    # Narrowed to what's actually implemented. Widen this only when you've
    # written and tested the groq/gemini judge branches — not before.
    eval_judge_provider: Literal["ollama"] = "ollama"

    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        alias="OLLAMA_BASE_URL",
    )
    ollama_judge_model: str = Field(
        default="llama3.2:latest",
        alias="OLLAMA_JUDGE_MODEL",
    )

    # Narrowed to what's actually implemented — no OPENAI_API_KEY exists
    # anywhere in this project, so "openai" was a promise with nothing behind it.
    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "rag_chunks"

    dense_top_k: int = 10
    sparse_top_k: int = 10

    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3

    rerank_top_n: int = 5
    confidence_threshold: float = 0.45

    api_host: str = "0.0.0.0"
    api_port: int = 8000  # matches .env, compose, and Dockerfile now — see step 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()