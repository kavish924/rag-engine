
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    llm_provider: Literal["groq"] = "groq"

    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
    )

    generation_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GENERATION_MODEL",
    )

    embedding_provider: Literal["local", "openai"] = "local"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

  
    openai_embedding_model: str = "text-embedding-3-small"

    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
    )


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
    api_port: int = 8000


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()