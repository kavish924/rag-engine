from typing import Literal, Optional
from pydantic import BaseModel, Field
class AskRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask against the indexed documents.",
    )

    retrieval_mode: Literal[
        "hybrid",
        "dense_only",
    ] = "hybrid"

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum number of chunks to keep after retrieval.",
    )

class Citation(BaseModel):

    marker: str

    chunk_id: str

    source_document: str

    section_heading: Optional[str] = None

    supported: bool

    excerpt: str

class ConfidenceBreakdown(BaseModel):
    retrieval_confidence: float

    citation_coverage: float

    answer_completeness: float

    composite: float
class RetrievedChunk(BaseModel):

    rank: int

    chunk_id: str

    score: Optional[float] = None

    source_document: str

    section_heading: Optional[str] = None

    preview: str

class PerformanceMetrics(BaseModel):

    retrieval_ms: float = 0.0

    generation_ms: float = 0.0

    verification_ms: float = 0.0

    confidence_ms: float = 0.0

    total_ms: float = 0.0
class TokenUsage(BaseModel):

    prompt_tokens: Optional[int] = None

    completion_tokens: Optional[int] = None

    total_tokens: Optional[int] = None

class AskResponse(BaseModel):
    answer: str

    citations: list[Citation]

    confidence: ConfidenceBreakdown

    retrieved_chunks: list[RetrievedChunk]

    timings: Optional[PerformanceMetrics] = None

    usage: Optional[TokenUsage] = None

    is_fallback: bool = False
class DocumentSummary(BaseModel):

    source_file: str

    num_chunks: int

    chunking_strategies_used: list[str]

    last_indexed_at: str


class DocumentListResponse(BaseModel):

    documents: list[DocumentSummary]
class IngestRequest(BaseModel):

    file_paths: list[str]

    chunking_strategy: Literal[
        "fixed_size",
        "recursive_structure",
        "semantic",
    ] = "recursive_structure"


class IngestResponse(BaseModel):

    ingested_files: list[str]

    chunks_created: int

    duplicates_skipped: int