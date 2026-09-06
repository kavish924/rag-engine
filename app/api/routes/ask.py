

import time

from fastapi import APIRouter

from app.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    ConfidenceBreakdown,
    RetrievedChunk,
    PerformanceMetrics,
)

from app.config import settings

from app.generation.citation_verifier import (
    verify_citations,
)

from app.generation.confidence_scorer import (
    score_confidence,
)

from app.generation.generator import (
    generate_answer,
)

from app.retrieval.retriever import (
    retrieve,
)


router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(
    request: AskRequest,
) -> AskResponse:

    total_start = time.perf_counter()


    retrieval_start = time.perf_counter()

    chunks = retrieve(
        request.question,
        mode=request.retrieval_mode,
        rerank_top_n=request.top_k,
    )

    retrieval_ms = (
        time.perf_counter()
        - retrieval_start
    ) * 1000


    if not chunks:

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        timings = PerformanceMetrics(
            retrieval_ms=round(
                retrieval_ms,
                1,
            ),
            generation_ms=0.0,
            verification_ms=0.0,
            confidence_ms=0.0,
            total_ms=round(
                total_ms,
                1,
            ),
        )

        return _fallback_response(
            reason=(
                "No relevant documents were found "
                "in the corpus for this question"
            ),
            retrieved_chunks=[],
            chunks=[],
            timings=timings,
        )

    generation_start = time.perf_counter()

    generation_result = generate_answer(
        request.question,
        chunks,
    )

    generation_ms = (
        time.perf_counter()
        - generation_start
    ) * 1000


    verification_start = time.perf_counter()

    verified_citations = verify_citations(
        generation_result["raw_citations"],
        chunks,
    )

    verification_ms = (
        time.perf_counter()
        - verification_start
    ) * 1000

    confidence_start = time.perf_counter()

    confidence = score_confidence(
        chunks=chunks,
        verified_citations=verified_citations,
        answer=generation_result["answer"],
        question=request.question,
    )

    confidence_ms = (
        time.perf_counter()
        - confidence_start
    ) * 1000

    retrieved_chunks = [
        RetrievedChunk(
            rank=i + 1,
            chunk_id=c["id"],
            score=c.get(
                "relevance_score"
            ),
            source_document=(
                c["metadata"].get(
                    "source_document",
                    "Unknown",
                )
            ),
            section_heading=(
                c["metadata"].get(
                    "section_heading"
                )
            ),
            preview=(
                c["text"][:150] + "..."
                if len(c["text"]) > 150
                else c["text"]
            ),
        )
        for i, c in enumerate(
            chunks
        )
    ]

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000

    timings = PerformanceMetrics(
        retrieval_ms=round(
            retrieval_ms,
            1,
        ),
        generation_ms=round(
            generation_ms,
            1,
        ),
        verification_ms=round(
            verification_ms,
            1,
        ),
        confidence_ms=round(
            confidence_ms,
            1,
        ),
        total_ms=round(
            total_ms,
            1,
        ),
    )
    _print_performance_metrics(
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        verification_ms=verification_ms,
        confidence_ms=confidence_ms,
        total_ms=total_ms,
    )

    if (
        confidence["composite"]
        < settings.confidence_threshold
    ):

        return _fallback_response(
            reason=(
                _build_low_confidence_reason(
                    confidence
                )
            ),
            retrieved_chunks=(
                retrieved_chunks
            ),
            chunks=chunks,
            timings=timings,
        )

    return AskResponse(
        answer=(
            generation_result["answer"]
        ),
        citations=[
            Citation(**citation)
            for citation
            in verified_citations
        ],
        confidence=(
            ConfidenceBreakdown(
                **confidence
            )
        ),
        retrieved_chunks=(
            retrieved_chunks
        ),
        timings=timings,
        is_fallback=False,
    )

def _print_performance_metrics(
    retrieval_ms: float,
    generation_ms: float,
    verification_ms: float,
    confidence_ms: float,
    total_ms: float,
) -> None:
    print(
        "\n"
        + "=" * 55
    )

    print(
        "RAG PIPELINE PERFORMANCE"
    )

    print(
        "=" * 55
    )

    print(
        f"Retrieval:             "
        f"{retrieval_ms / 1000:.2f}s"
    )

    print(
        f"Generation:            "
        f"{generation_ms / 1000:.2f}s"
    )

    print(
        f"Citation verification: "
        f"{verification_ms / 1000:.4f}s"
    )

    print(
        f"Confidence scoring:    "
        f"{confidence_ms / 1000:.2f}s"
    )

    print(
        "-" * 55
    )

    print(
        f"Total:                 "
        f"{total_ms / 1000:.2f}s"
    )

    print(
        "=" * 55
        + "\n"
    )

def _build_low_confidence_reason(
    confidence: dict,
) -> str:

    weak_points = []

    if (
        confidence[
            "retrieval_confidence"
        ]
        < 0.4
    ):

        weak_points.append(
            "the retrieved chunks weren't "
            "strongly relevant to the question"
        )

    if (
        confidence[
            "citation_coverage"
        ]
        < 0.4
    ):

        weak_points.append(
            "the generated claims couldn't "
            "be verified against the source "
            "chunks"
        )

    if (
        confidence[
            "answer_completeness"
        ]
        < 0.4
    ):

        weak_points.append(
            "the answer didn't fully address "
            "the question"
        )

    if not weak_points:

        return (
            "overall confidence in this answer "
            "was too low to return it directly"
        )

    return "; ".join(
        weak_points
    )

def _fallback_response(
    reason: str,
    retrieved_chunks: list[
        RetrievedChunk
    ],
    chunks: list[dict] | None = None,
    timings: PerformanceMetrics | None = None,
) -> AskResponse:

    checkable_docs = sorted(
        {
            c["metadata"].get(
                "source_document",
                "unknown",
            )
            for c in (
                chunks or []
            )
        }
    )


    doc_hint = (
        (
            " You may want to check: "
            + ", ".join(
                checkable_docs
            )
            + "."
        )
        if checkable_docs
        else ""
    )

    answer = (
        "I don't have enough confidence "
        "to answer this reliably — "
        f"{reason}."
        f"{doc_hint}"
    )


    if timings is None:

        timings = PerformanceMetrics(
            retrieval_ms=0.0,
            generation_ms=0.0,
            verification_ms=0.0,
            confidence_ms=0.0,
            total_ms=0.0,
        )
    return AskResponse(
        answer=answer,
        citations=[],
        confidence=(
            ConfidenceBreakdown(
                retrieval_confidence=0.0,
                citation_coverage=0.0,
                answer_completeness=0.0,
                composite=0.0,
            )
        ),
        retrieved_chunks=(
            retrieved_chunks
        ),
        timings=timings,
        is_fallback=True,
    )