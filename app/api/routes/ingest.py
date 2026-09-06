
from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.ingestion.pipeline import run_ingestion

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    try:
        result = run_ingestion(
            file_paths=request.file_paths,
            strategy=request.chunking_strategy,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
    
        raise HTTPException(status_code=400, detail=str(e))

    return IngestResponse(
        ingested_files=result["files_processed"],
        chunks_created=result["chunks_created"],
        duplicates_skipped=result["duplicates_skipped"],
    )
