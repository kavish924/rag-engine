
from fastapi import FastAPI

from app.api.routes import ask, documents, ingest

app = FastAPI(
    title="RAG Engine API",
    description="Hybrid-retrieval, citation-verified Q&A over a document corpus.",
    version="0.1.0",
)

app.include_router(ask.router, prefix="/v1", tags=["ask"])
app.include_router(documents.router, prefix="/v1", tags=["documents"])
app.include_router(ingest.router, prefix="/v1", tags=["ingest"])


@app.get("/health")
def health():
    return {"status": "ok"}
