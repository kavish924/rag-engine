# rag-engine

A Retrieval-Augmented Generation system built to demonstrate hybrid retrieval, multi-strategy chunking, and a rigorous evaluation harness over a document corpus. Built as a portfolio project — the goal is to show production-RAG tradeoffs (retrieval fusion, citation grounding, eval methodology), not to be a finished product.

## Status

This project is under active development. The eval numbers below are **preliminary** — see  before treating any score as representative of system quality.

## Architecture

Three services, orchestrated with Docker Compose:

| Service | Role | Depends on |
|---|---|---|
| `chromadb` | Vector store (dense embeddings + persisted chunks) | — |
| `api` | FastAPI backend — retrieval, generation, citation verification | `chromadb` |
| `seed` | One-shot ingestion job — parses corpus, chunks, embeds, writes to Chroma | `chromadb` |
| `frontend` | Streamlit UI, calls `api` over HTTP | `api` |

### Retrieval

Hybrid dense + sparse (BM25) retrieval, fused with Reciprocal Rank Fusion, followed by cross-encoder reranking:

- Dense: `sentence-transformers` embeddings (`BAAI/bge-small-en-v1.5`) against ChromaDB
- Sparse: BM25 (`rank-bm25`)
- Fusion: weighted RRF (`RRF_DENSE_WEIGHT` / `RRF_SPARSE_WEIGHT`, configurable)
- Rerank: cross-encoder over the top fused candidates (`RERANK_TOP_N`)

### Chunking

Three strategies, selectable at ingestion time via `--strategy`:

- `fixed_size` — naive fixed-length windows
- `recursive_structure` — structure-aware splitting (headings, sections)
- `semantic` — embedding-similarity-based chunk boundaries

### Generation & citation

- Generation: Gemini, via its OpenAI-compatible endpoint (`app/generation/generator.py`)
- Citations: answers are expected to cite retrieved chunks inline (`[1]`, `[2]`, ...); a verifier checks each citation's excerpt against the source chunk using exact-substring or near-verbatim (`SequenceMatcher`, 0.90 threshold) matching

### Evaluation

A four-metric harness (`eval/run_eval.py`) scores each answer on:

- **Correctness** — judged against a golden answer
- **Faithfulness** — are generated claims grounded in retrieved context
- **Retrieval relevance** — did retrieval surface the expected source documents
- **Citation accuracy** — do cited excerpts actually appear in their cited chunk

Correctness and faithfulness are scored by an LLM judge (`eval/llm_judge_client.py`), currently Ollama-only (`llama3.2:latest`, local, CPU). The harness checkpoints per-case results and supports partial reruns.

## Setup

### Prerequisites

- Docker + Docker Compose
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/))
- Ollama running locally with `llama3.2` pulled, if you intend to run evaluation

### Configuration

Copy `.env.example` to `.env` and fill in:

```
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
GEMINI_API_KEY=<your key>
GEMINI_MODEL=gemini-3.5-flash-lite

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_JUDGE_MODEL=llama3.2:latest

CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_COLLECTION=rag_chunks

DENSE_TOP_K=10
SPARSE_TOP_K=10
RRF_DENSE_WEIGHT=0.7
RRF_SPARSE_WEIGHT=0.3
RERANK_TOP_N=5
CONFIDENCE_THRESHOLD=0.45

API_HOST=0.0.0.0
API_PORT=8000
```

**Never commit `.env`.** It's already in `.gitignore` — keep it that way, and rotate any key that's ever been pasted somewhere it shouldn't (chat logs, screenshots, issue trackers).

### Run

```bash
docker compose build
docker compose up chromadb api frontend
```

Seed the corpus (one-shot, run separately):

```bash
docker compose run --rm seed
```

Drop sample documents (`.pdf`, `.md`, `.txt`, `.html`) into `scripts/sample_corpus/` before seeding.

- API: `http://localhost:8000` (`/health` for a liveness check)
- Frontend: `http://localhost:8501`

### Run evaluation

Evaluation runs locally (not inside Docker) against the `chromadb` and `api` containers:

```bash
python -m eval.run_eval --delay 1.5
```

Useful flags:

- `--limit N` — run only the first N cases, for fast iteration
- `--delay` — seconds between cases (be considerate of Gemini rate limits)

Set `EVAL_VERBOSE=true` or `CITATION_DEBUG=true` for per-case retrieval/citation debug output; both default to off.

## Project structure

```
rag-engine/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.seed
├── Dockerfile.frontend
├── requirements/
│   ├── backend.txt
│   ├── frontend.txt
│   └── seed.txt
├── requirements-dev.txt
├── app/
│   ├── config.py
│   ├── main.py
│   ├── api/routes/
│   ├── ingestion/pipeline.py
│   ├── retrieval/retriever.py
│   └── generation/
│       ├── generator.py
│       └── citation_verifier.py
├── frontend/
│   └── streamlit_app.py
├── scripts/
│   ├── seed_corpus.py
│   └── sample_corpus/
└── eval/
    ├── run_eval.py
    ├── llm_judge_client.py
    ├── golden_dataset.jsonl
    ├── metrics/
    │   ├── correctness.py
    │   ├── faithfulness.py
    │   ├── retrieval_relevance.py
    │   └── citation_accuracy.py
    └── results/