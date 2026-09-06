# RAG Engine — Hybrid Retrieval, Citation-Verified Q&A

A production-oriented Retrieval-Augmented Generation system built from the
ground up: hybrid (dense + sparse) retrieval with RRF fusion, configurable
chunking strategies, a citation-verified generation layer, a composite
confidence scorer, and a full LLM-as-judge evaluation harness — all served
behind a FastAPI backend with a Streamlit dashboard, containerized with
Docker Compose.

This isn't a LangChain quickstart. It's an attempt to build the parts most
RAG tutorials skip: verifying that citations actually support the claims
they're attached to, knowing when to say "I don't know," and having an
evaluation suite that can tell you why a change helped or hurt.

## Why this exists

Most RAG demos stop at "it retrieves chunks and the LLM answers." The
interesting engineering problems start after that:

- Retrieval that only works for prose is a liability for technical docs
  full of exact-match tokens (function names, config keys, error codes) —
  hence dense and sparse retrieval, fused with RRF.
- An LLM will confidently cite a chunk that doesn't actually support its
  claim. So citations are verified post-hoc, not trusted.
- "Which chunking strategy is best?" is an empirical question, not a
  guess — so chunking strategy is a swappable, evaluable parameter, and
  there's a script that compares all three head-to-head.
- You can't improve what you don't measure — so every generation is
  scored on correctness, faithfulness, retrieval relevance, and citation
  accuracy against a hand-written golden dataset.

## Features

- **Hybrid retrieval** - dense (ChromaDB + `text-embedding-3-small`) and
  sparse (BM25) search, combined with Reciprocal Rank Fusion, followed by
  an LLM-as-judge reranking pass.
-  **Configurable chunking** - fixed-size, structure-aware (recursive/
  header-based), and semantic (topic-boundary) strategies, selectable
  per ingestion run and directly comparable via the eval harness.
- **Grounded generation** — prompt templates that constrain the model to
  the retrieved context, with inline citations parsed from the response.
- **Citation verification** — every citation-claim pair is checked
  against the actual source chunk before it's surfaced to the user or
  counted toward confidence.
- **Confidence scoring** — a composite score gates low-confidence answers
  behind a structured "I don't know" fallback instead of a hallucination.
- **Evaluation framework** — four LLM-as-judge metrics (correctness,
  faithfulness, retrieval relevance, citation accuracy) run against a
  50+ question golden dataset, plus a dedicated chunking-strategy
  comparison script.
- **API + dashboard** — a FastAPI service (`/v1/ask`, `/v1/documents`,
  `/v1/ingest`) and a Streamlit dashboard for asking questions and
  inspecting citations, confidence, and hybrid-vs-dense-only retrieval
  side by side.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI, Pydantic v2 |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | Anthropic Claude (default), OpenAI (swappable) |
| Dashboard | Streamlit |
| Infra | Docker Compose |
| Testing | pytest, pytest-asyncio, httpx |

## Project structure

```
rag-engine/
├── app/
│   ├── main.py                       # FastAPI entrypoint
│   ├── config.py                     # Settings (env vars, model names, retrieval weights)
│   ├── api/
│   │   ├── schemas.py                 # Pydantic request/response models
│   │   └── routes/
│   │       ├── ask.py                   # POST /v1/ask
│   │       ├── documents.py             # GET  /v1/documents
│   │       └── ingest.py                # POST /v1/ingest
│   ├── ingestion/                    # Phase 1 — load, chunk, embed, dedup
│   │   ├── loaders.py                    # multi-format loader (md/txt/html/pdf)
│   │   ├── chunkers/
│   │   │   ├── fixed_size.py                # baseline: fixed-size + overlap
│   │   │   ├── recursive_structure.py       # structure-aware: split by headers
│   │   │   └── semantic.py                  # semantic: split on topic boundaries
│   │   ├── embeddings.py                 # text-embedding-3-small wrapper
│   │   ├── dedup.py                      # near-duplicate detection (cosine > 0.95)
│   │   └── pipeline.py                   # orchestrates the full ingestion flow
│   ├── retrieval/                    # Phase 2 — hybrid retrieval engine
│   │   ├── dense.py                      # vector search (ChromaDB)
│   │   ├── sparse_bm25.py                # BM25 keyword search
│   │   ├── fusion.py                     # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py                   # LLM-as-judge rerank
│   │   └── retriever.py                  # orchestrates dense+sparse+fusion+rerank
│   ├── generation/                   # Phase 3 — grounded generation + trust layer
│   │   ├── prompts.py                    # grounded system prompt templates
│   │   ├── generator.py                  # calls the LLM, parses citations
│   │   ├── citation_verifier.py          # verifies each citation-claim pair
│   │   └── confidence_scorer.py          # composite confidence score
│   └── storage/                      # Vector store + BM25 index wrappers
│       ├── vector_store.py
│       └── bm25_store.py
├── eval/                             # Phase 4 — evaluation framework
│   ├── golden_dataset.jsonl           # 50+ hand-written Q&A pairs
│   ├── metrics/
│   │   ├── correctness.py                # LLM-as-judge vs. golden answer
│   │   ├── faithfulness.py               # are claims grounded in context?
│   │   ├── retrieval_relevance.py        # were the right chunks retrieved?
│   │   └── citation_accuracy.py          # do citations support claims?
│   ├── run_eval.py                    # runs full suite, produces report
│   └── chunking_comparison.py         # compares the 3 chunking strategies
├── frontend/
│   └── streamlit_app.py               # ask questions, inspect citations/confidence,
│                                        # toggle hybrid vs. dense-only retrieval
├── scripts/
│   ├── seed_corpus.py                 # indexes sample_corpus/ for reviewers
│   └── sample_corpus/                 # sample documentation set
├── tests/                            # unit tests per module
├── docs/
│   ├── architecture.md                # data flow + key design decisions
│   └── case_study.md                  # eval results write-up
├── docker-compose.yml                 # API + ChromaDB + frontend
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Architecture

```
Docs (md/txt/html/pdf)
  -> loaders.py (normalize + metadata)
  -> chunkers/{fixed_size,recursive_structure,semantic}.py
  -> embeddings.py (text-embedding-3-small)
  -> dedup.py (cosine > 0.95 -> skip)
  -> storage: vector_store.py (ChromaDB) + bm25_store.py, kept in sync

Query
  -> retrieval/dense.py               (top-k cosine)
  -> retrieval/sparse_bm25.py         (top-k BM25)
  -> retrieval/fusion.py              (RRF, weighted)
  -> retrieval/reranker.py            (top-20 -> top-5)
  -> generation/generator.py          (grounded answer + raw citations)
  -> generation/citation_verifier.py  (per-claim verification)
  -> generation/confidence_scorer.py  (composite score)
  -> if confidence < threshold: structured fallback ("I don't know")
  -> API response (app/api/routes/ask.py)
```

Two indexes (ChromaDB + BM25) are written to together at ingestion time and
kept in sync, so retrieval can draw on both without a separate sync step.
Chunking strategy is stored as chunk-level metadata rather than a global
setting, which is what lets `eval/chunking_comparison.py` run the identical
eval suite across all three strategies without re-architecting anything.
Citations are treated as a hypothesis the generator makes, not a fact —
`citation_verifier.py` checks each one against the actual chunk content
before it's shown to the user or counted in the confidence score.

See [`docs/architecture.md`](docs/architecture.md) for more detail.

## Quickstart

```bash
cp .env.example .env          # add your OPENAI_API_KEY / ANTHROPIC_API_KEY
docker-compose up --build     # spins up API + ChromaDB + dashboard
python scripts/seed_corpus.py # indexes the sample corpus
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Running without Docker

```bash
pip install -r requirements.txt
python -m app.main               # start the API
python -m scripts.seed_corpus    # index the sample corpus
streamlit run frontend/streamlit_app.py
```

> This project uses `python -m module.name` invocation throughout so that
> imports from the `app` package resolve correctly regardless of OS or
> working directory.

## Configuration

Key settings (see `app/config.py` / `.env.example`):

| Setting | Default | Purpose |
|---|---|---|
| `llm_provider` | `anthropic` | `anthropic` or `openai` |
| `generation_model` | `claude-sonnet-4-6` | Generator model |
| `embedding_model` | `text-embedding-3-small` | Embedding model |
| `dense_top_k` / `sparse_top_k` | `10` / `10` | Candidates pulled from each index pre-fusion |
| `rrf_dense_weight` / `rrf_sparse_weight` | `0.7` / `0.3` | RRF fusion weights |
| `rerank_top_n` | `5` | Chunks kept after reranking |
| `confidence_threshold` | `0.45` | Below this, the API returns a structured "I don't know" |

## Running evals

```bash
python -m eval.run_eval               
python -m eval.chunking_comparison    
```

Results are scored against `eval/golden_dataset.jsonl` (50+ hand-written
Q&A pairs) using an LLM-as-judge for each metric. See
[`docs/case_study.md`](docs/case_study.md) for a write-up of results once
the suite has been run end-to-end.

## Build order

| Phase | Focus | Folder(s) |
|---|---|---|
| 1. Ingestion & Chunking | Load, chunk, embed, dedup | `app/ingestion/` |
| 2. Hybrid Retrieval | Dense + sparse + RRF + rerank | `app/retrieval/` |
| 3. Generation & Citations | Grounded generation, verified citations, confidence | `app/generation/` |
| 4. Evaluation Framework | LLM-as-judge metrics, golden dataset | `eval/` |
| 5. API & Dashboard | FastAPI routes, Streamlit UI | `app/api/`, `frontend/`, `docker-compose.yml` |
| 6. Portfolio Polish | Docs, architecture write-up, case study | `docs/` |

## Testing

```bash
python -m pytest tests/
```
