# Step 25 — FastAPI

## Goal

Expose the pipeline that steps 02–24 already proved through the four endpoints
promised by the roadmap:

- `POST /query`
- `POST /documents`
- `GET /health`
- `GET /metrics`

FastAPI is packaging here, not a second implementation of retrieval, caching,
tracing, or indexing.

## Contract

### `POST /query`

Accept a non-empty question, optional conversation history, and optional
metadata filters. Call `cached_answer()` and return the existing `Answer`
model. The endpoint never asks for `Answer.contexts`, so corpus text is not
echoed in the response. Invalid input is FastAPI's `422`; an unavailable
pipeline is a content-free `503`. Require the `X-API-Key` header configured by
`RAG_API_KEY`.

### `POST /documents`

Accept one caller-identified Markdown document, run the existing
`clean -> chunk -> embed -> Qdrant` pipeline, replace any points with the same
`document_id`, invalidate the process-local BM25 index, atomically advance the
cache corpus revision, and clear old cache entries after a successful write.
Upsert new deterministic point IDs before deleting obsolete tail IDs, so an
upsert failure preserves the previous document. Require `X-API-Key`. Return the
document ID and number of chunks indexed. A document removed by the existing
stub rule is a `400`; an unavailable provider or store is a `503`.

### `GET /health`

Report readiness for Qdrant and, only when configured, Redis. Return `200` when
all required dependencies answer and `503` otherwise. Do not require model
calls or spend money for a health probe.

### `GET /metrics`

Return the content-free aggregate already defined by `trace_report.py`. With no
trace configured or no trace file yet, return an empty aggregate rather than an
error. A malformed trace remains an operational failure and returns `503`.

## Shape

- `app/api/models.py`: HTTP-only request and response models.
- `app/api/main.py`: app factory, dependency seams, and four routes.
- `app/ingestion/index.py`: reusable one-document indexing operation.
- `app/observability/metrics.py`: promote trace loading and aggregation out of
  the CLI; the CLI becomes a caller.
- `tests/test_api.py`: endpoint contract, validation, failure mapping, and
  dependency injection without network access.
- `tests/test_ingestion_index.py`: indexing order, replacement, and cache
  invalidation without provider access.

## Execution

1. Write endpoint and indexing tests; verify they fail because the modules do
   not exist.
2. Add the HTTP models and reusable indexing function.
3. Add the app factory and routes using synchronous path operations so FastAPI
   runs the blocking SDK calls in its worker pool.
4. Promote the trace summary and keep `scripts/trace_report.py` compatible.
5. Run Ruff, formatting, strict mypy, and the complete pytest suite.
6. Update `README.md` and `docs/roadmap.md` with the shipped contract and real
   test evidence.

## Explicitly deferred

- `GET /documents` and `DELETE /documents/{id}` from the broad original brief:
  step 25's roadmap promises only the four endpoints above.
- Rate limiting, CORS policy, and multi-worker trace export: none has a product
  requirement yet.
- Docker packaging and CI: steps 26–28.
