# Global Plan

The single place that answers: where is this project, what comes next, and why.

- **Brief:** [`information.md`](../information.md) — the original 30-step learning order.
- **Public front page:** [`README.md`](../README.md) (French) — phases 0-17.
- **Step plans:** [`docs/superpowers/plans/`](superpowers/plans/) — one executable plan per step.

## Current state

**Phase 0 and steps 02-07 are done. Step 08 (basic RAG) is next.**

Shipped and verified:

| Artifact | Evidence |
|---|---|
| Python 3.12 workspace locked with `uv` | `pyproject.toml`, `uv.lock`, `.python-version` |
| Package boundaries | `app/{api,core,ingestion,retrieval,generation,evaluation,models}/__init__.py` — all empty |
| Quality gates | Ruff, mypy strict on `app`, pytest, pinned pre-commit hooks |
| Local Qdrant | `compose.yaml`, pinned `qdrant/qdrant:v1.19.1`, healthcheck, loopback-bound |
| Smoke test | `tests/test_environment.py` — imports only |
| FastAPI corpus, fetched and loaded | `scripts/fetch_corpus.py`, `app/ingestion/loader.py` — 155 Markdown files |
| Markdown cleaning | `app/ingestion/clean.py` — 155 in, 8 stubs dropped, 1 463 414 → 1 041 001 chars (71%), 0 code blocks lost |
| Basic chunking | `app/ingestion/chunk.py` — 147 docs → 1 607 chunks, median 795 chars, 10 oversized (a whole code fence each), reconstruction tested |
| Embeddings, cached | `app/ingestion/embed.py` — 1 607 chunks → 1 536 dims via `text-embedding-3-small`, 404 s cold / 0.12 s warm, 13.3 MB sqlite cache, $0.006 |
| Shared settings | `app/core/config.py` — one `Settings`, `.env` loaded once through an `lru_cache`d accessor |
| Qdrant indexing | `app/retrieval/store.py`, `scripts/index_corpus.py` — 1 607 points, cosine, payload indexes on `source`/`document_id`/`language`, 4.4 s end to end; a re-run leaves 1 607 points in 3.5 s |
| Similarity search | `app/retrieval/search.py`, `scripts/search.py` — ranked `ScoredChunk`s, 119 ms warm / 0.9-1.8 s cold, 15 tests; `HTTPException 422` recorded as the hybrid-search target (1 chunk in the top 50 contains the token, at rank 5) |

The corpus is indexed and queryable. Nothing generates an answer yet: no LLM, no endpoint.

## The three rules

1. **Measure, then improve.** No retrieval or generation change lands without a number next to it. The README's results table is the scoreboard; an em dash means "not measured", never "good enough".
2. **Hand-roll before you framework.** `embed()`, `search()`, `build_context()`, `generate()` are written by hand first (the brief is explicit about this). LangChain arrives when composing retrievers, rerankers and tracing is genuinely cheaper than not having it — around step 17, not step 05.
3. **Dependencies arrive with the step that uses them.** No `ragas` in `pyproject.toml` before step 21, no `redis` before step 23.

## Step map

`information.md` lists 30 steps; the README groups them into 18 phases; git tags mark the milestones. Same road, three resolutions:

| Steps | README phase | Tag | What you can demo at the end |
|---|---|---|---|
| 01 | Phase 0 — Foundation | — | A clean repo that builds and a healthy Qdrant. **Done.** |
| 02-08 | Phase 1 — Minimal RAG | `v0.2` | Ask a question about FastAPI, get an answer from your own corpus. |
| 09 | Phase 10 — Citations | `v0.3` | Every factual sentence carries `[n]` and resolves to a real source. |
| 10-11 | Phase 4 — Retrieval evaluation | `v0.4` | A baseline table: Recall@5, Recall@10, MRR, NDCG. The project is presentable here. |
| 12 | Phase 2 — Chunking | `v0.4+` | Five chunking strategies, one winner, chosen by Recall@5. |
| 13 | Phase 3 — Metadata | — | `source=fastapi` filters, and per-source recall numbers. |
| 14-16 | Phase 5-6 — Hybrid + RRF | `v0.5`-`v0.6` | Dense + BM25 fused; `HTTPException 422` finally retrieves. |
| 17 | Phase 6 — Reranking | `v0.7` | Top-30 recall, top-5 precision, measured latency cost. |
| 18-19 | Phase 7-8 — Query transforms | `v0.8`-`v0.9` | "et pour docker ?" resolves against conversation history. |
| 20 | Phase 9 — Compression | `v1.2` | Same answer quality, fewer context tokens. |
| 21 | Phase 11 — RAGAS | `v1.3` | Faithfulness and answer relevance, not just retrieval. |
| 22 | Phase 12 — Guardrails | `v1.6` | Refuses when retrieval is weak; survives injected instructions in documents. |
| 23 | Phase 13 — Cache | `v1.4` | Cache-hit latency and cost deltas. |
| 24 | Phase 15 — Observability | `v1.5` | Per-query trace: scores, prompt, tokens, latency, cost. |
| 25 | Phase 14 — FastAPI | `v1.0` | `POST /query`, `POST /documents`, `GET /health`, `GET /metrics`. |
| 26-28 | Phase 16-17 — Docker, tests, CI | `v1.0` | `docker compose up` and a green pipeline with RAG regression gates. |
| 29-30 | — | — | Benchmark dashboard and the README that tells the whole story. |

## Plans written so far

Detailed, executable plans exist for steps 02-11:

| Step | Plan |
|---|---|
| 02 Document ingestion | [`2026-09-11-step-02-document-ingestion.md`](superpowers/plans/2026-09-11-step-02-document-ingestion.md) |
| 03 Cleaning | [`2026-09-11-step-03-cleaning.md`](superpowers/plans/2026-09-11-step-03-cleaning.md) |
| 04 Basic chunking | [`2026-09-11-step-04-basic-chunking.md`](superpowers/plans/2026-09-11-step-04-basic-chunking.md) |
| 05 Embeddings | [`2026-09-11-step-05-embeddings.md`](superpowers/plans/2026-09-11-step-05-embeddings.md) |
| 06 Qdrant indexing | [`2026-09-11-step-06-qdrant.md`](superpowers/plans/2026-09-11-step-06-qdrant.md) |
| 07 Similarity search | [`2026-09-11-step-07-similarity-search.md`](superpowers/plans/2026-09-11-step-07-similarity-search.md) |
| 08 Basic RAG | [`2026-09-11-step-08-basic-rag.md`](superpowers/plans/2026-09-11-step-08-basic-rag.md) |
| 09 Citations | [`2026-09-11-step-09-citations.md`](superpowers/plans/2026-09-11-step-09-citations.md) |
| 10 Evaluation dataset | [`2026-09-11-step-10-evaluation-dataset.md`](superpowers/plans/2026-09-11-step-10-evaluation-dataset.md) |
| 11 Retrieval metrics | [`2026-09-11-step-11-retrieval-metrics.md`](superpowers/plans/2026-09-11-step-11-retrieval-metrics.md) |

**Steps 12-30 are deliberately unplanned.** Every one of them is a decision that rule 1 says must be made against measurements: which chunking strategy wins, whether BM25 helps this corpus, whether reranking pays for its latency, whether multi-query is worth 4x the cost. Writing those plans now would mean inventing the answers. Each plan gets written at the start of its own step, with step 11's numbers in hand.

## How a step lands

Same shape every time, no exceptions:

1. Write the step plan in `docs/superpowers/plans/` if it does not exist yet.
2. Add only the dependencies that step needs; `uv lock` and commit the lockfile in the same change.
3. Write the test first, watch it fail, then the code. Quality gates must pass: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.
4. Re-run the benchmark from step 11 once it exists. Record the real numbers — including the regressions.
5. Update the README in the same commit: current state, roadmap checkbox, new commands, results row.
6. Tag when a milestone in the step map is reached.

## Shape of the code

Target layout, and the step that creates each file. Nothing is created before its step.

```text
app/
├── core/config.py            # 06  Settings (pydantic-settings), single source of env truth
├── models/documents.py       # 02  RawDocument
├── models/chunks.py          # 04  Chunk, ChunkMetadata
├── models/answers.py         # 08  Answer, Source, RetrievalStats
├── ingestion/loader.py       # 02  filesystem -> RawDocument
├── ingestion/clean.py        # 03  markdown -> clean text
├── ingestion/chunk.py        # 04  text -> Chunk list
├── ingestion/embed.py        # 05  texts -> vectors, with a sqlite cache
├── retrieval/store.py        # 06  Qdrant collection, upsert
├── retrieval/search.py       # 07  query -> scored chunks
├── generation/context.py     # 08  chunks -> numbered context block
├── generation/llm.py          # 08  single LLM call
├── generation/answer.py      # 08  retrieve -> context -> prompt -> Answer
├── generation/citations.py   # 09  parse and validate [n] references
├── evaluation/dataset.py     # 10  load and validate the eval set
├── evaluation/metrics.py     # 11  recall@k, precision@k, mrr, ndcg, hit rate
├── evaluation/benchmark.py   # 11  run the set, emit a markdown row
└── api/                      # 25  FastAPI, last
scripts/
├── fetch_corpus.py           # 02
├── index_corpus.py           # 06
├── ask.py                    # 08
└── benchmark.py              # 11
```

`api/` is last on purpose. A CLI proves the pipeline works; HTTP is packaging.
