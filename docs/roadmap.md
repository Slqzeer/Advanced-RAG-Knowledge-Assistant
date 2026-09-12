# Global Plan

The single place that answers: where is this project, what comes next, and why.

- **Brief:** [`information.md`](../information.md) — the original 30-step learning order.
- **Public front page:** [`README.md`](../README.md) (French) — phases 0-17.
- **Step plans:** [`docs/superpowers/plans/`](superpowers/plans/) — one executable plan per step.

## Current state

**Phase 0 and steps 02-16 are done — `v0.6` is tagged. Step 17 (reranking) is next,
and it now has something to rerank.**

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
| Basic chunking | `app/ingestion/chunk.py` — recursive splitting, 147 docs → 1 607 chunks, median 795 chars, 10 oversized (a whole code fence each), reconstruction tested |
| Embeddings, cached | `app/ingestion/embed.py` — 1 607 chunks → 1 536 dims via `text-embedding-3-small`, 404 s cold / 0.12 s warm, 13.3 MB sqlite cache, $0.006 |
| Shared settings | `app/core/config.py` — one `Settings`, `.env` loaded once through an `lru_cache`d accessor |
| Qdrant indexing | `app/retrieval/store.py`, `scripts/index_corpus.py` — 1 607 points, cosine, payload indexes on `source`/`document_id`/`language`, 4.4 s end to end; a re-run leaves 1 607 points in 3.5 s |
| Similarity search | `app/retrieval/search.py`, `scripts/search.py` — ranked `ScoredChunk`s, 119 ms warm / 0.9-1.8 s cold, 15 tests; `HTTPException 422` recorded here as the hybrid-search target (1 chunk in the top 50 contains the token, at rank 5) — steps 14-16 settled it, see below |
| Minimal RAG | `app/generation/{context,llm,answer}.py`, `app/models/answers.py`, `scripts/ask.py` — `gpt-4o-mini` at `temperature=0`, 1.5-3.7 s end to end, 851-1 021 tokens/question (~$0.0003), 23 tests with no network; a French question answers in French, an out-of-corpus question answers "I do not know" |
| Citation validation | `app/generation/citations.py` — every `[n]` parsed and checked against the context actually supplied, sources renumbered from 1 in the text *and* the list, `--strict` fails instead of warning |
| Evaluation dataset | `app/evaluation/dataset.py`, `scripts/validate_dataset.py` — 50 questions, 5 categories, document-level ground truth cross-checked against the cleaned corpus, 5 held out |
| Retrieval metrics | `app/evaluation/metrics.py`, `app/evaluation/benchmark.py`, `scripts/benchmark.py` — Recall@K, Precision@K, MRR, Hit Rate@K, NDCG@K over deduplicated documents, per-category breakdown, p50/p95 latency, history in `data/eval/results.jsonl` with the git commit of each run |
| Chunking comparison | `app/ingestion/chunk.py` — four strategies behind one registry (`recursive`, `fixed`, `sentence`, `semantic`), one Qdrant collection each, nine runs; `sentence` wins Recall@5 0.776 against the 0.713 baseline and becomes the default, 147 docs → 1 484 chunks |
| BM25 index | `app/retrieval/bm25.py` — hand-rolled Okapi BM25 with Lucene's IDF variant, built by scrolling the same collection dense search queries, 1 484 chunks, avg length 126 tokens, ~200 ms build / 2 ms per query, 20 tests, no new dependency |
| Hybrid retrieval | `app/retrieval/search.py` — `RETRIEVERS` keyed `dense`\|`lexical`\|`hybrid`, `rrf()` on ranks only, `matches_filters` giving the lexical branch filter parity; six measured runs. Best fusion row `hybrid-k60-d20`: Recall@5 **0.737** (against 0.776 dense) but Recall@10 **0.829** (against 0.785). The pre-registered rule was not met, so `dense` stays the default |
| Metadata filtering | `app/ingestion/loader.py`, `app/retrieval/search.py` — `doc_type` derived from the corpus layout, indexed, filterable through a generic `filters=` mapping; per-facet recall and an oracle run recording the ceiling on facet routing at **+0.000 Recall@5** |

The loop is closed, verified and measured: a question goes in, a grounded answer with resolved citations comes out of `scripts/ask.py`, and 38 answerable questions give it a score. No HTTP endpoint yet — that is step 25.

Four things later steps own:

- **`HTTPException 422` is resolved as a finding, not as a fix.** Steps 14-16 gave the retriever the missing capability — BM25's rank-1 hit for `HTTPException 422` does contain the literal token, and `--mode hybrid` surfaces `reference/exceptions` at rank 2 — and the answer is still an honest "I do not know". The four chunks in the corpus containing `422` are two release notes and an OpenAPI JSON example; none of them explains what the code means. The refusal is correct. No later step owns this: the target was chosen at step 07, before step 11 said where the gap actually was, and `exact` was already the strongest category at 0.892.
- **Recall@10 minus Recall@5 is now 0.092**, up from 0.009 — RRF fusion opened it tenfold. This was the condition step 17 needed and it is met: 9.2 points of recall now separate rank 5 from rank 10, against 0.9 before, which is exactly what a reranker reorders. **Step 17 is worth running, and it runs on hybrid candidates rather than dense ones.** This is the phase's real result; it is not what the phase set out to achieve.
- **Generation is ~95 % of the latency.** 1.5-3.7 s per question against ~35 ms of warm retrieval. Any latency work before step 23's cache would be optimising the wrong 5 %.
- **A score threshold cannot carry refusal.** The 7 out-of-corpus questions score 0.305-0.459, the answerable ones 0.341-0.664. Step 22 needs something other than a floor.
- **Facet routing has a ceiling of zero.** Step 13's oracle — every question filtered to its own ground-truth `doc_type` — moves Recall@5 by +0.000 and Recall@10 by +0.000; the whole aggregate delta (MRR +0.046) is four questions reordered. Only 16 of 38 answerable questions even have single-facet ground truth, and on the other 22 any single-facet filter drops a relevant document by construction. This is the input step 18 needs: do not build a query-side facet router for this corpus.

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
| 12 | Phase 2 — Chunking | `v0.4+` | Four chunking strategies, one winner, chosen by Recall@5. **Done — `sentence`, Recall@5 0.776.** |
| 13 | Phase 3 — Metadata | `v0.4+` | A derived `doc_type` facet, a generic `filters=` mapping, per-facet recall. **Done — the routing ceiling is +0.000 Recall@5.** |
| 14-16 | Phase 5-6 — Hybrid + RRF | `v0.5`-`v0.6` | Dense + BM25 fused. **Done — hybrid loses Recall@5 (0.737 vs 0.776) and wins Recall@10 (0.829 vs 0.785); `dense` stays the default, and the Recall@10−Recall@5 gap opens from 0.009 to 0.092.** |
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

Detailed, executable plans exist for steps 02-16:

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
| 12 Chunking experiments | [`2026-09-12-step-12-chunking.md`](superpowers/plans/2026-09-12-step-12-chunking.md) |
| 13 Metadata filtering | [`2026-09-12-step-13-metadata.md`](superpowers/plans/2026-09-12-step-13-metadata.md) |
| 14-16 Hybrid search and RRF | [`2026-09-13-step-14-16-hybrid-rrf.md`](superpowers/plans/2026-09-13-step-14-16-hybrid-rrf.md), design: [`2026-09-13-hybrid-rrf-design.md`](superpowers/specs/2026-09-13-hybrid-rrf-design.md) |

**Steps 17-30 are deliberately unplanned.** Every one of them is a decision that rule 1 says must be made against measurements: which chunking strategy wins, whether BM25 helps this corpus, whether reranking pays for its latency, whether multi-query is worth 4x the cost. Writing those plans now would mean inventing the answers. Each plan gets written at the start of its own step, with step 11's numbers in hand.

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
