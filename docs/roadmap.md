# Global Plan

The single place that answers: where is this project, what comes next, and why.

- **Brief:** [`information.md`](../information.md) — the original 30-step learning order.
- **Public front page:** [`README.md`](../README.md) (French) — phases 0-20.
- **Step plans:** [`docs/superpowers/plans/`](superpowers/plans/) — one executable plan per step.

## Current state

**Phase 0 and steps 02-20 are done — `v1.0` is tagged. Step 21 (RAGAS) is next,
and step 20 hands it a specific question: its own headline gain is a retrieval
number, and the one question it made worse proves that number is blind to what
step 21 measures.**

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
| Cross-encoder reranking | `app/retrieval/rerank.py` — `RERANKERS` keyed `flashrank`\|`cohere`, a backend returning only `(index, score)` pairs while `rerank()` owns ordering, tie-breaking and the `score`/`rerank_score` split; `search(rerank=)` orthogonal to `mode`, reachable from all three scripts and `answer_question()`. The ceiling was measured first: the dense d30 pool holds **0.884** Recall@30 against 0.785 at rank 5. Eight measured rows. Best row `rerank-flashrank-dense-d30`: Recall@5 **0.779** (+0.002 on the 0.776 baseline), capture **−0.067**, p50 **1 141 ms** against a 400 ms budget, `code` **−0.100**. All three clauses of the pre-registered rule fail, so `RERANK_MODEL` stays empty. The Cohere rows are unmeasured — no API key |
| Query transforms | `app/retrieval/transform.py` — `TRANSFORMS` keyed `rewrite`\|`multi`, with parsing, capping, de-duplication and the fallback to the original query owned once for both; `search(transform=)` orthogonal to `mode` and `rerank`, fanning N queries through the chosen retriever and fusing with the existing `rrf()`; `contextualize()` deliberately outside the registry and above `search()`, in `answer_question(history=)`. Reachable from all three scripts. No new dependency — the first step since 13 to add none. Six measured rows. Best row `multi-n2-dense`: Recall@5 **0.765** against the 0.776 baseline (**−0.011**), but Recall@10 **0.807** (+0.022) and MRR **0.867** (+0.057, the project's best). `rewrite-standalone` loses outright at **0.684**, with `exact` **−0.183**. Clause 1 of the pre-registered rule fails (clauses 2 and 3 pass), so `QUERY_TRANSFORM` stays empty |
| Conversational fixture | `data/eval/conversations.jsonl`, `scripts/benchmark_conversations.py`, `app/evaluation/dataset.py` — `EvalConversation` subclasses `EvalQuestion` so it inherits every validator of the frozen set, and `load_dataset(model=)` avoids a second loader. Ten referential follow-ups built to one rule: the ground truth must be unreachable from the follow-up alone. Two rows: `conv-raw` Recall@5 **0.100** against `conv-rewrite` **0.600** (**+0.500**), MRR 0.127 → 0.567. The frozen 50-question set does not change |
| Contextual compression | `app/generation/compress.py`, `scripts/benchmark_answers.py` — `COMPRESSORS` keyed `embedding`, where an entry only *scores* units and `compress()` owns splitting, the greedy budget, document-order reassembly with a `[…]` marker and the frozen-model rebuild; `sentence_spans()` promoted out of `chunk.py` so the compressor splits exactly as the indexer does, fenced code included; it runs in `answer_question()` **between** `search()` and `build_context()`. No new dependency, no tokeniser — the budget is characters, the published number is `usage.prompt_tokens`. The budget was measured first, not estimated: the top_k=5 context is 4 000 chars at p50, of which 312 are headers, so `COMPRESS_BUDGET_CHARS` is **3 688**. Six retrieval rows, six answer rows. Best row `compress-embedding-d20`: Recall@context **0.814** against a 0.721 control and a 0.836 ceiling, no per-category regression, `multi_doc` **0.552 → 0.792**, refusal **0.316 → 0.211**, +79 ms p50. All three clauses of the pre-registered rule pass — the first time in five steps — so `COMPRESS_METHOD=embedding` and `COMPRESS_CANDIDATES=20` |
| Metadata filtering | `app/ingestion/loader.py`, `app/retrieval/search.py` — `doc_type` derived from the corpus layout, indexed, filterable through a generic `filters=` mapping; per-facet recall and an oracle run recording the ceiling on facet routing at **+0.000 Recall@5** |

The loop is closed, verified and measured: a question goes in, a grounded answer with resolved citations comes out of `scripts/ask.py`, and 38 answerable questions give it a score. No HTTP endpoint yet — that is step 25.

Four things later steps own:

- **`HTTPException 422` is resolved as a finding, not as a fix.** Steps 14-16 gave the retriever the missing capability — BM25's rank-1 hit for `HTTPException 422` does contain the literal token, and `--mode hybrid` surfaces `reference/exceptions` at rank 2 — and the answer is still an honest "I do not know". The four chunks in the corpus containing `422` are two release notes and an OpenAPI JSON example; none of them explains what the code means. The refusal is correct. No later step owns this: the target was chosen at step 07, before step 11 said where the gap actually was, and `exact` was already the strongest category at 0.892.
- **The headroom is real and a generic cross-encoder does not capture it.** Step 17 consumed step 16's `Recall@10 − Recall@5 = 0.092` precondition and replaced it with measurement. The pools do hold the documents: dense d30 goes 0.785 → **0.884** at Recall@30 (headroom 0.099), hybrid d30 0.768 → 0.890 (0.123), hybrid d50 0.743 → **0.917** (0.173). FlashRank recovers none of it — best capture **+0.018**, best absolute Recall@5 **0.779** against a 0.806 bar, for 1 141 ms. Two further facts steps 18-19 should carry: **Recall@20 equals Recall@30 in every pool**, so depth 20 is where these pools stop finding anything new and a 30-deep pool is 50 % more cross-encoder work for nothing reachable; and the reranker **moves precision between categories rather than adding any** — `conceptual` +0.083, `code` −0.100, total +0.002 — because `ms-marco-MiniLM-L-12-v2` is trained on natural-language web passages and promotes prose over code-bearing chunks. **Steps 18-19 consumed this and answered it: the gap is not a ranking problem at all.** Multi-query expansion — a different pool, built from up to five paraphrases rather than one query — gains **+0.057 MRR and +0.022 Recall@10 while losing 0.011 Recall@5**. It reorders what the pool already held and adds nothing new to it, which is the same shape as the reranker's result arrived at from the opposite direction. And reranking a wider multi-query pool costs `code` **−0.100 again**, so the category regression was never the pool's composition: `ms-marco-MiniLM-L-12-v2` buries code-bearing chunks however they are presented. A domain-adapted or code-aware reranker remains the only version worth retrying. **Step 20 consumed this and finally converted it.** If no ranking stage can reach rank 6-20, the remaining lever is making a chunk cheaper so more of the pool fits: a 20-chunk pool compressed into the budget five whole chunks occupied moves Recall@context **0.721 → 0.814** against a 0.836 ceiling, taking 81 % of what was reachable, for +79 ms and no category regression. `multi_doc`, which carried most of the gap, goes **0.552 → 0.792**. The `code` regression the reranker caused does *not* recur at document level — `code` is flat at 0.800 — but see the next bullet, because it recurs somewhere the retrieval metric cannot see.
- **Generation is ~95 % of the latency.** 1.5-3.7 s per question against ~35 ms of warm retrieval. Any latency work before step 23's cache would be optimising the wrong 5 %.
- **A score threshold cannot carry refusal.** The 7 out-of-corpus questions score 0.305-0.459, the answerable ones 0.341-0.664. Step 22 needs something other than a floor.
- **A score threshold cannot even be shared across runs.** Every multi-query row reports `abstention_rate` **1.000** against a 0.143 baseline, and that is an artefact, not a finding: RRF replaces every cosine with `1/(k+rank)` around 0.03, so all 45 questions fall under the 0.35 floor, answerable ones included. `rrf()`'s own docstring already said this; steps 18-19 are where it bites a recorded number. The one transform row that keeps cosines is `rewrite-standalone`, which skips fusion on its single query, and there abstention moves 0.143 → **0.286** for real — a rewritten query scores lower against everything, the threshold included. Step 22 inherits both the real number and the fact that any refusal rule must be defined per scoring scheme, not per project.
- **Step 21 inherits a gap that step 20 proved is real, on a named question.** Step 20's own headline is a retrieval number, and it also measured the thing that number cannot see. `q018` — *"How do I write a test that calls my own endpoints?"* — regressed from a cited answer to a refusal while its category's Recall@context did not move at all: the right documents were in the context both times, but the uncompressed context carried three code blocks and the compressed one carried none. A fenced block is one unit by design (half a fence is broken code), and a greedy per-character budget lets cheap prose outbid a 900-char example every time. So the answer is: **yes, the gap between Recall@context and answer quality is real, and it is not noise — it has a mechanism.** That is exactly what RAGAS exists to measure, and step 21 should also test scoring a unit by relevance **per character** rather than relevance alone. Two further inputs: the refusal rate, blunt as it is, moved the right way (0.316 → 0.211) and agreed with the retrieval number, so it is a usable cheap proxy; and both `--unanswerable` arms abstain 7/7 either way, so a wider compressed pool did not make the model over-answer.
- **The published `recall@k` counts distinct documents, not chunks.** `run_benchmark` deduplicates chunks to documents over the whole retrieved list and slices `[:k]` afterwards, so how deep into the chunk pool those k documents reach is a function of `--top-k`. Step 20 found this by measuring a control that disagreed with the plan by 0.064 and ruling out retrieval first (the top-5 chunks are byte-identical at limit 5 and limit 30). Nothing was changed: rewriting the metric now would invalidate every row in `results.jsonl`, and rows stay comparable **to each other**. The rule this leaves behind is narrow and must be kept: **never compare two rows run at different `--top-k`.**
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
| 17 | Phase 6 — Reranking | `v0.7` | Top-30 recall, top-5 precision, measured latency cost. **Done — the pool holds 0.884 Recall@30 against 0.785 at rank 5, and FlashRank captures none of it: Recall@5 0.779 (+0.002), p50 1 141 ms, `code` −0.100; `RERANK_MODEL` stays empty.** |
| 18-19 | Phase 7-8 — Query transforms | `v0.8`-`v0.9` | "et pour docker ?" resolves against conversation history. **Done — resolution works and is the step's one clear win (Recall@5 0.100 → 0.600 on a ten-conversation fixture), but neither `rewrite` (0.684) nor `multi` (best 0.765) beats the 0.776 baseline; `QUERY_TRANSFORM` stays empty.** |
| 20 | Phase 9 — Compression | `v1.0` | Same answer quality, fewer context tokens. **Done, and the brief was wrong twice over: the tokens went *up* 11 %, and the win is recall, not cost.** A 20-chunk pool compressed into the budget 5 whole chunks occupied moves Recall@context 0.721 → **0.814** (ceiling 0.836), `multi_doc` 0.552 → **0.792**, refusal 0.316 → **0.211**, +79 ms. First pre-registered rule met in five steps; `COMPRESS_METHOD=embedding`. |
| 21 | Phase 11 — RAGAS | `v1.3` | Faithfulness and answer relevance, not just retrieval. |
| 22 | Phase 12 — Guardrails | `v1.6` | Refuses when retrieval is weak; survives injected instructions in documents. |
| 23 | Phase 13 — Cache | `v1.4` | Cache-hit latency and cost deltas. |
| 24 | Phase 15 — Observability | `v1.5` | Per-query trace: scores, prompt, tokens, latency, cost. |
| 25 | Phase 14 — FastAPI | `v1.0` | `POST /query`, `POST /documents`, `GET /health`, `GET /metrics`. |
| 26-28 | Phase 16-17 — Docker, tests, CI | `v1.0` | `docker compose up` and a green pipeline with RAG regression gates. |
| 29-30 | — | — | Benchmark dashboard and the README that tells the whole story. |

**The tag column past step 20 is provisional.** It was written to track README phase
numbers rather than step order, which is why it reads `v1.3` at step 21 and `v1.0` at
step 25. Step 20 shipped as **`v1.0`** — the next tag after the `v0.9` actually in the
repo — so that number is now taken, and each remaining tag is assigned when its step
lands rather than promised here.

## Plans written so far

Detailed, executable plans exist for steps 02-20:

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
| 17 Cross-encoder reranking | [`2026-09-13-step-17-reranking.md`](superpowers/plans/2026-09-13-step-17-reranking.md), design: [`2026-09-13-reranking-design.md`](superpowers/specs/2026-09-13-reranking-design.md) |
| 18-19 Query transforms | [`2026-09-13-step-18-19-query-transforms.md`](superpowers/plans/2026-09-13-step-18-19-query-transforms.md), design: [`2026-09-13-query-transforms-design.md`](superpowers/specs/2026-09-13-query-transforms-design.md) |
| 20 Contextual compression | [`2026-09-13-step-20-compression.md`](superpowers/plans/2026-09-13-step-20-compression.md), design: [`2026-09-13-compression-design.md`](superpowers/specs/2026-09-13-compression-design.md), transcripts: [`transcripts-step-20.md`](superpowers/plans/transcripts-step-20.md) |

**Steps 21-30 are deliberately unplanned.** Every one of them is a decision that rule 1 says must be made against measurements: whether a score floor can carry refusal at all, what a cache key has to include. Writing those plans now would mean inventing the answers — steps 12-20 each answered their own question only by running it, four of them answered 'no', and step 20 answered 'yes' only after two of its own pre-registered numbers turned out to be measuring the wrong thing. Each plan gets written at the start of its own step, with step 11's numbers in hand.

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
├── retrieval/bm25.py         # 14  hand-rolled Okapi BM25 index
├── retrieval/rerank.py       # 17  cross-encoder rescoring of a shortlist
├── retrieval/transform.py    # 18-19  query rewriting and expansion
├── generation/context.py     # 08  chunks -> numbered context block
├── generation/llm.py          # 08  single LLM call
├── generation/answer.py      # 08  retrieve -> context -> prompt -> Answer
├── generation/citations.py   # 09  parse and validate [n] references
├── generation/compress.py    # 20  ranked chunks -> shorter ranked chunks, to a budget
├── evaluation/dataset.py     # 10  load and validate the eval set
├── evaluation/metrics.py     # 11  recall@k, precision@k, mrr, ndcg, hit rate
├── evaluation/benchmark.py   # 11  run the set, emit a markdown row
└── api/                      # 25  FastAPI, last
scripts/
├── fetch_corpus.py           # 02
├── index_corpus.py           # 06
├── ask.py                    # 08
├── benchmark.py              # 11
├── benchmark_conversations.py # 18  the conv-raw / conv-rewrite delta
└── benchmark_answers.py      # 20  refusal rate, context chars, real prompt_tokens
```

`api/` is last on purpose. A CLI proves the pipeline works; HTTP is packaging.
