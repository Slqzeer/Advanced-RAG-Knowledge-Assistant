# Step 07 — Similarity Search

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Ask a question in natural language and get back the `top_k` chunks that should answer it, with scores. Retrieval only — no LLM in this step.

**Architecture:** `app/retrieval/search.py` with one function, `search()`. It is the seam the rest of the project is built on: step 08 calls it, step 11 benchmarks it, steps 14-19 replace its internals without changing its signature.

**Tech stack:** nothing new.

## Decisions

**The signature is designed once, now, for everything that comes later.**

```python
def search(query: str, *, top_k: int = 5, source: str | None = None, ...) -> list[ScoredChunk]
```

Hybrid search, RRF, reranking and multi-query all eventually live behind this call. Getting the return type right now — a chunk plus a score plus a rank — means eight later steps are internal changes instead of refactors across the codebase. This is the one place in Phase 1 where thinking one step ahead is cheaper than not.

**`ScoredChunk` wraps a `Chunk` rather than subclassing it.** A score is not a property of a chunk; it is a property of a chunk *with respect to a query*. Also, step 17's reranker produces a *second* score for the same chunk, and `rerank_score` as a sibling field is obvious, where a mutated `score` is a debugging trap.

**`rank` is stored explicitly, starting at 1.** Every metric in step 11 (MRR, NDCG, Recall@K) is a function of rank. Recomputing it from list position in four places is how an off-by-one ends up in a published benchmark number.

**No caching here.** Step 23 adds it, deliberately, once there is a latency number to improve. A retrieval cache added now would silently hide the latency changes steps 14-19 are meant to measure.

**Scores are returned raw, as the engine reports them.** No normalising, no rescaling. Cosine scores from Qdrant are comparable across queries for this collection; invented normalisation is a layer that lies.

## File map

- Create `app/models/chunks.py` addition — `ScoredChunk`.
- Create `app/retrieval/search.py` — `search()`, `chunk_from_payload()`.
- Create `scripts/search.py` — a tiny CLI for eyeballing results.
- Create `tests/test_retrieval_search.py`.
- Modify `README.md`.

## Tasks

### Task 1: `ScoredChunk`

- [ ] **Step 1: Failing test** — `ScoredChunk` exposes `chunk`, `score: float`, `rank: int` (≥1), and `rerank_score: float | None = None`; `rank=0` is rejected.

- [ ] **Step 2: Implement** in `app/models/chunks.py`, next to `Chunk`.

- [ ] **Step 3: Green.** Commit: `feat(models): add ScoredChunk`.

### Task 2: The search function

- [ ] **Step 1: Failing unit tests**, with a fake Qdrant client and a fake embedder.

1. `search("x", top_k=3)` returns 3 `ScoredChunk`s with `rank` 1, 2, 3 in that order.
2. Scores are passed through unchanged, in descending order.
3. The query is embedded exactly once per call.
4. `source="fastapi"` produces a `query_filter` on the `source` payload key; `source=None` produces no filter.
5. A payload missing an optional key (`url`, `section`) maps to `None` instead of raising — Qdrant drops null payload values, so this *will* happen against a real collection.
6. A payload missing a required key raises a clear error naming the key, not a `KeyError` from deep inside pydantic.
7. An empty or whitespace-only query raises `ValueError` before any embedding call is made. Embedding the empty string returns a valid vector that retrieves plausible-looking garbage; failing fast is the only honest behaviour.
8. `top_k=0` raises.

- [ ] **Step 2: Write `app/retrieval/search.py`**

```python
def chunk_from_payload(payload: Mapping[str, Any]) -> Chunk: ...

def search(
    query: str,
    *,
    top_k: int = 5,
    source: str | None = None,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
    embedder: Callable[[str], list[float]] | None = None,
) -> list[ScoredChunk]: ...
```

Flow: validate the query → `embed_query` → `client.query_points(collection_name=..., query=vector, query_filter=..., limit=top_k, with_payload=True)` → map each hit to `ScoredChunk(chunk=chunk_from_payload(hit.payload), score=hit.score, rank=i)`.

`client` and `embedder` are injectable defaults, as in step 05. That is what makes every test above run with no server and no API key.

Check the installed client's search API against its own docs; `query_points` is the current universal entry point and the older `search` method is deprecated.

- [ ] **Step 3: Integration test** under the `requires_qdrant` marker: index three chunks with deliberately distinct content (one about dependency injection, one about Docker memory limits, one about Markdown formatting) into a throwaway collection, query `"how does dependency injection work"`, and assert the first chunk ranks 1. If it does not, something upstream is broken and no later measurement means anything.

- [ ] **Step 4: Green.** Commit: `feat(retrieval): add dense similarity search`.

### Task 3: Look at real results

- [ ] **Step 1: Write `scripts/search.py`** — `uv run python scripts/search.py "question" --top-k 5 [--source fastapi]`. Print rank, score, `document_id`, `section`, and the first 200 characters of each chunk.

- [ ] **Step 2: Run the queries that matter**

```powershell
uv run python scripts/search.py "Comment fonctionne l'injection de dependances dans FastAPI ?"
uv run python scripts/search.py "How does dependency injection work in FastAPI?"
uv run python scripts/search.py "HTTPException 422"
uv run python scripts/search.py "Depends"
uv run python scripts/search.py "how to protect an API"
```

- [ ] **Step 3: Write down what you observe.** Specifically:
  - Does the French question retrieve the English documents? (Cross-lingual behaviour of the embedding model, measured rather than assumed.)
  - Does `HTTPException 422` retrieve the right page? **It probably does not, and that failure is the entire justification for steps 14-16.** Record it now, verbatim, so the hybrid-search step has a real before-and-after instead of a claim from a blog post.
  - Are the top results from the same document? That is a chunking/diversity observation for step 12.

Put these observations in the commit message and in the README. They are the project's first findings.

- [ ] **Step 4: Commit.** `feat(retrieval): add a search CLI`.

### Task 4: Documentation

- [ ] Update `README.md`: current state, roadmap, the search command, and a short "Premiers constats" section with the observations above. Commit: `docs: record the first retrieval observations`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
docker compose up -d qdrant --wait ; uv run pytest -m requires_qdrant
uv run python scripts/search.py "How does dependency injection work in FastAPI?"
```

## Definition of done

- `search()` returns ranked, scored chunks and its signature is ready for hybrid search and reranking.
- Missing optional payload fields do not crash; missing required ones fail clearly.
- Empty queries are rejected before any API call.
- Unit tests run with no Qdrant and no API key.
- The `HTTPException 422` behaviour is recorded, whatever it turns out to be.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| BM25 and hybrid fusion | steps 14-16 |
| Reranking | step 17 |
| Query rewriting | step 18 |
| Retrieval caching | step 23, after latency is measured |
| MMR / diversity re-ordering | the eval set shows single-document result crowding |
| Score normalisation | two retrievers with incomparable scores need fusing (and RRF needs ranks, not scores) |
