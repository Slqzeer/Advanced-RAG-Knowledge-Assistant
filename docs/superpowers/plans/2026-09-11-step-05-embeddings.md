# Step 05 — Embeddings

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Turn chunk texts into vectors, cheaply and repeatably, so that re-running the pipeline twenty times during evaluation costs nothing the second time.

**Architecture:** One module, `app/ingestion/embed.py`, exposing two functions — `embed_texts()` for documents and `embed_query()` for a single query — with a persistent on-disk cache in front of the provider call.

**Tech stack:** `openai` (new dependency) and stdlib `sqlite3` for the cache.

## Decisions

**Provider: OpenAI `text-embedding-3-small`, 1536 dimensions.** `OPENAI_API_KEY` is already in `.env.example` and the brief names OpenAI. The alternative, local `BAAI/bge-small-en-v1.5`, is free per call but drags in `sentence-transformers` and `torch` — hundreds of megabytes and a model download — to solve a cost problem the cache already solves. Verify the current model list and dimensions against the provider docs when implementing; do not trust a number written in a plan.

**A persistent cache, and it is the whole reason this step is its own step.** Steps 10-19 re-run the pipeline constantly while changing retrieval, not embeddings. Without a cache you pay for every identical chunk every time and wait minutes per run; with one, the second run is instant and offline. A single sqlite file, ~20 lines, stdlib, survives restarts and parallel reads.

Cache key is `sha256(model + "\x00" + text)`. The model goes in the key because switching models must not silently serve stale vectors — the most expensive debugging session in this project is the one where half your index is in a different embedding space.

**No `Embedder` protocol yet.** One provider, one implementation; an interface with a single implementation is speculative. The seam appears for free when a second provider is added: the two functions are already the interface.

**Batch at 100 texts per request, retry 429 and 5xx with exponential backoff**, capped at 5 attempts. This is not over-engineering — embedding a few thousand chunks *will* hit a rate limit, and a crash halfway through a 10-minute run that loses all uncached work is the failure this prevents. The cache makes the retry cheap anyway: restarting resumes from where it stopped.

**Queries and documents go through the same function.** `text-embedding-3-*` needs no instruction prefix. If step 12 or later switches to a BGE-family model, query prefixes become mandatory — which is precisely why `embed_query` exists as a separate entry point now, even though today it just delegates.

## File map

- Create `app/ingestion/embed.py` — `embed_texts()`, `embed_query()`, `EmbeddingCache`.
- Create `tests/test_ingestion_embed.py`.
- Modify `pyproject.toml`, `uv.lock` — add `openai`.
- Modify `.env.example` — document `EMBEDDING_MODEL`, `EMBEDDING_CACHE_PATH`.
- Modify `README.md`.

## Tasks

### Task 1: Add the dependency

- [ ] **Step 1**

```powershell
uv add "openai>=1.109,<2"
uv lock --check
uv run python -c "import openai; print(openai.__version__)"
```

- [ ] **Step 2: Commit.** `chore: add the openai client`.

### Task 2: The cache

- [ ] **Step 1: Write the failing tests**

1. A fresh cache returns `None` for an unknown key.
2. After `put`, `get` returns the identical vector (floats, same length, same order).
3. Same text under a different model name misses.
4. `get_many` returns a dict covering only the hits, preserving nothing about misses.
5. The cache file is created with its parent directories.
6. Reopening the cache on the same path returns previously stored vectors.

- [ ] **Step 2: Write `EmbeddingCache` in `app/ingestion/embed.py`**

One table: `CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL)`.

Store the vector as a `struct.pack(f"<{len(v)}f", *v)` blob, not JSON. A 1536-float vector is 6 KB packed and ~30 KB as JSON text, and the pack/unpack is one line each.

Methods: `get(key)`, `get_many(keys)`, `put_many(items)`. Batch the writes in one transaction. Open with `check_same_thread=False` and set `PRAGMA journal_mode=WAL`.

- [ ] **Step 3: Green.** Commit: `feat(ingestion): add a persistent embedding cache`.

### Task 3: The embedding functions

- [ ] **Step 1: Write the failing tests**, all against a fake client — no network in the test suite, ever.

1. `embed_texts` on 3 texts calls the client once and returns 3 vectors in input order.
2. **Order is preserved with a partial cache hit:** 5 texts where 2 are cached returns 5 vectors in the original order, and the client was asked for only the 3 missing ones. This is the bug worth testing: an off-by-one in the merge silently attaches the wrong vector to the wrong chunk, and nothing downstream notices except a mysteriously bad Recall@5.
3. A second call with the same texts makes no client call at all.
4. 250 texts with `batch_size=100` makes 3 client calls.
5. Duplicate texts in one input list produce one client entry and two identical output vectors.
6. Empty input returns `[]` and makes no call.
7. A client raising a rate-limit error twice then succeeding returns vectors; a client always failing raises after the attempt cap. Patch the sleep so the test is fast.
8. A response whose vector length differs from the expected dimension raises — do not let a wrong-dimension vector reach Qdrant.

- [ ] **Step 2: Write the functions**

```python
def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    cache: EmbeddingCache | None = None,
    client: Any | None = None,
    batch_size: int = 100,
) -> list[list[float]]: ...

def embed_query(text: str, **kwargs) -> list[float]: ...
```

Flow: hash every text → `cache.get_many` → collect the unique misses preserving first-seen order → batch-call the provider → validate dimensions → `cache.put_many` → assemble the output list **by hash lookup, not by zip** (that is what makes test 2 pass).

`client` and `cache` are injectable parameters defaulting to `None`, built lazily when absent. That is the entire dependency-injection story here: two default arguments, no container, no factory.

- [ ] **Step 3: Green, `mypy app` clean.** Commit: `feat(ingestion): embed texts with caching and retries`.

### Task 4: Embed the real corpus

- [ ] **Step 1: Cost check before spending anything**

```powershell
uv run python -c "from pathlib import Path; from app.ingestion.loader import load_documents; from app.ingestion.clean import clean_document; from app.ingestion.chunk import chunk_documents; docs=[d for d in (clean_document(x) for x in load_documents(Path('data/raw/fastapi'),'fastapi')) if d]; cs=chunk_documents(docs); chars=sum(len(c.text) for c in cs); print('chunks',len(cs),'chars',chars,'approx tokens',chars//4)"
```

Multiply the approximate token count by the model's published price. If the number surprises you, stop and reconsider the corpus size before spending.

- [ ] **Step 2: Embed, twice**

Run the embedding over all chunks, timing it. Then run it again. The second run must take under a second and make zero requests. Record both timings and the cache file size.

- [ ] **Step 3: Verify one vector by hand** — embed `"dependency injection"` and `"Depends"` and `"kubernetes memory limits"`, and check that cosine similarity of the first two exceeds the first and third. If it does not, the vectors are wrong and every later measurement is meaningless.

- [ ] **Step 4: Commit the timings** in the message.

### Task 5: Documentation

- [ ] Update `.env.example` (`EMBEDDING_MODEL`, `EMBEDDING_CACHE_PATH=data/processed/embeddings.sqlite`), confirm the cache path is gitignored, and update `README.md` with the model, dimension, cold/warm timings and cost. Commit: `docs: document the embedding stage`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
```

The test suite must pass with no network access and no API key set. Verify that explicitly by unsetting the key in a shell and re-running.

## Definition of done

- Vectors come back in input order even with partial cache hits — proven by test.
- The whole corpus embeds once; a second run is free and offline.
- Dimension mismatches raise instead of propagating.
- Tests never touch the network.
- Cold run time, warm run time and real cost are recorded.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| A local BGE embedder | per-call cost or offline work actually becomes a constraint |
| An `Embedder` protocol | a second provider exists |
| Async / concurrent batches | embedding time is the bottleneck you are trying to fix |
| Matryoshka dimension truncation | index size or recall trade-offs matter |
| Cache eviction | the sqlite file gets uncomfortably large |
