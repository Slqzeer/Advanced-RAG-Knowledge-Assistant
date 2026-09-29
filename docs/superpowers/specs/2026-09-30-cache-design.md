# Cache Design

Step 23 of [`docs/roadmap.md`](../../roadmap.md); Phase 13 of
[`information.md`](../../../information.md); tag assigned when the step lands.

## Purpose

The brief asks for three caches (embedding, retrieval, LLM response) behind Redis,
and the step map asks for one deliverable: **cache-hit latency and cost deltas**.

Step 23 builds an answer cache in two layers. The **exact** layer serves a repeated
question under identical settings. The **semantic** layer serves a *paraphrase* of
a question already answered. The semantic layer is the only part of this step that
can be wrong: a false hit hands one question's answer to a different question. So
it ships only if a pre-registered rule, read on a fixture written before the
sweep, says it may.

As at steps 13, 16, 17, 19, 20 and 22, the deliverable is the measurement. A step
23 that concludes "no threshold separates paraphrases from near-misses on this
corpus" ships with the same care as one that turns the semantic layer on.

## Facts this design is built on

- **The embedding cache already exists.** `app/ingestion/embed.py` keeps a sqlite
  cache keyed by `(model, text)`. It covers corpus chunks, query embeddings
  (`embed_query()`), and the compressor's per-sentence embeddings
  (`compress.py::_default_embedder`). Cold 404 s, warm 0.12 s. Nothing to build.
- **Retrieval is ~5 % of latency.** ~35 ms warm, plus +79 ms p50 for compression,
  against 1.5-3.7 s of generation. A retrieval cache would optimise the wrong 5 %
  (roadmap, "Four things later steps own"). Not built.
- **Generation is ~95 % of latency and all of the per-query spend.** That is what
  the answer cache removes on a hit.
- **Similarity does not separate near-misses here.** At step 22, the 7
  unanswerable questions scored 0.335-0.394 at rank 1, while answerable ones went
  as low as 0.306. That is a retrieval cosine, not a question-to-question cosine,
  but it is the reason the semantic layer gets an adversarial fixture and a
  zero-false-hit rule rather than a threshold picked by eye.
- **Temperature 0 is not deterministic on this generator.** Step 22 sent 38
  byte-identical contexts twice: 30 of 38 answers differed and 4 refusals flipped.
  A cache hit returns *one* sample, frozen. That is the intended behaviour and is
  stated, not hidden.
- **A cache must never reach a benchmark.** OmniRoute's response cache replayed 80
  of one arm's 90 answers until every generation call sent `X-OmniRoute-No-Cache`
  (`06db6d0`). This design makes the bypass structural, not a flag.
- **Redis 8 ships vector search in the open-source image.** Vector sets
  (`VADD`/`VSIM`, with `FILTER` on JSON attributes) are built in since 8.0, so one
  container serves both layers. No Redis Stack, no `redisvl`.

## Scope

In:

- `app/generation/cache.py`: `cached_answer()` and `clear_cache()`.
- `Answer.cache: Literal["exact", "semantic"] | None`.
- Three settings: `redis_url`, `cache_ttl_s`, `cache_semantic_threshold`.
- A pinned `redis:8.x` service in `compose.yaml`.
- The `redis` (redis-py) dependency.
- `data/eval/cache_pairs.jsonl`: an 83-pair fixture.
- `scripts/benchmark_cache.py`: the offline sweep and three replay arms.
- `scripts/ask.py` goes through `cached_answer()`.
- `scripts/index_corpus.py` clears the cache after a successful upsert.

Out:

- **A retrieval cache.** ~100 ms at most; see above.
- **Semantic lookups on follow-ups.** See "Follow-ups" below.
- **`redisvl` and its `SemanticCache`.** Rule 2: hand-roll before you framework.
  Its threshold and key rules would be its own, not ones this project measured.
- **API wiring.** Step 25 calls `cached_answer()`; nothing HTTP lands here.

## Alternatives considered

**Where the cache sits.**

1. **A wrapper around `answer_question()` (chosen).** Benchmarks already call
   `answer_question()` directly, so they bypass the cache by construction: no
   `cache=False` to forget. `answer_question()`, whose docstring calls it
   "deliberately boring and deliberately short", is not touched.
2. **Inside `answer_question()`, after `contextualize()`.** It would give follow-ups
   semantic lookups on the resolved query. Rejected: every benchmark would need an
   explicit bypass, which is the footgun step 22 already paid for.
3. **At the HTTP layer, step 25.** Nothing to measure until then, and the step map
   puts the cache first.

**Where the cache lives.**

1. **Redis 8 for both layers (chosen).** Native TTL and eviction, vector sets for
   the semantic layer, shared by step 25's workers. Rule 3 already reserves `redis`
   for this step.
2. **Qdrant for semantic, sqlite for exact.** No new dependency, but TTL and
   eviction are hand-written, and it ignores the brief's Redis.
3. **sqlite for both, brute-force cosine.** Smallest, but the scan grows with the
   cache and there is no TTL. A poor base for step 25.
4. **`redisvl`'s `SemanticCache`.** Least code; rejected under rule 2.

### Follow-ups

Follow-ups (a call with `history`) get exact lookups only. **Assumption, not
measurement:** a semantic hit on a follow-up needs the history resolved first (one
LLM call, ~1-2 s) before we know whether it hits, and we expect repeated follow-ups
to be rare. There is no traffic to measure that on yet; step 24's per-query trace
can count it, and the corner is marked with a `ponytail:` comment.

## Architecture

```text
cached_answer(question, history, **kwargs)
  │
  ├─ redis_url empty ────────────────────────────► answer_question(...)
  │
  ├─ exact key  = sha256(canonical JSON)
  │    GET rag:cache:answer:<key> ── hit ────────► Answer(cache="exact")
  │
  ├─ no history and threshold set:
  │    VSIM rag:cache:qvec FP32 <query vector> WITHSCORES COUNT 1
  │         FILTER '.fp == "<fingerprint>"'
  │    score ≥ threshold and GET answer:<sha> ────► Answer(cache="semantic")
  │    score ≥ threshold, answer expired ────────► VREM, treat as a miss
  │
  └─ miss ──► answer_question(...)
               refusal is None ──► SET answer:<key> EX ttl
                                   VADD qvec NOQUANT <vector> <key> SETATTR {"fp": ...}
```

### The key

**Exact key** = `sha256` over one canonical JSON object (sorted keys, no
whitespace):

- `question`, with outer whitespace stripped and inner runs collapsed to one space.
  **Case is kept**: `Depends` and `depends` are different things in these docs.
  Paraphrases are the semantic layer's job, not the normaliser's.
- `history`, verbatim, truncated to `history_turns` exactly as `contextualize()`
  truncates it.
- **All of `settings.model_dump()` minus a denylist**: secrets (`*_api_key`),
  local paths (`*_path`, `corpus_dir`), `qdrant_url`, `redis_url`,
  `cache_ttl_s`, `cache_semantic_threshold`, and `judge_*`. A denylist rather
  than an allowlist: a setting added at a later step changes the key
  automatically, and forgetting one costs a cache miss, never a stale answer.
- The resolved per-call kwargs: `top_k`, `mode`, `transform`, `transform_n`,
  `rerank`, `rerank_candidates`, `compress`, `compress_candidates`,
  `compress_budget`, `filters`, `max_context_chars`, `strict`,
  `include_contexts`. `include_contexts` is in the key because it changes what the
  returned `Answer` carries.

**Settings fingerprint** = `sha256` of the same object without `question` and
`history`. It is the `fp` attribute on every vector-set element, and `VSIM`'s
`FILTER` requires it to match: a paraphrase can only hit an answer produced under
identical settings.

### Invalidation

1. **On re-index.** `scripts/index_corpus.py` calls `clear_cache()` after a
   successful upsert: the one place the index changes. If Redis is unreachable it
   prints a warning and exits 0; indexing must not fail because of a cache.
2. **TTL**, 7 days by default (`CACHE_TTL_S`), native to Redis.
3. **Refusals are never stored**, `no_context` or `model_declined`, so a refusal
   cannot outlive the re-index that would fix it.

No per-query Qdrant call fingerprints the index. Clearing on re-index covers it,
and a lookup that asks Qdrant first spends part of the latency it exists to save.

### Redis

`compose.yaml` gains a `redis` service: `redis:8.x` pinned to an exact tag (as
Qdrant is), loopback-bound, healthcheck `redis-cli ping`, and
`--maxmemory 256mb --maxmemory-policy volatile-lru --save "" --appendonly no`.
It is a cache: losing it on restart means cold again. `volatile-lru` evicts only
keys with a TTL, so the vector set (no TTL) is never evicted wholesale.

All keys sit under the `rag:cache:` prefix:

| Key | Type | Content |
|---|---|---|
| `rag:cache:answer:<sha>` | string, `EX = cache_ttl_s` | `Answer.model_dump_json()` |
| `rag:cache:qvec` | vector set | element `<sha>`, the query embedding, attribute `{"fp": "<fingerprint>"}` |

- **`NOQUANT` on every `VADD`.** Vector sets quantise to int8 by default, which
  would make the scores Redis compares differ from the cosines the offline sweep
  measured.
- **Lazy cleanup.** Vector-set elements have no per-element TTL. When `VSIM`
  returns a `<sha>` whose answer has expired, the lookup is a miss and the element
  is `VREM`'d on the spot. No sweeper.
- **`VSIM`'s score scale is checked, not assumed.** A `requires_redis` test inserts
  known vectors and pins the mapping from `VSIM` score to cosine, so
  `cache_semantic_threshold` means the same thing in the sweep and in Redis.
- `clear_cache()` = `DEL rag:cache:qvec` plus `SCAN MATCH rag:cache:answer:*` and
  `DEL`. Nothing outside the prefix is touched.

### `app/generation/cache.py` (new)

```python
def cached_answer(
    question: str,
    *,
    history: Sequence[Mapping[str, str]] | None = None,
    settings: Settings | None = None,
    client: Redis | None = None,
    embedder: Callable[[str], list[float]] | None = None,
    answerer: Callable[..., Answer] = answer_question,
    **kwargs: Any,
) -> Answer: ...

def clear_cache(client: Redis | None = None, *, settings: Settings | None = None) -> int: ...
```

`client`, `embedder` and `answerer` are injectable, the same pattern as
`retriever` and `llm` in `answer_question()`. `embedder` defaults to
`embed_query()`, whose sqlite cache already holds the query vector that dense
retrieval computes, so the semantic lookup adds no embedding spend.

On a hit, `latency_ms` is the lookup's own time, not the stored generation's, so
the measurement reads straight off the returned `Answer`.

### `app/models/answers.py`

One field: `cache: Literal["exact", "semantic"] | None = None`. `None` means the
answer was generated on this call.

### Configuration

Following the repo's "empty means off" idiom:

| Setting | Default | Meaning |
|---|---|---|
| `redis_url` | `""` | Empty: no cache, no connection attempt. Tests and every existing command behave exactly as today. |
| `cache_ttl_s` | `604800` | 7 days. |
| `cache_semantic_threshold` | `None` | `None`: the semantic layer is off. Set only if the sweep passes the rule below. |

### Failure handling

The cache is an optimisation. It must never fail a query.

- A Redis error on lookup or store (`ConnectionError`, `TimeoutError`) adds a
  warning to `Answer.warnings` and falls through to `answer_question()`.
- The socket timeout is 200 ms, so a dead Redis costs at most that per call, never
  a hang.
- A stored entry that fails to deserialise (for example, `Answer`'s schema changed
  since it was written) is a miss, and its key is deleted.

## Dependencies

`redis` (redis-py), added with `uv lock` in the same commit. It ships vector-set
commands; those are used rather than raw `execute_command`.

## Measurement

### The fixture

`data/eval/cache_pairs.jsonl`, one pair per line:
`{id, anchor_id, anchor, probe, kind}`, where `kind` is `paraphrase`,
`near_miss` or `unanswerable`. It loads through `load_dataset(model=)` with its
own pydantic model, as `EvalConversation` does.

- **38 `paraphrase`**: one per answerable question. Same intent and same
  ground-truth documents, worded to share as few content words with the anchor as
  reads naturally.
- **38 `near_miss`**: one per answerable question. Shares at least half its
  content words with the anchor, but the correct answer is different (a different
  document, or a different section of the same one).
- **7 `unanswerable`**: each out-of-corpus question, paired with the answerable
  anchor it is closest to. The adversarial case, on purpose.

Written by hand, reviewed by the user, and committed on its own **before**
`benchmark_cache.py` exists, so the history shows the fixture came first. Not
generated by a model: an LLM's paraphrases tend to reuse the original's wording,
which makes should-hit pairs easy and the measured recall optimistic.

### The sweep (offline, no generation)

`scripts/benchmark_cache.py --sweep` embeds every anchor and probe through
`embed_query()` (sqlite cache, about $0.0001) and computes plain cosine per pair.
It reports:

- the paraphrase cosine distribution against the must-miss distribution
  (`near_miss` + `unanswerable`, 45 pairs), and their overlap;
- the **candidate threshold: the highest must-miss cosine + 0.02**. The margin is
  pre-registered. The lowest threshold with zero false hits would sit exactly on
  this fixture's edge, which is fitting the threshold to the data it is judged on.

### Rule S: the semantic layer

Read at the candidate threshold:

1. **0 of 45** must-miss pairs hit, and
2. **≥ 50 %** of the 38 paraphrases hit (≥ 19).

Both clauses pass: `CACHE_SEMANTIC_THRESHOLD` is set to the candidate. Either
fails: it stays `None`, and the published finding is how far the two
distributions overlap.

### Replay arms

1. **`cache-cold`**: `clear_cache()`, then the 38 answerable anchors through
   `cached_answer()`. All misses: p50/p95 latency, `usage` tokens per answer.
2. **`cache-exact`**: the same 38 again.
3. **`cache-semantic`**, only if Rule S passed: the 38 paraphrases. Hit count,
   p50/p95, and a pair-for-pair check that Redis's hits equal the sweep's
   prediction. That catches a `VSIM` score scale or HNSW approximation that
   differs from plain cosine.

Reported for each arm: hits, p50/p95 latency, tokens per miss, tokens saved per
hit, and dollars at the generation model's list price. Through OmniRoute the real
bill may differ, and the README says so. Rows go into `data/eval/results.jsonl`
with the git commit, like every other benchmark. Generation spend: 38 calls in arm
1; arms 2 and 3 add none.

### Rule E: the exact layer

On `cache-exact`: **38 of 38** hits, answer text byte-identical to `cache-cold`,
hit p50 **< 50 ms**. Expected to pass trivially. It is pre-registered so that a
broken key shows up as a failed row, not as an assumption.

## Acceptance

- Quality gates pass: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.
- The fixture is committed before the sweep script.
- The sweep, Rule S's verdict, and all run arms are in `results.jsonl`, the README
  (French) and the roadmap, regressions and a failed Rule S included.
- `docs/roadmap.md` is updated: current state, the step-map row, a bullet on what
  later steps inherit (including the follow-up assumption, in the wording above,
  handed to step 24 to count), and the plans table.
- The tag is the next free one when the step lands.

## Testing

**Default suite** (no Redis, no key, no spend):

- **Fingerprint**, parametrized over `Settings.model_fields`: changing any
  non-denylisted field changes the key; changing a denylisted one does not. A
  setting added later is covered without editing the test.
- The question's whitespace is collapsed and its case kept; history is truncated
  to `history_turns`; each per-call kwarg changes the key, `include_contexts`
  included.
- `redis_url=""`: the stub `answerer` is called, and no connection is attempted.
- **Dead Redis** (an unused local port, real redis-py): the answer comes back, it
  carries a warning, and the call returns within the timeout plus a small margin.
- Fixture validators: 38/38/7 by `kind`, every `anchor_id` exists in
  `questions.jsonl`, and every `unanswerable` anchor is an answerable question.

**`@pytest.mark.requires_redis`**, against the compose container, skipped when it
is not up. No fake Redis: faking vector sets would mean re-implementing the thing
under test.

- A miss stores, the next call is an exact hit with identical text, and
  `cache == "exact"`.
- Neither refusal kind is stored.
- A semantic hit requires a matching `fp`; the same probe under different settings
  misses.
- An expired answer with a leftover vector element is a miss, and the element is
  `VREM`'d.
- `clear_cache()` removes the prefix and nothing else.
- **`NOQUANT` and the score scale**: known vectors in, and `VSIM`'s score maps to
  cosine as the threshold assumes. This test gates Rule S's threshold before
  anything uses it.
