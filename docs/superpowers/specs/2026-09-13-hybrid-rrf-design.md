# Hybrid Search and Reciprocal Rank Fusion Design

Steps 14, 15 and 16 of [`docs/roadmap.md`](../../roadmap.md); Phase 5-6 of
[`information.md`](../../../information.md); tags `v0.5` and `v0.6`.

## Purpose

Add a lexical retriever alongside dense retrieval, fuse the two with Reciprocal
Rank Fusion, and measure whether the combination beats dense retrieval on this
corpus. The deliverable is the measurement, not the retriever: a result of "BM25
does not help here" is shipped with the same care as a result of "it does", in
the same way step 13 shipped a facet-routing ceiling of +0.000.

## Scope

Included:

- a hand-rolled BM25 index over the chunks already stored in Qdrant;
- Reciprocal Rank Fusion over ranked lists;
- three retrieval modes behind the existing `search()` seam, selectable from
  configuration and from every script;
- payload filtering on the lexical branch, so step 13's `filters=` keeps working
  in every mode;
- a parameter sweep over the RRF constant and the per-branch candidate depth;
- a recorded verdict, and the default flipped only if the pre-registered rule in
  "Acceptance" is met.

Excluded, with the step that owns each:

- Qdrant sparse vectors and server-side fusion — considered and rejected below;
- cross-encoder reranking (step 17);
- query rewriting and multi-query retrieval (steps 18-19);
- stemming, lemmatisation and stopword lists — rejected below;
- weighted score fusion — rejected below;
- any change to `data/eval/questions.jsonl`. The evaluation set is frozen for
  this phase.

## Facts this design is built on (measured 2026-09-12, commit `51f053d`)

The baseline is the `dense-sentence-doctype` row of `data/eval/results.jsonl`:
Recall@5 0.776, Recall@10 0.785, MRR 0.810, NDCG@5 0.713, p50 latency ~35 ms
warm, over 38 answerable questions.

Per category:

| category | n | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| exact | 10 | 0.892 | 0.892 | 0.950 |
| code | 10 | 0.850 | 0.850 | 0.850 |
| conceptual | 10 | 0.733 | 0.733 | 0.603 |
| multi_doc | 8 | 0.594 | 0.635 | 0.844 |

Four facts follow, and they shape every decision in this document.

**`exact` is already the strongest category.** It is also the only category BM25
has a mechanism to improve. `conceptual` (0.733) and `multi_doc` (0.594) are the
weak ones, and both are dense retrieval's home ground: a lexical retriever has
no way to know that "protect an API" means "authentication". The honest prior
for this phase is therefore a small or negative aggregate delta.

**The `HTTPException 422` failure is not in the evaluation set.** It is a
qualitative `scripts/ask.py` transcript quoted in the README. The nearest
benchmark question, `q023` ("HTTPException", ground truth
`tutorial/handling-errors` and `reference/exceptions`), already scores Recall@5
1.00 and MRR 1.00. Only two Markdown files in the corpus contain the token
`422` — `advanced/additional-responses.md` and `release-notes.md` — and neither
is ground truth for any question. Surfacing them can only lower the measured
number while improving the demo. The evaluation set stays frozen and 422 stays
qualitative; see "Acceptance".

**`release-notes.md` is a lexical magnet.** It is the largest file in the corpus
and contains nearly every token in it. Under raw TF-IDF it would dominate every
lexical query. The only thing preventing that is BM25's document-length
normalisation, which makes the `b` parameter load-bearing rather than
decorative, and makes the test in "Testing" that pins it a correctness test
rather than a nicety.

**`Recall@10 - Recall@5` is 0.009.** The roadmap records this as the number to
watch: while it stays near zero the retriever misses documents outright and a
reranker has nothing to reorder. Opening this gap is the strongest argument for
this phase, and if hybrid retrieval fails to open it, that finding belongs in the
roadmap before step 17 starts.

## Alternatives considered

**Qdrant named sparse vectors with `modifier=IDF`.** Rejected. It is the
architecture `information.md` sketches and `store.py` currently promises in a
`ponytail:` comment, and it would give filtering and fusion server-side for
free. Against it: Qdrant's IDF modifier supplies only the inverse-document-
frequency factor, so the term-frequency saturation and length normalisation —
`k1` and `b`, the parts that actually make BM25 work on a corpus containing
`release-notes.md` — would still be computed in application code. The formula
would end up split across two systems, which is the wrong shape for the step
whose purpose is to understand it, and it would cost a recreate-and-reupsert
of the nine collections step 12 left behind. Roadmap rule 2 says hand-roll
before you framework. The stale comment in `store.py` is corrected as part of
task 1.

**`fastembed`'s `Bm25` sparse encoder.** Rejected. It writes the least code and
pulls `onnxruntime` to do it, and it outsources precisely the thing step 14
exists to learn. Roadmap rule 3 also applies: a dependency arrives with the step
that needs it, and this step does not need it.

**`qdrant_client.hybrid.fusion.reciprocal_rank_fusion`.** Rejected on fit, not
on principle — it is an already-installed dependency and would otherwise win.
It operates on `ScoredPoint` objects, and the lexical branch produces no Qdrant
points at all, so using it would mean fabricating `ScoredPoint`s to hand back to
a helper that returns them. RRF is six lines; the fabrication would be longer
than the function.

**Weighted score fusion instead of, or alongside, RRF.** Rejected for this
phase. It requires per-query min-max normalisation of two incomparable score
scales, which `search()`'s own docstring calls "a layer that lies", and it would
roughly double the number of runs to answer a question subordinate to "does
lexical retrieval help at all". If RRF shows a real gain, comparing fusion
methods becomes a well-motivated follow-up; if it does not, the comparison is
moot.

**Stemming and a stopword list.** Rejected. BM25's IDF term already drives
stopwords towards zero weight, which is the mechanism a stoplist crudely
approximates. A hand-rolled Porter stemmer is roughly a hundred lines for an
unmeasured gain on a corpus of technical identifiers, where stemming is as
likely to destroy signal as to add it.

**Composing the hybrid retriever inside `scripts/benchmark.py` only.** Rejected.
`run_benchmark` accepts any `str -> list[ScoredChunk]` callable, so this is the
smallest possible diff that produces a number. It also guarantees that
`scripts/ask.py` can never use the winning retriever, which defeats the phase.

## Architecture

Two files carry the phase: one new module for the index, and the existing
`search()` seam for composition.

### `app/retrieval/bm25.py` (new)

```python
def tokenize(text: str) -> list[str]
class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75)
    def search(self, query: str, top_k: int,
               predicate: Callable[[Chunk], bool] | None = None) -> list[ScoredChunk]
def build_index(client: QdrantClient, collection: str) -> BM25Index
```

`tokenize` is `re.findall(r"\w+", text.lower())`. `\w+` rather than
`[a-z0-9_]+` because a French query carries accents and the corpus carries
identifiers; both must survive. It is applied identically to documents and
queries, so `HTTPException 422` becomes `["httpexception", "422"]` and matches a
chunk containing either token.

`BM25Index.__init__` builds, in one pass over the chunks: an inverted index
`postings: dict[str, dict[int, int]]` mapping term to document index to term
frequency; `doc_len: list[int]`; the mean document length; and
`idf: dict[str, float]`.

IDF uses the Lucene variant, `log(1 + (N - df + 0.5) / (df + 0.5))`, not the
textbook Robertson-Sparck-Jones form. The textbook form returns a negative
weight for any term appearing in more than half the documents, which makes
common terms actively subtract from a document's score — a wrong answer that
looks plausible. The two forms are the same line count.

`BM25Index.search` accumulates scores only over documents appearing in the
postings of a query term, never over all 1 484 chunks, and returns
`ScoredChunk`s with `rank` assigned from 1. The optional `predicate` is how
payload filters reach the lexical branch; it is applied while accumulating, so a
filtered query returns `top_k` results rather than `top_k` minus the ones that
were discarded afterwards.

`build_index` scrolls the collection with `with_payload=True,
with_vectors=False` and feeds each record through the existing
`store.chunk_from_payload`, so the lexical index and the dense index are
guaranteed to hold the same chunk set. Pagination accumulates every batch: the
loop shape in Qdrant's own documentation discards the first one.

Index construction over 1 484 chunks costs roughly 200 ms, dominated by the
scroll. A query costs under 5 ms. Neither is persisted between processes;
`scripts/ask.py` spends 1.5-3.7 s in generation, so a 200 ms build is noise
there, and the benchmark builds once for 45 questions. A persisted index is an
optimisation with no problem to solve yet, and is marked as such in the code.

### `app/retrieval/search.py` (extended)

```python
RetrievalMode = Literal["dense", "lexical", "hybrid"]
RETRIEVERS: dict[str, Callable[..., list[ScoredChunk]]]   # dense | lexical | hybrid

def rrf(rankings: Sequence[Sequence[ScoredChunk]], *, k: int, top_k: int) -> list[ScoredChunk]
def matches_filters(chunk: Chunk, filters: Filters | None) -> bool
def default_index(settings: Settings, collection: str) -> BM25Index

def search(query: str, *, top_k: int = 5, mode: str | None = None,
           candidates: int | None = None, rrf_k: int | None = None,
           filters: Filters | None = None, collection: str | None = None,
           settings: Settings | None = None, client: QdrantClient | None = None,
           embedder: Embedder | None = None,
           index: BM25Index | None = None) -> list[ScoredChunk]
```

The signature and return type stay backward compatible: every existing caller —
`app/generation/answer.py`, `scripts/ask.py`, `scripts/search.py`,
`scripts/benchmark.py` — keeps working untouched, and `mode` defaults to the new
`RETRIEVAL_MODE` setting. This mirrors step 12's registry-and-setting pattern
(`chunk.STRATEGIES` plus `CHUNK_STRATEGY`), which is how this repository already
compares N variants and promotes a winner. Step 17 registers a fourth key rather
than re-plumbing five call sites.

`rrf` implements `score = sum over rankings of 1 / (k + rank)`, deduplicating on
`chunk_id`, and re-ranks from 1. It takes ranked lists and ignores their scores
entirely, which is the property that makes it applicable to two incomparable
scales without normalising either.

`matches_filters` lives beside `build_filter` rather than in `bm25.py` on
purpose: they are two translations of the same `Filters` mapping, and separating
them is how one grows support for a key the other silently ignores. It honours
the same contract — a scalar matches one value, a sequence matches any of them,
several keys are ANDed — and rejects unindexed keys through the same check, so a
typo fails identically in both modes.

`default_index` mirrors the existing `default_embedder`: a thin accessor over an
`lru_cache`d builder keyed on `(qdrant_url, collection)`, so the 45 queries of a
benchmark run share one index while the unit tests bypass it entirely by passing
`index=`.

### Data flow, hybrid mode

```text
query
  |
  +-- embed --> Qdrant query_points, limit=candidates, query_filter --+
  |                                                                  |
  +-- tokenize --> BM25Index.search, top_k=candidates, predicate ----+
                                                                     |
                                                                     v
                                                           rrf(k=rrf_k)
                                                                     |
                                                                     v
                                                          top_k ScoredChunks
```

`candidates` is the per-branch retrieval depth before fusion, and it is the
parameter that can actually move recall. Fusing two top-10 lists cannot surface
a document that neither branch ranked in its top 10; only depth can. This is why
the sweep in "Acceptance" varies depth as well as `k`.

### Configuration

Three settings in `app/core/config.py`, documented in `.env.example`:

| setting | default | why |
|---|---|---|
| `retrieval_mode` | `dense` | unchanged behaviour until a measurement earns the change |
| `retrieval_candidates` | `50` | the midpoint of the swept range |
| `rrf_k` | `60` | the constant from the original RRF paper |

`scripts/benchmark.py` gains `--mode`, `--candidates` and `--rrf-k`, and records
all three in the run's `config` so a row in `results.jsonl` remains
interpretable without its shell history. `scripts/search.py` and
`scripts/ask.py` gain `--mode`.

`app/evaluation/benchmark.py` gains one column, `mode`, in `SUMMARY_COLUMNS`.
The RRF constant and the candidate depth are not given columns: the run label
carries them (`hybrid-k60-d50`), and widening the summary table by three columns
to duplicate what the label already says would make step 12's rows harder to
read for no gain.

## What RRF changes about a score

Today `ScoredChunk.score` is a raw cosine similarity, roughly 0.3 to 0.6, and
`search()`'s docstring defends leaving it raw. After fusion the score is an RRF
score, roughly 0.03, on a scale with no relation to similarity.

This is accepted rather than worked around. `score` remains the value that
actually produced the ranking, because a score field reporting a number other
than the one used to rank is the more dangerous lie. Two consequences follow,
both handled by documentation rather than code:

- `DEFAULT_ABSTENTION_THRESHOLD = 0.35` is meaningless outside dense mode, so
  `abstention_rate` must not be compared between a dense row and a hybrid row.
  The `mode` recorded in each run's `config` is what makes this checkable. The
  roadmap already states that a score floor cannot carry refusal; step 22 owns
  the problem, and this phase does not pre-emptively solve it.
- `per_question.top_score` in the history changes meaning with `mode` for the
  same reason. It stays recorded, because step 22 retuning a threshold without
  re-running is worth more than a uniform scale.

No new field is added to `ScoredChunk`. `rerank_score` is reserved for step 17
and stays empty here.

## Latency measurement

`scripts/benchmark.py` builds the BM25 index before entering the timed loop
whenever `mode` is not `dense`. Left inside the loop, the ~200 ms construction
would be charged to the first question, and the `p50 ms` column would stop
meaning "per-query retrieval latency" — which is the only thing it is used for.

## Testing

New file `tests/test_retrieval_bm25.py`, and additions to
`tests/test_retrieval_search.py`. Every test runs without a network and without
an API key, following the injection pattern the existing 15 search tests already
use: `client`, `embedder` and now `index` are parameters.

Two tests carry the design's real risk and are written first:

**Length normalisation.** A short chunk containing a term once must outrank a
long chunk containing the same term once. This is the `b = 0.75` behaviour, and
without it `release-notes.md` wins every lexical query in the corpus. A test that
only checks "a matching chunk is returned" would pass against a broken
implementation.

**Exact-token retrieval.** A chunk containing `422` must be retrieved for the
query `HTTPException 422`. This is the capability dense retrieval demonstrably
lacks and the reason the phase exists.

Also covered: an empty query raises, as in dense mode; a query whose every term
is absent returns an empty list rather than arbitrary chunks; IDF is positive for
a term present in every document; `rrf` places a document ranked second by both
branches above one ranked first by only one; `rrf` deduplicates on `chunk_id`
across branches; `matches_filters` agrees with `build_filter` on scalars,
sequences, multiple ANDed keys and unindexed keys; and hybrid mode with
`filters=` returns only matching chunks from both branches.

Quality gates before every commit, unchanged:
`uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.

## Tasks

| Task | Step | Deliverable | Tag |
|---|---|---|---|
| 1 | 14 | `bm25.py`, its tests, `mode="lexical"`, `--mode` on `scripts/search.py`, the corrected `store.py` comment, and a `bm25-sentence` benchmark row | — |
| 2 | 15 | `matches_filters`, filters honoured on the lexical branch, tests | — |
| 3 | 15 | `rrf`, `mode="hybrid"`, `--mode`/`--candidates`/`--rrf-k`, the pre-loop index build, a `hybrid-k60-d50` row | `v0.5` |
| 4 | 16 | the parameter sweep, the verdict, the default flipped only if the rule below is met | `v0.6` |
| 5 | — | the 422 before/after transcript, README (Phase 5-6, current state, results table, commands), `docs/roadmap.md` | — |

Task 2 precedes task 3 deliberately: hybrid mode cannot honestly claim filter
parity, and `--oracle-filter` cannot be trusted, until the lexical branch filters
too.

No dependency is added, and `pyproject.toml` and `uv.lock` are untouched by the
whole phase. Every benchmark run is free: the embedding cache serves the query
vectors and the lexical branch makes no API call.

## Acceptance

The sweep in task 4 is four runs beyond the `hybrid-k60-d50` baseline of task 3,
varying one parameter at a time from `k=60, depth=50`:

```text
bm25-sentence        lexical only, the step 14 number
hybrid-k20-d50       hybrid-k60-d50       hybrid-k100-d50
hybrid-k60-d20                            hybrid-k60-d100
```

**The decision rule is fixed before the first run.** `RETRIEVAL_MODE` changes
from `dense` to `hybrid` if and only if the best sweep row reaches aggregate
Recall@5 of **0.786 or better** — the 0.776 baseline plus 0.01 — **and** no
per-category Recall@5 regresses by more than 0.05 against the baseline. The
second clause exists because the aggregate can be bought by improving `exact`
while quietly breaking `conceptual`.

If the rule is not met, the default stays `dense`. The code, the tests and every
measured row still ship, and the finding is recorded in the README's results
table and in the roadmap's current state in the same terms step 13 used for its
routing ceiling. A phase that concludes "this corpus does not benefit from
lexical retrieval" has satisfied roadmap rule 1; one that quietly widens the
criterion until the number qualifies has not.

Reported either way, whatever the verdict:

- aggregate Recall@5, Recall@10, MRR, NDCG@5 and p50/p95 latency, per run;
- the full per-category table, because a single aggregate hides which kind of
  question changed;
- `Recall@10 - Recall@5` against the 0.009 baseline. If hybrid retrieval does not
  open this gap, step 17's reranker has nothing to reorder, and that goes into
  `docs/roadmap.md` before step 17 begins rather than after it disappoints;
- the `HTTPException 422` transcript from `scripts/ask.py`, before and after,
  labelled as the qualitative evidence it is and attached to no metric.
