# Cross-Encoder Reranking Design

Step 17 of [`docs/roadmap.md`](../../roadmap.md); Phase 6 of
[`information.md`](../../../information.md); tag `v0.7`.

## Purpose

Insert a reranking stage between retrieval and generation: retrieve a deep
candidate pool, score every candidate against the query with a cross-encoder,
and keep the best five. Two backends are built — a local ONNX model and a hosted
API — selectable one at a time, and the winner is promoted only if it clears a
rule fixed before the first run.

The deliverable is the measurement, not the reranker. Step 13 shipped a facet
routing ceiling of +0.000 and step 16 shipped a default that did not change;
a step 17 that concludes "this corpus does not benefit from reranking" ships
with the same care, in the same table.

## Scope

Included:

- a candidate-pool ceiling measurement, run before any reranker exists;
- `app/retrieval/rerank.py`: a `RERANKERS` registry, one entry per backend;
- a FlashRank (local ONNX cross-encoder) backend;
- a Cohere Rerank backend;
- a `rerank=` parameter on `search()`, orthogonal to `mode`, threaded through
  `answer_question` and all three scripts;
- three settings, documented in `.env.example`;
- a run matrix over `{backend} x {candidate pool}` and a depth sweep on the
  winner;
- a recorded verdict, and `RERANK_MODEL` promoted from empty only if the
  pre-registered rule in "Acceptance" is met.

Excluded, with the step that owns each:

- query rewriting and multi-query retrieval (steps 18-19);
- context compression (step 20) — the reranker changes *which* five chunks reach
  the prompt, not how many tokens each one costs;
- answer-quality metrics for the reranked pipeline (step 21) — `scripts/ask.py`
  gains the flag here, but faithfulness and answer relevance are RAGAS's job and
  this step measures retrieval only;
- ensembling two rerankers, or fusing their scores. One backend at a time, by
  decision;
- fine-tuning a reranker on this corpus. There are 38 answerable questions;
  fine-tuning on them and then evaluating on them measures nothing;
- any change to `data/eval/questions.jsonl`. The evaluation set stays frozen, as
  it has since step 10.

## Facts this design is built on

From `data/eval/results.jsonl`, 38 answerable questions, chunking strategy
`sentence`:

| label | Recall@5 | Recall@10 | MRR | NDCG@5 | p50 |
|---|---|---|---|---|---|
| `dense-sentence-doctype` (the default) | 0.776 | 0.785 | 0.810 | 0.713 | 65 ms |
| `bm25-sentence` | 0.605 | 0.632 | 0.570 | 0.534 | 2 ms |
| `hybrid-k60-d20` (best fusion row) | 0.737 | **0.829** | 0.748 | 0.680 | 83 ms |
| `hybrid-k60-d50` | 0.721 | 0.807 | 0.757 | 0.676 | 89 ms |

Four facts follow, and they shape every decision in this document.

**The reranker's ceiling is the pool's recall, and it is unmeasured.** A
reranker reorders; it cannot retrieve. Its best possible Recall@5 is the
Recall@N of the pool handed to it. The deepest pool ever measured on this
project is N=10, because `scripts/benchmark.py` has `KS = (1, 3, 5, 10)` and
`--top-k` defaults to `max(KS)`. Recall@20 and Recall@30 have never been run.
Everything in this step is downstream of that number, so measuring it is task 1
and no reranker code is written until it is recorded.

**The known headroom is small.** Against the 0.776 default, reranking a hybrid
top-10 pool perfectly would reach 0.829 — **+0.053, and that is the ceiling, not
the expectation.** Anything beyond it has to come from a deeper pool. This is
the honest prior for the step and it is why the acceptance rule in this document
is expressed as a fraction of measured headroom rather than as an absolute
number invented in advance.

**The step 16 winner cannot be reranked as it stands.** `_hybrid` raises when
`candidates < top_k`, and `hybrid-k60-d20` fuses two 20-deep branches, so its
fused pool holds at most 40 unique chunks and cannot be queried at `--top-k 30`
without changing `candidates` — which changes the ranking, which makes it a
different row. The ceiling runs therefore re-derive the pool at depth 30 and 50
and the reranker is measured against *those*, never against the d20 row.

**Generation is ~95 % of end-to-end latency.** 1.5-3.7 s per question against
~35-90 ms of warm retrieval. A reranker adding 100-400 ms is a real cost to
retrieval and a rounding error to the user, which is why the latency clause in
"Acceptance" is a ceiling rather than a target.

## Alternatives considered

**`sentence-transformers.CrossEncoder`.** Rejected. It is the first option
`information.md` names and the most standard answer, and it pulls `torch` — on
Windows CPU that is roughly 2.5 GB of wheels to run a 30-pair scoring loop.
FlashRank runs the same class of model (`ms-marco-MiniLM-L-12-v2`) through
`onnxruntime` at a fraction of the install, and the thing being learned here —
that a cross-encoder scores `(query, passage)` jointly rather than embedding
each side independently — is identical in both. Roadmap rule 3 says a dependency
arrives with the step that uses it; it does not say the heaviest one does.

**`gpt-4o-mini` as a listwise reranker.** Rejected as the primary, though it
needs no new dependency at all. It costs money per benchmark run, adds 1-3 s,
is not fully deterministic even at `temperature=0`, and answers "does a
cross-encoder help?" with something that is not a cross-encoder. It stays
available as a `RERANKERS` entry anyone can add later; the registry is the whole
extension mechanism.

**Registering the reranker as a fourth `RETRIEVERS` key.** Rejected, and the
comment at `app/retrieval/search.py:217` that predicts it is corrected as part
of task 2 — the same way step 14 corrected the stale `ponytail:` comment in
`store.py`. Reranking is orthogonal to retrieval mode: it composes with dense,
lexical and hybrid alike, and this step must measure at least two of those
pairings. As a mode key, two backends across two pools is four keys, six once
step 18 wants a reranked multi-query, and the registry stops meaning "pick a
retriever". As a parameter it is two registries of three and two entries.

**A free-standing `rerank()` composed by each caller.** Rejected on the ground
step 16 already settled. `search()` has four callers; making each one remember
to retrieve deep and then trim guarantees that one of them eventually does not,
and a pipeline that reranks in the benchmark but not in `scripts/ask.py` is the
exact failure the hybrid spec rejected under "composing the hybrid retriever
inside `scripts/benchmark.py` only". `rerank()` is still a public function with
its own tests; it is simply also wired into the seam.

**Qdrant's server-side reranking / ColBERT multivectors.** Rejected for this
step. It would mean re-indexing nine collections with a second vector per chunk
and moves the scoring into the database, which is the wrong shape for the step
whose purpose is to understand what a cross-encoder does. It is a legitimate
follow-up if reranking proves its worth.

## Architecture

### `app/retrieval/rerank.py` (new)

```python
Reranker = Callable[[str, Sequence[ScoredChunk], int], list[ScoredChunk]]
RERANKERS: dict[str, Reranker]          # flashrank | cohere

def rerank(query: str, candidates: Sequence[ScoredChunk], *, model: str,
           top_k: int, settings: Settings | None = None) -> list[ScoredChunk]
```

`rerank()` validates the model name against `RERANKERS`, dispatches, and returns
`top_k` `ScoredChunk`s with `rerank_score` set, `score` left exactly as the
retriever wrote it, and `rank` reassigned from 1. An empty candidate list
returns an empty list rather than calling a backend, because both backends
charge — in latency or in money — for being asked to rank nothing.

Ties break on `chunk_id`, as `rrf()` already does, so two runs of one commit
agree.

**`_flashrank`** builds a `RerankRequest` from `{"id": chunk_id, "text": text}`
passages and reads back the `score` field. The `Ranker` session is built by an
`lru_cache`d accessor keyed on the model name, mirroring `default_embedder` and
`default_index`.

**`_cohere`** sends `documents=[chunk.text for chunk in candidates]` with
`top_n=top_k` and maps each returned `RerankResponseResultsItem` back through
its `index` into the candidate list. That mapping is the one genuinely dangerous
line in this file — the API answers with positions, not documents, and an
off-by-one pairs every score with the wrong chunk while still producing a
plausible-looking ranking. It is tested first, in task 3.

### `app/retrieval/search.py` (extended)

```python
def search(query: str, *, top_k: int = 5, mode: str | None = None,
           candidates: int | None = None, rrf_k: int | None = None,
           rerank: str | None = None, rerank_candidates: int | None = None,
           filters: Filters | None = None, collection: str | None = None,
           settings: Settings | None = None, client: QdrantClient | None = None,
           embedder: Embedder | None = None,
           index: BM25Index | None = None) -> list[ScoredChunk]
```

With `rerank` unset — the default, from an empty `RERANK_MODEL` — the function
behaves exactly as it does today, and a test pins that. With `rerank` set, the
retriever is called at `top_k=rerank_candidates` and its output is piped through
`rerank()` down to `top_k`.

Three depths are now in play and their constraint is validated in `search()`:

```text
candidates  >=  rerank_candidates  >=  top_k
```

In hybrid mode `candidates` is per-branch depth *before* fusion and
`rerank_candidates` is how deep the *fused* list goes into the cross-encoder.
Conflating them is how a run silently reranks twelve chunks while its label says
thirty, and a short pool reads downstream as "the reranker did not help".

### Data flow, hybrid + rerank

```text
query
  |
  +-- embed    --> Qdrant, limit=candidates ------+
  |                                               |
  +-- tokenize --> BM25Index, top_k=candidates ---+
                                                  |
                                                  v
                                        rrf(k=rrf_k, top_k=rerank_candidates)
                                                  |
                                                  v
                                  cross-encoder score(query, chunk.text)
                                                  |
                                                  v
                                         top_k ScoredChunks
                                    (rerank_score set, score preserved)
```

### Configuration

| setting | default | why |
|---|---|---|
| `rerank_model` | `""` | empty means off; unchanged behaviour until a measurement earns the change |
| `rerank_candidates` | `30` | the pool depth `information.md` sketches, and the midpoint of the swept range |
| `flashrank_model` | `ms-marco-MiniLM-L-12-v2` | ~34 MB, the best-precision small model; the ~4 MB nano default trades away what this step is measuring |

`cohere_api_key` is promoted from an ignored extra — `config.py` has carried the
comment "`.env` carries keys for later phases (COHERE_API_KEY)" since step 06 —
to a real `str | None` field, defaulting to `None` so the test suite and every
API-free command still import the module without a key.

`scripts/benchmark.py`, `scripts/search.py` and `scripts/ask.py` each gain
`--rerank` and `--rerank-candidates`, and `answer_question` threads `rerank`
through exactly as it already threads `mode`. `benchmark.py` records both in the
run's `config`, so a row in `results.jsonl` stays interpretable without its
shell history.

`SUMMARY_COLUMNS` gains one column, `rerank`. The candidate depth does not get a
column: the label carries it (`rerank-flashrank-hybrid-d30`), and step 16 already
declined to widen the table with `rrf_k` for the same reason.

### `KS` and the ceiling

`scripts/benchmark.py` changes `KS = (1, 3, 5, 10)` to `KS = (1, 3, 5, 10, 20,
30)`. `--top-k` already defaults to `max(KS)` and `run_benchmark` already filters
`ks` to those `<= top_k`, so every existing invocation keeps working and no past
row is invalidated — rows written before this change simply have no `recall@20`
key, which `summarise` already handles with `.get(..., 0.0)`.

## What `score` means once a reranker runs

`app/models/chunks.py:58` settled this at step 08: a second field, not a mutated
`score`, "so *what did dense retrieval think?* stays answerable". That is
honoured — and it does put a crack in the rule `search()` defends elsewhere,
that `score` is always the number that produced the ranking.

The resolution is documentation, not code: **`rerank_score is not None` means
`rerank_score` produced the ranking.** Both numbers stay recorded, so a failure
can be read as "the retriever never had it" or "the reranker buried it", which
is the question this step exists to answer. Two consequences, both already
precedented:

- `DEFAULT_ABSTENTION_THRESHOLD = 0.35` was already meaningless outside dense
  mode after RRF; a `rerank_score` on a different scale again changes nothing
  about that. Step 22 owns refusal.
- `per_question.top_score` keeps its current meaning — the retriever's score —
  and each run's `config` records `rerank`, which is what makes a reranked row
  and a dense row checkably non-comparable. Same guard step 16 built for `mode`.

No new field is added to `ScoredChunk`. `rerank_score` was reserved for this
step and this step fills it.

## Latency measurement

The FlashRank ONNX session costs 1-3 s to build on first use. It is constructed
before the timed loop in `scripts/benchmark.py`, exactly as the ~200 ms BM25
build already is: left inside, it lands entirely on question 1 and the `p50 ms`
column stops meaning per-query retrieval latency, which is the only thing it is
used for.

FlashRank downloads its model on first use. That download is a one-line warm-up
documented in the README and performed before any benchmark or test run; it
never happens inside a test and it never happens inside the timed loop.

## Dependencies

`flashrank` and `cohere` are added to `pyproject.toml` in this step, with
`uv lock` and the lockfile committed in the same change, per roadmap rule 3.
`flashrank` pulls `onnxruntime`, `tokenizers` and `numpy`; its `[listwise]`
extra, which would pull a 7B LLM reranker, is not installed.

This step also breaks a small streak: the Cohere rows are the first in
`results.jsonl` that cannot be reproduced offline or for free. The cost is about
$0.23 for the whole matrix. It is accepted and stated in the README next to
those rows, because "the free local model is good enough" is only a result if
the paid one was actually run.

## Testing

New file `tests/test_retrieval_rerank.py`, plus additions to
`tests/test_retrieval_search.py`. `RERANKERS` is injectable and both backends
take an injectable client, so every unit test runs with no network, no API key
and no model download — the pattern the existing retrieval tests already use.

Three tests carry the design's real risk and are written first:

**Cohere index mapping.** Given a fake client returning
`[{index: 2, relevance_score: 0.9}, {index: 0, relevance_score: 0.4}]` against a
three-candidate list, the result must be the third candidate then the first,
each carrying its own score. An off-by-one here silently pairs every score with
the wrong chunk and still returns a well-formed, plausible ranking — the failure
mode no aggregate metric would catch.

**`rerank=None` changes nothing.** `search()` with no reranker returns exactly
what it returns today, for dense, lexical and hybrid. This is the test that lets
the other four callers stay untouched.

**Reordering actually happens.** A fake reranker that inverts its input must
produce inverted `rank`s starting at 1, with `score` preserved from the
retriever and `rerank_score` set from the backend. A test asserting only "five
chunks came back" passes against a reranker that does nothing at all.

Also covered: an unknown `rerank` name raises, naming the available keys, as
`mode` already does; `rerank_candidates < top_k` raises; `candidates <
rerank_candidates` raises in hybrid mode; an empty candidate list returns empty
without calling a backend; ties break on `chunk_id`; `answer_question` threads
`rerank` to its retriever. One `@pytest.mark.requires_model` integration test
scores a real pair with the real FlashRank model and is skipped by default, like
the existing `requires_qdrant` marker.

Quality gates before every commit, unchanged:
`uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.

## Tasks

| Task | Deliverable | Tag |
|---|---|---|
| 1 | `KS` extended to 20 and 30; the three ceiling rows; the headroom recorded and the go/no-go taken | — |
| 2 | `rerank.py` with the FlashRank backend and its tests; `rerank=`/`rerank_candidates=` on `search()`; the corrected comment at `search.py:217`; `--rerank` on three scripts; `rerank` threaded through `answer_question`; the three settings and `.env.example` | — |
| 3 | the Cohere backend, the index-mapping tests first; `cohere_api_key` promoted to a real field | — |
| 4 | the run matrix, the depth sweep on the winner, the verdict, `RERANK_MODEL` flipped only if the rule below is met | `v0.7` |
| 5 | README (Phase 6, current state, results table, commands, the FlashRank warm-up line) and `docs/roadmap.md` | — |

Task 1 precedes every line of reranker code deliberately. If the ceiling comes
back flat, tasks 2-5 are a different conversation, and finding that out after
adding two dependencies would be finding it out too late.

## Acceptance

### Task 1 — the ceiling, and the go/no-go

```text
dense-d30-ceiling     --mode dense                              --top-k 30
hybrid-d30-ceiling    --mode hybrid --rrf-k 60 --candidates 30  --top-k 30
hybrid-d50-ceiling    --mode hybrid --rrf-k 60 --candidates 50  --top-k 30
```

Recorded from each: Recall@5, @10, @20, @30, and **headroom = Recall@30 −
Recall@5** for that pool.

**If the best pool's headroom is below 0.03, step 17 stops here.** The finding —
that this corpus puts nearly every retrievable document in the top five already,
so there is nothing for a reranker to promote — is written into the README's
results table and the roadmap's current state in the same terms step 13 used for
its +0.000 routing ceiling, and no dependency is added. That is a completed step
under roadmap rule 1, not an abandoned one.

### Task 4 — the run matrix

Four rows, both backends against both pools, at candidate depth 30:

```text
rerank-flashrank-dense-d30      rerank-flashrank-hybrid-d30
rerank-cohere-dense-d30         rerank-cohere-hybrid-d30
```

then a depth sweep on the best pairing: `d10`, `d20`, `d50`. Roughly ten rows in
all. FlashRank rows are free; the Cohere rows cost about $0.23 total at
$2/1000 searches.

### The decision rule, fixed before the first run

`RERANK_MODEL` changes from empty to the winning backend **if and only if all
three hold**:

1. **Headroom capture.** `(reranked Recall@5 − pool Recall@5) / (pool Recall@30
   − pool Recall@5)` is **at least 0.50**, and the absolute gain over the
   **0.776** default is **at least +0.03**. The ratio is the honest measure of
   whether the cross-encoder is doing its job; the absolute floor stops a large
   fraction of a tiny headroom from qualifying.
2. **No category is sacrificed.** No per-category Recall@5 regresses by more
   than **0.05** against the 0.776 baseline's per-category table. An aggregate
   can be bought by improving `exact` while quietly breaking `conceptual`, and
   step 16 was decided on exactly this clause.
3. **Latency stays bounded.** p50 retrieval latency stays under **400 ms**.
   Generous on purpose, given generation dominates end to end — but it exists so
   that "+0.06 recall for three seconds" cannot pass silently.

If the rule is not met, `RERANK_MODEL` stays empty. The code, the tests, both
backends and every measured row still ship, and the verdict is recorded exactly
as step 16 recorded hybrid's. A phase that widens its criterion until the number
qualifies has not satisfied rule 1.

### Reported either way, whatever the verdict

- aggregate Recall@5, @10, @20, @30, MRR, NDCG@5 and p50/p95 latency, per run;
- the full per-category table, because one aggregate hides which kind of
  question changed;
- **headroom capture per run**, as defined above — the number that actually says
  whether the cross-encoder works, separate from whether the pool was any good;
- **FlashRank against Cohere, head to head**, on the same pool at the same
  depth. "The free local model is within noise of the paid API" is a publishable
  result and is most of why both were built;
- the latency cost of each backend, isolated: pool retrieval and reranking timed
  separately, so the 400 ms clause is checkable rather than asserted;
- the `HTTPException 422` transcript from `scripts/ask.py` once more, labelled as
  the qualitative evidence it is and attached to no metric. The roadmap already
  records the refusal as correct; if reranking changes it, that is a finding
  about the reranker, not a fix.
