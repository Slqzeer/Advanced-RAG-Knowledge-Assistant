# Contextual Compression Design

Step 20 of [`docs/roadmap.md`](../../roadmap.md); Phase 9 of
[`information.md`](../../../information.md); tag `v1.2`.

## Purpose

Put the documents the pool already holds in front of the model, without paying
for them whole.

`information.md` states this phase as a token-reduction exercise: a 1 000-token
chunk of which 150 tokens matter, an extractor, and a table of tokens, latency,
cost and quality. **On this corpus that exercise measures nothing.** Five
`sentence` chunks are roughly 4 000 characters; a real question costs 851-1 289
tokens end to end, about $0.0003, and `MAX_CONTEXT_CHARS = 12 000` has never once
bound. Cutting 40 % of the context saves $0.0001 and no measurable latency —
generation is 95 % of the wall clock and it is dominated by *output* tokens.

The lever that is real is the one steps 17-19 handed over. The pools hold the
documents and no ranking stage can surface them: dense retrieval reaches
**0.785** Recall@5 and **0.884** Recall@20, FlashRank recovered **+0.002** of
that 0.099 gap for 1 141 ms, and multi-query expansion reordered the pool
without adding to it. Compression is the last mechanism that can act on that
gap, because it does not reorder anything — it changes what a chunk *costs*, so
twenty chunks can occupy the prompt five whole chunks occupy today.

So step 20 fixes the token budget and moves the pool, rather than fixing the
pool and moving the budget. The deliverable, as at steps 13, 16, 17 and 19, is
the measurement. A step 20 that concludes "sentence extraction loses more than
the wider pool gains on this corpus" ships with the same care, in the same table.

## Scope

Included:

- `app/generation/compress.py`: a `COMPRESSORS` registry with one entry,
  `embedding`, and a `compress()` that owns splitting, scoring, budgeting and
  re-assembly for every entry;
- a public `sentence_spans()` in `app/ingestion/chunk.py`, promoted from the
  existing private `_sentence_units()`, so the compressor and the `sentence`
  chunking strategy split text the same way and a code fence stays whole in both;
- a `compress=` / `compress_candidates=` pair on `answer_question()`, threaded
  exactly as `rerank` / `rerank_candidates` already are;
- `--compress` and `--compress-candidates` on `scripts/ask.py` and
  `scripts/benchmark.py`;
- three settings, documented in `.env.example`;
- **Recall@context**, a new aggregate computed by reusing `run_benchmark`
  unchanged, plus the refusal-rate arm that Recall@context is blind to;
- a six-row run matrix and a recorded verdict, with `COMPRESS_METHOD` promoted
  from empty only if the pre-registered rule in "Acceptance" is met;
- the correction of the stale `ponytail:` comment at `context.py:16`.

Excluded, with the step that owns each:

- **parent-child retrieval.** The README's "écarté volontairement" table promises
  it here, and this design breaks that promise deliberately. Parent-child is the
  *inverse* trade — retrieve a small child, send a large parent, which grows the
  context rather than shrinking it — so at a fixed budget it competes with
  compression for the same characters instead of composing with it. It also needs
  a parent id in every payload, which means a re-index. Measuring both in one
  step measures neither: that is the mistake `llm.py`'s prompt version marker
  exists to prevent. Task 6 rewrites the README row to name the follow-up that
  owns it, rather than leaving a promise this step silently did not keep;
- a tokeniser and token-denominated chunk sizes (`tiktoken`). See "The budget
  unit" below: the budget is set in characters and the number *reported* is
  `usage.prompt_tokens`, straight off the API response. Token-denominated
  chunking is a re-chunk and a re-index, and it belongs with parent-child;
- faithfulness, answer relevance and context precision (step 21). This step has
  two quality signals and both are free; RAGAS is the step that judges an answer;
- refusal policy (step 22). This step *reports* refusal rate because it is the
  clause that catches a compressor cutting the answer out of a surviving
  document, and changes no policy on the strength of it;
- an LLM-based extractor. See "Alternatives considered";
- caching compressed contexts (step 23). Sentence embeddings already land in the
  step 05 sqlite cache, which is the only part worth caching;
- any change to `data/eval/questions.jsonl`. Frozen since step 10.

## Facts this design is built on

From `data/eval/results.jsonl`, 38 answerable questions, strategy `sentence`,
collection `chunks`, run `dense-d30-ceiling`:

| category | n | Recall@5 | Recall@20 | gap |
|---|---|---|---|---|
| `multi_doc` | 8 | 0.635 | 0.865 | **+0.230** |
| `conceptual` | 10 | 0.733 | 0.817 | +0.084 |
| `exact` | 10 | 0.892 | 0.950 | +0.058 |
| `code` | 10 | 0.850 | 0.900 | +0.050 |
| **aggregate** | 38 | **0.785** | **0.884** | **+0.099** |

Six facts follow, and they shape every decision below.

**The gap is a `multi_doc` gap.** Weighted by category size, the 0.099
decomposes as `multi_doc` **0.048**, `conceptual` 0.022, `exact` 0.015, `code`
0.013. **Half the gap comes from eight questions — a fifth of the set — whose
ground truth spans two or more documents.** That is not an accident of the
sample: a question needing three documents cannot be satisfied by
a top-5 that spends its five slots on three chunks of the best one. Compression
is the tool whose shape matches that failure — it buys *distinct documents per
character*, which is exactly what `multi_doc` is short of. Every other stage this
project has tried bought ordering, which `multi_doc` already had.

**The token budget has never bound, so it has to be created.** `MAX_CONTEXT_CHARS`
is 12 000 and today's context is roughly a third of that. A compression
experiment against a ceiling nothing touches is not an experiment. Task 1
measures the real k=5 context size over all 38 questions and *that* becomes
`COMPRESS_BUDGET_CHARS`, so every arm is held at the cost of today's prompt.

**No ranking stage can close this gap.** Step 17 measured it directly: FlashRank
over the d30 pool reached 0.779 against the 0.776 baseline, capture −0.067, for
1 141 ms, and cost `code` −0.100. Step 19 arrived at the same wall from the
opposite side: multi-query gained +0.057 MRR and +0.022 Recall@10 while losing
0.011 Recall@5 — it reordered the pool. Compression does not rank. It is the
only untried mechanism that changes what reaches the prompt rather than in which
order.

**Recall@20 equals Recall@30 in every pool.** Step 17 measured it; it is why
`COMPRESS_CANDIDATES` defaults to 20 and not 30. Thirty chunks is 50 % more
sentence embedding for nothing reachable.

**A generic prose model buries code, twice measured.** `ms-marco-MiniLM-L-12-v2`
cost `code` −0.100 over a single-query pool at step 17 and −0.100 again over a
multi-query pool at step 19, which is how step 19 established the cause was the
model and not the pool. A sentence-level extractor is the most likely thing this
project has built to repeat that failure, because a code fence split into
"sentences" is nonsense and a fence scored against a prose query loses. Both are
designed against: `sentence_spans()` treats a fence as one indivisible unit, and
acceptance clause 2 names `code` explicitly.

**Two collections are in play and the arms must not straddle them.**
`dense-sentence-doctype` reports Recall@5 **0.776** on collection
`chunks_sentence`; `dense-d30-ceiling` reports **0.785** on collection `chunks`.
Same strategy, same chunk parameters, different collection. The 0.009 is
immaterial as a quantity and fatal as a methodology: a six-row matrix whose
baseline came from a different collection is six rows that cannot be compared to
their own control. **Every arm in this step runs on `chunks`, and the baseline is
0.785.** A `compress-off-k5` control row re-measures it rather than assuming it.

## Alternatives considered

**An LLM extractor, one call per chunk.** This is what `information.md` sketches
and it is rejected on cost and on confounding. Twenty chunks times 38 questions
is 760 `gpt-4o-mini` calls per arm, four such arms, and +20× latency on a stage
whose whole premise is that it is nearly free. Batching all twenty chunks into
one call removes the latency and replaces it with a worse problem: the model then
reads the entire pool and decides what matters, which is generation. The
measurement would no longer be "does a wider pool help" but "does calling the
model twice help", and those are not separable afterwards.

**LLMLingua.** A real dependency that does token-level compression properly, at a
step where the hand-rolled version is about thirty-five lines. Roadmap rule 2
says hand-roll first, and rule 3 says a dependency arrives with the step that
uses it. If the embedding compressor clears the acceptance rule and the remaining
headroom is still visible, LLMLingua is the obvious follow-up and this design's
registry is how it lands: one more `COMPRESSORS` entry.

**BM25 over sentences, reusing `bm25.tokenize()`.** Genuinely the laziest thing
that could work — zero cost, zero latency, zero API calls — and rejected as the
*default* on evidence already in the history. `bm25-sentence` scored 0.605
Recall@5 against dense's 0.776 at chunk level, and a sentence is shorter than a
chunk, so its IDF is noisier still. It stays available as a second `COMPRESSORS`
entry if the embedding arm's cost ever matters, which at one cached batch per
question it does not.

**Compression inside `search()`, as a retrieval stage.** Rejected. `search()`
returns ranked chunks; a retriever that rewrites chunk text has stopped being
one, and the change propagates: `scripts/search.py` would print truncated chunks
with no query context to explain them, and step 23's cache key would have to
carry the compression method and budget to stay correct. This is the same seam
argument steps 18-19 used to keep `contextualize()` above `search()`.

**Compression inside `build_context()`.** Tempting, because that function already
owns the budget and already drops chunks, and its `ponytail:` comment names this
step. Rejected on the commitment its own docstring makes: *"No I/O, no model, no
state. Everything here is a pure function of its arguments, which is why the
tests are exhaustive: this is the last place the corpus is still verbatim."* A
compressor embeds a query. Putting it there means either breaking that property
or injecting an embedder into the one function this project deliberately kept
free of them. `build_context()` keeps its contract and receives shorter chunks.

**A new `assemble()` wrapper owning compression and context building, with
`build_context()` demoted.** Rejected for the reason steps 18-19 rejected the
equivalent `retrieve()` wrapper: it touches every caller and re-opens a seam
steps 08 and 09 were built on, to save one line in `answer_question()`.

**Scoring sentences against the *chunk* vector rather than the query.** Rejected.
It selects the sentences most representative of the chunk, which is
summarisation. The question is what answers the query, and the query vector is
already computed by the retrieval that produced these chunks.

## Architecture

### `app/ingestion/chunk.py` (one promotion)

```python
def sentence_spans(text: str) -> list[tuple[int, int]]
```

`_sentence_units()` made public, with `_protected_spans()` applied inside rather
than passed in. It already handles abbreviations (`_is_abbreviation`) and already
treats a fenced code block as one unbreakable span. The `sentence` chunking
strategy keeps calling the private path it calls today; nothing about step 12's
measured winner changes. This is a rename and a wrapper, not a rewrite: the
compressor must split exactly as the indexer does, and two splitters that are
"the same for now" are two splitters that diverge at the next edit.

### `app/generation/compress.py` (new)

```python
Compressor = Callable[[Sequence[ScoredChunk], str, int, Settings, Embedder], list[ScoredChunk]]
COMPRESSORS: dict[str, Compressor]                     # embedding

def compress(chunks: Sequence[ScoredChunk], query: str, *,
             method: str | None = None,
             budget_chars: int | None = None,
             settings: Settings | None = None,
             embedder: Embedder | None = None) -> list[ScoredChunk]
```

Mirrors `RETRIEVERS`, `RERANKERS`, `TRANSFORMS` and `STRATEGIES`: the registry is
how this project compares N variants and promotes a winner, and it is what makes
"add LLMLingua later" one function rather than a refactor.

`compress()` owns everything that is not scoring — validation, splitting,
budgeting, re-assembly, the `ScoredChunk` rebuild — for the same reason
`expand()` owns parsing and capping for both transforms: the only thing that
genuinely differs between a future BM25 or LLMLingua entry and this one is how a
sentence gets a number.

`method=""` forces compression off and `method=None` reads `COMPRESS_METHOD`,
the convention `rerank=` and `transform=` already established, which is how an
uncompressed baseline stays runnable after a default flips.

`embedder` is injectable so every unit test runs with no network, no key and no
spend — the pattern `answer_question`, `search` and `expand` all use.

### The `embedding` compressor

```text
chunks (rank order, d20)
  |
  +-- sentence_spans() per chunk   -> units; a code fence is one unit
  |
  +-- embed_texts(all units) ------> one batched call, step 05 sqlite cache
  |
  +-- cosine(unit, query vector) --> one score per unit
  |
  +-- greedy fill to budget_chars -> keep set
  |
  +-- re-emit per chunk, document order, "[…]" at each gap
  |
  v
chunks with shortened .text, rank order preserved, empty ones removed
```

Five properties, each of which is a test:

**Survivors are re-emitted in document order, not score order.** Score order
produces a paragraph that contradicts itself across sentence boundaries. The
selection is by score; the rendering is by position.

**A gap is marked with `[…]`.** Without it the model reads two non-adjacent
sentences as contiguous prose, and the failure it produces — a confident answer
stitched from two unrelated clauses — is indistinguishable from a hallucination
in the output. One string, and it is the difference between a compressed context
and a misleading one.

**A code fence is kept whole or dropped whole.** Half a fence is not shorter
code, it is broken code, and a corpus of API documentation is mostly fences. This
is the single design decision aimed at the `code −0.100` that steps 17 and 19
each measured.

**Rank order across chunks is preserved.** The compressor selects; it does not
re-rank. Steps 16, 17 and 19 spent this project's entire ranking budget and the
conclusion was that ranking is not where the gap is. A compressor that also
reordered would confound the two in a single number.

**A chunk with no surviving sentence disappears from the output.** It is then
`dropped` in `RetrievalStats`, whose `used + dropped == retrieved` invariant is
validated on construction. `answer_question` counts `retrieved` as the size of
the pool it asked for, so a d20 run reporting `20 retrieved, 7 used, 13 dropped`
is the honest description of what happened, and the `retrieved` column stops
meaning `top_k` — which it silently did before this step, because nothing ever
dropped anything.

### `app/generation/answer.py` (extended)

```python
def answer_question(question: str, *, ...,
                    compress: str | None = None,
                    compress_candidates: int | None = None, ...) -> Answer
```

```text
query -> search(top_k=compress_candidates or top_k)
      -> compress(chunks, query, budget_chars=...)
      -> build_context(chunks, max_chars=MAX_CONTEXT_CHARS)
      -> prompt -> LLM -> validate_citations -> Answer
```

**`top_k` is widened to `compress_candidates` only when compression is on.** With
`COMPRESS_METHOD` empty the function retrieves `top_k` and passes the chunks
through untouched, and a test pins that the resulting context is byte-identical
to today's. This is what lets step 25's endpoint, step 23's cache and the four
existing callers stay as they are.

`build_context()` keeps `MAX_CONTEXT_CHARS` as its own outer guard. Two budgets
is not duplication: the compressor's budget is the experiment's fixed variable
and `build_context`'s is the prompt's hard ceiling, and with compression off the
second is the only one that exists.

### The budget unit

**Characters, not tokens, and `tiktoken` does not become a dependency.**

The `ponytail:` comment at `context.py:16` predicts step 20 replaces the
character budget with a real token budget "when the difference starts costing
money". It does not: the difference is worth about $0.0001 a question. Task 6
corrects that comment in place — the practice step 14 used on `store.py` and step
17 used on `search.py:217` — rather than adding a per-model tokeniser to satisfy
a prediction the measurements have since answered.

The number *reported* is `usage.prompt_tokens`, which the API returns on every
call and which is the true count rather than an estimate. If two arms drift more
than a few percent apart in real tokens at the same character budget — plausible,
since code tokenises denser than prose — the character budget is nudged and the
run repeated, and the drift is recorded. An estimate that disagrees with the bill
is worse than no estimate.

### Configuration

| setting | default | why |
|---|---|---|
| `compress_method` | `""` | empty means off; unchanged behaviour until a measurement earns the change, as `rerank_model` and `query_transform` already do |
| `compress_candidates` | `20` | step 17 measured Recall@20 == Recall@30 in every pool; 30 is 50 % more embedding for nothing reachable |
| `compress_budget_chars` | set by task 1 | today's real k=5 context size, so every arm is held at the cost of today's prompt |

`SUMMARY_COLUMNS` gains one column, `compress`, rendered `-` when absent for the
reason the `rerank` and `transform` columns already are: every row written before
this step genuinely had no compressor, and an empty cell reads as a missing
value. `compress_candidates` and the budget get no column — the label carries
them (`compress-embedding-d20`), as step 16 declined a column for `rrf_k`, step
17 for `rerank_candidates` and step 19 for `transform_n`.

### Scripts

`scripts/ask.py` and `scripts/benchmark.py` gain `--compress` (choices
`["", *sorted(COMPRESSORS)]`, default from `COMPRESS_METHOD`),
`--compress-candidates` and `--compress-budget`.

`scripts/search.py` does **not**. It prints ranked chunks, and compression does
not change the ranking; printing truncated text there would show a shorter list
of the same documents in the same order and teach the reader nothing.

`scripts/ask.py` gains one line of output — characters before and after, and the
`prompt_tokens` the call actually billed — because the qualitative evidence this
step owes is "what did it cut", and that is unreadable from an aggregate.

## Dependencies

**None.** `text-embedding-3-small` through the existing `embed_texts()`, into the
existing sqlite cache. This is the second step since 13 to add nothing to
`pyproject.toml`.

## Measurement

### Recall@context

The fraction of a question's ground-truth documents present in the **final
context**, deduplicated to documents exactly as every existing metric is.

It is computed by **reusing `run_benchmark` unchanged**, with a retriever closure
that returns post-compression chunks — the same trick `scripts/benchmark.py`
already uses for `--oracle-filter`, including its guard that no two questions may
share a text. Nothing in `metrics.py` changes: `recall_at_k` over the returned
list with `k` equal to its length is Recall@context by definition.

For an uncompressed k=5 run it is identically Recall@5, which is what makes the
arms comparable and what makes the `compress-off-k5` control worth running.

**What it cannot see:** whether the sentences kept from a surviving document
contain the answer. A compressor that keeps the right document and the wrong
three sentences of it scores identically to one that keeps the answer. That is
the entire reason for the second signal.

### Refusal rate

`scripts/ask.py` over the 38 answerable questions per arm, counting the exact
refusal sentence with `citations.REFUSAL_MARKERS`, which already exists and
already matches it in both languages.

If compression cut the answer out of a document that survived, the model says so
itself, in a fixed string, for free. It is a blunt binary signal and it is the
only one available before RAGAS lands at step 21 — and bluntness is the point: it
has no tuning surface, so it cannot be tuned into agreement.

Roughly $0.01 and two minutes per arm at 1.5-3.7 s a question.

### The run matrix

```text
compress-off-k5               top_k=5,  no compression    the control, on `chunks`
compress-off-d20              top_k=20, no compression    the ceiling, 4x the tokens
compress-embedding-k5         d5  -> budget               the literal brief
compress-embedding-d20        d20 -> budget               the headline arm
compress-embedding-d10        d10 -> budget               is half the pool enough
compress-embedding-d20-b1.5x  d20 -> 1.5x budget          what more budget buys
```

`compress-off-k5` and `compress-off-d20` cost no LLM call and exist to bracket
everything else: 0.785 and 0.884 measured on the same collection, on the same
day, by the same code. Quoting step 17's numbers instead would be quoting a run
made before this step's code existed.

`compress-embedding-k5` is the literal `information.md` brief and it is the
control that separates the two things the headline arm changes at once. If it
loses ground against `compress-off-k5`, extraction damages context, and any gain
at d20 is a wider pool paying for that damage — a fact worth knowing before
reading the headline row, which is why it runs before it.

Total spend, every arm and both signals: under **$0.05**.

## Acceptance

### The decision rule, fixed before the first run

`COMPRESS_METHOD` changes from empty to `embedding` **if and only if all three
hold**:

1. **Recall@context ≥ 0.835** — +0.05 on the **0.785** baseline, which is a
   little over half the 0.099 available. Higher than step 19's +0.03 threshold on
   purpose: this gain is mechanical rather than speculative, because the pool
   provably holds the documents, so the question is what fraction survives the
   cut. Capturing under half of a gap this project can already see does not earn
   a stage in the pipeline.
2. **No per-category Recall@context regresses by more than 0.05** against the
   `compress-off-k5` control's per-category table, with **`code` named
   explicitly**. Steps 16, 17 and 19 were each decided on this clause, and a
   sentence extractor over a corpus of API documentation is the most likely thing
   this project has built to repeat the `code −0.100` that two separate steps
   already measured.
3. **Refusal rate on the 38 answerable questions does not exceed the
   `compress-off-k5` control's.** The clause that clause 1 is structurally blind
   to, and the only one that looks inside a surviving document.

**One setting takes one value.** The rule is read against each
`compress-embedding` arm independently; if more than one clears it, the higher
Recall@context wins and sets both `COMPRESS_METHOD` and `COMPRESS_CANDIDATES`,
and the others are recorded as measured-and-not-promoted. This is step 19's
`rewrite`-versus-`multi` clause, restated for pool depth.

**No latency clause.** The compressor adds one batched embedding call — warm,
cached, a few milliseconds — against 1.5-3.7 s of generation. Step 17's 400 ms
budget existed because a cross-encoder cost 1 141 ms and step 19's 2 000 ms
budget existed because an LLM call ran before retrieval; neither applies. Latency
is reported, not gated, and if it turns out to be gateable that is a finding
against this paragraph.

If the rule is not met, `COMPRESS_METHOD` stays empty. The compressor, the tests,
the new metric and every measured row still ship, and the verdict is recorded
exactly as steps 13, 16, 17 and 19 recorded theirs. A phase that widens its
criterion until the number qualifies has not satisfied rule 1.

### Reported either way, whatever the verdict

- Recall@context, Recall@5/@10/@20, MRR, NDCG@5 and p50/p95 per arm;
- the **full per-category table**, with `multi_doc` read first: it contributes
  0.048 of the 0.099 aggregate gap from a fifth of the questions, and it is the
  category this mechanism is shaped for. An aggregate that moves without
  `multi_doc` moving means something other than the intended mechanism produced
  it, and the row is then evidence against the design rather than for it;
- **real `prompt_tokens`** per arm, p50 and p95, from the API response — the
  number `information.md`'s table asks for, measured rather than estimated;
- **characters in, characters out, and the compression ratio** per arm;
- refusal rate on the 38 answerable questions, and separately on the 7
  unanswerable ones, where a *drop* would be the interesting direction: a wider
  compressed pool giving the model more plausible-looking material to answer from
  is precisely what step 22 needs to know;
- **three transcripts, verbatim**: one `multi_doc` question where compression is
  expected to help, one `code` question where the fence rule is what is on trial,
  and one question where it made the answer worse. The third is not optional. The
  `HTTPException 422` transcript is in the README because a qualitative failure
  told this project something four aggregate rows did not.

## Testing

New file `tests/test_generation_compress.py`, plus additions to
`tests/test_generation_answer.py` and `tests/test_ingestion_chunk.py`. The
`embedder` is injectable everywhere, so every unit test runs with no network, no
key and no spend.

Four tests carry this design's real risk and are written first:

**Compression off changes nothing, byte for byte.** `answer_question` with
`compress=""` produces a context identical to today's, and `compress()` with
`method=""` returns its input unchanged. This is the test that lets the four
existing callers stay untouched.

**A code fence is never split.** A chunk whose fence alone exceeds the budget
either keeps the fence whole or drops the chunk — never emits half of one. This
is the `code −0.100` guard and it is the test most likely to fail first.

**Survivors come back in document order with `[…]` at every gap.** Given a fake
embedder scoring sentence 3 above sentence 1, the output reads `1 […] 3` and not
`3 […] 1`. A test asserting only "the text got shorter" passes against a
compressor that scrambles the paragraph.

**A fully-dropped chunk is accounted for.** A chunk with no surviving sentence
leaves the output, is absent from `sources`, and `RetrievalStats` still satisfies
`used + dropped == retrieved` against the pool size that was actually retrieved.
This invariant is validated in the model and has never been exercised by a real
drop.

Also covered: an unknown `method` raises, naming the available keys, as `mode`,
`rerank` and `transform` already do; an empty query raises; a budget larger than
the input is a no-op; a budget smaller than the single best sentence still
returns that sentence, never an empty context (the same reasoning
`build_context()` uses to keep its first chunk over budget); `sentence_spans()`
returns exactly what `_sentence_units()` returned, over the existing chunking
fixtures; `compress_candidates` widens `top_k` only when compression is on.

Quality gates before every commit, unchanged:
`uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.

## Tasks

| Task | Deliverable | Tag |
|---|---|---|
| 1 | today's real k=5 context size over the 38 questions — characters and `prompt_tokens`, p50 and p95 — recorded, and `COMPRESS_BUDGET_CHARS` set to it | — |
| 2 | `sentence_spans()` promoted public; `compress.py`, `COMPRESSORS`, the `embedding` entry, the gap marker and the fence rule, tests written first | — |
| 3 | `compress=` / `compress_candidates=` on `answer_question()`, the widened `top_k`, the `RetrievalStats` accounting; `--compress` on two scripts; three settings and `.env.example` | — |
| 4 | Recall@context via the `run_benchmark` closure, the `compress` summary column, the two `compress-off` bracket rows | — |
| 5 | the four `compress-embedding` rows, the refusal arm, the three transcripts | — |
| 6 | the verdict; `COMPRESS_METHOD` flipped only if the rule is met; the `context.py:16` comment corrected; README (phase 9, results table, the rewritten parent-child row) and `docs/roadmap.md` | `v1.2` |

Task 1 precedes every line of code deliberately. It is the cheapest measurement
that can change the plan — if today's context turns out to be 8 000 characters
rather than 4 000, the budget, the pool depth and probably the verdict all move —
and it is the ordering step 17 used when its ceiling run saved four rows of
reranker work and step 19 used when its rewrite row reshaped the matrix.
