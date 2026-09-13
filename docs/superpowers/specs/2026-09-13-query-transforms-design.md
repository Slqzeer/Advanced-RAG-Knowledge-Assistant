# Query Transforms Design

Steps 18-19 of [`docs/roadmap.md`](../../roadmap.md); Phases 7-8 of
[`information.md`](../../../information.md); tags `v0.8`-`v0.9`.

## Purpose

Work on the *question* rather than on the index. Two transforms, one seam:

- **rewriting** — turn a question into a better standalone query, either by
  reformulating it or by resolving it against what was asked before;
- **multi-query expansion** — generate several phrasings of one question,
  retrieve with each, and fuse the rankings.

Step 17 handed this step its brief in one sentence: *the gap is not a ranking
problem a generic model solves.* A reranker reorders a pool and is capped by
that pool's recall. A query transform changes the query, so it can retrieve a
document no pool ever held. That is the whole reason these two steps are worth
running and it is what the acceptance rule below is built around.

The deliverable is the measurement, not the transform. Step 13 shipped a facet
routing ceiling of +0.000, step 16 shipped a default that did not change, and
step 17 shipped a reranker that stays off. A step 18-19 that concludes "this
corpus does not benefit from query transforms" ships with the same care, in the
same table.

## Scope

Included:

- `app/retrieval/transform.py`: a `TRANSFORMS` registry with `rewrite` and
  `multi`, plus a `contextualize()` that is deliberately *not* in it;
- a `transform=` parameter on `search()`, orthogonal to `mode` and to `rerank=`,
  reachable from all three scripts;
- a `history=` parameter on `answer_question()` and `--history` on
  `scripts/ask.py`;
- three settings, documented in `.env.example`;
- an aggregate row for standalone rewriting against the frozen evaluation set;
- a small conversational fixture, `data/eval/conversations.jsonl`, and the
  `conv-raw` / `conv-rewrite` delta it exists to produce;
- a run matrix over `{N} x {mode}` for multi-query, and one reranked row;
- a recorded verdict, and `QUERY_TRANSFORM` promoted from empty only if the
  pre-registered rule in "Acceptance" is met.

Excluded, with the step that owns each:

- context compression (step 20) — a transform changes *which* chunks reach the
  prompt, not how many tokens each one costs;
- answer-quality metrics for the transformed pipeline (step 21) — `ask.py` gains
  the flags here, but faithfulness and answer relevance are RAGAS's job;
- refusal and the abstention threshold (step 22) — this step *reports* what
  multi-query does to `abstention_rate` because the number is already computed,
  and changes no policy on the strength of it;
- a cache for transform outputs (step 23) — 38 questions at one `gpt-4o-mini`
  call each is roughly $0.001 and 20 s per run, which is not a cost worth a
  sqlite file;
- conversation *storage* and multi-turn session state (step 25) — this step
  takes a history it is handed and returns a standalone query; nothing here
  remembers anything between calls;
- a router that applies a transform only to some questions. `information.md`
  suggests it ("je ne l'active donc que pour certaines requêtes") and it is a
  legitimate follow-up, but it is a second decision layered on an unmeasured
  first one. Measure unconditional first;
- HyDE, step-back prompting, and any third transform. The registry is the
  extension mechanism; adding entries later costs one function each;
- any change to `data/eval/questions.jsonl`. The evaluation set stays frozen, as
  it has since step 10. The conversational fixture is a **new, separate file**.

## Facts this design is built on

From `data/eval/results.jsonl`, 38 answerable questions, chunking strategy
`sentence`:

| label | Recall@5 | Recall@10 | Recall@30 | MRR | p50 |
|---|---|---|---|---|---|
| `dense-sentence-doctype` (the default) | **0.776** | 0.785 | — | 0.810 | 65 ms |
| `dense-d30-ceiling` | 0.785 | 0.836 | **0.884** | — | 34 ms |
| `hybrid-d30-ceiling` | 0.768 | 0.873 | 0.890 | — | 40 ms |
| `hybrid-d50-ceiling` | 0.743 | 0.879 | **0.917** | — | 44 ms |
| `rerank-flashrank-dense-d30` (best reranked) | 0.779 | 0.862 | 0.884 | — | 1 141 ms |

Five facts follow, and they shape every decision in this document.

**The pools saturate at depth 20.** Step 17 measured Recall@20 equal to
Recall@30 in every pool. Twenty chunks deep, a single query has found everything
it is ever going to find on this corpus. Deepening the pool is exhausted as a
strategy; changing the query is the only thing left that can add a document.

**A reranker is capped by its pool and a transform is not.** This is why step
17's acceptance rule cannot be reused. "Headroom capture" —
`(gain) / (Recall@30 − Recall@5)` — was the honest measure for a stage that can
only reorder what it was given. Multi-query has no such ceiling: three phrasings
retrieve three pools, and their union can exceed the Recall@30 of any one of
them. The rule here is therefore expressed in absolute terms.

**The evaluation questions are already well-formed.** They were hand-written at
step 10 to read like real questions, not like keyword queries. Rewriting them is
as likely to blur them as to sharpen them, and that is a result worth having
cheaply. It is why the standalone rewriting row is task 3, before the multi-query
matrix, rather than bundled into it.

**Generation is ~95 % of end-to-end latency, but a transform is not free the way
a reranker was.** Step 17 could afford a 400 ms retrieval budget because
retrieval was a rounding error against 1.5-3.7 s of generation. A transform adds
an LLM call *before* retrieval even starts — 500-1 500 ms — plus N times the
retrieval cost. `information.md` saw this coming: "Multi-query augmente
légèrement le recall mais multiplie la latence". The budget below is 2 000 ms and
is stated honestly as roughly +50 % end to end: a cost the user feels.

**`rrf()` already fuses N rankings.** It is typed
`rankings: Sequence[Sequence[ScoredChunk]]` and reads ranks only. Step 19's
fusion shipped at step 16; this step writes none.

## Alternatives considered

**Both transforms inside `search()`, via a `history=` parameter.** Rejected.
Multi-query expansion belongs inside the seam — it fans out and fuses, which is
retrieval work, and step 16 already settled that a stage every caller must
remember to compose is a stage one caller will eventually forget. Conversation
history is different in kind. A `search()` that knows what a conversation is
pushes that dependency into step 23's cache key and step 25's endpoint, both of
which would then have to carry it. Contextualisation runs in
`answer_question()`, where the conversation already lives, and hands `search()`
the one thing it understands: a query string.

**Both transforms above `search()`, in a free-standing module each caller
composes.** Rejected on the ground step 16 already settled and step 17 restated.
`search()` has four callers; making each one expand, fan out and fuse guarantees
that one of them does not, and a pipeline that expands in the benchmark but not
in `scripts/ask.py` is precisely the failure the hybrid spec rejected. It also
duplicates the N-way fan-out and the `rrf()` call in four places.

**A new `retrieve()` wrapper owning transform, search and rerank, with `search()`
demoted to an inner primitive.** Rejected for this step, though it is the
cleanest layering on paper. It touches every caller and every retrieval test, and
it re-opens a seam steps 08, 11, 16 and 17 were each built on top of. The
docstring at `search.py` promising "no LLM here" is corrected instead — the same
way step 17 corrected the stale prediction at `search.py:217` and step 14
corrected the stale `ponytail:` comment in `store.py`. A comment that has stopped
being true is fixed where it is, not routed around.

**Structured JSON output for the generated queries.** Rejected. It is what
`information.md` sketches, and it would mean adding a `response_format` parameter
to `complete()` — the one function in the project that talks to a model, kept to
one signature on purpose so that swapping providers is editing its body. One
query per line costs a `splitlines()`, a strip of leading `1.` / `-` ordinals,
and a cap at N. The failure mode JSON would protect against is handled by the
fallback below, which is needed regardless.

**Registering the transforms as `RETRIEVERS` keys.** Rejected for the reason step
17 already recorded when it declined to make reranking a fourth mode: a transform
composes with every retrieval mode and with the reranker. As registry keys, two
transforms across three modes is six entries and twelve once a reranker is
involved, and `RETRIEVERS` stops meaning "pick a retriever". As a parameter it is
one more registry of two entries.

**Fine-tuning or few-shot-priming the rewriter on this corpus.** Rejected. There
are 38 answerable questions; tuning a prompt against them and then evaluating on
them measures the tuning, not the transform.

## Architecture

### `app/retrieval/transform.py` (new)

```python
Transform = Callable[[str, int, Settings, Completer], list[str]]
TRANSFORMS: dict[str, Transform]                       # rewrite | multi

def expand(query: str, *, transform: str, n: int | None = None,
           settings: Settings | None = None,
           llm: Completer | None = None) -> list[str]

def contextualize(question: str, history: Sequence[Mapping[str, str]], *,
                  settings: Settings | None = None,
                  llm: Completer | None = None) -> str
```

`expand()` validates the transform name against `TRANSFORMS`, dispatches, parses,
and returns at least one query. `llm` is injectable so every unit test runs with
no network and no key — the pattern `answer_question` already uses.

**`contextualize()` lives in this file and is not a `TRANSFORMS` key.** It is a
query transform by nature, so it belongs beside the others; it takes a history
and returns a single string, so it cannot sit behind a registry whose contract is
`str -> list[str]`, and it must never be reachable from `search()`. Both facts
are stated in its docstring, because "why is this not in the registry?" is the
first question a reader will have.

Two registry entries:

**`rewrite`** returns `[rewritten]` — one query, replacing the original. This is
a real experiment and not a formality: it asks whether LLM reformulation of an
already well-formed question helps retrieval on this corpus at all.

**`multi`** returns `[original, p1, ..., p(n-1)]`. **The original query is always
first and always kept.** N phrasings that all drift in the same direction is how
an expansion loses ground the raw query already held; keeping the original makes
the raw ranking a floor that fusion can only build on. It also means `multi` at
`n=1` is exactly today's behaviour, which is a free sanity check.

### Parsing, and the fallback

One query per line. The parser strips blank lines, leading ordinals (`1.`, `2)`,
`-`, `*`) and surrounding quotes, drops any line that survives as empty, and caps
the list at `n`.

**If parsing yields nothing usable, `expand()` returns `[query]`.** An API hiccup
or a chatty preamble in the middle of a 38-question benchmark must degrade to
exactly today's behaviour, never zero a question. `run_benchmark` already catches
per-question exceptions and reports the count, but a degraded row is better than
a lost one.

This fallback is **not silent**. The queries actually used are recorded in each
run's `per_question` rows, so a row where the fallback fired shows a single query
equal to the original and says so on inspection. A fallback nobody can see is a
transform that quietly stopped working three commits ago.

### `app/retrieval/search.py` (extended)

```python
def search(query: str, *, top_k: int = 5, mode: str | None = None,
           transform: str | None = None, transform_n: int | None = None,
           candidates: int | None = None, rrf_k: int | None = None,
           rerank: str | None = None, rerank_candidates: int | None = None,
           filters: Filters | None = None, collection: str | None = None,
           settings: Settings | None = None, client: QdrantClient | None = None,
           embedder: Embedder | None = None, index: BM25Index | None = None,
           llm: Completer | None = None) -> list[ScoredChunk]
```

`transform` follows `rerank`'s convention exactly: `None` reads
`QUERY_TRANSFORM`, and the empty string forces it off, which is how an
untransformed baseline stays runnable once the default flips.

With `transform` unset — the default, from an empty `QUERY_TRANSFORM` — the
function behaves exactly as it does today, and a test pins that for all three
modes.

```text
queries = expand(query, transform=..., n=transform_n)

len(queries) == 1  ->  RETRIEVERS[mode](queries[0], top_k=depth, ...)
len(queries)  > 1  ->  rrf([RETRIEVERS[mode](q, top_k=depth, ...) for q in queries],
                          k=rrf_k, top_k=depth)

                   ->  rerank (unchanged) -> top_k
```

**The `len(queries) == 1` branch is correctness, not an optimisation.** `rrf()`
over a single ranking preserves its order but overwrites every `score` with
`1 / (k + rank)`. Routing the `rewrite` row through it would silently replace
cosine similarities with fusion constants and make its `top_score` column
incomparable to every dense row in the history, for no change in ranking
whatsoever.

### Data flow, multi + hybrid + rerank

```text
query
  |
  +-- expand() --> [q0 (original), q1, q2]
                        |
        for each query: |
                        +-- embed    --> Qdrant,   limit=candidates --+
                        |                                             +-- rrf(rrf_k)
                        +-- tokenize --> BM25Index, top_k=candidates -+       |
                                                                              v
                                                     rrf(rrf_k, top_k=rerank_candidates)
                                                                              |
                                                                              v
                                                              cross-encoder, top_k
```

RRF runs twice in this path — once per query to fuse the two branches, once
across queries. That composes because RRF reads ranks and never scores: fusing
fused *ranks* is well defined in a way that fusing fused scores would not be. The
code says so in a comment at the call site.

### What `score` means once a transform runs

Unchanged from step 16's answer, and it needs no new field. When more than one
query is retrieved, the reported `score` is the RRF score — around 0.03, not a
cosine's 0.3-0.6 — because it is what produced the ranking. Two consequences,
both already precedented:

- `DEFAULT_ABSTENTION_THRESHOLD = 0.35` was already meaningless outside dense
  mode after step 16 and outside an unreranked run after step 17. Step 22 owns
  refusal.
- each run's `config` records `transform` and `transform_n`, which is what makes
  a transformed row and a dense row checkably non-comparable. Same guard step 16
  built for `mode` and step 17 for `rerank`.

### `app/generation/answer.py` (extended)

```python
def answer_question(question: str, *,
                    history: Sequence[Mapping[str, str]] | None = None,
                    transform: str | None = None, transform_n: int | None = None,
                    ...) -> Answer
```

`history` is typed as a sequence of `{"role", "content"}` mappings — OpenAI's own
message shape. No new model: it is what `complete()` already speaks and what
step 25's endpoint will receive off the wire, and inventing a `Turn` class here
would mean translating in both directions for no gain.

**An empty or absent history returns the question unchanged, without an LLM
call.** Paying 500-1 500 ms and a billed call to rewrite a first-turn question
into itself is the failure this parameter would otherwise ship on every
single-turn request in the project.

Only the last `HISTORY_TURNS = 4` turns are sent. An unbounded history is a
prompt that grows until it breaks, and the turn that disambiguates a follow-up is
almost always the previous one.

`transform` and `transform_n` thread straight through to the retriever, exactly
as `mode` and `rerank` already do, so the measured winner of step 19 reaches the
answer and not only the benchmark.

### `app/evaluation/dataset.py` (extended)

```python
class EvalConversation(EvalQuestion):
    history: list[dict[str, str]] = []

def load_dataset(path, *, categories=None,
                 model: type[EvalQuestion] = EvalQuestion) -> list[EvalQuestion]
```

One added field and one added parameter. `EvalConversation` inherits every
validator the frozen set already enforces — a category from the `Literal`,
non-empty ground truth, no ground truth on an `unanswerable` — and
`load_dataset` keeps its line-numbered errors and its duplicate-id check for
free. Writing a second loader for a file that differs by one field is how the two
drift.

### Scripts

`scripts/search.py`, `scripts/benchmark.py` and `scripts/ask.py` each gain
`--transform` (choices `["", *sorted(TRANSFORMS)]`, default from
`QUERY_TRANSFORM`) and `--transform-n`.

`scripts/ask.py` also gains `--history`, repeatable, alternating user and
assistant starting with user:

```bash
uv run python scripts/ask.py "et pour docker ?" \
  --history "Comment limiter la mémoire d'un container Kubernetes ?" \
  --history "Vous pouvez définir resources.limits.memory dans le manifeste [1]."
```

One flag rather than two carries full fidelity and matches the fixture's shape,
so the CLI demo and the measured runs exercise the same code path.

`scripts/benchmark_conversations.py` (new, ~50 lines) is described under "Step 18,
track B" below.

### Configuration

| setting | default | why |
|---|---|---|
| `query_transform` | `""` | empty means off; unchanged behaviour until a measurement earns the change |
| `multi_query_n` | `3` | the number `information.md` sketches, and the midpoint of the swept range |
| `history_turns` | `4` | enough to resolve a follow-up; bounded so the prompt cannot grow without limit |

`SUMMARY_COLUMNS` gains one column, `transform`. `transform_n` does not get a
column: the label carries it (`multi-n3-dense`), as step 16 declined a column for
`rrf_k` and step 17 for `rerank_candidates`.

`summarise()` renders a missing `transform` as `-` rather than an empty cell, for
the reason the `rerank` column already does: every row written before this step
genuinely had no transform, and an empty cell reads as a missing value.

## Dependencies

**None.** `gpt-4o-mini` through the existing `complete()`. This is the first step
since 13 that adds nothing to `pyproject.toml`, which roadmap rule 3 makes a
virtue rather than an accident.

## Measurement

### Step 18, track A — the standalone rewriting row

```text
rewrite-standalone    --transform rewrite    (dense, sentence, no rerank)
```

Compared against `dense-sentence-doctype` at **0.776**. This is step 18's
aggregate number and it runs before any multi-query code exists, for the same
reason step 17 measured its ceiling before writing a reranker: if reformulating
an already well-formed question degrades retrieval on this corpus, that reshapes
what multi-query is expected to do, and finding it out after the matrix has run
is finding it out too late.

### Step 18, track B — the conversational fixture

`data/eval/conversations.jsonl`, roughly ten conversations, each a short history
plus a follow-up plus document-level ground truth, validated against the corpus
by `scripts/validate_dataset.py` like the frozen set.

**The construction rule: every follow-up's ground-truth document must be
unreachable from the follow-up alone.** This is the rule that makes the fixture
mean anything. `"et pour docker ?"` — the roadmap's own headline — is a *weak*
case: it carries its own discriminating noun, so a bare dense search finds
`fastapi:deployment/docker` with no rewriting at all. It stays as the qualitative
demo transcript because it is the example the roadmap promises, and the measured
fixture is built from genuinely referential follow-ups instead:
`"and how do I test that?"`, `"what about the async version?"`,
`"can I do the same thing with a header?"`.

Scored by **reusing `run_benchmark` unchanged**. Each conversation's history is
bound into the retriever closure, keyed on the question text — the exact trick
`scripts/benchmark.py` already uses for `--oracle-filter`, including its guard
that two questions must not share a text. Two labelled runs land in
`results.jsonl` like any other:

| label | what it retrieves |
|---|---|
| `conv-raw` | the bare follow-up |
| `conv-rewrite` | `contextualize(follow-up, history)` |

The delta between them is the number step 18 exists to produce.

**Ten conversations is a capability check, not a promotion criterion.** Nothing
flips `QUERY_TRANSFORM` on the strength of ten hand-written follow-ups; the
sample is too small and it was written by the same person who wrote the
rewriter's prompt. Only track A and step 19's matrix feed that decision. The
fixture answers a different and equally necessary question: does the thing work
at all?

### Step 19 — the run matrix

```text
multi-n2-dense      multi-n3-dense      multi-n5-dense
multi-n3-hybrid     multi-n3-dense-rerank-flashrank
```

`multi-n3-hybrid` earns a row because step 17 measured the hybrid d50 pool at
**0.917** Recall@30, the largest pool in the project. Multi-query and hybrid
retrieval are the two *widening* tools this project has, and pairing them is
exactly what step 17's verdict points at.

`multi-n3-dense-rerank-flashrank` costs nothing — the model is local, already
built, already warmed up — and answers something step 17 could not. Step 17 found
the reranker moved precision between categories rather than adding any, with
`code` at −0.100, and attributed that to the model rather than the pool. A wider,
differently-composed pool is the cheapest available test of that attribution.

Roughly six runs. Total spend for every run in this document, both tracks
included: under **$0.05**.

## Acceptance

### Task 3 — the standalone rewriting go/no-go

`rewrite-standalone` is recorded whatever it says. There is no stopping condition
attached to it: unlike step 17's ceiling measurement, a poor result here does not
invalidate multi-query, because expansion keeps the original query and rewriting
replaces it. A regression is a finding that informs the matrix, not a reason to
skip it.

### The decision rule, fixed before the first run

`QUERY_TRANSFORM` changes from empty to the winning transform **if and only if
all three hold**:

1. **Absolute gain.** Recall@5 is at least **+0.03** over the **0.776** default.
   No headroom-capture clause, because a transform has no pool ceiling to capture
   a fraction of — see "Facts this design is built on".
2. **No category is sacrificed.** No per-category Recall@5 regresses by more than
   **0.05** against the 0.776 baseline's per-category table. An aggregate can be
   bought by improving `conceptual` while quietly breaking `code`; steps 16 and
   17 were each decided on this clause.
3. **Latency stays bounded.** p50 retrieval latency, LLM call included, stays
   under **2 000 ms**. Stated honestly: that is roughly +50 % end to end against
   1.5-3.7 s of generation — a cost the user feels, unlike the reranker's, which
   is why the budget is a ceiling and not a target.

**One setting takes one value.** `rewrite` and `multi` compete for the same slot;
if both clear the rule, the higher Recall@5 wins and the other is recorded as
measured-and-not-promoted.

If the rule is not met, `QUERY_TRANSFORM` stays empty. Both transforms, the
fixture, the tests and every measured row still ship, and the verdict is recorded
exactly as steps 13, 16 and 17 recorded theirs. A phase that widens its criterion
until the number qualifies has not satisfied rule 1.

### Reported either way, whatever the verdict

- aggregate Recall@5, @10, @20, @30, MRR, NDCG@5 and p50/p95 latency, per run;
- the full per-category table, because one aggregate hides which kind of question
  changed;
- **the LLM call timed separately from retrieval**, so clause 3 is checkable
  rather than asserted — a single p50 that mixes a 900 ms completion with a 65 ms
  vector search cannot be reasoned about;
- **tokens per question** for the transform, alongside the generation cost
  `scripts/ask.py` already prints;
- **the generated queries verbatim** for a handful of questions — the qualitative
  evidence, attached to no metric, in the spirit of the `HTTPException 422`
  transcript. A recall number cannot tell you the rewriter turned a French
  question into English, or three paraphrases into three near-duplicates;
- **`abstention_rate` on the 7 unanswerable questions.** It is already computed
  and costs nothing to report. Multi-query paraphrases an out-of-corpus question
  three ways and retrieves three times the plausible-looking garbage; if widening
  retrieval makes refusal *harder*, that is a finding step 22 needs and this is
  the only step that will see it. Reported here, acted on there;
- the `conv-raw` / `conv-rewrite` delta and the `fastapi:deployment/docker`
  transcript, labelled as the capability evidence they are.

## Testing

New file `tests/test_retrieval_transform.py`, plus additions to
`tests/test_retrieval_search.py`, `tests/test_generation_answer.py` and
`tests/test_evaluation_dataset.py`. `llm` is injectable everywhere, so every unit
test runs with no network, no key and no spend.

Three tests carry the design's real risk and are written first:

**`transform=None` changes nothing.** `search()` with no transform returns
exactly what it returns today, for dense, lexical and hybrid. This is the test
that lets the four callers stay untouched.

**Fusion actually happens across queries.** A fake transform returning three
queries must cause three retriever calls, and a chunk returned by all three must
outrank a chunk returned by one. A test asserting only "five chunks came back"
passes against a transform that expands into three copies of the same query.

**Garbage output falls back to the original, visibly.** An LLM returning a
preamble, an empty string, or only blank lines must yield `[query]` — and the
recorded queries for that call must show it, so the fallback is inspectable
rather than silent.

Also covered: an unknown `transform` name raises, naming the available keys, as
`mode` and `rerank` already do; the parser strips ordinals, bullets and quotes;
the list is capped at `n`; `multi` always places the original first;
`len(queries) == 1` does **not** pass through `rrf()`, and the returned `score`
is still the retriever's; `contextualize` with empty history returns the question
unchanged **and makes no LLM call**; `contextualize` sends at most
`HISTORY_TURNS` turns; `answer_question(history=...)` passes the *standalone*
query to its retriever; `EvalConversation` inherits the frozen set's validators;
`load_dataset(model=EvalConversation)` reports the same line-numbered errors.

Quality gates before every commit, unchanged:
`uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.

## Tasks

| Task | Deliverable | Tag |
|---|---|---|
| 1 | `transform.py`, `expand()`, `TRANSFORMS` with `rewrite` and `multi`; the parser, the cap and the fallback, tests written first | — |
| 2 | `transform=` / `transform_n=` on `search()`, the `len == 1` correctness branch, the corrected "no LLM here" docstring, the `rrf()`-twice comment; `--transform` on three scripts; the three settings and `.env.example` | — |
| 3 | the `rewrite-standalone` row against the 0.776 baseline, recorded whatever it says | — |
| 4 | `contextualize()`, `history=` on `answer_question`, `--history` on `ask.py`, the `fastapi:deployment/docker` transcript | — |
| 5 | `conversations.jsonl` built to the referential rule, `EvalConversation`, `load_dataset(model=)`, `scripts/benchmark_conversations.py`, the `conv-raw` / `conv-rewrite` delta | `v0.8` |
| 6 | the multi-query matrix, the N sweep, the verdict, `QUERY_TRANSFORM` flipped only if the rule above is met | `v0.9` |
| 7 | README (phases 7-8, current state, results table, commands) and `docs/roadmap.md` | — |

Task 1 precedes every line of wiring deliberately, and task 3 precedes the
matrix. The cheapest measurement that can change the plan runs first, which is
the ordering step 17 used and the reason its ceiling run saved four rows of
reranker work.
