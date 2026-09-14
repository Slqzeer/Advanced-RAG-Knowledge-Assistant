# RAGAS Evaluation Design

Step 21 of [`docs/roadmap.md`](../../roadmap.md); Phase 11 of
[`information.md`](../../../information.md); tag assigned when the step lands.

## Purpose

Measure the thing every retrieval number in this project is structurally blind
to: whether the answer is true, and whether it answers the question.

Step 20 is the reason this is not an abstract exercise. Its headline is
**Recall@context 0.721 → 0.814**, a retrieval number, and in the same run it
measured a counter-example on a named question. `q018` — *"How do I write a test
that calls my own endpoints?"* — went from a cited answer to a refusal while its
category's Recall@context did not move at all. The right documents were in the
context both times. The uncompressed context carried three code blocks; the
compressed one carried none, because a fenced block is one unit by design and a
greedy per-character budget lets cheap prose outbid a 900-character example.

So the gap between "the documents are in the context" and "the answer is good"
is real, it is not noise, and it has a mechanism. Every metric this project
owns — Recall@K, MRR, NDCG, Recall@context — scores the first and cannot see the
second. Refusal rate, step 20's cheap proxy, saw it only because the failure
happened to be a refusal; it would have missed a confidently wrong answer
entirely.

Step 21 builds the instrument that sees it, proves the instrument can fail, and
then spends it on the one question step 20 left open.

The deliverable, as at steps 13, 16, 17, 19 and 20, is the measurement. A step 21
that concludes "length-penalised scoring loses more than it gains on this corpus"
ships with the same care, in the same table.

## Scope

Included:

- `app/evaluation/judge.py`: a `METRICS` registry over three reference-free RAGAS
  metrics — faithfulness, response relevancy, context precision without
  reference — and a `judge()` that is the only place `ragas` is imported;
- `include_contexts=` on `answer_question()` and `Answer.contexts`, because the
  context text RAGAS judges is not reachable from today's return value;
- `COMPRESS_LENGTH_PENALTY`, one float in `compress()` that covers step 20's
  hypothesis, its inverse, and today's shipped behaviour at α = 0;
- `--ragas` on `scripts/benchmark_answers.py`, default off, writing per-question
  and aggregate scores into the existing `data/eval/answers.jsonl` rows;
- two settings, documented in `.env.example`;
- a two-arm calibration bracket, a four-arm α sweep, and a recorded verdict, with
  `COMPRESS_LENGTH_PENALTY` promoted off 0.0 only if the pre-registered rule in
  "Acceptance" is met.

Excluded, with the reason or the step that owns each:

- **`LLMContextRecall`, answer correctness and factual correctness.** All three
  need a written reference answer per question, and `EvalQuestion` carries
  document-level ground truth only — by an explicit step 10 decision, because
  chunk-level labels do not survive a re-chunk. Writing 38 grounded reference
  answers creates a second frozen artefact that must stay in sync with the corpus
  and whose quality silently caps every metric computed from it. They are an em
  dash in the results table, which is what rule 1 says an unmeasured cell is;
- **judging the 20-chunk retrieved pool.** RAGAS is fed the contexts that
  actually reached the model — post-compression, roughly five units. Faithfulness
  is a claim about what the model was shown, and context precision over a pool
  the model never saw would cost twenty verdicts a question to score a prompt
  that does not exist;
- **citation correctness as a fourth metric.** `validate_citations` already
  parses and checks every `[n]`, and the warnings it emits are already on the
  `Answer`. Aggregating them is cheap and it is not what this step is for; it
  belongs with step 24's per-query trace, where the warning list is already being
  recorded;
- **refusal policy (step 22).** This step *reports* refusal rate on the same row
  as faithfulness — the two came from one run, which is the point — and changes
  no policy on the strength of it;
- **a separate `scripts/benchmark_ragas.py`.** See "Alternatives considered";
- **replacing Recall@context.** It stays, unchanged, on every row. The step 20
  finding is that the two metrics disagree on a named question; a design that
  retires one of them deletes the evidence;
- **any change to `data/eval/questions.jsonl`.** Frozen since step 10.

## Facts this design is built on

From `data/eval/answers.jsonl` and `data/eval/results.jsonl`, 38 answerable
questions, strategy `sentence`, generation `gpt-4o-mini` at `temperature=0`:

- the shipped default `compress-embedding-d20` reaches **Recall@context 0.814**
  against a **0.721** control and a **0.836** ceiling — 81 % of what was
  reachable — for **+79 ms** p50;
- `multi_doc`, which carried most of the gap, goes **0.552 → 0.792**; `code` is
  **flat at 0.800**;
- refusal rate on the 38 answerable questions goes **0.316 → 0.211**, agreeing
  with the retrieval number;
- both `--unanswerable` arms abstain **7/7** either way, so the wider compressed
  pool did not make the model over-answer;
- **`q018` regressed from a cited answer to a refusal** while `code`'s
  Recall@context did not move. This is the one row where the two signals
  disagree, and it is the reason this step exists;
- generation is **~95 % of wall clock** — 1.5-3.7 s against ~35 ms of warm
  retrieval — so judge latency is a cost line, not a gate.

From the code:

- `Answer` carries `sources` (index, document_id, title, url, section, chunk_id,
  score) and `context_chars`, and **not the context text**. The contexts RAGAS
  judges are assembled inside `answer_question()` and discarded;
- `compress.py`'s module docstring is explicit that a `Compressor` scores units
  and nothing else, while `compress()` owns splitting, budgeting, re-assembly and
  the frozen-model rebuild;
- `scripts/benchmark_answers.py` already runs all 38 questions through
  `answer_question()` and records refusal, context chars, real `prompt_tokens`
  and latency, one JSONL row per run with the git commit;
- `mypy` runs `strict` on `app` with `ignore_missing_imports = true`, so an
  untyped third-party package does not need stubs;
- every dependency in `pyproject.toml` is pinned to a minor range.

From the RAGAS documentation (fetched at design time, v0.4 line):

- `ragas.llms.llm_factory("gpt-4o", client=OpenAI())` wires a judge over a plain
  `openai` client. `LangchainLLMWrapper` is **deprecated** — the LangChain
  dependency this step would have dragged in a year ago is no longer required;
- `ResponseRelevancy` needs embeddings as well as an LLM: it generates
  `strictness` questions from the answer and takes their cosine against the
  original question;
- `Faithfulness` decomposes the answer into claims and returns a per-claim
  verdict against the contexts — roughly two judge calls per question.

## Alternatives considered

**Hand-rolling the three judges.** This project hand-rolled Okapi BM25, RRF and
the compressor, and rule 2 says hand-roll before you framework. It was rejected
here on the one ground that separates an instrument from a component: a
hand-rolled faithfulness score is comparable to nothing, including the next
person's. The value of a published metric *is* the published definition. BM25
had a paper to implement against and a correctness test; a judge prompt has
neither, and a subtly weak claim-decomposition prompt fails by reporting 0.9 for
everything — the one failure mode that looks like success. Containment answers
the rest: `judge.py` is the only importer, exactly as `rerank.py` is the only
importer of `flashrank`.

**A separate `scripts/benchmark_ragas.py`.** It is the file-per-benchmark
precedent this repo already set three times, and it was rejected because that
precedent has a stated reason that does not apply here.
`benchmark_conversations.py` and `benchmark_answers.py` are separate from
`benchmark.py` because `run_benchmark`'s seam is `str -> list[ScoredChunk]` and
produces ranking metrics, which neither of them produces. `benchmark_answers.py`'s
seam is `answer_question()` — which is precisely what RAGAS needs, question,
answer and contexts from one pass. A separate script would pay for a second full
generation run over 38 questions to reproduce answers that already exist, and
would then put refusal rate and faithfulness in two files that must be joined by
label. Step 20's finding is that those two numbers disagreed; they belong on one
row.

**Judging offline from a persisted run.** Attractive for one real reason:
re-judging with a different judge model would cost no generation at all, and a
judge failure would never lose the generation. Rejected as premature — it
requires `answers.jsonl` to carry full answer and context text for every row,
and the judge model is frozen for this step by the decision below. If step 24 or
a later judge comparison makes re-judging routine, this becomes the obvious
refactor and the `--ragas` flag is where it hooks in.

**`gpt-4o-mini` as its own judge.** Cheapest, and defensible on the grounds that
the comparison is between arms so a constant self-preference bias cancels — the
same argument that lets `recall@k`'s document-dedup quirk stand. Rejected because
the absolute faithfulness number is then not credible on its own, only the delta
is, and that caveat would have to be carried in prose next to the number forever.
Judge ≠ generator also means the number survives steps 22-25 changing the
generator.

**A new `COMPRESSORS` entry for length-penalised scoring.** Rejected because it
puts a selection concern in a scoring slot. `compress.py` commits in its own
docstring to entries that score and a `compress()` that budgets; length
penalisation is budgeting. As one float in `compress()` it also applies to any
future entry for free, α = 0 reproduces today's behaviour byte-for-byte — which
makes the control provable rather than asserted — and negative values come free,
which matters because of the next section.

## Architecture

### `app/evaluation/judge.py` (new)

```python
METRICS: dict[str, Callable[[RagasLLM, RagasEmbeddings], SingleTurnMetric]]
# "faithfulness" | "relevancy" | "context_precision"

def judge(
    samples: Sequence[JudgeSample],
    *,
    metrics: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> list[dict[str, float]]: ...
```

`JudgeSample` is a small frozen model — `question_id`, `question`, `answer`,
`contexts: list[str]` — so the script builds what it has and the module owns the
translation into `SingleTurnSample`. No ragas type crosses the module boundary in
either direction; `judge()` returns plain floats keyed by metric name, one dict
per sample, in input order.

The registry mirrors `RETRIEVERS`, `RERANKERS`, `TRANSFORMS`, `COMPRESSORS` and
`STRATEGIES`: it is how this project names N variants and reports them in one
table. Entries are factories rather than instances because both metrics that need
an LLM need the *same* configured one, and constructing them at import time would
build a judge — and read a key — for every process that imports the module,
including the test suite.

An unknown metric name raises and names the available keys, as `mode`, `rerank`,
`transform` and `compress` already do.

A per-sample judge failure is recorded on the sample and does not abort the run,
for the reason `benchmark_answers.py` already states for generation: one
transient API error must not cost a 38-question run.

### `app/generation/answer.py` and `app/models/answers.py` (extended)

`Answer.contexts: list[str] = []`, populated only when
`answer_question(..., include_contexts=True)`.

The contexts RAGAS judges are the compressed chunk texts assembled between
`compress_chunks()` and the LLM call, and today they are discarded. Rebuilding
them in the harness by re-running search and compression would duplicate the
pipeline and could diverge from what was actually sent, which is the one thing a
faithfulness score must not do.

Default empty keeps step 25's `POST /query` body unchanged: an HTTP response that
echoes several thousand characters of corpus text is not the endpoint anyone
wants. This is the same move `usage` got at step 08 — capture it at the seam
rather than retrofit every call site later.

### `app/generation/compress.py` (extended)

One line in `compress()`, applied between scoring and the greedy fill:

```python
effective = raw / max(len(unit), 1) ** settings.compress_length_penalty
```

- **α = 0.0** — today's shipped behaviour, byte-for-byte. Sort by relevance, fill
  the budget in that order.
- **α = 1.0** — relevance per character: step 20's stated hypothesis, and the
  fractional-knapsack-optimal ordering for maximum total relevance mass inside a
  fixed budget.
- **α < 0** — rewards long units.

**This design records a disagreement with step 20's hypothesis before the run,
because pre-registering a prediction is what makes the result mean anything.**
Step 20 proposed relevance-per-character to stop a 900-character fence losing to
cheap prose. Dividing by length *penalises* that fence. The fence already has two
strikes — code embeds poorly against a natural-language question, so its raw
cosine is low, and it is expensive — and α > 0 adds a third. The prediction is
therefore: **α = 1.0 raises the aggregate and hurts `code`**, reproducing the
shape of the step 17 and step 19 reranker results from a third direction, and the
direction that rescues `q018` is **α = −0.5**. Because α is one float, testing
both costs two rows instead of two algorithms. If α = 1.0 wins on `code` as well,
this paragraph is wrong and the step says so in the README.

### `scripts/benchmark_answers.py` (extended)

`--ragas` (default off), `--ragas-metrics` (default all three), `--judge-model`.
The generation loop is unchanged except for `include_contexts=args.ragas`;
judging runs once after it, over the collected samples, in one batch.

Refused questions are judged like any other. A refusal scoring near zero on
relevancy is the signal, not an error case to skip — skipping it is how a
compressor that refuses more often scores better.

New keys on the existing row: `ragas` (the aggregate per metric, plus per
category), and one score dict per entry in `per_question`. The row keeps
`refusal_rate`, `context_chars_*` and `prompt_tokens_*` exactly as today, so a
`--ragas` row and an ordinary row remain comparable, and `config` gains
`judge_model` and `compress_length_penalty`.

`--no-save` already exists and is what the calibration fixtures below use.

### Configuration

```
judge_model: str = "gpt-4o"          # JUDGE_MODEL
compress_length_penalty: float = 0.0 # COMPRESS_LENGTH_PENALTY
```

`judge_model` is deliberately not `generation_model`. `compress_length_penalty`
defaults to 0.0 — unchanged behaviour until a measurement earns the change, the
same posture `rerank_model` and `query_transform` still hold. Both documented in
`.env.example` in French, matching the file.

## Dependencies

`ragas>=0.4,<0.5`, added in the same change as `uv lock` and the lockfile,
per rule 3. Nothing else: `llm_factory` over the `openai` client the project
already depends on replaces the LangChain path, which is deprecated upstream.

Whatever `ragas` pulls transitively — `instructor`, `numpy`, `pandas` and
`datasets` are likely — is recorded in the plan's first task by reading the
resolved lockfile, not guessed here. If the transitive tree turns out to be
larger than the rest of the project combined, that is a fact worth writing down
next to the decision, not a reason to reopen it mid-step.

## Measurement

### The calibration bracket

Two arms, run before any α is swept, reproducing step 20's control and winner at
generation level:

| Label | Config |
|---|---|
| `ragas-k5` | `--compress "" --top-k 5` |
| `ragas-d20-a0` | shipped default: `embedding`, d20, α 0.0 |

**This bracket is the actual deliverable of step 21.** Step 20 measured
+0.093 Recall@context between these two configurations and, on `q018`, a
regression the metric could not see. The question is whether faithfulness and
relevancy agree with the aggregate, agree with `q018`, or disagree with both.
Any of the three is a result; the first would mean Recall@context has been an
adequate proxy all along, which is worth knowing before step 22 builds guardrails
on top of it.

### The α sweep

Four arms on the d20 pool: **α ∈ {0.0, 0.5, 1.0, −0.5}**. α = 0.0 is
`ragas-d20-a0` re-used, not re-run.

### Reported either way, whatever the verdict

- mean faithfulness, response relevancy and context precision per arm, with the
  **full per-category table**, `code` and `multi_doc` read first;
- **`q018`'s own three scores, by name, in every arm.** It is the question this
  step was handed; an aggregate that moves while `q018` does not has not answered
  it;
- refusal rate, `context_chars` and real `prompt_tokens` on the same row, from
  the same run;
- **judge cost and wall clock per arm** — the honest price of the instrument,
  and the number that decides whether `--ragas` is affordable in step 28's CI;
- **two transcripts, verbatim**: `q018` under the best arm and under the control,
  with their scores. Step 20 put the `HTTPException 422` transcript in the README
  because a qualitative failure told this project something four aggregate rows
  did not.

## Acceptance

### Part A — the instrument reports nothing until it proves it can fail

Fixed before the first arm, and run first:

1. a refusal on an answerable question scores **response relevancy < 0.3**;
2. a synthetic answer carrying a claim absent from its contexts scores
   **faithfulness < 0.5**;
3. a context list whose single relevant chunk sits *below* three irrelevant ones
   scores **context precision < 0.6**;
4. a complete, correct answer to the same question scores **response relevancy
   > 0.7**.

Clause 3 was corrected after Task 6 measured what this metric actually is.
`ContextPrecisionWithoutReference` is average precision over **ranked** verdicts, not
precision over a set: a relevant chunk at rank 1 scores 1.0 however much junk follows
it. The original fixture put the relevant chunk first and therefore measured the
metric's definition rather than the judge's discrimination — it returned 0.9999999999
and could not have returned anything else. The threshold is untouched at 0.6; only the
fixture changed, so that the clause tests the property it always claimed to.

**This has a consequence for how the arms are read.** `compress()` preserves rank order
and never reorders, so across every arm in this step the top context is the same chunk
the retriever ranked first. Context precision will therefore be high and nearly flat
between arms, and it is the least informative of the three metrics for this comparison.
It is still reported — it is what catches an arm that pads the context below the useful
material — but a step 21 conclusion must not rest on it.

Clause 4 was added after Task 1 measured a correct but terse answer at **0.278**
relevancy - close enough to clause 1's threshold to matter. Three clauses that
each check a *bad* answer scores low would pass against a metric stuck near zero,
and that metric would then report every arm as equally bad. The pair proves
separation, not just a low number.

If any of the four misses, the wiring is wrong, **no arm is run and no
compressor conclusion is drawn.** A judge that returns 0.9 for everything
produces a full results table that is entirely noise, and nothing downstream
would catch it.

### Part B — the decision rule for `COMPRESS_LENGTH_PENALTY`

It changes from 0.0 **if and only if all three hold** against the `ragas-d20-a0`
control:

1. **mean faithfulness ≥ control − 0.01.** A compressor that improves relevance
   by dropping the material the answer was grounded in has made the system worse;
   this is the clause that catches it.
2. **mean response relevancy ≥ control + 0.03.** Step 19's threshold, and the
   same reasoning: below that, a stage is not worth the parameter it adds.
3. **no category loses more than 0.05 mean relevancy, with `code` named
   explicitly.** Steps 16, 17, 19 and 20 were each decided on this clause, and
   `code` is the exact mechanism under test — the design predicts above that
   α > 0 will fail this one.

**One setting takes one value.** The rule is read against each α independently;
if more than one clears it, the higher relevancy wins, and the others are
recorded as measured-and-not-promoted.

**No latency clause.** The penalty is an exponent on a float already computed;
the judge runs offline and is never in the answer path.

If the rule is not met, `COMPRESS_LENGTH_PENALTY` stays 0.0. The judge module,
the tests, the `--ragas` flag and every measured row still ship, and the verdict
is recorded exactly as steps 13, 16, 17 and 19 recorded theirs. A phase that
widens its criterion until the number qualifies has not satisfied rule 1.

## Testing

New `tests/test_evaluation_judge.py`, plus additions to
`tests/test_generation_compress.py` and `tests/test_generation_answer.py`. Every
unit test runs with no network, no key and no spend; `judge()` takes its metrics
from the registry, and the tests register fakes.

Four tests carry this design's real risk and are written first:

**α = 0 changes nothing, byte for byte.** `compress()` at
`compress_length_penalty=0.0` returns output identical to today's over the
existing compression fixtures. This is what makes the control a measurement
rather than an assumption, and it guards every step 20 row in `answers.jsonl`.

**α reorders a known pair.** Given a short unit at cosine 0.60 and a long unit at
cosine 0.55, α = 0 keeps the short one first and α = 1 flips the order; α = −0.5
flips it the other way. A test asserting only "the output got shorter" passes
against a penalty that does nothing.

**`Answer.contexts` is empty unless asked, and is what was sent when asked.**
With `include_contexts=True` the list equals the compressed chunk texts that
reached `build_context`, in order — not the retrieved pool, and not the rendered
context block with its `[n] source — title` headers.

**A judge failure on one sample does not lose the other 37.** A metric that
raises on sample 12 leaves 37 scored rows and one recorded error, matching how
the script already handles a generation failure.

Also covered: an unknown metric name raises and names the available keys; an
empty `contexts` list is scored rather than crashing, because a refusal with no
context is a real row; `--ragas` off leaves the JSONL row shape byte-identical to
today's; `judge()` returns results in input order.

Part A's three fixtures are real judge calls and are marked `requires_api`, a new
pytest marker alongside `requires_qdrant` and `requires_model`, and documented in
`pyproject.toml` the same way.

Quality gates before every commit, unchanged:
`uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.

## Tasks

| Task | Deliverable |
|---|---|
| 1 | `ragas` added and locked; the resolved transitive tree recorded; a five-line spike proving `llm_factory` + one `SingleTurnSample` scores end to end, before any project code is written |
| 2 | `judge.py`, `METRICS`, `JudgeSample`, `judge()`, the per-sample failure path; tests written first |
| 3 | `include_contexts=` on `answer_question()` and `Answer.contexts`; `COMPRESS_LENGTH_PENALTY` in `compress()`; two settings and `.env.example`; the α = 0 byte-identity test |
| 4 | `--ragas` on `benchmark_answers.py`, the new row keys, the per-question score dicts |
| 5 | **Part A's three calibration fixtures, run and recorded.** No arm runs until they pass |
| 6 | the calibration bracket: `ragas-k5` and `ragas-d20-a0`, with `q018` read by name |
| 7 | the α sweep, the two transcripts, judge cost and wall clock |
| 8 | the verdict; `COMPRESS_LENGTH_PENALTY` flipped only if the rule is met; README (phase 11, results table, new commands) and `docs/roadmap.md` |

Task 1 precedes every line of project code deliberately. `ragas` v0.4 deprecated
the wrapper API that most of its own documentation still shows, and the cheapest
way to find out what the installed version actually wants is to call it once.
This is the ordering step 17 used when its ceiling run saved four rows of
reranker work, step 19 used when its rewrite row reshaped the matrix, and step 20
used when task 1 set the budget the whole design turned on.

Task 5 is a gate, not a step. Everything after it is meaningless if it fails.
