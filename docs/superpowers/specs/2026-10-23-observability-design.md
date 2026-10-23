# Observability Design

Step 24 of [`docs/roadmap.md`](../../roadmap.md); Phase 15 of
[`information.md`](../../../information.md); tag assigned when the step lands.

## Purpose

The pipeline already returns an `Answer` with aggregate retrieval counts, token
usage and total latency. That is enough to print a useful CLI footer, but not
enough to explain one bad answer after the fact. The resolved query, ranked
chunks, reranker scores and exact prompt disappear before `answer_question()`
returns.

Step 24 writes one durable, schema-versioned trace per top-level query. A trace
must answer four questions without rerunning the model:

1. What did the user ask, and what standalone query did conversation rewriting
   produce?
2. Which chunks were ranked, which reached the prompt, and which were cited?
3. What did the model receive and return, how many tokens did it use, and what
   did that cost at the configured price?
4. Was the result generated, refused, or served by a cache layer, and where did
   the time go?

The trace file is also the input Step 29's dashboard can read. Step 24 does not
build that dashboard.

## Facts this design is built on

- `Answer` already carries the response, cited sources, retrieval counts,
  context characters, token usage, total latency, model, warnings, refusal and
  cache layer. Those fields are reused rather than duplicated under new names.
- `answer_question()` is the one seam that sees the resolved query, ranked
  `ScoredChunk` objects and exact prompt. Capturing outside it would require
  repeating retrieval, which could diverge and spend money.
- `cached_answer()` is the outer shipping seam. A cache hit never enters
  `answer_question()`, so only the wrapper can emit a truthful hit trace.
- Benchmarks intentionally call `answer_question()` directly to bypass the
  answer cache. Direct calls must still trace when tracing is enabled.
- `ScoredChunk` already keeps retrieval `score`, final `rank`, and optional
  `rerank_score`. No retrieval model change is needed.
- The default cache has semantic lookups disabled. Follow-ups receive exact
  lookups only. Step 23 explicitly handed this step the job of counting
  follow-ups and exact repeats.
- Prompt and completion text can contain user or corpus data. Current
  OpenTelemetry guidance makes content capture suppressible. This design does
  the same: tracing itself is opt-in, and prompt/response bodies need a second
  opt-in.

## Scope

Included:

- `app/observability/trace.py`: frozen trace models, one per-query recorder and
  an append-only JSONL sink;
- `TRACE_PATH`, `TRACE_CONTENT`, and optional input/output list-price settings;
- instrumentation at `cached_answer()` and `answer_question()`;
- stage latencies for contextualisation, retrieval, compression/context
  assembly and generation/validation;
- ranked chunk identity, retrieval score, reranker score, prompt inclusion and
  citation inclusion;
- cache outcome, refusal outcome, token usage and computed generation cost;
- `scripts/trace_report.py`: counts, latency/cost summaries, cache layers, and
  the number of follow-ups and exact-repeat follow-ups;
- a no-network writer benchmark and a pre-registered overhead rule.

Excluded:

- **An observability backend or dashboard.** Step 29 owns the dashboard. A local
  JSONL artifact is inspectable now and is the smallest stable input to it.
- **OTLP export and automatic FastAPI instrumentation.** There is no HTTP
  service or collector until steps 25-26. Adding the OpenTelemetry SDK and an
  exporter now would produce no durable result the project can inspect. The
  trace ID and field names are chosen so Step 25 can map one record to a root
  span and the stage timings to child spans without changing this schema.
- **Metrics endpoint.** `GET /metrics` belongs to Step 25.
- **RAGAS scores.** They are offline evaluation results, not facts available on
  a production query.
- **Logging secrets.** API keys, base URLs, settings dumps and environment
  values never enter a trace.
- **Changing retrieval or generation behaviour.** With `TRACE_PATH` empty, the
  execution path and returned `Answer` remain unchanged.

## Alternatives considered

### OpenTelemetry only

The Python API can create custom spans, and without an SDK/exporter it is a
no-op. That is useful for libraries but not sufficient here: Step 24 must
*record* queries, and the repository has no collector to receive them. Shipping
API calls that disappear, or pulling the collector forward from Dockerisation,
both add machinery without completing the step. Defer export, keep compatible
IDs.

### Put the trace fields on `Answer`

Rejected. Step 25 returns `Answer` as the HTTP response body; returning the full
prompt, completion and twenty ranked chunks would leak internals and inflate
every response. A trace is an operational record, not the public answer model.

### Rebuild the prompt in `scripts/ask.py`

Rejected. `--show-context` already demonstrates why this is fragile: it repeats
search and compression. A trace that can disagree with the call it describes is
worse than no trace.

### One callback on every pipeline stage

Rejected for this step. `answer_question()` already owns the stage boundaries
and returned objects needed by the brief. A single explicit recorder reaches
the same facts with fewer public interfaces. If Step 25 later needs distributed
child spans inside Qdrant, Redis or the model client, OpenTelemetry
instrumentation can add them independently.

## Architecture

```text
cached_answer(question, history, ...)
  |
  +-- start QueryRecorder when TRACE_PATH is set
  |
  +-- cache hit ------------------------------+
  |                                           |
  +-- miss -> answer_question(..., trace=recorder)
                |
                +-- contextualize             |
                +-- retrieve ranked chunks    |
                +-- compress / build context  |
                +-- generate / validate       |
                                                    |
  +-- finish once, at the outermost owner <---------+
        -> one QueryTrace JSON object + newline
```

`answer_question()` creates and finishes its own recorder only when no recorder
was supplied. This gives direct benchmark calls one trace while preventing a
`cached_answer()` miss from writing twice.

Trace writing is optional operational work. A write failure never replaces a
valid answer or masks the pipeline's original exception. On a successful query,
the returned answer receives a `trace write failed: ...` warning.

## Trace schema

`QueryTrace` is a frozen Pydantic model with these groups:

| Group | Fields |
|---|---|
| Identity | `schema_version=1`, 32-lowercase-hex `trace_id`, UTC `started_at` |
| Query | `question`, `resolved_query`, `history_turns`, `follow_up`, `follow_up_repeat` |
| Configuration | generation model, retrieval mode, transform, reranker, compressor, prompt version, requested `top_k` |
| Retrieval | ordered `TraceChunk[]`: chunk/document IDs, final rank, retrieval score, optional rerank score, `used`, `cited` |
| Content | optional system prompt, user prompt and model response |
| Result | refusal reason, cache layer, warnings |
| Usage | provider token dictionary and optional `cost_usd` |
| Timing | contextualise, retrieval, context, generation and total milliseconds |
| Failure | optional exception type and message for failed calls |

The original and resolved query are always present once tracing is enabled:
they are the identity of the operation being diagnosed. `TRACE_CONTENT=false`
omits system/user prompts and the model response. Enabling `TRACE_PATH` is
therefore consent to store queries; enabling `TRACE_CONTENT` is consent to
store the larger prompt and response bodies. Both facts are stated beside the
settings in `.env.example`.

`TraceChunk.used` means it reached `build_context()`; `cited` means its
`chunk_id` survived citation validation into `Answer.sources`. Those are
different facts and neither is inferred from list position after the record is
written.

## Cost

Prices are configuration, not constants:

```text
GENERATION_USD_PER_MTOK_INPUT=
GENERATION_USD_PER_MTOK_OUTPUT=
```

When both prices and provider token counts exist:

```text
cost_usd = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
```

Otherwise `cost_usd` is `null`. A model name is not a price: gateways can bill
the same wire-model ID differently, and hard-coding today's public price would
silently falsify later traces.

## Cache and follow-ups

- A miss records the full pipeline.
- An exact or semantic hit records the cache layer, lookup latency and stored
  answer metadata. Retrieval chunks and prompts are empty because neither
  operation happened on that call.
- `follow_up = history_turns > 0`.
- `follow_up_repeat = follow_up and cache == "exact"`.

The report prints both counts and their ratio. This answers Step 23's assumption
with traffic rather than inventing a fixture: if exact-repeat follow-ups are
material, resolving before a semantic lookup becomes worth measuring. Until
there is traffic, the report says zero observations, not zero demand.

## Storage and retention

The default path is empty: tracing is off and no directory is touched. The
documented local path is `data/processed/traces.jsonl`, already ignored by Git.
Every record is one UTF-8 JSON object followed by one newline. Parent directories
are created on first write.

The file contains potentially sensitive query content and is never committed.
Rotation, retention and multi-process export belong with the long-running API
and deployment in steps 25-26. Step 24's CLI is single-process; this limit is
marked with a `shortcut:` comment at the append function.

## Acceptance rule

**Rule O (one truthful trace, negligible local overhead):**

1. exactly one schema-valid record is appended for each generated answer, exact
   cache hit, semantic cache hit, structural refusal and raised pipeline error;
2. tracing disabled appends nothing and does not alter the returned `Answer`;
3. a generated record reproduces the resolved query, ordered chunk IDs/scores,
   used/cited flags, prompt (when enabled), response, usage and total latency
   observed by the test doubles;
4. on a representative 4,000-character prompt, append-only JSONL overhead is
   below **5 ms p95** across 1,000 local writes.

The threshold is deliberately generous relative to generation's measured
2,622 ms median. If local disk cannot clear it, the sink must be buffered or
replaced before it reaches the request path.
