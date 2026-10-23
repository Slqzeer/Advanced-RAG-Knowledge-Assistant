# Step 24: Per-query observability implementation plan

> Execute in order. Write each named test first, see it fail for the intended
> reason, then add only the production code that makes it pass.

**Goal:** Produce one durable, privacy-controlled trace for every top-level RAG
query, including rewritten query, ranked chunks and scores, exact prompt and
response when enabled, usage, latency, configured cost, refusal and cache
outcome. Provide a local report that counts follow-ups and exact repeats.

**Design:** [`2026-10-23-observability-design.md`](../specs/2026-10-23-observability-design.md)

**No new dependency.** Pydantic, already installed, validates the record;
`pathlib`, `json`, `time`, `datetime` and `uuid` cover the sink and timing.
OpenTelemetry export waits for the API/collector steps; see the design.

---

## Task 1: Freeze the trace contract and settings

**Files:**

- Create: `app/observability/__init__.py`
- Create: `app/observability/trace.py`
- Create: `tests/test_observability_trace.py`
- Modify: `app/core/config.py`
- Modify: `.env.example`

- [ ] Add failing tests for a minimal `QueryTrace`, a generated trace with a
      ranked chunk, rejection of malformed trace IDs, and cost with/without both
      configured prices.
- [ ] Add frozen Pydantic models `TraceChunk`, `TraceTimings`, and `QueryTrace`.
      Use `schema_version: Literal[1] = 1`; validate `trace_id` as 32 lowercase
      hexadecimal characters; use timezone-aware UTC datetimes.
- [ ] Add these settings, all preserving today's behaviour by default:

```python
trace_path: Path | None = None
trace_content: bool = False
generation_usd_per_mtok_input: float | None = None
generation_usd_per_mtok_output: float | None = None
```

- [ ] In `.env.example`, document that enabling `TRACE_PATH` stores original
      queries, and `TRACE_CONTENT=true` additionally stores prompts, retrieved
      text inside those prompts, and responses. Leave all four values empty or
      false; show `data/processed/traces.jsonl` as the local example.
- [ ] Run:

```powershell
uv run pytest tests/test_observability_trace.py tests/test_core_config.py -q
uv run ruff check app/observability tests/test_observability_trace.py app/core/config.py
uv run mypy app
```

---

## Task 2: Implement the recorder and JSONL sink

**Files:**

- Modify: `app/observability/trace.py`
- Modify: `tests/test_observability_trace.py`

- [ ] Add tests that `write_trace()` creates the parent directory, appends two
      independently parseable lines without overwriting, preserves Unicode, and
      writes nothing when the path is `None`.
- [ ] Add a small mutable `QueryRecorder` internal to this module. It owns the
      trace ID, start timestamp, stage stopwatch values and fields collected
      before a frozen `QueryTrace` can exist. Keep mutation out of the public
      models.
- [ ] Give the recorder explicit methods for the orchestration seam:

```python
recorder.resolved(query, elapsed_ms)
recorder.retrieved(chunks, elapsed_ms)
recorder.context_built(chunks, used, system, user, elapsed_ms)
recorder.generated(response, usage, elapsed_ms)
recorder.completed(answer, settings) -> QueryTrace
recorder.failed(error, total_ms) -> QueryTrace
```

- [ ] `completed()` derives `used` and `cited` from the actual chunks/sources,
      computes cost only when both prices exist, and drops prompt/response bodies
      unless `trace_content` is true.
- [ ] `write_trace(path, trace)` appends `trace.model_dump_json()` plus `\n` in
      UTF-8. Mark its single-process assumption:

```python
# shortcut: append is single-process; switch to OTLP or a writer service before multi-worker API deployment.
```

- [ ] Keep the sink free of global state. Tests pass a temporary path; production
      reads `settings.trace_path` once at recorder creation.
- [ ] Run the Task 1 commands again.

---

## Task 3: Trace generated answers at the source of truth

**Files:**

- Modify: `app/generation/answer.py`
- Modify: `tests/test_generation_answer.py`
- Modify: `tests/test_observability_trace.py`

- [ ] Add an explicit `trace: QueryRecorder | None = None` keyword to
      `answer_question()`. It is runtime instrumentation, not part of answer
      semantics.
- [ ] Ownership rule: when tracing is configured and no recorder was supplied,
      `answer_question()` creates and finishes one; when supplied, it only fills
      it. This is how direct benchmark calls write once and cache misses do not
      write twice.
- [ ] Time and record these existing boundaries without moving them:

  1. `contextualize()`;
  2. `retriever()`;
  3. compression, guard and `build_context()` together;
  4. model call plus citation validation.

- [ ] Record the exact `system` and formatted `user` strings passed to `llm`, not
      a reconstruction. For `no_context`, record no model prompt and an empty
      usage dictionary because no model call occurred.
- [ ] Add tests proving ordered IDs, raw and rerank scores, used/cited flags,
      resolved query, exact prompt/response, tokens and timings come from the
      same call. Add a `TRACE_CONTENT=false` case that keeps metadata and omits
      bodies.
- [ ] Add a raised-retriever test: one failed trace is written, the original
      exception is re-raised, and a sink error does not mask it.
- [ ] On a successful answer, a sink error returns the answer with one
      `trace write failed: ...` warning. Do not make tracing a new failure mode.
- [ ] Run:

```powershell
uv run pytest tests/test_generation_answer.py tests/test_observability_trace.py -q
uv run ruff check app/generation/answer.py app/observability tests/test_generation_answer.py
uv run mypy app
```

---

## Task 4: Trace cache hits and count repeated follow-ups

**Files:**

- Modify: `app/generation/cache.py`
- Modify: `tests/test_generation_cache.py`
- Modify: `tests/test_observability_trace.py`

- [ ] Add an explicit recorder parameter outside `**kwargs` so instrumentation
      can never enter `fingerprint()` or change a cache key.
- [ ] When tracing is configured, `cached_answer()` owns the recorder and passes
      it into the default `answer_question()` on a miss. Finish it exactly once
      after the lookup or generated answer.
- [ ] Test exact and semantic hits. Their traces have the cache layer, stored
      answer metadata and lookup total latency, but no retrieved chunks, prompt
      or generation timing because those operations did not occur.
- [ ] Test a call with history:

  - miss: `follow_up=true`, `follow_up_repeat=false`;
  - exact hit: `follow_up=true`, `follow_up_repeat=true`;
  - no history exact hit: both false.

- [ ] Preserve fail-open cache behaviour. A cache lookup failure followed by a
      successful generation produces one generated trace with the existing
      cache warning.
- [ ] Run:

```powershell
uv run pytest tests/test_generation_cache.py tests/test_generation_answer.py tests/test_observability_trace.py -q
uv run ruff check app/generation/cache.py app/observability tests/test_generation_cache.py
uv run mypy app
```

---

## Task 5: Make traces inspectable without a dashboard

**Files:**

- Create: `scripts/trace_report.py`
- Create: `tests/test_trace_report.py`

- [ ] Implement `load_traces(path) -> list[QueryTrace]`; report a line number for
      invalid JSON/schema instead of silently skipping it.
- [ ] Implement a pure `summarize(traces)` returning:

  - total, generated, refused and failed;
  - exact and semantic hits;
  - follow-ups, exact-repeat follow-ups and repeat rate (or `None` with no
    follow-ups);
  - total latency p50/p95;
  - prompt/completion tokens and known USD cost totals;
  - content-capture state counts.

- [ ] CLI: `uv run python scripts/trace_report.py [PATH]`; default to
      `TRACE_PATH`, fail clearly if both are empty, and print no raw prompt,
      response or query content.
- [ ] Test an empty file, mixed generated/hit/refusal records, no-follow-up
      division, and malformed line reporting.
- [ ] Run:

```powershell
uv run pytest tests/test_trace_report.py tests/test_observability_trace.py -q
uv run ruff check scripts/trace_report.py tests/test_trace_report.py
uv run mypy app
```

---

## Task 6: Measure Rule O

**Files:**

- Create: `scripts/benchmark_observability.py`
- Create: `tests/test_benchmark_observability.py`
- Create: `data/eval/observability.jsonl` (one measured summary row)

- [ ] Build a deterministic representative `QueryTrace`: twenty ranked chunks,
      a 4,000-character user prompt, a response, and typical token usage. No
      model, Redis or Qdrant call is allowed.
- [ ] Write it 1,000 times to a temporary file using the production sink. Record
      p50/p95 append latency, bytes per trace, schema-valid line count and total
      bytes. Delete the temporary file normally when the process exits; never
      write benchmark payloads into the repository.
- [ ] Implement Rule O's mechanical verdict: 1,000 valid lines and p95 below
      5 ms. Append only the summary row, including timestamp and git commit, to
      `data/eval/observability.jsonl`.
- [ ] Unit-test the percentile/verdict functions with small fixed inputs.
- [ ] Run:

```powershell
uv run python scripts/benchmark_observability.py
uv run pytest tests/test_benchmark_observability.py -q
```

- [ ] If Rule O fails, do not hide it by raising the threshold. Buffer writes or
      move the sink off the request path, rerun, and record both rows.

---

## Task 7: End-to-end proof, docs and milestone

**Files:**

- Modify: `README.md` (French)
- Modify: `docs/roadmap.md`

- [ ] Start Qdrant and Redis, set a temporary ignored `TRACE_PATH`, and make:

  1. one cold standalone query;
  2. the same query for an exact hit;
  3. one follow-up with history;
  4. the same follow-up for an exact-repeat hit.

- [ ] Run `scripts/trace_report.py` and verify four records, two exact hits, two
      follow-ups and one exact-repeat follow-up. This run spends at most two
      generation calls. Record the configured model and prices; never commit the
      trace file itself.
- [ ] README: add Phase 15 in the established French voice, the trace schema,
      privacy switches, report command, Rule O numbers and the follow-up counts.
      Check the Phase 15 box only after the proof passes.
- [ ] Roadmap: change current state to Step 24 done/Step 25 next, add the
      observability artifact row and step-map result, consume the follow-up
      assumption with the measured count, and mark only steps 25-30 unplanned.
- [ ] Run every quality gate:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
git diff --check
```

- [ ] Commit in reviewable slices, then tag the next free version after `v1.2`.
      Do not merge to `main` inside this plan; hand off the green branch and its
      measured verdict first.
