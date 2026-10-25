# Step 29 — Benchmark dashboard

## Goal

Turn the benchmark histories already committed by steps 11, 21, 23, and 24
into one inspectable dashboard:

```text
data/eval/{results,answers,cache,observability}.jsonl
    -> validated compact snapshot
    -> dashboard/index.html
```

The dashboard compares retrieval quality and latency, answer quality and
refusals, and the cache/observability verdicts. It does not run a benchmark or
invent a metric.

## Measured starting point

- `results.jsonl`: 40 retrieval runs, 1.1 MB because every row retains its
  per-question evidence.
- `answers.jsonl`: 20 answer runs, including the RAGAS arms.
- `cache.jsonl`: 3 cache experiments.
- `observability.jsonl`: 1 write-overhead experiment.
- The histories contain 64 rows in total and are already versioned. Step 11
  explicitly named `results.jsonl` as this dashboard's source.
- Step 28 is complete: the first post-merge `main` workflow passed both the
  quality/coverage and real retrieval-regression jobs.

## Decision

Build a static, offline dashboard with no new dependency.

- Python's standard library validates JSONL and emits only aggregate fields to
  `dashboard/data.js`; per-question text and evidence stay out of the browser
  snapshot.
- Plain HTML, CSS, and JavaScript render the charts, filters, and accessible
  tables. No CDN is required, so opening `dashboard/index.html` works offline.
- The latest row for a repeated label is the comparison row. The full histories
  remain untouched and continue to be the audit trail.
- Retrieval and answer runs remain separate because their metrics have
  different denominators and must not be plotted as if they were one series.

## Execution

1. Add tests for line-numbered JSONL validation, latest-label selection, compact
   snapshot shape, and deterministic JavaScript output.
2. Add `app/evaluation/dashboard.py` and `scripts/build_dashboard.py`.
3. Add the static dashboard shell and generate its committed `data.js` snapshot.
4. Inspect the dashboard at desktop and mobile widths, then run the repository
   quality and coverage gates.
5. Update the README and roadmap with the actual dashboard contents and command.

## Definition of done

- One command rebuilds the dashboard from committed histories without network
  access or credentials.
- Invalid JSON names its file and line instead of being skipped.
- Repeated labels appear once, using their newest row.
- No `per_question` or `per_probe` payload enters the browser snapshot.
- Retrieval runs can be searched, filtered, compared in a quality/latency plot,
  and inspected by category.
- Answer, cache, and observability results have their own truthful summaries.
- The generated dashboard opens directly from disk and remains usable on a
  narrow viewport.

## Explicitly deferred

- A live benchmark runner: benchmark commands already own execution, and a Web
  UI that can spend API calls expands the trust boundary for no step-29 need.
- A hosted dashboard and authentication: there is no deployment target.
- Plotting per-question content: aggregates answer the comparison question and
  avoid publishing query or answer text.
- A frontend framework or chart package: the data set is small and the two
  required plots fit the browser platform.
