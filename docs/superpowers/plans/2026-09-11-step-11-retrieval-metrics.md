# Step 11 — Retrieval Metrics and Benchmark

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Turn the step 10 dataset into numbers: Recall@K, Precision@K, MRR, Hit Rate, NDCG@K, plus latency. Produce the baseline row that every later step is compared against. This is `v0.4` and the point at which the project becomes presentable.

**Architecture:** `app/evaluation/metrics.py` is pure arithmetic over ranked id lists — no Qdrant, no embeddings, no I/O. `app/evaluation/benchmark.py` runs the dataset through a retriever and aggregates. `scripts/benchmark.py` is the CLI that writes a results row.

**Tech stack:** stdlib `math`, `statistics`, `json`. No new dependency — these five metrics are about 40 lines of arithmetic, and `pytrec_eval` or `ranx` would be a dependency plus a format conversion for code you should be able to write and explain in an interview.

## Decisions

**Metrics take `list[str]` of retrieved document ids and `set[str]` of relevant ids. Nothing else.** Not `ScoredChunk`, not a Qdrant response. That keeps every metric testable against hand-computed examples, which is the only way to know they are right. A wrong MRR implementation produces plausible numbers forever.

**Chunk results are deduplicated to documents before scoring, preserving best rank.** Retrieval returns chunks; labels are documents (step 10). Two chunks from the same document are one retrieved document, at the better of the two ranks. Not deduplicating inflates Precision@K and quietly makes chunking strategies that return many small chunks from one page look better than they are — the exact bug that would corrupt step 12.

**`@K` is computed on the deduplicated document list.** State this in the README. `Recall@5` meaning "5 chunks" versus "5 distinct documents" are different numbers, and a benchmark whose definition is ambiguous is not a benchmark.

**`unanswerable` questions are excluded from Recall/Precision/MRR/NDCG and scored separately.** Recall over an empty relevant set is undefined; dividing by zero or scoring it as 1.0 both distort the aggregate. They get their own metric: *abstention rate*, the fraction where the top result's score falls below a threshold, recorded now and acted on in step 22.

**Per-category breakdown is the primary output, not the single aggregate.** One overall Recall@5 hides the entire story. The `exact` category is expected to be poor; when step 14 lifts `exact` from 0.4 to 0.8 while leaving `conceptual` flat, *that* is the finding worth writing up — and it is invisible in an average.

**Latency is recorded as p50 and p95, not a mean.** Retrieval latency is long-tailed; a mean hides the tail that a user actually notices. Two lines with `statistics.quantiles`.

**Results are appended to a JSONL history file, not overwritten.** `data/eval/results.jsonl`, one row per run, with a git commit hash, a timestamp, the config that produced it, and the numbers. Committed. This file *is* the README's results table and step 29's dashboard; reconstructing it from git history later is miserable.

**Held-out questions are excluded by default** and run only with `--include-held-out`.

## File map

- Create `app/evaluation/metrics.py` — the five metrics plus helpers.
- Create `app/evaluation/benchmark.py` — `run_benchmark()`, `BenchmarkResult`.
- Create `scripts/benchmark.py` — CLI, writes JSONL and a Markdown row.
- Create `data/eval/results.jsonl`.
- Create `tests/test_evaluation_metrics.py`, `tests/test_evaluation_benchmark.py`.
- Modify `README.md`.

## Tasks

### Task 1: The metrics

- [ ] **Step 1: Failing tests with hand-computed expected values.** Every number below is worked out by hand in the test, with the arithmetic in a comment. This is the one module in the project where "looks about right" is not acceptable.

Helpers:
1. `dedupe_to_documents(["a#1", "a#2", "b#0"])` → `["a", "b"]`, preserving best rank.
2. Deduplication keeps first (best) occurrence order.

Recall@K:
3. Relevant `{a, b}`, retrieved `[a, x, b]`, K=3 → 1.0.
4. Same, K=2 → 0.5.
5. Relevant `{a}`, retrieved `[x, y]` → 0.0.
6. K larger than the retrieved list does not raise and does not pad.

Precision@K:
7. Relevant `{a, b}`, retrieved `[a, x, b]`, K=3 → 2/3.
8. Fewer than K results retrieved: the denominator is K, not the list length. Document this choice in a docstring — both conventions exist and the comparison is only valid if one is used consistently.

MRR:
9. First relevant at rank 1 → 1.0; rank 2 → 0.5; rank 4 → 0.25.
10. No relevant result → 0.0.
11. MRR uses only the *first* relevant hit, even with three relevant documents retrieved.

Hit Rate@K:
12. Any relevant document in the top K → 1.0, else 0.0.

NDCG@K:
13. Relevant `{a, b}`, retrieved `[a, b, x]`, K=3 → 1.0 (perfect ranking).
14. Retrieved `[x, a, b]`, K=3 → DCG = 1/log2(3) + 1/log2(4), IDCG = 1/log2(2) + 1/log2(3); assert the exact ratio.
15. Binary gains, log2 discount, IDCG built from `min(len(relevant), K)` — a relevant set larger than K must not make a perfect ranking score below 1.0.
16. No relevant result → 0.0.

Edge cases:
17. Empty retrieved list → every metric 0.0, no exception.
18. Empty relevant set raises `ValueError` — that case belongs to the abstention path, and silently returning 0.0 or 1.0 would poison the aggregate.
19. K=0 raises.

- [ ] **Step 2: Write `app/evaluation/metrics.py`.** Plain functions, full docstrings stating the convention each one follows.

- [ ] **Step 3: Green.** Commit: `feat(evaluation): add retrieval metrics`.

### Task 2: The benchmark runner

- [ ] **Step 1: Failing tests** with a fake retriever returning scripted results.

1. A 3-question dataset produces per-question rows and an aggregate.
2. Aggregates are means over answerable questions only.
3. `unanswerable` questions appear in the abstention metric and **nowhere** in Recall/MRR/NDCG. Assert the answerable count, not just the values.
4. Per-category breakdown keys match the categories present in the dataset.
5. A retriever raising on one question records the failure and continues — losing a 40-question run to one transient API error is unacceptable. Assert the failure count surfaces in the result.
6. Latency p50/p95 are populated.
7. The result dict is JSON-serialisable (no numpy types, no `Path`, no pydantic object) — this is what gets appended to the history file.

- [ ] **Step 2: Write `app/evaluation/benchmark.py`**

```python
def run_benchmark(
    questions: Sequence[EvalQuestion],
    retriever: Callable[[str], list[ScoredChunk]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    label: str,
    config: Mapping[str, Any] | None = None,
) -> BenchmarkResult: ...
```

Retrieve `max(ks)` results once per question and compute every K from that single list. Re-querying per K quadruples the run time and cost for identical results.

`BenchmarkResult` holds `label`, `timestamp`, `git_commit`, `config`, `aggregate`, `per_category`, `per_question`, `latency_p50_ms`, `latency_p95_ms`, `failures`.

- [ ] **Step 3: Green.** Commit: `feat(evaluation): add the benchmark runner`.

### Task 3: The CLI

- [ ] **Step 1: Write `scripts/benchmark.py`**

Arguments: `--label` (required — an unlabelled row in the history is useless), `--top-k`, `--source`, `--include-held-out`, `--no-save`, `--compare <label>`.

Output: a Markdown table of the aggregate, a per-category table, latency, failures. Append one JSON row to `data/eval/results.jsonl` unless `--no-save`. `--compare` prints a delta table against a previous label — the thing you will actually look at for the next twenty steps, so build it now rather than eyeballing two tables side by side forever.

Capture the git commit with `git rev-parse --short HEAD` and mark it dirty when the worktree is not clean. A benchmark row attributed to a clean commit that was actually run on uncommitted code is a lie you will believe later.

- [ ] **Step 2: Run the baseline**

```powershell
docker compose up -d qdrant --wait
uv run python scripts/benchmark.py --label "dense-baseline"
```

- [ ] **Step 3: Read the per-category numbers carefully.** Expectations worth checking against reality:
  - `exact` should be clearly worse than `conceptual`. If it is not, the `exact` questions are not exact enough — fix the dataset, not the retriever.
  - `multi_doc` Recall@5 should be low, because 5 slots rarely cover 3 documents.
  - Recall@10 well above Recall@5 means reranking (step 17) has room to work; similar values mean the retriever is missing documents entirely and reranking cannot help.

  Write these observations down. They are the reasoning that makes steps 12-19 deliberate instead of a list of techniques applied because a blog post said so.

- [ ] **Step 4: Commit.** `feat(evaluation): add the benchmark CLI` plus the baseline results row.

### Task 4: Documentation and the tag

- [ ] **Step 1: Fill in the README results table** with the real baseline — the first row with numbers instead of em dashes. Add the per-category table and the metric definitions (especially the document-level `@K` convention and the Precision@K denominator). Add the benchmark command.
- [ ] **Step 2: Add a short "Méthode" note** stating the rule: every later row is produced by the same command against the same dataset, and regressions get published alongside improvements.
- [ ] **Step 3: Commit and tag.** `docs: publish the retrieval baseline`, then `git tag v0.4`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
uv run python scripts/benchmark.py --label "smoke" --no-save
```

Metric unit tests must pass with no Qdrant and no API key.

## Definition of done

- Five metrics, each verified against a hand-computed example.
- Chunk results deduplicated to documents before scoring.
- `unanswerable` questions excluded from ranking metrics and measured as abstention.
- Per-category breakdown and p50/p95 latency reported.
- One baseline row in `data/eval/results.jsonl` and in the README, with a git commit attached.
- `--compare` works.
- `v0.4` is tagged.

## After this step

Steps 12-30 each get their own plan, written at the start of the step. They are not written yet **because their designs depend on these numbers**: which chunking strategy wins, whether BM25 helps this corpus, whether reranking earns its latency, whether multi-query is worth the cost. Writing those plans before the baseline exists would mean inventing the answers, which is precisely the habit this step exists to break.

The next plan to write is step 12 (chunking experiments), and its first input is the per-category table produced here.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| `ranx` / `pytrec_eval` | you need a metric that is genuinely hard to implement correctly |
| Statistical significance testing | two candidates differ by less than a question or two and the call is unclear |
| Graded relevance | binary NDCG stops discriminating |
| Generation metrics | step 21 (RAGAS) |
| A plotted dashboard | step 29 — the JSONL history is already the data source |
| Cost tracking per run | step 24 (observability) |
