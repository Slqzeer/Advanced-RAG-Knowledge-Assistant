# Step 12 — Chunking Experiments

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [x]` for tracking.

**Goal:** Compare four chunking strategies against the step 11 baseline, pick one winner by Recall@5, promote it, and publish the table — including the strategies that lost. This is the first step where the project behaves like a search system instead of an LLM wrapper.

**Baseline to beat** (`dense-baseline`, commit `4640e02`, recursive @ 1000/200):

| | Recall@5 | Recall@10 | MRR | NDCG@5 |
|---|---|---|---|---|
| **overall** (38 answerable) | **0.713** | 0.737 | 0.788 | 0.658 |
| exact (10) | 0.792 | 0.825 | 0.950 | 0.789 |
| code (10) | 0.800 | 0.800 | 0.833 | 0.743 |
| conceptual (10) | 0.633 | 0.667 | 0.478 | 0.456 |
| multi_doc (8) | 0.604 | 0.635 | 0.917 | 0.638 |

Latency p50 57 ms, p95 89 ms. Abstention rate 0.143.

Recall@10 sits only 0.024 above Recall@5. The retriever is not mis-ranking documents it found — it is **missing them entirely**, which is exactly the failure chunking can move and reranking (step 17) cannot.

**Architecture:** four splitter functions in `app/ingestion/chunk.py` behind a `STRATEGIES` dict, all with the same `text -> list[tuple[int, int]]` signature, so `chunk_document` and everything downstream are untouched. One Qdrant collection per strategy. The comparison table is generated from `data/eval/results.jsonl`, which already holds every number.

**Tech stack:** no new dependency. Sentence splitting is a regex; semantic splitting is cosine distance over sentence embeddings the project already knows how to produce. `nltk`, `spacy` and `langchain-text-splitters` each buy one function's worth of code in exchange for a model download, a tokenizer, or an abstraction the next four steps would have to route around.

## Decisions

**Every strategy returns character offsets, not strings.** Same contract `split_text` already has. It keeps the reconstruction test — drop each overlap, get the original text back — applicable to all four, and that test is the only thing standing between a subtly lossy splitter and a benchmark number that looks fine.

**`fixed` is deliberately naive and must stay that way.** Hard cut every `size - overlap` characters: no separators, no code protection, mid-word boundaries. It is the control. A `fixed` that quietly protects code fences is not a floor, and the comparison loses its bottom. The test asserts it splits a fence, so nobody "fixes" it in six months.

**`semantic` takes an embedder as an argument, never an import.** `app/ingestion/chunk.py` is network-free and its tests run with no API key. Threading `embed_batch: Callable[[list[str]], list[list[float]]]` through the one strategy that needs it preserves both, and makes the interesting behaviour — where does it cut? — testable against scripted vectors instead of against OpenAI.

**Semantic threshold is a per-document percentile, not a global constant.** Cut where the cosine distance between adjacent sentences exceeds that document's own 95th percentile. A fixed threshold like 0.15 means something different on a tutorial page than on an API reference; a percentile adapts and has no magic number to defend. It is exposed as a parameter so the sweep can move it if the number turns out to matter.

**One collection per strategy: `chunks_{strategy}`.** The alternative — `--recreate` the one collection before each run — makes every comparison irreproducible without a full reindex, and makes "why did semantic lose this question?" unanswerable once the next variant is indexed. Five collections cost about 10 MB each and one optional `collection` argument on `search()`.

**The embedding cache is shared, not per-strategy.** It is keyed on `sha256(model + text)`, so identical spans across strategies hit, and only genuinely-new text costs. The whole step is about $0.04 including the size sweep.

**No driver script.** `results.jsonl` already records label, config and aggregate per run. A `--summary` flag on `scripts/benchmark.py` that filters labels by glob and prints one table is ~20 lines; a driver script would be a `for` loop wrapping two CLIs that already do their jobs. If the matrix ever outgrows nine commands, write it then.

**The decision rule is written down before the first run.** Winner is the highest overall Recall@5. Anything inside **0.026** — one question out of 38 — is a tie, broken by `conceptual` Recall@5, because 0.633 is where the baseline is weakest. Round 3 runs only if round 2's spread exceeds that same 0.026. A rule invented after seeing the table is not a rule.

**Parent-child is not compared here, and the README says why.** Step 10 labels relevance at *document* level on purpose. Parent-child retrieval — embed a small child, return a large parent — changes which *text* reaches the LLM, not which *documents* reach the top-K. Recall@5 would score it identically to whatever child chunking it wraps. Measuring it needs context-token count and answer quality, which is step 20 and step 21. Reporting a Recall@5 for it would be a number with no information in it.

## File map

- Modify `app/ingestion/chunk.py` — `split_fixed`, `split_sentences`, `split_semantic`, `STRATEGIES`, `strategy` parameter on `chunk_document`/`chunk_documents`.
- Modify `app/core/config.py` — `chunk_strategy: str = "recursive"` (becomes the winner in Task 5).
- Modify `app/retrieval/search.py` — optional `collection` argument.
- Modify `scripts/index_corpus.py` — `--strategy`, `--collection`, `--chunk-size`, `--overlap`.
- Modify `scripts/benchmark.py` — `--collection`, `--summary <glob>`, `strategy` in the recorded config.
- Modify `tests/test_ingestion_chunk.py`, `tests/test_retrieval_search.py`.
- Modify `README.md`, `docs/roadmap.md`, `.env.example`.

## Tasks

### Task 1: The three new splitters

- [ ] **Step 1: Failing tests.** The existing coverage and reconstruction properties get applied to every strategy, parametrised — that is the whole reason offsets were the contract.

Shared properties, run against all four strategies:
1. Spans tile the text: no gaps, each span starts at or before the previous span's end.
2. Dropping each overlap reconstructs the input exactly.
3. No span is empty; every span is within `[0, len(text))`.
4. Blank or whitespace-only input returns `[]`.
5. `overlap >= chunk_size` raises `ValueError`.

`fixed`:
6. Cuts land exactly on `size - overlap` strides, regardless of content.
7. **Splits a fenced code block.** Asserted, not tolerated — this is the control and its naivety is the feature.

`sentence`:
8. Boundaries fall after `.`, `?`, `!` followed by whitespace, not after `e.g.` or a version number.
9. A sentence longer than `chunk_size` becomes its own oversized chunk rather than being cut mid-sentence.
10. Never cuts inside a `_protected_spans` region.
11. Overlap is whole sentences, never a partial one.

`semantic` (fake embedder, scripted vectors — no network, no key):
12. Three sentences whose vectors are `[1,0]`, `[1,0]`, `[0,1]` cut between the second and third and nowhere else.
13. Uniform vectors produce no semantic cut; the result is bounded by `chunk_size` alone.
14. A run of sentences exceeding `chunk_size` is cut at the size cap even with no distance spike.
15. Never cuts inside a `_protected_spans` region.
16. The embedder is called once, with every sentence, not once per sentence.

Registry:
17. `STRATEGIES` has exactly the four keys; `chunk_documents(..., strategy="nope")` raises with the valid names in the message.
18. `chunk_document(..., strategy="recursive")` is byte-identical to today's output — the baseline must not move by accident.

- [ ] **Step 2: Write the splitters.** Reuse `_protected_spans`, `_skip_protected`, `_boundary` and `_merge`; three of the four are packing loops over an existing helper. Keep `split_text` exactly as it is and register it as `recursive`.

- [ ] **Step 3: Green.** `uv run pytest tests/test_ingestion_chunk.py`. Commit: `feat(ingestion): add fixed, sentence and semantic chunking`.

### Task 2: Plumbing — collection and strategy through the CLIs

- [ ] **Step 1: Failing tests.** `search(query, collection="other")` queries `other` and not `settings.qdrant_collection`; omitting it falls back to settings. Fake client, as in the existing search tests.

- [ ] **Step 2: Thread the arguments.**
  - `search(..., collection: str | None = None)`.
  - `index_corpus.py`: `--strategy` (default from settings), `--collection` (default from settings), `--chunk-size`, `--overlap`. Print the strategy in the summary line — a run whose parameters are not on screen is a run you cannot attribute later.
  - `benchmark.py`: `--collection`, and `"strategy"` added to the recorded `config` dict alongside the `chunk_size`/`chunk_overlap` already there.
  - `config.py`: `chunk_strategy: str = "recursive"`; `.env.example` gets `CHUNK_STRATEGY=recursive` with a one-line comment.

- [ ] **Step 3: Green plus a smoke run.**

```powershell
uv run python scripts/index_corpus.py --strategy sentence --limit 5 --dry-run
```

Commit: `feat(ingestion): select chunking strategy and collection per run`.

### Task 3: The summary table

- [ ] **Step 1: Failing test.** `--summary` over a two-row fake history prints both labels with their Recall@5, and a glob matching nothing exits non-zero with a message rather than printing an empty table.

- [ ] **Step 2: Write it.** `--summary <glob>` reads `data/eval/results.jsonl`, keeps the last run per matching label, and prints one table: label, strategy, chunk_size, overlap, Recall@5, Recall@10, MRR, NDCG@5, conceptual Recall@5, p50 ms. Reuse the existing `table()` helper.

- [ ] **Step 3: Green.** Commit: `feat(evaluation): summarise benchmark history by label`.

### Task 4: Run the matrix

Qdrant up first: `docker compose up -d qdrant --wait`.

- [ ] **Step 1: Round 1 — four strategies at 1000/200.**

```powershell
foreach ($s in "recursive","fixed","sentence","semantic") {
  uv run python scripts/index_corpus.py --strategy $s --collection "chunks_$s" --recreate
  uv run python scripts/benchmark.py --label "chunk-$s-1000-200" --collection "chunks_$s"
}
uv run python scripts/benchmark.py --summary "chunk-*"
```

Record the chunk count per strategy from the indexer's summary line — it is half the explanation of any Recall difference, and it is not in `results.jsonl` otherwise.

- [ ] **Step 2: Apply the decision rule.** Highest overall Recall@5; ties inside 0.026 broken by `conceptual` Recall@5. Write the winner down before running anything else.

- [ ] **Step 3: Round 2 — size sweep on the winner.** 1000 is already done.

```powershell
foreach ($n in 500,1500) {
  uv run python scripts/index_corpus.py --strategy <winner> --collection "chunks_<winner>_$n" --chunk-size $n --overlap 200 --recreate
  uv run python scripts/benchmark.py --label "chunk-<winner>-$n-200" --collection "chunks_<winner>_$n"
}
```

- [ ] **Step 4: Round 3 — overlap, only if round 2 earned it.** If the spread across 500/1000/1500 is below 0.026, stop and say so in the README: the corpus is insensitive to chunk size in this range, which is itself a finding. Otherwise run overlap 0 and 400 at the best size.

- [ ] **Step 5: Read the per-category tables, not just the aggregate.** Things worth checking against reality, and worth writing down whichever way they land:
  - `fixed` should lose, and most visibly on `code` — it is the only strategy allowed to cut a fence open. If it does not lose, the other three are doing work the corpus does not reward, and that is the headline.
  - `semantic` should help `conceptual` most, since that is the category whose answers span a whole explanatory passage. If it helps `exact` instead, the threshold is cutting on formatting rather than on meaning.
  - Smaller chunks usually raise Recall@5 and lower precision, because more documents fit in five slots. If Recall@5 rises while `hit_rate@5` is flat, the gain is dedup arithmetic, not better retrieval — say so.
  - If no strategy beats `dense-baseline` by more than 0.026, **the baseline wins and step 12's result is "chunking is not the bottleneck here"**. That is a publishable outcome and the correct one to publish; it makes the case for step 14 stronger, not weaker.

- [ ] **Step 6: Commit the results.** `feat(evaluation): benchmark four chunking strategies` with the new `results.jsonl` rows.

### Task 5: Promote the winner

- [ ] **Step 1: Rewire the default.** `chunk_strategy` and, if the sweep moved them, `chunk_size`/`chunk_overlap` in `config.py` and `.env.example`.

- [ ] **Step 2: Reindex the primary collection and re-benchmark.**

```powershell
uv run python scripts/index_corpus.py --recreate
uv run python scripts/benchmark.py --label "dense-<winner>" --compare "dense-baseline"
```

This row, not `dense-baseline`, is what steps 13-30 compare against. Losing collections stay on disk until step 14 wants the space.

- [ ] **Step 3: Sanity-check the pipeline end to end**, because a chunking change touches every answer, not just every score:

```powershell
uv run python scripts/ask.py "Comment FastAPI gere les dependances avec Depends ?"
uv run python scripts/ask.py "HTTPException 422"
```

Citations must still resolve (step 09) and the out-of-corpus question must still answer "I do not know". A chunking strategy that raises Recall@5 while breaking citation resolution is a regression, and the benchmark cannot see it.

- [ ] **Step 4: Commit.** `feat(ingestion): promote <winner> chunking as the default`.

### Task 6: Documentation

- [ ] **Step 1: README.** The four-strategy table, the size sweep, the per-category deltas against `dense-baseline`, the new commands, the roadmap checkbox. Publish the losers with their numbers — a table with only the winner in it is marketing.
- [ ] **Step 2: The parent-child note.** One short paragraph: what it is, why a document-level Recall@5 cannot distinguish it, which step measures it instead. This is the most interesting thing the step learns and it belongs in the README, not only in this plan.
- [ ] **Step 3: Fix `docs/roadmap.md`.** Its "Current state" section still says steps 02-08 are done and step 09 is next; 09, 10 and 11 shipped and `v0.4` is tagged. Bring it up to date, add step 12's row, and add this plan to the plans table.
- [ ] **Step 4: Commit.** `docs: publish the chunking comparison`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
uv run python scripts/benchmark.py --summary "chunk-*"
```

Every chunking test must pass with no Qdrant, no API key and no network — including the semantic one, which is why the embedder is an argument.

## Definition of done

- Four strategies behind one registry, each covered by the tiling and reconstruction properties.
- `fixed` is asserted naive; `recursive` is asserted byte-identical to the pre-step baseline.
- One collection per strategy; `search()` takes an optional collection.
- Round 1 run and summarised; the decision rule applied as written, before the numbers were seen.
- Size sweep on the winner; round 3 run, or explicitly skipped with its reason recorded.
- Winner promoted, `chunks` reindexed, `dense-<winner>` row in `results.jsonl` with a `--compare` delta against `dense-baseline`.
- `ask.py` still answers and still cites after the reindex.
- README carries all four strategies with their numbers, the parent-child note, and the roadmap is no longer stale.

## After this step

Step 13 (metadata filtering) is next and is cheap: the payload indexes on `source`/`document_id`/`language` already exist from step 06, so it is a query-time change plus per-source recall numbers.

The number to watch across steps 12-17 is **Recall@10 minus Recall@5**, currently 0.024. While that gap stays near zero, the retriever is missing documents and reranking cannot help; step 14's hybrid search is the fix. Once hybrid opens the gap, step 17's reranker has something to work with. If chunking widens it here, say so — it changes which of the next steps matters most.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| Parent-child retrieval | step 20 — context tokens and answer quality are its metrics, Recall@5 is blind to it |
| Token-based chunk sizes | step 20, when a context limit actually binds |
| The full 24-run grid | the winner's size curve is not flat *and* a strategy x size interaction is plausible |
| Chunk-level relevance labels | never — document-level labels are precisely what makes re-chunking comparable |
| A driver script for the matrix | the matrix outgrows nine commands |
| Tuning the semantic percentile | the four-strategy table shows semantic close enough to the winner that the threshold could decide it |
| Deleting the losing collections | step 14 needs the disk |
