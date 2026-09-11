# Step 04 — Basic Chunking

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Split clean documents into overlapping chunks that carry full metadata, at the brief's baseline settings — `chunk_size=1000`, `overlap=200`. This is the baseline every later chunking experiment is measured against, so it must be boring and reproducible, not clever.

**Architecture:** One pure function, `chunk_document(document) -> list[Chunk]`, built on a generic recursive text splitter. Hand-written, per the brief's explicit instruction not to start with LangChain.

**Tech stack:** stdlib. No `langchain-text-splitters` — recursive character splitting is about 40 lines, and writing them is the point of the step.

## Decisions

**Characters, not tokens.** `chunk_size` counts characters. A token-accurate splitter means `tiktoken`, a model-specific encoding, and a dependency, to buy a unit that only matters when you are fighting a context limit. You are not, yet. Document the ratio (~4 chars per token for English prose, worse for code) and move on. Step 20 revisits this when context budget becomes real.

**Recursive splitting on a separator hierarchy**, most semantic first:

```python
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
```

Try to split on the first separator that yields pieces under `chunk_size`; recurse into pieces still too large with the next separator. The `""` terminal guarantees termination on pathological input (a 5000-character single-token blob) instead of looping or raising.

**Code fences are atomic.** A fence split down the middle produces two chunks of garbage: one missing its opening, one missing its context. So a fenced block is never split internally. A block larger than `chunk_size` becomes its own oversized chunk — deliberately, with a size warning logged. This is the known ceiling of the step; it is marked in code.

**Overlap is measured backwards from the chunk boundary** and snapped to the nearest preceding separator, so an overlap starts at a sentence or line boundary rather than mid-word.

**Metadata is attached now, filtered later.** Step 13 is "metadata filtering", but the fields must be carried from the moment chunks exist or every chunk in the index needs rebuilding. Every `Chunk` carries `document_id`, `source`, `title`, `url`, `language`, `section`, `chunk_index`, `char_start`, `char_end`.

**`section` is the nearest enclosing heading**, tracked while splitting. It is what turns a citation from "FastAPI docs" into "FastAPI — Dependencies / Sub-dependencies", and it costs one variable to maintain.

**`chunk_id = f"{document_id}#{chunk_index}"`.** Readable, and it makes the step 10 decision explicit: chunk ids *change* when chunking parameters change, which is exactly why ground truth is labelled at document level.

## File map

- Create `app/models/chunks.py` — `Chunk`.
- Create `app/ingestion/chunk.py` — `split_text()`, `chunk_document()`, `chunk_documents()`.
- Create `tests/test_ingestion_chunk.py`.
- Modify `README.md`.

## Tasks

### Task 1: The chunk model

- [x] **Step 1: Failing test** — a `Chunk` with `char_end <= char_start` is rejected; `chunk_id` is derived, not passed in.

- [x] **Step 2: Write `app/models/chunks.py`**

Frozen pydantic model: `document_id`, `source`, `title`, `url: str | None`, `language`, `section: str | None`, `chunk_index: int`, `text: str` (`min_length=1`), `char_start: int`, `char_end: int`. A computed `chunk_id` property returning `f"{document_id}#{chunk_index}"`. A model validator rejecting `char_end <= char_start`.

Add `to_payload()` returning the flat dict that step 06 stores in Qdrant. Keeping the payload shape next to the model means the indexer and the search result mapper cannot drift apart.

- [x] **Step 3: Green.** Commit: `feat(ingestion): add Chunk model`.

### Task 2: The recursive splitter

- [x] **Step 1: Write the failing tests**

1. Text shorter than `chunk_size` yields exactly one chunk equal to the input.
2. No chunk exceeds `chunk_size`, **except** a chunk that is a single oversized code fence.
3. **Coverage:** concatenating chunks with overlaps removed reproduces the original text. Assert this on a 10 KB fixture — it is the test that catches dropped content, the worst possible chunking bug.
4. Consecutive chunks overlap, and the overlap is between 1 and `overlap` characters.
5. A fenced code block of 300 characters inside 2000 characters of prose appears **whole** inside exactly one chunk.
6. A fenced block of 1500 characters with `chunk_size=1000` yields one oversized chunk containing the whole fence, and a warning is logged.
7. Splitting prefers `\n## ` boundaries: a document with three well-sized `##` sections yields chunks that each start at a heading.
8. `chunk_index` is contiguous from 0, and `char_start`/`char_end` slice the source back to the chunk text.
9. `section` on a chunk after `## Dependencies` is `"Dependencies"`; chunks before any heading have `section is None`.
10. Degenerate inputs: empty string → `[]`; whitespace-only → `[]`; a single 5000-character word with no separators → chunks of exactly `chunk_size`, no infinite loop. Put a hard iteration guard in the splitter and test that it is never hit.

- [x] **Step 2: Write `app/ingestion/chunk.py`**

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]: ...
def chunk_document(document: RawDocument, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]: ...
def chunk_documents(documents: Iterable[RawDocument], **kwargs) -> list[Chunk]: ...
```

`split_text` returns `(start, end)` offset pairs rather than strings — offsets make the coverage test possible, give `char_start`/`char_end` for free, and keep one copy of the text in memory.

Implementation notes:

- Find fenced-block spans first (reuse the regex from step 03, or move it to a shared helper — do not write it twice). Treat each span as an unsplittable unit.
- Recurse: for each candidate piece too long, re-split with the next separator in the list.
- Track the current heading while walking, for `section`.
- Leave a `ponytail:` comment on the oversized-fence branch naming the ceiling: *oversized code fences become single oversized chunks; revisit if the eval set shows them hurting recall.*

- [x] **Step 3: Green, `mypy app` clean.** Commit: `feat(ingestion): split documents into overlapping chunks`.

### Task 3: Chunk the real corpus and look at the distribution

- [x] **Step 1: Measure**

```powershell
uv run python -c "from pathlib import Path; from statistics import median; from app.ingestion.loader import load_documents; from app.ingestion.clean import clean_document; from app.ingestion.chunk import chunk_documents; docs=[d for d in (clean_document(x) for x in load_documents(Path('data/raw/fastapi'),'fastapi')) if d]; cs=chunk_documents(docs); sizes=sorted(len(c.text) for c in cs); print('docs',len(docs),'chunks',len(cs),'median',median(sizes),'max',sizes[-1],'oversized',sum(1 for s in sizes if s>1000))"
```

Record: document count, chunk count, chunks per document, median and max chunk size, oversized-chunk count.

- [x] **Step 2: Sanity-read ten chunks** — the longest, the shortest, three random ones, and every oversized one. Each should be readable on its own. A chunk that is pure navigation links or a lone heading is a cleaning bug, not a chunking bug: fix it in step 03's module and add the test there.

- [x] **Step 3: Commit the measurements** in the message.

### Task 4: Documentation

- [x] Update `README.md`: current state, Phase 1 roadmap, and a short "Chunking" note giving the baseline parameters and the measured distribution. Phase 2 will compare against exactly these numbers, so they need to be written down somewhere permanent. Commit: `docs: record the chunking baseline`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
```

## Definition of done

- Chunks reconstruct the source text with no loss (coverage test green).
- No code fence is ever split internally.
- Every chunk carries full metadata including `section`.
- The baseline distribution (count, median, max, oversized) is recorded in the README.
- Degenerate inputs terminate.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| Token-accurate sizing | context budget actually binds (step 20) |
| Semantic / parent-child / sentence chunking | step 12, where they get compared on numbers |
| Configurable separators per source | a second source needs a different hierarchy |
| A `Chunker` protocol | step 12 needs to swap strategies behind one call |
| Table-aware splitting | the eval set shows split tables losing answers |
