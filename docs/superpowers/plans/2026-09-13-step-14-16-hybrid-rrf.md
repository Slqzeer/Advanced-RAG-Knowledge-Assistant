# Steps 14-16 — Hybrid Search and RRF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hand-rolled BM25 retriever alongside dense retrieval, fuse the two with Reciprocal Rank Fusion, and measure whether the combination beats dense retrieval on this corpus.

**Architecture:** A new `app/retrieval/bm25.py` builds an inverted index from the chunks already stored in Qdrant, so the lexical and dense retrievers rank the same population. `app/retrieval/search.py` grows a `RETRIEVERS` registry keyed `"dense" | "lexical" | "hybrid"` and a `mode=` parameter defaulting to a new `RETRIEVAL_MODE` setting — the same registry-plus-setting pattern step 12 used for chunking strategies, so `search()`'s signature and return type stay backward compatible and step 17 registers a fourth key instead of re-plumbing five call sites. Fusion is rank-based only; no score normalisation is introduced anywhere.

**Tech Stack:** Python 3.12, pydantic v2, qdrant-client, pytest. **No new dependencies.**

**Spec:** [`docs/superpowers/specs/2026-09-13-hybrid-rrf-design.md`](../specs/2026-09-13-hybrid-rrf-design.md) — read it before starting. It carries the measured facts, the six rejected alternatives, and the pre-registered acceptance rule that Task 6 applies.

## Global Constraints

- **No new dependencies.** Roadmap rule 3. `pyproject.toml` and `uv.lock` are untouched by every task in this plan. If you reach for `rank_bm25`, `fastembed` or `nltk`, stop — the spec rejects all three by name.
- **Quality gates must pass before every commit:** `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.
- **TDD, no exceptions.** Write the failing test, run it, watch it fail for the *stated* reason, then write the code.
- **Line length 100**, enforced by Ruff. `app` is type-checked with mypy `strict`.
- **`data/eval/questions.jsonl` is frozen.** No task in this plan adds, removes or re-annotates an evaluation question. The spec explains why at length.
- **Every benchmark run targets the same collection as the baseline:** always pass `--collection chunks_sentence --strategy sentence`. Without them the run silently uses the `chunks` collection and is not comparable to anything in `data/eval/results.jsonl`.
- **Tests never touch the network, Qdrant, or an API key.** Inject `client`, `embedder` and `index`, following the existing pattern in `tests/test_retrieval_search.py`.
- **Baseline to beat, from the `dense-sentence-doctype` row:** Recall@5 **0.776**, Recall@10 **0.785**, MRR **0.810**, NDCG@5 **0.713**. Per category Recall@5: exact 0.892, code 0.850, conceptual 0.733, multi_doc 0.594.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `app/retrieval/bm25.py` | create | Tokenising, the BM25 inverted index and its scoring, building one from a Qdrant collection, and the process-wide cached accessor. Knows nothing about dense retrieval or fusion. |
| `app/retrieval/search.py` | modify | Composition only: filter translation (both directions), RRF, the three mode implementations, and the `search()` dispatcher. |
| `app/retrieval/store.py` | modify | One stale comment corrected (line ~55). |
| `app/core/config.py` | modify | Three new settings. |
| `app/generation/answer.py` | modify | Thread `mode` through to the retriever. |
| `app/evaluation/benchmark.py` | modify | One new summary column. |
| `scripts/search.py` | modify | `--mode`. |
| `scripts/ask.py` | modify | `--mode`. |
| `scripts/benchmark.py` | modify | `--mode`, `--candidates`, `--rrf-k`; pre-loop index build; record all three in `config`. |
| `tests/test_retrieval_bm25.py` | create | The index in isolation: tokenising, IDF, length normalisation, exact-token retrieval, scrolling. |
| `tests/test_retrieval_search.py` | modify | RRF, `matches_filters`, and the three modes through `search()`. |
| `tests/test_evaluation_benchmark.py` | modify | Two positional assertions shifted by the new column. |
| `.env.example` | modify | Document the three new settings. |
| `README.md`, `docs/roadmap.md` | modify | Task 7. |

**One deliberate deviation from the spec:** the spec listed `default_index` under `search.py`; it lives in `bm25.py` instead, so every line that constructs an index sits in one module. `search.py` imports it.

## Task Map

| Task | Step | Deliverable | Tag |
|---|---|---|---|
| 1 | 14 | `bm25.py` and its tests — the index, standalone | — |
| 2 | 14 | `mode="lexical"` in `search()`, `--mode` on two scripts, the `bm25-sentence` number | — |
| 3 | 15 | Filters honoured on the lexical branch | — |
| 4 | 15 | `rrf()`, `mode="hybrid"`, the `hybrid-k60-d50` number | `v0.5` |
| 5 | 16 | The parameter sweep and the recorded verdict | `v0.6` |
| 6 | 16 | Apply the acceptance rule to the default | — |
| 7 | — | The 422 transcript, README, roadmap | — |

---

### Task 1: The BM25 index, standalone

**Files:**
- Create: `app/retrieval/bm25.py`
- Test: `tests/test_retrieval_bm25.py`

**Interfaces:**
- Consumes: `app.models.chunks.Chunk` and `ScoredChunk`; `app.retrieval.store.chunk_from_payload`; `app.core.config.Settings`.
- Produces, relied on by Tasks 2 and 4:
  - `tokenize(text: str) -> list[str]`
  - `BM25Index(chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75)` with attributes `chunks: list[Chunk]`, `postings: dict[str, dict[int, int]]`, `doc_len: list[int]`, `avg_len: float`, `idf: dict[str, float]`
  - `BM25Index.search(query: str, top_k: int, predicate: Callable[[Chunk], bool] | None = None) -> list[ScoredChunk]`
  - `build_index(client: QdrantClient, collection: str) -> BM25Index`
  - `default_index(settings: Settings, collection: str) -> BM25Index`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval_bm25.py`:

```python
from typing import Any

import pytest

from app.models.chunks import Chunk
from app.retrieval.bm25 import BM25Index, build_index, tokenize


def chunk(index: int, text: str, **overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "document_id": "fastapi:tutorial/handling-errors",
        "source": "fastapi",
        "title": "Handling Errors",
        "url": None,
        "doc_type": "tutorial",
        "section": None,
        "chunk_index": index,
        "text": text,
        "char_start": index * 100,
        "char_end": index * 100 + max(len(text), 1),
    }
    fields.update(overrides)
    return Chunk(**fields)


# --- tokenising ------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_non_word_characters() -> None:
    assert tokenize("HTTPException 422, raised!") == ["httpexception", "422", "raised"]


def test_tokenize_keeps_accented_characters() -> None:
    """A French question must not lose its vowels before it reaches the index."""
    assert tokenize("Comment gérer les requêtes") == ["comment", "gérer", "les", "requêtes"]


def test_tokenize_keeps_digit_only_tokens() -> None:
    """`422` is the whole reason this module exists."""
    assert "422" in tokenize("returns a 422 response")


# --- the index -------------------------------------------------------------


def test_an_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="no chunks"):
        BM25Index([])


def test_idf_is_positive_for_a_term_in_every_document() -> None:
    """The textbook log((N - df + 0.5) / (df + 0.5)) goes negative here, which
    would make a common term subtract from a document's score."""
    index = BM25Index([chunk(0, "shared term"), chunk(1, "shared term")])
    assert index.idf["shared"] > 0


def test_a_rarer_term_outweighs_a_common_one() -> None:
    index = BM25Index([chunk(0, "common rare"), chunk(1, "common word"), chunk(2, "common word")])
    assert index.idf["rare"] > index.idf["common"]


def test_a_short_document_outranks_a_long_one_for_the_same_single_match() -> None:
    """Length normalisation, the `b` parameter. Without it `release-notes.md` —
    the largest file in the corpus, containing nearly every token in it — wins
    every lexical query."""
    short = chunk(0, "validation error")
    long = chunk(1, "validation " + " ".join(f"filler{n}" for n in range(200)))
    index = BM25Index([short, long])
    results = index.search("validation", top_k=2)
    assert [scored.chunk.chunk_index for scored in results] == [0, 1]
    assert results[0].score > results[1].score


def test_an_exact_token_is_retrievable() -> None:
    """The capability dense retrieval demonstrably lacks: the step 07 finding."""
    index = BM25Index(
        [
            chunk(0, "You can return a 422 Unprocessable Entity response"),
            chunk(1, "Handling errors with a plain HTTP exception"),
        ]
    )
    [top] = index.search("HTTPException 422", top_k=1)
    assert top.chunk.chunk_index == 0


def test_results_are_ranked_from_one() -> None:
    index = BM25Index([chunk(i, f"error number {i}") for i in range(3)])
    assert [scored.rank for scored in index.search("error", top_k=3)] == [1, 2, 3]


def test_scores_are_descending() -> None:
    index = BM25Index([chunk(0, "error error error"), chunk(1, "error"), chunk(2, "unrelated")])
    scores = [scored.score for scored in index.search("error", top_k=2)]
    assert scores == sorted(scores, reverse=True)


def test_a_query_with_no_matching_term_returns_nothing() -> None:
    """Not "the least bad chunk": an empty list is the honest answer, and the
    fusion in task 4 relies on it rather than on padding."""
    index = BM25Index([chunk(0, "dependency injection")])
    assert index.search("kubernetes", top_k=5) == []


def test_top_k_caps_the_result_count() -> None:
    index = BM25Index([chunk(i, "error") for i in range(10)])
    assert len(index.search("error", top_k=3)) == 3


def test_ties_are_broken_deterministically_by_chunk_id() -> None:
    """Two identical chunks must not swap places between benchmark runs."""
    index = BM25Index([chunk(1, "error"), chunk(0, "error")])
    first = [scored.chunk.chunk_id for scored in index.search("error", top_k=2)]
    second = [scored.chunk.chunk_id for scored in index.search("error", top_k=2)]
    assert first == second == sorted(first)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_is_rejected(query: str) -> None:
    index = BM25Index([chunk(0, "anything")])
    with pytest.raises(ValueError, match="empty"):
        index.search(query, top_k=5)


def test_top_k_zero_is_rejected() -> None:
    index = BM25Index([chunk(0, "anything")])
    with pytest.raises(ValueError, match="top_k"):
        index.search("anything", top_k=0)


def test_the_predicate_excludes_chunks_before_ranking() -> None:
    """Applied while accumulating, not afterwards, so a filtered query still
    returns top_k results instead of top_k minus the discarded ones."""
    index = BM25Index(
        [
            chunk(0, "error in a tutorial", doc_type="tutorial"),
            chunk(1, "error in a reference", doc_type="reference"),
        ]
    )
    results = index.search("error", top_k=5, predicate=lambda c: c.doc_type == "reference")
    assert [scored.chunk.doc_type for scored in results] == ["reference"]


# --- building from Qdrant --------------------------------------------------


class FakeRecord:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class FakeScrollClient:
    """Hands back the payloads in pages, mimicking Qdrant's offset protocol."""

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def scroll(self, **kwargs: Any) -> tuple[list[FakeRecord], int | None]:
        self.calls.append(kwargs)
        page = len(self.calls) - 1
        records = [FakeRecord(payload) for payload in self.pages[page]]
        more = page + 1 < len(self.pages)
        return records, (page + 1 if more else None)


def test_build_index_keeps_every_page_including_the_first() -> None:
    """Qdrant's own documented loop shape discards the first batch."""
    pages = [[chunk(0, "first page").to_payload()], [chunk(1, "second page").to_payload()]]
    index = build_index(FakeScrollClient(pages), "chunks_sentence")  # type: ignore[arg-type]
    assert len(index.chunks) == 2
    assert {c.chunk_index for c in index.chunks} == {0, 1}


def test_build_index_asks_for_payloads_and_not_vectors() -> None:
    """1 536 floats per point, 1 484 points, for data BM25 never reads."""
    client = FakeScrollClient([[chunk(0, "only page").to_payload()]])
    build_index(client, "chunks_sentence")  # type: ignore[arg-type]
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["with_vectors"] is False
    assert client.calls[0]["collection_name"] == "chunks_sentence"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_bm25.py -v`

Expected: every test fails at collection with `ModuleNotFoundError: No module named 'app.retrieval.bm25'`.

- [ ] **Step 3: Write the module**

Create `app/retrieval/bm25.py`:

```python
"""Lexical retrieval, hand-rolled. The other half of step 14's hybrid search.

Dense retrieval finds "authentication" when the question says "protect an API".
It cannot find the literal token ``422``: only two documents in the corpus
contain it, and cosine similarity has no way to care. BM25 is the exact inverse,
which is why step 16 fuses the two rather than choosing between them.

The index is built from the chunks already in Qdrant, through the same
``chunk_from_payload`` dense search uses, so the two retrievers cannot drift onto
different chunk sets. Roughly 200 ms for the 1 484-chunk corpus, dominated by
the scroll; a query costs under 5 ms.
"""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.models.chunks import Chunk, ScoredChunk
from app.retrieval.store import chunk_from_payload

# Term-frequency saturation and length normalisation. The Robertson defaults;
# step 16 sweeps the RRF constant and the candidate depth, not these two.
K1 = 1.5
B = 0.75
SCROLL_BATCH = 256

TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    r"""Lowercased runs of ``\w``. Applied to documents and queries alike.

    ``\w`` rather than ``[a-z0-9_]`` because a French question carries accents
    and dropping them would make it unmatchable. No stemming and no stopword
    list: BM25's IDF already drives "the" to near zero, which is what a stoplist
    crudely approximates, and a hand-rolled stemmer is a hundred lines of
    guesswork over a corpus whose vocabulary is mostly identifiers.
    """
    return TOKEN.findall(text.lower())


class BM25Index:
    """An inverted index over chunk text, scored with Okapi BM25.

    Built once per process and queried many times, so construction does the work
    the query would otherwise repeat: term frequencies, document lengths and IDF
    are all precomputed here.
    """

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = K1, b: float = B) -> None:
        if not chunks:
            # An empty index silently returns nothing for every query, which
            # reads downstream as "BM25 does not help" rather than "the
            # collection was empty or misspelled".
            raise ValueError("cannot build a BM25 index over no chunks")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: list[int] = []
        for index, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.text)
            self.doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self.postings[term][index] = freq
        # `or 1.0` guards a corpus of blank chunks; the division below would
        # otherwise raise ZeroDivisionError on the first query.
        self.avg_len = sum(self.doc_len) / len(self.doc_len) or 1.0
        total = len(self.chunks)
        # Lucene's IDF variant. The textbook log((N - df + 0.5) / (df + 0.5))
        # turns negative once df > N/2, so a term appearing in most documents
        # would *subtract* from their scores — wrong in a way that looks
        # plausible right up until you debug a ranking.
        self.idf = {
            term: math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def search(
        self,
        query: str,
        top_k: int,
        predicate: Callable[[Chunk], bool] | None = None,
    ) -> list[ScoredChunk]:
        """The ``top_k`` best-matching chunks, best first, BM25 scores as computed.

        Scores accumulate only over documents that appear in a query term's
        postings — never over all 1 484 chunks. A query whose every term is
        unknown returns an empty list rather than the least bad chunk.

        ``predicate`` is how payload filters reach this branch; it is applied
        while accumulating, so a filtered query still returns ``top_k`` results
        rather than ``top_k`` minus the ones that were discarded.
        """
        if not query.strip():
            raise ValueError("query is empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        scores: dict[int, float] = defaultdict(float)
        # A set, so a word repeated in the question does not count twice.
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            idf = self.idf[term]
            for index, freq in posting.items():
                if predicate is not None and not predicate(self.chunks[index]):
                    continue
                norm = 1 - self.b + self.b * self.doc_len[index] / self.avg_len
                scores[index] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)

        # chunk_id breaks ties: two equally scored chunks must not swap places
        # between two benchmark runs of the same commit.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        return [
            ScoredChunk(chunk=self.chunks[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(ranked[:top_k], start=1)
        ]


def build_index(client: QdrantClient, collection: str) -> BM25Index:
    """Scroll every chunk out of ``collection`` and index it.

    ``with_vectors=False`` because BM25 never reads them and 1 536 floats per
    point is the whole cost of the scroll. The loop keeps every page including
    the first, unlike the loop shape in Qdrant's own documentation.
    """
    chunks: list[Chunk] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        chunks.extend(chunk_from_payload(record.payload or {}) for record in records)
        if offset is None:
            return BM25Index(chunks)


@lru_cache(maxsize=4)
def _index_for(qdrant_url: str, collection: str) -> BM25Index:
    return build_index(QdrantClient(url=qdrant_url), collection)


def default_index(settings: Settings, collection: str) -> BM25Index:
    """The process-wide index for one collection, built on first use.

    Mirrors ``search.default_embedder``: a benchmark run of 45 questions pays the
    ~200 ms build once. Keyed on the URL string rather than on ``settings``
    because ``Settings`` is not hashable.

    ponytail: rebuilt per process, not persisted. A sqlite-backed index would
    save ~200 ms on a CLI call that already spends 1.5-3.7 s in generation.
    """
    return _index_for(settings.qdrant_url, collection)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_bm25.py -v`

Expected: PASS, 20 tests (the empty-query test is parametrized three ways).

If `test_a_short_document_outranks_a_long_one_for_the_same_single_match` fails, the `b` term is wrong — that test is the whole reason length normalisation is in the plan. Do not adjust the test to match the code.

- [ ] **Step 5: Run the full gates**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest
```

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/bm25.py tests/test_retrieval_bm25.py
git commit -m "feat(retrieval): hand-rolled BM25 index over the indexed chunks

Okapi BM25 with Lucene's IDF variant, built by scrolling the collection
dense search already uses so the two retrievers rank the same population.
No new dependency: the spec rejects rank_bm25 and fastembed by name.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 2: `mode="lexical"` through `search()`, and the step 14 number

**Files:**
- Modify: `app/core/config.py`, `app/retrieval/search.py`, `app/retrieval/store.py`, `scripts/search.py`, `scripts/benchmark.py`, `.env.example`
- Test: `tests/test_retrieval_search.py`

**Interfaces:**
- Consumes: everything Task 1 produced.
- Produces, relied on by Tasks 3-5:
  - `search.RETRIEVERS: dict[str, Retrieve]` with keys `"dense"` and `"lexical"` (Task 4 adds `"hybrid"`)
  - `search(query, *, top_k=5, mode=None, filters=None, collection=None, settings=None, client=None, embedder=None, index=None)`
  - `Settings.retrieval_mode: str = "dense"`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_search.py`:

```python
# --- retrieval modes (step 14) --------------------------------------------


def bm25_index(*texts: str) -> BM25Index:
    return BM25Index(
        [Chunk.model_validate(make_payload(i, text=text)) for i, text in enumerate(texts)]
    )


def test_the_default_mode_is_dense_and_queries_qdrant() -> None:
    client = FakeClient()
    run(client=client, top_k=3)
    assert len(client.calls) == 1


def test_lexical_mode_does_not_touch_qdrant_or_the_embedder() -> None:
    """The point of a lexical branch: no vector, no API call, no server round trip."""
    client, embedder = FakeClient(), FakeEmbedder()
    run(
        "HTTPException 422",
        mode="lexical",
        client=client,
        embedder=embedder,
        index=bm25_index("a 422 response", "unrelated prose"),
        top_k=1,
    )
    assert client.calls == []
    assert embedder.calls == []


def test_lexical_mode_retrieves_an_exact_token_dense_search_misses() -> None:
    [top] = run(
        "HTTPException 422",
        mode="lexical",
        index=bm25_index("returns a 422 response", "generic error handling prose"),
        top_k=1,
    )
    assert "422" in top.chunk.text


def test_lexical_mode_ranks_from_one() -> None:
    results = run(
        "error",
        mode="lexical",
        index=bm25_index("error one", "error two", "error three"),
        top_k=3,
    )
    assert [scored.rank for scored in results] == [1, 2, 3]


def test_an_unknown_mode_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="sparse"):
        run(mode="sparse")


def test_the_mode_falls_back_to_the_settings() -> None:
    """A benchmark row that says `lexical` must not have run dense retrieval."""
    client, embedder = FakeClient(), FakeEmbedder()
    run(
        "error",
        settings=Settings(qdrant_collection="chunks", retrieval_mode="lexical"),
        client=client,
        embedder=embedder,
        index=bm25_index("an error occurred"),
        top_k=1,
    )
    assert client.calls == []
```

Add to the imports at the top of the file:

```python
from app.retrieval.bm25 import BM25Index
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_search.py -v -k "mode or lexical"`

Expected: FAIL with `TypeError: search() got an unexpected keyword argument 'mode'` (and `'index'`).

- [ ] **Step 3: Add the setting**

In `app/core/config.py`, after the `top_k` line:

```python
    # Step 14-16 compared three. The default changes only if the sweep meets the
    # rule in the step 14-16 design doc: dense | lexical | hybrid.
    retrieval_mode: str = "dense"
```

- [ ] **Step 4: Document it**

In `.env.example`, after the `GENERATION_MODEL` block:

```bash
# Recherche (étapes 14-16)
# dense | lexical | hybrid — comparées à l'étape 16, table dans le README.
RETRIEVAL_MODE=dense
```

- [ ] **Step 5: Restructure `search()` around the registry**

In `app/retrieval/search.py`, replace the body of `build_filter` down to the end of `search` with the following. `parse_filters` and `default_embedder` are unchanged and stay where they are.

First, extract the shared validation and add the imports:

```python
from typing import Any

from app.retrieval.bm25 import BM25Index, default_index
```

```python
def _check_filters(filters: Filters) -> None:
    """Reject keys that are not indexed and values that can never match.

    Shared by ``build_filter`` and ``matches_filters`` so a filter is accepted or
    rejected identically whichever branch ends up applying it — a typo must not
    be caught on the dense side and silently honoured on the lexical one.
    """
    unknown = sorted(set(filters) - set(INDEXED_FIELDS))
    if unknown:
        raise ValueError(f"not an indexed field: {', '.join(unknown)}; have {INDEXED_FIELDS}")
    for key, value in filters.items():
        if not isinstance(value, str) and not list(value):
            # MatchAny([]) is a filter that matches nothing at all.
            raise ValueError(f"{key} was given an empty list of values")
```

Then `build_filter` loses its own copies of those two checks:

```python
def build_filter(filters: Filters | None) -> Filter | None:
    """Turn ``{"doc_type": "tutorial", "source": ["fastapi", "starlette"]}`` into a
    Qdrant filter: a scalar matches one value, a sequence matches any of them, and
    several keys are ANDed.
    """
    if not filters:
        return None
    _check_filters(filters)
    conditions: list[Condition] = []
    for key, value in filters.items():
        if isinstance(value, str):
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
    return Filter(must=conditions)
```

Now the mode implementations and the dispatcher, replacing the old `search` body:

```python
def _dense(
    query: str,
    *,
    top_k: int,
    filters: Filters | None,
    collection: str,
    client: QdrantClient,
    embedder: Embedder,
    **_: Any,
) -> list[ScoredChunk]:
    """Cosine nearest neighbours. Scores are raw, as Qdrant reports them: cosine
    scores are comparable across queries for this collection, and invented
    normalisation is a layer that lies."""
    hits = client.query_points(
        collection_name=collection,
        query=embedder(query),
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
    ).points
    return [
        ScoredChunk(chunk=chunk_from_payload(hit.payload or {}), score=hit.score, rank=rank)
        for rank, hit in enumerate(hits, start=1)
    ]


def _lexical(
    query: str, *, top_k: int, filters: Filters | None, index: BM25Index, **_: Any
) -> list[ScoredChunk]:
    """BM25 over the same chunks, no vector and no server round trip."""
    return index.search(query, top_k)


Retrieve = Callable[..., list[ScoredChunk]]
# Mirrors chunk.STRATEGIES: the registry is how this project compares N variants
# and promotes a winner. Step 17's reranker registers a key here.
RETRIEVERS: dict[str, Retrieve] = {"dense": _dense, "lexical": _lexical}


def search(
    query: str,
    *,
    top_k: int = 5,
    mode: str | None = None,
    filters: Filters | None = None,
    collection: str | None = None,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    index: BM25Index | None = None,
) -> list[ScoredChunk]:
    """The ``top_k`` chunks best matching ``query``, best first.

    ``mode`` selects the retriever and defaults to ``RETRIEVAL_MODE``. The
    signature and the return type are the seam steps 08, 11 and 17 are built on:
    adding a retriever means adding a key to ``RETRIEVERS``, never changing this.

    ``client``, ``embedder`` and ``index`` are injectable so the unit tests run
    with no server and no API key. Only what the chosen mode needs is built, so
    lexical mode never opens a connection.
    """
    if not query.strip():
        # The empty string embeds fine and retrieves plausible-looking garbage.
        raise ValueError("query is empty")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    settings = settings or get_settings()
    mode = mode or settings.retrieval_mode
    if mode not in RETRIEVERS:
        raise ValueError(f"unknown retrieval mode {mode!r}; have {sorted(RETRIEVERS)}")
    collection = collection or settings.qdrant_collection

    if mode in ("dense", "hybrid"):
        client = client or get_client(settings)
        embedder = embedder or default_embedder(settings)
    if mode in ("lexical", "hybrid"):
        index = index or default_index(settings, collection)

    return RETRIEVERS[mode](
        query,
        top_k=top_k,
        filters=filters,
        collection=collection,
        client=client,
        embedder=embedder,
        index=index,
    )
```

Note `_lexical` ignores `filters` for now — Task 3 is the test that forces it to stop.

- [ ] **Step 6: Correct the stale comment in `store.py`**

`app/retrieval/store.py` line ~55 promises something this phase decided against. Replace:

```python
    # ponytail: one unnamed dense vector. Step 14 adds BM25 and will want a named
    # sparse vector alongside, which means recreating the collection — fine, the
    # step 05 cache makes re-indexing free.
```

with:

```python
    # ponytail: one unnamed dense vector, and it stays that way. Step 14 put BM25
    # in-process instead of in a named sparse vector: Qdrant's IDF modifier
    # supplies only the IDF factor, so k1 and b would still live in Python and
    # the formula would be split across two systems. See the step 14-16 design
    # doc; a named sparse vector is the upgrade path if the corpus outgrows an
    # in-process index.
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_search.py -v`

Expected: PASS — the 6 new tests and all 15 existing ones. The existing filter tests must still pass unchanged; `_check_filters` preserved their error messages verbatim.

- [ ] **Step 8: Add `--mode` to `scripts/search.py`**

After the `--filter` argument:

```python
    parser.add_argument(
        "--mode", choices=sorted(RETRIEVERS), help="default: RETRIEVAL_MODE"
    )
```

Change the import and the call:

```python
from app.retrieval.search import RETRIEVERS, parse_filters, search  # noqa: E402
```

```python
    results = search(
        args.query, top_k=args.top_k, mode=args.mode, filters=parse_filters(args.filter)
    )
```

And the docstring's usage line:

```python
    uv run python scripts/search.py "question" [--top-k 5] [--mode lexical]
                                              [--filter doc_type=tutorial]
```

- [ ] **Step 9: Add `--mode` to `scripts/benchmark.py`**

After the `--collection` argument:

```python
    parser.add_argument(
        "--mode", choices=sorted(RETRIEVERS), help="default: RETRIEVAL_MODE"
    )
```

Change the import:

```python
from app.retrieval.search import RETRIEVERS, parse_filters, search  # noqa: E402
```

Pass it in `retrieve`:

```python
        return search(
            text,
            top_k=args.top_k,
            mode=args.mode,
            filters=filters or None,
            collection=collection,
            settings=settings,
        )
```

And record it in `config`, immediately after `"top_k": args.top_k,`:

```python
            "mode": args.mode or settings.retrieval_mode,
```

Add the usage line to the module docstring, after the oracle one:

```python
    uv run python scripts/benchmark.py --label "bm25-sentence" --mode lexical
```

- [ ] **Step 10: Run the full gates**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest
```

- [ ] **Step 11: Eyeball the retriever on the real corpus**

Qdrant must be up: `docker compose up -d qdrant`.

```bash
uv run python scripts/search.py "HTTPException 422" --mode lexical
uv run python scripts/search.py "HTTPException 422" --mode dense
```

Expected: the lexical run surfaces a chunk containing the literal `422` — the step 07 finding, for the first time. The dense run does not. If lexical mode instead returns chunks from `release-notes.md` for every query, length normalisation is not working; go back to Task 1 rather than continuing.

- [ ] **Step 12: Measure BM25 alone**

```bash
uv run python scripts/benchmark.py --label "bm25-sentence" --mode lexical \
  --collection chunks_sentence --strategy sentence --compare "dense-sentence-doctype"
```

Expect a *worse* aggregate than 0.776. Lexical-only retrieval has no way to answer a conceptual question, and the spec says so in advance. Record the number regardless — it is the denominator the fusion in Task 4 has to beat.

- [ ] **Step 13: Commit**

```bash
git add app/core/config.py app/retrieval/search.py app/retrieval/store.py \
  scripts/search.py scripts/benchmark.py tests/test_retrieval_search.py \
  .env.example data/eval/results.jsonl
git commit -m "feat(retrieval): select a retriever with mode=, and measure BM25 alone

RETRIEVERS keyed dense|lexical behind the unchanged search() signature,
mirroring step 12's chunking registry. Only what the chosen mode needs is
constructed, so lexical retrieval opens no connection and spends nothing.
Records the bm25-sentence row and corrects store.py's stale promise of a
named sparse vector.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 3: Filters on the lexical branch

**Files:**
- Modify: `app/retrieval/search.py`
- Test: `tests/test_retrieval_search.py`

**Interfaces:**
- Consumes: `_check_filters`, `_lexical`, `BM25Index.search`'s `predicate` from Tasks 1-2.
- Produces, relied on by Task 4: `matches_filters(chunk: Chunk, filters: Filters | None) -> bool`.

Task 3 comes before hybrid mode deliberately: hybrid retrieval cannot honestly claim filter parity, and `--oracle-filter` cannot be trusted, while half the retriever ignores `filters=`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_search.py`:

```python
# --- filters on the lexical branch (step 15) ------------------------------


def filtered_index() -> BM25Index:
    return BM25Index(
        [
            Chunk.model_validate(
                make_payload(0, text="error in a tutorial", doc_type="tutorial", source="fastapi")
            ),
            Chunk.model_validate(
                make_payload(1, text="error in a reference", doc_type="reference", source="fastapi")
            ),
            Chunk.model_validate(
                make_payload(2, text="error in starlette", doc_type="tutorial", source="starlette")
            ),
        ]
    )


def test_matches_filters_accepts_a_scalar_match() -> None:
    chunk = Chunk.model_validate(make_payload(0, doc_type="tutorial"))
    assert matches_filters(chunk, {"doc_type": "tutorial"})
    assert not matches_filters(chunk, {"doc_type": "reference"})


def test_matches_filters_accepts_any_of_a_sequence() -> None:
    chunk = Chunk.model_validate(make_payload(0, doc_type="tutorial"))
    assert matches_filters(chunk, {"doc_type": ["tutorial", "advanced"]})
    assert not matches_filters(chunk, {"doc_type": ["reference", "advanced"]})


def test_matches_filters_ands_several_keys() -> None:
    chunk = Chunk.model_validate(make_payload(0, doc_type="tutorial", source="fastapi"))
    assert matches_filters(chunk, {"doc_type": "tutorial", "source": "fastapi"})
    assert not matches_filters(chunk, {"doc_type": "tutorial", "source": "starlette"})


def test_matches_filters_with_no_filters_matches_everything() -> None:
    chunk = Chunk.model_validate(make_payload(0))
    assert matches_filters(chunk, None)
    assert matches_filters(chunk, {})


def test_matches_filters_rejects_an_unindexed_key_like_build_filter_does() -> None:
    """The two translations must accept and reject exactly the same mappings."""
    chunk = Chunk.model_validate(make_payload(0))
    with pytest.raises(ValueError, match="doctype"):
        matches_filters(chunk, {"doctype": "tutorial"})


def test_matches_filters_rejects_an_empty_value_list_like_build_filter_does() -> None:
    chunk = Chunk.model_validate(make_payload(0))
    with pytest.raises(ValueError, match="doc_type"):
        matches_filters(chunk, {"doc_type": []})


def test_lexical_mode_honours_a_scalar_filter() -> None:
    results = run("error", mode="lexical", index=filtered_index(), filters={"doc_type": "reference"}, top_k=5)
    assert [scored.chunk.doc_type for scored in results] == ["reference"]


def test_lexical_mode_honours_a_sequence_filter() -> None:
    results = run(
        "error", mode="lexical", index=filtered_index(), filters={"source": ["starlette"]}, top_k=5
    )
    assert [scored.chunk.source for scored in results] == ["starlette"]


def test_lexical_mode_ands_two_filters() -> None:
    results = run(
        "error",
        mode="lexical",
        index=filtered_index(),
        filters={"doc_type": "tutorial", "source": "fastapi"},
        top_k=5,
    )
    assert [scored.chunk.chunk_index for scored in results] == [0]


def test_lexical_mode_rejects_an_unindexed_filter_key_before_searching() -> None:
    """Eagerly, not once per candidate chunk: a typo must fail the call."""
    with pytest.raises(ValueError, match="doctype"):
        run("error", mode="lexical", index=filtered_index(), filters={"doctype": "tutorial"})
```

Add `matches_filters` to the import line:

```python
from app.retrieval.search import matches_filters, parse_filters, search
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_search.py -v -k "matches_filters or lexical_mode_honours or lexical_mode_ands or lexical_mode_rejects"`

Expected: FAIL with `ImportError: cannot import name 'matches_filters'`.

- [ ] **Step 3: Write `matches_filters`**

In `app/retrieval/search.py`, directly after `build_filter`:

```python
def matches_filters(chunk: Chunk, filters: Filters | None) -> bool:
    """``build_filter``'s twin for the lexical branch: the same mapping, applied
    in Python because a BM25 index has no server to push a filter to.

    Kept beside ``build_filter`` and sharing ``_check_filters`` on purpose. Two
    translations of one mapping living in different modules is how one of them
    grows support for a key the other silently ignores, and a filter that half
    applies is worse than one that fails.
    """
    if not filters:
        return True
    _check_filters(filters)
    for key, value in filters.items():
        actual = getattr(chunk, key)
        if isinstance(value, str):
            if actual != value:
                return False
        elif actual not in list(value):
            return False
    return True
```

Add `Chunk` to the import from `app.models.chunks`:

```python
from app.models.chunks import Chunk, ScoredChunk
```

- [ ] **Step 4: Apply it in `_lexical`**

Replace `_lexical`:

```python
def _lexical(
    query: str, *, top_k: int, filters: Filters | None, index: BM25Index, **_: Any
) -> list[ScoredChunk]:
    """BM25 over the same chunks, no vector and no server round trip.

    Filters are validated once here rather than per candidate chunk, so a
    misspelled key fails the call instead of quietly matching nothing.
    """
    if filters:
        _check_filters(filters)
    return index.search(query, top_k, predicate=lambda chunk: matches_filters(chunk, filters))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_search.py -v`

Expected: PASS, all tests including the 15 original ones.

- [ ] **Step 6: Run the full gates**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest
```

- [ ] **Step 7: Confirm on the real corpus that the filter actually narrows**

```bash
uv run python scripts/search.py "error" --mode lexical --filter doc_type=reference
```

Expected: every result's `document_id` starts `fastapi:reference/`.

- [ ] **Step 8: Commit**

```bash
git add app/retrieval/search.py tests/test_retrieval_search.py
git commit -m "feat(retrieval): honour payload filters on the lexical branch

matches_filters is build_filter's twin, sharing one validation helper so a
key either branch accepts is a key both accept. Without it hybrid mode
would apply a filter to half the retriever, and --oracle-filter would be
measuring something other than what it claims.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 4: RRF and `mode="hybrid"`

**Files:**
- Modify: `app/core/config.py`, `app/retrieval/search.py`, `app/generation/answer.py`, `app/evaluation/benchmark.py`, `scripts/ask.py`, `scripts/benchmark.py`, `.env.example`
- Test: `tests/test_retrieval_search.py`, `tests/test_evaluation_benchmark.py`

**Interfaces:**
- Consumes: `_dense`, `_lexical`, `matches_filters`, `RETRIEVERS` from Tasks 2-3.
- Produces, relied on by Task 5:
  - `rrf(rankings: Sequence[Sequence[ScoredChunk]], *, k: int, top_k: int) -> list[ScoredChunk]`
  - `RETRIEVERS["hybrid"]`, and `search(..., candidates=None, rrf_k=None)`
  - `Settings.retrieval_candidates: int = 50`, `Settings.rrf_k: int = 60`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_search.py`:

```python
# --- RRF and hybrid mode (step 16) ----------------------------------------


def scored_list(*indices: int) -> list[ScoredChunk]:
    """A ranked list of chunks, identified by chunk_index, ranked from 1."""
    return [
        ScoredChunk(
            chunk=Chunk.model_validate(make_payload(index)), score=1.0 - 0.1 * rank, rank=rank
        )
        for rank, index in enumerate(indices, start=1)
    ]


def indices_of(results: list[ScoredChunk]) -> list[int]:
    return [scored.chunk.chunk_index for scored in results]


def test_rrf_prefers_a_chunk_ranked_second_by_both_over_one_ranked_first_by_one() -> None:
    """The whole reason to fuse ranks: broad agreement beats a single strong vote.
    Chunk 9 scores 2/(60+2) = 0.0323; chunk 1 scores 1/(60+1) = 0.0164."""
    fused = rrf([scored_list(1, 9), scored_list(2, 9)], k=60, top_k=3)
    assert indices_of(fused)[0] == 9


def test_rrf_deduplicates_a_chunk_appearing_in_both_rankings() -> None:
    fused = rrf([scored_list(1, 2), scored_list(2, 1)], k=60, top_k=10)
    assert sorted(indices_of(fused)) == [1, 2]


def test_rrf_ignores_the_input_scores_entirely() -> None:
    """A cosine and a BM25 score share no scale; using them would invent a
    comparison the numbers do not support."""
    high = scored_list(1)
    low = [ScoredChunk(chunk=high[0].chunk, score=0.0001, rank=1)]
    assert rrf([low], k=60, top_k=1)[0].score == rrf([high], k=60, top_k=1)[0].score


def test_rrf_ranks_from_one_and_scores_descending() -> None:
    fused = rrf([scored_list(1, 2, 3)], k=60, top_k=3)
    assert [scored.rank for scored in fused] == [1, 2, 3]
    assert [s.score for s in fused] == sorted([s.score for s in fused], reverse=True)


def test_rrf_score_is_the_fused_score_not_a_cosine() -> None:
    """Documented consequence: abstention_rate is not comparable across modes."""
    assert rrf([scored_list(1)], k=60, top_k=1)[0].score == pytest.approx(1 / 61)


def test_rrf_caps_at_top_k() -> None:
    assert len(rrf([scored_list(1, 2, 3, 4, 5)], k=60, top_k=2)) == 2


def test_rrf_with_an_empty_ranking_still_fuses_the_other() -> None:
    """A lexical query whose terms are all unknown returns nothing; hybrid mode
    must degrade to dense rather than fail."""
    assert indices_of(rrf([scored_list(1, 2), []], k=60, top_k=2)) == [1, 2]


def test_rrf_rejects_a_k_below_one() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        rrf([scored_list(1)], k=0, top_k=1)


def test_hybrid_mode_queries_both_branches_at_the_candidate_depth() -> None:
    client, embedder = FakeClient(), FakeEmbedder()
    run(
        "error",
        mode="hybrid",
        client=client,
        embedder=embedder,
        index=bm25_index("an error occurred", "another error"),
        top_k=2,
        candidates=50,
    )
    assert client.calls[0]["limit"] == 50
    assert embedder.calls == ["error"]


def test_hybrid_mode_returns_top_k_not_candidates() -> None:
    results = run(
        "error",
        mode="hybrid",
        client=FakeClient(),
        index=bm25_index("an error occurred", "another error"),
        top_k=3,
        candidates=10,
    )
    assert len(results) == 3
    assert [scored.rank for scored in results] == [1, 2, 3]


def test_candidates_below_top_k_is_rejected() -> None:
    """Fusing two top-3 lists cannot produce 10 results; a silent short list
    would read downstream as a recall drop."""
    with pytest.raises(ValueError, match="candidates"):
        run("error", mode="hybrid", top_k=10, candidates=3, index=bm25_index("error"))


def test_hybrid_mode_honours_filters_on_both_branches() -> None:
    client = FakeClient()
    run(
        "error",
        mode="hybrid",
        client=client,
        index=filtered_index(),
        filters={"doc_type": "reference"},
        top_k=5,
        candidates=5,
    )
    [condition] = client.calls[0]["query_filter"].must
    assert condition.key == "doc_type"


def test_candidates_and_rrf_k_fall_back_to_the_settings() -> None:
    client = FakeClient()
    run(
        "error",
        mode="hybrid",
        settings=Settings(qdrant_collection="chunks", retrieval_candidates=30, rrf_k=20),
        client=client,
        index=bm25_index("an error occurred"),
        top_k=5,
    )
    assert client.calls[0]["limit"] == 30
```

Add `rrf` to the import line:

```python
from app.retrieval.search import matches_filters, parse_filters, rrf, search
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_search.py -v -k "rrf or hybrid or candidates"`

Expected: FAIL with `ImportError: cannot import name 'rrf'`.

- [ ] **Step 3: Add the two settings**

In `app/core/config.py`, after `retrieval_mode`:

```python
    # Per-branch depth before fusion. Fusing two top-10 lists cannot surface a
    # document neither branch ranked top-10, so this is the parameter that moves
    # recall; step 16 sweeps it.
    retrieval_candidates: int = 50
    # The constant from the original RRF paper. Larger flattens the rank
    # weighting, smaller sharpens it.
    rrf_k: int = 60
```

- [ ] **Step 4: Document them**

In `.env.example`, extend the block added in Task 2:

```bash
# Recherche (étapes 14-16)
# dense | lexical | hybrid — comparées à l'étape 16, table dans le README.
RETRIEVAL_MODE=dense
# Profondeur par branche avant fusion, et la constante k de RRF.
RETRIEVAL_CANDIDATES=50
RRF_K=60
```

- [ ] **Step 5: Write `rrf` and `_hybrid`**

In `app/retrieval/search.py`, after `_lexical`:

```python
def rrf(
    rankings: Sequence[Sequence[ScoredChunk]], *, k: int, top_k: int
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: ``sum(1 / (k + rank))`` over the lists a chunk is in.

    Ranks only, never scores, and that is the entire point. A cosine similarity
    and a BM25 score share no scale; normalising them per query would invent a
    comparison the numbers do not support, which is the "layer that lies" this
    module refuses elsewhere.

    The returned ``score`` is the fused one — around 0.03, not a cosine's
    0.3-0.6. It is what produced the ranking, so it is what gets reported, and
    that makes ``benchmark.DEFAULT_ABSTENTION_THRESHOLD`` meaningless outside
    dense mode. Step 22 owns refusal; each run records its ``mode`` so the two
    are never compared by accident.
    """
    if k < 1:
        raise ValueError(f"rrf_k must be at least 1, got {k}")
    fused: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for scored in ranking:
            chunk_id = scored.chunk.chunk_id
            fused[chunk_id] += 1.0 / (k + scored.rank)
            chunks.setdefault(chunk_id, scored.chunk)
    # chunk_id breaks ties so two runs of one commit agree.
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [
        ScoredChunk(chunk=chunks[chunk_id], score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1)
    ]


def _hybrid(
    query: str,
    *,
    top_k: int,
    candidates: int,
    rrf_k: int,
    filters: Filters | None,
    collection: str,
    client: QdrantClient,
    embedder: Embedder,
    index: BM25Index,
) -> list[ScoredChunk]:
    """Both branches to ``candidates`` depth, then fused to ``top_k``."""
    dense = _dense(
        query,
        top_k=candidates,
        filters=filters,
        collection=collection,
        client=client,
        embedder=embedder,
    )
    lexical = _lexical(query, top_k=candidates, filters=filters, index=index)
    return rrf([dense, lexical], k=rrf_k, top_k=top_k)
```

Add the import and register the mode:

```python
from collections import defaultdict
```

```python
RETRIEVERS: dict[str, Retrieve] = {"dense": _dense, "lexical": _lexical, "hybrid": _hybrid}
```

- [ ] **Step 6: Thread `candidates` and `rrf_k` through `search()`**

Add the two parameters after `mode`:

```python
    candidates: int | None = None,
    rrf_k: int | None = None,
```

After the `collection` line in the body:

```python
    candidates = candidates or settings.retrieval_candidates
    if candidates < top_k:
        # Fusing two top-3 lists cannot produce 10 results, and a silently short
        # list reads downstream as a recall drop rather than a bad flag.
        raise ValueError(f"candidates ({candidates}) must be at least top_k ({top_k})")
```

And pass them to the retriever:

```python
    return RETRIEVERS[mode](
        query,
        top_k=top_k,
        candidates=candidates,
        rrf_k=rrf_k or settings.rrf_k,
        filters=filters,
        collection=collection,
        client=client,
        embedder=embedder,
        index=index,
    )
```

Extend the docstring with one paragraph:

```python
    ``candidates`` is the per-branch depth hybrid mode retrieves before fusing,
    and ``rrf_k`` the fusion constant. Both default to their settings and are
    ignored by the single-branch modes.
```

- [ ] **Step 7: Run the retrieval tests**

Run: `uv run pytest tests/test_retrieval_search.py tests/test_retrieval_bm25.py -v`

Expected: PASS.

- [ ] **Step 8: Thread `mode` through `answer_question`**

`scripts/ask.py --mode hybrid` cannot reach the pipeline otherwise, and Task 7's 422 transcript needs it.

In `app/generation/answer.py`, add a parameter after `top_k`:

```python
    mode: str | None = None,
```

Document it in the docstring, after the ``strict`` paragraph:

```python
    ``mode`` selects the retriever; it is threaded straight through so the
    measured winner of step 16 reaches the answer, not only the benchmark.
```

And pass it at the retriever call:

```python
    chunks = retriever(
        question, top_k=top_k or settings.top_k, mode=mode, filters=filters, settings=settings
    )
```

- [ ] **Step 9: Add `--mode` to `scripts/ask.py`**

After the `--filter` argument:

```python
    parser.add_argument(
        "--mode", choices=sorted(RETRIEVERS), help="default: RETRIEVAL_MODE"
    )
```

Change the import:

```python
from app.retrieval.search import RETRIEVERS, parse_filters, search  # noqa: E402
```

Both call sites gain it:

```python
        chunks = search(args.question, top_k=args.top_k or 5, mode=args.mode, filters=filters)
```

```python
    answer = answer_question(
        args.question, top_k=args.top_k, mode=args.mode, filters=filters, strict=args.strict
    )
```

And the docstring's usage line:

```python
    uv run python scripts/ask.py "question" [--top-k 5] [--mode hybrid]
                                           [--filter doc_type=tutorial] [--show-context]
```

- [ ] **Step 10: Add the `mode` summary column**

In `app/evaluation/benchmark.py`, insert `"mode"` into `SUMMARY_COLUMNS` right after `"label"`:

```python
SUMMARY_COLUMNS = (
    "label",
    "mode",
    "strategy",
    "size",
    "overlap",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@5",
    "conceptual r@5",
    "p50 ms",
)
```

And emit it in `summarise`, as the first entry after `label`:

```python
    return [
        [
            label,
            str(row["config"].get("mode", "dense")),
            str(row["config"].get("strategy", "?")),
```

`"dense"` rather than `"?"` as the fallback because every row written before this task genuinely was a dense run.

- [ ] **Step 11: Shift the two positional assertions the new column moved**

In `tests/test_evaluation_benchmark.py`, `test_the_summary_keeps_one_row_per_matching_label`:

```python
    assert [row[2] for row in rows] == ["fixed", "sentence"]
    assert rows[0][5] == "0.500"
```

And in `test_the_summary_keeps_only_the_last_run_of_a_repeated_label`:

```python
    assert rows[0][5] == "0.900"
```

- [ ] **Step 12: Add the three flags and the pre-loop build to `scripts/benchmark.py`**

After the `--mode` argument added in Task 2:

```python
    parser.add_argument(
        "--candidates",
        type=int,
        help="per-branch depth before fusion; default: RETRIEVAL_CANDIDATES",
    )
    parser.add_argument("--rrf-k", type=int, help="RRF constant; default: RRF_K")
```

Change the import to pull in the index builder:

```python
from app.retrieval.bm25 import default_index  # noqa: E402
```

Immediately before the `def retrieve` definition:

```python
    mode = args.mode or settings.retrieval_mode
    if mode != "dense":
        # Built here, outside the timed loop, on purpose: the ~200 ms
        # construction charged to the first question would make the p50 column
        # stop meaning per-query retrieval latency, which is all it is used for.
        default_index(settings, collection)
```

Pass the two new parameters in `retrieve`:

```python
        return search(
            text,
            top_k=args.top_k,
            mode=mode,
            candidates=args.candidates,
            rrf_k=args.rrf_k,
            filters=filters or None,
            collection=collection,
            settings=settings,
        )
```

And record them, replacing the `"mode"` line from Task 2:

```python
            "mode": mode,
            "candidates": args.candidates or settings.retrieval_candidates,
            "rrf_k": args.rrf_k or settings.rrf_k,
```

Add the usage line to the module docstring:

```python
    uv run python scripts/benchmark.py --label "hybrid-k60-d50" --mode hybrid --candidates 50
```

- [ ] **Step 13: Run the full gates**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest
```

- [ ] **Step 14: Eyeball hybrid mode on the real corpus**

```bash
uv run python scripts/search.py "HTTPException 422" --mode hybrid
```

Expected: a mix — the `handling-errors` chunks dense search finds, plus at least one chunk containing the literal `422`. Scores around 0.03, not 0.3: that is the documented RRF consequence, not a bug.

- [ ] **Step 15: Measure hybrid at the default parameters**

```bash
uv run python scripts/benchmark.py --label "hybrid-k60-d50" --mode hybrid \
  --candidates 50 --rrf-k 60 --collection chunks_sentence --strategy sentence \
  --compare "dense-sentence-doctype"
```

Record whatever comes out. Do not tune anything yet — Task 5 is the sweep, and adjusting parameters now to chase a number before the sweep exists is how a pre-registered rule gets quietly abandoned.

- [ ] **Step 16: Commit and tag**

```bash
git add app/core/config.py app/retrieval/search.py app/generation/answer.py \
  app/evaluation/benchmark.py scripts/ask.py scripts/benchmark.py \
  tests/test_retrieval_search.py tests/test_evaluation_benchmark.py \
  .env.example data/eval/results.jsonl
git commit -m "feat(retrieval): fuse dense and BM25 with reciprocal rank fusion

rrf() consumes ranks and ignores scores, so no per-query normalisation is
invented for two incomparable scales. mode= reaches answer_question, so
the retriever step 16 measures is the one ask.py uses. The index is built
before the timed loop, keeping p50 a per-query number.

Records the hybrid-k60-d50 row.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
git tag v0.5
```

---

### Task 5: The sweep

**Files:**
- Modify: `data/eval/results.jsonl` (appended by each run)

No code changes. Four runs, one parameter moved at a time from `k=60, depth=50`. Every run is free: the embedding cache serves the query vectors and BM25 makes no API call.

- [ ] **Step 1: Confirm the starting point is on the record**

```bash
uv run python scripts/benchmark.py --summary "*sentence*"
uv run python scripts/benchmark.py --summary "hybrid-*"
```

Expected: `dense-sentence-doctype`, `bm25-sentence` and `hybrid-k60-d50` all present, with a populated `mode` column. If `hybrid-k60-d50` is missing, Task 4 Step 15 did not run.

- [ ] **Step 2: Sweep the RRF constant**

```bash
uv run python scripts/benchmark.py --label "hybrid-k20-d50" --mode hybrid \
  --candidates 50 --rrf-k 20 --collection chunks_sentence --strategy sentence

uv run python scripts/benchmark.py --label "hybrid-k100-d50" --mode hybrid \
  --candidates 50 --rrf-k 100 --collection chunks_sentence --strategy sentence
```

- [ ] **Step 3: Sweep the candidate depth**

```bash
uv run python scripts/benchmark.py --label "hybrid-k60-d20" --mode hybrid \
  --candidates 20 --rrf-k 60 --collection chunks_sentence --strategy sentence

uv run python scripts/benchmark.py --label "hybrid-k60-d100" --mode hybrid \
  --candidates 100 --rrf-k 60 --collection chunks_sentence --strategy sentence
```

- [ ] **Step 4: Read the whole table at once**

```bash
uv run python scripts/benchmark.py --summary "*sentence*"
uv run python scripts/benchmark.py --summary "hybrid-*"
```

Six rows: the dense baseline, BM25 alone, and the four fusion variants. Read the
per-category numbers out of the history rather than re-running anything — every
run already stored them:

```bash
uv run python -c "import json; rows=[json.loads(l) for l in open('data/eval/results.jsonl',encoding='utf-8')]; [print(r['label'], {k: round(v,3) for k,v in r['aggregate'].items() if k in ('recall@5','recall@10','mrr','ndcg@5')}, {c: round(s['recall@5'],3) for c,s in r['per_category'].items()}) for r in rows if r['label'].startswith(('hybrid-','bm25-','dense-sentence-doctype'))]"
```

Write down, for the best row: aggregate Recall@5, Recall@10, MRR, NDCG@5, p50/p95, the four per-category Recall@5 values, and `Recall@10 − Recall@5`.

- [ ] **Step 5: Commit the history**

```bash
git add data/eval/results.jsonl
git commit -m "chore(evaluation): record the RRF constant and candidate depth sweep

Four runs from k=60/depth=50, one parameter at a time. Inputs to the
acceptance rule in the step 14-16 design doc; the verdict is the next
commit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 6: Apply the acceptance rule

**Files:**
- Modify: `app/core/config.py`, `.env.example` — **only if the rule is met**

The rule, from the spec, fixed before any run: `RETRIEVAL_MODE` changes from `dense` to `hybrid` **if and only if** the best sweep row reaches aggregate Recall@5 of **0.786 or better** *and* no per-category Recall@5 regresses by more than **0.05** against exact 0.892, code 0.850, conceptual 0.733, multi_doc 0.594.

- [ ] **Step 1: Evaluate both clauses explicitly, in writing**

State in the commit body: the best label, its Recall@5, whether clause 1 passes, each of the four per-category deltas, and whether clause 2 passes. Both clauses, every time, even when the first one obviously fails — the second is what catches an aggregate bought by improving `exact` while breaking `conceptual`.

- [ ] **Step 2A: If the rule is met — flip the default**

In `app/core/config.py`:

```python
    # Step 16 measured three modes: hybrid won Recall@5 <VALUE> against the
    # 0.776 dense baseline, with no category regressing more than 0.05.
    retrieval_mode: str = "hybrid"
    retrieval_candidates: int = <WINNING DEPTH>
    rrf_k: int = <WINNING K>
```

Update the same three values in `.env.example`, then re-run the gates and confirm the default path still works:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest
uv run python scripts/ask.py "How do I handle validation errors?"
```

- [ ] **Step 2B: If the rule is not met — leave the default alone**

Change no setting. This is a result, not a failure: the code, the tests and six measured rows all ship, and Task 7 records the finding in the same terms step 13 used for its routing ceiling. Do not widen the criterion, do not add a seventh parameter combination hoping for a better row, and do not re-scope the verdict onto a favourable category slice — the spec rejects all three by name.

- [ ] **Step 3: Commit the verdict**

```bash
git add app/core/config.py .env.example
git commit -F - <<'MSG'
feat(retrieval): <hybrid becomes the default | keep dense as the default>

<Best label> reached Recall@5 <value> against the 0.776 dense baseline
(<pass|fail>: the rule needs 0.786). Per-category Recall@5 deltas: exact
<d>, code <d>, conceptual <d>, multi_doc <d> (<pass|fail>: the rule allows
-0.05). Recall@10 - Recall@5 moved from 0.009 to <value>.

<One sentence on what that means for step 17.>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
MSG
git tag v0.6
```

If Step 2B applied and no file changed, commit the tag alone with `git tag v0.6` and record the verdict in Task 7's documentation commit instead.

---

### Task 7: The 422 transcript and the documentation

**Files:**
- Modify: `README.md`, `docs/roadmap.md`

- [ ] **Step 1: Capture the 422 transcript, both modes**

```bash
uv run python scripts/ask.py "What does HTTPException 422 mean?" --mode dense
uv run python scripts/ask.py "What does HTTPException 422 mean?" --mode hybrid
```

Keep both outputs verbatim. This is the phase's qualitative evidence and it is attached to no metric — label it that way in the README. If hybrid mode still answers "I do not know", say so: an honest negative transcript is worth more than a quietly omitted one.

- [ ] **Step 2: README — the results table**

Fill the `Recherche hybride` row (currently all em dashes, around line 1000) with the winning run's Recall@5, Recall@10, MRR, NDCG@5 and p50. Add a `BM25 seul` row from `bm25-sentence` so the fusion has a visible denominator.

- [ ] **Step 3: README — Phase 5-6 and the current state**

Tick the Phase 5 roadmap checkbox (around line 974). Update the current-state section and the stack table's `Recherche lexicale` row from `Planifié` to the shipped mode. Document the new commands:

```bash
uv run python scripts/search.py "HTTPException 422" --mode hybrid
uv run python scripts/benchmark.py --label "hybrid-k60-d50" --mode hybrid --candidates 50
```

- [ ] **Step 4: README — the honest finding**

Replace the three passages that promise step 14 will fix 422 (around lines 235, 327, 644 and 799) with what actually happened. Whatever the verdict, the paragraph must say:

- that `exact` was already the strongest category at 0.892 before any of this, so the phase's headline target was never the aggregate's weak point;
- the measured aggregate delta, and the per-category table;
- what `Recall@10 − Recall@5` did, and what that means for step 17;
- that the 422 transcript is qualitative evidence with no ground truth behind it.

- [ ] **Step 5: `docs/roadmap.md` — current state**

- Change the headline from "Step 14 (BM25) is next" to steps 14-16 done and step 17 next.
- Add the two shipped artifacts to the table: the BM25 index, and hybrid retrieval with its measured numbers.
- Rewrite the first bullet in "Four things later steps own" — `HTTPException 422` is no longer step 14's target, it is a resolved-or-not finding.
- Update the `Recall@10 minus Recall@5` bullet with the new value, and state plainly whether step 17 is still worth running.
- Tick the `14-16` row in the step map.
- Add this plan and the spec to "Plans written so far".

- [ ] **Step 6: Verify the docs against reality**

Every number in the README and the roadmap must appear in `data/eval/results.jsonl`. Check each one:

```bash
uv run python scripts/benchmark.py --summary "*"
```

A results cell that does not match a recorded run is exactly the thing rule 1 exists to prevent.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/roadmap.md
git commit -m "docs: publish the hybrid search and RRF results

Records what the sweep actually measured, including that exact was the
strongest category at 0.892 before the phase began and the 422 case never
had ground truth in the evaluation set. States what Recall@10 - Recall@5
did and what that means for step 17.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: the BM25 module and its three design calls (Task 1); the registry, the settings and the corrected `store.py` comment (Task 2); `matches_filters` and filter parity (Task 3); `rrf`, `_hybrid`, the scripts, the `mode` column and the pre-loop index build (Task 4); the four-run sweep (Task 5); the pre-registered rule (Task 6); the 422 transcript and both documents (Task 7). The spec's "What RRF changes about a score" section is implemented as `rrf`'s docstring plus `test_rrf_score_is_the_fused_score_not_a_cosine`, and its latency section as Task 4 Step 12.

**Two things the spec did not anticipate**, both found by reading the existing tests and folded in above:

- `answer_question` calls `retriever(question, top_k=..., filters=..., settings=...)` with no `mode`, so `ask.py --mode` could not have reached the pipeline. Task 4 Step 8 threads it through. Without this, Task 7's 422 transcript would have been impossible to produce.
- `tests/test_evaluation_benchmark.py` asserts on `summarise` rows by position (`row[1]`, `rows[0][4]`). Inserting the `mode` column shifts both. Task 4 Step 11 updates them explicitly rather than leaving a mystery failure.

**Placeholders.** The only bracketed values are in Task 6, where the winning parameters cannot be known before Task 5 runs, and in Task 7's documentation, which reports measurements. Every code step carries complete code.

**Type consistency.** `BM25Index.search(query, top_k, predicate=...)` is called with that shape in `_lexical`; `rrf(rankings, *, k, top_k)` is called that way in `_hybrid` and in every test; `default_index(settings, collection)` is called that way in `search()` and in `scripts/benchmark.py`; `matches_filters(chunk, filters)` matches its uses in `_lexical`'s lambda and the tests. `Retrieve = Callable[..., list[ScoredChunk]]` covers all three mode functions, whose differing tails are absorbed by `**_: Any`.
