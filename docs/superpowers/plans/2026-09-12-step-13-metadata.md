# Step 13 — Metadata Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every chunk a `doc_type` facet, make `search()` filter on any indexed payload field, and measure what a perfect facet filter would be worth.

**Architecture:** `doc_type` is derived from the corpus path's first segment in `loader.py` — one function, no mapping table, generalises to any future source. It rides through `RawDocument` → `Chunk` → the Qdrant payload, where it joins `source`/`document_id`/`language` as an indexed keyword field. `search()`'s `source: str | None` parameter is replaced by a generic `filters: Mapping[str, str | Sequence[str]] | None`, so steps 14-17 add facets without touching six call sites again. The step's deliverable is not the filter but the number it produces: an oracle run that filters each eval question to its ground-truth facet, giving the ceiling on facet routing before step 18 considers building a router.

**Tech Stack:** Python 3.12, pydantic v2, qdrant-client, pytest. No new dependencies.

**Spec:** No separate spec file — this plan was agreed in the brainstorming session of 2026-09-12 and carries the design inline. The governing documents are [`docs/roadmap.md`](../../roadmap.md) (step 13 row, "How a step lands") and [`information.md`](../../../information.md) (Phase 3).

## Global Constraints

- **No new dependencies.** Roadmap rule 3. `pyproject.toml` and `uv.lock` are untouched by this step.
- **Quality gates must pass before every commit:** `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`.
- **TDD, no exceptions.** Write the failing test, watch it fail, then the code.
- **Measure, then improve.** Roadmap rule 1. No number is written into the README that did not come out of a benchmark run recorded in `data/eval/results.jsonl`.
- **Tests never touch the network.** The existing `FakeClient`/`FakeEmbedder` fixtures in `tests/test_retrieval_search.py` are the pattern; integration tests that need Qdrant use the existing `indexed` fixture.
- **Python line length 100**, ruff-formatted, mypy strict on `app`.

## Facts this plan is built on (measured 2026-09-12)

Corpus path prefixes, 155 documents:

```
tutorial 40   advanced 31   (root) 22   reference 20   how-to 12   deployment 9
learn 1   about 1   resources 1
```

Eval set, 45 non-held-out questions, 38 answerable:

```
single-facet ground truth:  16   (tutorial 8, root 5, advanced 2, deployment 1)
multi-facet  ground truth:  22
```

**Those 16 are the whole oracle experiment.** A perfect router can help at most 16 of 38 questions; on the other 22 any single-facet filter removes at least one relevant document by construction. `tutorial` (n=8) and `root` (n=5) are the only per-facet buckets with enough questions to read; `advanced` (n=2) and `deployment` (n=1) are reported with their `n` next to them and are not conclusions.

No two questions in `data/eval/questions.jsonl` share the same text, which is what makes the oracle's `{question_text: facets}` dict safe.

---

### Task 1: `doc_type` on `RawDocument`

**Files:**
- Modify: `app/ingestion/loader.py`
- Modify: `app/models/documents.py`
- Test: `tests/test_ingestion_loader.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `app.ingestion.loader.doc_type_of(relative_path: str) -> str` and the field `RawDocument.doc_type: str` (required, no default). Task 2 copies the field onto `Chunk`; Task 5 imports `doc_type_of` to bucket eval questions.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingestion_loader.py`:

```python
from app.ingestion.loader import doc_type_of


def test_doc_type_is_the_first_path_segment() -> None:
    assert doc_type_of("tutorial/first-steps.md") == "tutorial"


def test_a_nested_path_still_reports_its_top_segment() -> None:
    assert doc_type_of("tutorial/dependencies/classes-as-dependencies.md") == "tutorial"


def test_a_file_at_the_corpus_root_is_root() -> None:
    assert doc_type_of("index.md") == "root"


def test_loading_a_document_sets_its_doc_type(tmp_path: Path) -> None:
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment" / "docker.md").write_text("# Docker\n\nbody\n", encoding="utf-8")
    [document] = load_documents(tmp_path, "fastapi")
    assert document.doc_type == "deployment"
```

`Path` and `load_documents` are already imported at the top of that file — check before adding duplicate imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ingestion_loader.py -v`
Expected: FAIL — `ImportError: cannot import name 'doc_type_of'`.

- [ ] **Step 3: Add the field to the model**

In `app/models/documents.py`, add to `RawDocument`, directly after `path: str`:

```python
    # Step 13's filterable facet: the corpus's own directory structure, which is
    # the only metadata that actually varies while there is one source.
    doc_type: str
```

- [ ] **Step 4: Add the derivation and use it**

In `app/ingestion/loader.py`, add above `load_document`:

```python
def doc_type_of(relative_path: str) -> str:
    """The corpus section a path belongs to: its first directory, or ``root``.

    Derived rather than mapped, so a second source with a different layout gets
    its own facets for free instead of silently collapsing into ``other``.
    """
    head, _, tail = relative_path.partition("/")
    return head if tail else "root"
```

In `load_document`, add to the `RawDocument(...)` call, after `path=relative,`:

```python
        doc_type=doc_type_of(relative),
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ingestion_loader.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`
Expected: all green. `tests/test_ingestion_clean.py:259` constructs a `RawDocument` directly and will now fail on the missing required field — add `doc_type="tutorial",` to that constructor.

- [ ] **Step 7: Commit**

```bash
git add app/models/documents.py app/ingestion/loader.py tests/test_ingestion_loader.py tests/test_ingestion_clean.py
git commit -m "feat(ingestion): derive a doc_type facet from the corpus layout

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 2: `doc_type` on `Chunk`, indexed in Qdrant

**Files:**
- Modify: `app/models/chunks.py`
- Modify: `app/ingestion/chunk.py:373-386` (the `Chunk(...)` construction inside `chunk_document`)
- Modify: `app/retrieval/store.py:23` (`INDEXED_FIELDS`)
- Test: `tests/test_models_chunks.py`, `tests/test_ingestion_chunk.py`, `tests/test_retrieval_store.py`
- Modify (fixtures only): `tests/test_evaluation_benchmark.py:14`, `tests/test_generation_answer.py:16`, `tests/test_generation_context.py`, `tests/test_retrieval_search.py:17`, `tests/test_retrieval_search.py:168`

**Interfaces:**
- Consumes: `RawDocument.doc_type` from Task 1.
- Produces: `Chunk.doc_type: str` (required, no default), present in `Chunk.to_payload()` output and in `INDEXED_FIELDS`. Task 3 validates filter keys against `INDEXED_FIELDS`.

**Why required and not defaulted:** `chunk_from_payload` validates payloads written by whichever indexer last ran. A default turns an un-reindexed collection into a run that reports a plausible facet distribution and is wrong. A required field turns it into a loud pydantic error whose fix is a 4-second reindex.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models_chunks.py`:

```python
def test_the_payload_carries_doc_type() -> None:
    assert make_chunk().to_payload()["doc_type"] == "tutorial"
```

Append to `tests/test_ingestion_chunk.py`:

```python
def test_chunks_inherit_the_documents_doc_type() -> None:
    document = RawDocument(
        document_id="fastapi:deployment/docker",
        source="fastapi",
        title="Docker",
        path="deployment/docker.md",
        doc_type="deployment",
        url=None,
        text="Some prose about containers. " * 60,
        content_hash="0" * 64,
    )
    chunks = chunk_document(document)
    assert chunks
    assert {chunk.doc_type for chunk in chunks} == {"deployment"}
```

Check the existing imports in `tests/test_ingestion_chunk.py` for `RawDocument` and `chunk_document` before adding them; match the module's existing helper for building a `RawDocument` if one already exists rather than writing a second one.

Append to `tests/test_retrieval_store.py`:

```python
def test_doc_type_is_a_payload_indexed_field() -> None:
    assert "doc_type" in INDEXED_FIELDS
```

Import `INDEXED_FIELDS` from `app.retrieval.store` at the top of that file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models_chunks.py tests/test_ingestion_chunk.py tests/test_retrieval_store.py -v`
Expected: FAIL — `KeyError: 'doc_type'`, a pydantic `ValidationError` for the unexpected `doc_type` argument to `RawDocument` if Task 1 was skipped, and an assertion failure on `INDEXED_FIELDS`.

- [ ] **Step 3: Add the field to `Chunk`**

In `app/models/chunks.py`, add to `Chunk` directly after `url: str | None = None`:

```python
    # Required, no default: a payload written before step 13 must fail loudly in
    # chunk_from_payload rather than report a made-up facet into a benchmark row.
    doc_type: str
```

Pydantic requires non-defaulted fields before defaulted ones only for dataclasses, not `BaseModel` — placing it here is fine and keeps it next to the other document-level metadata.

- [ ] **Step 4: Copy it through the chunker**

In `app/ingestion/chunk.py`, inside the `Chunk(...)` construction in `chunk_document`, add after `url=document.url,`:

```python
                doc_type=document.doc_type,
```

- [ ] **Step 5: Index it in Qdrant**

In `app/retrieval/store.py`, change line 23:

```python
INDEXED_FIELDS = ("source", "document_id", "language", "doc_type")
```

The comment above it still reads "so step 13's metadata filtering is a query-time change instead of a re-index" — update it to reflect that step 13 has now arrived:

```python
# Indexed keyword fields. Anything filterable must be here: an unindexed filter
# still works but scans, and search.build_filter() rejects keys that are not.
```

- [ ] **Step 6: Fix the remaining `Chunk(...)` fixtures**

Every direct `Chunk(...)` construction now needs `doc_type`. Add `doc_type="tutorial",` to each:

- `tests/test_models_chunks.py:8`
- `tests/test_evaluation_benchmark.py:14`
- `tests/test_generation_answer.py:16`
- `tests/test_retrieval_search.py:17`
- `tests/test_retrieval_search.py:168`
- `tests/test_generation_context.py` — grep for `Chunk(` and fix each site
- `tests/test_retrieval_store.py:21`

- [ ] **Step 7: Run the full gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`
Expected: all green. A remaining failure naming `doc_type` is a fixture missed in step 6.

- [ ] **Step 8: Commit**

```bash
git add app/models/chunks.py app/ingestion/chunk.py app/retrieval/store.py tests/
git commit -m "feat(retrieval): carry doc_type into the payload and index it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 3: `source=` becomes a generic `filters=`

**Files:**
- Modify: `app/retrieval/search.py` (replace `source_filter`, change `search`'s signature)
- Modify: `app/generation/answer.py:38,59` (the `source` parameter and the retriever call)
- Modify: `scripts/search.py`, `scripts/ask.py`, `scripts/benchmark.py` (CLI flag)
- Test: `tests/test_retrieval_search.py`, `tests/test_generation_answer.py:146`

**Interfaces:**
- Consumes: `INDEXED_FIELDS` from Task 2.
- Produces:
  - `app.retrieval.search.build_filter(filters: Mapping[str, str | Sequence[str]] | None) -> Filter | None`
  - `app.retrieval.search.parse_filters(pairs: Sequence[str]) -> dict[str, list[str]]`
  - `search(query, *, top_k=5, filters=None, collection=None, settings=None, client=None, embedder=None)` — `source` is **gone**, not deprecated.
  - `answer_question(question, *, top_k=None, filters=None, ...)` — likewise.

  Task 5 calls `search(..., filters=...)` and `parse_filters`.

**This is one task, not two.** Removing `source=` breaks `answer.py` and three scripts at the type level; mypy is red until every caller moves. Splitting it would produce a commit that fails the gates.

- [ ] **Step 1: Write the failing tests**

In `tests/test_retrieval_search.py`, replace `test_a_source_becomes_a_payload_filter` and `test_no_source_means_no_filter` with:

```python
def test_a_scalar_filter_becomes_a_match_value() -> None:
    client = FakeClient()
    run(client=client, filters={"source": "fastapi"})
    [condition] = client.calls[0]["query_filter"].must
    assert condition.key == "source"
    assert condition.match.value == "fastapi"


def test_a_sequence_filter_becomes_a_match_any() -> None:
    client = FakeClient()
    run(client=client, filters={"doc_type": ["tutorial", "advanced"]})
    [condition] = client.calls[0]["query_filter"].must
    assert condition.key == "doc_type"
    assert condition.match.any == ["tutorial", "advanced"]


def test_two_filters_are_anded_in_a_stable_order() -> None:
    client = FakeClient()
    run(client=client, filters={"doc_type": "tutorial", "source": "fastapi"})
    assert [c.key for c in client.calls[0]["query_filter"].must] == ["doc_type", "source"]


def test_no_filters_means_no_filter() -> None:
    client = FakeClient()
    run(client=client)
    assert client.calls[0]["query_filter"] is None


def test_an_empty_filters_mapping_means_no_filter() -> None:
    client = FakeClient()
    run(client=client, filters={})
    assert client.calls[0]["query_filter"] is None


def test_an_unindexed_filter_key_is_rejected_by_name() -> None:
    """A typo'd key would otherwise scan the whole collection and match nothing."""
    with pytest.raises(ValueError, match="doctype"):
        run(filters={"doctype": "tutorial"})


def test_an_empty_value_list_is_rejected() -> None:
    """MatchAny([]) matches nothing, which reads as 'retrieval is broken'."""
    with pytest.raises(ValueError, match="doc_type"):
        run(filters={"doc_type": []})


def test_parse_filters_reads_key_value_pairs() -> None:
    assert parse_filters(["source=fastapi"]) == {"source": ["fastapi"]}


def test_parse_filters_splits_comma_separated_values() -> None:
    assert parse_filters(["doc_type=tutorial,advanced"]) == {"doc_type": ["tutorial", "advanced"]}


def test_parse_filters_merges_a_repeated_key() -> None:
    assert parse_filters(["doc_type=tutorial", "doc_type=advanced"]) == {
        "doc_type": ["tutorial", "advanced"]
    }


def test_parse_filters_rejects_a_pair_with_no_equals() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_filters(["doc_type"])


def test_parse_filters_rejects_an_empty_value() -> None:
    with pytest.raises(ValueError, match="doc_type"):
        parse_filters(["doc_type="])
```

Update the import at the top of the file:

```python
from app.retrieval.search import build_filter, parse_filters, search
```

(`build_filter` is imported for symmetry with the module's public surface even though these tests exercise it through `search`; if ruff flags it as unused, drop it from the import.)

Also update the integration test at `tests/test_retrieval_search.py:221`:

```python
def test_a_filter_that_matches_nothing_returns_nothing(indexed: Settings) -> None:
    assert search("dependency injection", filters={"source": "django"}, settings=indexed) == []


def test_a_doc_type_filter_restricts_results_to_that_facet(indexed: Settings) -> None:
    results = search(
        "dependency injection", top_k=5, filters={"doc_type": "tutorial"}, settings=indexed
    )
    assert results
    assert {scored.chunk.doc_type for scored in results} == {"tutorial"}
```

In `tests/test_generation_answer.py`, replace `test_the_retriever_gets_top_k_and_the_source_filter` at line 146 with:

```python
def test_the_retriever_gets_top_k_and_the_filters() -> None:
    retriever = FakeRetriever()
    ask(retriever=retriever, top_k=7, filters={"doc_type": "tutorial"})
    assert retriever.calls[0]["top_k"] == 7
    assert retriever.calls[0]["filters"] == {"doc_type": "tutorial"}
```

`FakeRetriever` and `ask` already exist in that file (lines 30 and 50); nothing else in it changes.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_search.py -v -k "filter"`
Expected: FAIL — `ImportError: cannot import name 'parse_filters'`.

- [ ] **Step 3: Replace `source_filter` in `app/retrieval/search.py`**

Change the imports at the top:

```python
from collections.abc import Callable, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.core.config import Settings, get_settings
from app.ingestion.embed import EmbeddingCache, embed_query
from app.models.chunks import ScoredChunk
from app.retrieval.store import INDEXED_FIELDS, chunk_from_payload, get_client

Embedder = Callable[[str], list[float]]
Filters = Mapping[str, str | Sequence[str]]
```

Delete `source_filter` entirely and put these two in its place:

```python
def build_filter(filters: Filters | None) -> Filter | None:
    """Turn ``{"doc_type": "tutorial", "source": ["fastapi", "starlette"]}`` into a
    Qdrant filter: a scalar matches one value, a sequence matches any of them, and
    several keys are ANDed.

    Keys are checked against ``INDEXED_FIELDS`` rather than passed through. An
    unindexed key is a full scan; a misspelled one (``doctype``) is a filter that
    silently matches nothing, which reads downstream as "retrieval is broken"
    rather than "the flag is wrong".
    """
    if not filters:
        return None
    unknown = sorted(set(filters) - set(INDEXED_FIELDS))
    if unknown:
        raise ValueError(f"not an indexed field: {', '.join(unknown)}; have {INDEXED_FIELDS}")

    conditions = []
    for key, value in filters.items():
        if isinstance(value, str):
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            continue
        values = list(value)
        if not values:
            # MatchAny([]) is a filter that matches nothing at all.
            raise ValueError(f"{key} was given an empty list of values")
        conditions.append(FieldCondition(key=key, match=MatchAny(any=values)))
    return Filter(must=conditions)


def parse_filters(pairs: Sequence[str]) -> dict[str, list[str]]:
    """``["doc_type=tutorial,advanced"]`` -> ``{"doc_type": ["tutorial", "advanced"]}``.

    Shared by the three scripts so ``--filter`` means the same thing everywhere.
    """
    parsed: dict[str, list[str]] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--filter expects key=value, got {pair!r}")
        values = [item.strip() for item in value.split(",") if item.strip()]
        if not values:
            raise ValueError(f"--filter {key} was given no value")
        parsed.setdefault(key.strip(), []).extend(values)
    return parsed
```

- [ ] **Step 4: Change `search`'s signature**

Replace `source: str | None = None,` in the signature with `filters: Filters | None = None,` and the call site:

```python
        query_filter=build_filter(filters),
```

Update the docstring paragraph that mentions `source` to describe `filters` instead:

```
    ``filters`` restricts the search to payload values: ``{"doc_type": "tutorial"}``
    or ``{"doc_type": ["tutorial", "advanced"]}``. Only fields in
    ``store.INDEXED_FIELDS`` are accepted, so a filter is always an index lookup.
```

- [ ] **Step 5: Thread it through `app/generation/answer.py`**

Change the signature parameter `source: str | None = None,` to `filters: Filters | None = None,`, import `Filters` from `app.retrieval.search`, and change line 59:

```python
    chunks = retriever(question, top_k=top_k or settings.top_k, filters=filters, settings=settings)
```

- [ ] **Step 6: Move the three scripts to `--filter`**

In `scripts/search.py`: import `parse_filters` alongside `search`, replace the `--source` argument with

```python
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable; e.g. --filter doc_type=tutorial --filter doc_type=tutorial,advanced",
    )
```

and the call with `search(args.query, top_k=args.top_k, filters=parse_filters(args.filter))`.

In `scripts/ask.py`: the same `--filter` argument, then

```python
    filters = parse_filters(args.filter)
```

near the top of `main`, and pass `filters=filters` to both the `search(...)` call in the `--show-context` branch and to `answer_question(...)`. Import `parse_filters` from `app.retrieval.search`.

In `scripts/benchmark.py`: replace `--source` with the same `--filter` argument; Task 5 wires it into the retriever, so for now just parse it and pass `filters=parse_filters(args.filter) or None` to `search` inside the lambda, and change the `config` dict entry `"source": args.source,` to `"filters": parse_filters(args.filter) or None,`.

Update each script's module docstring usage line to show `--filter` instead of `--source`.

- [ ] **Step 7: Run the full gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`
Expected: all green. The integration tests marked with the `indexed` fixture need a running Qdrant and a collection indexed by Task 2's code — if `test_a_doc_type_filter_restricts_results_to_that_facet` fails with a pydantic error naming `doc_type`, the local collection is stale; Task 6 reindexes it. Run the reindex from Task 6 Step 1 now if that happens, then re-run.

- [ ] **Step 8: Commit**

```bash
git add app/retrieval/search.py app/generation/answer.py scripts/ tests/
git commit -m "feat(retrieval): filter on any indexed payload field

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 4: per-facet recall in the benchmark

**Files:**
- Modify: `app/evaluation/benchmark.py` (`BenchmarkResult`, `run_benchmark`)
- Modify: `scripts/benchmark.py` (`print_result`)
- Test: `tests/test_evaluation_benchmark.py`

**Interfaces:**
- Consumes: `doc_type_of` from Task 1.
- Produces: `BenchmarkResult.per_doc_type: dict[str, dict[str, float]]`, and a `doc_types: list[str]` key on every answerable row in `per_question`. Task 5 reads neither; Task 6 reads the printed table.

**Why only single-facet questions:** a question whose ground truth spans `tutorial` and `advanced` belongs to no single bucket. Counting it in both inflates whichever bucket the retriever happened to hit; dropping it is the only honest option, so the bucket's `n` is printed next to every score.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation_benchmark.py`:

```python
def test_single_facet_questions_are_bucketed_by_doc_type() -> None:
    questions = [
        EvalQuestion(
            question_id="q1",
            question="how do dependencies work",
            category="conceptual",
            relevant_document_ids=["fastapi:tutorial/dependencies"],
        ),
        EvalQuestion(
            question_id="q2",
            question="how do i deploy",
            category="conceptual",
            relevant_document_ids=["fastapi:deployment/docker"],
        ),
    ]
    result = run_benchmark(questions, perfect, label="t", ks=(1,))
    assert sorted(result.per_doc_type) == ["deployment", "tutorial"]
    assert result.per_doc_type["tutorial"]["questions"] == 1.0


def test_a_multi_facet_question_is_in_no_bucket() -> None:
    questions = [
        EvalQuestion(
            question_id="q1",
            question="how do dependencies work in production",
            category="multi_doc",
            relevant_document_ids=["fastapi:tutorial/dependencies", "fastapi:deployment/docker"],
        )
    ]
    result = run_benchmark(questions, perfect, label="t", ks=(1,))
    assert result.per_doc_type == {}
    assert result.answerable == 1


def test_an_unanswerable_question_is_in_no_bucket() -> None:
    questions = [
        EvalQuestion(
            question_id="q1",
            question="what is the capital of france",
            category="unanswerable",
            relevant_document_ids=[],
        )
    ]
    result = run_benchmark(questions, perfect, label="t", ks=(1,))
    assert result.per_doc_type == {}
```

`perfect` is the scripted retriever already defined at `tests/test_evaluation_benchmark.py:46`; `EvalQuestion` is already imported there. Note that the file's existing `question()` helper builds ids like `doc-a` with no `source:` prefix, so every pre-existing test question buckets as `root` — harmless, since no existing test asserts on `per_doc_type`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_benchmark.py -v -k "facet or bucket"`
Expected: FAIL — `AttributeError: 'BenchmarkResult' object has no attribute 'per_doc_type'`.

- [ ] **Step 3: Add the field and the derivation**

In `app/evaluation/benchmark.py`, add the import:

```python
from app.ingestion.loader import doc_type_of
```

Add a module-level helper below `DEFAULT_ABSTENTION_THRESHOLD`:

```python
def question_doc_type(question: EvalQuestion) -> str | None:
    """The single facet a question's ground truth lives in, or ``None``.

    ``None`` for the 22 of 38 answerable questions whose relevant documents span
    two directories: they belong in no bucket, and putting them in both is how a
    per-facet table starts reporting numbers that cannot be reproduced.
    """
    facets = {
        doc_type_of(document_id.partition(":")[2])
        for document_id in question.relevant_document_ids
    }
    return facets.pop() if len(facets) == 1 else None
```

Add the field to `BenchmarkResult`, after `failures`:

```python
    per_doc_type: dict[str, dict[str, float]] = field(default_factory=dict)
```

- [ ] **Step 4: Populate it in `run_benchmark`**

In the per-question loop, in the `else:` branch that handles answerable questions, add before `answerable_rows.append(row)`:

```python
            row["doc_type"] = question_doc_type(question)
```

After the `per_category` computation, add:

```python
    facets = sorted({row["doc_type"] for row in answerable_rows if row["doc_type"]})
    per_doc_type = {
        facet: {"questions": float(len(rows)), **_mean_rows(rows, metric_keys)}
        for facet in facets
        if (rows := [row for row in answerable_rows if row["doc_type"] == facet])
    }
```

and pass `per_doc_type=per_doc_type,` in the `BenchmarkResult(...)` construction.

- [ ] **Step 5: Print it**

In `scripts/benchmark.py`, at the end of `print_result` (before the `for failure in result.failures:` loop), add:

```python
    # Only the single-facet questions are in here; `n` is printed so a bucket of
    # one is read as a bucket of one.
    if result.per_doc_type:
        print()
        print(
            table(
                ["doc_type", "n", *CATEGORY_COLUMNS],
                [
                    [
                        facet,
                        f"{scores['questions']:.0f}",
                        *[f"{scores[column]:.3f}" for column in CATEGORY_COLUMNS],
                    ]
                    for facet, scores in result.per_doc_type.items()
                ],
            )
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_benchmark.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add app/evaluation/benchmark.py scripts/benchmark.py tests/test_evaluation_benchmark.py
git commit -m "feat(evaluation): break recall down by doc_type

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 5: the oracle filter

**Files:**
- Modify: `scripts/benchmark.py` (`--oracle-filter` flag, the retriever lambda, the `config` dict)
- Test: none — this is fifteen lines of script wiring over `build_filter` and `question_doc_type`, both tested in Tasks 3 and 4.

**Interfaces:**
- Consumes: `parse_filters` and `search(filters=)` from Task 3, `question_doc_type` from Task 4.
- Produces: `uv run python scripts/benchmark.py --oracle-filter --label ...`, which Task 6 runs.

**Why this does not touch `run_benchmark`:** `Retriever = Callable[[str], list[ScoredChunk]]` is the seam steps 14-19 swap hybrid search, RRF and reranking into. Widening it so one experiment can see the question object would cost every one of those steps a signature change. A dict keyed on question text, closed over by the lambda, is exact — no two questions in the dataset share a text — and costs the seam nothing.

- [ ] **Step 1: Add the flag**

In `scripts/benchmark.py`'s argument parser, after `--filter`:

```python
    parser.add_argument(
        "--oracle-filter",
        action="store_true",
        help="filter each question to its ground-truth doc_type — the ceiling on "
        "facet routing, not a retriever you can ship",
    )
```

- [ ] **Step 2: Build the lookup and the retriever**

Import `question_doc_type` from `app.evaluation.benchmark` and replace the `run_benchmark(...)` retriever lambda with:

```python
    base_filters = parse_filters(args.filter)
    # Keyed on the question text because run_benchmark hands the retriever a
    # string: widening that seam for one experiment would cost steps 14-19 a
    # signature change each. Exact as long as no two questions share a text.
    oracle = {q.question: question_doc_type(q) for q in questions} if args.oracle_filter else {}
    if args.oracle_filter and len(oracle) != len(questions):
        raise SystemExit("two questions share the same text; the oracle lookup would be wrong")

    def retrieve(text: str) -> list[ScoredChunk]:
        filters = dict(base_filters)
        if facet := oracle.get(text):
            filters["doc_type"] = facet
        return search(
            text,
            top_k=args.top_k,
            filters=filters or None,
            collection=collection,
            settings=settings,
        )

    result = run_benchmark(
        questions,
        retrieve,
        ks=[k for k in KS if k <= args.top_k],
        label=args.label,
        config={...},  # unchanged from what is already there, plus Step 3 below
        abstention_threshold=args.abstention_threshold,
    )
```

Only the second argument changes: the lambda becomes the `retrieve` function defined above it. `ks`, `label`, `config` and `abstention_threshold` keep the exact values they already have in the file. Import `ScoredChunk` from `app.models.chunks` for the annotation.

- [ ] **Step 3: Record it in the config**

Add to the `config` dict passed to `run_benchmark`:

```python
            "oracle_filter": args.oracle_filter,
```

A row in `results.jsonl` that does not say whether the oracle was on is a row nobody can interpret in three weeks.

- [ ] **Step 4: Run the gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`
Expected: all green.

- [ ] **Step 5: Smoke-test it without saving**

Run: `uv run python scripts/benchmark.py --label smoke --oracle-filter --no-save`
Expected: a normal benchmark report, a `doc_type` table, and no line appended to `data/eval/results.jsonl`. This needs Qdrant running and a collection indexed by Task 2's code — if it fails with a pydantic error naming `doc_type`, run Task 6 Step 1 first.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark.py
git commit -m "feat(evaluation): measure the ceiling on doc_type routing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 6: reindex, measure, record

**Files:**
- Modify: `data/eval/results.jsonl` (appended by the runs)

**Interfaces:**
- Consumes: everything above.
- Produces: two rows in `data/eval/results.jsonl` (`dense-sentence-doctype` and `dense-sentence-oracle-filter`) that Task 7 quotes.

No code here. This is the step's actual output, and it is the one part that cannot be faked.

- [ ] **Step 1: Reindex every collection**

`Chunk` gained a required field, so every payload written before Task 2 is now unreadable. Embeddings are cached, so this costs seconds and no money. Confirm Qdrant is up first (`docker compose up -d`), then:

```bash
uv run python scripts/index_corpus.py --strategy sentence   --collection chunks_sentence
uv run python scripts/index_corpus.py --strategy recursive  --collection chunks_recursive
uv run python scripts/index_corpus.py --strategy fixed      --collection chunks_fixed
uv run python scripts/index_corpus.py --strategy semantic   --collection chunks_semantic
uv run python scripts/index_corpus.py
```

The last one reindexes whatever `QDRANT_COLLECTION` points at. Check `.env` first — if it already names one of the four above, that run is a no-op repeat and can be skipped. Each run should report the same point count as before (`sentence`: 1 484) — a changed count means the chunker changed, which Task 2 did not intend.

- [ ] **Step 2: Confirm the facet actually landed**

```bash
uv run python scripts/search.py "how do dependencies work" --filter doc_type=tutorial --top-k 5
uv run python scripts/search.py "how do dependencies work" --filter doc_type=deployment --top-k 5
```

Expected: two visibly different result sets, every `document_id` in the first under `tutorial/`, every one in the second under `deployment/`. If both return the same thing, the filter is not reaching Qdrant.

Also confirm the guard fires:

```bash
uv run python scripts/search.py "anything" --filter doctype=tutorial
```

Expected: a `ValueError` naming `doctype` and listing the indexed fields.

- [ ] **Step 3: Re-run the unfiltered baseline**

The corpus was reindexed, so the old `dense-sentence` row is no longer strictly comparable even though nothing about chunking changed. Establish the post-reindex baseline:

```bash
uv run python scripts/benchmark.py --label "dense-sentence-doctype" --collection chunks_sentence --strategy sentence --compare "dense-sentence"
```

Expected: deltas of `0.000` on every metric. **A non-zero delta here means Task 2 changed the chunking, not just the payload — stop and find out why before running the oracle.** Record the numbers either way.

- [ ] **Step 4: Run the oracle**

```bash
uv run python scripts/benchmark.py --label "dense-sentence-oracle-filter" --collection chunks_sentence --strategy sentence --oracle-filter --compare "dense-sentence-doctype"
```

- [ ] **Step 5: Write down what actually happened**

Capture, from the two runs' output:

- aggregate Recall@5, Recall@10, MRR, NDCG@5 for both labels, and the delta
- the `doc_type` table from the baseline run (n per bucket, Recall@5 per bucket)
- p50/p95 latency for both — a filtered query on an indexed keyword field should be no slower, and if it is, that is a finding
- the abstention rate on both, since filtering changes the top score on the 7 unanswerable questions

**Expected shape of the result, stated in advance so the run cannot be rationalised afterwards:** the oracle should lift the 16 single-facet questions and leave the 22 multi-facet ones flat or worse, netting out to a small aggregate change in either direction. If the aggregate lift is large, check that the oracle is not accidentally filtering the multi-facet questions to a single facet — `question_doc_type` returns `None` for those and the lambda must skip them.

- [ ] **Step 6: Commit the history**

```bash
git add data/eval/results.jsonl
git commit -m "chore(evaluation): record the doc_type baseline and oracle runs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

### Task 7: documentation

**Files:**
- Modify: `README.md` (French — current state, Phase 3 checkbox, commands, results table)
- Modify: `docs/roadmap.md` (current state table, step map, plans table)

**Interfaces:**
- Consumes: the numbers from Task 6. **Do not start this task before Task 6 has produced them** — the roadmap's rule 1 is that an em dash means "not measured", and inventing a plausible number here is the one failure this whole step exists to prevent.

- [ ] **Step 1: README — Phase 3 and the current state**

- Tick `- [ ] **Phase 3 — Métadonnées**` at `README.md:867`.
- Update the "Étape 12 terminée" paragraph at `README.md:37` to "Étape 13 terminée" and add a sentence on the facet and the oracle number.
- `README.md:140` currently says the payload indexes exist "so step 13's filtering is a query-time change". Update it to past tense: the indexes were there, step 13 used them, and `doc_type` joined them.
- `README.md:89` lists the `Chunk` fields — add `doc_type` and note it is derived from the corpus layout.
- Add the new commands to whichever section documents `scripts/search.py` and `scripts/ask.py`:

```bash
uv run python scripts/search.py "question" --filter doc_type=tutorial
uv run python scripts/ask.py "question" --filter doc_type=tutorial,advanced
uv run python scripts/benchmark.py --label "oracle" --oracle-filter
```

- Add the two Task 6 rows to the results table.

- [ ] **Step 2: README — the honest finding**

Write the finding in one short paragraph, in French, next to the results table. It must say: 16 of 38 answerable questions have single-facet ground truth; the oracle is the ceiling and it is `<measured delta>`; a router that guesses is wrong on the 22 multi-facet questions by construction; therefore step 18 should/should not build one. Fill in the actual verdict from Task 6, not this sentence's placeholder.

- [ ] **Step 3: roadmap.md — current state**

- Change the bold line under "Current state" to "Phase 0 and steps 02-13 are done... Step 14 (BM25) is next."
- Add a row to the shipped table:

```markdown
| Metadata filtering | `app/ingestion/loader.py`, `app/retrieval/search.py` — `doc_type` derived from the corpus layout, indexed, filterable through a generic `filters=` mapping; per-facet recall and an oracle run recording the ceiling on facet routing at <measured> |
```

- In the "Four things later steps own" list, add or amend an entry recording the oracle ceiling as the input step 18 needs.
- Step map: mark row `13` **Done** with the number.
- Plans table: add the row `| 13 Metadata filtering | [2026-09-12-step-13-metadata.md](superpowers/plans/2026-09-12-step-13-metadata.md) |`, and change "Steps 13-30 are deliberately unplanned" to "Steps 14-30".

- [ ] **Step 4: Verify the docs against reality**

Run every command block you added to the README, exactly as written. A README command that does not run is worse than no README command.

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/roadmap.md
git commit -m "docs: publish the doc_type facet and the routing ceiling

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest"
```

---

## Out of scope, and where it goes instead

- **Query-side facet inference.** Step 18 (query rewriting), and only if Task 6's ceiling says it is worth anything.
- **A second corpus source.** It changes the retrieval pool, so every number from steps 11-13 becomes non-comparable and the eval set needs new questions. Its own step, after the retrieval stack has stopped moving.
- **`language` filtering.** One value in the corpus, so there is nothing to measure. The field stays indexed and unused.
- **`section` filtering.** Free-text headings, hundreds of distinct values, no eval labels at that granularity. `EvalQuestion.relevant_sections` exists and is empty; step 17's reranker is the thing that operates below document granularity.
