# Cross-Encoder Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieve a deep candidate pool, rescore it with a cross-encoder, keep the best five, and measure whether that beats the 0.776 Recall@5 default.

**Architecture:** A new `app/retrieval/rerank.py` holds a `RERANKERS` registry of *scorers* — each backend only returns `(candidate index, score)` pairs, and the shared `rerank()` does all sorting, tie-breaking and `ScoredChunk` rebuilding once. `search()` gains a `rerank=` parameter orthogonal to `mode`: when set, it calls the retriever at `rerank_candidates` depth and pipes the result through `rerank()` down to `top_k`. Two backends: FlashRank (local ONNX) and Cohere Rerank, selectable one at a time.

**Tech Stack:** Python 3.12, `uv`, pydantic / pydantic-settings, Qdrant, pytest, ruff, mypy strict. New: `flashrank` (pulls `onnxruntime`, `tokenizers`, `numpy`), `cohere`.

**Spec:** [`docs/superpowers/specs/2026-09-13-reranking-design.md`](../specs/2026-09-13-reranking-design.md)

## Global Constraints

- **Measure, then improve.** No retrieval change lands without a number next to it. An em dash in the README results table means "not measured", never "good enough".
- **Dependencies arrive with the step that uses them.** `flashrank` and `cohere` are added in this step, with `uv lock` and the lockfile committed in the same change.
- **TDD, always.** Write the failing test, watch it fail, then the minimal code. Never the reverse.
- **Quality gates before every commit:** `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`
- **Unit tests run with no network, no API key and no model download.** Backends are injectable; the one test that touches a real model is behind a marker and skipped by default.
- `line-length = 100`, ruff `select = ["B", "E", "F", "I", "SIM", "UP"]`, mypy `strict = true` on `app`.
- **The evaluation set is frozen.** `data/eval/questions.jsonl` is not edited by this step.
- **One backend at a time.** No ensembling, no score fusion between rerankers.
- Every benchmark row is appended to `data/eval/results.jsonl` and committed.
- Comments explain *why*, not *what* — match the density and voice of `app/retrieval/search.py`.

---

### Task 1: Measure the candidate-pool ceiling

No reranker code. A reranker cannot exceed the recall of the pool it reorders, and the deepest pool ever measured here is N=10. This task produces the number the rest of the plan is judged against, and it can end the step.

**Files:**
- Modify: `scripts/benchmark.py:39` (the `KS` constant)
- Modify: `data/eval/results.jsonl` (three appended rows)
- Modify: `docs/roadmap.md` (the recorded headroom)

**Interfaces:**
- Consumes: nothing.
- Produces: three rows in `data/eval/results.jsonl` labelled `dense-d30-ceiling`, `hybrid-d30-ceiling`, `hybrid-d50-ceiling`, each carrying `aggregate["recall@20"]` and `aggregate["recall@30"]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation_benchmark.py`, using the `question()` and `scored()` factories already defined at the top of that file. This pins that a deep `ks` flows through to `aggregate` without disturbing the existing keys — a relevant document sitting at rank 25 must count for Recall@30 and not for Recall@10.

```python
def test_deep_ks_separate_a_rank_25_hit_from_a_rank_5_one() -> None:
    """Step 17's whole premise: the pool holds documents the top five miss."""
    only = question("a")  # relevant_document_ids == ["doc-a"]
    buried = scored(*[f"noise-{i}" for i in range(24)], "doc-a", *[f"tail-{i}" for i in range(5)])

    result = run_benchmark([only], lambda _: buried, ks=(1, 3, 5, 10, 20, 30), label="deep")

    assert result.ks == [1, 3, 5, 10, 20, 30]
    assert result.aggregate["recall@30"] == pytest.approx(1.0)
    assert result.aggregate["recall@20"] == pytest.approx(0.0)
    assert result.aggregate["recall@5"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluation_benchmark.py::test_deep_ks_separate_a_rank_25_hit_from_a_rank_5_one -v`

Expected: it may well PASS immediately — `run_benchmark` takes `ks` as a parameter and already sorts and filters it, so nothing in `app/` needs to change. That is the right outcome and not a reason to skip the test: it is what lets you trust the `recall@30` column in the rows about to be written. If it fails, fix `run_benchmark` before going further.

- [ ] **Step 3: Widen `KS` in the benchmark script**

`scripts/benchmark.py`, replacing the existing `KS = (1, 3, 5, 10)`:

```python
# 20 and 30 are step 17's ceiling: a reranker cannot exceed the recall of the
# pool it reorders, so the pool's recall at its own depth is the target to beat.
# --top-k defaults to max(KS) and run_benchmark filters ks to those <= top_k, so
# every existing invocation is unchanged and no past row is invalidated.
KS = (1, 3, 5, 10, 20, 30)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_benchmark.py -v`

Expected: PASS, all of them. Then the full gates: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`

- [ ] **Step 5: Start Qdrant and confirm the corpus is indexed**

```bash
docker compose up -d qdrant
uv run python scripts/index_corpus.py --strategy sentence
```

Expected: 1 484 points in the `chunks` collection. If the collection is already populated the re-run is idempotent and takes ~3.5 s.

- [ ] **Step 6: Run the three ceiling benchmarks**

```bash
uv run python scripts/benchmark.py --label "dense-d30-ceiling"  --mode dense                             --top-k 30
uv run python scripts/benchmark.py --label "hybrid-d30-ceiling" --mode hybrid --rrf-k 60 --candidates 30 --top-k 30
uv run python scripts/benchmark.py --label "hybrid-d50-ceiling" --mode hybrid --rrf-k 60 --candidates 50 --top-k 30
```

`hybrid-k60-d20` — the step 16 winner — is deliberately absent: `_hybrid` raises when `candidates < top_k`, and a 20-deep per-branch fusion cannot be queried at `--top-k 30`. The reranker is measured against the pools above, never against the d20 row.

Every run is free: the embedding cache serves the query vectors and BM25 makes no API call.

- [ ] **Step 7: Compute the headroom and take the go/no-go**

For each pool, `headroom = Recall@30 − Recall@5`. Write the three numbers down.

```bash
uv run python scripts/benchmark.py --summary "*-ceiling"
```

**If the best pool's headroom is below 0.03, stop here.** Record the finding in `docs/roadmap.md` and the README results table in the same terms step 13 used for its +0.000 routing ceiling, skip tasks 2-7, and go straight to task 8 to write it up. No dependency is added. That is a completed step under rule 1, not an abandoned one.

Otherwise, note which pool (`dense` or `hybrid` at which depth) has the most headroom — task 7 measures both, but this tells you what to expect.

- [ ] **Step 8: Commit**

```bash
git add scripts/benchmark.py tests/test_evaluation_benchmark.py data/eval/results.jsonl docs/roadmap.md
git commit -m "$(cat <<'EOF'
chore(evaluation): measure the reranker's ceiling at depth 20 and 30

A reranker reorders; it cannot retrieve. Its best possible Recall@5 is the
Recall@N of the pool handed to it, and this project had never run a pool
deeper than 10.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
```

---

### Task 2: `rerank()` and the `RERANKERS` registry

The shared half of the feature: validation, dispatch, ordering, tie-breaking and score semantics — everything both backends need and neither should duplicate. No real backend yet, so this task has no new dependency and its tests need nothing but a fake.

**Files:**
- Create: `app/retrieval/rerank.py`
- Test: `tests/test_retrieval_rerank.py`

**Interfaces:**
- Consumes: `app.models.chunks.ScoredChunk` (fields `chunk`, `score`, `rank`, `rerank_score`), `app.core.config.Settings`.
- Produces:
  - `Scorer = Callable[[str, Sequence[ScoredChunk], int, Settings], list[tuple[int, float]]]` — a backend receives `(query, candidates, top_k, settings)` and returns `(index into candidates, relevance score)` pairs, unsorted, possibly fewer than `len(candidates)`.
  - `RERANKERS: dict[str, Scorer]` — the registry. Task 3 adds `"flashrank"`, task 6 adds `"cohere"`.
  - `rerank(query: str, candidates: Sequence[ScoredChunk], *, model: str, top_k: int, settings: Settings | None = None) -> list[ScoredChunk]`
  - `warm_up(model: str, settings: Settings) -> None` — builds whatever the named backend loads lazily. A no-op for backends with nothing to load.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval_rerank.py`:

```python
from collections.abc import Sequence

import pytest

from app.core.config import Settings
from app.models.chunks import Chunk, ScoredChunk
from app.retrieval.rerank import RERANKERS, Scorer, rerank, warm_up

SETTINGS = Settings(openai_api_key=None)


def scored(index: int, score: float) -> ScoredChunk:
    chunk = Chunk(
        document_id=f"fastapi:tutorial/page-{index}",
        source="fastapi",
        title=f"Page {index}",
        doc_type="tutorial",
        chunk_index=0,
        text=f"chunk body number {index}",
        char_start=0,
        char_end=20,
    )
    return ScoredChunk(chunk=chunk, score=score, rank=index + 1)


def pool(size: int = 4) -> list[ScoredChunk]:
    """A retriever's output: descending scores, ranks from 1."""
    return [scored(index, 0.9 - 0.1 * index) for index in range(size)]


def register(name: str, scorer: Scorer) -> None:
    """Registered per test and removed by the fixture below."""
    RERANKERS[name] = scorer


@pytest.fixture(autouse=True)
def _restore_registry() -> "Iterator[None]":
    original = dict(RERANKERS)
    yield
    RERANKERS.clear()
    RERANKERS.update(original)


def reverse_scorer(
    query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
) -> list[tuple[int, float]]:
    """Scores the last candidate highest — a no-op reranker cannot fake this."""
    return [(index, float(index)) for index in range(len(candidates))]


# --- the reordering actually happens ---------------------------------------


def test_reordering_actually_happens() -> None:
    register("reverse", reverse_scorer)
    results = rerank("q", pool(4), model="reverse", top_k=4)
    assert [r.chunk.document_id for r in results] == [
        "fastapi:tutorial/page-3",
        "fastapi:tutorial/page-2",
        "fastapi:tutorial/page-1",
        "fastapi:tutorial/page-0",
    ]


def test_rank_is_reassigned_from_one() -> None:
    register("reverse", reverse_scorer)
    assert [r.rank for r in rerank("q", pool(4), model="reverse", top_k=4)] == [1, 2, 3, 4]


def test_the_retriever_score_survives_and_the_rerank_score_is_recorded() -> None:
    """Both numbers stay answerable: 'the retriever never had it' and 'the
    reranker buried it' are different failures."""
    register("reverse", reverse_scorer)
    [best] = rerank("q", pool(4), model="reverse", top_k=1)
    assert best.chunk.document_id == "fastapi:tutorial/page-3"
    assert best.score == pytest.approx(0.6)  # what dense retrieval thought
    assert best.rerank_score == pytest.approx(3.0)


def test_top_k_trims_after_reordering_not_before() -> None:
    register("reverse", reverse_scorer)
    results = rerank("q", pool(10), model="reverse", top_k=3)
    assert [r.chunk.document_id for r in results] == [
        "fastapi:tutorial/page-9",
        "fastapi:tutorial/page-8",
        "fastapi:tutorial/page-7",
    ]


# --- errors and edges ------------------------------------------------------


def test_an_unknown_reranker_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="unknown reranker"):
        rerank("q", pool(), model="nope", top_k=3)


def test_top_k_below_one_is_rejected() -> None:
    register("reverse", reverse_scorer)
    with pytest.raises(ValueError, match="top_k"):
        rerank("q", pool(), model="reverse", top_k=0)


def test_an_empty_pool_returns_empty_without_calling_a_backend() -> None:
    """Both backends charge for being asked to rank nothing — in latency or in
    money."""
    calls: list[str] = []

    def recording(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        calls.append(query)
        return []

    register("recording", recording)
    assert rerank("q", [], model="recording", top_k=3) == []
    assert calls == []


def test_ties_break_on_chunk_id_so_two_runs_agree() -> None:
    def flat(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        return [(index, 0.5) for index in range(len(candidates))]

    register("flat", flat)
    ids = [r.chunk.chunk_id for r in rerank("q", pool(4), model="flat", top_k=4)]
    assert ids == sorted(ids)


def test_a_backend_may_return_fewer_pairs_than_it_was_given() -> None:
    """Cohere's top_n returns only the best n; the result is those n, ranked."""

    def partial(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        return [(2, 0.9), (0, 0.4)]

    register("partial", partial)
    results = rerank("q", pool(4), model="partial", top_k=4)
    assert [r.chunk.document_id for r in results] == [
        "fastapi:tutorial/page-2",
        "fastapi:tutorial/page-0",
    ]


def test_the_scorer_is_told_how_many_results_are_wanted() -> None:
    """Cohere bills per search and returns top_n; passing it through is the
    difference between ranking 30 documents and paying to return 30."""
    seen: list[int] = []

    def recording(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        seen.append(top_k)
        return [(0, 1.0)]

    register("recording", recording)
    rerank("q", pool(4), model="recording", top_k=2)
    assert seen == [2]


def test_warm_up_is_a_no_op_for_an_unknown_backend() -> None:
    """It runs before a timed loop; a crash there would fail a benchmark that
    would otherwise have worked."""
    warm_up("nope", SETTINGS)
```

Add `from collections.abc import Iterator, Sequence` at the top — the fixture's return annotation needs `Iterator`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_rerank.py -v`

Expected: FAIL at collection — `ModuleNotFoundError: No module named 'app.retrieval.rerank'`.

- [ ] **Step 3: Write `app/retrieval/rerank.py`**

```python
"""Reorder a shortlist with a cross-encoder. Retrieval finds; this ranks.

A bi-encoder embeds the query and the chunk separately and compares two vectors
that never met — which is what makes it fast enough to run over 1 484 chunks. A
cross-encoder reads the pair together and scores it jointly, which is far more
accurate and far too slow to run over a corpus. So it runs over the 30 chunks a
retriever already shortlisted. That two-stage shape is the whole idea, and it is
why ``rerank_candidates`` matters more here than any model choice.

A backend is a ``Scorer``: it returns ``(index into candidates, score)`` pairs
and nothing else. Ordering, tie-breaking, trimming and rebuilding
``ScoredChunk``s happen once, here, for every backend — the only thing that
genuinely differs between a local ONNX model and a hosted API is the scoring.
"""

from collections.abc import Callable, Sequence

from app.core.config import Settings, get_settings
from app.models.chunks import ScoredChunk

Scorer = Callable[[str, Sequence[ScoredChunk], int, Settings], list[tuple[int, float]]]

# Mirrors search.RETRIEVERS and chunk.STRATEGIES: the registry is how this
# project compares N variants and promotes a winner. One at a time, by decision:
# there is no ensembling here and no fusion of two rerankers' scores.
RERANKERS: dict[str, Scorer] = {}


def rerank(
    query: str,
    candidates: Sequence[ScoredChunk],
    *,
    model: str,
    top_k: int,
    settings: Settings | None = None,
) -> list[ScoredChunk]:
    """The ``top_k`` candidates a cross-encoder thinks best answer ``query``.

    ``score`` is left exactly as the retriever wrote it and ``rerank_score``
    carries the new number, so "the retriever never had it" and "the reranker
    buried it" stay different, answerable failures. ``rerank_score is not None``
    is what says which of the two produced the ranking.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if model not in RERANKERS:
        raise ValueError(f"unknown reranker {model!r}; have {sorted(RERANKERS)}")
    if not candidates:
        # Both backends charge for being asked to rank nothing: one in a model
        # invocation, the other in a billed API call.
        return []

    settings = settings or get_settings()
    scored = RERANKERS[model](query, candidates, top_k, settings)
    # chunk_id breaks ties so two runs of one commit agree, as rrf() already does.
    ordered = sorted(scored, key=lambda pair: (-pair[1], candidates[pair[0]].chunk.chunk_id))
    return [
        # model_copy rather than a fresh ScoredChunk: the chunk and the
        # retriever's score come along untouched, which is the point.
        candidates[index].model_copy(update={"rerank_score": score, "rank": rank})
        for rank, (index, score) in enumerate(ordered[:top_k], start=1)
    ]


def warm_up(model: str, settings: Settings) -> None:
    """Build whatever ``model`` loads lazily, before a timed loop starts.

    A local model costs 1-3 s to load; charged to the first question of a
    benchmark it would make the ``p50 ms`` column stop meaning per-query
    retrieval latency, which is the only thing it is used for. Backends with
    nothing to preload — anything that calls out over the network per query —
    are a no-op, and so is an unknown name: this runs before the work, and
    failing here would fail a run that would otherwise have succeeded.
    """
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_rerank.py -v`

Expected: PASS, 11 tests. Then: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/rerank.py tests/test_retrieval_rerank.py
git commit -m "$(cat <<'EOF'
feat(retrieval): add the reranker registry and shared ordering

Backends score; rerank() orders. A Scorer returns (index, score) pairs and
nothing else, so tie-breaking, trimming and the score/rerank_score split are
written once rather than once per backend.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
```

---

### Task 3: The FlashRank backend

**Files:**
- Modify: `app/retrieval/rerank.py`
- Modify: `app/core/config.py` (one setting)
- Modify: `pyproject.toml`, `uv.lock`
- Test: `tests/test_retrieval_rerank.py`

**Interfaces:**
- Consumes: `Scorer`, `RERANKERS`, `warm_up` from task 2; `Settings` from `app.core.config`.
- Produces: `RERANKERS["flashrank"]`; `Settings.flashrank_model: str = "ms-marco-MiniLM-L-12-v2"`; `FLASHRANK_CACHE: Path`; `warm_up("flashrank", settings)` loads the ONNX session.

- [ ] **Step 1: Add the dependency**

```bash
uv add "flashrank>=0.2,<1"
```

This pulls `onnxruntime`, `tokenizers` and `numpy`. Do **not** install the `[listwise]` extra — it pulls a 7B LLM reranker this step does not use. If resolution fails on the bound, widen it to `"flashrank"` and let `uv` pick, then pin the caret manually to match the repo's `>=X,<Y` style.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_retrieval_rerank.py`:

```python
def test_flashrank_is_registered() -> None:
    assert "flashrank" in RERANKERS


def test_flashrank_maps_scores_back_by_list_position() -> None:
    """The passage id is the candidate's index, not its chunk_id: a string
    round-trip through a third-party library is one more place to lose a
    mapping, and the index is already unique and already an int."""
    from app.retrieval import rerank as module

    captured: dict[str, object] = {}

    class FakeRanker:
        def rerank(self, request: object) -> list[dict[str, object]]:
            captured["passages"] = request.passages  # type: ignore[attr-defined]
            captured["query"] = request.query  # type: ignore[attr-defined]
            # Deliberately out of order and incomplete, like a real one.
            return [{"id": 2, "text": "x", "score": 0.91}, {"id": 0, "text": "y", "score": 0.12}]

    module._ranker.cache_clear()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(module, "_ranker", lambda name: FakeRanker())
    try:
        pairs = module._flashrank("what is a dependency", pool(4), 2, SETTINGS)
    finally:
        monkey.undo()

    assert pairs == [(2, pytest.approx(0.91)), (0, pytest.approx(0.12))]
    assert captured["query"] == "what is a dependency"
    assert [p["id"] for p in captured["passages"]] == [0, 1, 2, 3]  # type: ignore[index]
    assert captured["passages"][0]["text"] == "chunk body number 0"  # type: ignore[index]
```

And the real-model integration test. Mirror the `requires_qdrant` pattern exactly — `tests/test_retrieval_search.py:211` and `tests/test_retrieval_store.py:76` both do `requires_qdrant = pytest.mark.requires_qdrant` and then decorate. **These markers are declarative, not deselecting:** `addopts = "-ra"` does not filter them, so marked tests run in a normal `uv run pytest`. That is why the warm-up in step 6 happens before you run the suite, and why the marker's job is to *name* the prerequisite rather than to hide the test.

```python
requires_model = pytest.mark.requires_model


@requires_model
def test_the_real_model_prefers_the_relevant_chunk() -> None:
    """The only test that loads ONNX weights, and the only one that proves the
    cross-encoder reads the pair rather than the passage: the relevant chunk is
    given the *worse* retriever score, so a reranker that passes its input
    through cannot pass this."""
    relevant = scored(0, 0.1)
    irrelevant = scored(1, 0.9)
    candidates = [
        relevant.model_copy(
            update={"chunk": relevant.chunk.model_copy(update={"text": "Depends() injects a dependency into a path operation."})}
        ),
        irrelevant.model_copy(
            update={"chunk": irrelevant.chunk.model_copy(update={"text": "Run the server with uvicorn on port 8000."})}
        ),
    ]
    [best] = rerank("how do dependencies work", candidates, model="flashrank", top_k=1)
    assert "Depends()" in best.chunk.text
```

Register the marker in `pyproject.toml` beside the existing one:

```toml
markers = [
    "requires_qdrant: needs a running Qdrant (docker compose up -d qdrant)",
    "requires_model: downloads and loads the FlashRank ONNX weights",
]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_rerank.py -v`

Expected: FAIL — `AttributeError: module 'app.retrieval.rerank' has no attribute '_ranker'`, and `test_flashrank_is_registered` fails on the empty registry.

- [ ] **Step 4: Implement the backend**

Add the setting to `app/core/config.py`, after `rrf_k`:

```python
    # ~34 MB. The ~4 MB nano default trades away exactly the ranking precision
    # step 17 is measuring, which would make the measurement meaningless.
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
```

Add to `app/retrieval/rerank.py`:

```python
from functools import lru_cache
from pathlib import Path
from typing import Any

# Beside the embedding cache rather than in /tmp: on Windows the library's
# default cache_dir is neither writable nor stable across runs, and re-downloading
# 34 MB per process is a cost with no upside.
FLASHRANK_CACHE = Path("data/processed/flashrank")


@lru_cache(maxsize=2)
def _ranker(model_name: str) -> Any:
    """The ONNX session, built once per process and per model name.

    Imported here rather than at module scope so that importing this module
    costs nothing: onnxruntime is tens of megabytes of shared library, and every
    test, every `--mode dense` run and every script that never reranks would pay
    for it at import time otherwise.
    """
    from flashrank import Ranker

    FLASHRANK_CACHE.mkdir(parents=True, exist_ok=True)
    return Ranker(model_name=model_name, cache_dir=str(FLASHRANK_CACHE))


def _flashrank(
    query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
) -> list[tuple[int, float]]:
    """Score every candidate locally. ``top_k`` is ignored: the model scores the
    whole shortlist either way, and trimming is ``rerank()``'s job."""
    from flashrank import RerankRequest

    passages = [
        {"id": index, "text": scored.chunk.text} for index, scored in enumerate(candidates)
    ]
    results = _ranker(settings.flashrank_model).rerank(RerankRequest(query=query, passages=passages))
    return [(int(result["id"]), float(result["score"])) for result in results]


RERANKERS["flashrank"] = _flashrank
```

And fill in `warm_up`, replacing its `return None`:

```python
    if model == "flashrank":
        _ranker(settings.flashrank_model)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_rerank.py -v -m "not requires_model"`

Expected: PASS. The marked test is excluded here only because the weights are not downloaded yet; step 6 downloads them and runs it.

- [ ] **Step 6: Warm up the model once, then run the integration test**

```bash
uv run python -c "from app.core.config import get_settings; from app.retrieval.rerank import warm_up; warm_up('flashrank', get_settings()); print('ready')"
uv run pytest tests/test_retrieval_rerank.py -m requires_model -v
```

Expected: a ~34 MB download into `data/processed/flashrank/` on the first call, then PASS. Add `data/processed/flashrank/` to `.gitignore` if `data/processed/` is not already ignored — check first.

- [ ] **Step 7: Full gates and commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`

```bash
git add app/retrieval/rerank.py app/core/config.py tests/test_retrieval_rerank.py pyproject.toml uv.lock .gitignore
git commit -m "$(cat <<'EOF'
feat(retrieval): add the FlashRank cross-encoder backend

ms-marco-MiniLM-L-12-v2 through onnxruntime rather than sentence-transformers
through torch: the same class of model for a fraction of the install, and the
thing being learned is identical either way.

onnxruntime is imported inside the scorer so that every dense run, every script
that never reranks and every test keeps paying nothing for it at import time.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
```

---

### Task 4: Wire `rerank=` into `search()`

**Files:**
- Modify: `app/retrieval/search.py` (signature, body, and the stale comment at line 217)
- Modify: `app/core/config.py` (two settings)
- Test: `tests/test_retrieval_search.py`

**Interfaces:**
- Consumes: `rerank` and `warm_up` from `app.retrieval.rerank`.
- Produces: `search(..., rerank: str | None = None, rerank_candidates: int | None = None, ...)`. `rerank=None` reads `Settings.rerank_model`; `rerank=""` forces it off. `Settings.rerank_model: str = ""`, `Settings.rerank_candidates: int = 30`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_retrieval_search.py`. Import the registry and register a fake in a fixture, exactly as `tests/test_retrieval_rerank.py` does:

```python
from collections.abc import Iterator

from app.retrieval.rerank import RERANKERS


def reverse_scorer(
    query: str, candidates: "Sequence[ScoredChunk]", top_k: int, settings: Settings
) -> list[tuple[int, float]]:
    return [(index, float(index)) for index in range(len(candidates))]


@pytest.fixture
def fake_reranker() -> Iterator[None]:
    RERANKERS["reverse"] = reverse_scorer
    yield
    del RERANKERS["reverse"]


def test_no_reranker_leaves_search_exactly_as_it_was() -> None:
    """The four existing callers stay untouched only if this holds."""
    assert [s.rank for s in run(top_k=3)] == [1, 2, 3]
    assert [s.rerank_score for s in run(top_k=3)] == [None, None, None]


def test_a_reranker_retrieves_deep_and_returns_top_k(fake_reranker: None) -> None:
    client = FakeClient()
    results = run(client=client, top_k=3, rerank="reverse", rerank_candidates=10)
    assert client.calls[0]["limit"] == 10  # the pool, not the answer
    assert len(results) == 3
    assert [s.rank for s in results] == [1, 2, 3]
    assert results[0].rerank_score == pytest.approx(9.0)


def test_the_reranker_defaults_to_the_setting(fake_reranker: None) -> None:
    client = FakeClient()
    results = run(
        client=client,
        settings=Settings(rerank_model="reverse", rerank_candidates=8, openai_api_key=None),
        top_k=2,
    )
    assert client.calls[0]["limit"] == 8
    assert results[0].rerank_score is not None


def test_an_empty_rerank_argument_forces_it_off(fake_reranker: None) -> None:
    """`--rerank ""` must be able to override a configured RERANK_MODEL, or a
    dense baseline becomes unrunnable once the default flips."""
    client = FakeClient()
    results = run(
        client=client,
        settings=Settings(rerank_model="reverse", rerank_candidates=8, openai_api_key=None),
        top_k=2,
        rerank="",
    )
    assert client.calls[0]["limit"] == 2
    assert results[0].rerank_score is None


def test_rerank_candidates_below_top_k_is_rejected(fake_reranker: None) -> None:
    with pytest.raises(ValueError, match="rerank_candidates"):
        run(top_k=10, rerank="reverse", rerank_candidates=3)


def test_hybrid_candidates_below_the_rerank_pool_is_rejected(fake_reranker: None) -> None:
    """The fused list is what the cross-encoder sees; two 5-deep branches cannot
    fill a 30-deep pool, and a short pool reads downstream as 'reranking did not
    help'."""
    with pytest.raises(ValueError, match="candidates"):
        run(
            "error",
            mode="hybrid",
            index=bm25_index("an error occurred"),
            top_k=5,
            candidates=5,
            rerank="reverse",
            rerank_candidates=30,
        )


def test_an_unknown_reranker_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reranker"):
        run(top_k=3, rerank="nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_search.py -v -k rerank`

Expected: FAIL — `TypeError: search() got an unexpected keyword argument 'rerank'`.

- [ ] **Step 3: Add the settings**

`app/core/config.py`, after `flashrank_model`:

```python
    # Empty means off: unchanged behaviour until step 17's measurement earns the
    # change. flashrank | cohere — compared at step 17, table in the README.
    rerank_model: str = ""
    # How deep the retrieved pool goes into the cross-encoder. This is the
    # parameter that bounds what reranking can do: a reranker reorders, it
    # cannot retrieve, so the pool's own recall is its ceiling.
    rerank_candidates: int = 30
```

- [ ] **Step 4: Wire it into `search()`**

In `app/retrieval/search.py`, add the import:

```python
from app.retrieval.rerank import rerank as apply_reranker
```

(aliased because `rerank` is also the parameter name — the parameter reads better than `rerank_model` at every call site, and shadowing an import silently is worse than one alias.)

Add both parameters to the signature after `rrf_k`:

```python
    rerank: str | None = None,
    rerank_candidates: int | None = None,
```

Extend the docstring with:

```
    ``rerank`` names a cross-encoder from ``rerank.RERANKERS`` and is orthogonal
    to ``mode``: it reorders whatever the chosen retriever produced. ``None``
    reads ``RERANK_MODEL``, and the empty string forces it off, which is how a
    dense baseline stays runnable once the default flips. ``rerank_candidates``
    is how deep the retrieved pool goes into the cross-encoder — in hybrid mode
    that is the depth of the *fused* list, which ``candidates`` (per-branch,
    pre-fusion) must be at least as large as.
```

Replace the end of the body — everything from `return RETRIEVERS[mode](` onwards:

```python
    model = settings.rerank_model if rerank is None else rerank
    depth = top_k
    if model:
        depth = rerank_candidates or settings.rerank_candidates
        if depth < top_k:
            # A pool shallower than the answer is a reranker with nothing to
            # choose between, and it reads downstream as "reranking did not help".
            raise ValueError(f"rerank_candidates ({depth}) must be at least top_k ({top_k})")

    results = RETRIEVERS[mode](
        query,
        top_k=depth,
        candidates=candidates or settings.retrieval_candidates,
        rrf_k=rrf_k or settings.rrf_k,
        filters=filters,
        collection=collection,
        client=client,
        embedder=embedder,
        index=index,
    )
    if not model:
        return results
    return apply_reranker(query, results, model=model, top_k=top_k, settings=settings)
```

Note there is no new check for `candidates >= rerank_candidates` in hybrid mode: passing `top_k=depth` into `_hybrid` makes its existing `candidates < top_k` guard do exactly that job. The test above pins it.

- [ ] **Step 5: Correct the stale comment**

`app/retrieval/search.py:217` currently reads `# and promotes a winner. Step 17's reranker registers a key here.` Replace that line with:

```python
# and promotes a winner. Reranking is NOT a key here: it composes with every
# mode, so it is a `rerank=` parameter and its own registry in rerank.py.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_search.py -v`

Expected: PASS, all of them — the pre-existing tests included. Then: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`

- [ ] **Step 7: Commit**

```bash
git add app/retrieval/search.py app/core/config.py tests/test_retrieval_search.py
git commit -m "$(cat <<'EOF'
feat(retrieval): rerank behind the search() seam, orthogonal to mode

search.py predicted step 17 would register a fourth RETRIEVERS key. It should
not: reranking composes with dense, lexical and hybrid alike, and as a mode key
two backends across two pools would be four keys and six once step 18 wants a
reranked multi-query. It is a parameter and its own registry; the comment is
corrected.

rerank=None reads RERANK_MODEL, rerank="" forces it off, so a dense baseline
stays runnable once the default flips.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
```

---

### Task 5: Reach the flag from every script and from the answer pipeline

**Files:**
- Modify: `app/generation/answer.py:31-64`
- Modify: `scripts/search.py`, `scripts/ask.py`, `scripts/benchmark.py`
- Modify: `app/evaluation/benchmark.py:75-113` (`SUMMARY_COLUMNS`, `summarise`)
- Modify: `.env.example`
- Test: `tests/test_generation_answer.py`

**Interfaces:**
- Consumes: `search(..., rerank=, rerank_candidates=)` from task 4; `warm_up` from task 2.
- Produces: `answer_question(..., rerank: str | None = None, rerank_candidates: int | None = None, ...)`; `--rerank` and `--rerank-candidates` on all three scripts; a `rerank` column in `SUMMARY_COLUMNS`; `rerank` and `rerank_candidates` in each benchmark row's `config`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_generation_answer.py`, using the `FakeRetriever`, `FakeLLM` and `SETTINGS` already defined at the top of that file (`FakeRetriever.calls` records `{"query": ..., **kwargs}` per call):

```python
def test_rerank_is_threaded_to_the_retriever() -> None:
    """The measured winner has to reach the answer, not only the benchmark."""
    retriever = FakeRetriever()
    answer_question(
        "how do dependencies work",
        rerank="flashrank",
        rerank_candidates=30,
        retriever=retriever,
        llm=FakeLLM(),
        settings=SETTINGS,
    )
    assert retriever.calls[0]["rerank"] == "flashrank"
    assert retriever.calls[0]["rerank_candidates"] == 30


def test_no_rerank_still_reaches_the_retriever_as_none() -> None:
    """search() reads RERANK_MODEL when it gets None, so answer_question must
    pass None through rather than dropping the keyword — otherwise a configured
    default would apply to the benchmark and not to ask.py."""
    retriever = FakeRetriever()
    answer_question("how do dependencies work", retriever=retriever, llm=FakeLLM(), settings=SETTINGS)
    assert retriever.calls[0]["rerank"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_generation_answer.py -v -k rerank`

Expected: FAIL, both — `TypeError: answer_question() got an unexpected keyword argument 'rerank'`.

- [ ] **Step 3: Thread it through `answer_question`**

`app/generation/answer.py`: add two parameters after `mode`:

```python
    rerank: str | None = None,
    rerank_candidates: int | None = None,
```

Extend the docstring line about `mode`:

```
    ``mode`` selects the retriever and ``rerank`` the cross-encoder that reorders
    its output; both are threaded straight through so the measured winners of
    steps 16 and 17 reach the answer, not only the benchmark.
```

And the call:

```python
    chunks = retriever(
        question,
        top_k=top_k or settings.top_k,
        mode=mode,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        filters=filters,
        settings=settings,
    )
```

- [ ] **Step 4: Add the flags to the three scripts**

In each of `scripts/search.py`, `scripts/ask.py` and `scripts/benchmark.py`, add after the existing `--mode` argument. Import `RERANKERS` from `app.retrieval.rerank` in each:

```python
    parser.add_argument(
        "--rerank",
        choices=sorted(RERANKERS),
        help='cross-encoder that reorders the pool; default: RERANK_MODEL, "" is off',
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        help="how deep the pool goes into the cross-encoder; default: RERANK_CANDIDATES",
    )
```

`argparse` with `choices` rejects `""`, so add `""` to the choices list in each: `choices=["", *sorted(RERANKERS)]`.

Then pass both through:

- `scripts/search.py`, in the `search(...)` call: `rerank=args.rerank, rerank_candidates=args.rerank_candidates`. Also widen the printed line so a rerank score is visible when there is one:

```python
        score = f"{scored.rerank_score:.4f}*" if scored.rerank_score is not None else f"{scored.score:.4f}"
        print(f"{scored.rank}. {score}  {chunk.document_id}  [{chunk.section or '-'}]")
```

with a comment: `# The asterisk marks a rerank score — a different scale from a cosine, and comparing the two by eye is the mistake it exists to prevent.`

- `scripts/ask.py`: pass both to the `search(...)` call inside the `--show-context` branch and to `answer_question(...)`.
- `scripts/benchmark.py`: add to the `retrieve` closure's `search(...)` call, and add two keys to the `config` dict:

```python
            "rerank": args.rerank or settings.rerank_model or None,
            "rerank_candidates": args.rerank_candidates or settings.rerank_candidates,
```

- [ ] **Step 5: Warm the model outside the timed loop**

`scripts/benchmark.py`, next to the existing `if mode != "dense": default_index(...)` block:

```python
    reranker = args.rerank if args.rerank is not None else settings.rerank_model
    if reranker:
        # Same reason as the BM25 build above: a 1-3 s model load charged to
        # question 1 would make the p50 column stop meaning per-query latency.
        warm_up(reranker, settings)
```

Import `warm_up` from `app.retrieval.rerank` alongside `RERANKERS`.

- [ ] **Step 6: Add the summary column**

`app/evaluation/benchmark.py`: insert `"rerank"` into `SUMMARY_COLUMNS` immediately after `"mode"`, and the matching cell into `summarise`'s row, immediately after the `mode` cell:

```python
            # "-" rather than "": every row written before step 17 genuinely had
            # no reranker, and an empty cell reads as a missing value.
            str(row["config"].get("rerank") or "-"),
```

The candidate depth gets no column: the label carries it (`rerank-flashrank-hybrid-d30`), and step 16 already declined to widen this table with `rrf_k` for the same reason.

- [ ] **Step 7: Document the settings in `.env.example`**

Append, in French, matching the file's existing voice:

```ini
# Reranking (étape 17)
# Vide = désactivé. flashrank | cohere — comparés à l'étape 17, table dans le README.
RERANK_MODEL=
# Profondeur du vivier envoyé au cross-encoder. Un reranker réordonne, il ne
# récupère pas : le rappel du vivier est son plafond.
RERANK_CANDIDATES=30
# ~34 Mo, téléchargé une fois dans data/processed/flashrank/.
FLASHRANK_MODEL=ms-marco-MiniLM-L-12-v2
```

and move `COHERE_API_KEY=` out of "Fournisseurs optionnels des phases futures" into this block — task 6 makes it a real setting.

- [ ] **Step 8: Verify end to end**

```bash
uv run pytest -v
uv run ruff check . && uv run ruff format --check . && uv run mypy app
uv run python scripts/search.py "how do dependencies work" --rerank flashrank --rerank-candidates 20
uv run python scripts/ask.py "How do I define a dependency?" --mode hybrid --rerank flashrank
```

Expected: the search output shows asterisked scores and a different order from the same query without `--rerank`; `ask.py` returns a grounded answer with resolved citations.

- [ ] **Step 9: Commit**

```bash
git add app/generation/answer.py app/evaluation/benchmark.py scripts/ .env.example tests/test_generation_answer.py
git commit -m "$(cat <<'EOF'
feat(retrieval): reach the reranker from every script and from ask.py

A retriever the benchmark can use and scripts/ask.py cannot is the failure the
hybrid spec rejected for composing hybrid inside benchmark.py only.

The model load is warmed before the timed loop, as the BM25 build already is:
1-3 s charged to question 1 would make the p50 column stop meaning per-query
retrieval latency.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
```

---

### Task 6: The Cohere backend

The one genuinely dangerous mapping in the feature: the API answers with positions into the list you sent, not with documents. An off-by-one pairs every score with the wrong chunk and still returns a well-formed, plausible ranking that no aggregate metric would catch. Its test is written first.

**Files:**
- Modify: `app/retrieval/rerank.py`
- Modify: `app/core/config.py`
- Modify: `pyproject.toml`, `uv.lock`
- Test: `tests/test_retrieval_rerank.py`

**Interfaces:**
- Consumes: `Scorer`, `RERANKERS` from task 2.
- Produces: `RERANKERS["cohere"]`; `_cohere(query, candidates, top_k, settings, client=None)` — the optional `client` is how the test injects a fake; `Settings.cohere_api_key: str | None = None`; `COHERE_MODEL: str = "rerank-v3.5"`.

- [ ] **Step 1: Add the dependency**

```bash
uv add "cohere>=5,<6"
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_retrieval_rerank.py`:

```python
class FakeCohereResult:
    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class FakeCohereResponse:
    def __init__(self, results: list[FakeCohereResult]) -> None:
        self.results = results


class FakeCohereClient:
    def __init__(self, results: list[FakeCohereResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def rerank(self, **kwargs: object) -> FakeCohereResponse:
        self.calls.append(kwargs)
        return FakeCohereResponse(self.results)


def test_cohere_is_registered() -> None:
    assert "cohere" in RERANKERS


def test_cohere_maps_every_score_back_to_the_chunk_it_scored() -> None:
    """The API answers with positions into the list it was sent. An off-by-one
    here pairs every score with the wrong chunk and still returns a well-formed
    ranking no aggregate metric would catch."""
    from app.retrieval.rerank import _cohere

    client = FakeCohereClient([FakeCohereResult(2, 0.93), FakeCohereResult(0, 0.11)])
    pairs = _cohere("what is a dependency", pool(4), 2, SETTINGS, client=client)

    assert pairs == [(2, pytest.approx(0.93)), (0, pytest.approx(0.11))]


def test_cohere_sends_the_chunk_texts_in_candidate_order() -> None:
    from app.retrieval.rerank import _cohere

    client = FakeCohereClient([FakeCohereResult(0, 1.0)])
    _cohere("q", pool(3), 2, SETTINGS, client=client)

    assert client.calls[0]["documents"] == [
        "chunk body number 0",
        "chunk body number 1",
        "chunk body number 2",
    ]
    assert client.calls[0]["query"] == "q"


def test_cohere_asks_for_only_as_many_as_are_wanted() -> None:
    """It bills per search; returning 30 when 5 are used is paid-for waste."""
    from app.retrieval.rerank import _cohere

    client = FakeCohereClient([FakeCohereResult(0, 1.0)])
    _cohere("q", pool(30), 5, SETTINGS, client=client)
    assert client.calls[0]["top_n"] == 5


def test_cohere_without_a_key_fails_before_the_network() -> None:
    from app.retrieval.rerank import _cohere

    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        _cohere("q", pool(3), 2, Settings(cohere_api_key=None, openai_api_key=None))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_rerank.py -v -k cohere`

Expected: FAIL — `ImportError: cannot import name '_cohere'`, and `test_cohere_is_registered` fails.

- [ ] **Step 4: Promote the API key to a real setting**

`app/core/config.py`. Replace the existing comment on `model_config` — it currently says `.env` carries `COHERE_API_KEY` for later phases, and that phase is now:

```python
    # extra="ignore": .env carries keys for phases that have not landed yet, and
    # a strict settings class would crash on them.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

and add the field beside `openai_api_key`:

```python
    # Step 17's hosted reranker. Not required: the test suite and every
    # FlashRank-only run must import this module without a Cohere key present.
    cohere_api_key: str | None = None
```

- [ ] **Step 5: Implement the backend**

Add to `app/retrieval/rerank.py`:

```python
# The current general-purpose rerank model; English-only variants exist but the
# corpus is bilingual by construction — the evaluation set asks in French.
COHERE_MODEL = "rerank-v3.5"


@lru_cache(maxsize=1)
def _cohere_client(api_key: str) -> Any:
    import cohere

    return cohere.ClientV2(api_key=api_key)


def _cohere(
    query: str,
    candidates: Sequence[ScoredChunk],
    top_k: int,
    settings: Settings,
    client: Any = None,
) -> list[tuple[int, float]]:
    """Score the shortlist through Cohere's hosted reranker.

    ``client`` is injectable so the mapping below is testable without a key and
    without a billed call. That mapping is the whole risk of this function: the
    API answers with ``index`` positions into the list it was sent, not with
    documents, so getting it wrong pairs every score with the wrong chunk and
    still produces a ranking that looks entirely reasonable.
    """
    if client is None:
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY is not set; --rerank cohere needs one")
        client = _cohere_client(settings.cohere_api_key)
    response = client.rerank(
        model=COHERE_MODEL,
        query=query,
        documents=[scored.chunk.text for scored in candidates],
        top_n=top_k,
    )
    return [(int(item.index), float(item.relevance_score)) for item in response.results]


RERANKERS["cohere"] = _cohere
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_rerank.py -v`

Expected: PASS. Then: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`

- [ ] **Step 7: One real call, to prove the wiring**

Put a real key in `.env` as `COHERE_API_KEY=`, then:

```bash
uv run python scripts/search.py "how do dependencies work" --rerank cohere --rerank-candidates 20
```

Expected: 5 results, asterisked scores, ordered differently from the same query without `--rerank`. Cost: one search, $0.002.

- [ ] **Step 8: Commit**

```bash
git add app/retrieval/rerank.py app/core/config.py tests/test_retrieval_rerank.py pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
feat(retrieval): add the Cohere Rerank backend

The API answers with index positions into the list it was sent, not with
documents. An off-by-one there pairs every score with the wrong chunk and still
returns a plausible ranking no aggregate metric would catch, so the client is
injectable and that mapping is the first test in the file.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
```

---

### Task 7: The run matrix, the sweep, and the verdict

**Files:**
- Modify: `data/eval/results.jsonl`
- Modify: `app/core/config.py` (only if the rule is met)
- Modify: `.env.example` (only if the rule is met)

**Interfaces:**
- Consumes: everything from tasks 1-6.
- Produces: ~10 rows in `data/eval/results.jsonl`; a recorded verdict; `Settings.rerank_model` flipped from `""` to the winner only if the rule below is met.

- [ ] **Step 1: Confirm the baseline and the pool numbers are in front of you**

```bash
uv run python scripts/benchmark.py --summary "dense-sentence-doctype"
uv run python scripts/benchmark.py --summary "*-ceiling"
```

Write down, for each pool: its Recall@5, its Recall@30, and `headroom = Recall@30 − Recall@5`. The default's Recall@5 is **0.776** and that is the number every absolute gain is measured against.

- [ ] **Step 2: Run the four-row matrix**

```bash
uv run python scripts/benchmark.py --label "rerank-flashrank-dense-d30"  --mode dense  --rerank flashrank --rerank-candidates 30
uv run python scripts/benchmark.py --label "rerank-flashrank-hybrid-d30" --mode hybrid --rrf-k 60 --candidates 30 --rerank flashrank --rerank-candidates 30
uv run python scripts/benchmark.py --label "rerank-cohere-dense-d30"     --mode dense  --rerank cohere    --rerank-candidates 30
uv run python scripts/benchmark.py --label "rerank-cohere-hybrid-d30"    --mode hybrid --rrf-k 60 --candidates 30 --rerank cohere    --rerank-candidates 30
```

`--top-k` defaults to `max(KS)` = 30, which is what the metrics need. FlashRank rows are free; the two Cohere rows are 38 searches each, about $0.15 total. A trial key rate-limits to roughly 10 searches a minute, so each Cohere row takes ~4 minutes — that is the API's limit, not a bug.

- [ ] **Step 3: Sweep the depth on the best pairing**

Take whichever of the four rows has the highest Recall@5 and re-run it at three more depths:

```bash
uv run python scripts/benchmark.py --label "rerank-<backend>-<pool>-d10" ... --rerank-candidates 10
uv run python scripts/benchmark.py --label "rerank-<backend>-<pool>-d20" ... --rerank-candidates 20
uv run python scripts/benchmark.py --label "rerank-<backend>-<pool>-d50" ... --rerank-candidates 50 --candidates 50
```

The d50 run needs `--candidates 50` in hybrid mode: the fused list cannot be 50 deep if each branch only retrieved 30.

- [ ] **Step 4: Compute headroom capture for every run**

For each reranked row, against its own pool's ceiling row:

```text
capture = (reranked Recall@5 − pool Recall@5) / (pool Recall@30 − pool Recall@5)
```

This is the number that says whether the cross-encoder is doing its job, separately from whether the pool was any good. Record it per run.

Get each backend's isolated latency cost the same way, by subtraction rather than by new plumbing:

```text
reranker cost = reranked row's p50 − its pool's ceiling row p50
```

Both were measured at the same depth over the same 38 questions, so the difference is the cross-encoder and nothing else. This is what makes clause 3 of the decision rule checkable, and it is why the ceiling rows belong in the published table rather than only in this plan.

- [ ] **Step 5: Apply the decision rule, fixed before the first run**

`RERANK_MODEL` changes from empty to the winning backend **if and only if all three hold**:

1. **Headroom capture ≥ 0.50** *and* absolute Recall@5 ≥ **0.806** (the 0.776 default plus 0.03).
2. **No per-category Recall@5 regresses by more than 0.05** against the baseline's per-category table: `exact` 0.892, `code` 0.850, `conceptual` 0.733, `multi_doc` 0.594.
3. **p50 retrieval latency < 400 ms.**

```bash
uv run python scripts/benchmark.py --label "rerank-<winner>" --compare "dense-sentence-doctype"
```

If all three hold, set `rerank_model` in `app/core/config.py` to the winner, update its comment with the measured numbers in the style of the `retrieval_mode` and `chunk_strategy` comments, and update `RERANK_MODEL=` in `.env.example`.

**If any clause fails, `rerank_model` stays `""`.** Both backends, every test and every measured row still ship, and the verdict is recorded exactly as step 16 recorded hybrid's. Do not widen the criterion to make a number qualify.

- [ ] **Step 6: Commit**

```bash
git add data/eval/results.jsonl app/core/config.py .env.example
git commit -m "$(cat <<'EOF'
chore(evaluation): record the reranking matrix and the verdict

<one line: the winning row, its Recall@5, its headroom capture, and whether
RERANK_MODEL moved>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
```

---

### Task 8: Publish the result

**Files:**
- Modify: `README.md` (Phase 6 section, current state, results table, commands)
- Modify: `docs/roadmap.md` (current state, step map row, plans table)

**Interfaces:**
- Consumes: every number from tasks 1 and 7.
- Produces: the published result and the `v0.7` tag.

- [ ] **Step 1: Update the README**

In French, matching the existing voice. Four edits:

1. **Results table** — a row per measured run, with the `rerank` column. Include the ceiling rows: they are what makes every capture figure readable.
2. **Current state** — what shipped, and the verdict in one sentence.
3. **Commands** — `--rerank` and `--rerank-candidates` on the three scripts, plus the FlashRank warm-up one-liner and the note that it downloads ~34 MB once into `data/processed/flashrank/`.
4. **A note on the Cohere rows** — the first rows in this project that cannot be reproduced offline or for free, about $0.15 for the matrix. Say so next to them; "the free local model is good enough" is only a result because the paid one was actually run.

Also capture the qualitative transcript, as steps 14-16 did:

```bash
uv run python scripts/ask.py "What does HTTPException 422 mean?" --mode hybrid
uv run python scripts/ask.py "What does HTTPException 422 mean?" --mode hybrid --rerank <winner>
```

Paste both into the README, labelled as the qualitative evidence they are and attached to no metric. The roadmap already records the refusal as correct — the four chunks containing `422` are two release notes and an OpenAPI example, and none of them explains the code. If reranking changes that answer, it is a finding about the reranker, not a fix.

- [ ] **Step 2: Update `docs/roadmap.md`**

1. **Current state table** — a row for the reranker: the files, the numbers, the verdict.
2. **The "four things later steps own" list** — the `Recall@10 − Recall@5` bullet was step 17's precondition and step 17 has now consumed it. Replace it with what step 17 found: the measured ceiling, the capture figure, and the input steps 18-19 now have.
3. **Step map** — mark row 17 done with its numbers, in the shape rows 12-16 already use.
4. **Plans written so far** — add this plan and its spec.
5. **Current state header** — `v0.7` tagged, step 18 next.

- [ ] **Step 3: Run the full gates one last time**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest
```

- [ ] **Step 4: Commit and tag**

```bash
git add README.md docs/roadmap.md docs/superpowers/plans/2026-09-13-step-17-reranking.md
git commit -m "$(cat <<'EOF'
docs: publish the reranking results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HMTrJTHmCuWCV9eJpcnest
EOF
)"
git tag v0.7
```

---

## Notes for the executor

**Task 1 can end the plan.** If the best pool's headroom is under 0.03, stop after task 1, jump to task 8, and publish the finding. Adding two dependencies and then discovering there was nothing to reorder is discovering it too late. This is the same shape as step 13's +0.000 routing ceiling — a completed step, not an abandoned one.

**The three depths are different things and mixing them up is the likeliest bug in this plan:**

| name | meaning | where |
|---|---|---|
| `top_k` | how many chunks the caller wants | everywhere |
| `candidates` | per-branch depth **before** RRF fusion | hybrid mode only |
| `rerank_candidates` | how deep the **retrieved or fused** list goes into the cross-encoder | when `rerank` is set |

The constraint is `candidates >= rerank_candidates >= top_k`, and in hybrid mode the first half of it is enforced by `_hybrid`'s existing guard rather than by new code.

**Do not add a `rerank_score` column to the per-question history or renumber anything.** `score` stays the retriever's number and `rerank_score` carries the new one; `rerank_score is not None` is what says which produced the ranking. Both stay recorded because "the retriever never had it" and "the reranker buried it" are different failures and this step exists to tell them apart.
