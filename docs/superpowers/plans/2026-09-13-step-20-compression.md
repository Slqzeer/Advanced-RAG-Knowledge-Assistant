# Contextual Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a 20-chunk retrieval pool into the token budget five whole chunks occupy today, so the 0.099 Recall gap steps 17-19 proved no ranking stage can close finally reaches the model — and measure whether it survives the cut.

**Architecture:** A new `app/generation/compress.py` holds a `COMPRESSORS` registry whose one entry, `embedding`, scores every sentence of every retrieved chunk against the query using the existing `embed_texts()` and its sqlite cache. `compress()` owns splitting (via a `sentence_spans()` promoted out of `chunk.py`, which already keeps a fenced code block unbreakable), greedy budgeting, re-assembly in document order with a `[…]` marker at each cut, and the `ScoredChunk` rebuild. It runs in `answer_question()` between `search()` and `build_context()` — never inside either.

**Tech Stack:** Python 3.12, `uv`, pydantic v2, Qdrant, OpenAI `text-embedding-3-small` and `gpt-4o-mini`, pytest, ruff, mypy strict. **No new dependency is added by this plan.**

**Spec:** [`docs/superpowers/specs/2026-09-13-compression-design.md`](../specs/2026-09-13-compression-design.md)

## Global Constraints

- **No new dependency.** Nothing is added to `pyproject.toml`. Roadmap rule 3: dependencies arrive with the step that uses them, and this step uses none.
- **No tokeniser.** The budget is denominated in **characters**; the number *reported* is `usage.prompt_tokens` off the API response. `tiktoken` is explicitly out of scope.
- **Quality gates before every commit:** `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest` — all four, all green.
- **Every unit test runs with no network, no key and no spend.** `embedder`, `llm`, `client` and `retriever` are injectable; use them.
- **Every measured arm runs on collection `chunks`.** Not `chunks_sentence`. The two disagree by 0.009 Recall@5 and a matrix that straddles them cannot be compared to its own control.
- **The baseline is Recall@5 = 0.785**, from `dense-d30-ceiling` on `chunks`. Not the 0.776 of `dense-sentence-doctype`.
- **`COMPRESS_METHOD` stays empty unless the acceptance rule in Task 6 is met, all three clauses.** A phase that widens its criterion until the number qualifies has not satisfied roadmap rule 1.
- **Defaults are off.** `compress=None` reads the setting; `compress=""` forces off. Identical to the `rerank=` and `transform=` convention already in `search()`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `app/generation/citations.py` | modify: promote `_is_refusal` → `is_refusal` | 1 |
| `scripts/benchmark_answers.py` | **create**: answer-side arm — refusal rate, context characters, real `prompt_tokens`. Sibling of `benchmark_conversations.py` | 1 |
| `data/eval/answers.jsonl` | **create**: one row per answer-side run. Separate from `results.jsonl`, whose rows are retrieval runs that `summarise()` reads by `recall@5` | 1 |
| `app/ingestion/chunk.py` | modify: promote `_sentence_units` → public `sentence_spans()` | 2 |
| `app/generation/compress.py` | **create**: `COMPRESSORS`, `compress()`, the `embedding` entry | 2 |
| `tests/test_generation_compress.py` | **create**: the compressor's tests | 2 |
| `app/core/config.py` | modify: three settings | 3 |
| `.env.example` | modify: document the three settings | 3 |
| `app/generation/answer.py` | modify: `compress=` / `compress_candidates=` / `compress_budget=`, the widened pool, the corrected `RetrievalStats` arithmetic | 3 |
| `scripts/ask.py` | modify: `--compress`, `--compress-candidates`, `--compress-budget`, the compression line of output | 3 |
| `scripts/benchmark.py` | modify: the same three flags, the closure, the `compress` config keys and summary column | 4 |
| `app/evaluation/benchmark.py` | modify: one entry in `SUMMARY_COLUMNS` | 4 |
| `README.md`, `docs/roadmap.md`, `app/generation/context.py` | modify: the verdict, the results table, the corrected `ponytail:` comment, the rewritten parent-child row | 6 |

`scripts/search.py` is deliberately **not** touched: it prints ranked chunks, compression does not change the ranking, and a truncated list of the same documents in the same order teaches a reader nothing.

---

### Task 1: The answer-side arm, and the budget it sets

Everything downstream is calibrated by one number nobody has measured: how many characters today's `top_k=5` context actually is. The spec's Task 1 says to measure it before writing any compressor. This plan builds that measurement as the script Task 5's refusal arm needs anyway, rather than as a throwaway — one script, used twice.

It is a separate script from `scripts/benchmark.py` for the reason `benchmark_conversations.py` is: `run_benchmark`'s retriever seam is `str -> list[ScoredChunk]` and four steps are built on it. An answer-side run produces refusals and token counts, not ranking metrics, so forcing it into `BenchmarkResult` would mean inventing `recall@5` values that `summarise()` would then render.

**Files:**
- Modify: `app/generation/citations.py:139` (`_is_refusal` → `is_refusal`)
- Modify: `app/generation/answer.py` (the one internal caller of `_is_refusal`, if any — grep first)
- Create: `scripts/benchmark_answers.py`
- Test: `tests/test_generation_citations.py` (one added test)

**Interfaces:**
- Consumes: `answer_question()` as it exists today; `load_dataset`, `EvalQuestion`; `print_result`-style output conventions from `scripts/benchmark.py`.
- Produces: `is_refusal(answer: str) -> bool` (public, used by Task 5); `scripts/benchmark_answers.py --label <str> [--compress ...] [--top-k N] [--no-save]` writing one JSON object per run to `data/eval/answers.jsonl` with keys `label`, `timestamp`, `git_commit`, `config`, `questions`, `refusals`, `refusal_rate`, `context_chars_p50`, `context_chars_p95`, `prompt_tokens_p50`, `prompt_tokens_p95`, `latency_p50_ms`, `per_question`.

- [ ] **Step 1: Find every caller of the private name**

Run: `grep -rn "_is_refusal" app/ tests/ scripts/`

Note each hit. They all get renamed in Step 3. Do not skip this — renaming a function whose callers you have not enumerated is how a "simple rename" breaks a test suite.

- [ ] **Step 2: Write the failing test for the public name**

Add to `tests/test_generation_citations.py`:

```python
from app.generation.citations import is_refusal


def test_is_refusal_is_public_and_matches_both_languages() -> None:
    assert is_refusal("I do not have enough information in the provided context to answer this.")
    assert is_refusal("Je ne dispose pas des informations nécessaires.")
    assert not is_refusal("FastAPI uses dependency injection through Depends [1].")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_generation_citations.py::test_is_refusal_is_public_and_matches_both_languages -v`

Expected: FAIL with `ImportError: cannot import name 'is_refusal'`.

- [ ] **Step 4: Rename the function and every caller**

In `app/generation/citations.py`, rename `_is_refusal` to `is_refusal` and extend its docstring:

```python
def is_refusal(answer: str) -> bool:
    """True when the answer declines rather than asserts.

    Public since step 20: it is that step's only signal that sees *inside* a
    surviving document. Recall@context cannot tell a context that kept the
    answer from one that kept the right document and the wrong three sentences
    of it; a model handed the second says so itself, in a fixed string, free.
    """
```

Rename every call site found in Step 1.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`

Expected: PASS, no test lost. If a test referenced `_is_refusal`, it is now green under the new name.

- [ ] **Step 6: Commit**

```bash
git add app/generation/citations.py tests/test_generation_citations.py
git commit -m "refactor(generation): make is_refusal public for step 20's answer arm"
```

- [ ] **Step 7: Write `scripts/benchmark_answers.py`**

```python
"""What did the answer cost, and did the model still answer at all?

    uv run python scripts/benchmark_answers.py --label answers-k5
    uv run python scripts/benchmark_answers.py --label answers-compress-d20 \
        --compress embedding --compress-candidates 20

One row per run in `data/eval/answers.jsonl`. Three numbers per row: how many
characters of context the prompt carried, how many `prompt_tokens` the provider
actually billed, and how often the model refused a question the dataset says is
answerable.

Separate from `scripts/benchmark.py` for the reason `benchmark_conversations.py`
is separate: `run_benchmark`'s seam is `str -> list[ScoredChunk]` and produces
ranking metrics. This produces neither. Forcing it into `BenchmarkResult` would
mean inventing a `recall@5` for `summarise()` to render.

Refusal rate is a blunt binary signal and that is exactly its value at step 20:
it has no tuning surface, so it cannot be tuned into agreement. RAGAS judges an
answer properly at step 21.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.generation.citations import is_refusal  # noqa: E402
from app.generation.compress import COMPRESSORS  # noqa: E402
from app.generation.answer import answer_question  # noqa: E402

HISTORY = Path("data/eval/answers.jsonl")
DEFAULT_DATASET = Path("data/eval/questions.jsonl")


def git_commit() -> str:
    """The commit a row was produced at. A number with no code beside it is a rumour."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def percentiles(values: list[float]) -> tuple[float, float]:
    """p50 and p95. One value is its own p50 and p95; none is zero."""
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--compress",
        choices=["", *sorted(COMPRESSORS)],
        default=None,
        help='sentence extractor; default: COMPRESS_METHOD, "" is off',
    )
    parser.add_argument("--compress-candidates", type=int, default=None)
    parser.add_argument("--compress-budget", type=int, default=None)
    parser.add_argument(
        "--unanswerable",
        action="store_true",
        help="run the 7 out-of-corpus questions instead; a refusal there is correct",
    )
    parser.add_argument("--no-save", action="store_true", help="print only, append nothing")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = get_settings()
    questions = [q for q in load_dataset(args.dataset) if not q.held_out]
    questions = [
        q for q in questions
        if (q.category == "unanswerable") == args.unanswerable
    ]
    if not questions:
        raise SystemExit("no questions selected")

    rows: list[dict[str, Any]] = []
    for question in questions:
        started = time.perf_counter()
        try:
            answer = answer_question(
                question.question,
                top_k=args.top_k,
                compress=args.compress,
                compress_candidates=args.compress_candidates,
                compress_budget=args.compress_budget,
                settings=settings,
            )
        except Exception as error:  # one transient API error must not cost a
            # 38-question run; it is recorded as a row, never swallowed.
            rows.append({"question_id": question.question_id, "error": f"{error}"})
            continue
        rows.append({
            "question_id": question.question_id,
            "category": question.category,
            "refused": is_refusal(answer.answer),
            "context_chars": answer.context_chars,
            "prompt_tokens": answer.usage.get("prompt_tokens", 0),
            "retrieved": answer.retrieval.retrieved,
            "used": answer.retrieval.used,
            "latency_ms": (time.perf_counter() - started) * 1000,
        })

    scored = [row for row in rows if "error" not in row]
    chars_p50, chars_p95 = percentiles([r["context_chars"] for r in scored])
    tokens_p50, tokens_p95 = percentiles([float(r["prompt_tokens"]) for r in scored])
    latency_p50, latency_p95 = percentiles([r["latency_ms"] for r in scored])
    refusals = sum(1 for row in scored if row["refused"])

    result = {
        "label": args.label,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": {
            "top_k": args.top_k or settings.top_k,
            "compress": args.compress if args.compress is not None else settings.compress_method,
            "compress_candidates": args.compress_candidates or settings.compress_candidates,
            "compress_budget": args.compress_budget or settings.compress_budget_chars,
            "collection": settings.qdrant_collection,
            "generation_model": settings.generation_model,
            "unanswerable": args.unanswerable,
        },
        "questions": len(scored),
        "failures": len(rows) - len(scored),
        "refusals": refusals,
        "refusal_rate": refusals / len(scored) if scored else 0.0,
        "context_chars_p50": chars_p50,
        "context_chars_p95": chars_p95,
        "prompt_tokens_p50": tokens_p50,
        "prompt_tokens_p95": tokens_p95,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "per_question": rows,
    }

    print(f"\n## {args.label}  ({result['questions']} questions, {result['failures']} failed)\n")
    print(f"refusal rate     {result['refusal_rate']:.3f}  ({refusals} of {len(scored)})")
    print(f"context chars    p50 {chars_p50:.0f}   p95 {chars_p95:.0f}")
    print(f"prompt tokens    p50 {tokens_p50:.0f}   p95 {tokens_p95:.0f}")
    print(f"latency ms       p50 {latency_p50:.0f}   p95 {latency_p95:.0f}")

    if not args.no_save:
        with HISTORY.open("a", encoding="utf-8") as history:
            history.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\nappended to {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Note:** this script references `answer.context_chars` and `answer_question(compress=...)`, neither of which exists until Task 3. That is deliberate — the script is written once, here, and first *run* in Step 9 below against a stub. If you prefer, write it in Task 3 instead; the plan keeps it here so Task 1's measurement and Task 5's refusal arm are visibly the same tool.

- [ ] **Step 8: Add `context_chars` to `Answer`**

In `app/models/answers.py`, add to `Answer`:

```python
    # The characters of context the prompt actually carried. Step 20's fixed
    # variable: a compression arm that reports a smaller Recall for a smaller
    # prompt has said nothing until both numbers are on the same row.
    context_chars: int = Field(ge=0, default=0)
```

In `app/generation/answer.py`, pass it: `context_chars=len(context)` in the `Answer(...)` construction. With compression off this is today's number, measured for the first time.

- [ ] **Step 9: Run the baseline, uncompressed**

Ensure Qdrant is up and `QDRANT_COLLECTION=chunks`. Run:

```bash
uv run python scripts/benchmark_answers.py --label answers-k5-baseline --compress ""
```

Expected: 38 questions, a refusal rate near 0.0, and — the number this task exists for — `context chars p50`.

- [ ] **Step 10: Record the number and set the budget**

Write the measured `context_chars_p50` into the spec's configuration table, replacing "set by task 1", and use it as `COMPRESS_BUDGET_CHARS`'s default in Task 3.

**If it comes back materially above ~6 000**, stop and re-read the spec's "Facts this design is built on": the estimate of ~4 000 came from a median chunk size, and a real number twice that changes what a d20 pool can be compressed into and therefore what Task 6's verdict can be. Say so before continuing rather than after the matrix has run.

- [ ] **Step 11: Commit**

```bash
git add scripts/benchmark_answers.py app/models/answers.py app/generation/answer.py data/eval/answers.jsonl
git commit -m "chore(evaluation): measure today's real context size and refusal baseline"
```

---

### Task 2: `sentence_spans()` and the compressor

**Files:**
- Modify: `app/ingestion/chunk.py:210` (promote `_sentence_units`), `chunk.py:268` (`split_sentences` calls the new name)
- Create: `app/generation/compress.py`
- Test: `tests/test_generation_compress.py` (create), `tests/test_ingestion_chunk.py` (one added test)

**Interfaces:**
- Consumes: `sentence_spans()` from Task 2's own first half; `embed_texts()` and `EmbeddingCache` from `app.ingestion.embed`; `ScoredChunk` and `Chunk` from `app.models.chunks`; `Settings` from `app.core.config`.
- Produces:
  - `sentence_spans(text: str) -> list[tuple[int, int]]`
  - `BatchEmbedder = Callable[[Sequence[str]], list[list[float]]]`
  - `Compressor = Callable[[str, Sequence[str], Settings, BatchEmbedder | None], list[float]]`
  - `COMPRESSORS: dict[str, Compressor]` with key `"embedding"`
  - `GAP = " […] "`
  - `compress(chunks: Sequence[ScoredChunk], query: str, *, method: str | None = None, budget_chars: int | None = None, settings: Settings | None = None, embedder: BatchEmbedder | None = None) -> list[ScoredChunk]`

**Why `Compressor` scores rather than compresses:** the spec sketched it as `(chunks, query, budget, settings, embedder) -> list[ScoredChunk]`. Reading the code says otherwise: splitting, budgeting, re-assembly and the frozen-model rebuild are identical for every conceivable entry, and the only thing a BM25 or LLMLingua entry would change is how a unit gets a number. So the registry entry is *scoring only*, and `compress()` owns the rest — the same division `expand()` uses, where parsing and capping are shared and only the prompt differs.

**Why `BatchEmbedder` and not `search.Embedder`:** `search.Embedder` is `Callable[[str], list[float]]` and embeds one query. This embeds ~200 sentences and must do it in one batched call. Reusing the name would hide a different shape behind a familiar one.

- [ ] **Step 1: Write the failing test for `sentence_spans`**

Add to `tests/test_ingestion_chunk.py`:

```python
from app.ingestion.chunk import _protected_spans, _sentence_units, sentence_spans


def test_sentence_spans_matches_the_private_splitter() -> None:
    text = "One. Two. Three.\n\n```py\nx = 1.  # not a boundary\ny = 2.\n```\n\nFour."
    assert sentence_spans(text) == _sentence_units(text, _protected_spans(text))


def test_sentence_spans_keeps_a_code_fence_in_one_span() -> None:
    text = "Intro sentence. \n\n```py\na = 1.\nb = 2.\n```\n\nOutro sentence."
    spans = sentence_spans(text)
    fence_start, fence_end = text.index("```py"), text.index("```\n\nOutro") + 3
    # Exactly one span contains the whole fence; none starts or ends inside it.
    assert sum(1 for lo, hi in spans if lo <= fence_start and hi >= fence_end) == 1
    assert not any(fence_start < lo < fence_end or fence_start < hi < fence_end for lo, hi in spans)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ingestion_chunk.py -k sentence_spans -v`

Expected: FAIL with `ImportError: cannot import name 'sentence_spans'`.

- [ ] **Step 3: Add `sentence_spans` and route `split_sentences` through it**

In `app/ingestion/chunk.py`, immediately after `_sentence_units`:

```python
def sentence_spans(text: str) -> list[tuple[int, int]]:
    """One span per sentence, tiling ``text``; a fenced code block is one span.

    Public because step 20's compressor must split exactly as this module does.
    Two splitters that are "the same for now" are two splitters that diverge at
    the next edit — and the thing that would diverge first is the fence rule,
    which is the one protecting the ``code`` category that steps 17 and 19 each
    measured a reranker destroying.
    """
    return _sentence_units(text, _protected_spans(text))
```

In `split_sentences`, replace `units = _sentence_units(text, _protected_spans(text))` with `units = sentence_spans(text)`. No behaviour changes; step 12's measured winner is untouched.

- [ ] **Step 4: Run the chunking suite**

Run: `uv run pytest tests/test_ingestion_chunk.py -v`

Expected: PASS, including every pre-existing `split_sentences` test.

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/chunk.py tests/test_ingestion_chunk.py
git commit -m "refactor(ingestion): promote sentence_spans so the compressor splits as the indexer does"
```

- [ ] **Step 6: Write the failing tests for the compressor**

Create `tests/test_generation_compress.py`:

```python
"""The compressor's contract. No network, no key, no spend: the embedder is faked."""

from collections.abc import Sequence

import pytest

from app.core.config import Settings
from app.generation.compress import COMPRESSORS, GAP, compress
from app.models.chunks import Chunk, ScoredChunk


def make_chunk(text: str, index: int = 0, document_id: str = "fastapi:a") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            document_id=document_id,
            source="fastapi",
            title="A",
            doc_type="tutorial",
            chunk_index=index,
            text=text,
            char_start=0,
            char_end=len(text),
        ),
        score=0.5,
        rank=index + 1,
    )


def fake_embedder(scores: dict[str, float]) -> "object":
    """Returns a 2-D vector per text: the query is (1, 0), a unit is (score, rest).

    Cosine against (1, 0) is then exactly ``scores[text]`` for any unit listed,
    and 0.0 for any that is not. Two dimensions rather than 1 536 because the
    compressor must not care, and a test that needs 1 536 numbers to say
    "this sentence matters more" is a test nobody edits.
    """
    import math

    def embed(texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for position, text in enumerate(texts):
            if position == 0:  # the query
                out.append([1.0, 0.0])
                continue
            score = scores.get(text.strip(), 0.0)
            out.append([score, math.sqrt(max(0.0, 1.0 - score * score))])
        return out

    return embed


SETTINGS = Settings(compress_method="embedding", compress_budget_chars=100, compress_candidates=20)


def test_compression_off_returns_the_input_unchanged() -> None:
    chunks = [make_chunk("One. Two. Three.")]
    assert compress(chunks, "q", method="", settings=SETTINGS) == chunks


def test_unknown_method_raises_and_names_the_keys() -> None:
    with pytest.raises(ValueError, match="unknown compressor"):
        compress([make_chunk("One.")], "q", method="nope", settings=SETTINGS)
    assert "embedding" in sorted(COMPRESSORS)


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="query is empty"):
        compress([make_chunk("One.")], "   ", method="embedding", settings=SETTINGS)


def test_a_budget_larger_than_the_input_is_a_no_op() -> None:
    chunks = [make_chunk("One. Two.")]
    out = compress(chunks, "q", method="embedding", budget_chars=10_000, settings=SETTINGS)
    assert out == chunks


def test_survivors_are_emitted_in_document_order_with_a_gap_marker() -> None:
    # Sentence 3 scores highest, sentence 1 next, sentence 2 not at all.
    text = "Alpha one. Beta two. Gamma three."
    embed = fake_embedder({"Alpha one.": 0.8, "Beta two.": 0.1, "Gamma three.": 0.9})
    out = compress(
        [make_chunk(text)], "q", method="embedding",
        budget_chars=len("Alpha one.") + len(" Gamma three.") + len(GAP),
        settings=SETTINGS, embedder=embed,
    )
    assert len(out) == 1
    assert out[0].chunk.text == f"Alpha one.{GAP}Gamma three."
    assert "Beta two." not in out[0].chunk.text


def test_a_code_fence_is_kept_whole_or_dropped_whole() -> None:
    fence = "```py\nfrom fastapi import FastAPI.\napp = FastAPI().\n```"
    text = f"Prose lead in. \n\n{fence}\n\nProse tail out."
    embed = fake_embedder({"Prose lead in.": 0.9})
    # A budget that fits the prose but not the fence.
    out = compress(
        [make_chunk(text)], "q", method="embedding",
        budget_chars=len("Prose lead in.") + 5, settings=SETTINGS, embedder=embed,
    )
    kept = out[0].chunk.text if out else ""
    assert "```" not in kept, "half a fence is broken code, not shorter code"


def test_a_chunk_with_no_surviving_sentence_disappears() -> None:
    keep, drop = make_chunk("Relevant sentence here.", 0, "fastapi:a"), make_chunk(
        "Irrelevant filler text.", 1, "fastapi:b"
    )
    embed = fake_embedder({"Relevant sentence here.": 0.9, "Irrelevant filler text.": 0.0})
    out = compress(
        [keep, drop], "q", method="embedding",
        budget_chars=len("Relevant sentence here."), settings=SETTINGS, embedder=embed,
    )
    assert [c.chunk.document_id for c in out] == ["fastapi:a"]


def test_rank_order_across_chunks_is_preserved() -> None:
    # The second chunk holds the better sentence; it must still come second.
    first = make_chunk("Weak but ranked first.", 0, "fastapi:a")
    second = make_chunk("Strong but ranked second.", 1, "fastapi:b")
    embed = fake_embedder({"Weak but ranked first.": 0.2, "Strong but ranked second.": 0.95})
    out = compress(
        [first, second], "q", method="embedding", budget_chars=200,
        settings=SETTINGS, embedder=embed,
    )
    assert [c.chunk.document_id for c in out] == ["fastapi:a", "fastapi:b"]


def test_the_best_sentence_survives_a_budget_too_small_for_it() -> None:
    embed = fake_embedder({"Alpha one.": 0.1, "Beta two.": 0.9})
    out = compress(
        [make_chunk("Alpha one. Beta two.")], "q", method="embedding",
        budget_chars=1, settings=SETTINGS, embedder=embed,
    )
    # Never an empty context: an empty one makes a wrong answer certain, the
    # same reasoning build_context() uses to keep its first chunk over budget.
    assert out and out[0].chunk.text.strip() == "Beta two."


def test_the_assembled_text_never_exceeds_the_budget() -> None:
    text = "One sentence. Two sentence. Three sentence. Four sentence. Five sentence."
    embed = fake_embedder({s.strip() + "." : 0.9 - i * 0.1 for i, s in enumerate(text.split("."))})
    budget = 40
    out = compress(
        [make_chunk(text)], "q", method="embedding", budget_chars=budget,
        settings=SETTINGS, embedder=embed,
    )
    assert sum(len(c.chunk.text) for c in out) <= budget


def test_the_char_span_still_points_at_the_original_document_region() -> None:
    text = "Alpha one. Beta two."
    embed = fake_embedder({"Alpha one.": 0.9})
    out = compress(
        [make_chunk(text)], "q", method="embedding", budget_chars=len("Alpha one."),
        settings=SETTINGS, embedder=embed,
    )
    # Provenance, not length: char_start/char_end say where the chunk was cut
    # from, which is what a citation needs and what compression does not change.
    assert (out[0].chunk.char_start, out[0].chunk.char_end) == (0, len(text))
```

- [ ] **Step 7: Run them to verify they fail**

Run: `uv run pytest tests/test_generation_compress.py -v`

Expected: every test FAILs with `ModuleNotFoundError: No module named 'app.generation.compress'`.

- [ ] **Step 8: Write `app/generation/compress.py`**

```python
"""Ranked chunks in, shorter ranked chunks out. The last stage before the prompt.

``information.md`` frames phase 9 as token reduction. On this corpus that
measures nothing: five ``sentence`` chunks are roughly 4 000 characters against a
``MAX_CONTEXT_CHARS`` of 12 000 that has never once bound, and 40 % off a
1 000-token prompt is $0.0001.

What compression is for *here* is the gap steps 17-19 proved no ranking stage can
close. Dense retrieval reaches 0.785 Recall@5 and 0.884 Recall@20; FlashRank
recovered +0.002 of it for 1 141 ms and multi-query only reordered the pool.
A stage that makes a chunk *cheaper* is the one remaining way rank 6-20 reaches
the model at today's budget — and half that gap is ``multi_doc``, which is short
of distinct documents per character, which is exactly what this buys.

A ``Compressor`` scores units and nothing else. ``compress`` owns splitting,
budgeting, re-assembly and the frozen-model rebuild, for the same reason
``expand()`` owns parsing and capping for both transforms: the only thing a
future BM25 or LLMLingua entry changes is how a sentence gets a number.
"""

import math
from collections.abc import Callable, Sequence

from app.core.config import Settings, get_settings
from app.ingestion.chunk import sentence_spans
from app.ingestion.embed import EmbeddingCache, embed_texts
from app.models.chunks import ScoredChunk

# Deliberately not `search.Embedder`, which is `Callable[[str], list[float]]` and
# embeds one query. This embeds every sentence of twenty chunks and must do it in
# one batched call; sharing the name would hide a different shape behind a
# familiar one.
BatchEmbedder = Callable[[Sequence[str]], list[list[float]]]

# One score per unit, in input order. Scoring only — see the module docstring.
Compressor = Callable[[str, Sequence[str], Settings, BatchEmbedder | None], list[float]]

# Mirrors RETRIEVERS, RERANKERS, TRANSFORMS and STRATEGIES: the registry is how
# this project compares N variants and promotes a winner, and it is what makes
# "add LLMLingua later" one function rather than a refactor.
COMPRESSORS: dict[str, Compressor] = {}

# Marks where text was cut. Without it the model reads two non-adjacent sentences
# as contiguous prose, and the confident answer stitched from two unrelated
# clauses is indistinguishable from a hallucination in the output. One string,
# and it is the difference between a compressed context and a misleading one.
GAP = " […] "


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Written out rather than assumed: `text-embedding-3-*` returns normalised
    vectors today, so a dot product would agree — and a BGE-family model later
    would not, silently."""
    norm = math.hypot(*left) * math.hypot(*right)
    return sum(x * y for x, y in zip(left, right, strict=True)) / norm if norm else 0.0


def _default_embedder(settings: Settings) -> BatchEmbedder:
    """The same sqlite cache the corpus was embedded with, so a repeated arm is free."""
    cache = EmbeddingCache(settings.embedding_cache_path)
    return lambda texts: embed_texts(list(texts), model=settings.embedding_model, cache=cache)


def _embedding(
    query: str, units: Sequence[str], settings: Settings, embedder: BatchEmbedder | None
) -> list[float]:
    """Cosine of each unit against the query, in one batched call.

    The query rides along in the same batch rather than in a call of its own: it
    is one more string, it lands in the same cache, and two calls where one will
    do is two chances for a transient error inside a 38-question run.
    """
    vectors = (embedder or _default_embedder(settings))([query, *units])
    return [_cosine(vectors[0], vector) for vector in vectors[1:]]


COMPRESSORS["embedding"] = _embedding


def _texts(
    chunks: Sequence[ScoredChunk], units: Sequence[tuple[int, int, str]], kept: set[int]
) -> dict[int, str]:
    """Chunk index -> compressed text: survivors in *document* order, GAP at cuts.

    Selection is by score; rendering is by position. Emitting in score order
    produces a paragraph that contradicts itself across sentence boundaries.
    """
    by_chunk: dict[int, list[int]] = {}
    for index in sorted(kept):  # units were built in document order
        by_chunk.setdefault(units[index][0], []).append(index)

    out: dict[int, str] = {}
    for chunk_index, indices in by_chunk.items():
        parts: list[str] = []
        for position, index in enumerate(indices):
            if position and units[index][1] != units[indices[position - 1]][1] + 1:
                parts.append(GAP)
            parts.append(units[index][2])
        text = "".join(parts).strip()
        if text:  # Chunk.text is min_length=1; an all-whitespace unit set is a drop
            out[chunk_index] = text
    return out


def compress(
    chunks: Sequence[ScoredChunk],
    query: str,
    *,
    method: str | None = None,
    budget_chars: int | None = None,
    settings: Settings | None = None,
    embedder: BatchEmbedder | None = None,
) -> list[ScoredChunk]:
    """The chunks, with only the sentences that answer ``query``, within a budget.

    ``method`` names an entry in ``COMPRESSORS``. ``None`` reads
    ``COMPRESS_METHOD`` and the empty string forces it off, which is how an
    uncompressed baseline stays runnable once a default flips — the convention
    ``rerank=`` and ``transform=`` already established.

    ``budget_chars`` counts **chunk text only**. Headers and the ``[n]`` numbering
    are added afterwards by ``build_context`` and are identical across arms, so
    excluding them keeps the fixed variable fixed. The number reported in the
    README is ``usage.prompt_tokens`` off the API response, not an estimate
    derived from this one.

    Rank order is preserved and never touched. Steps 16, 17 and 19 spent this
    project's entire ranking budget; a compressor that also reordered would
    confound the two in a single number.

    ``embedder`` is injectable so every unit test runs with no network, no key
    and no spend — the pattern ``answer_question``, ``search`` and ``expand`` use.
    """
    settings = settings or get_settings()
    method = settings.compress_method if method is None else method
    if not method:
        return list(chunks)
    if method not in COMPRESSORS:
        raise ValueError(f"unknown compressor {method!r}; have {sorted(COMPRESSORS)}")
    if not query.strip():
        # The empty string embeds fine and then scores every sentence equally,
        # which silently degrades to "keep the first ones that fit".
        raise ValueError("query is empty")
    if not chunks:
        return []

    budget = budget_chars or settings.compress_budget_chars
    if budget < 1:
        raise ValueError(f"budget_chars must be at least 1, got {budget}")

    # (chunk index, unit index within that chunk, text) for every unit.
    units = [
        (chunk_index, unit_index, scored.chunk.text[start:end])
        for chunk_index, scored in enumerate(chunks)
        for unit_index, (start, end) in enumerate(sentence_spans(scored.chunk.text))
    ]
    if sum(len(text) for _, _, text in units) <= budget:
        return list(chunks)  # nothing to cut, and nothing to pay an embedder for

    scores = COMPRESSORS[method](query, [text for _, _, text in units], settings, embedder)
    order = sorted(range(len(units)), key=lambda index: scores[index], reverse=True)

    # Greedy by score, charging each unit its own length. The first is kept even
    # if it alone blows the budget, for build_context's reason: an empty context
    # makes a wrong answer certain, an oversized one merely makes a long prompt.
    kept: set[int] = set()
    total = 0
    for index in order:
        cost = len(units[index][2])
        if kept and total + cost > budget:
            continue
        kept.add(index)
        total += cost

    # The assembled text also carries GAP markers, which the loop above did not
    # charge for. One trim pass enforces the real budget rather than an estimate
    # of it; it runs a handful of times at most.
    while len(kept) > 1 and sum(len(t) for t in _texts(chunks, units, kept).values()) > budget:
        kept.discard(min(kept, key=lambda index: scores[index]))

    texts = _texts(chunks, units, kept)
    return [
        # char_start/char_end are deliberately unchanged: they say where in the
        # document this chunk was cut from, which is what a citation needs and
        # what compression does not alter.
        scored.model_copy(update={"chunk": scored.chunk.model_copy(update={"text": texts[index]})})
        for index, scored in enumerate(chunks)
        if index in texts
    ]
```

- [ ] **Step 9: Add the three settings so the tests can construct `Settings`**

The tests build `Settings(compress_method=..., compress_budget_chars=..., compress_candidates=...)`. Add them now (Task 3 documents them in `.env.example`). In `app/core/config.py`, after the `history_turns` block:

```python
    # Step 20. Empty means off: unchanged behaviour until a measurement earns
    # the change. embedding — measured at step 20, table in the README.
    compress_method: str = ""
    # How deep the retrieved pool goes into the compressor. Step 17 measured
    # Recall@20 == Recall@30 in every pool, so 30 is 50 % more sentence
    # embedding for nothing reachable.
    compress_candidates: int = 20
    # Characters of chunk text the compressed context may carry, headers
    # excluded. Set to the measured size of today's top_k=5 context, so an arm
    # that widens the pool is held at the cost of today's prompt.
    compress_budget_chars: int = 4000  # REPLACE with Task 1 Step 10's measured p50
```

- [ ] **Step 10: Run the compressor tests**

Run: `uv run pytest tests/test_generation_compress.py -v`

Expected: PASS, all twelve.

If `test_a_code_fence_is_kept_whole_or_dropped_whole` fails, do **not** relax the assertion — it is the guard for the `code` category that two prior steps measured a reranker destroying. Check that `sentence_spans` is being called (not a local re-split) and that the fence is genuinely one unit for the text in the test.

- [ ] **Step 11: Run the gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q`

Expected: all green.

- [ ] **Step 12: Commit**

```bash
git add app/generation/compress.py app/core/config.py tests/test_generation_compress.py
git commit -m "feat(generation): sentence-level contextual compression behind a registry"
```

---

### Task 3: Wire it into the pipeline

**Files:**
- Modify: `app/generation/answer.py`
- Modify: `scripts/ask.py`
- Modify: `.env.example`
- Test: `tests/test_generation_answer.py`

**Interfaces:**
- Consumes: `compress()`, `COMPRESSORS` from Task 2; `compress_method` / `compress_candidates` / `compress_budget_chars` from Task 2 Step 9.
- Produces: `answer_question(..., compress: str | None = None, compress_candidates: int | None = None, compress_budget: int | None = None)`; `Answer.context_chars`; `--compress`, `--compress-candidates`, `--compress-budget` on `scripts/ask.py`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_generation_answer.py`:

```python
def test_compression_off_leaves_the_context_byte_identical(monkeypatch) -> None:
    """The test that lets every existing caller stay untouched."""
    chunks = [make_scored_chunk("One. Two. Three.", index=i) for i in range(3)]
    seen: list[str] = []

    def llm(system: str, user: str, *, model: str) -> tuple[str, dict[str, int]]:
        seen.append(user)
        return "An answer [1].", {"total_tokens": 10, "prompt_tokens": 8}

    answer = answer_question(
        "q", compress="", retriever=lambda *a, **k: chunks, llm=llm, settings=SETTINGS
    )
    expected, _, _ = build_context(chunks)
    assert expected in seen[0]
    assert answer.context_chars == len(expected)


def test_compress_candidates_widens_top_k_only_when_compression_is_on() -> None:
    asked: list[int] = []

    def retriever(query: str, **kwargs) -> list[ScoredChunk]:
        asked.append(kwargs["top_k"])
        return [make_scored_chunk("One. Two.", index=0)]

    def llm(system: str, user: str, *, model: str) -> tuple[str, dict[str, int]]:
        return "An answer [1].", {"total_tokens": 10}

    answer_question("q", compress="", top_k=5, compress_candidates=20,
                    retriever=retriever, llm=llm, settings=SETTINGS)
    assert asked == [5]

    answer_question("q", compress="embedding", top_k=5, compress_candidates=20,
                    retriever=retriever, llm=llm, settings=SETTINGS,
                    compress_budget=10_000)
    assert asked == [5, 20]


def test_retrieval_stats_report_the_pool_not_the_survivors() -> None:
    """A d20 run that compresses to seven chunks reports 20 retrieved, not 7."""
    pool = [make_scored_chunk(f"Sentence {i} here.", index=i) for i in range(20)]

    def llm(system: str, user: str, *, model: str) -> tuple[str, dict[str, int]]:
        return "An answer [1].", {"total_tokens": 10}

    answer = answer_question(
        "q", compress="embedding", top_k=5, compress_candidates=20, compress_budget=40,
        retriever=lambda *a, **k: pool, llm=llm, settings=SETTINGS,
        embedder=lambda texts: [[1.0, 0.0]] + [[0.5, 0.866]] * (len(texts) - 1),
    )
    assert answer.retrieval.retrieved == 20
    assert answer.retrieval.used + answer.retrieval.dropped == 20
    assert answer.retrieval.used < 20
```

Reuse whatever `make_scored_chunk` / `SETTINGS` helpers already exist at the top of `tests/test_generation_answer.py`; if the helper is named differently there, use that name rather than adding a second one.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_generation_answer.py -k "compress" -v`

Expected: FAIL with `TypeError: answer_question() got an unexpected keyword argument 'compress'`.

- [ ] **Step 3: Wire `answer_question`**

In `app/generation/answer.py`, add the import:

```python
from app.generation.compress import BatchEmbedder
from app.generation.compress import compress as compress_chunks
```

Add three parameters to the signature, beside `rerank_candidates`:

```python
    compress: str | None = None,
    compress_candidates: int | None = None,
    compress_budget: int | None = None,
    embedder: BatchEmbedder | None = None,
```

Add to the docstring:

```
    ``compress`` names a sentence extractor from ``compress.COMPRESSORS``. It
    runs *after* retrieval and *before* ``build_context``: a retriever that
    rewrites chunk text has stopped being one, and ``build_context`` commits in
    its own docstring to no I/O and no model. ``None`` reads ``COMPRESS_METHOD``
    and the empty string forces it off.

    ``compress_candidates`` is the pool depth retrieved when compression is on —
    the whole point of the stage, since step 17 measured the pool holding
    documents no ranking could surface. With compression off, ``top_k`` is
    retrieved unchanged and nothing about this function's behaviour differs from
    step 19's.
```

Replace the retrieval and context block:

```python
    method = settings.compress_method if compress is None else compress
    wanted = top_k or settings.top_k
    depth = wanted
    if method:
        depth = compress_candidates or settings.compress_candidates
        if depth < wanted:
            # A pool shallower than the answer is a compressor with nothing to
            # choose between; the same guard search() applies to rerank_candidates.
            raise ValueError(f"compress_candidates ({depth}) must be at least top_k ({wanted})")

    chunks = retriever(
        query,
        top_k=depth,
        mode=mode,
        transform=transform,
        transform_n=transform_n,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        filters=filters,
        settings=settings,
    )
    # Counted before compression: how many chunks the retriever produced is a
    # retrieval fact, and a d20 run that compresses to seven must report 20.
    pool = len(chunks)
    chunks = compress_chunks(
        chunks, query, method=method, budget_chars=compress_budget,
        settings=settings, embedder=embedder,
    )
    context, sources, _ = build_context(chunks, max_chars=max_context_chars)
    used = len(sources)
```

and the `Answer(...)` construction:

```python
        retrieval=RetrievalStats(retrieved=pool, used=used, dropped=pool - used),
        context_chars=len(context),
```

Delete the now-unused `dropped` variable binding from `build_context`'s return (keep the third slot as `_`): with compression on, `build_context`'s own count is a subset of the chunks compression already removed, and reporting both would double-count.

- [ ] **Step 4: Run the answer tests**

Run: `uv run pytest tests/test_generation_answer.py -v`

Expected: PASS, including every pre-existing test. If a pre-existing test asserted `retrieval.dropped == 0` for a five-chunk run, it still passes: `pool == used` when nothing is compressed and nothing exceeds `MAX_CONTEXT_CHARS`.

- [ ] **Step 5: Add the flags to `scripts/ask.py`**

Beside the `--rerank` arguments:

```python
    parser.add_argument(
        "--compress",
        choices=["", *sorted(COMPRESSORS)],
        default=None,
        help='sentence extractor run before the prompt; default: COMPRESS_METHOD, "" is off',
    )
    parser.add_argument(
        "--compress-candidates",
        type=int,
        default=None,
        help="pool depth retrieved when compression is on; default: COMPRESS_CANDIDATES",
    )
    parser.add_argument(
        "--compress-budget",
        type=int,
        default=None,
        help="characters of chunk text the compressed context may carry",
    )
```

with `from app.generation.compress import COMPRESSORS  # noqa: E402` at the top.

Thread all three into the `answer_question(...)` call.

- [ ] **Step 6: Make `--show-context` tell the truth**

In the `if args.show_context:` block, compress before building the context — otherwise the flag prints a prompt that is not the one sent, which is precisely the bug it exists to catch:

```python
        chunks = search(
            args.question,
            top_k=args.compress_candidates or 20 if args.compress else (args.top_k or 5),
            mode=args.mode,
            rerank=args.rerank,
            rerank_candidates=args.rerank_candidates,
            transform=args.transform,
            transform_n=args.transform_n,
            filters=filters,
        )
        chunks = compress(
            chunks, args.question, method=args.compress, budget_chars=args.compress_budget
        )
        context, _, _ = build_context(chunks)
```

with `from app.generation.compress import COMPRESSORS, compress  # noqa: E402`.

- [ ] **Step 7: Add the compression line of output**

After the existing stats line in `scripts/ask.py`:

```python
    if answer.context_chars:
        prompt_tokens = answer.usage.get("prompt_tokens", 0)
        print(f"context: {answer.context_chars} chars  |  {prompt_tokens} prompt tokens")
```

The qualitative evidence this step owes is "what did it cut", and that is unreadable from an aggregate.

- [ ] **Step 8: Document the settings in `.env.example`**

Append:

```
# Compression contextuelle (étape 20)
# Vide = désactivé. embedding — mesurée à l'étape 20, table dans le README.
COMPRESS_METHOD=
# Profondeur du vivier envoyé au compresseur. L'étape 17 a mesuré
# Recall@20 == Recall@30 dans tous les viviers : 30 coûte 50 % d'embeddings en
# plus pour rien d'atteignable.
COMPRESS_CANDIDATES=20
# Caractères de texte de chunk que le contexte compressé peut porter, en-têtes
# exclus. Calé sur la taille mesurée du contexte top_k=5 actuel, pour qu'un
# vivier élargi coûte exactement le prompt d'aujourd'hui.
COMPRESS_BUDGET_CHARS=4000
```

with the real number from Task 1 Step 10 in place of 4000.

- [ ] **Step 9: Smoke-test the real pipeline**

```bash
uv run python scripts/ask.py "How do I declare a dependency in FastAPI?" \
  --compress embedding --compress-candidates 20 --show-context
```

Expected: the printed context shows sentences with `[…]` markers between cuts, the answer still cites `[n]`, and the new line reports fewer characters than the same command with `--compress ""`.

- [ ] **Step 10: Run the gates and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q
git add app/generation/answer.py scripts/ask.py .env.example tests/test_generation_answer.py
git commit -m "feat(generation): reach compression from answer_question and ask.py"
```

---

### Task 4: Recall@context and the two bracket rows

**Recall@context is `recall@K` for any `K` at least as large as the longest context.** `recall_at_k` slices `retrieved[:k]`, so a `k` larger than the list is the whole list. Running the benchmark at `--top-k 20` with a retriever closure that returns post-compression chunks therefore makes the existing `recall@20` column exactly this metric — no change to `metrics.py`, no change to `run_benchmark`.

**`precision@K` is meaningless on these rows** and must not be quoted from them: its denominator is `k` even when fewer results came back, which its own docstring says is deliberate. It penalises a compressor for compressing.

**Files:**
- Modify: `scripts/benchmark.py`
- Modify: `app/evaluation/benchmark.py` (`SUMMARY_COLUMNS`)
- Test: `tests/test_evaluation_benchmark.py`

**Interfaces:**
- Consumes: `compress()` and `COMPRESSORS` from Task 2; `run_benchmark`, `KS`, `HISTORY` unchanged.
- Produces: `scripts/benchmark.py --compress <str> --compress-candidates <int> --compress-budget <int>`; `config` keys `compress`, `compress_candidates`, `compress_budget`; a `compress` column in `SUMMARY_COLUMNS`.

- [ ] **Step 1: Write the failing test for the summary column**

Add to `tests/test_evaluation_benchmark.py`:

```python
def test_summarise_renders_a_missing_compressor_as_a_dash() -> None:
    history = [{
        "label": "old-row",
        "config": {"strategy": "sentence", "chunk_size": 1000, "chunk_overlap": 200},
        "aggregate": {"recall@5": 0.776},
        "per_category": {},
        "latency_p50_ms": 65.0,
    }]
    row = summarise(history, "old-row")[0]
    assert "-" in row, "every row written before step 20 genuinely had no compressor"
    assert len(row) == len(SUMMARY_COLUMNS)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_evaluation_benchmark.py -k compressor -v`

Expected: FAIL on the length assertion — `SUMMARY_COLUMNS` has one fewer entry than the row.

- [ ] **Step 3: Add the column**

In `app/evaluation/benchmark.py`, add `"compress"` to `SUMMARY_COLUMNS` immediately after `"transform"`, and in `summarise()`'s row construction, immediately after the `transform` entry:

```python
            # "-" rather than "": every row written before step 20 genuinely had
            # no compressor, and an empty cell reads as a missing value.
            str(row["config"].get("compress") or "-"),
```

`compress_candidates` and the budget get no column — the label carries them (`compress-embedding-d20`), as step 16 declined a column for `rrf_k`, step 17 for `rerank_candidates` and step 19 for `transform_n`.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_evaluation_benchmark.py -v`

Expected: PASS.

- [ ] **Step 5: Add the flags and the closure to `scripts/benchmark.py`**

Add the three arguments exactly as in Task 3 Step 5 (same names, same help text), with `from app.generation.compress import COMPRESSORS, compress  # noqa: E402`.

In the `retrieve` closure, after the `search(...)` call:

```python
    compressor = args.compress if args.compress is not None else settings.compress_method

    def retrieve(text: str) -> list[ScoredChunk]:
        filters: dict[str, str | list[str]] = dict(base_filters)
        if facet := oracle.get(text):
            filters["doc_type"] = facet
        chunks = search(
            text,
            top_k=args.top_k,
            mode=mode,
            candidates=args.candidates,
            rrf_k=args.rrf_k,
            rerank=args.rerank,
            rerank_candidates=args.rerank_candidates,
            transform=args.transform,
            transform_n=args.transform_n,
            llm=recorder,
            filters=filters or None,
            collection=collection,
            settings=settings,
        )
        # Recall@context: run_benchmark scores whatever this returns, so
        # returning the post-compression list makes recall@20 the fraction of
        # ground-truth documents that actually reached the prompt. The same
        # closure trick --oracle-filter already uses.
        return compress(
            chunks, text, method=compressor,
            budget_chars=args.compress_budget, settings=settings,
        )
```

Add to the `config` dict passed to `run_benchmark`:

```python
            "compress": compressor or None,
            "compress_candidates": args.compress_candidates or settings.compress_candidates,
            "compress_budget": args.compress_budget or settings.compress_budget_chars,
```

**Note on `--top-k`:** for a compression arm, `--top-k` *is* the pool depth — `search()` is called with it directly, and `compress` reduces what comes back. Pass `--top-k 20` for a d20 arm. `--compress-candidates` exists on this script for symmetry with `ask.py` and is recorded in `config`; it does not drive the closure, because `run_benchmark` already owns the depth through `--top-k`. State this in the script's module docstring so nobody wires it twice.

- [ ] **Step 6: Run the two bracket rows**

Ensure Qdrant is up and `QDRANT_COLLECTION=chunks`.

```bash
uv run python scripts/benchmark.py --label compress-off-k5  --top-k 5  --compress ""
uv run python scripts/benchmark.py --label compress-off-d20 --top-k 20 --compress ""
```

Expected: `compress-off-k5` reports `recall@5 ≈ 0.785`; `compress-off-d20` reports `recall@20 ≈ 0.884`. These are this step's own control and ceiling, measured by this step's own code on the same collection on the same day — quoting step 17's numbers instead would be quoting a run made before any of this existed.

**If either is more than 0.01 off the expected value, stop.** Something differs about the collection or the settings, and every arm after this is uninterpretable until it is explained.

- [ ] **Step 7: Run the gates and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q
git add scripts/benchmark.py app/evaluation/benchmark.py tests/test_evaluation_benchmark.py data/eval/results.jsonl
git commit -m "chore(evaluation): measure Recall@context and bracket it at 0.785 and 0.884"
```

---

### Task 5: The four compression arms and the refusal arm

**Files:**
- Modify: `data/eval/results.jsonl`, `data/eval/answers.jsonl` (appended by the runs)
- Create: `docs/superpowers/plans/transcripts-step-20.md` — the three verbatim transcripts

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: six rows in `results.jsonl`, five rows in `answers.jsonl`, three transcripts.

- [ ] **Step 1: Run the literal-brief control first**

```bash
uv run python scripts/benchmark.py --label compress-embedding-k5 --top-k 5 --compress embedding
```

This is `information.md`'s brief: the same pool, sentences stripped. It runs **before** the headline arm because it separates the two things the headline changes at once. If it loses ground against `compress-off-k5`, extraction damages context, and any gain at d20 is a wider pool paying for that damage — read it before reading the rest.

- [ ] **Step 2: Run the headline arm and the two sweeps**

```bash
uv run python scripts/benchmark.py --label compress-embedding-d20      --top-k 20 --compress embedding
uv run python scripts/benchmark.py --label compress-embedding-d10      --top-k 10 --compress embedding
uv run python scripts/benchmark.py --label compress-embedding-d20-b1.5x --top-k 20 --compress embedding \
    --compress-budget <1.5 x the Task 1 budget>
```

- [ ] **Step 3: Print the matrix**

```bash
uv run python scripts/benchmark.py --summary "compress-*"
```

Record the table. Read `recall@20` as Recall@context on every compressed row; ignore `precision@*` entirely, for the reason stated at the top of Task 4.

- [ ] **Step 4: Run the refusal arms**

```bash
uv run python scripts/benchmark_answers.py --label answers-compress-k5  --compress embedding --top-k 5
uv run python scripts/benchmark_answers.py --label answers-compress-d20 --compress embedding \
    --compress-candidates 20
uv run python scripts/benchmark_answers.py --label answers-unanswerable-baseline --compress "" --unanswerable
uv run python scripts/benchmark_answers.py --label answers-unanswerable-d20 --compress embedding \
    --compress-candidates 20 --unanswerable
```

The two `--unanswerable` rows are reported, not gated. A *drop* in refusal there is the interesting direction: a wider compressed pool giving the model more plausible-looking material to answer from is exactly what step 22 needs to know, and this is the only step that will see it.

- [ ] **Step 5: Capture the three transcripts**

Pick one `multi_doc` question (the category carrying 0.048 of the 0.099 gap), one `code` question (where the fence rule is on trial), and one question where compression made the answer **worse**. Run each twice and paste both outputs verbatim into `docs/superpowers/plans/transcripts-step-20.md`:

```bash
uv run python scripts/ask.py "<question>" --compress "" --show-context
uv run python scripts/ask.py "<question>" --compress embedding --compress-candidates 20 --show-context
```

**The third transcript is not optional.** The `HTTPException 422` transcript is in the README because a qualitative failure told this project something four aggregate rows did not. If no question got worse, write that sentence down explicitly — "no regression found among the 38" is a finding, "we did not look" is not.

- [ ] **Step 6: Commit the measurements**

```bash
git add data/eval/results.jsonl data/eval/answers.jsonl docs/superpowers/plans/transcripts-step-20.md
git commit -m "chore(evaluation): record the compression matrix and the refusal arms"
```

---

### Task 6: The verdict, the corrections, the docs

**Files:**
- Modify: `app/core/config.py` (the default, only if the rule is met)
- Modify: `app/generation/context.py:16` (the stale `ponytail:` comment)
- Modify: `README.md`, `docs/roadmap.md`
- Tag: `v1.2`

- [ ] **Step 1: Apply the decision rule, all three clauses**

`COMPRESS_METHOD` changes from empty to `embedding` **if and only if**:

1. **Recall@context ≥ 0.835** — +0.05 on the 0.785 `compress-off-k5` control, a little over half the 0.099 available.
2. **No per-category Recall@context regresses by more than 0.05** against `compress-off-k5`'s per-category table, **`code` checked explicitly**.
3. **Refusal rate on the 38 answerable questions does not exceed** `answers-k5-baseline`'s.

Read each clause against each `compress-embedding` arm independently. If more than one clears all three, the higher Recall@context wins and sets both `COMPRESS_METHOD` and `COMPRESS_CANDIDATES`; the others are recorded as measured-and-not-promoted.

Write the verdict down — which clauses passed, which failed, with the numbers — before editing any default. If the rule is not met, `COMPRESS_METHOD` stays empty and every row, test and line of code still ships, exactly as steps 13, 16, 17 and 19 recorded theirs.

- [ ] **Step 2: Correct the stale `ponytail:` comment**

In `app/generation/context.py`, replace lines 16-17:

```python
# ponytail: step 20 replaces this with a real token budget when compression
# arrives and the difference starts costing money.
```

with:

```python
# Step 20 measured the prediction below and retired it: the difference is worth
# about $0.0001 a question, so a per-model tokeniser was never the cost it was
# predicted to be. This stays the prompt's hard outer ceiling; the compressor's
# own budget (COMPRESS_BUDGET_CHARS) is the one that binds, and the number
# reported is usage.prompt_tokens off the API response, not an estimate.
```

A comment that has stopped being true is fixed where it is, the practice step 14 used on `store.py` and step 17 used on `search.py:217`.

- [ ] **Step 3: Rewrite the parent-child row in the README**

In the "Écarté volontairement" table, replace:

```
| Parent-child retrieval | étape 20 — ses métriques sont les tokens de contexte et la qualité de réponse, le Recall@5 y est aveugle |
| Tailles de chunk en tokens | étape 20, quand une limite de contexte contraindra vraiment |
```

with rows that name what actually happened: step 20 measured compression at a fixed budget and did not measure parent-child, because parent-child is the inverse trade — it grows the context rather than shrinking it, so at a fixed budget it competes with compression for the same characters instead of composing with it — and it needs a parent id in every payload, which is a re-index. Name the follow-up that owns both it and token-denominated chunk sizes.

Do not leave the old rows pointing at a step that has closed. A deferred item whose owner has shipped without it is an item nobody owns.

- [ ] **Step 4: Update the README**

- phase 9 section: the mechanism, the reframe (why token reduction measures nothing here), the six-row table, the per-category table with `multi_doc` read first, the refusal arms, the three transcripts, the verdict;
- the results table: the new rows, with real `prompt_tokens` p50 beside them;
- the current-state list at the top: one line for compression, in the register the others use.

- [ ] **Step 5: Update `docs/roadmap.md`**

- the header line: step 20 done, step 21 next;
- one row in the "Shipped and verified" table;
- the "four things later steps own" list: replace step 20's entry with what step 20 actually found, and state what step 21 inherits — specifically whether the gap between Recall@context and answer quality turned out to be real, since that is the gap RAGAS exists to measure.

- [ ] **Step 6: Run the gates**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q
```

- [ ] **Step 7: Commit and tag**

```bash
git add -A
git commit -m "docs: publish the contextual compression results"
git tag v1.2
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: `compress.py` and `sentence_spans()` → Task 2; the `answer_question` wiring, settings, `.env.example` and scripts → Task 3; Recall@context and the bracket rows → Task 4; the four arms, refusal rate and transcripts → Task 5; the acceptance rule, the `context.py` comment and the docs → Task 6. The spec's Task 1 ("measure today's context size") is Task 1 here, built as `scripts/benchmark_answers.py` so the same tool serves Task 5's refusal arm.

**Three deviations from the spec, each deliberate:**

1. **`Compressor` is scoring-only.** The spec typed it as returning `list[ScoredChunk]`. Reading the code showed splitting, budgeting and re-assembly are identical for every possible entry, so they belong in `compress()` — the division `expand()` already uses.
2. **`BatchEmbedder`, not `Embedder`.** `search.Embedder` is `Callable[[str], list[float]]`; this needs `Callable[[Sequence[str]], list[list[float]]]`. Reusing the name would hide a different shape.
3. **`Answer` gains `context_chars`.** The spec assumed the number was already available. It is not, and Task 1's measurement needs it.

**One thing the spec asserted that the plan can't yet honour:** `COMPRESS_BUDGET_CHARS = 4000` is a placeholder in Task 2 Step 9, with an explicit REPLACE marker and a gate at Task 1 Step 10 that stops the work if the measured value is materially different. That is the spec's own instruction — the budget is set by measurement, not by estimate — so the placeholder is the plan working as designed rather than a gap in it.

**Type consistency checked:** `compress(chunks, query, *, method, budget_chars, settings, embedder)` is called with exactly those keywords in `answer.py`, `ask.py` and `benchmark.py`; `is_refusal` is used under its public name in `benchmark_answers.py` only after Task 1 renames it; `Answer.context_chars` is written in Task 1 Step 8 and read in Task 1 Step 7's script and Task 3 Step 7's output line.
