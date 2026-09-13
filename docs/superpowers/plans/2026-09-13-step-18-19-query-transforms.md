# Query Transforms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Steps 18-19 — work on the *question* rather than on the index: rewrite it, resolve it against conversation history, or expand it into several phrasings whose rankings are fused — and measure whether any of that beats the 0.776 Recall@5 default.

**Architecture:** One new module, `app/retrieval/transform.py`, holding a `TRANSFORMS` registry (`rewrite` | `multi`) and a `contextualize()` that is deliberately *not* in it. Expansion goes **inside** `search()` as a `transform=` parameter, fanning N queries through `RETRIEVERS[mode]` and fusing with the `rrf()` that already exists. Contextualisation stays **above** `search()`, in `answer_question()`, so retrieval never learns what a conversation is.

**Tech Stack:** Python 3.12, `uv`, pydantic / pydantic-settings, Qdrant, OpenAI `gpt-4o-mini` via the existing `app/generation/llm.py`. **No new dependency** — the first step since 13 to add nothing to `pyproject.toml`.

**Spec:** [`docs/superpowers/specs/2026-09-13-query-transforms-design.md`](../specs/2026-09-13-query-transforms-design.md)

## Global Constraints

- **Quality gates before every commit:** `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest`. All four must pass.
- **`mypy` runs in strict mode on `app/`.** Every new function is fully annotated. `scripts/` is not type-checked but is still linted.
- **Ruff line length is 100.** Long prompt strings get `# noqa: E501` with a reason, as `llm.py:SYSTEM_PROMPT` already does.
- **No new dependency.** Nothing is added to `pyproject.toml` in this step.
- **`data/eval/questions.jsonl` is frozen.** It has not changed since step 10 and does not change here. The conversational fixture is a **new, separate file**.
- **Every unit test runs with no network, no API key, no model download.** `llm`, `client`, `embedder` and `index` are all injectable.
- **`RERANK_MODEL` stays empty** and `RETRIEVAL_MODE` stays `dense`. Steps 16 and 17 decided those; nothing here re-opens them.
- **Each benchmark row is appended to `data/eval/results.jsonl`** with the git commit that produced it. Never edit a past row.
- **Commit style:** lowercase conventional commits scoped to the package — `feat(retrieval):`, `chore(evaluation):`, `docs:`.
- **`.env.example` comments are in French.** New settings get French comments to match the file they live in.
- **Windows:** every script already calls `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`. New scripts do too.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `app/retrieval/transform.py` | **create** | `TRANSFORMS` registry, `expand()`, `contextualize()`, `parse_queries()`, the three prompts |
| `app/generation/llm.py` | modify | gains the canonical `Completer` type alias |
| `app/generation/answer.py` | modify | imports `Completer` instead of defining it; gains `history=`, `transform=`, `transform_n=` |
| `app/retrieval/search.py` | modify | gains `transform=`, `transform_n=`, `llm=`; the fan-out and the `len == 1` branch; two corrected docstrings |
| `app/core/config.py` | modify | three settings: `query_transform`, `multi_query_n`, `history_turns` |
| `app/evaluation/dataset.py` | modify | `EvalConversation`; `load_dataset(model=)` |
| `app/evaluation/benchmark.py` | modify | `SUMMARY_COLUMNS` gains `transform` |
| `scripts/search.py` | modify | `--transform`, `--transform-n` |
| `scripts/ask.py` | modify | `--transform`, `--transform-n`, `--history` |
| `scripts/benchmark.py` | modify | `--transform`, `--transform-n`, `RecordingLLM`, config keys, per-question recording |
| `scripts/validate_dataset.py` | modify | `--conversations` |
| `scripts/benchmark_conversations.py` | **create** | the `conv-raw` / `conv-rewrite` runs |
| `data/eval/conversations.jsonl` | **create** | ~10 referential follow-ups with ground truth |
| `tests/test_retrieval_transform.py` | **create** | the registry, the parser, the fallback, `contextualize` |
| `tests/test_retrieval_search.py` | modify | the `transform=` wiring tests |
| `tests/test_generation_answer.py` | modify | the `history=` threading tests |
| `tests/test_evaluation_dataset.py` | modify | `EvalConversation` and `load_dataset(model=)` |
| `.env.example` | modify | the three settings, French comments |
| `README.md`, `docs/roadmap.md` | modify | phases 7-8, current state, results table |

---

### Task 1: `transform.py` — the registry, the parser, the fallback

**Files:**
- Create: `app/retrieval/transform.py`
- Modify: `app/generation/llm.py` (add the `Completer` alias), `app/generation/answer.py:22` (import it instead of defining it)
- Modify: `app/core/config.py` (three settings), `.env.example`
- Test: `tests/test_retrieval_transform.py`

**Interfaces:**
- Consumes: `app.core.config.Settings` / `get_settings`; `app.generation.llm.complete`.
- Produces:
  - `Completer = Callable[..., tuple[str, dict[str, int]]]` in `app/generation/llm.py`
  - `TRANSFORMS: dict[str, Transform]` with keys `"rewrite"` and `"multi"`
  - `parse_queries(text: str, *, limit: int) -> list[str]`
  - `expand(query: str, *, transform: str, n: int | None = None, settings: Settings | None = None, llm: Completer | None = None) -> list[str]`
  - `contextualize(question: str, history: Sequence[Mapping[str, str]], *, settings: Settings | None = None, llm: Completer | None = None) -> str`
  - `Settings.query_transform: str = ""`, `Settings.multi_query_n: int = 3`, `Settings.history_turns: int = 4`

- [ ] **Step 1: Add the three settings**

In `app/core/config.py`, after the `rerank_candidates` field and before `corpus_dir`:

```python
    # Steps 18-19. Empty means off: unchanged behaviour until a measurement
    # earns the change. rewrite | multi — compared at step 19, table in the README.
    query_transform: str = ""
    # How many queries `multi` retrieves with, the original included. The
    # original is always kept, so n=1 is exactly today's behaviour.
    multi_query_n: int = 3
    # How many conversation turns reach the rewriter. An unbounded history is a
    # prompt that grows until it breaks, and the turn that disambiguates a
    # follow-up is almost always the previous one.
    history_turns: int = 4
```

- [ ] **Step 2: Document them in `.env.example`**

Append to `.env.example` (French, matching the file):

```bash
# Transformations de requete (etapes 18-19)
# Vide = desactive. rewrite | multi — compares a l'etape 19, table dans le README.
QUERY_TRANSFORM=
# Nombre de requetes utilisees par `multi`, la requete originale incluse.
# L'originale est toujours conservee : n=1 equivaut au comportement actuel.
MULTI_QUERY_N=3
# Nombre de tours de conversation transmis au reecriveur.
HISTORY_TURNS=4
```

- [ ] **Step 3: Move the `Completer` alias into `llm.py`**

In `app/generation/llm.py`, after the imports:

```python
from collections.abc import Callable
from typing import Any

from app.ingestion.embed import build_client

# The shape every caller of `complete` may substitute: the tests inject one, and
# so do `answer_question` and `expand`. Defined here, beside the only real
# implementation, so the three call sites cannot drift into three aliases.
Completer = Callable[..., tuple[str, dict[str, int]]]
```

In `app/generation/answer.py`, delete the local `Completer = Callable[..., tuple[str, dict[str, int]]]` line and import it instead:

```python
from app.generation.llm import SYSTEM_PROMPT, USER_TEMPLATE, Completer, complete
```

Leave `Retriever = Callable[..., list[ScoredChunk]]` in `answer.py` untouched.

- [ ] **Step 4: Run the gates to confirm the move broke nothing**

Run: `uv run mypy app && uv run pytest -q`
Expected: PASS, unchanged test count.

- [ ] **Step 5: Write the failing tests**

Create `tests/test_retrieval_transform.py`:

```python
"""Query transforms with a fake model: no network, no key, no spend."""

from typing import Any

import pytest

from app.core.config import Settings
from app.retrieval.transform import (
    TRANSFORMS,
    contextualize,
    expand,
    parse_queries,
)

SETTINGS = Settings(_env_file=None, generation_model="test-model", multi_query_n=3, history_turns=4)


class FakeLLM:
    """Answers with a canned completion and records how it was called."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, system: str, user: str, **kwargs: Any) -> tuple[str, dict[str, int]]:
        self.calls.append({"system": system, "user": user, **kwargs})
        return self.text, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


# --- the parser ------------------------------------------------------------


def test_plain_lines_become_queries() -> None:
    assert parse_queries("first query\nsecond query", limit=5) == ["first query", "second query"]


def test_numbered_and_bulleted_ordinals_are_stripped() -> None:
    text = "1. how to configure cors\n2) enabling cors middleware\n- cors origins setting\n* cors"
    assert parse_queries(text, limit=5) == [
        "how to configure cors",
        "enabling cors middleware",
        "cors origins setting",
        "cors",
    ]


def test_surrounding_quotes_are_stripped() -> None:
    assert parse_queries('"how do dependencies work"', limit=1) == ["how do dependencies work"]


def test_blank_lines_are_dropped() -> None:
    assert parse_queries("\n\nreal query\n   \n", limit=5) == ["real query"]


def test_a_preamble_line_ending_in_a_colon_is_dropped() -> None:
    """The single most common way a chatty model poisons the query list."""
    assert parse_queries("Here are three queries:\nfirst\nsecond", limit=5) == ["first", "second"]


def test_the_list_is_capped_at_the_limit() -> None:
    assert parse_queries("a\nb\nc\nd", limit=2) == ["a", "b"]


def test_output_with_nothing_usable_parses_to_an_empty_list() -> None:
    assert parse_queries("   \n\n  ", limit=3) == []


# --- expand() --------------------------------------------------------------


def test_rewrite_returns_exactly_one_query() -> None:
    llm = FakeLLM("how to configure memory limits for docker containers")
    assert expand("et pour docker ?", transform="rewrite", settings=SETTINGS, llm=llm) == [
        "how to configure memory limits for docker containers"
    ]


def test_multi_always_keeps_the_original_first() -> None:
    """N phrasings that all drift is how expansion loses ground the raw query held."""
    llm = FakeLLM("fastapi security authentication\nfastapi oauth2 authentication")
    queries = expand("How does auth work?", transform="multi", n=3, settings=SETTINGS, llm=llm)
    assert queries[0] == "How does auth work?"
    assert queries == [
        "How does auth work?",
        "fastapi security authentication",
        "fastapi oauth2 authentication",
    ]


def test_multi_asks_for_one_fewer_than_n_because_the_original_counts() -> None:
    llm = FakeLLM("a\nb\nc\nd\ne")
    assert len(expand("q", transform="multi", n=3, settings=SETTINGS, llm=llm)) == 3


def test_multi_at_n_one_is_todays_behaviour_and_makes_no_call() -> None:
    llm = FakeLLM("unused")
    assert expand("q", transform="multi", n=1, settings=SETTINGS, llm=llm) == ["q"]
    assert llm.calls == []


def test_multi_drops_a_paraphrase_identical_to_the_original() -> None:
    llm = FakeLLM("q\nsomething else")
    assert expand("q", transform="multi", n=3, settings=SETTINGS, llm=llm) == ["q", "something else"]


def test_multi_deduplicates_repeated_paraphrases() -> None:
    llm = FakeLLM("same\nsame\nother")
    assert expand("q", transform="multi", n=5, settings=SETTINGS, llm=llm) == ["q", "same", "other"]


def test_garbage_output_falls_back_to_the_original_query() -> None:
    """An API hiccup mid-benchmark must degrade to today's behaviour, never zero a row."""
    assert expand("q", transform="rewrite", settings=SETTINGS, llm=FakeLLM("")) == ["q"]
    assert expand("q", transform="multi", n=3, settings=SETTINGS, llm=FakeLLM("  \n ")) == ["q"]


def test_an_unknown_transform_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="hyde"):
        expand("q", transform="hyde", settings=SETTINGS, llm=FakeLLM("x"))


def test_an_empty_query_is_rejected_before_any_call() -> None:
    llm = FakeLLM("x")
    with pytest.raises(ValueError, match="empty"):
        expand("   ", transform="rewrite", settings=SETTINGS, llm=llm)
    assert llm.calls == []


def test_n_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        expand("q", transform="multi", n=0, settings=SETTINGS, llm=FakeLLM("x"))


def test_n_defaults_to_the_setting() -> None:
    llm = FakeLLM("a\nb\nc\nd")
    settings = Settings(_env_file=None, generation_model="test-model", multi_query_n=2)
    assert expand("q", transform="multi", settings=settings, llm=llm) == ["q", "a"]


def test_the_registry_holds_exactly_the_two_measured_transforms() -> None:
    assert sorted(TRANSFORMS) == ["multi", "rewrite"]


# --- contextualize() -------------------------------------------------------


def test_empty_history_returns_the_question_and_makes_no_call() -> None:
    """Paying 500-1500 ms to rewrite a first-turn question into itself is the
    failure this parameter would otherwise ship on every single-turn request."""
    llm = FakeLLM("unused")
    assert contextualize("How does auth work?", [], settings=SETTINGS, llm=llm) == (
        "How does auth work?"
    )
    assert llm.calls == []


def test_a_follow_up_is_resolved_against_the_history() -> None:
    llm = FakeLLM("How are memory limits configured for Docker containers?")
    history = [
        {"role": "user", "content": "Comment limiter la memoire d'un container Kubernetes ?"},
        {"role": "assistant", "content": "Via resources.limits.memory [1]."},
    ]
    assert contextualize("et pour docker ?", history, settings=SETTINGS, llm=llm) == (
        "How are memory limits configured for Docker containers?"
    )


def test_only_the_last_history_turns_are_sent() -> None:
    llm = FakeLLM("resolved")
    history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    contextualize("and that?", history, settings=SETTINGS, llm=llm)
    sent = llm.calls[0]["user"]
    assert "turn 9" in sent
    assert "turn 5" not in sent


def test_an_unusable_rewrite_falls_back_to_the_raw_question() -> None:
    llm = FakeLLM("")
    history = [{"role": "user", "content": "How does auth work?"}]
    assert contextualize("and oauth?", history, settings=SETTINGS, llm=llm) == "and oauth?"


def test_contextualize_rejects_an_empty_question() -> None:
    with pytest.raises(ValueError, match="empty"):
        contextualize("  ", [{"role": "user", "content": "x"}], settings=SETTINGS, llm=FakeLLM("y"))
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_transform.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.retrieval.transform'`

- [ ] **Step 7: Write `transform.py`**

Create `app/retrieval/transform.py`:

```python
"""Work on the question, not on the index.

Step 17 measured the alternative and closed it: a cross-encoder reorders a pool
and is capped by that pool's recall, and every pool on this corpus saturates by
depth 20. A transform changes the *query*, so it can retrieve a document no pool
ever held. That is the one lever left.

A transform is a ``Transform``: it takes a query and returns a list of queries.
``rewrite`` returns one, replacing the original; ``multi`` returns several with
the original kept first. Parsing, capping, de-duplication and the fallback to
the original happen once, here, for every entry — the only thing that genuinely
differs between them is the prompt.

``contextualize`` lives in this file because it is a query transform by nature,
and is deliberately **not** a ``TRANSFORMS`` key: it takes a conversation and
returns a single string, so it does not fit a registry whose contract is
``str -> list[str]``, and it must never be reachable from ``search()``. Retrieval
does not get to learn what a conversation is — that dependency would propagate
into step 23's cache key and step 25's endpoint.
"""

import re
from collections.abc import Callable, Mapping, Sequence

from app.core.config import Settings, get_settings
from app.generation.llm import Completer, complete

Transform = Callable[[str, int, Settings, Completer], list[str]]

# Mirrors search.RETRIEVERS, rerank.RERANKERS and chunk.STRATEGIES: the registry
# is how this project compares N variants and promotes a winner. One at a time,
# by decision — QUERY_TRANSFORM takes one value and there is no chaining here.
TRANSFORMS: dict[str, Transform] = {}

# The corpus is English and the evaluation set asks in French, so a rewriter that
# also translates is the realistic transform rather than a separate feature. It
# does mean the `rewrite-standalone` row confounds reformulation with
# translation; task 3 records that and names the run that would separate them.
REWRITE_SYSTEM = """\
You rewrite a user's question into a single search query for a documentation search engine.

Rules:
- Output exactly one line: the rewritten query. No preamble, no numbering, no quotes, no explanation.
- Keep every proper noun, symbol, error code and number from the question.
- Prefer the vocabulary technical documentation uses.
- Write the query in English even when the question is not."""  # noqa: E501 — one rule per line, wrapped is worse

MULTI_SYSTEM = """\
You generate alternative phrasings of a question for a documentation search engine.

Rules:
- Output exactly {n} lines, one query per line. No preamble, no numbering, no quotes, no explanation.
- Each line must be a complete, standalone search query.
- Vary the vocabulary: use the words technical documentation would use, not only the user's.
- Keep every proper noun, symbol, error code and number from the question.
- Write the queries in English even when the question is not."""  # noqa: E501 — one rule per line, wrapped is worse

# Deliberately does NOT translate, unlike REWRITE_SYSTEM. This transform is
# measured as a delta (conv-raw against conv-rewrite), so it must do exactly one
# thing: resolve the reference. Folding translation in would make the delta
# measure two changes at once, which is the mistake llm.py's version marker
# exists to prevent.
CONTEXTUALIZE_SYSTEM = """\
You rewrite a follow-up question into a standalone one, using the conversation before it.

Rules:
- Output exactly one line: the standalone question. No preamble, no quotes, no explanation.
- Resolve every pronoun and every ellipsis ("and for X?", "what about that?") against the conversation.
- Keep the follow-up's own language. Do not translate.
- If the follow-up already stands alone, return it unchanged."""  # noqa: E501 — one rule per line, wrapped is worse

CONTEXTUALIZE_TEMPLATE = """Conversation:
{history}

Follow-up: {question}"""

# "1. ", "2) ", "- ", "* ", "• " — every way a model numbers a list it was asked
# not to number.
ORDINAL = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def parse_queries(text: str, *, limit: int) -> list[str]:
    """One query per line, cleaned, at most ``limit`` of them.

    Line-oriented rather than JSON on purpose: structured output would mean
    adding a ``response_format`` parameter to ``complete()``, the one function in
    the project that talks to a model, kept to one signature so that swapping
    providers is editing its body. The failure this would protect against is
    handled by ``expand``'s fallback, which is needed either way.
    """
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = ORDINAL.sub("", line).strip().strip("\"'").strip()
        # A trailing colon is a preamble ("Here are three queries:"), not a
        # query. It is the single most common way a chatty model poisons a list.
        if cleaned and not cleaned.endswith(":"):
            queries.append(cleaned)
    return queries[:limit]


def _rewrite(query: str, n: int, settings: Settings, llm: Completer) -> list[str]:
    """One query, replacing the original. ``n`` is ignored: a rewrite is a rewrite."""
    text, _ = llm(REWRITE_SYSTEM, query, model=settings.generation_model)
    return parse_queries(text, limit=1)


TRANSFORMS["rewrite"] = _rewrite


def _multi(query: str, n: int, settings: Settings, llm: Completer) -> list[str]:
    """``n`` queries, the original first and always kept.

    Keeping the original makes the raw ranking a floor that fusion can only build
    on: N phrasings that all drift the same way is how an expansion loses ground
    the user's own words already held. It also makes n=1 exactly today's
    behaviour, which is a free sanity check.
    """
    if n < 2:
        return [query]
    text, _ = llm(MULTI_SYSTEM.format(n=n - 1), query, model=settings.generation_model)
    # dict.fromkeys de-duplicates while preserving order; a model asked for three
    # phrasings returns the same one twice more often than you would hope.
    paraphrases = dict.fromkeys(parse_queries(text, limit=n - 1))
    paraphrases.pop(query, None)
    return [query, *paraphrases]


TRANSFORMS["multi"] = _multi


def expand(
    query: str,
    *,
    transform: str,
    n: int | None = None,
    settings: Settings | None = None,
    llm: Completer | None = None,
) -> list[str]:
    """The queries ``transform`` wants to retrieve with. Never empty.

    ``llm`` is injectable so every unit test runs with no network and no key —
    the pattern ``answer_question`` already uses.

    **A transform that produces nothing usable returns ``[query]``.** An API
    hiccup or a chatty preamble in the middle of a 38-question benchmark must
    degrade to exactly today's behaviour, never zero a question. The fallback is
    not silent: ``scripts/benchmark.py`` records the raw completion per question,
    so a row where it fired says so on inspection.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}; have {sorted(TRANSFORMS)}")
    if not query.strip():
        # The empty string expands into plausible-looking garbage and then
        # retrieves against it, which search() already refuses one layer down.
        raise ValueError("query is empty")
    settings = settings or get_settings()
    n = n or settings.multi_query_n
    if n < 1:
        raise ValueError(f"transform_n must be at least 1, got {n}")
    return TRANSFORMS[transform](query, n, settings, llm or complete) or [query]


def contextualize(
    question: str,
    history: Sequence[Mapping[str, str]],
    *,
    settings: Settings | None = None,
    llm: Completer | None = None,
) -> str:
    """``"et pour docker ?"`` plus what came before it, as one standalone question.

    ``history`` is a sequence of ``{"role", "content"}`` mappings — OpenAI's own
    message shape, because it is what ``complete()`` already speaks and what step
    25's endpoint will receive off the wire.

    An empty history returns the question unchanged **without an LLM call**.
    Every single-turn question in the project takes that path, and paying a
    billed call to rewrite a question into itself is the failure this function
    would otherwise ship by default.
    """
    if not question.strip():
        raise ValueError("question is empty")
    if not history:
        return question
    settings = settings or get_settings()
    turns = list(history)[-settings.history_turns :]
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    text, _ = (llm or complete)(
        CONTEXTUALIZE_SYSTEM,
        CONTEXTUALIZE_TEMPLATE.format(history=transcript, question=question),
        model=settings.generation_model,
    )
    resolved = parse_queries(text, limit=1)
    return resolved[0] if resolved else question
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_transform.py -q`
Expected: PASS, 24 tests.

- [ ] **Step 9: Run the full gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add app/retrieval/transform.py app/generation/llm.py app/generation/answer.py \
        app/core/config.py .env.example tests/test_retrieval_transform.py
git commit -m "feat(retrieval): add the query transforms registry

rewrite and multi behind one TRANSFORMS registry, with the parser, the cap,
the de-duplication and the fallback to the original owned once for both.
contextualize lives here too and is deliberately not a registry key: it takes
a conversation, and search() must never learn what one is.

A transform that produces nothing usable returns the original query, so an API
hiccup mid-benchmark degrades to today's behaviour instead of zeroing a row.

Completer moves to llm.py so answer.py and transform.py share one alias."
```

---

### Task 2: Wire `transform=` into `search()`

**Files:**
- Modify: `app/retrieval/search.py` (module docstring, `search()` docstring and signature, the fan-out)
- Test: `tests/test_retrieval_search.py` (append a new section)

**Interfaces:**
- Consumes: `expand` and `TRANSFORMS` from Task 1; `Completer` from `app/generation/llm.py`.
- Produces: `search(..., transform: str | None = None, transform_n: int | None = None, llm: Completer | None = None)`. `transform=None` reads `QUERY_TRANSFORM`; `transform=""` forces it off. Behaviour with no transform is **byte-identical** to today.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_search.py`. Add `TRANSFORMS` and `Settings` to the existing imports at the top of the file:

```python
from app.retrieval.transform import TRANSFORMS
```

Then append this section at the end of the file:

```python
# --- query transforms (steps 18-19) ---------------------------------------


def three_queries(query: str, n: int, settings: Settings, llm: Any) -> list[str]:
    return [query, f"{query} rephrased", f"{query} again"]


def one_query(query: str, n: int, settings: Settings, llm: Any) -> list[str]:
    return [f"{query} rewritten"]


@pytest.fixture
def fake_transforms() -> Iterator[None]:
    TRANSFORMS["triple"] = three_queries
    TRANSFORMS["single"] = one_query
    yield
    del TRANSFORMS["triple"]
    del TRANSFORMS["single"]


def test_no_transform_leaves_search_exactly_as_it_was() -> None:
    """The four existing callers stay untouched only if this holds, for every mode."""
    assert [s.rank for s in run(top_k=3)] == [1, 2, 3]
    assert [s.score for s in run(top_k=3)] == [pytest.approx(0.9 - 0.1 * i) for i in range(3)]
    lexical = run("error", mode="lexical", index=bm25_index("error one", "error two"), top_k=2)
    assert [s.rank for s in lexical] == [1, 2]


def test_a_transform_retrieves_once_per_query(fake_transforms: None) -> None:
    embedder = FakeEmbedder()
    run(embedder=embedder, transform="triple", top_k=3)
    assert embedder.calls == [
        "how do dependencies work",
        "how do dependencies work rephrased",
        "how do dependencies work again",
    ]


def test_a_single_query_transform_does_not_pass_through_rrf(fake_transforms: None) -> None:
    """rrf() over one ranking preserves the order but overwrites every score with
    1/(k+rank). Routing `rewrite` through it would replace cosines with fusion
    constants and make top_score incomparable to every dense row in the history."""
    embedder = FakeEmbedder()
    results = run(embedder=embedder, transform="single", top_k=3)
    assert embedder.calls == ["how do dependencies work rewritten"]
    assert results[0].score == pytest.approx(0.9)


def test_a_chunk_found_by_every_query_outranks_one_found_by_a_single_query(
    fake_transforms: None,
) -> None:
    """A test asserting only "five chunks came back" passes against a transform
    that expands into three copies of the same query."""

    class PerQueryClient(FakeClient):
        """Chunk 0 comes back for every query; chunk 1 only for the first."""

        def query_points(self, **kwargs: Any) -> FakeResponse:
            self.calls.append(kwargs)
            shared = make_payload(0)
            if len(self.calls) == 1:
                return FakeResponse([FakeHit(0.5, make_payload(1)), FakeHit(0.4, shared)])
            return FakeResponse([FakeHit(0.9, shared)])

    results = run(client=PerQueryClient(), transform="triple", top_k=2)
    assert results[0].chunk.chunk_index == 0
    assert results[1].chunk.chunk_index == 1


def test_the_transform_defaults_to_the_setting(fake_transforms: None) -> None:
    embedder = FakeEmbedder()
    run(
        embedder=embedder,
        settings=Settings(_env_file=None, qdrant_collection="chunks", query_transform="single"),
        top_k=1,
    )
    assert embedder.calls == ["how do dependencies work rewritten"]


def test_an_empty_transform_argument_forces_it_off(fake_transforms: None) -> None:
    """`--transform ""` must override a configured QUERY_TRANSFORM, or an
    untransformed baseline becomes unrunnable once the default flips."""
    embedder = FakeEmbedder()
    run(
        embedder=embedder,
        settings=Settings(_env_file=None, qdrant_collection="chunks", query_transform="single"),
        transform="",
        top_k=1,
    )
    assert embedder.calls == ["how do dependencies work"]


def test_an_unknown_transform_is_rejected_by_name(fake_transforms: None) -> None:
    with pytest.raises(ValueError, match="hyde"):
        run(transform="hyde")


def test_a_transform_composes_with_a_reranker(fake_reranker: None, fake_transforms: None) -> None:
    """transform= and rerank= are orthogonal; both must survive being combined."""
    results = run(transform="triple", rerank="reverse", rerank_candidates=10, top_k=3)
    assert len(results) == 3
    assert results[0].rerank_score is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_retrieval_search.py -q -k transform`
Expected: FAIL — `TypeError: search() got an unexpected keyword argument 'transform'`

- [ ] **Step 3: Correct the two stale docstrings**

In `app/retrieval/search.py`, replace the module docstring:

```python
"""Query the index: a question in, ranked chunks out.

``search()`` is the seam the rest of the project is built on. Step 08 calls it,
step 11 benchmarks it, and steps 14-19 replace its internals — hybrid retrieval,
RRF, reranking, query expansion — by adding parameters to it rather than by
composing a pipeline in each of its four callers. Step 16 settled why: a stage
every caller must remember to compose is a stage one caller eventually does not,
and a pipeline that expands in the benchmark but not in ``scripts/ask.py`` is a
measurement of something nobody ships.

Step 18 made one line of this docstring false and it is corrected rather than
worked around: this module *does* call a language model now, when ``transform``
is set. It still never calls one to *generate an answer* — that is
``app/generation/`` — and it still never learns what a conversation is.
Contextual rewriting lives in ``answer_question``, where the conversation does.
"""
```

- [ ] **Step 4: Add the imports and the parameters**

In `app/retrieval/search.py`, add to the imports:

```python
from app.generation.llm import Completer
from app.retrieval.transform import expand
```

Add three parameters to `search()`, after `rerank_candidates`:

```python
    transform: str | None = None,
    transform_n: int | None = None,
```

and after `index`:

```python
    llm: Completer | None = None,
```

Add to the `search()` docstring, after the `rerank` paragraph:

```
    ``transform`` names a query transform from ``transform.TRANSFORMS`` and is
    orthogonal to both ``mode`` and ``rerank``: it changes the *query*, then the
    chosen retriever runs once per query produced and ``rrf()`` fuses the
    rankings. ``None`` reads ``QUERY_TRANSFORM``, and the empty string forces it
    off, which is how an untransformed baseline stays runnable once the default
    flips. ``transform_n`` is how many queries ``multi`` produces, the original
    included.

    Contextual rewriting is deliberately not reachable from here. It needs a
    conversation, and a ``search()`` that knows what one is would push that
    dependency into step 23's cache key and step 25's endpoint. It lives in
    ``answer_question``.
```

- [ ] **Step 5: Implement the fan-out**

In `search()`, replace the single `results = RETRIEVERS[mode](...)` call with the block below. Everything before it (validation, `settings`, `mode`, `collection`, the client/embedder/index construction, `model` and `depth`) is unchanged, and the `if not model: return results` tail is unchanged.

```python
    transform = settings.query_transform if transform is None else transform
    queries = (
        expand(query, transform=transform, n=transform_n, settings=settings, llm=llm)
        if transform
        else [query]
    )

    def retrieve(text: str) -> list[ScoredChunk]:
        return RETRIEVERS[mode](
            text,
            top_k=depth,
            candidates=candidates or settings.retrieval_candidates,
            rrf_k=rrf_k or settings.rrf_k,
            filters=filters,
            collection=collection,
            client=client,
            embedder=embedder,
            index=index,
        )

    if len(queries) == 1:
        # Not an optimisation — correctness. rrf() over a single ranking keeps
        # its order but overwrites every score with 1/(k+rank), which would
        # replace cosine similarities with fusion constants for no change in
        # ranking at all, and make this row's top_score incomparable to every
        # dense row already in the history.
        results = retrieve(queries[0])
    else:
        # In hybrid mode this is the second rrf() in the path: once per query to
        # fuse the two branches, then once across queries. That composes because
        # RRF reads ranks and never scores — fusing fused *ranks* is well
        # defined in a way that fusing fused scores would not be.
        results = rrf(
            [retrieve(text) for text in queries], k=rrf_k or settings.rrf_k, top_k=depth
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_search.py -q`
Expected: PASS — the new transform tests plus every pre-existing one, unchanged.

- [ ] **Step 7: Add `--transform` to `scripts/search.py`**

Add the import:

```python
from app.retrieval.transform import TRANSFORMS  # noqa: E402
```

Add the arguments after `--rerank-candidates`:

```python
    parser.add_argument(
        "--transform",
        choices=["", *sorted(TRANSFORMS)],
        help='query transform applied before retrieval; default: QUERY_TRANSFORM, "" is off',
    )
    parser.add_argument(
        "--transform-n", type=int, help="queries `multi` produces, original included; MULTI_QUERY_N"
    )
```

and thread them into the `search(...)` call:

```python
        transform=args.transform,
        transform_n=args.transform_n,
```

- [ ] **Step 8: Run the gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add app/retrieval/search.py scripts/search.py tests/test_retrieval_search.py
git commit -m "feat(retrieval): expand behind the search() seam, orthogonal to mode

transform= runs the chosen retriever once per query produced and fuses the
rankings with the rrf() step 16 already shipped. A single-query transform
deliberately skips rrf(): fusing one ranking would overwrite every cosine with
1/(k+rank) for no change in order.

Corrects the module docstring's 'No LLM here', which step 18 makes false.
Reranking still composes on top, and contextual rewriting still does not live
here."
```

---

### Task 3: The `rewrite-standalone` row

**Files:**
- Modify: `scripts/benchmark.py` (arguments, `RecordingLLM`, config keys, per-question recording)
- Modify: `app/evaluation/benchmark.py:76-89` (`SUMMARY_COLUMNS` and `summarise`)
- Modify: `data/eval/results.jsonl` (appended by the run)

**Interfaces:**
- Consumes: `search(transform=...)` from Task 2; `parse_queries` is *not* needed — the raw completion is recorded verbatim.
- Produces: a `rewrite-standalone` row in `data/eval/results.jsonl`, whose `config` carries `transform` and `transform_n` and whose `per_question` rows carry `transform_output` and `transform_ms`.

- [ ] **Step 1: Add the `transform` column to the summary table**

In `app/evaluation/benchmark.py`, add `"transform"` to `SUMMARY_COLUMNS` immediately after `"rerank"`:

```python
SUMMARY_COLUMNS = (
    "label",
    "mode",
    "rerank",
    "transform",
    "strategy",
    ...
)
```

and in `summarise()`, immediately after the `rerank` cell:

```python
            # "-" rather than "": every row written before step 18 genuinely had
            # no transform, and an empty cell reads as a missing value.
            str(row["config"].get("transform") or "-"),
```

- [ ] **Step 2: Add a test for the new column**

Append to `tests/test_evaluation_benchmark.py`:

```python
def test_a_row_written_before_step_18_reports_no_transform() -> None:
    """An empty cell reads as a missing value; "-" reads as "there wasn't one"."""
    history = [
        {
            "label": "dense-sentence",
            "config": {"mode": "dense", "strategy": "sentence"},
            "aggregate": {"recall@5": 0.776},
            "per_category": {},
            "latency_p50_ms": 35.0,
        }
    ]
    [row] = summarise(history, "dense-*")
    assert row[SUMMARY_COLUMNS.index("transform")] == "-"
```

Add `SUMMARY_COLUMNS` to that file's imports from `app.evaluation.benchmark` if it is not already there.

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_evaluation_benchmark.py -q -k transform`
Expected: FAIL — `ValueError: 'transform' is not in tuple` or an `IndexError`.

- [ ] **Step 4: Run it again to verify it passes**

The edit in Step 1 already satisfies it.

Run: `uv run pytest tests/test_evaluation_benchmark.py -q`
Expected: PASS.

- [ ] **Step 5: Add the recording LLM and the flags to `scripts/benchmark.py`**

Add the imports:

```python
from app.generation.llm import complete  # noqa: E402
from app.retrieval.transform import TRANSFORMS  # noqa: E402
```

Add this class after `load_history`:

```python
class RecordingLLM:
    """`complete`, plus a note of what it said and how long it took, per question.

    Keyed on the user message because both transforms send the question verbatim
    as their user message, so the key is exact rather than order-coupled: a
    question that fails mid-run cannot shift every later row by one.

    The *raw* completion is kept, not the parsed query list. A chatty preamble or
    an empty response is then visible in the history exactly as the model
    produced it, which is what makes expand()'s fallback inspectable instead of
    silent — and it needs no second copy of the parser living out here.
    """

    def __init__(self) -> None:
        self.seen: dict[str, tuple[str, float, int]] = {}

    def __call__(self, system: str, user: str, **kwargs: Any) -> tuple[str, dict[str, int]]:
        started = time.perf_counter()
        text, usage = complete(system, user, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.seen[user] = (text, elapsed_ms, usage.get("total_tokens", 0))
        return text, usage
```

Add `import time` to the imports at the top of the file.

Add the arguments after `--rerank-candidates`:

```python
    parser.add_argument(
        "--transform",
        choices=["", *sorted(TRANSFORMS)],
        help='query transform applied before retrieval; default: QUERY_TRANSFORM, "" is off',
    )
    parser.add_argument(
        "--transform-n", type=int, help="queries `multi` produces, original included; MULTI_QUERY_N"
    )
```

- [ ] **Step 6: Thread the recorder through the retriever and into the rows**

Replace the `retrieve` closure's `search(...)` call to pass the recorder and the two new flags. Immediately before `def retrieve`:

```python
    transform = args.transform if args.transform is not None else settings.query_transform
    recorder = RecordingLLM() if transform else None
```

Inside `retrieve`, add to the `search(...)` call:

```python
            transform=args.transform,
            transform_n=args.transform_n,
            llm=recorder,
```

After `result = run_benchmark(...)` and before `print_result(result)`:

```python
    if recorder:
        # Merged after the fact rather than widened into run_benchmark: the
        # retriever seam is `str -> list[ScoredChunk]` and four steps are built
        # on it. Questions are keyed by text, as --oracle-filter already does.
        by_text = {question.question: question.question_id for question in questions}
        outputs = {
            by_text[text]: value for text, value in recorder.seen.items() if text in by_text
        }
        for row in result.per_question:
            if found := outputs.get(row["question_id"]):
                row["transform_output"], row["transform_ms"], row["transform_tokens"] = found
        timings = [row["transform_ms"] for row in result.per_question if "transform_ms" in row]
        tokens = [row["transform_tokens"] for row in result.per_question if "transform_tokens" in row]
        if timings:
            print(
                f"\nTransform: p50 {statistics.median(timings):.0f} ms"
                f" of {result.latency_p50_ms:.0f} ms total p50"
                f"  |  {statistics.mean(tokens):.0f} tokens/question"
            )
```

Add `import statistics` to the imports at the top of the file.

Add the two keys to the `config=` mapping passed to `run_benchmark`, after `"rerank_candidates"`:

```python
            "transform": transform or None,
            "transform_n": args.transform_n or settings.multi_query_n,
```

- [ ] **Step 7: Verify the wiring with a dry run that spends nothing**

Run: `uv run python scripts/benchmark.py --label smoke --no-save --transform "" --top-k 5`
Expected: the usual tables, no `Transform:` line, nothing appended. This confirms the untransformed path is untouched before any money is spent.

- [ ] **Step 8: Start Qdrant and run the measurement**

```bash
docker compose up -d
uv run python scripts/benchmark.py --label "rewrite-standalone" --transform rewrite \
  --compare "dense-sentence-doctype"
```

Expected: a delta table against 0.776, a `Transform: p50 ... ms` line, and `appended to data/eval/results.jsonl`.

- [ ] **Step 9: Record the finding**

Read the comparison table and write down, for the commit message and for Task 7:

- Recall@5, Recall@10, MRR, NDCG@5 and the delta against 0.776;
- the per-category table, specifically whether `code` and `exact` moved;
- transform p50 against total p50;
- two or three `transform_output` values read straight out of the new row, so the prose can say what the rewriter actually did.

**This row is recorded whatever it says.** There is no stopping condition: a regression here does not invalidate multi-query, because `rewrite` *replaces* the original query and `multi` *keeps* it.

**One confound to state plainly in the commit and the README:** `REWRITE_SYSTEM` both reformulates and translates to English, so a gain here has two possible causes. If and only if this row wins, a follow-up row `rewrite-standalone-no-translate` — the same prompt with the last rule removed — separates them. Do not run it if the row loses; there would be nothing to attribute.

- [ ] **Step 10: Commit**

```bash
git add app/evaluation/benchmark.py scripts/benchmark.py \
        tests/test_evaluation_benchmark.py data/eval/results.jsonl
git commit -m "chore(evaluation): measure standalone query rewriting

rewrite-standalone against the 0.776 dense-sentence-doctype baseline:
Recall@5 <VALUE> (<DELTA>), transform p50 <VALUE> ms of <VALUE> ms total.

<One sentence on what the per-category table did.>

The row confounds reformulation with translation — REWRITE_SYSTEM does both.
<Named follow-up run, or 'Not worth separating: the row loses.'>

Each run records the raw completion and its latency per question, so the
fallback to the original query is inspectable rather than silent."
```

---

### Task 4: `contextualize()` reaches the answer pipeline

**Files:**
- Modify: `app/generation/answer.py` (signature, the contextualisation call, the docstring)
- Modify: `scripts/ask.py` (`--history`, `--transform`, `--transform-n`)
- Test: `tests/test_generation_answer.py`

**Interfaces:**
- Consumes: `contextualize` from Task 1; `search(transform=...)` from Task 2.
- Produces: `answer_question(question, *, history: Sequence[Mapping[str, str]] | None = None, transform: str | None = None, transform_n: int | None = None, ...)`. The retriever receives the **standalone** query, never the raw follow-up.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generation_answer.py`:

```python
# --- conversation history (step 18) ----------------------------------------

HISTORY = [
    {"role": "user", "content": "Comment limiter la memoire d'un container Kubernetes ?"},
    {"role": "assistant", "content": "Via resources.limits.memory [1]."},
]


def test_the_retriever_receives_the_standalone_query_not_the_follow_up() -> None:
    """The whole point of step 18: "et pour docker ?" retrieves nothing on its own."""
    retriever = FakeRetriever()
    ask(
        "et pour docker ?",
        history=HISTORY,
        retriever=retriever,
        llm=FakeLLM("Set a memory limit [1]."),
    )
    assert retriever.calls[0]["query"] == "Set a memory limit [1]."


def test_no_history_means_no_rewriting_call_at_all() -> None:
    """Every single-turn question in the project takes this path."""
    llm = FakeLLM("Use Depends() [1].")
    retriever = FakeRetriever()
    ask("How do dependencies work?", retriever=retriever, llm=llm)
    assert retriever.calls[0]["query"] == "How do dependencies work?"
    assert len(llm.calls) == 1  # the answer only; nothing rewrote the question


def test_the_transform_is_threaded_to_the_retriever() -> None:
    retriever = FakeRetriever()
    ask("How do dependencies work?", transform="multi", transform_n=2, retriever=retriever)
    assert retriever.calls[0]["transform"] == "multi"
    assert retriever.calls[0]["transform_n"] == 2
```

Note the first test's expectation: `ask()` injects one `FakeLLM` for both the rewrite and the answer, so the rewritten query is that fake's canned text. That is deliberate — it proves the retriever got whatever `contextualize` returned rather than the raw follow-up.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generation_answer.py -q -k "history or transform or standalone"`
Expected: FAIL — `TypeError: answer_question() got an unexpected keyword argument 'history'`

- [ ] **Step 3: Extend `answer_question`**

In `app/generation/answer.py`, add the import:

```python
from app.retrieval.transform import contextualize
```

Add the parameters after `question`:

```python
    history: Sequence[Mapping[str, str]] | None = None,
    transform: str | None = None,
    transform_n: int | None = None,
```

and add `from collections.abc import Callable, Mapping, Sequence` to the imports.

Add to the docstring, after the `mode`/`rerank` paragraph:

```
    ``history`` is the conversation so far, as ``{"role", "content"}`` mappings.
    Given one, the question is resolved into a standalone query before retrieval
    — ``"et pour docker ?"`` becomes a question that means something on its own.
    This is the one place that happens: ``search()`` takes a query string and is
    never told a conversation exists.

    ``transform`` and ``transform_n`` are threaded straight through to the
    retriever, exactly as ``mode`` and ``rerank`` already are, so the measured
    winner of step 19 reaches the answer and not only the benchmark.
```

Immediately after the `settings = settings or get_settings()` / `retriever` / `llm` / `started` block, before the `retriever(...)` call:

```python
    # Before retrieval and above search(): resolving a follow-up needs the
    # conversation, and pushing that into the retrieval seam would carry it on
    # into step 23's cache key and step 25's endpoint. With no history this
    # returns the question and makes no call.
    query = contextualize(question, history or [], settings=settings, llm=llm)
```

and change the retriever call's first argument from `question` to `query`, adding the two new parameters:

```python
    chunks = retriever(
        query,
        top_k=top_k or settings.top_k,
        mode=mode,
        transform=transform,
        transform_n=transform_n,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        filters=filters,
        settings=settings,
    )
```

Leave the `USER_TEMPLATE.format(context=context, question=question)` call using **`question`**, not `query`: the model answers the question the user actually asked, and is grounded in context retrieved for the resolved one.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_generation_answer.py -q`
Expected: PASS, including every pre-existing test.

- [ ] **Step 5: Add the flags to `scripts/ask.py`**

Add the import:

```python
from app.retrieval.transform import TRANSFORMS  # noqa: E402
```

Add the arguments:

```python
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        metavar="TEXT",
        help="repeatable; alternates user, assistant, user, ... starting with user",
    )
    parser.add_argument(
        "--transform",
        choices=["", *sorted(TRANSFORMS)],
        help='query transform applied before retrieval; default: QUERY_TRANSFORM, "" is off',
    )
    parser.add_argument(
        "--transform-n", type=int, help="queries `multi` produces, original included; MULTI_QUERY_N"
    )
```

After `filters = parse_filters(args.filter)`:

```python
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": text}
        for index, text in enumerate(args.history)
    ]
```

Thread `history=history`, `transform=args.transform` and `transform_n=args.transform_n` into the `answer_question(...)` call, and `transform=args.transform`, `transform_n=args.transform_n` into the `--show-context` `search(...)` call.

Update the module docstring's usage block to mention `--history`.

- [ ] **Step 6: Run the demo transcript**

```bash
uv run python scripts/ask.py "et pour docker ?" \
  --history "Comment limiter la memoire d'un container Kubernetes ?" \
  --history "Definissez resources.limits.memory dans le manifeste [1]."
```

Expected: a grounded answer citing `fastapi:deployment/docker`. Save the full transcript for Task 7.

Also run it **without** `--history` and save that transcript too. The pair is the qualitative evidence; the contrast is the point.

- [ ] **Step 7: Run the gates**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add app/generation/answer.py scripts/ask.py tests/test_generation_answer.py
git commit -m "feat(generation): resolve a follow-up against the conversation

answer_question takes history= and resolves the question into a standalone
query before retrieval. Above search() deliberately: resolving a follow-up
needs the conversation, and search() must not carry that into step 23's cache
key or step 25's endpoint.

No history means no call at all, which is the path every single-turn question
in the project takes.

The prompt is still built from the question the user asked, grounded in context
retrieved for the resolved one."
```

---

### Task 5: The conversational fixture and its delta — `v0.8`

**Files:**
- Create: `data/eval/conversations.jsonl`
- Create: `scripts/benchmark_conversations.py`
- Modify: `app/evaluation/dataset.py` (`EvalConversation`, `load_dataset(model=)`)
- Modify: `scripts/validate_dataset.py` (`--conversations`)
- Test: `tests/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: `contextualize` (Task 1), `search` (Task 2), `run_benchmark` / `question_doc_type` (unchanged).
- Produces: `EvalConversation(EvalQuestion)` with `history: list[dict[str, str]] = []`; `load_dataset(path, *, categories=None, model: type[EvalQuestion] = EvalQuestion)`; two rows `conv-raw` and `conv-rewrite` in `data/eval/results.jsonl`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluation_dataset.py`:

```python
def test_a_conversation_carries_its_history(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    path.write_text(
        '{"question_id": "c001", "question": "et pour docker ?",'
        ' "category": "conceptual", "relevant_document_ids": ["fastapi:deployment/docker"],'
        ' "history": [{"role": "user", "content": "Et pour Kubernetes ?"}]}\n',
        encoding="utf-8",
    )
    [conversation] = load_dataset(path, model=EvalConversation)
    assert conversation.history == [{"role": "user", "content": "Et pour Kubernetes ?"}]
    assert conversation.relevant_document_ids == ["fastapi:deployment/docker"]


def test_a_conversation_inherits_the_frozen_sets_validators(tmp_path: Path) -> None:
    """One added field must not buy an exemption from the label rules."""
    path = tmp_path / "conversations.jsonl"
    path.write_text(
        '{"question_id": "c001", "question": "et pour docker ?",'
        ' "category": "conceptual", "relevant_document_ids": [], "history": []}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relevant_document_ids must not be empty"):
        load_dataset(path, model=EvalConversation)


def test_a_conversation_file_reports_the_same_line_numbered_errors(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"conversations\.jsonl:1"):
        load_dataset(path, model=EvalConversation)
```

Add `EvalConversation` to that file's imports from `app.evaluation.dataset`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluation_dataset.py -q -k conversation`
Expected: FAIL — `ImportError: cannot import name 'EvalConversation'`

- [ ] **Step 3: Extend `dataset.py`**

Add after the `EvalQuestion` class:

```python
class EvalConversation(EvalQuestion):
    """A labelled question plus the turns that came before it.

    Subclasses rather than duplicates ``EvalQuestion``: it inherits every
    validator the frozen set already enforces — a category from the ``Literal``,
    non-empty ground truth, no ground truth on an ``unanswerable`` — and a second
    loader for a file that differs by one field is how the two quietly drift.

    ``history`` uses OpenAI's ``{"role", "content"}`` shape, the same one
    ``contextualize`` and step 25's endpoint take.
    """

    history: list[dict[str, str]] = []
```

Change `load_dataset`'s signature and its one validation line:

```python
def load_dataset(
    path: Path,
    *,
    categories: Sequence[str] | None = None,
    model: type[EvalQuestion] = EvalQuestion,
) -> list[EvalQuestion]:
```

and inside the loop:

```python
            question = model.model_validate_json(line)
```

Add to the docstring:

```
    ``model`` selects the row type: ``EvalConversation`` for a file that carries
    conversation history. Everything else — the comment lines, the line-numbered
    errors, the duplicate-id check — is identical, which is the point.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluation_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Build the fixture**

Create `data/eval/conversations.jsonl`. **The construction rule, and it is the whole point of the file: every follow-up's ground-truth document must be unreachable from the follow-up alone.** A follow-up that carries its own discriminating noun tests nothing — `conv-raw` would score well and the delta would measure nothing.

Before writing a row, check the follow-up against the index:

```bash
uv run python scripts/search.py "and how do I test that?" --top-k 5
```

If the ground-truth document appears, the follow-up is too self-sufficient — make it more referential.

Ten rows, ids `c001`-`c010`, drawn from documents that exist in `data/raw/fastapi/`. Two examples in the exact required format:

```json
{"question_id": "c001", "question": "et pour docker ?", "category": "conceptual", "relevant_document_ids": ["fastapi:deployment/docker"], "history": [{"role": "user", "content": "Comment deployer une application FastAPI derriere HTTPS ?"}, {"role": "assistant", "content": "Vous placez un proxy de terminaison TLS devant l'application [1]."}], "notes": "roadmap headline; weak by construction - carries its own noun"}
{"question_id": "c002", "question": "and how do I test that?", "category": "conceptual", "relevant_document_ids": ["fastapi:tutorial/testing"], "history": [{"role": "user", "content": "How do I override a dependency for one route?"}, {"role": "assistant", "content": "Use app.dependency_overrides with the dependency as the key [1]."}], "notes": "referential: 'that' is the only link to testing"}
```

Keep `c001` and label it in `notes` as the weak case it is: it is the roadmap's headline and belongs in the file, but it is not what the delta rests on. The other nine must pass the check above.

Draw the remaining eight from these documents, all confirmed present in `data/raw/fastapi/`. For each, the history establishes the subject and the follow-up refers to it without naming it:

| target document | history establishes | follow-up refers |
|---|---|---|
| `fastapi:tutorial/testing` | overriding a dependency | "and how do I test that?" |
| `fastapi:async` | writing a route function | "when should I make it async?" |
| `fastapi:tutorial/header-params` | reading a query parameter | "can I do the same thing with a header?" |
| `fastapi:tutorial/handling-errors` | returning a response model | "what if I need to fail instead?" |
| `fastapi:tutorial/bigger-applications` | one app file with several routes | "at what point should I split it up?" |
| `fastapi:tutorial/security/oauth2-jwt` | a login route returning a token | "how do I sign those?" |
| `fastapi:tutorial/sql-databases` | a Pydantic model | "how do I persist one of those?" |
| `fastapi:deployment/server-workers` | running with uvicorn | "how do I run more than one of them?" |

Verify each `relevant_document_ids` value against the corpus before writing the row — `ls data/raw/fastapi/tutorial/` and friends — and adjust the id if a page has a different slug. Step 7 will catch any that are wrong, but catching them here is cheaper. Assign each row the category that matches what it actually asks (`conceptual`, `code`, `exact`); they do not all have to be `conceptual`.

- [ ] **Step 6: Add `--conversations` to the validator**

`scripts/validate_dataset.py` hardcodes `MIN_PER_CATEGORY = 8` and `MIN_QUESTIONS = 40`, which a ten-row fixture fails by construction. Those floors belong to the frozen 50-question set, not to this file.

Add the argument:

```python
    parser.add_argument(
        "--conversations",
        action="store_true",
        help="load the file as conversations; skips the category and count floors, "
        "which are calibrated for the frozen 50-question set",
    )
```

Import `EvalConversation`, and change the load:

```python
    questions = load_dataset(
        args.dataset, model=EvalConversation if args.conversations else EvalQuestion
    )
```

Guard the two floor checks so they are skipped under `--conversations`:

```python
    problems = validate_dataset(questions, known)
    if not args.conversations:
        problems += [
            f"category {category!r} has {counts[category]} questions, minimum {MIN_PER_CATEGORY}"
            for category in CATEGORIES
            if counts[category] < MIN_PER_CATEGORY
        ]
        if len(questions) < MIN_QUESTIONS:
            problems.append(f"{len(questions)} questions, minimum {MIN_QUESTIONS}")
```

Import `EvalQuestion` alongside `EvalConversation` from `app.evaluation.dataset`.

- [ ] **Step 7: Validate the fixture**

Run: `uv run python scripts/validate_dataset.py --dataset data/eval/conversations.jsonl --conversations`
Expected: `no problems`. Every `relevant_document_ids` entry must exist in the corpus; fix any that do not before going further.

Then confirm the frozen set is untouched:

Run: `uv run python scripts/validate_dataset.py`
Expected: `no problems`, 50 questions, unchanged.

- [ ] **Step 8: Write `scripts/benchmark_conversations.py`**

```python
"""Does resolving a follow-up against its conversation actually retrieve better?

    uv run python scripts/benchmark_conversations.py --label conv-raw --raw
    uv run python scripts/benchmark_conversations.py --label conv-rewrite

Two runs, one delta. `--raw` searches the bare follow-up; without it the
follow-up is resolved by `contextualize` first. The gap between the two rows is
the number step 18 exists to produce, and it lands in the same
`data/eval/results.jsonl` as every other row.

Separate from `scripts/benchmark.py` because a conversation is a different
shape, not a different flag: `run_benchmark`'s retriever seam is
`str -> list[ScoredChunk]`, and four steps are built on that. The history is
bound into the closure and keyed on the question text, exactly as
`--oracle-filter` already does.

Ten conversations is a capability check, not a promotion criterion. Nothing
flips QUERY_TRANSFORM on the strength of this file.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.benchmark import run_benchmark  # noqa: E402
from app.evaluation.dataset import EvalConversation, load_dataset  # noqa: E402
from app.models.chunks import ScoredChunk  # noqa: E402
from app.retrieval.search import search  # noqa: E402
from app.retrieval.transform import contextualize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import HISTORY, KS, load_history, print_comparison, print_result  # noqa: E402

DEFAULT_DATASET = Path("data/eval/conversations.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--raw", action="store_true", help="search the bare follow-up, resolving nothing"
    )
    parser.add_argument("--compare", help="print a delta table against a previous label")
    parser.add_argument("--no-save", action="store_true", help="print only, append nothing")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = get_settings()
    conversations = load_dataset(args.dataset, model=EvalConversation)
    # Keyed on the question text because run_benchmark hands the retriever a
    # string, exactly as --oracle-filter is. Exact as long as no two follow-ups
    # share a text, which the guard below enforces.
    histories = {c.question: getattr(c, "history", []) for c in conversations}
    if len(histories) != len(conversations):
        raise SystemExit("two follow-ups share the same text; the history lookup would be wrong")

    def retrieve(text: str) -> list[ScoredChunk]:
        query = text if args.raw else contextualize(text, histories[text], settings=settings)
        return search(query, top_k=args.top_k, settings=settings)

    result = run_benchmark(
        conversations,
        retrieve,
        ks=[k for k in KS if k <= args.top_k],
        label=args.label,
        config={
            "top_k": args.top_k,
            "mode": settings.retrieval_mode,
            "contextualize": not args.raw,
            "dataset": str(args.dataset),
            "collection": settings.qdrant_collection,
            "generation_model": settings.generation_model,
            "history_turns": settings.history_turns,
        },
    )
    print_result(result)

    status = 0
    if args.compare:
        status = print_comparison(result, args.compare, load_history(HISTORY))
    if not args.no_save:
        with HISTORY.open("a", encoding="utf-8") as history:
            history.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        print(f"\nappended to {HISTORY}")
    return status


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9: Run both measurements**

```bash
docker compose up -d
uv run python scripts/benchmark_conversations.py --label "conv-raw" --raw
uv run python scripts/benchmark_conversations.py --label "conv-rewrite" --compare "conv-raw"
```

Expected: `conv-raw` scores near the floor on the nine referential rows; `conv-rewrite` scores like ordinary questions. Record Recall@5 for both and the delta.

If `conv-raw` scores *well*, the fixture is too self-sufficient — go back to Step 5 and make the follow-ups more referential. That is a fixture bug, not a result.

- [ ] **Step 10: Run the gates and commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q`

```bash
git add app/evaluation/dataset.py scripts/validate_dataset.py \
        scripts/benchmark_conversations.py data/eval/conversations.jsonl \
        data/eval/results.jsonl tests/test_evaluation_dataset.py
git commit -m "chore(evaluation): measure follow-up resolution on its own fixture

conv-raw Recall@5 <VALUE> against conv-rewrite <VALUE> over 10 conversations.

The frozen 50-question set has no conversation history to score this against
and does not change here. The fixture is built to one rule: every follow-up's
ground truth must be unreachable from the follow-up alone, or conv-raw scores
well and the delta measures nothing. c001 is the roadmap's own 'et pour
docker ?' and is labelled in notes as the weak case it is.

run_benchmark is reused unchanged, with the history bound into the closure and
keyed on question text, as --oracle-filter already does.

Ten conversations is a capability check, not a promotion criterion."
```

- [ ] **Step 11: Tag**

```bash
git tag v0.8
```

---

### Task 6: The multi-query matrix and the verdict — `v0.9`

**Files:**
- Modify: `data/eval/results.jsonl` (appended by each run)
- Modify: `app/core/config.py` (only if the rule is met)
- Modify: `.env.example` (only if the rule is met)

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: five to six rows in `data/eval/results.jsonl`, and a decision on `QUERY_TRANSFORM`.

- [ ] **Step 1: Warm the embedding cache and start Qdrant**

```bash
docker compose up -d
uv run python scripts/benchmark.py --label warm --no-save --top-k 5
```

Expected: runs clean. Paraphrases are new strings and will miss the embedding cache, which is expected and cheap.

- [ ] **Step 2: Run the N sweep on dense**

```bash
uv run python scripts/benchmark.py --label "multi-n2-dense" --transform multi --transform-n 2
uv run python scripts/benchmark.py --label "multi-n3-dense" --transform multi --transform-n 3 \
  --compare "dense-sentence-doctype"
uv run python scripts/benchmark.py --label "multi-n5-dense" --transform multi --transform-n 5
```

- [ ] **Step 3: Run the hybrid pairing**

```bash
uv run python scripts/benchmark.py --label "multi-n3-hybrid" --transform multi --transform-n 3 \
  --mode hybrid --candidates 50 --rrf-k 60
```

Step 17 measured the hybrid d50 pool at **0.917** Recall@30, the largest in the project. Multi-query and hybrid are the two widening tools this project has; this row is the pairing.

- [ ] **Step 4: Run the reranked row**

```bash
uv run python scripts/benchmark.py --label "multi-n3-dense-rerank-flashrank" \
  --transform multi --transform-n 3 --rerank flashrank --rerank-candidates 30
```

Free — the model is local and already downloaded. It answers what step 17 could not: it found the reranker moved precision between categories (`code` −0.100) and blamed the model rather than the pool. A wider, differently-composed pool is the cheapest test of that attribution.

- [ ] **Step 5: Print the matrix**

```bash
uv run python scripts/benchmark.py --summary "multi-*"
uv run python scripts/benchmark.py --summary "*"
```

- [ ] **Step 6: Apply the pre-registered rule**

Fixed before the first run, in the spec. `QUERY_TRANSFORM` moves off `""` **if and only if all three hold** for the best row:

1. **Absolute gain** — Recall@5 at least **+0.03** over **0.776**.
2. **No category sacrificed** — no per-category Recall@5 down more than **0.05** against the 0.776 baseline's per-category table.
3. **Latency bounded** — p50 retrieval, transform LLM call included, under **2 000 ms**.

`rewrite` (Task 3) and `multi` compete for the same setting. If both clear it, the higher Recall@5 wins and the other is recorded as measured-and-not-promoted.

Write the three clause verdicts down explicitly — pass or fail with the number beside each. **If any clause fails, `QUERY_TRANSFORM` stays empty.** Everything still ships and the verdict is recorded exactly as steps 13, 16 and 17 recorded theirs. Widening the criterion until the number qualifies is not satisfying rule 1.

- [ ] **Step 7: Flip the default — only if all three clauses passed**

In `app/core/config.py`:

```python
    query_transform: str = "multi"
```

Replace the comment above it with the measured verdict — the numbers, not a claim. In `.env.example`, set `QUERY_TRANSFORM=multi`.

Then confirm the default actually took effect:

```bash
uv run python scripts/benchmark.py --label "default-check" --no-save --top-k 5
```

Expected: the `Transform: p50 ... ms` line appears with no `--transform` flag given.

**If any clause failed, skip this step entirely.** Leave `query_transform: str = ""` and go to Step 8.

- [ ] **Step 8: Also record the abstention finding**

From any `multi-*` row, read `aggregate.abstention_rate` and compare it to the baseline's. Multi-query paraphrases an out-of-corpus question three ways and retrieves three times the plausible-looking garbage.

Write down whether widening retrieval made refusal harder. It costs nothing — the number is already computed — and step 22 is the only consumer. Reported here, acted on there.

- [ ] **Step 9: Commit**

```bash
git add data/eval/results.jsonl app/core/config.py .env.example
git commit -m "chore(evaluation): record the multi-query matrix and the verdict

Best row <LABEL>: Recall@5 <VALUE> against the 0.776 baseline (<DELTA>),
p50 <VALUE> ms of which <VALUE> ms is the transform call.

Clause 1 (gain >= +0.03): <PASS/FAIL, value>
Clause 2 (no category below -0.05): <PASS/FAIL, worst category and value>
Clause 3 (p50 < 2000 ms): <PASS/FAIL, value>

QUERY_TRANSFORM <stays empty | becomes multi>.

Abstention on the 7 unanswerable questions: <VALUE> against <VALUE> baseline —
<one sentence on whether widening retrieval makes refusal harder>. Step 22 owns
acting on it.

The reranked row answers step 17's open question: <one sentence on whether a
wider pool changed the code -0.100 result>."
```

- [ ] **Step 10: Tag**

```bash
git tag v0.9
```

---

### Task 7: Publish the result

**Files:**
- Modify: `README.md` (phases 7-8, current state, results table, commands)
- Modify: `docs/roadmap.md` (current state, the shipped-artifacts table, the step map, the plan index)

**Interfaces:**
- Consumes: every number recorded in Tasks 3, 5 and 6.

- [ ] **Step 1: Read the two documents' existing shape**

Run: `sed -n '1,80p' README.md` and `sed -n '1,60p' docs/roadmap.md`

Match the existing voice exactly. The README is in French; the roadmap is in English. Neither gets restructured here.

- [ ] **Step 2: Update the README**

- add phases 7-8 alongside the existing phase sections;
- add every new row to the results table — `rewrite-standalone`, the five `multi-*` rows, `conv-raw`, `conv-rewrite` — with the real numbers, and an em dash for anything genuinely not measured;
- add the commands: `--transform`, `--transform-n`, `--history`, and `scripts/benchmark_conversations.py`;
- include the two `scripts/ask.py` transcripts from Task 4 Step 6 — with and without `--history` — as the qualitative evidence for step 18, labelled as attached to no metric;
- quote two or three `transform_output` values verbatim so a reader can see what the transforms actually produced;
- state the `rewrite-standalone` translation confound in one sentence;
- state the verdict plainly, whichever way it went.

- [ ] **Step 3: Update `docs/roadmap.md`**

- **Current state:** change the header from "Phase 0 and steps 02-17 are done — `v0.7` is tagged" to cover 18-19 and `v0.9`, and name what step 20 should carry forward;
- **Shipped table:** add a `Query transforms` row and a `Conversational fixture` row, in the style of the `Cross-encoder reranking` row — artifact, then evidence, then the measured verdict;
- **The "four things later steps own" list:** update it. Step 17's entry says "the input for steps 18-19"; replace that with what steps 18-19 actually found. Add the abstention finding from Task 6 Step 8 as an input step 22 owns, beside the existing "a score threshold cannot carry refusal" bullet;
- **Step map:** mark row `18-19` done with its verdict, as rows `14-16` and `17` already are;
- **Plan index:** add this plan and its spec to the table, matching the `17 Cross-encoder reranking` row's format;
- **The tree near line 131:** add `retrieval/transform.py  # 18-19  query rewriting and expansion`.

- [ ] **Step 4: Verify every number in the prose against the history file**

Run:

```bash
uv run python scripts/benchmark.py --summary "*"
```

Read every figure you wrote into the README and the roadmap against this table. A number retyped from memory into a published results table is exactly the lie `git_commit()` exists to prevent.

- [ ] **Step 5: Run the gates one last time**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy app && uv run pytest -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/roadmap.md
git commit -m "docs: publish the query transform results

Steps 18-19 measured and recorded. <One sentence on the verdict.>

The roadmap's step 17 entry said the gap is not a ranking problem a generic
model solves; this records what changing the query did about it instead."
```

---

## Notes for the executor

**The verdict is the deliverable.** Three of the last four steps on this project ended with the default unchanged — facet routing at +0.000, hybrid losing Recall@5, the reranker capturing none of its headroom — and each is written up as carefully as a win. A step 18-19 that ends with `QUERY_TRANSFORM=""` is a completed step. Do not reach for a bigger model, a fourth transform or a widened clause to avoid that outcome.

**Task order is deliberate.** Task 3 measures standalone rewriting before any multi-query row is run, for the reason step 17 measured its pool ceiling before writing a reranker: the cheapest measurement that can change the plan runs first.

**Three things will be tempting and are out of scope**, each with the step that owns it: a cache for transform outputs (step 23), a router that transforms only some questions (a follow-up, and unmeasurable before the unconditional version is), and changing the abstention threshold on the strength of what multi-query does to it (step 22).

**`data/eval/questions.jsonl` does not change.** If a task seems to need it to, the task is wrong.

**Every `<VALUE>` and `<PASS/FAIL>` in a commit message above is a real number to fill in** from the run you just did — not a placeholder to leave in place.
