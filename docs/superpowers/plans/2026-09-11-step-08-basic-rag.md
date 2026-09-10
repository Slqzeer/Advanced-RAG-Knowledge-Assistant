# Step 08 — Basic RAG

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Close the loop. Question in, grounded answer out, assembled by hand: `search → build_context → prompt → LLM → Answer`. This is `v0.2` and the first thing worth showing anyone.

**Architecture:** Three small modules instead of one, because they fail for different reasons and are tested differently — `generation/context.py` (pure, string assembly), `generation/llm.py` (the only place that talks to a model), `generation/answer.py` (the orchestrator, ~20 lines).

**Tech stack:** the `openai` client from step 05. No new dependency, no LangChain — the brief is explicit that this pipeline gets written by hand first, and the whole point is being able to explain every arrow in it.

## Decisions

**One provider, chosen by config, defaulting to the one whose key you already have.** The OpenAI key is already required for embeddings, so generation through the same SDK means one dependency and one credential. Anthropic (`claude-sonnet-5`) is a drop-in: one function body in `llm.py`, different message shape, same signature. Write `llm.py` so that swapping is editing one function, not threading an abstraction through three modules. Check the current model ids and pricing against the provider's own documentation — never from memory, and never from this plan.

**`temperature=0`.** A RAG system that answers differently on a second identical call cannot be evaluated. Steps 11 and 21 depend on this.

**The context block is numbered from 1 and carries source headers:**

```text
[1] FastAPI — Dependencies / First Steps
<chunk text>

[2] FastAPI — Sub-dependencies
<chunk text>
```

Numbering starts at 1 because that is what the model is asked to cite, and because `[0]` reads as an error to both a human and a model. The header is built from `title` and `section`, which is exactly why step 04 tracked `section`.

**Context is capped by characters, with whole chunks only.** A budget of ~12 000 characters (~3 000 tokens) for `top_k=5`. Never truncate a chunk mid-way to fit — a half chunk is a half fact and the model will confidently complete it. Drop the lowest-ranked chunk instead, and report how many were dropped.

**The response model is the final API contract, written now.** `Answer` has `answer`, `sources: list[Source]`, `retrieval: RetrievalStats`, `latency_ms`, `model`. This is the shape `POST /query` returns at step 25 and the shape step 24's trace logs. Designing it once means step 25 is a FastAPI wrapper over an existing object, not a redesign.

**The system prompt is a module constant, not an f-string built inline.** Prompts are the thing you change most often and compare most carefully; it needs one findable home and a version comment.

**A CLI, not an endpoint.** `scripts/ask.py` proves the pipeline. FastAPI at step 25 is packaging, and adding it now means maintaining a server through seventeen steps that do not need one.

## File map

- Create `app/models/answers.py` — `Source`, `RetrievalStats`, `Answer`.
- Create `app/generation/context.py` — `build_context()`.
- Create `app/generation/llm.py` — `complete()`, `SYSTEM_PROMPT`.
- Create `app/generation/answer.py` — `answer_question()`.
- Create `scripts/ask.py`.
- Create `tests/test_generation_context.py`, `tests/test_generation_answer.py`.
- Modify `app/core/config.py`, `.env.example`, `README.md`.

## Tasks

### Task 1: The answer models

- [ ] **Step 1: Failing tests** — `Source` requires `index: int` (≥1), `document_id`, `title`, and allows `url: str | None`, `section: str | None`, `chunk_id`, `score`; `RetrievalStats` carries `retrieved`, `used`, `dropped`; `Answer` requires a non-empty `answer`.

- [ ] **Step 2: Write `app/models/answers.py`.** `Source.index` is the citation number the answer text refers to, so it must match the context block exactly.

- [ ] **Step 3: Green.** Commit: `feat(models): add the answer response model`.

### Task 2: Context building

- [ ] **Step 1: Failing tests** — pure functions, so be thorough here; this is where silent corruption happens.

1. Three chunks produce a block containing `[1]`, `[2]`, `[3]` in rank order.
2. The header of each entry contains the chunk's `title` and, when present, its `section`.
3. A chunk with `section=None` still produces a valid header with no trailing separator.
4. `build_context` returns both the block and the `Source` list, and `Source.index` matches the number in the block for every entry.
5. With a budget smaller than the total, the lowest-ranked chunks are dropped, the kept ones are **complete**, and `dropped` counts correctly.
6. With a budget smaller than the *first* chunk alone, exactly one chunk is kept anyway — returning an empty context is worse than a long one, and the model call should still happen. Assert this explicitly.
7. An empty chunk list returns an empty block and an empty source list, and the caller is expected to handle it (tested in Task 4).
8. Chunk text is never altered: a chunk containing `[1]` or a code fence appears byte-for-byte in the block.

- [ ] **Step 2: Write `app/generation/context.py`**

```python
MAX_CONTEXT_CHARS = 12_000

def build_context(chunks: Sequence[ScoredChunk], *, max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[Source], int]: ...
```

Returns `(context_block, sources, dropped_count)`.

- [ ] **Step 3: Green.** Commit: `feat(generation): build a numbered context block`.

### Task 3: The LLM call

- [ ] **Step 1: Write `app/generation/llm.py`**

```python
SYSTEM_PROMPT = """..."""   # versioned with a comment

def complete(system: str, user: str, *, model: str, client: Any | None = None, temperature: float = 0.0) -> tuple[str, dict[str, int]]: ...
```

Returns the text and a token-usage dict. Capture usage from the first call — step 24 needs it and retrofitting it means touching every call site.

`SYSTEM_PROMPT`, first version (step 09 hardens it):

```text
You answer questions about technical documentation using only the provided context.
If the context does not contain the answer, say that you do not know.
Do not use prior knowledge. Do not invent APIs, flags or version numbers.
Cite the context entry for each factual statement using its bracketed number, e.g. [1].
Answer in the language of the question.
```

"Answer in the language of the question" is there because the corpus is English and the brief's example question is French. Without it the system answers in the corpus language and looks broken.

Retry on rate limits with the same backoff helper as step 05 — extract it to `app/core/retry.py` rather than writing it twice.

- [ ] **Step 2: Manual check** — one real call, confirm text and usage come back. No automated test hits the network.

- [ ] **Step 3: Commit.** `feat(generation): add the LLM call`.

### Task 4: The orchestrator

- [ ] **Step 1: Failing tests**, with a fake retriever and a fake LLM.

1. `answer_question("q")` returns an `Answer` whose `sources` match the retrieved chunks and whose `retrieval.retrieved`/`used`/`dropped` are consistent.
2. The user prompt passed to the LLM contains both the context block and the question.
3. **Zero retrieved chunks: no LLM call is made**, and the answer is the configured "not enough information" message with an empty source list. Paying for a call that can only hallucinate is the one unforgivable bug in a RAG pipeline; this is the test that prevents it.
4. `latency_ms` is populated and positive.
5. `model` on the `Answer` matches the configured model.
6. An LLM exception propagates rather than being swallowed into a fake answer.

- [ ] **Step 2: Write `app/generation/answer.py`**

```python
def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    source: str | None = None,
    settings: Settings | None = None,
    retriever: Callable[..., list[ScoredChunk]] | None = None,
    llm: Callable[..., tuple[str, dict[str, int]]] | None = None,
) -> Answer: ...
```

Twenty lines: validate, retrieve, short-circuit on empty, build context, call, assemble, time it with `time.perf_counter()`.

- [ ] **Step 3: Green, `mypy app` clean.** Commit: `feat(generation): answer questions from retrieved context`.

### Task 5: The CLI and the first real answers

- [ ] **Step 1: Write `scripts/ask.py`** — `uv run python scripts/ask.py "question" [--top-k 5] [--source fastapi] [--show-context]`. Print the answer, then the sources with titles and URLs, then the retrieval stats and latency. `--show-context` prints the exact prompt, which is the debugging tool you will use most.

- [ ] **Step 2: Ask real questions and save the transcripts**

```powershell
uv run python scripts/ask.py "Comment fonctionne l'injection de dependances dans FastAPI ?"
uv run python scripts/ask.py "How does dependency injection work in FastAPI?"
uv run python scripts/ask.py "What does HTTPException 422 mean?"
uv run python scripts/ask.py "How do I limit memory for a Kubernetes pod?"
```

The last one is deliberate: the corpus does not cover it. **The system must say it does not know.** If it answers confidently, the prompt is not doing its job and that is a step 09 problem you now know about before writing step 09.

- [ ] **Step 3: Record** the French-question result, the refusal result, latency, and token counts. Commit: `feat: add the ask CLI`.

### Task 6: Documentation and the tag

- [ ] **Step 1: Update `README.md`** — current state says Phase 1 complete, tick the Phase 1 roadmap box, add the `ask` command, and add a short transcript of one real question and answer with its sources. A README that shows the system working beats three paragraphs claiming it does.
- [ ] **Step 2: Commit and tag.** `docs: document the minimal RAG pipeline`, then `git tag v0.2`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
uv run python scripts/ask.py "How does dependency injection work in FastAPI?"
```

The test suite passes with no network and no API key.

## Definition of done

- A real question gets a real answer from the real corpus, with sources and latency.
- An out-of-corpus question gets an admission of ignorance, not an invention.
- Zero retrieved chunks means zero LLM calls.
- Chunks are never truncated mid-text to fit the budget.
- `Answer` is the shape step 25 will serve and step 24 will log.
- `v0.2` is tagged.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| Citation validation | step 09 — this step only *asks* for citations |
| Streaming responses | a user interface exists to stream to |
| Conversation history | step 18, where query rewriting needs it |
| LangChain chains | composing retrievers and rerankers is genuinely harder by hand (≈ step 17) |
| Token-accurate context budgeting | step 20, with compression |
| A FastAPI endpoint | step 25 |
| Answer caching | step 23 |
