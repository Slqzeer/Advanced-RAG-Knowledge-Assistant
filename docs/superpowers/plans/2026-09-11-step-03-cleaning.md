# Step 03 — Cleaning

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Turn raw MkDocs Markdown into text worth embedding, without destroying the parts that make a technical corpus searchable.

**Architecture:** One pure function, `clean_markdown(text) -> str`, composed of small named sub-steps applied in a fixed order. Pure string in, string out: no I/O, no config, trivially testable.

**Tech stack:** stdlib `re` and `unicodedata`. No Markdown parser — a regex pipeline over MkDocs Markdown is a few dozen lines, and a real parser (`markdown-it-py` + a renderer) is a dependency plus an AST walk for output that is thrown away anyway.

## Decisions

**Code blocks survive verbatim. This is the most important rule in the step.** The brief's own motivating example for hybrid search is the query `HTTPException 422` — a string that only exists inside code. Strip or reformat fenced blocks and you quietly destroy the corpus's exact-match value, then spend step 14 wondering why BM25 does nothing. The cleaner extracts fenced blocks to placeholders first and restores them last; every whitespace and inline rule runs only on prose.

**Clean, do not summarize.** Cleaning removes things that are noise *for retrieval*: front matter, HTML comments, MkDocs machinery. It never rewrites, truncates or paraphrases, because an LLM reads this text later and needs it intact.

**Headings are kept as text.** `## Dependencies with yield` is a strong retrieval signal and it is how step 04 finds split points. Keeping the `#` markers costs nothing.

**Cleaning is not chunking.** No length logic here. The only length decision is dropping a document whose cleaned prose is under 200 characters — a stub or a redirect page, pure noise in the index.

## File map

- Create `app/ingestion/clean.py` — `clean_markdown()`, `clean_document()`.
- Create `tests/test_ingestion_clean.py`.
- Modify `README.md`.

## Tasks

### Task 1: The cleaning pipeline

- [ ] **Step 1: Write the failing tests** — one per rule, each a short literal input and expected output.

1. **Code fences survive byte-for-byte**, including internal blank lines and indentation. Use a fence containing `raise HTTPException(status_code=422)` and assert that exact substring is present afterwards.
2. A `~~~`-fenced block survives too.
3. Indented (4-space) code blocks survive.
4. A leading `---` front-matter block is removed; a `---` horizontal rule in the middle of the document is not.
5. `<!-- comment -->` is removed, including multi-line comments.
6. MkDocs snippet includes (`--8<-- "path"`) and `{!...!}` includes are removed.
7. An admonition marker line (`!!! tip "Title"`) loses the marker and keeps `Title`; the indented body is de-indented.
8. `{ .annotate }` / `{#anchor}` attribute lists are stripped from heading and paragraph ends.
9. Inline links collapse to their text: `[Depends](../x.md)` → `Depends`. Bare autolinks `<https://x>` keep the URL.
10. Image tags `![alt](src)` are removed entirely, `alt` included — an alt text is not content.
11. Three or more consecutive blank lines collapse to two; trailing whitespace per line is removed.
12. Unicode is NFKC-normalised and non-breaking spaces become regular spaces.
13. `clean_document()` returns `None` for a document whose cleaned prose is shorter than 200 characters.
14. **Idempotence:** `clean_markdown(clean_markdown(x)) == clean_markdown(x)` for every fixture above.

- [ ] **Step 2: Write `app/ingestion/clean.py`**

```python
MIN_PROSE_LENGTH = 200

def clean_markdown(text: str) -> str: ...
def clean_document(document: RawDocument) -> RawDocument | None: ...
```

`clean_markdown` order is fixed and must not be shuffled:

1. NFKC-normalise, normalise line endings to `\n`, replace NBSP.
2. Strip leading front matter.
3. **Extract every fenced and indented code block into a placeholder** (`\x00CODE0\x00`) and keep the originals in a list.
4. Remove HTML comments, MkDocs includes, attribute lists, image tags.
5. Flatten admonitions: drop the `!!!` marker, keep the quoted title as a line, de-indent the body by 4.
6. Collapse inline links to their text; leave autolinks alone.
7. Strip trailing whitespace per line; collapse 3+ blank lines to 2.
8. **Restore the code placeholders.**
9. `strip()`.

Use a placeholder character that cannot appear in the source (`\x00`) so step 4-7's regexes can never corrupt a placeholder. Assert on restore that every placeholder was found — a missing one means a regex ate it, and that is a silent data-loss bug worth failing loudly on.

`clean_document` returns a copy with the cleaned text and a recomputed `content_hash`, or `None` when the prose (text minus code blocks) is too short.

- [ ] **Step 3: Tests green, `mypy app` clean.** Commit: `feat(ingestion): clean markdown for retrieval`.

### Task 2: Check it against the real corpus

- [ ] **Step 1: Eyeball the worst cases**

```powershell
uv run python -c "from pathlib import Path; from app.ingestion.loader import load_documents; from app.ingestion.clean import clean_document; docs=[d for d in (clean_document(x) for x in load_documents(Path('data/raw/fastapi'),'fastapi')) if d]; print(len(docs)); print(max(docs,key=lambda d:len(d.text)).text[:1500])"
```

Then print the *shortest* surviving document and the dropped count. Read them. You are looking for: mangled code, leftover `{!`, `--8<--`, stray `</div>`, admonition bodies still indented. Add a test for anything you find, then fix it.

- [ ] **Step 2: Record the numbers** in the commit message: documents in, documents dropped, total characters before and after. These are the first real measurements of the project.

- [ ] **Step 3: Commit.** `test(ingestion): cover cleaning edge cases from the real corpus`.

### Task 3: Documentation

- [ ] Update `README.md` current state and the Phase 1 roadmap line. State the code-block guarantee explicitly — it is a design promise later steps depend on. Commit: `docs: document the cleaning stage`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
```

## Definition of done

- Every fenced and indented code block in the corpus survives cleaning unchanged.
- `clean_markdown` is idempotent on every fixture.
- Front matter, comments, MkDocs includes, attribute lists and images are gone.
- Stub documents are dropped, and the drop count is known.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| A real Markdown AST parser | the regex pipeline produces a bug that regexes genuinely cannot fix |
| Language detection | the corpus stops being English-only |
| Boilerplate / near-duplicate detection | the eval set shows duplicate chunks crowding out real answers |
| HTML-to-text conversion | an HTML source is actually ingested |
| Sentence segmentation | step 12 tries sentence chunking |
