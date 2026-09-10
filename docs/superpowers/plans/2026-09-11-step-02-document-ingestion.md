# Step 02 — Document Ingestion

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Get a real technical corpus onto disk and turn it into validated `RawDocument` objects. No cleaning, no chunking, no embeddings.

**Architecture:** Two pieces that stay separate forever — *acquisition* (a script that writes bytes into `data/raw/`, run by a human, never in tests) and *loading* (a pure function over the filesystem, fully testable against a committed fixture tree).

**Tech stack:** stdlib only (`pathlib`, `hashlib`, `re`) plus `pydantic`, already installed via FastAPI. No new dependency.

## Decisions

**Corpus: FastAPI's own docs, Markdown, one shallow git clone.** The brief's example question is about FastAPI `Depends`, so the corpus that answers it comes first. `git clone --depth 1 --filter=blob:none --sparse` beats writing a crawler and beats an HTTP scraper that fights rate limits and HTML. Python, Docker and Kubernetes docs are the same shape (MkDocs Markdown in a git repo), so adding them later is one entry in a list.

**PDFs are deferred.** They need a parser, layout heuristics and a separate cleaning path, and they teach nothing a Markdown corpus does not. Add them when the corpus is the bottleneck.

**`document_id` is derived, stable and human-readable:** `f"{source}:{relative_path_without_extension}"` → `fastapi:tutorial/dependencies/index`. It survives re-cloning, re-cleaning and re-chunking. Step 10 labels ground truth against this id, which is exactly why it must not be a random UUID or a content hash.

**`content_hash` is a separate field.** sha256 of the raw bytes. It answers "did this document change since I embedded it" — needed by step 05's cache and step 06's re-indexing.

## File map

- Create `scripts/fetch_corpus.py` — clone sources into `data/raw/<source>/`.
- Create `app/models/documents.py` — `RawDocument`.
- Create `app/ingestion/loader.py` — `load_documents()`, `load_document()`, `iter_markdown_files()`.
- Create `tests/data/corpus/` — a 4-file fixture tree, committed.
- Create `tests/test_ingestion_loader.py`.
- Modify `README.md` — current state, roadmap, commands.

## Tasks

### Task 1: The document model

- [ ] **Step 1: Write the failing test**

In `tests/test_ingestion_loader.py`, one test: `RawDocument` rejects empty `text` and empty `document_id`. Run `uv run pytest` → FAIL on import.

- [ ] **Step 2: Write `app/models/documents.py`**

A frozen pydantic model. Fields: `document_id: str`, `source: str`, `title: str`, `path: str` (relative, forward slashes), `url: str | None`, `language: str = "en"`, `text: str`, `content_hash: str`. Constrain `document_id` and `text` with `min_length=1`.

Keep `url` nullable. A locally dropped file has no URL, and pretending otherwise produces citations that link nowhere.

- [ ] **Step 3: Test passes.** Commit: `feat(ingestion): add RawDocument model`.

### Task 2: The loader

- [ ] **Step 1: Build the fixture tree**

Under `tests/data/corpus/fastapi/`, four small files:

```text
docs/tutorial/first-steps.md      # h1 title, one code fence
docs/tutorial/dependencies.md     # YAML front matter, then an h1
docs/index.md                     # h1 only
docs/img/logo.png                 # must be ignored
```

Commit them. This is the only corpus the test suite touches, so tests stay offline and deterministic.

- [ ] **Step 2: Write the failing tests**

1. Loading the fixture root yields exactly 3 documents; the `.png` is excluded.
2. `document_id == "fastapi:docs/tutorial/dependencies"` — forward slashes on Windows too.
3. Title comes from the first `# ` heading, and falls back to the filename stem (hyphens → spaces, title-cased) when there is none.
4. Front matter does not leak into `title`.
5. `content_hash` is stable across two loads and differs between two documents.
6. Two loads return the same order (sorted by `document_id`).

- [ ] **Step 3: Write `app/ingestion/loader.py`**

```python
def iter_markdown_files(root: Path) -> Iterator[Path]: ...
def load_document(path: Path, root: Path, source: str, base_url: str | None = None) -> RawDocument: ...
def load_documents(root: Path, source: str, base_url: str | None = None) -> list[RawDocument]: ...
```

Rules: glob `**/*.md` and `**/*.markdown`; skip anything under a dot-directory; read with `encoding="utf-8", errors="replace"`; build `path` with `relative_to(root).as_posix()`; sort the result by `document_id`.

Title extraction: first line matching `^#\s+(.+)$`, searched **after** skipping a leading `---` front-matter block. Strip trailing `#` characters and inline backticks from the captured text.

`url`: `f"{base_url}/{relative_path_without_suffix}/"` when `base_url` is given, else `None`.

- [ ] **Step 4: Tests pass, `mypy app` clean.** Commit: `feat(ingestion): load markdown documents from disk`.

### Task 3: Corpus acquisition script

- [ ] **Step 1: Write `scripts/fetch_corpus.py`**

A module-level list is the entire configuration:

```python
SOURCES = [
    Source(
        name="fastapi",
        repo="https://github.com/fastapi/fastapi.git",
        docs_subdir="docs/en/docs",
        base_url="https://fastapi.tiangolo.com",
    ),
]
```

Per source: `git clone --depth 1 --filter=blob:none --sparse` into a temp directory, `git sparse-checkout set <docs_subdir>`, then copy that subdirectory to `data/raw/<name>/`. Skip a source whose target directory already exists unless `--force` is passed. Print a per-source file count.

Use `subprocess.run(argv_list, check=True)`. Never `shell=True`.

- [ ] **Step 2: Run it for real**

```powershell
uv run python scripts/fetch_corpus.py
uv run python -c "from pathlib import Path; print(sum(1 for _ in Path('data/raw/fastapi').rglob('*.md')))"
```

Expected: several hundred Markdown files. `data/raw/*` is already gitignored, so nothing is committed.

- [ ] **Step 3: Smoke-load the real corpus**

```powershell
uv run python -c "from pathlib import Path; from app.ingestion.loader import load_documents; d = load_documents(Path('data/raw/fastapi'), 'fastapi', 'https://fastapi.tiangolo.com'); print(len(d), d[0].document_id, d[0].title)"
```

Expected: a plausible count and a real title. If titles come back as filename fallbacks everywhere, the front-matter skip is wrong — fix it before moving on.

- [ ] **Step 4: Commit.** `feat(ingestion): add corpus fetch script`.

### Task 4: Documentation

- [ ] **Step 1: Update `README.md`** — current state mentions step 02 with the real document count; tick the ingestion part of the Phase 1 roadmap line; add `uv run python scripts/fetch_corpus.py` to the commands section, noting it writes into untracked `data/raw/`.
- [ ] **Step 2: Commit.** `docs: document corpus ingestion`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
```

All green, new tests visible in the pytest output.

## Definition of done

- `load_documents()` returns stable, sorted, validated `RawDocument` objects from both the fixture tree and the real corpus.
- Document ids are readable and reproducible across re-clones.
- Tests need no network and no Docker.
- `data/raw/` is still untracked.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| PDF and HTML loaders | the corpus genuinely needs a non-Markdown source |
| Incremental, changed-only loading | full loads get slow enough to notice |
| Parallel file reads | load time exceeds a few seconds |
| A `DocumentLoader` ABC | a second loader exists and actually shares the interface |
