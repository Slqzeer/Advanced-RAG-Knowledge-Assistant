# Step 10 — Retrieval Evaluation Dataset

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. Steps use `- [ ]` for tracking.

**Goal:** Build the labelled question set that every later step is judged against. No metrics yet (step 11), no retrieval changes. Just ground truth, and the loader that refuses to let it rot.

**Architecture:** A committed JSONL file plus `app/evaluation/dataset.py` to load and validate it. The file is data, the module is the guard.

**Tech stack:** stdlib `json`, `pydantic`.

## Decisions

**This is the most important step in the project and it is mostly typing.** Everything from step 12 to step 30 is a comparison, and a comparison needs a yardstick. Get this wrong — 8 questions, vague labels, leaked answers — and every number in the README afterwards is decoration. Budget real time for it.

**Ground truth is labelled at *document* level, not chunk level.** This is the decision the whole step turns on. Chunk ids change every time `chunk_size` changes, so chunk-level labels would have to be redone for each of the five strategies compared in step 12 — and nobody relabels 40 questions five times, so the comparison quietly never happens. Document ids (step 02) are stable across re-chunking, re-cleaning and re-cloning. A retrieved chunk counts as a hit when its `document_id` is in `relevant_document_ids`.

The cost is honest and worth naming: document-level labels cannot distinguish "retrieved the right page, wrong section" from "retrieved the right section". Step 12 can add optional `relevant_sections` for the handful of questions where that distinction matters.

**Questions are written *from* the corpus, not from imagination.** Open a document, find something a real person would ask about it, write the question in their words — not the document's. A question that reuses the document's exact phrasing tests string matching, not retrieval, and makes dense search look better than it is.

**Four question categories, tagged, because the whole point is to find out *where* retrieval fails:**

| Category | Example | What it probes |
|---|---|---|
| `conceptual` | "How does dependency injection work?" | dense retrieval's home ground |
| `exact` | "What does HTTPException 422 mean?" | identifiers and error codes — the BM25 case |
| `code` | "How do I declare a query parameter with a default?" | code-fence retrieval |
| `multi_doc` | "How do dependencies and security interact?" | questions needing 2+ documents |
| `unanswerable` | "How do I configure nginx?" | refusal behaviour — **ground truth is an empty list** |

Without the `exact` and `unanswerable` buckets, steps 14 and 22 have nothing to prove and their numbers would be invented.

**40-50 questions, with at least 8 per category.** Fewer than ~30 and a single question swings Recall@5 by more than the improvements you are measuring. More than ~60 and you will not finish, and an unfinished dataset is a zero-question dataset.

**Some questions have several relevant documents, and that is not a mistake.** Recall@K is only meaningful when `len(relevant_document_ids)` varies; if every question has exactly one answer, Recall@K collapses into Hit Rate and two of step 11's five metrics become redundant.

**The dataset is committed to git**, unlike the corpus. It is small, hand-made, and the single most expensive artefact to recreate. `data/eval/` must be excluded from the `data/*` gitignore rule — check this explicitly, because the existing `.gitignore` ignores `data/raw/*` and `data/processed/*` only, and a new `data/eval/` directory needs no exception but *does* need confirming with `git check-ignore`.

**Never tune retrieval by looking at individual dataset failures repeatedly.** That is overfitting to 45 questions. Add a handful of held-out questions that are only looked at when a decision feels too good to be true.

## File map

- Create `data/eval/questions.jsonl` — the labelled set.
- Create `data/eval/README.md` — the labelling rules, so a future you labels consistently.
- Create `app/evaluation/dataset.py` — `EvalQuestion`, `load_dataset()`, `validate_dataset()`.
- Create `scripts/validate_dataset.py`.
- Create `tests/test_evaluation_dataset.py`, `tests/data/eval/` fixtures.
- Modify `README.md`.

## Tasks

### Task 1: The schema and loader

- [x] **Step 1: Failing tests**, against small fixture files with deliberately broken rows.

1. A valid 3-row file loads into 3 `EvalQuestion` objects.
2. A malformed JSON line raises an error **naming the line number**. Debugging "invalid JSON" in a 50-line file without a line number is a waste of an afternoon.
3. A duplicate `question_id` raises.
4. An unknown `category` raises and lists the allowed values.
5. An empty `question` raises.
6. A non-`unanswerable` question with an empty `relevant_document_ids` raises — an unlabelled question scores 0 forever and silently drags every metric down.
7. An `unanswerable` question with a non-empty `relevant_document_ids` raises.
8. Blank lines and `#` comment lines in the file are skipped.
9. `load_dataset(..., categories=["exact"])` filters.

- [x] **Step 2: Write `app/evaluation/dataset.py`**

```python
Category = Literal["conceptual", "exact", "code", "multi_doc", "unanswerable"]

class EvalQuestion(BaseModel):
    question_id: str
    question: str
    category: Category
    relevant_document_ids: list[str]
    relevant_sections: list[str] = []
    notes: str | None = None
    held_out: bool = False

def load_dataset(path: Path, *, categories: Sequence[str] | None = None) -> list[EvalQuestion]: ...
def validate_dataset(questions: Sequence[EvalQuestion], known_document_ids: set[str]) -> list[str]: ...
```

`validate_dataset` is separate from `load_dataset` because it needs the corpus, which the unit tests do not have. It returns a list of problems rather than raising, so the CLI can print all of them at once instead of one per run.

- [x] **Step 3: Green.** Commit: `feat(evaluation): add the evaluation dataset loader`.

### Task 2: The corpus cross-check

- [x] **Step 1: Write `scripts/validate_dataset.py`** — load the corpus document ids (from Qdrant, or by loading and cleaning the corpus), load the dataset, run `validate_dataset`, print every problem, exit non-zero if any.

Must catch: a `relevant_document_id` that does not exist in the corpus (a typo, or a document dropped by step 03's stub filter — both happen, and both make a question permanently unanswerable while looking fine).

- [x] **Step 2: Print a distribution summary** — count per category, mean relevant documents per question, held-out count. You will want this number in the README.

- [x] **Step 3: Commit.** `feat(evaluation): add a dataset validation script`.

### Task 3: Write the questions

This is the long task. It is manual on purpose.

- [x] **Step 1: Write `data/eval/README.md`** first — the rules above, in a page. Labelling consistency across two sittings depends on it existing before you start.

- [x] **Step 2: Label `conceptual` (≥10) and `code` (≥8)**

Workflow per question: pick a document, read it, write the question as a user would ask it, then search the corpus yourself for any *other* document that also answers it and add those ids too. Missing a genuinely relevant document penalises a retriever for being right, which is how a good change gets rejected by a bad yardstick.

- [x] **Step 3: Label `exact` (≥8)** — error codes, status codes, decorator and class names, CLI flags, exception types. Use the identifier as the user would type it, without surrounding prose. These are the questions dense search is expected to fail; that expected failure is the evidence for step 14.

- [x] **Step 4: Label `multi_doc` (≥8)** — questions genuinely spanning documents, each with 2-4 relevant ids.

- [x] **Step 5: Label `unanswerable` (≥8)** — plausible technical questions the corpus does not cover, with `relevant_document_ids: []`. Make them *near* the corpus (another web framework, a neighbouring tool) rather than absurd; "how do I bake bread" tests nothing.

- [x] **Step 6: Mark ~5 questions `held_out: true`**, spread across categories.

- [x] **Step 7: Validate and fix**

```powershell
uv run python scripts/validate_dataset.py
```

Iterate to zero problems.

- [x] **Step 8: Spot-check your own labels.** Take 5 questions, run step 07's search CLI, and read the top 10 results. If a result is clearly relevant and not in your label list, your labels are wrong — fix them now, before any number depends on them. Expect to find two or three.

- [x] **Step 9: Commit.** `feat(evaluation): add the labelled retrieval evaluation set`.

### Task 4: Documentation

- [x] **Step 1: Update `README.md`** — a short "Évaluation" section giving the question count, the category distribution, the document-level labelling decision and its rationale. That rationale is exactly the kind of reasoning the brief says should be visible in this project.
- [x] **Step 2: Commit.** `docs: document the evaluation dataset`.

## Verification

```powershell
uv run ruff check . ; uv run ruff format --check . ; uv run mypy app ; uv run pytest
uv run python scripts/validate_dataset.py
git check-ignore data/eval/questions.jsonl   # must print nothing
```

## Definition of done

- 40+ questions, ≥8 per category, validating clean against the real corpus.
- Every `relevant_document_id` exists in the corpus.
- Labels are document-level and survive re-chunking.
- Labelling rules are written down in `data/eval/README.md`.
- ~5 questions held out.
- The dataset is committed to git.

## Deliberately skipped

| Skipped | Add when |
|---|---|
| LLM-generated questions | the hand-written set is too small to detect a difference (and re-check them by hand anyway) |
| Chunk-level or passage-level labels | step 12 needs section granularity on specific questions |
| Graded relevance (0-3 instead of binary) | NDCG on binary labels stops discriminating |
| Reference answers for generation scoring | step 21 (RAGAS) |
| Multi-turn conversations | step 18 (query rewriting) |
| Inter-annotator agreement | a second person labels |
