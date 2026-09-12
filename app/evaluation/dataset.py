"""Load and check the labelled question set every later step is judged against.

Labels are at *document* level, never chunk level: chunk ids change whenever
``chunk_size`` does, so chunk labels would have to be redone for each strategy
compared in step 12 — and nobody relabels forty questions five times, so the
comparison quietly never happens. Document ids survive re-chunking.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["conceptual", "exact", "code", "multi_doc", "unanswerable"]
CATEGORIES: tuple[str, ...] = get_args(Category)


class EvalQuestion(BaseModel):
    """One labelled question. ``relevant_document_ids`` is the ground truth."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: Category
    relevant_document_ids: list[str]
    # Step 12 may need section granularity on the handful of questions where
    # "right page, wrong section" is the distinction that matters.
    relevant_sections: list[str] = []
    notes: str | None = None
    held_out: bool = False

    @model_validator(mode="after")
    def _check_labels(self) -> "EvalQuestion":
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if self.category == "unanswerable":
            if self.relevant_document_ids:
                raise ValueError("an unanswerable question must have no relevant_document_ids")
        elif not self.relevant_document_ids:
            # An unlabelled question scores zero forever and drags every metric
            # down without ever looking wrong.
            raise ValueError("relevant_document_ids must not be empty")
        return self


def load_dataset(path: Path, *, categories: Sequence[str] | None = None) -> list[EvalQuestion]:
    """Read a JSONL question file, skipping blank and ``#`` comment lines.

    Every error names the line number: chasing "invalid JSON" through a fifty
    line file without one wastes an afternoon.
    """
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            question = EvalQuestion.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        if question.question_id in seen:
            raise ValueError(f"{path}:{number}: duplicate question_id {question.question_id!r}")
        seen.add(question.question_id)
        questions.append(question)
    if categories is None:
        return questions
    wanted = set(categories)
    return [question for question in questions if question.category in wanted]


def validate_dataset(questions: Sequence[EvalQuestion], known_document_ids: set[str]) -> list[str]:
    """Return every label that points at a document the corpus does not have.

    Separate from :func:`load_dataset` because it needs the corpus, which the
    unit tests do not have, and it returns problems instead of raising so the
    CLI prints all of them at once rather than one per run.
    """
    return [
        f"{question.question_id}: unknown document_id {document_id!r}"
        for question in questions
        for document_id in question.relevant_document_ids
        if document_id not in known_document_ids
    ]
