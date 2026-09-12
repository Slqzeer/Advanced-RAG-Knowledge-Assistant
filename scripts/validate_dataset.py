"""Check the labelled question set against the real corpus.

    uv run python scripts/validate_dataset.py [--dataset data/eval/questions.jsonl]

Catches the two failures that are invisible by eye: a ``relevant_document_id``
that no longer exists (a typo, or a document dropped by step 03's stub filter),
and a category that quietly fell below its minimum while questions were added.
Both leave a question permanently unanswerable while the file still looks fine.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.dataset import CATEGORIES, load_dataset, validate_dataset  # noqa: E402
from app.ingestion.clean import clean_document  # noqa: E402
from app.ingestion.loader import load_documents  # noqa: E402

DEFAULT_DATASET = Path("data/eval/questions.jsonl")
MIN_PER_CATEGORY = 8
MIN_QUESTIONS = 40


def corpus_document_ids(corpus_dir: Path) -> set[str]:
    """The ids that survive cleaning — the same set the index was built from.

    Loading and cleaning rather than reading Qdrant: this must run before
    anything is indexed, and cleaning 150 files costs a second.
    """
    return {
        cleaned.document_id
        for name in sorted(p.name for p in corpus_dir.iterdir() if p.is_dir())
        for document in load_documents(corpus_dir / name, name)
        if (cleaned := clean_document(document)) is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = get_settings()
    if not settings.corpus_dir.exists():
        raise SystemExit(f"no corpus under {settings.corpus_dir}; run scripts/fetch_corpus.py")
    known = corpus_document_ids(settings.corpus_dir)
    questions = load_dataset(args.dataset)

    counts = Counter(question.category for question in questions)
    labelled = [len(q.relevant_document_ids) for q in questions if q.category != "unanswerable"]
    held_out = sum(question.held_out for question in questions)

    print(f"{args.dataset}: {len(questions)} questions against {len(known)} documents\n")
    for category in CATEGORIES:
        flag = "" if counts[category] >= MIN_PER_CATEGORY else f"  < {MIN_PER_CATEGORY}"
        print(f"  {category:<14} {counts[category]:>3}{flag}")
    print(f"\n  mean relevant documents per answerable question: {statistics.mean(labelled):.2f}")
    print(f"  held out: {held_out}")

    problems = validate_dataset(questions, known)
    problems += [
        f"category {category!r} has {counts[category]} questions, minimum {MIN_PER_CATEGORY}"
        for category in CATEGORIES
        if counts[category] < MIN_PER_CATEGORY
    ]
    if len(questions) < MIN_QUESTIONS:
        problems.append(f"{len(questions)} questions, minimum {MIN_QUESTIONS}")

    if problems:
        print(f"\n{len(problems)} problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nno problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
