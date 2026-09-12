"""Run the evaluation set against dense retrieval and publish the numbers.

    uv run python scripts/benchmark.py --label "dense-baseline"
    uv run python scripts/benchmark.py --label "hybrid" --compare "dense-baseline"

Every row is appended to ``data/eval/results.jsonl`` with the git commit that
produced it. That file is the README's results table and step 29's dashboard;
rebuilding it from git history later is miserable, so it is committed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.benchmark import (  # noqa: E402
    DEFAULT_ABSTENTION_THRESHOLD,
    BenchmarkResult,
    run_benchmark,
)
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.retrieval.search import search  # noqa: E402

DEFAULT_DATASET = Path("data/eval/questions.jsonl")
HISTORY = Path("data/eval/results.jsonl")
KS = (1, 3, 5, 10)
CATEGORY_COLUMNS = ("recall@5", "recall@10", "precision@5", "mrr", "ndcg@5")


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines += [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def print_result(result: BenchmarkResult) -> None:
    print(f"\n## {result.label}  ({result.git_commit}, {result.timestamp})\n")
    print(
        f"{result.questions} questions: {result.answerable} answerable,"
        f" {result.unanswerable} unanswerable, {len(result.failures)} failed\n"
    )
    print(
        table(
            ["Metric", *[f"@{k}" for k in result.ks]],
            [
                [metric, *[f"{result.aggregate[f'{metric}@{k}']:.3f}" for k in result.ks]]
                for metric in ("recall", "precision", "hit_rate", "ndcg")
            ],
        )
    )
    print(f"\nMRR: {result.aggregate['mrr']:.3f}")
    if "abstention_rate" in result.aggregate:
        print(f"Abstention rate (unanswerable): {result.aggregate['abstention_rate']:.3f}")
    print(f"Latency: p50 {result.latency_p50_ms:.0f} ms, p95 {result.latency_p95_ms:.0f} ms\n")

    # The per-category table is the point of the exercise: one overall Recall@5
    # hides which kind of question is actually broken.
    print(
        table(
            ["Category", "n", *CATEGORY_COLUMNS],
            [
                [
                    category,
                    f"{scores['questions']:.0f}",
                    *[f"{scores[column]:.3f}" for column in CATEGORY_COLUMNS],
                ]
                for category, scores in result.per_category.items()
            ],
        )
    )
    for failure in result.failures:
        print(f"\n!! {failure['question_id']}: {failure['error']}")


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def print_comparison(result: BenchmarkResult, label: str, history: list[dict[str, Any]]) -> int:
    previous = [row for row in history if row["label"] == label]
    if not previous:
        print(f"\nno run labelled {label!r} in {HISTORY}")
        return 1
    baseline = previous[-1]
    print(f"\n## {result.label} vs {label} ({baseline['git_commit']})\n")
    rows = [
        [
            metric,
            f"{baseline['aggregate'].get(metric, 0.0):.3f}",
            f"{value:.3f}",
            f"{value - baseline['aggregate'].get(metric, 0.0):+.3f}",
        ]
        for metric, value in result.aggregate.items()
    ]
    rows.append(
        [
            "latency_p50_ms",
            f"{baseline['latency_p50_ms']:.0f}",
            f"{result.latency_p50_ms:.0f}",
            f"{result.latency_p50_ms - baseline['latency_p50_ms']:+.0f}",
        ]
    )
    print(table(["Metric", label, result.label, "delta"], rows))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="names this run in the history file")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument("--source", help="restrict to one corpus, e.g. fastapi")
    parser.add_argument("--collection", help="default: QDRANT_COLLECTION")
    # Recorded, not used: the collection cannot tell us how it was chunked, and a
    # summary row that says "recursive" for every strategy is worse than no row.
    parser.add_argument("--strategy", help="how --collection was chunked; recorded only")
    parser.add_argument("--chunk-size", type=int, help="ditto; recorded only")
    parser.add_argument("--overlap", type=int, help="ditto; recorded only")
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="print only, append nothing")
    parser.add_argument("--compare", help="print a delta table against a previous label")
    parser.add_argument("--abstention-threshold", type=float, default=DEFAULT_ABSTENTION_THRESHOLD)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = get_settings()
    questions = load_dataset(args.dataset)
    if not args.include_held_out:
        questions = [question for question in questions if not question.held_out]

    collection = args.collection or settings.qdrant_collection
    result = run_benchmark(
        questions,
        lambda text: search(
            text, top_k=args.top_k, source=args.source, collection=collection, settings=settings
        ),
        ks=[k for k in KS if k <= args.top_k],
        label=args.label,
        config={
            "top_k": args.top_k,
            "source": args.source,
            "dataset": str(args.dataset),
            "held_out": args.include_held_out,
            "collection": collection,
            "embedding_model": settings.embedding_model,
            "strategy": args.strategy or settings.chunk_strategy,
            "chunk_size": args.chunk_size or settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap if args.overlap is None else args.overlap,
            "abstention_threshold": args.abstention_threshold,
        },
        abstention_threshold=args.abstention_threshold,
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
