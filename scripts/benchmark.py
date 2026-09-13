"""Run the evaluation set against dense retrieval and publish the numbers.

    uv run python scripts/benchmark.py --label "dense-baseline"
    uv run python scripts/benchmark.py --label "tutorial" --filter doc_type=tutorial
    uv run python scripts/benchmark.py --label "oracle" --oracle-filter
    uv run python scripts/benchmark.py --label "bm25-sentence" --mode lexical
    uv run python scripts/benchmark.py --label "hybrid-k60-d50" --mode hybrid --candidates 50
    uv run python scripts/benchmark.py --label "hybrid" --compare "dense-baseline"
    uv run python scripts/benchmark.py --summary "chunk-*"

Every row is appended to ``data/eval/results.jsonl`` with the git commit that
produced it. That file is the README's results table and step 29's dashboard;
rebuilding it from git history later is miserable, so it is committed.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.benchmark import (  # noqa: E402
    DEFAULT_ABSTENTION_THRESHOLD,
    SUMMARY_COLUMNS,
    BenchmarkResult,
    question_doc_type,
    run_benchmark,
    summarise,
)
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.generation.llm import complete  # noqa: E402
from app.models.chunks import ScoredChunk  # noqa: E402
from app.retrieval.bm25 import default_index  # noqa: E402
from app.retrieval.rerank import RERANKERS, warm_up  # noqa: E402
from app.retrieval.search import RETRIEVERS, parse_filters, search  # noqa: E402
from app.retrieval.transform import TRANSFORMS  # noqa: E402

DEFAULT_DATASET = Path("data/eval/questions.jsonl")
HISTORY = Path("data/eval/results.jsonl")
# 20 and 30 are step 17's ceiling: a reranker cannot exceed the recall of the
# pool it reorders, so the pool's recall at its own depth is the target to beat.
# --top-k defaults to max(KS) and run_benchmark filters ks to those <= top_k, so
# every existing invocation is unchanged and no past row is invalidated.
KS = (1, 3, 5, 10, 20, 30)
CATEGORY_COLUMNS = ("recall@5", "recall@10", "precision@5", "mrr", "ndcg@5")


def category_columns(ks: list[int]) -> list[str]:
    """CATEGORY_COLUMNS minus whatever this run's --top-k never computed.

    run_benchmark filters ks to those <= top_k, so `--top-k 5` produces no
    recall@10 and a hardcoded column list raises KeyError on a run that is
    otherwise fine.
    """
    computed = {"mrr", *(f"{m}@{k}" for m in ("recall", "precision", "ndcg") for k in ks)}
    return [column for column in CATEGORY_COLUMNS if column in computed]


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
    columns = category_columns(result.ks)
    print(
        table(
            ["Category", "n", *columns],
            [
                [
                    category,
                    f"{scores['questions']:.0f}",
                    *[f"{scores[column]:.3f}" for column in columns],
                ]
                for category, scores in result.per_category.items()
            ],
        )
    )
    # Only the single-facet questions are in here; `n` is printed so a bucket of
    # one is read as a bucket of one.
    if result.per_doc_type:
        print()
        print(
            table(
                ["doc_type", "n", *columns],
                [
                    [
                        facet,
                        f"{scores['questions']:.0f}",
                        *[f"{scores[column]:.3f}" for column in columns],
                    ]
                    for facet, scores in result.per_doc_type.items()
                ],
            )
        )
    for failure in result.failures:
        print(f"\n!! {failure['question_id']}: {failure['error']}")


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
    parser.add_argument("--label", help="names this run in the history file")
    parser.add_argument(
        "--summary", help="print past runs whose label matches this glob, then exit"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=max(KS))
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable; e.g. --filter doc_type=tutorial --filter doc_type=tutorial,advanced",
    )
    parser.add_argument(
        "--oracle-filter",
        action="store_true",
        help="filter each question to its ground-truth doc_type — the ceiling on "
        "facet routing, not a retriever you can ship",
    )
    parser.add_argument("--collection", help="default: QDRANT_COLLECTION")
    parser.add_argument("--mode", choices=sorted(RETRIEVERS), help="default: RETRIEVAL_MODE")
    parser.add_argument(
        "--rerank",
        choices=["", *sorted(RERANKERS)],
        help='cross-encoder that reorders the pool; default: RERANK_MODEL, "" is off',
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        help="how deep the pool goes into the cross-encoder; default: RERANK_CANDIDATES",
    )
    parser.add_argument(
        "--transform",
        choices=["", *sorted(TRANSFORMS)],
        help='query transform applied before retrieval; default: QUERY_TRANSFORM, "" is off',
    )
    parser.add_argument(
        "--transform-n", type=int, help="queries `multi` produces, original included; MULTI_QUERY_N"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        help="per-branch depth before fusion; default: RETRIEVAL_CANDIDATES",
    )
    parser.add_argument("--rrf-k", type=int, help="RRF constant; default: RRF_K")
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

    if args.summary:
        rows = summarise(load_history(HISTORY), args.summary)
        if not rows:
            print(f"no run in {HISTORY} whose label matches {args.summary!r}")
            return 1
        print(f"\n## runs matching {args.summary!r}\n")
        print(table(list(SUMMARY_COLUMNS), rows))
        return 0
    if not args.label:
        parser.error("--label is required unless --summary is given")

    settings = get_settings()
    questions = load_dataset(args.dataset)
    if not args.include_held_out:
        questions = [question for question in questions if not question.held_out]

    collection = args.collection or settings.qdrant_collection
    base_filters = parse_filters(args.filter)
    # Keyed on the question text because run_benchmark hands the retriever a
    # string: widening that seam for one experiment would cost steps 14-19 a
    # signature change each. Exact as long as no two questions share a text.
    oracle = {q.question: question_doc_type(q) for q in questions} if args.oracle_filter else {}
    if args.oracle_filter and len(oracle) != len(questions):
        raise SystemExit("two questions share the same text; the oracle lookup would be wrong")

    mode = args.mode or settings.retrieval_mode
    if mode != "dense":
        # Built here, outside the timed loop, on purpose: the ~200 ms
        # construction charged to the first question would make the p50 column
        # stop meaning per-query retrieval latency, which is all it is used for.
        default_index(settings, collection)

    reranker = args.rerank if args.rerank is not None else settings.rerank_model
    if reranker:
        # Same reason as the BM25 build above: a 1-3 s model load charged to
        # question 1 would make the p50 column stop meaning per-query latency.
        warm_up(reranker, settings)

    transform = args.transform if args.transform is not None else settings.query_transform
    recorder = RecordingLLM() if transform else None

    def retrieve(text: str) -> list[ScoredChunk]:
        filters: dict[str, str | list[str]] = dict(base_filters)
        if facet := oracle.get(text):
            filters["doc_type"] = facet
        return search(
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

    result = run_benchmark(
        questions,
        retrieve,
        ks=[k for k in KS if k <= args.top_k],
        label=args.label,
        config={
            "top_k": args.top_k,
            "mode": mode,
            "candidates": args.candidates or settings.retrieval_candidates,
            "rrf_k": args.rrf_k or settings.rrf_k,
            "rerank": args.rerank or settings.rerank_model or None,
            "rerank_candidates": args.rerank_candidates or settings.rerank_candidates,
            "transform": transform or None,
            "transform_n": args.transform_n or settings.multi_query_n,
            "filters": base_filters or None,
            "oracle_filter": args.oracle_filter,
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
    if recorder:
        # Merged after the fact rather than widened into run_benchmark: the
        # retriever seam is `str -> list[ScoredChunk]` and four steps are built
        # on it. Questions are keyed by text, as --oracle-filter already does.
        by_text = {question.question: question.question_id for question in questions}
        outputs = {by_text[text]: value for text, value in recorder.seen.items() if text in by_text}
        for row in result.per_question:
            if found := outputs.get(row["question_id"]):
                row["transform_output"], row["transform_ms"], row["transform_tokens"] = found
        timings = [row["transform_ms"] for row in result.per_question if "transform_ms" in row]
        tokens = [
            row["transform_tokens"] for row in result.per_question if "transform_tokens" in row
        ]
        if timings:
            print(
                f"\nTransform: p50 {statistics.median(timings):.0f} ms"
                f" of {result.latency_p50_ms:.0f} ms total p50"
                f"  |  {statistics.mean(tokens):.0f} tokens/question"
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
