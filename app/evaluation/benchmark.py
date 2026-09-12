"""Run the labelled question set through a retriever and aggregate the numbers.

The retriever is any ``str -> list[ScoredChunk]`` callable, which is what keeps
this module honest: steps 14-19 replace dense search with hybrid, RRF and
reranking, and each one is benchmarked by passing a different callable here.
"""

import math
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Any

from app.evaluation.dataset import EvalQuestion
from app.evaluation.metrics import (
    dedupe_to_documents,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.ingestion.loader import doc_type_of
from app.models.chunks import ScoredChunk

Retriever = Callable[[str], list[ScoredChunk]]

# Cosine scores on this collection sit around 0.3-0.6; below this the top hit is
# usually unrelated. Recorded now, acted on in step 22 — and every per-question
# top score is kept in the history so the threshold can be retuned without a rerun.
DEFAULT_ABSTENTION_THRESHOLD = 0.35


def question_doc_type(question: EvalQuestion) -> str | None:
    """The single facet a question's ground truth lives in, or ``None``.

    ``None`` for the 22 of 38 answerable questions whose relevant documents span
    two directories: they belong in no bucket, and putting them in both is how a
    per-facet table starts reporting numbers that cannot be reproduced.
    """
    facets = {
        doc_type_of(document_id.partition(":")[2]) for document_id in question.relevant_document_ids
    }
    return facets.pop() if len(facets) == 1 else None


@dataclass(frozen=True)
class BenchmarkResult:
    """One run. ``to_dict()`` is what gets appended to ``data/eval/results.jsonl``."""

    label: str
    timestamp: str
    git_commit: str
    config: dict[str, Any]
    ks: list[int]
    questions: int
    answerable: int
    unanswerable: int
    aggregate: dict[str, float]
    per_category: dict[str, dict[str, float]]
    per_question: list[dict[str, Any]] = field(repr=False)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    failures: list[dict[str, str]] = field(default_factory=list)
    per_doc_type: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUMMARY_COLUMNS = (
    "label",
    "strategy",
    "size",
    "overlap",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@5",
    "conceptual r@5",
    "p50 ms",
)


def summarise(history: Sequence[Mapping[str, Any]], pattern: str) -> list[list[str]]:
    """One row per label matching ``pattern``, newest run of each, ready for a table.

    ``results.jsonl`` already holds every number a comparison needs, so a matrix
    of runs is a filter over it rather than a driver script that re-runs them.
    """
    latest: dict[str, Mapping[str, Any]] = {
        row["label"]: row for row in history if fnmatch(row["label"], pattern)
    }
    return [
        [
            label,
            str(row["config"].get("strategy", "?")),
            str(row["config"].get("chunk_size", "?")),
            str(row["config"].get("chunk_overlap", "?")),
            *[f"{row['aggregate'].get(m, 0.0):.3f}" for m in ("recall@5", "recall@10", "mrr")],
            f"{row['aggregate'].get('ndcg@5', 0.0):.3f}",
            f"{row['per_category'].get('conceptual', {}).get('recall@5', 0.0):.3f}",
            f"{row['latency_p50_ms']:.0f}",
        ]
        for label, row in latest.items()
    ]


def git_commit() -> str:
    """``<short sha>`` or ``<short sha>-dirty``.

    A row attributed to a clean commit but produced by uncommitted code is a lie
    you will believe three steps later, when the numbers refuse to reproduce.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. Defined for a single sample, unlike
    ``statistics.quantiles``, and a benchmark run of one question happens."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _score(documents: Sequence[str], relevant: set[str], ks: Sequence[int]) -> dict[str, float]:
    scores = {"mrr": reciprocal_rank(documents, relevant)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(documents, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(documents, relevant, k)
        scores[f"hit_rate@{k}"] = hit_rate_at_k(documents, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(documents, relevant, k)
    return scores


def _mean_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    return {key: statistics.mean([float(row[key]) for row in rows]) for key in keys} if rows else {}


def run_benchmark(
    questions: Sequence[EvalQuestion],
    retriever: Retriever,
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    label: str,
    config: Mapping[str, Any] | None = None,
    abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
) -> BenchmarkResult:
    """Retrieve once per question at ``max(ks)`` and compute every K from it.

    Re-querying per K would quadruple the run time and the cost for identical
    results. ``unanswerable`` questions are excluded from every ranking metric —
    recall over an empty relevant set is undefined — and measured as abstention
    instead: the fraction whose top score falls below ``abstention_threshold``.
    """
    if not questions:
        raise ValueError("no questions to benchmark")
    ks = sorted(ks)
    answerable_rows: list[dict[str, Any]] = []
    unanswerable_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    latencies: list[float] = []

    for question in questions:
        started = time.perf_counter()
        try:
            chunks = retriever(question.question)
        except Exception as error:  # one transient API error must
            # not cost a forty-question run; the count is reported, not swallowed.
            failures.append({"question_id": question.question_id, "error": f"{error}"})
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)

        documents = dedupe_to_documents([scored.chunk.chunk_id for scored in chunks])
        top_score = chunks[0].score if chunks else 0.0
        row: dict[str, Any] = {
            "question_id": question.question_id,
            "category": question.category,
            "retrieved_chunks": len(chunks),
            "retrieved_documents": len(documents),
            "top_score": top_score,
            "latency_ms": latency_ms,
        }
        if question.category == "unanswerable":
            row["abstained"] = float(top_score < abstention_threshold)
            unanswerable_rows.append(row)
        else:
            row.update(_score(documents, set(question.relevant_document_ids), ks))
            row["doc_type"] = question_doc_type(question)
            answerable_rows.append(row)

    metric_keys = list(_score(["x"], {"x"}, ks))
    aggregate = _mean_rows(answerable_rows, metric_keys)
    if unanswerable_rows:
        aggregate["abstention_rate"] = statistics.mean(
            [float(row["abstained"]) for row in unanswerable_rows]
        )

    categories = sorted({row["category"] for row in answerable_rows})
    per_category = {
        category: {
            "questions": float(len(rows)),
            **_mean_rows(rows, metric_keys),
        }
        for category in categories
        if (rows := [row for row in answerable_rows if row["category"] == category])
    }

    facets = sorted({row["doc_type"] for row in answerable_rows if row["doc_type"]})
    per_doc_type = {
        facet: {"questions": float(len(rows)), **_mean_rows(rows, metric_keys)}
        for facet in facets
        if (rows := [row for row in answerable_rows if row["doc_type"] == facet])
    }

    return BenchmarkResult(
        label=label,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        config=dict(config or {}),
        ks=list(ks),
        questions=len(questions),
        answerable=len(answerable_rows),
        unanswerable=len(unanswerable_rows),
        aggregate=aggregate,
        per_category=per_category,
        per_question=[*answerable_rows, *unanswerable_rows],
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        failures=failures,
        per_doc_type=per_doc_type,
    )
