"""Deterministic acceptance rules for a fresh retrieval benchmark."""

from collections.abc import Mapping
from typing import Any

AGGREGATE_METRICS = ("recall@5", "recall@10", "recall@20", "mrr")
COUNT_FIELDS = ("questions", "answerable", "unanswerable")


def regression_failures(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    aggregate_tolerance: float = 0.02,
    category_tolerance: float = 0.05,
) -> list[str]:
    """Return every reason ``candidate`` fails the recorded retrieval baseline."""
    failures: list[str] = []
    retrieval_failures = candidate.get("failures", [])
    if retrieval_failures:
        count = len(retrieval_failures)
        failures.append(f"{count} retrieval failure{'s' if count != 1 else ''}")

    for field in COUNT_FIELDS:
        before, after = baseline.get(field), candidate.get(field)
        if after != before:
            failures.append(f"{field} changed from {before} to {after}")

    baseline_aggregate = baseline.get("aggregate", {})
    candidate_aggregate = candidate.get("aggregate", {})
    for metric in AGGREGATE_METRICS:
        if metric not in candidate_aggregate:
            failures.append(f"candidate is missing aggregate {metric}")
            continue
        _check_drop(
            failures,
            f"aggregate {metric}",
            float(baseline_aggregate[metric]),
            float(candidate_aggregate[metric]),
            aggregate_tolerance,
        )

    baseline_categories = baseline.get("per_category", {})
    candidate_categories = candidate.get("per_category", {})
    for category, scores in baseline_categories.items():
        if category not in candidate_categories:
            failures.append(f"candidate is missing category {category}")
            continue
        _check_drop(
            failures,
            f"category {category} recall@5",
            float(scores["recall@5"]),
            float(candidate_categories[category]["recall@5"]),
            category_tolerance,
        )
    return failures


def _check_drop(
    failures: list[str], label: str, baseline: float, candidate: float, tolerance: float
) -> None:
    drop = baseline - candidate
    if drop - tolerance > 1e-12:
        failures.append(
            f"{label} regressed by {drop:.3f} "
            f"({baseline:.3f} -> {candidate:.3f}; limit {tolerance:.3f})"
        )
