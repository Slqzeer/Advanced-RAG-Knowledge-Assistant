from copy import deepcopy
from typing import Any

from app.evaluation.regression import regression_failures


def result() -> dict[str, Any]:
    return {
        "questions": 45,
        "answerable": 38,
        "unanswerable": 7,
        "failures": [],
        "aggregate": {"recall@5": 0.77, "recall@10": 0.81, "recall@20": 0.82, "mrr": 0.81},
        "per_category": {
            "code": {"recall@5": 0.80},
            "conceptual": {"recall@5": 0.68},
            "exact": {"recall@5": 0.89},
            "multi_doc": {"recall@5": 0.68},
        },
    }


def test_an_unchanged_benchmark_passes() -> None:
    baseline = result()
    assert regression_failures(deepcopy(baseline), baseline) == []


def test_retrieval_failures_and_fixture_drift_fail_the_gate() -> None:
    baseline = result()
    candidate = deepcopy(baseline)
    candidate["questions"] = 44
    candidate["failures"] = [{"question_id": "q018", "error": "timeout"}]

    failures = regression_failures(candidate, baseline)

    assert any("1 retrieval failure" in failure for failure in failures)
    assert any("questions changed from 45 to 44" in failure for failure in failures)


def test_aggregate_metrics_may_drop_by_at_most_two_points() -> None:
    baseline = result()
    candidate = deepcopy(baseline)
    candidate["aggregate"]["recall@10"] = 0.789

    failures = regression_failures(candidate, baseline)

    assert failures == ["aggregate recall@10 regressed by 0.021 (0.810 -> 0.789; limit 0.020)"]


def test_each_category_may_drop_by_at_most_five_points() -> None:
    baseline = result()
    candidate = deepcopy(baseline)
    candidate["per_category"]["code"]["recall@5"] = 0.749

    failures = regression_failures(candidate, baseline)

    assert failures == ["category code recall@5 regressed by 0.051 (0.800 -> 0.749; limit 0.050)"]


def test_missing_metrics_and_categories_fail_instead_of_becoming_zero() -> None:
    baseline = result()
    candidate = deepcopy(baseline)
    del candidate["aggregate"]["mrr"]
    del candidate["per_category"]["exact"]

    failures = regression_failures(candidate, baseline)

    assert "candidate is missing aggregate mrr" in failures
    assert "candidate is missing category exact" in failures
