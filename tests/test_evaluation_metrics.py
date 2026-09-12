"""Every expected value here is worked out by hand, with the arithmetic shown.

This is the one module where "looks about right" is not acceptable: a wrong MRR
produces plausible numbers forever, and every step from 12 onwards is judged
against them.
"""

import math

import pytest

from app.evaluation.metrics import (
    dedupe_to_documents,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_dedupe_keeps_one_entry_per_document() -> None:
    assert dedupe_to_documents(["a#1", "a#2", "b#0"]) == ["a", "b"]


def test_dedupe_preserves_the_best_rank() -> None:
    # b first at rank 1, a at rank 2: order is by best occurrence, not by id.
    assert dedupe_to_documents(["b#7", "a#3", "b#1", "a#0"]) == ["b", "a"]


def test_dedupe_splits_on_the_last_hash_only() -> None:
    # document ids carry colons and slashes; only the trailing #index is ours.
    assert dedupe_to_documents(["fastapi:tutorial/first#2"]) == ["fastapi:tutorial/first"]


def test_recall_finds_everything() -> None:
    # 2 of 2 relevant in the top 3
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0


def test_recall_is_cut_off_at_k() -> None:
    # b sits at rank 3, outside K=2: 1 of 2
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5


def test_recall_with_no_hit_is_zero() -> None:
    assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_k_larger_than_the_result_list_does_not_raise() -> None:
    assert recall_at_k(["a"], {"a"}, 10) == 1.0


def test_precision_counts_hits_over_k() -> None:
    # 2 hits in 3 slots
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)


def test_precision_divides_by_k_even_when_fewer_results_came_back() -> None:
    # 1 hit, K=5, only 2 results: 1/5, never 1/2.
    assert precision_at_k(["a", "x"], {"a"}, 5) == pytest.approx(0.2)


def test_reciprocal_rank_by_position() -> None:
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y", "z", "a"], {"a"}) == 0.25


def test_reciprocal_rank_without_a_hit_is_zero() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_reciprocal_rank_uses_only_the_first_hit() -> None:
    # three relevant documents retrieved, the first at rank 2: still 1/2.
    assert reciprocal_rank(["x", "a", "b", "c"], {"a", "b", "c"}) == 0.5


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_rate_at_k(["x", "a"], {"a"}, 1) == 0.0


def test_ndcg_of_a_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == 1.0


def test_ndcg_of_an_imperfect_ranking() -> None:
    # DCG  = 1/log2(3) + 1/log2(4)   (hits at ranks 2 and 3)
    # IDCG = 1/log2(2) + 1/log2(3)   (hits at ranks 1 and 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_is_capped_at_k() -> None:
    # 3 relevant documents, 2 slots: retrieving 2 of them in order is perfect.
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2) == 1.0


def test_ndcg_without_a_hit_is_zero() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k])
def test_an_empty_result_list_scores_zero(metric: object) -> None:
    assert metric([], {"a"}, 5) == 0.0  # type: ignore[operator]
    assert reciprocal_rank([], {"a"}) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k])
def test_an_empty_relevant_set_raises(metric: object) -> None:
    # unanswerable questions belong to the abstention path; scoring them 0.0 or
    # 1.0 here would poison every aggregate.
    with pytest.raises(ValueError, match="relevant"):
        metric(["a"], set(), 5)  # type: ignore[operator]
    with pytest.raises(ValueError, match="relevant"):
        reciprocal_rank(["a"], set())


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k])
def test_k_below_one_raises(metric: object) -> None:
    with pytest.raises(ValueError, match="k"):
        metric(["a"], {"a"}, 0)  # type: ignore[operator]
