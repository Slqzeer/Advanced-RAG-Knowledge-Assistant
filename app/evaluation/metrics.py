"""The five ranking metrics, as arithmetic over id lists. No Qdrant, no I/O.

Every function takes a ranked ``list[str]`` of retrieved *document* ids and a
``set[str]`` of relevant ones, and nothing else. That is what makes them
testable against hand-computed examples, which is the only way to know a metric
is right: a wrong implementation produces plausible numbers forever.

Two conventions are fixed here and must stay fixed, because a benchmark whose
definitions drift is not a benchmark:

* **``@K`` counts distinct documents**, not chunks — call
  :func:`dedupe_to_documents` on retrieval output first.
* **Precision@K divides by K**, not by the number of results returned.
"""

import math
from collections.abc import Sequence


def dedupe_to_documents(chunk_ids: Sequence[str]) -> list[str]:
    """``["a#1", "a#2", "b#0"]`` -> ``["a", "b"]``, keeping the best rank.

    Retrieval returns chunks, labels are documents. Skipping this inflates
    Precision@K and makes a chunking strategy that returns five slices of one
    page look better than one that returns five different pages — the exact bug
    that would corrupt step 12's comparison.
    """
    documents: dict[str, None] = {}
    for chunk_id in chunk_ids:
        documents.setdefault(chunk_id.rsplit("#", 1)[0], None)
    return list(documents)


def _check(relevant: set[str], k: int) -> None:
    if not relevant:
        raise ValueError("relevant must not be empty; unanswerable questions are scored separately")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents found in the top ``k``."""
    _check(relevant, k)
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` *slots* that are relevant.

    The denominator is ``k`` even when fewer than ``k`` results came back. Both
    conventions exist; this one penalises a retriever that returns three results
    when it was asked for five, which is the behaviour worth penalising.
    """
    _check(relevant, k)
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """``1 / rank`` of the *first* relevant document, 0.0 if there is none.

    MRR is the mean of this across questions — aggregation lives in the
    benchmark, so this stays a per-question number that can be inspected.
    """
    _check(relevant, 1)
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in relevant:
            return 1 / rank
    return 0.0


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if anything relevant is in the top ``k``, else 0.0."""
    _check(relevant, k)
    return float(bool(set(retrieved[:k]) & relevant))


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain NDCG with the usual ``log2(rank + 1)`` discount.

    The ideal ranking is built from ``min(len(relevant), k)`` hits, so a
    relevant set larger than ``k`` cannot make a perfect top-``k`` score below
    1.0 — otherwise multi-document questions would look broken by construction.
    """
    _check(relevant, k)
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
        if document_id in relevant
    )
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg
