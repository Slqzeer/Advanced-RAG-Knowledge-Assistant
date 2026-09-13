"""Reorder a shortlist with a cross-encoder. Retrieval finds; this ranks.

A bi-encoder embeds the query and the chunk separately and compares two vectors
that never met — which is what makes it fast enough to run over 1 484 chunks. A
cross-encoder reads the pair together and scores it jointly, which is far more
accurate and far too slow to run over a corpus. So it runs over the 30 chunks a
retriever already shortlisted. That two-stage shape is the whole idea, and it is
why ``rerank_candidates`` matters more here than any model choice.

A backend is a ``Scorer``: it returns ``(index into candidates, score)`` pairs
and nothing else. Ordering, tie-breaking, trimming and rebuilding
``ScoredChunk``s happen once, here, for every backend — the only thing that
genuinely differs between a local ONNX model and a hosted API is the scoring.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.models.chunks import ScoredChunk

Scorer = Callable[[str, Sequence[ScoredChunk], int, Settings], list[tuple[int, float]]]

# Mirrors search.RETRIEVERS and chunk.STRATEGIES: the registry is how this
# project compares N variants and promotes a winner. One at a time, by decision:
# there is no ensembling here and no fusion of two rerankers' scores.
RERANKERS: dict[str, Scorer] = {}


def rerank(
    query: str,
    candidates: Sequence[ScoredChunk],
    *,
    model: str,
    top_k: int,
    settings: Settings | None = None,
) -> list[ScoredChunk]:
    """The ``top_k`` candidates a cross-encoder thinks best answer ``query``.

    ``score`` is left exactly as the retriever wrote it and ``rerank_score``
    carries the new number, so "the retriever never had it" and "the reranker
    buried it" stay different, answerable failures. ``rerank_score is not None``
    is what says which of the two produced the ranking.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if model not in RERANKERS:
        raise ValueError(f"unknown reranker {model!r}; have {sorted(RERANKERS)}")
    if not candidates:
        # Both backends charge for being asked to rank nothing: one in a model
        # invocation, the other in a billed API call.
        return []

    settings = settings or get_settings()
    scored = RERANKERS[model](query, candidates, top_k, settings)
    # chunk_id breaks ties so two runs of one commit agree, as rrf() already does.
    ordered = sorted(scored, key=lambda pair: (-pair[1], candidates[pair[0]].chunk.chunk_id))
    return [
        # model_copy rather than a fresh ScoredChunk: the chunk and the
        # retriever's score come along untouched, which is the point.
        candidates[index].model_copy(update={"rerank_score": score, "rank": rank})
        for rank, (index, score) in enumerate(ordered[:top_k], start=1)
    ]


def warm_up(model: str, settings: Settings) -> None:
    """Build whatever ``model`` loads lazily, before a timed loop starts.

    A local model costs 1-3 s to load; charged to the first question of a
    benchmark it would make the ``p50 ms`` column stop meaning per-query
    retrieval latency, which is the only thing it is used for. Backends with
    nothing to preload — anything that calls out over the network per query —
    are a no-op, and so is an unknown name: this runs before the work, and
    failing here would fail a run that would otherwise have succeeded.
    """
    if model == "flashrank":
        _ranker(settings.flashrank_model)


# Beside the embedding cache rather than in /tmp: on Windows the library's
# default cache_dir is neither writable nor stable across runs, and re-downloading
# 34 MB per process is a cost with no upside.
FLASHRANK_CACHE = Path("data/processed/flashrank")


@lru_cache(maxsize=2)
def _ranker(model_name: str) -> Any:
    """The ONNX session, built once per process and per model name.

    Imported here rather than at module scope so that importing this module
    costs nothing: onnxruntime is tens of megabytes of shared library, and every
    test, every `--mode dense` run and every script that never reranks would pay
    for it at import time otherwise.
    """
    from flashrank import Ranker

    FLASHRANK_CACHE.mkdir(parents=True, exist_ok=True)
    return Ranker(model_name=model_name, cache_dir=str(FLASHRANK_CACHE))


def _flashrank(
    query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
) -> list[tuple[int, float]]:
    """Score every candidate locally. ``top_k`` is ignored: the model scores the
    whole shortlist either way, and trimming is ``rerank()``'s job."""
    from flashrank import RerankRequest

    passages = [{"id": index, "text": scored.chunk.text} for index, scored in enumerate(candidates)]
    results = _ranker(settings.flashrank_model).rerank(
        RerankRequest(query=query, passages=passages)
    )
    return [(int(result["id"]), float(result["score"])) for result in results]


RERANKERS["flashrank"] = _flashrank
