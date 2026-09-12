"""Query the index: a question in, ranked chunks out. No LLM here.

``search()`` is the seam the rest of the project is built on. Step 08 calls it,
step 11 benchmarks it, and steps 14-19 replace its internals — hybrid retrieval,
RRF, reranking, query rewriting — without touching this signature. Everything
here is deliberately thin so that stays true.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Condition, FieldCondition, Filter, MatchAny, MatchValue

from app.core.config import Settings, get_settings
from app.ingestion.embed import EmbeddingCache, embed_query
from app.models.chunks import Chunk, ScoredChunk
from app.retrieval.bm25 import BM25Index, default_index
from app.retrieval.store import INDEXED_FIELDS, chunk_from_payload, get_client

Embedder = Callable[[str], list[float]]
Filters = Mapping[str, str | Sequence[str]]


def _check_filters(filters: Filters) -> None:
    """Reject keys that are not indexed and values that can never match.

    Shared by ``build_filter`` and ``matches_filters`` so a filter is accepted or
    rejected identically whichever branch ends up applying it — a typo must not
    be caught on the dense side and silently honoured on the lexical one.
    """
    unknown = sorted(set(filters) - set(INDEXED_FIELDS))
    if unknown:
        raise ValueError(f"not an indexed field: {', '.join(unknown)}; have {INDEXED_FIELDS}")
    for key, value in filters.items():
        if not isinstance(value, str) and not list(value):
            # MatchAny([]) is a filter that matches nothing at all.
            raise ValueError(f"{key} was given an empty list of values")


def build_filter(filters: Filters | None) -> Filter | None:
    """Turn ``{"doc_type": "tutorial", "source": ["fastapi", "starlette"]}`` into a
    Qdrant filter: a scalar matches one value, a sequence matches any of them, and
    several keys are ANDed.

    Keys are checked against ``INDEXED_FIELDS`` rather than passed through. An
    unindexed key is a full scan; a misspelled one (``doctype``) is a filter that
    silently matches nothing, which reads downstream as "retrieval is broken"
    rather than "the flag is wrong".
    """
    if not filters:
        return None
    _check_filters(filters)
    conditions: list[Condition] = []
    for key, value in filters.items():
        if isinstance(value, str):
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
    return Filter(must=conditions)


def matches_filters(chunk: Chunk, filters: Filters | None) -> bool:
    """``build_filter``'s twin for the lexical branch: the same mapping, applied
    in Python because a BM25 index has no server to push a filter to.

    Kept beside ``build_filter`` and sharing ``_check_filters`` on purpose. Two
    translations of one mapping living in different modules is how one of them
    grows support for a key the other silently ignores, and a filter that half
    applies is worse than one that fails.
    """
    if not filters:
        return True
    _check_filters(filters)
    for key, value in filters.items():
        actual = getattr(chunk, key)
        if isinstance(value, str):
            if actual != value:
                return False
        elif actual not in list(value):
            return False
    return True


def parse_filters(pairs: Sequence[str]) -> dict[str, list[str]]:
    """``["doc_type=tutorial,advanced"]`` -> ``{"doc_type": ["tutorial", "advanced"]}``.

    Shared by the three scripts so ``--filter`` means the same thing everywhere.
    """
    parsed: dict[str, list[str]] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--filter expects key=value, got {pair!r}")
        values = [item.strip() for item in value.split(",") if item.strip()]
        if not values:
            raise ValueError(f"--filter {key} was given no value")
        parsed.setdefault(key.strip(), []).extend(values)
    return parsed


def default_embedder(settings: Settings) -> Embedder:
    """The same cache the corpus was embedded with: a repeated benchmark query
    costs nothing. This is not a *retrieval* cache — step 23 adds that, once
    there is a latency number worth improving."""
    cache = EmbeddingCache(settings.embedding_cache_path)
    return lambda text: embed_query(text, model=settings.embedding_model, cache=cache)


def _dense(
    query: str,
    *,
    top_k: int,
    filters: Filters | None,
    collection: str,
    client: QdrantClient,
    embedder: Embedder,
    **_: Any,
) -> list[ScoredChunk]:
    """Cosine nearest neighbours. Scores are raw, as Qdrant reports them: cosine
    scores are comparable across queries for this collection, and invented
    normalisation is a layer that lies."""
    hits = client.query_points(
        collection_name=collection,
        query=embedder(query),
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
    ).points
    return [
        ScoredChunk(chunk=chunk_from_payload(hit.payload or {}), score=hit.score, rank=rank)
        for rank, hit in enumerate(hits, start=1)
    ]


def _lexical(
    query: str, *, top_k: int, filters: Filters | None, index: BM25Index, **_: Any
) -> list[ScoredChunk]:
    """BM25 over the same chunks, no vector and no server round trip.

    Filters are validated once here rather than per candidate chunk, so a
    misspelled key fails the call instead of quietly matching nothing.
    """
    if filters:
        _check_filters(filters)
    return index.search(query, top_k, predicate=lambda chunk: matches_filters(chunk, filters))


def rrf(rankings: Sequence[Sequence[ScoredChunk]], *, k: int, top_k: int) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: ``sum(1 / (k + rank))`` over the lists a chunk is in.

    Ranks only, never scores, and that is the entire point. A cosine similarity
    and a BM25 score share no scale; normalising them per query would invent a
    comparison the numbers do not support, which is the "layer that lies" this
    module refuses elsewhere.

    The returned ``score`` is the fused one — around 0.03, not a cosine's
    0.3-0.6. It is what produced the ranking, so it is what gets reported, and
    that makes ``benchmark.DEFAULT_ABSTENTION_THRESHOLD`` meaningless outside
    dense mode. Step 22 owns refusal; each run records its ``mode`` so the two
    are never compared by accident.
    """
    if k < 1:
        raise ValueError(f"rrf_k must be at least 1, got {k}")
    fused: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for scored in ranking:
            chunk_id = scored.chunk.chunk_id
            fused[chunk_id] += 1.0 / (k + scored.rank)
            chunks.setdefault(chunk_id, scored.chunk)
    # chunk_id breaks ties so two runs of one commit agree.
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [
        ScoredChunk(chunk=chunks[chunk_id], score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1)
    ]


def _hybrid(
    query: str,
    *,
    top_k: int,
    candidates: int,
    rrf_k: int,
    filters: Filters | None,
    collection: str,
    client: QdrantClient,
    embedder: Embedder,
    index: BM25Index,
) -> list[ScoredChunk]:
    """Both branches to ``candidates`` depth, then fused to ``top_k``.

    ``candidates`` is validated here rather than in ``search()``: the
    single-branch modes ignore it, and rejecting a call they would have served
    correctly names a parameter that branch never reads.
    """
    if candidates < top_k:
        # Fusing two top-3 lists cannot produce 10 results, and a silently short
        # list reads downstream as a recall drop rather than a bad flag.
        raise ValueError(f"candidates ({candidates}) must be at least top_k ({top_k})")
    dense = _dense(
        query,
        top_k=candidates,
        filters=filters,
        collection=collection,
        client=client,
        embedder=embedder,
    )
    lexical = _lexical(query, top_k=candidates, filters=filters, index=index)
    return rrf([dense, lexical], k=rrf_k, top_k=top_k)


Retrieve = Callable[..., list[ScoredChunk]]
# Mirrors chunk.STRATEGIES: the registry is how this project compares N variants
# and promotes a winner. Step 17's reranker registers a key here.
RETRIEVERS: dict[str, Retrieve] = {"dense": _dense, "lexical": _lexical, "hybrid": _hybrid}


def search(
    query: str,
    *,
    top_k: int = 5,
    mode: str | None = None,
    candidates: int | None = None,
    rrf_k: int | None = None,
    filters: Filters | None = None,
    collection: str | None = None,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    index: BM25Index | None = None,
) -> list[ScoredChunk]:
    """The ``top_k`` chunks best matching ``query``, best first.

    ``mode`` selects the retriever and defaults to ``RETRIEVAL_MODE``. The
    signature and the return type are the seam steps 08, 11 and 17 are built on:
    adding a retriever means adding a key to ``RETRIEVERS``, never changing this.

    ``client``, ``embedder`` and ``index`` are injectable so the unit tests run
    with no server and no API key. Only what the chosen mode needs is built, so
    lexical mode never opens a connection.

    ``filters`` restricts the search to payload values: ``{"doc_type": "tutorial"}``
    or ``{"doc_type": ["tutorial", "advanced"]}``. Only fields in
    ``store.INDEXED_FIELDS`` are accepted, so a filter is always an index lookup.

    ``candidates`` is the per-branch depth hybrid mode retrieves before fusing,
    and ``rrf_k`` the fusion constant. Both default to their settings and are
    ignored by the single-branch modes.

    ``collection`` overrides the configured one. Step 12 indexes one collection
    per chunking strategy so a comparison stays reproducible: recreating a single
    collection before each run makes "why did semantic lose this question?"
    unanswerable the moment the next variant is indexed.
    """
    if not query.strip():
        # The empty string embeds fine and retrieves plausible-looking garbage.
        raise ValueError("query is empty")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    settings = settings or get_settings()
    mode = mode or settings.retrieval_mode
    if mode not in RETRIEVERS:
        raise ValueError(f"unknown retrieval mode {mode!r}; have {sorted(RETRIEVERS)}")
    collection = collection or settings.qdrant_collection

    if mode in ("dense", "hybrid"):
        client = client or get_client(settings)
        embedder = embedder or default_embedder(settings)
    if mode in ("lexical", "hybrid"):
        index = index or default_index(settings, collection)

    return RETRIEVERS[mode](
        query,
        top_k=top_k,
        candidates=candidates or settings.retrieval_candidates,
        rrf_k=rrf_k or settings.rrf_k,
        filters=filters,
        collection=collection,
        client=client,
        embedder=embedder,
        index=index,
    )
