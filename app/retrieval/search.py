"""Query the index: a question in, ranked chunks out. No LLM here.

``search()`` is the seam the rest of the project is built on. Step 08 calls it,
step 11 benchmarks it, and steps 14-19 replace its internals — hybrid retrieval,
RRF, reranking, query rewriting — without touching this signature. Everything
here is deliberately thin so that stays true.
"""

from collections.abc import Callable, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Condition, FieldCondition, Filter, MatchAny, MatchValue

from app.core.config import Settings, get_settings
from app.ingestion.embed import EmbeddingCache, embed_query
from app.models.chunks import ScoredChunk
from app.retrieval.store import INDEXED_FIELDS, chunk_from_payload, get_client

Embedder = Callable[[str], list[float]]
Filters = Mapping[str, str | Sequence[str]]


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
    unknown = sorted(set(filters) - set(INDEXED_FIELDS))
    if unknown:
        raise ValueError(f"not an indexed field: {', '.join(unknown)}; have {INDEXED_FIELDS}")

    conditions: list[Condition] = []
    for key, value in filters.items():
        if isinstance(value, str):
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            continue
        values = list(value)
        if not values:
            # MatchAny([]) is a filter that matches nothing at all.
            raise ValueError(f"{key} was given an empty list of values")
        conditions.append(FieldCondition(key=key, match=MatchAny(any=values)))
    return Filter(must=conditions)


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


def search(
    query: str,
    *,
    top_k: int = 5,
    filters: Filters | None = None,
    collection: str | None = None,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
) -> list[ScoredChunk]:
    """The ``top_k`` chunks closest to ``query``, best first, scores as Qdrant
    reports them.

    Scores are raw: no normalising, no rescaling. Cosine scores are comparable
    across queries for this collection, and invented normalisation is a layer
    that lies. ``client`` and ``embedder`` are injectable so the unit tests run
    with no server and no API key.

    ``filters`` restricts the search to payload values: ``{"doc_type": "tutorial"}``
    or ``{"doc_type": ["tutorial", "advanced"]}``. Only fields in
    ``store.INDEXED_FIELDS`` are accepted, so a filter is always an index lookup.

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
    embedder = embedder or default_embedder(settings)
    client = client or get_client(settings)

    hits = client.query_points(
        collection_name=collection or settings.qdrant_collection,
        query=embedder(query),
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        ScoredChunk(chunk=chunk_from_payload(hit.payload or {}), score=hit.score, rank=rank)
        for rank, hit in enumerate(hits, start=1)
    ]
