"""Query the index: a question in, ranked chunks out. No LLM here.

``search()`` is the seam the rest of the project is built on. Step 08 calls it,
step 11 benchmarks it, and steps 14-19 replace its internals — hybrid retrieval,
RRF, reranking, query rewriting — without touching this signature. Everything
here is deliberately thin so that stays true.
"""

from collections.abc import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import Settings, get_settings
from app.ingestion.embed import EmbeddingCache, embed_query
from app.models.chunks import ScoredChunk
from app.retrieval.store import chunk_from_payload, get_client

Embedder = Callable[[str], list[float]]


def source_filter(source: str | None) -> Filter | None:
    """``source`` is a payload-indexed keyword field, so this filters rather than scans."""
    if source is None:
        return None
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


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
    source: str | None = None,
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
        query_filter=source_filter(source),
        limit=top_k,
        with_payload=True,
    ).points

    return [
        ScoredChunk(chunk=chunk_from_payload(hit.payload or {}), score=hit.score, rank=rank)
        for rank, hit in enumerate(hits, start=1)
    ]
