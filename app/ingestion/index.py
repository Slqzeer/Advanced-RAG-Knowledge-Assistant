from collections.abc import Callable, Sequence
from functools import partial

from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.generation.cache import bump_corpus_revision, clear_cache
from app.ingestion.chunk import chunk_document
from app.ingestion.clean import clean_document
from app.ingestion.embed import EmbeddingCache, embed_texts
from app.models.chunks import Chunk
from app.models.documents import RawDocument
from app.retrieval.bm25 import invalidate_index
from app.retrieval.store import get_client, replace_document_chunks

Embedder = Callable[[list[str]], list[list[float]]]
Replacer = Callable[[QdrantClient, str, Sequence[Chunk], Sequence[Sequence[float]]], int]
CacheClearer = Callable[..., int]
CacheRevisionBumper = Callable[..., int]
IndexInvalidator = Callable[[], None]


def index_document(
    document: RawDocument,
    *,
    settings: Settings | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    replacer: Replacer = replace_document_chunks,
    index_invalidator: IndexInvalidator = invalidate_index,
    revision_bumper: CacheRevisionBumper = bump_corpus_revision,
    cache_clearer: CacheClearer = clear_cache,
) -> int:
    """Clean, chunk, embed, and replace one document in the live index."""
    settings = settings or get_settings()
    cleaned = clean_document(document)
    if cleaned is None:
        raise ValueError("document has too little useful content")

    chunks = chunk_document(
        cleaned,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.chunk_strategy,
    )
    if not chunks:
        raise ValueError("document produced no chunks")

    if embedder is None:
        cache = EmbeddingCache(settings.embedding_cache_path)
        embedder = partial(embed_texts, model=settings.embedding_model, cache=cache)
    vectors = embedder([chunk.text for chunk in chunks])
    count = replacer(client or get_client(settings), settings.qdrant_collection, chunks, vectors)
    index_invalidator()
    revision_bumper(settings=settings)
    cache_clearer(settings=settings)
    return count
