"""Everything that touches Qdrant lives here.

Point ids are ``uuid5(NAMESPACE, chunk_id)``: Qdrant accepts only integers or
UUIDs, and a deterministic one turns re-indexing into an idempotent upsert
instead of an accumulation of duplicates.
"""

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from app.core.config import Settings, get_settings
from app.models.chunks import Chunk

# A constant, never uuid1/uuid4: point ids must be the same on every machine and
# every run, or idempotence is gone.
NAMESPACE = uuid.UUID("1f1e7a64-0b9e-5c33-9f0e-2a6c4d8b1e77")
# Indexed keyword fields. Anything filterable must be here: an unindexed filter
# still works but scans, and search.build_filter() rejects keys that are not.
INDEXED_FIELDS = ("source", "document_id", "language", "doc_type")
BATCH_SIZE = 256


def get_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, chunk_id))


def chunk_from_payload(payload: Mapping[str, Any]) -> Chunk:
    """Inverse of :meth:`Chunk.to_payload`.

    One line, and that is the point: a hand-written field-by-field mapper is the
    thing that silently loses ``url`` three steps after someone adds a field.
    """
    return Chunk.model_validate(payload)


def ensure_collection(
    client: QdrantClient, name: str, vector_size: int, *, recreate: bool = False
) -> None:
    """Create the collection if it is missing. Cheap to call on every run."""
    if client.collection_exists(name):
        if not recreate:
            return
        client.delete_collection(name)

    # ponytail: one unnamed dense vector, and it stays that way. Step 14 put BM25
    # in-process instead of in a named sparse vector: Qdrant's IDF modifier
    # supplies only the IDF factor, so k1 and b would still live in Python and
    # the formula would be split across two systems. See the step 14-16 design
    # doc; a named sparse vector is the upgrade path if the corpus outgrows an
    # in-process index.
    client.create_collection(
        collection_name=name,
        # Cosine because that is what the embedding model is trained for. Dot
        # product on un-normalised vectors gives rankings that are wrong in ways
        # that look plausible.
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    for field in INDEXED_FIELDS:
        client.create_payload_index(
            collection_name=name, field_name=field, field_schema=PayloadSchemaType.KEYWORD
        )


def upsert_chunks(
    client: QdrantClient,
    name: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    *,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Store chunks with their vectors and return the number of points written."""
    if len(chunks) != len(vectors):
        # A mismatch here mislabels every vector in the batch, and nothing
        # downstream notices except an unexplained drop in recall.
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    points = [
        PointStruct(id=point_id(chunk.chunk_id), vector=list(vector), payload=chunk.to_payload())
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=name, points=points[start : start + batch_size], wait=True)
    return len(points)


def replace_document_chunks(
    client: QdrantClient,
    name: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> int:
    """Replace every indexed chunk for one document, including stale tail chunks."""
    document_ids = {chunk.document_id for chunk in chunks}
    if len(document_ids) != 1:
        raise ValueError("replacement requires chunks from exactly one document")
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    ensure_collection(client, name, len(vectors[0]))
    document_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=next(iter(document_ids))))]
    )
    old_ids: set[int | str | uuid.UUID] = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=name,
            scroll_filter=document_filter,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        old_ids.update(record.id for record in records)
        if offset is None:
            break

    count = upsert_chunks(client, name, chunks, vectors)
    new_ids = {point_id(chunk.chunk_id) for chunk in chunks}
    stale_ids = old_ids - new_ids
    if stale_ids:
        client.delete(
            collection_name=name,
            points_selector=PointIdsList(points=list(stale_ids)),
            wait=True,
        )
    return count


def collection_stats(client: QdrantClient, name: str) -> dict[str, Any]:
    """Point count and vector geometry, for the indexer's summary line."""
    vectors = client.get_collection(name).config.params.vectors
    return {
        "points": client.count(collection_name=name, exact=True).count,
        "vector_size": vectors.size if isinstance(vectors, VectorParams) else None,
        "distance": vectors.distance if isinstance(vectors, VectorParams) else None,
    }
