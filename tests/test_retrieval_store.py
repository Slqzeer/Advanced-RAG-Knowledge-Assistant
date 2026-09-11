import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.models.chunks import Chunk
from app.retrieval.store import (
    INDEXED_FIELDS,
    chunk_from_payload,
    collection_stats,
    ensure_collection,
    get_client,
    point_id,
    upsert_chunks,
)


def make_chunk(index: int) -> Chunk:
    return Chunk(
        document_id="fastapi:tutorial/first-steps",
        source="fastapi",
        title="First Steps",
        url="https://fastapi.tiangolo.com/tutorial/first-steps/",
        language="en",
        section="Check it",
        chunk_index=index,
        text=f"chunk body number {index}",
        char_start=index * 100,
        char_end=index * 100 + 20,
    )


CHUNKS = [make_chunk(i) for i in range(3)]
VECTORS = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]


# --- no server needed ------------------------------------------------------


def test_point_id_is_deterministic() -> None:
    assert point_id("fastapi:index#0") == point_id("fastapi:index#0")


def test_point_id_differs_between_chunks() -> None:
    assert point_id("fastapi:index#0") != point_id("fastapi:index#1")


def test_point_id_is_a_uuid() -> None:
    assert str(uuid.UUID(point_id("fastapi:index#0"))) == point_id("fastapi:index#0")


def test_the_payload_round_trips_every_model_field() -> None:
    """Field-by-field on purpose: adding a field to Chunk without carrying it
    through the payload is how search results lose their url three steps later."""
    chunk = make_chunk(7)
    restored = chunk_from_payload(chunk.to_payload())
    for field in Chunk.model_fields:
        assert getattr(restored, field) == getattr(chunk, field), field
    assert restored == chunk


def test_the_payload_carries_the_chunk_id() -> None:
    assert make_chunk(7).to_payload()["chunk_id"] == "fastapi:tutorial/first-steps#7"


def test_a_vector_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="2 vectors"):
        upsert_chunks(None, "x", CHUNKS, VECTORS[:2])  # type: ignore[arg-type]


# --- against a real Qdrant -------------------------------------------------

requires_qdrant = pytest.mark.requires_qdrant


@pytest.fixture(scope="module")
def client() -> Iterator[QdrantClient]:
    connection = get_client(get_settings())
    try:
        connection.get_collections()
    except Exception as error:  # noqa: BLE001 - any transport failure means "not up"
        pytest.skip(f"Qdrant is not running ({error}); docker compose up -d qdrant")
    yield connection
    connection.close()


@pytest.fixture
def collection(client: QdrantClient) -> Iterator[str]:
    name = f"test_chunks_{uuid.uuid4().hex[:8]}"
    yield name
    if client.collection_exists(name):
        client.delete_collection(name)


@requires_qdrant
def test_ensure_collection_is_idempotent(client: QdrantClient, collection: str) -> None:
    ensure_collection(client, collection, 4)
    upsert_chunks(client, collection, CHUNKS, VECTORS)
    ensure_collection(client, collection, 4)
    assert collection_stats(client, collection)["points"] == 3


@requires_qdrant
def test_payload_indexes_exist(client: QdrantClient, collection: str) -> None:
    ensure_collection(client, collection, 4)
    schema = client.get_collection(collection).payload_schema
    assert set(INDEXED_FIELDS) <= set(schema)


@requires_qdrant
def test_upserting_three_chunks_stores_three_points(client: QdrantClient, collection: str) -> None:
    ensure_collection(client, collection, 4)
    assert upsert_chunks(client, collection, CHUNKS, VECTORS) == 3
    assert collection_stats(client, collection)["points"] == 3


@requires_qdrant
def test_upserting_the_same_chunks_twice_stores_three_points(
    client: QdrantClient, collection: str
) -> None:
    """The idempotence guarantee: the indexer is safe to run twenty times."""
    ensure_collection(client, collection, 4)
    upsert_chunks(client, collection, CHUNKS, VECTORS)
    upsert_chunks(client, collection, CHUNKS, VECTORS)
    assert collection_stats(client, collection)["points"] == 3


@requires_qdrant
def test_recreate_empties_the_collection(client: QdrantClient, collection: str) -> None:
    ensure_collection(client, collection, 4)
    upsert_chunks(client, collection, CHUNKS, VECTORS)
    ensure_collection(client, collection, 4, recreate=True)
    assert collection_stats(client, collection)["points"] == 0


@requires_qdrant
def test_a_stored_point_keeps_its_text_and_metadata(client: QdrantClient, collection: str) -> None:
    ensure_collection(client, collection, 4)
    upsert_chunks(client, collection, CHUNKS, VECTORS)
    [record] = client.retrieve(collection, ids=[point_id(CHUNKS[1].chunk_id)], with_payload=True)
    assert record.payload is not None
    assert chunk_from_payload(record.payload) == CHUNKS[1]
