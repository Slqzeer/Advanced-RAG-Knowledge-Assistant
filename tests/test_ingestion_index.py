from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.ingestion.index import index_document
from app.models.documents import RawDocument


def document(text: str = "Useful documentation. " * 30) -> RawDocument:
    return RawDocument(
        document_id="team:guide",
        source="team",
        title="Guide",
        path="api/team:guide.md",
        doc_type="guide",
        url=None,
        text=text,
        content_hash="0" * 64,
    )


def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, embedding_cache_path=tmp_path / "embeddings.sqlite")


def test_index_document_reuses_the_pipeline_and_invalidates_answers(tmp_path: Path) -> None:
    calls: list[tuple[str, Any]] = []

    def embedder(texts: list[str]) -> list[list[float]]:
        calls.append(("embed", len(texts)))
        return [[0.1, 0.2] for _ in texts]

    def replacer(client: object, name: str, chunks: Any, vectors: Any) -> int:
        calls.append(("replace", (name, chunks[0].document_id, len(vectors))))
        return len(chunks)

    def clearer(*, settings: Settings) -> int:
        calls.append(("clear", settings.qdrant_collection))
        return 4

    def invalidate() -> None:
        calls.append(("invalidate", None))

    def bump(*, settings: Settings) -> int:
        calls.append(("bump", settings.qdrant_collection))
        return 1

    count = index_document(
        document(),
        settings=settings(tmp_path),
        client=object(),
        embedder=embedder,
        replacer=replacer,
        index_invalidator=invalidate,
        revision_bumper=bump,
        cache_clearer=clearer,
    )

    assert count > 0
    assert [name for name, _ in calls] == [
        "embed",
        "replace",
        "invalidate",
        "bump",
        "clear",
    ]
    assert calls[1][1][0] == "chunks"


def test_failed_replacement_does_not_invalidate_readers(tmp_path: Path) -> None:
    invalidated = False

    def fail(*args: Any) -> int:
        raise RuntimeError("Qdrant rejected the upsert")

    def invalidate() -> None:
        nonlocal invalidated
        invalidated = True

    with pytest.raises(RuntimeError, match="rejected"):
        index_document(
            document(),
            settings=settings(tmp_path),
            client=object(),
            embedder=lambda texts: [[0.1, 0.2] for _ in texts],
            replacer=fail,
            index_invalidator=invalidate,
        )

    assert invalidated is False


def test_index_document_rejects_a_stub_before_embedding(tmp_path: Path) -> None:
    called = False

    def embedder(texts: list[str]) -> list[list[float]]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ValueError, match="too little useful content"):
        index_document(
            document("tiny"),
            settings=settings(tmp_path),
            client=object(),
            embedder=embedder,
        )

    assert called is False
