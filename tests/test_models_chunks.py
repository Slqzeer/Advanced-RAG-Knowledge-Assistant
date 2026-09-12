import pytest
from pydantic import ValidationError

from app.models.chunks import Chunk, ScoredChunk


def make_chunk() -> Chunk:
    return Chunk(
        document_id="fastapi:tutorial/dependencies",
        source="fastapi",
        title="Dependencies",
        url="https://fastapi.tiangolo.com/tutorial/dependencies/",
        section="First steps",
        doc_type="tutorial",
        chunk_index=0,
        text="Depends() declares a dependency.",
        char_start=0,
        char_end=32,
    )


def test_a_scored_chunk_carries_its_chunk_score_and_rank() -> None:
    scored = ScoredChunk(chunk=make_chunk(), score=0.82, rank=1)
    assert scored.chunk.text.startswith("Depends()")
    assert scored.score == 0.82
    assert scored.rank == 1
    # Step 17 fills this in; until then its absence is the honest answer.
    assert scored.rerank_score is None


def test_rank_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        ScoredChunk(chunk=make_chunk(), score=0.5, rank=0)


def test_the_optional_payload_fields_default_to_none() -> None:
    """Qdrant drops null payload values, so a stored chunk with no url comes
    back without the key at all."""
    payload = make_chunk().to_payload()
    del payload["url"], payload["section"]
    restored = Chunk.model_validate(payload)
    assert restored.url is None
    assert restored.section is None


def test_the_payload_carries_doc_type() -> None:
    assert make_chunk().to_payload()["doc_type"] == "tutorial"
