"""A slice of a cleaned document, with everything the index and the citation need."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Chunk(BaseModel):
    """One retrievable unit: text plus the metadata that survives into Qdrant."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    source: str
    title: str
    url: str | None = None
    # Required, no default: a payload written before step 13 must fail loudly in
    # chunk_from_payload rather than report a made-up facet into a benchmark row.
    doc_type: str
    language: str = "en"
    section: str | None = None
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int

    @property
    def chunk_id(self) -> str:
        """Readable and derived: it changes when the chunking parameters change."""
        return f"{self.document_id}#{self.chunk_index}"

    @model_validator(mode="after")
    def _check_span(self) -> "Chunk":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self

    def to_payload(self) -> dict[str, Any]:
        """The flat dict stored alongside the vector. Lives here so the indexer and
        the search result mapper cannot drift apart."""
        return {"chunk_id": self.chunk_id, **self.model_dump()}


class ScoredChunk(BaseModel):
    """A chunk together with how well it answered one query.

    Wraps rather than subclasses ``Chunk``: a score belongs to the pair, not to
    the chunk. ``rank`` is stored, not derived from list position, because every
    metric in step 11 is a function of it and recomputing it in four places is
    how an off-by-one reaches a published benchmark.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    # Step 17's reranker writes here. A second field rather than a mutated
    # ``score``, so "what did dense retrieval think?" stays answerable.
    rerank_score: float | None = None
