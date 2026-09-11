"""A slice of a cleaned document, with everything the index and the citation need."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Chunk(BaseModel):
    """One retrievable unit: text plus the metadata that survives into Qdrant."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    source: str
    title: str
    url: str | None
    language: str = "en"
    section: str | None
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
