from pydantic import BaseModel, ConfigDict, Field


class RawDocument(BaseModel):
    """A source document as read from disk, before cleaning or chunking."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    source: str
    title: str
    path: str
    url: str | None
    language: str = "en"
    text: str = Field(min_length=1)
    content_hash: str
