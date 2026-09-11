"""What the pipeline returns: an answer, what it was built from, what it cost.

Written at step 08 but shaped for step 25: this is the body of ``POST /query``
and the record step 24 traces. Designing it once means the endpoint is a wrapper
over an existing object rather than a redesign of one.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Source(BaseModel):
    """One context entry the answer is allowed to cite.

    ``index`` is the number printed in the context block and the number the
    answer text refers to. It is stored rather than derived from list position
    because step 09 validates ``[n]`` against it, and a citation pointing at the
    wrong document is the failure mode a RAG system is judged on.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    document_id: str = Field(min_length=1)
    title: str
    url: str | None = None
    section: str | None = None
    chunk_id: str
    score: float


class RetrievalStats(BaseModel):
    """How many chunks came back, how many reached the model, how many did not."""

    model_config = ConfigDict(frozen=True)

    retrieved: int = Field(ge=0)
    used: int = Field(ge=0)
    dropped: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_split(self) -> "RetrievalStats":
        if self.used + self.dropped != self.retrieved:
            raise ValueError(
                f"used ({self.used}) + dropped ({self.dropped}) "
                f"must equal retrieved ({self.retrieved})"
            )
        return self


class Answer(BaseModel):
    """A grounded answer with everything needed to check it and to bill it."""

    model_config = ConfigDict(frozen=True)

    # An empty answer is a failed call wearing a success's clothes.
    answer: str = Field(min_length=1)
    sources: list[Source] = Field(default_factory=list)
    retrieval: RetrievalStats
    latency_ms: float = Field(ge=0)
    model: str
    # Captured from the first call rather than retrofitted at step 24, when
    # adding it would mean touching every call site. Empty when no call happened.
    usage: dict[str, int] = Field(default_factory=dict)
