from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _content_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content is blank")
        return value


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)
    filters: dict[str, str | list[str]] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _question_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question is blank")
        return value


class DocumentRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    doc_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    url: str | None = None
    language: str = Field(min_length=1, default="en")


class DocumentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    chunks_indexed: int = Field(ge=1)


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "unavailable"]
    qdrant: bool
    redis: bool | None


class MetricsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int = Field(ge=0)
    generated: int = Field(ge=0)
    refused: int = Field(ge=0)
    failed: int = Field(ge=0)
    exact_hits: int = Field(ge=0)
    semantic_hits: int = Field(ge=0)
    follow_ups: int = Field(ge=0)
    follow_up_repeats: int = Field(ge=0)
    follow_up_repeat_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    known_cost_usd: float = Field(ge=0)
    content_captured: int = Field(ge=0)

    @classmethod
    def empty(cls) -> "MetricsResponse":
        return cls(
            total=0,
            generated=0,
            refused=0,
            failed=0,
            exact_hits=0,
            semantic_hits=0,
            follow_ups=0,
            follow_up_repeats=0,
            follow_up_repeat_rate=None,
            latency_p50_ms=None,
            latency_p95_ms=None,
            prompt_tokens=0,
            completion_tokens=0,
            known_cost_usd=0,
            content_captured=0,
        )
