import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import Settings
from app.models.answers import Answer
from app.models.chunks import ScoredChunk


class TraceChunk(BaseModel):
    """One ranked chunk and whether it reached or supported the answer."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float
    rerank_score: float | None = None
    used: bool = False
    cited: bool = False


class TraceTimings(BaseModel):
    """Measured stage times; zero means the stage did not run on this call."""

    model_config = ConfigDict(frozen=True)

    contextualize_ms: float = Field(ge=0, default=0)
    retrieval_ms: float = Field(ge=0, default=0)
    context_ms: float = Field(ge=0, default=0)
    generation_ms: float = Field(ge=0, default=0)
    total_ms: float = Field(ge=0)


class QueryTrace(BaseModel):
    """The versioned operational record consumed by the future dashboard."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    started_at: datetime
    question: str = Field(min_length=1)
    resolved_query: str | None = None
    history_turns: int = Field(ge=0, default=0)
    follow_up: bool = False
    follow_up_repeat: bool = False
    model: str
    retrieval_mode: str
    transform: str | None = None
    reranker: str | None = None
    compressor: str | None = None
    prompt_version: str
    top_k: int = Field(ge=1)
    chunks: list[TraceChunk] = Field(default_factory=list)
    system_prompt: str | None = None
    user_prompt: str | None = None
    response: str | None = None
    refusal: Literal["no_context", "model_declined"] | None = None
    cache: Literal["exact", "semantic"] | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = Field(ge=0, default=None)
    timings: TraceTimings
    error_type: str | None = None
    error_message: str | None = None

    @field_validator("started_at")
    @classmethod
    def _started_at_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")
        return value


def generation_cost(usage: Mapping[str, int], settings: Settings) -> float | None:
    """Cost at deployment-configured prices, or unknown when either is absent."""
    price_in = settings.generation_usd_per_mtok_input
    price_out = settings.generation_usd_per_mtok_output
    if price_in is None or price_out is None:
        return None
    return (
        usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out
    ) / 1_000_000


def write_trace(path: Path | None, trace: QueryTrace) -> None:
    """Append one UTF-8 JSONL record; no path means tracing is disabled."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # shortcut: append is single-process; switch to OTLP or a writer service
    # before multi-worker API deployment.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(trace.model_dump_json() + "\n")


class QueryRecorder:
    """Mutable collection state that freezes into one `QueryTrace` at the edge."""

    def __init__(
        self,
        question: str,
        history: Sequence[Mapping[str, str]],
        settings: Settings,
        *,
        mode: str | None = None,
        transform: str | None = None,
        rerank: str | None = None,
        compress: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.question = question
        self.history_turns = min(len(history), settings.history_turns)
        self.settings = settings
        self.mode = mode or settings.retrieval_mode
        self.transform = settings.query_transform if transform is None else transform
        self.reranker = settings.rerank_model if rerank is None else rerank
        self.compressor = settings.compress_method if compress is None else compress
        self.top_k = top_k or settings.top_k
        self.trace_id = uuid.uuid4().hex
        self.started_at = datetime.now(UTC)
        self._started = time.perf_counter()
        self._resolved_query: str | None = None
        self._chunks: list[ScoredChunk] = []
        self._used_ids: set[str] = set()
        self._system_prompt: str | None = None
        self._user_prompt: str | None = None
        self._response: str | None = None
        self._usage: dict[str, int] = {}
        self._contextualize_ms = 0.0
        self._retrieval_ms = 0.0
        self._context_ms = 0.0
        self._generation_ms = 0.0

    def resolved(self, query: str, elapsed_ms: float) -> None:
        self._resolved_query = query
        self._contextualize_ms = elapsed_ms

    def retrieved(self, chunks: Sequence[ScoredChunk], elapsed_ms: float) -> None:
        self._chunks = list(chunks)
        self._retrieval_ms = elapsed_ms

    def context_built(
        self,
        chunks: Sequence[ScoredChunk],
        used: int,
        system: str | None,
        user: str | None,
        elapsed_ms: float,
    ) -> None:
        self._used_ids = {item.chunk.chunk_id for item in chunks[:used]}
        self._system_prompt = system
        self._user_prompt = user
        self._context_ms = elapsed_ms

    def generated(self, response: str | None, usage: Mapping[str, int], elapsed_ms: float) -> None:
        self._response = response
        self._usage = dict(usage)
        self._generation_ms = elapsed_ms

    def completed(self, answer: Answer) -> QueryTrace:
        cited = {source.chunk_id for source in answer.sources}
        content = self.settings.trace_content
        return QueryTrace(
            trace_id=self.trace_id,
            started_at=self.started_at,
            question=self.question,
            resolved_query=self._resolved_query,
            history_turns=self.history_turns,
            follow_up=self.history_turns > 0,
            follow_up_repeat=self.history_turns > 0 and answer.cache == "exact",
            model=answer.model,
            retrieval_mode=self.mode,
            transform=self.transform or None,
            reranker=self.reranker or None,
            compressor=self.compressor or None,
            prompt_version=self.settings.prompt_version,
            top_k=self.top_k,
            chunks=[
                TraceChunk(
                    chunk_id=item.chunk.chunk_id,
                    document_id=item.chunk.document_id,
                    rank=item.rank,
                    score=item.score,
                    rerank_score=item.rerank_score,
                    used=item.chunk.chunk_id in self._used_ids,
                    cited=item.chunk.chunk_id in cited,
                )
                for item in self._chunks
            ],
            system_prompt=self._system_prompt if content else None,
            user_prompt=self._user_prompt if content else None,
            response=self._response if content else None,
            refusal=answer.refusal,
            cache=answer.cache,
            warnings=answer.warnings,
            usage=self._usage or answer.usage,
            cost_usd=generation_cost(self._usage or answer.usage, self.settings),
            timings=TraceTimings(
                contextualize_ms=self._contextualize_ms,
                retrieval_ms=self._retrieval_ms,
                context_ms=self._context_ms,
                generation_ms=self._generation_ms,
                total_ms=(time.perf_counter() - self._started) * 1000,
            ),
        )

    def failed(self, error: Exception) -> QueryTrace:
        return QueryTrace(
            trace_id=self.trace_id,
            started_at=self.started_at,
            question=self.question,
            resolved_query=self._resolved_query,
            history_turns=self.history_turns,
            follow_up=self.history_turns > 0,
            model=self.settings.generation_model,
            retrieval_mode=self.mode,
            transform=self.transform or None,
            reranker=self.reranker or None,
            compressor=self.compressor or None,
            prompt_version=self.settings.prompt_version,
            top_k=self.top_k,
            chunks=[
                TraceChunk(
                    chunk_id=item.chunk.chunk_id,
                    document_id=item.chunk.document_id,
                    rank=item.rank,
                    score=item.score,
                    rerank_score=item.rerank_score,
                    used=item.chunk.chunk_id in self._used_ids,
                )
                for item in self._chunks
            ],
            usage=self._usage,
            cost_usd=generation_cost(self._usage, self.settings),
            timings=TraceTimings(
                contextualize_ms=self._contextualize_ms,
                retrieval_ms=self._retrieval_ms,
                context_ms=self._context_ms,
                generation_ms=self._generation_ms,
                total_ms=(time.perf_counter() - self._started) * 1000,
            ),
            error_type=type(error).__name__,
            error_message=str(error),
        )


def record_answer(recorder: QueryRecorder, answer: Answer) -> Answer:
    """Persist a successful trace without making observability a failure mode."""
    try:
        write_trace(recorder.settings.trace_path, recorder.completed(answer))
    except Exception as error:
        return answer.model_copy(
            update={"warnings": [*answer.warnings, f"trace write failed: {error}"]}
        )
    return answer


def record_failure(recorder: QueryRecorder, error: Exception) -> None:
    """Best-effort failure trace; never mask the pipeline's original exception."""
    with suppress(Exception):
        write_trace(recorder.settings.trace_path, recorder.failed(error))
