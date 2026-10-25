import hashlib
import secrets
from collections.abc import Callable
from contextlib import suppress
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from redis import Redis

from app.api.models import (
    DocumentRequest,
    DocumentResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
)
from app.core.config import Settings, get_settings
from app.generation.cache import cached_answer, connect
from app.ingestion.index import index_document
from app.models.answers import Answer
from app.models.documents import RawDocument
from app.observability.metrics import read_metrics
from app.retrieval.store import get_client

Answerer = Callable[..., Answer]
DocumentIndexer = Callable[..., int]
HealthChecker = Callable[[Settings], HealthResponse]
MetricsReader = Callable[[Settings], MetricsResponse]


def check_dependencies(settings: Settings) -> HealthResponse:
    qdrant = False
    redis: bool | None = None
    with suppress(Exception):
        qdrant = get_client(settings).collection_exists(settings.qdrant_collection)
    if settings.redis_url:
        try:
            cache: Redis = connect(settings)
            redis = bool(cache.ping())
        except Exception:
            redis = False
    return HealthResponse(
        status="ok" if qdrant and redis is not False else "unavailable",
        qdrant=qdrant,
        redis=redis,
    )


def configured_metrics(settings: Settings) -> MetricsResponse:
    return MetricsResponse.model_validate(read_metrics(settings.trace_path))


def _raw_document(request: DocumentRequest) -> RawDocument:
    return RawDocument(
        document_id=request.document_id,
        source=request.source,
        title=request.title,
        path=f"api/{request.document_id}.md",
        doc_type=request.doc_type,
        url=request.url,
        language=request.language,
        text=request.text,
        content_hash=hashlib.sha256(request.text.encode()).hexdigest(),
    )


def create_app(
    *,
    settings: Settings | None = None,
    answerer: Answerer = cached_answer,
    document_indexer: DocumentIndexer = index_document,
    health_checker: HealthChecker = check_dependencies,
    metrics_reader: MetricsReader = configured_metrics,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Advanced RAG Knowledge Assistant", version="1.0.0")
    api_key_header = APIKeyHeader(
        name="X-API-Key",
        scheme_name="RagApiKey",
        description="API key configured through RAG_API_KEY.",
        auto_error=False,
    )

    def require_api_key(
        provided: Annotated[str | None, Security(api_key_header)],
    ) -> None:
        if not settings.rag_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is not configured",
            )
        if provided is None or not secrets.compare_digest(provided, settings.rag_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid API key",
            )

    @app.post(
        "/query",
        response_model=Answer,
        response_model_exclude={"contexts"},
        dependencies=[Depends(require_api_key)],
    )
    def query(request: QueryRequest) -> Answer:
        try:
            return answerer(
                request.question,
                history=[message.model_dump() for message in request.history],
                filters=request.filters or None,
                settings=settings,
            )
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="query service unavailable",
            ) from error

    @app.post(
        "/documents",
        response_model=DocumentResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_api_key)],
    )
    def documents(request: DocumentRequest) -> DocumentResponse:
        try:
            count = document_indexer(_raw_document(request), settings=settings)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="document service unavailable",
            ) from error
        return DocumentResponse(document_id=request.document_id, chunks_indexed=count)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse | JSONResponse:
        result = health_checker(settings)
        if result.status == "ok":
            return result
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=result.model_dump(),
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        try:
            return metrics_reader(settings)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="metrics unavailable",
            ) from error

    return app


app = create_app()
