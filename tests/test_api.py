from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.models import HealthResponse, MetricsResponse
from app.core.config import Settings
from app.models.answers import Answer, RetrievalStats


def answer(**updates: Any) -> Answer:
    base: dict[str, Any] = {
        "answer": "Dependency injection supplies declared dependencies [1].",
        "sources": [],
        "retrieval": RetrievalStats(retrieved=1, used=1, dropped=0),
        "context_chars": 30,
        "contexts": ["private retrieved text"],
        "latency_ms": 12.5,
        "model": "test-model",
    }
    return Answer(**(base | updates))


def client(*, authenticated: bool = True, **overrides: Any) -> TestClient:
    defaults: dict[str, Callable[..., Any]] = {
        "answerer": lambda question, **kwargs: answer(),
        "document_indexer": lambda document, **kwargs: 2,
        "health_checker": lambda settings: HealthResponse(status="ok", qdrant=True, redis=None),
        "metrics_reader": lambda settings: MetricsResponse.empty(),
    }
    test_client = TestClient(
        create_app(
            settings=Settings(_env_file=None, rag_api_key="test-api-key"),
            **(defaults | overrides),
        )
    )
    if authenticated:
        test_client.headers["X-API-Key"] = "test-api-key"
    return test_client


def test_post_routes_require_the_configured_api_key() -> None:
    query = client(authenticated=False).post("/query", json={"question": "hello"})
    document = client(authenticated=False).post(
        "/documents",
        json={
            "document_id": "team:guide",
            "source": "team",
            "title": "Guide",
            "doc_type": "guide",
            "text": "Useful documentation " * 20,
        },
    )

    assert query.status_code == document.status_code == 401


def test_post_routes_fail_closed_when_authentication_is_unconfigured() -> None:
    response = TestClient(
        create_app(
            settings=Settings(_env_file=None),
            answerer=lambda question, **kwargs: answer(),
        )
    ).post("/query", json={"question": "hello"}, headers={"X-API-Key": "anything"})

    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication is not configured"}


def test_query_returns_the_existing_answer_without_context_text() -> None:
    response = client().post("/query", json={"question": "How does DI work?"})

    assert response.status_code == 200
    assert response.json()["answer"].startswith("Dependency injection")
    assert "contexts" not in response.json()


def test_query_threads_history_and_filters_to_the_pipeline() -> None:
    seen: dict[str, Any] = {}

    def fake_answerer(question: str, **kwargs: Any) -> Answer:
        seen.update(question=question, **kwargs)
        return answer()

    response = client(answerer=fake_answerer).post(
        "/query",
        json={
            "question": "And for Docker?",
            "history": [{"role": "user", "content": "How do I deploy FastAPI?"}],
            "filters": {"doc_type": ["deployment", "advanced"]},
        },
    )

    assert response.status_code == 200
    assert seen["question"] == "And for Docker?"
    assert seen["history"] == [{"role": "user", "content": "How do I deploy FastAPI?"}]
    assert seen["filters"] == {"doc_type": ["deployment", "advanced"]}


def test_query_rejects_a_blank_question_before_calling_the_pipeline() -> None:
    response = client().post("/query", json={"question": "   "})

    assert response.status_code == 422


def test_query_hides_pipeline_failure_details() -> None:
    def fail(question: str, **kwargs: Any) -> Answer:
        raise RuntimeError("secret provider response")

    response = client(answerer=fail).post("/query", json={"question": "hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "query service unavailable"}


def test_documents_builds_a_document_and_returns_the_indexed_count() -> None:
    seen: dict[str, Any] = {}

    def fake_indexer(document: Any, **kwargs: Any) -> int:
        seen["document"] = document
        return 3

    response = client(document_indexer=fake_indexer).post(
        "/documents",
        json={
            "document_id": "team:runbook",
            "source": "team",
            "title": "Runbook",
            "doc_type": "operations",
            "text": "Useful documentation " * 20,
            "url": "https://example.test/runbook",
        },
    )

    assert response.status_code == 201
    assert response.json() == {"document_id": "team:runbook", "chunks_indexed": 3}
    assert seen["document"].document_id == "team:runbook"
    assert seen["document"].path == "api/team:runbook.md"
    assert len(seen["document"].content_hash) == 64


def test_documents_maps_a_rejected_stub_to_bad_request() -> None:
    def reject(document: Any, **kwargs: Any) -> int:
        raise ValueError("document has too little useful content")

    response = client(document_indexer=reject).post(
        "/documents",
        json={
            "document_id": "team:stub",
            "source": "team",
            "title": "Stub",
            "doc_type": "notes",
            "text": "short",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "document has too little useful content"}


def test_health_uses_503_when_a_required_dependency_is_down() -> None:
    response = client(
        health_checker=lambda settings: HealthResponse(
            status="unavailable", qdrant=False, redis=None
        )
    ).get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "qdrant": False, "redis": None}


def test_metrics_returns_an_empty_content_free_snapshot() -> None:
    response = client().get("/metrics")

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert "question" not in response.json()


def test_api_does_not_claim_frontend_routes() -> None:
    response = client(authenticated=False).get("/")

    assert response.status_code == 404
