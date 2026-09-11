import pytest
from pydantic import ValidationError

from app.models.answers import Answer, RetrievalStats, Source


def make_source(**overrides: object) -> Source:
    fields: dict[str, object] = {
        "index": 1,
        "document_id": "fastapi:tutorial/dependencies",
        "title": "Dependencies",
        "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "section": "First steps",
        "chunk_id": "fastapi:tutorial/dependencies#0",
        "score": 0.81,
    }
    fields.update(overrides)
    return Source(**fields)  # type: ignore[arg-type]


def make_answer(**overrides: object) -> Answer:
    fields: dict[str, object] = {
        "answer": "Use Depends() [1].",
        "sources": [make_source()],
        "retrieval": RetrievalStats(retrieved=5, used=4, dropped=1),
        "latency_ms": 812.0,
        "model": "gpt-4o-mini",
    }
    fields.update(overrides)
    return Answer(**fields)  # type: ignore[arg-type]


def test_a_source_carries_what_a_citation_needs() -> None:
    source = make_source()
    assert (source.index, source.title, source.chunk_id) == (
        1,
        "Dependencies",
        "fastapi:tutorial/dependencies#0",
    )


def test_citation_numbers_start_at_one() -> None:
    """[0] reads as an error to a human and to a model; the prompt asks for [1]."""
    with pytest.raises(ValidationError, match="index"):
        make_source(index=0)


@pytest.mark.parametrize("field", ["url", "section"])
def test_optional_source_metadata_defaults_to_none(field: str) -> None:
    fields = {"url": None, "section": None}
    assert getattr(make_source(**fields), field) is None


def test_a_source_without_a_document_id_is_rejected() -> None:
    """A citation that cannot be traced back to a document is worse than none."""
    with pytest.raises(ValidationError, match="document_id"):
        make_source(document_id="")


def test_retrieval_stats_add_up() -> None:
    stats = RetrievalStats(retrieved=5, used=4, dropped=1)
    assert stats.used + stats.dropped == stats.retrieved


def test_an_inconsistent_split_is_rejected() -> None:
    with pytest.raises(ValidationError, match="used"):
        RetrievalStats(retrieved=5, used=4, dropped=0)


def test_an_empty_answer_is_rejected() -> None:
    """Returning an empty string as an answer hides a failed call as a success."""
    with pytest.raises(ValidationError, match="answer"):
        make_answer(answer="")


def test_an_answer_with_no_sources_is_valid() -> None:
    """Zero retrieved chunks still produces an Answer, just an unsourced one."""
    answer = make_answer(sources=[], retrieval=RetrievalStats(retrieved=0, used=0, dropped=0))
    assert answer.sources == []
    assert answer.usage == {}


def test_an_answer_is_json_serialisable() -> None:
    """This is the shape step 25 serves over HTTP; it has to round-trip now."""
    answer = make_answer()
    assert Answer.model_validate_json(answer.model_dump_json()) == answer
