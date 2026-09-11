"""The pipeline, with a fake retriever and a fake model: no server, no key."""

from typing import Any

import pytest

from app.core.config import Settings
from app.generation.answer import NO_CONTEXT_ANSWER, answer_question
from app.models.answers import Answer
from app.models.chunks import Chunk, ScoredChunk

SETTINGS = Settings(_env_file=None, generation_model="test-model", top_k=3)


def scored(rank: int, text: str = "body") -> ScoredChunk:
    chunk = Chunk(
        document_id=f"fastapi:doc-{rank}",
        source="fastapi",
        title="Dependencies",
        url=None,
        section="First steps",
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )
    return ScoredChunk(chunk=chunk, score=1.0 - 0.1 * rank, rank=rank)


class FakeRetriever:
    def __init__(self, chunks: list[ScoredChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else [scored(i) for i in (1, 2, 3)]
        self.calls: list[dict[str, Any]] = []

    def __call__(self, query: str, **kwargs: Any) -> list[ScoredChunk]:
        self.calls.append({"query": query, **kwargs})
        return self.chunks


class FakeLLM:
    def __init__(self, text: str = "Use Depends() [1].") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, system: str, user: str, **kwargs: Any) -> tuple[str, dict[str, int]]:
        self.calls.append({"system": system, "user": user, **kwargs})
        return self.text, {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


def ask(question: str = "How do dependencies work?", **kwargs: Any) -> Answer:
    kwargs.setdefault("retriever", FakeRetriever())
    kwargs.setdefault("llm", FakeLLM())
    kwargs.setdefault("settings", SETTINGS)
    return answer_question(question, **kwargs)


def test_the_sources_are_the_retrieved_chunks_in_rank_order() -> None:
    answer = ask()
    assert [source.document_id for source in answer.sources] == [
        "fastapi:doc-1",
        "fastapi:doc-2",
        "fastapi:doc-3",
    ]
    assert [source.index for source in answer.sources] == [1, 2, 3]


def test_the_retrieval_stats_match_what_reached_the_model() -> None:
    answer = ask()
    assert (answer.retrieval.retrieved, answer.retrieval.used, answer.retrieval.dropped) == (
        3,
        3,
        0,
    )


def test_dropped_chunks_are_counted_and_not_cited() -> None:
    retriever = FakeRetriever([scored(i, "x" * 100) for i in (1, 2, 3)])
    answer = ask(retriever=retriever, max_context_chars=300)
    assert (answer.retrieval.used, answer.retrieval.dropped) == (2, 1)
    assert len(answer.sources) == 2


def test_the_prompt_carries_both_the_context_and_the_question() -> None:
    llm = FakeLLM()
    ask(
        "How do dependencies work?", llm=llm, retriever=FakeRetriever([scored(1, "Depends() body")])
    )
    prompt = llm.calls[0]["user"]
    assert "Depends() body" in prompt
    assert "How do dependencies work?" in prompt
    assert "[1]" in prompt


def test_zero_retrieved_chunks_means_zero_model_calls() -> None:
    """Paying for a call whose only possible output is a hallucination is the one
    unforgivable bug in a RAG pipeline."""
    llm = FakeLLM()
    answer = ask(retriever=FakeRetriever([]), llm=llm)
    assert llm.calls == []
    assert answer.answer == NO_CONTEXT_ANSWER
    assert answer.sources == []
    assert answer.retrieval.retrieved == 0
    assert answer.usage == {}


def test_the_retriever_gets_top_k_and_the_source_filter() -> None:
    retriever = FakeRetriever()
    ask(retriever=retriever, top_k=7, source="fastapi")
    assert retriever.calls[0]["top_k"] == 7
    assert retriever.calls[0]["source"] == "fastapi"


def test_top_k_falls_back_to_the_settings() -> None:
    retriever = FakeRetriever()
    ask(retriever=retriever)
    assert retriever.calls[0]["top_k"] == SETTINGS.top_k


def test_latency_is_measured_and_positive() -> None:
    assert ask().latency_ms > 0


def test_the_model_and_its_usage_are_reported() -> None:
    answer = ask()
    assert answer.model == "test-model"
    assert answer.usage["total_tokens"] == 120


def test_the_configured_model_is_the_one_called() -> None:
    llm = FakeLLM()
    ask(llm=llm)
    assert llm.calls[0]["model"] == "test-model"


def test_a_model_failure_propagates() -> None:
    """A swallowed exception becomes an answer that looks real and is not."""

    def broken(system: str, user: str, **kwargs: Any) -> tuple[str, dict[str, int]]:
        raise RuntimeError("rate limited")

    with pytest.raises(RuntimeError, match="rate limited"):
        ask(llm=broken)


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_an_empty_question_is_rejected_before_anything_is_spent(question: str) -> None:
    retriever, llm = FakeRetriever(), FakeLLM()
    with pytest.raises(ValueError, match="empty"):
        ask(question, retriever=retriever, llm=llm)
    assert retriever.calls == []
    assert llm.calls == []


def test_an_empty_model_response_fails_loudly() -> None:
    with pytest.raises(ValueError, match="answer"):
        ask(llm=FakeLLM(text="  "))
