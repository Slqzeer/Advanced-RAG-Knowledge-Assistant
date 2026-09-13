"""The pipeline, with a fake retriever and a fake model: no server, no key."""

from typing import Any

import pytest

from app.core.config import Settings
from app.generation.answer import NO_CONTEXT_ANSWER, answer_question
from app.generation.context import build_context
from app.models.answers import Answer
from app.models.chunks import Chunk, ScoredChunk

SETTINGS = Settings(_env_file=None, generation_model="test-model", top_k=3)


def scored(rank: int, text: str = "body") -> ScoredChunk:
    chunk = Chunk(
        document_id=f"fastapi:doc-{rank}",
        source="fastapi",
        title="Dependencies",
        url=None,
        doc_type="tutorial",
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


def test_the_sources_are_the_cited_chunks_in_citation_order() -> None:
    """Step 09 changed this contract: step 08 returned all three retrieved chunks,
    which overstates provenance when the answer used one. What was retrieved is
    still reported, in ``retrieval``, where it is a retrieval fact."""
    answer = ask(llm=FakeLLM("Caching first [2], then Depends() [1]."))
    assert [source.document_id for source in answer.sources] == ["fastapi:doc-2", "fastapi:doc-1"]
    assert [source.index for source in answer.sources] == [1, 2]
    assert answer.answer == "Caching first [1], then Depends() [2]."


def test_an_invented_citation_is_stripped_and_warned_about() -> None:
    answer = ask(llm=FakeLLM("Depends() [1] caches [9]."))
    assert answer.answer == "Depends() [1] caches."
    assert len(answer.sources) == 1
    assert answer.warnings and "9" in answer.warnings[0]


def test_strict_mode_turns_that_warning_into_a_failure() -> None:
    with pytest.raises(ValueError, match=r"\[9\]"):
        ask(llm=FakeLLM("Depends() [1] caches [9]."), strict=True)


def test_the_retrieval_stats_match_what_reached_the_model() -> None:
    answer = ask()
    assert (answer.retrieval.retrieved, answer.retrieval.used, answer.retrieval.dropped) == (
        3,
        3,
        0,
    )


def test_dropped_chunks_are_counted_and_not_cited() -> None:
    retriever = FakeRetriever([scored(i, "x" * 100) for i in (1, 2, 3)])
    answer = ask(retriever=retriever, max_context_chars=300, llm=FakeLLM("Both [1][2]."))
    assert (answer.retrieval.used, answer.retrieval.dropped) == (2, 1)
    assert [source.document_id for source in answer.sources] == ["fastapi:doc-1", "fastapi:doc-2"]


def test_the_stats_count_what_reached_the_model_not_what_was_cited() -> None:
    """``used`` is a retrieval fact. Deriving it from the citation count would
    make a model that cites lazily look like a retriever that returned less."""
    answer = ask(llm=FakeLLM("Only the first one [1]."))
    assert (answer.retrieval.retrieved, answer.retrieval.used, answer.retrieval.dropped) == (
        3,
        3,
        0,
    )
    assert len(answer.sources) == 1


def test_an_uncited_answer_is_flagged() -> None:
    answer = ask(llm=FakeLLM("Dependencies are resolved per request and cached. " * 3))
    assert answer.sources == []
    assert answer.warnings == ["answer_without_citations"]


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
    # Nothing to cite is not a flaw when there was nothing to cite from.
    assert answer.warnings == []


def test_a_refusal_with_context_available_is_not_flagged() -> None:
    """The prompt's exact refusal sentence. A warning here would train whoever
    reads the output to ignore warnings."""
    refusal = "I do not have enough information in the provided context to answer this."
    answer = ask(llm=FakeLLM(refusal))
    assert (answer.answer, answer.sources, answer.warnings) == (refusal, [], [])


def test_the_retriever_gets_top_k_and_the_filters() -> None:
    # compress="" pins the pool to top_k: since step 20 the default compressor
    # widens it to COMPRESS_CANDIDATES, which is a different test's subject.
    retriever = FakeRetriever()
    ask(retriever=retriever, compress="", top_k=7, filters={"doc_type": "tutorial"})
    assert retriever.calls[0]["top_k"] == 7
    assert retriever.calls[0]["filters"] == {"doc_type": "tutorial"}


def test_top_k_falls_back_to_the_settings() -> None:
    retriever = FakeRetriever()
    ask(retriever=retriever, compress="")
    assert retriever.calls[0]["top_k"] == SETTINGS.top_k


def test_the_default_compressor_is_on_and_widens_the_pool() -> None:
    """Step 20 flipped COMPRESS_METHOD; a default nobody asserts is a default
    that gets reverted by the next refactor without a test going red."""
    settings = Settings(_env_file=None, generation_model="test-model", top_k=3)
    assert settings.compress_method == "embedding"

    retriever = FakeRetriever()
    # Budget above the input, so compress() short-circuits before any embedder:
    # the subject here is the widened depth, not the scoring.
    ask(retriever=retriever, settings=settings, compress_budget=10_000)
    assert retriever.calls[0]["top_k"] == settings.compress_candidates == 20


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


def test_rerank_is_threaded_to_the_retriever() -> None:
    """The measured winner has to reach the answer, not only the benchmark."""
    retriever = FakeRetriever()
    answer_question(
        "how do dependencies work",
        rerank="flashrank",
        rerank_candidates=30,
        retriever=retriever,
        llm=FakeLLM(),
        settings=SETTINGS,
    )
    assert retriever.calls[0]["rerank"] == "flashrank"
    assert retriever.calls[0]["rerank_candidates"] == 30


def test_no_rerank_still_reaches_the_retriever_as_none() -> None:
    """search() reads RERANK_MODEL when it gets None, so answer_question must
    pass None through rather than dropping the keyword — otherwise a configured
    default would apply to the benchmark and not to ask.py."""
    retriever = FakeRetriever()
    answer_question(
        "how do dependencies work", retriever=retriever, llm=FakeLLM(), settings=SETTINGS
    )
    assert retriever.calls[0]["rerank"] is None


# --- conversation history (step 18) ----------------------------------------

HISTORY = [
    {"role": "user", "content": "Comment limiter la memoire d'un container Kubernetes ?"},
    {"role": "assistant", "content": "Via resources.limits.memory [1]."},
]


def test_the_retriever_receives_the_standalone_query_not_the_follow_up() -> None:
    """The whole point of step 18: "et pour docker ?" retrieves nothing on its own."""
    retriever = FakeRetriever()
    ask(
        "et pour docker ?",
        history=HISTORY,
        retriever=retriever,
        llm=FakeLLM("Set a memory limit [1]."),
    )
    assert retriever.calls[0]["query"] == "Set a memory limit [1]."


def test_no_history_means_no_rewriting_call_at_all() -> None:
    """Every single-turn question in the project takes this path."""
    llm = FakeLLM("Use Depends() [1].")
    retriever = FakeRetriever()
    ask("How do dependencies work?", retriever=retriever, llm=llm)
    assert retriever.calls[0]["query"] == "How do dependencies work?"
    assert len(llm.calls) == 1  # the answer only; nothing rewrote the question


def test_the_transform_is_threaded_to_the_retriever() -> None:
    retriever = FakeRetriever()
    ask("How do dependencies work?", transform="multi", transform_n=2, retriever=retriever)
    assert retriever.calls[0]["transform"] == "multi"
    assert retriever.calls[0]["transform_n"] == 2


def test_compression_off_leaves_the_context_byte_identical() -> None:
    """The test that lets every existing caller stay untouched."""
    chunks = [scored(rank, f"One {rank}. Two {rank}. Three {rank}.") for rank in (1, 2, 3)]
    llm = FakeLLM()
    answer = ask(compress="", retriever=FakeRetriever(chunks), llm=llm)
    expected, _, _ = build_context(chunks)
    assert expected in llm.calls[0]["user"]
    assert answer.context_chars == len(expected)


def test_compress_candidates_widens_top_k_only_when_compression_is_on() -> None:
    retriever = FakeRetriever([scored(1, "One. Two.")])
    ask(compress="", top_k=5, compress_candidates=20, retriever=retriever)
    assert [call["top_k"] for call in retriever.calls] == [5]

    ask(
        compress="embedding",
        top_k=5,
        compress_candidates=20,
        compress_budget=10_000,
        retriever=retriever,
    )
    assert [call["top_k"] for call in retriever.calls] == [5, 20]


def test_retrieval_stats_report_the_pool_not_the_survivors() -> None:
    """A d20 run that compresses to a handful of chunks reports 20 retrieved."""
    pool = [scored(rank, f"Sentence {rank} here. Filler {rank} there.") for rank in range(1, 21)]
    answer = ask(
        compress="embedding",
        top_k=5,
        compress_candidates=20,
        compress_budget=40,
        retriever=FakeRetriever(pool),
        embedder=lambda texts: [[1.0, 0.0]] + [[0.5, 0.866]] * (len(texts) - 1),
    )
    assert answer.retrieval.retrieved == 20
    assert answer.retrieval.used + answer.retrieval.dropped == 20
    assert answer.retrieval.used < 20


def test_a_pool_shallower_than_the_answer_is_refused() -> None:
    with pytest.raises(ValueError, match="compress_candidates"):
        ask(compress="embedding", top_k=10, compress_candidates=5)
