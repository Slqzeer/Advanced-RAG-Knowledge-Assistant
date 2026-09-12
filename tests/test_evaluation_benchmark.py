"""The runner, against a scripted retriever: no Qdrant, no API key, no network."""

import json
from typing import Any

import pytest

from app.evaluation.benchmark import run_benchmark, summarise
from app.evaluation.dataset import EvalQuestion
from app.models.chunks import Chunk, ScoredChunk


def chunk(document_id: str, index: int = 0) -> Chunk:
    return Chunk(
        document_id=document_id,
        source="fastapi",
        title=document_id,
        doc_type="tutorial",
        chunk_index=index,
        text="body",
        char_start=0,
        char_end=10,
    )


def scored(*document_ids: str, score: float = 0.8) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=chunk(document_id), score=score - rank / 100, rank=rank)
        for rank, document_id in enumerate(document_ids, start=1)
    ]


def question(question_id: str, category: str = "conceptual", **overrides: object) -> EvalQuestion:
    fields: dict[str, object] = {
        "question_id": question_id,
        "question": f"question {question_id}",
        "category": category,
        "relevant_document_ids": [] if category == "unanswerable" else [f"doc-{question_id}"],
    }
    fields.update(overrides)
    return EvalQuestion.model_validate(fields)


QUESTIONS = [question("a"), question("b", "exact"), question("c")]


def perfect(text: str) -> list[ScoredChunk]:
    return scored(f"doc-{text.rsplit(' ', 1)[-1]}", "doc-other")


def test_every_question_gets_a_row_and_an_aggregate() -> None:
    result = run_benchmark(QUESTIONS, perfect, label="test")

    assert [row["question_id"] for row in result.per_question] == ["a", "b", "c"]
    assert result.aggregate["recall@5"] == 1.0
    assert result.aggregate["mrr"] == 1.0
    assert result.label == "test"


def test_aggregates_are_means_over_answerable_questions_only() -> None:
    def half(text: str) -> list[ScoredChunk]:
        # finds doc-a at rank 1, misses everything else
        return scored("doc-a", "doc-other")

    result = run_benchmark(QUESTIONS, half, label="test")

    assert result.answerable == 3
    assert result.aggregate["recall@5"] == pytest.approx(1 / 3)
    assert result.aggregate["mrr"] == pytest.approx(1 / 3)


def test_unanswerable_questions_are_scored_only_as_abstention() -> None:
    questions = [*QUESTIONS, question("d", "unanswerable")]

    result = run_benchmark(questions, perfect, label="test", abstention_threshold=0.9)

    assert result.answerable == 3
    assert result.unanswerable == 1
    # every answerable question was perfect; the aggregate must not be diluted
    assert result.aggregate["recall@5"] == 1.0
    # top score 0.79 < 0.9, so the retriever would have abstained
    assert result.aggregate["abstention_rate"] == 1.0
    assert "unanswerable" not in result.per_category


def test_abstention_is_zero_when_the_top_score_clears_the_threshold() -> None:
    result = run_benchmark(
        [question("d", "unanswerable")], perfect, label="test", abstention_threshold=0.5
    )
    assert result.aggregate["abstention_rate"] == 0.0


def test_per_category_keys_match_the_dataset() -> None:
    result = run_benchmark(QUESTIONS, perfect, label="test")

    assert set(result.per_category) == {"conceptual", "exact"}
    assert result.per_category["exact"]["questions"] == 1
    assert result.per_category["conceptual"]["recall@5"] == 1.0


def test_a_failing_retriever_does_not_lose_the_run() -> None:
    def flaky(text: str) -> list[ScoredChunk]:
        if text.endswith("b"):
            raise RuntimeError("rate limited")
        return perfect(text)

    result = run_benchmark(QUESTIONS, flaky, label="test")

    assert len(result.failures) == 1
    assert result.failures[0]["question_id"] == "b"
    assert "rate limited" in result.failures[0]["error"]
    # the two survivors are still scored, and the failure is not counted as a miss
    assert result.answerable == 2
    assert result.aggregate["recall@5"] == 1.0


def test_latency_percentiles_are_populated() -> None:
    result = run_benchmark(QUESTIONS, perfect, label="test")
    assert result.latency_p50_ms >= 0.0
    assert result.latency_p95_ms >= result.latency_p50_ms


def test_the_result_is_json_serialisable() -> None:
    result = run_benchmark(QUESTIONS, perfect, label="test", config={"top_k": 10})

    row = json.loads(json.dumps(result.to_dict()))

    assert row["label"] == "test"
    assert row["config"] == {"top_k": 10}
    assert row["git_commit"]
    assert row["timestamp"]


def test_chunks_are_deduplicated_to_documents_before_scoring() -> None:
    def repeated(text: str) -> list[ScoredChunk]:
        # five chunks of one page: one document retrieved, not five
        return [
            ScoredChunk(chunk=chunk("doc-other", index), score=0.9, rank=index + 1)
            for index in range(4)
        ] + scored("doc-a")

    result = run_benchmark([question("a")], repeated, label="test", ks=(5,))

    assert result.aggregate["precision@5"] == pytest.approx(1 / 5)
    assert result.per_question[0]["retrieved_documents"] == 2


# --- the history summary ---------------------------------------------------


def history_row(label: str, recall5: float, **config: object) -> dict[str, Any]:
    return {
        "label": label,
        "config": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200, **config},
        "aggregate": {"recall@5": recall5, "recall@10": recall5, "mrr": 0.5, "ndcg@5": 0.4},
        "per_category": {"conceptual": {"recall@5": recall5 - 0.1}},
        "latency_p50_ms": 57.0,
    }


def test_the_summary_keeps_one_row_per_matching_label() -> None:
    history = [
        history_row("chunk-fixed-1000-200", 0.5, strategy="fixed"),
        history_row("dense-baseline", 0.713),
        history_row("chunk-sentence-1000-200", 0.6, strategy="sentence"),
    ]
    rows = summarise(history, "chunk-*")
    assert [row[0] for row in rows] == ["chunk-fixed-1000-200", "chunk-sentence-1000-200"]
    assert [row[2] for row in rows] == ["fixed", "sentence"]
    assert rows[0][5] == "0.500"


def test_the_summary_keeps_only_the_last_run_of_a_repeated_label() -> None:
    history = [history_row("chunk-fixed-1000-200", 0.5), history_row("chunk-fixed-1000-200", 0.9)]
    rows = summarise(history, "chunk-*")
    assert len(rows) == 1
    assert rows[0][5] == "0.900"


def test_a_glob_matching_nothing_summarises_nothing() -> None:
    assert summarise([history_row("dense-baseline", 0.713)], "chunk-*") == []


def test_single_facet_questions_are_bucketed_by_doc_type() -> None:
    questions = [
        EvalQuestion(
            question_id="q1",
            question="how do dependencies work",
            category="conceptual",
            relevant_document_ids=["fastapi:tutorial/dependencies"],
        ),
        EvalQuestion(
            question_id="q2",
            question="how do i deploy",
            category="conceptual",
            relevant_document_ids=["fastapi:deployment/docker"],
        ),
    ]
    result = run_benchmark(questions, perfect, label="t", ks=(1,))
    assert sorted(result.per_doc_type) == ["deployment", "tutorial"]
    assert result.per_doc_type["tutorial"]["questions"] == 1.0


def test_a_multi_facet_question_is_in_no_bucket() -> None:
    questions = [
        EvalQuestion(
            question_id="q1",
            question="how do dependencies work in production",
            category="multi_doc",
            relevant_document_ids=["fastapi:tutorial/dependencies", "fastapi:deployment/docker"],
        )
    ]
    result = run_benchmark(questions, perfect, label="t", ks=(1,))
    assert result.per_doc_type == {}
    assert result.answerable == 1


def test_an_unanswerable_question_is_in_no_bucket() -> None:
    questions = [
        EvalQuestion(
            question_id="q1",
            question="what is the capital of france",
            category="unanswerable",
            relevant_document_ids=[],
        )
    ]
    result = run_benchmark(questions, perfect, label="t", ks=(1,))
    assert result.per_doc_type == {}
