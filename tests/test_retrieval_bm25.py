from typing import Any

import pytest

from app.models.chunks import Chunk
from app.retrieval.bm25 import BM25Index, build_index, tokenize


def chunk(index: int, text: str, **overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "document_id": "fastapi:tutorial/handling-errors",
        "source": "fastapi",
        "title": "Handling Errors",
        "url": None,
        "doc_type": "tutorial",
        "section": None,
        "chunk_index": index,
        "text": text,
        "char_start": index * 100,
        "char_end": index * 100 + max(len(text), 1),
    }
    fields.update(overrides)
    return Chunk(**fields)


# --- tokenising ------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_non_word_characters() -> None:
    assert tokenize("HTTPException 422, raised!") == ["httpexception", "422", "raised"]


def test_tokenize_keeps_accented_characters() -> None:
    """A French question must not lose its vowels before it reaches the index."""
    assert tokenize("Comment gérer les requêtes") == ["comment", "gérer", "les", "requêtes"]


def test_tokenize_keeps_digit_only_tokens() -> None:
    """`422` is the whole reason this module exists."""
    assert "422" in tokenize("returns a 422 response")


# --- the index -------------------------------------------------------------


def test_an_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="no chunks"):
        BM25Index([])


def test_idf_is_positive_for_a_term_in_every_document() -> None:
    """The textbook log((N - df + 0.5) / (df + 0.5)) goes negative here, which
    would make a common term subtract from a document's score."""
    index = BM25Index([chunk(0, "shared term"), chunk(1, "shared term")])
    assert index.idf["shared"] > 0


def test_a_rarer_term_outweighs_a_common_one() -> None:
    index = BM25Index([chunk(0, "common rare"), chunk(1, "common word"), chunk(2, "common word")])
    assert index.idf["rare"] > index.idf["common"]


def test_a_short_document_outranks_a_long_one_for_the_same_single_match() -> None:
    """Length normalisation, the `b` parameter. Without it `release-notes.md` —
    the largest file in the corpus, containing nearly every token in it — wins
    every lexical query."""
    short = chunk(0, "validation error")
    long = chunk(1, "validation " + " ".join(f"filler{n}" for n in range(200)))
    index = BM25Index([short, long])
    results = index.search("validation", top_k=2)
    assert [scored.chunk.chunk_index for scored in results] == [0, 1]
    assert results[0].score > results[1].score


def test_an_exact_token_is_retrievable() -> None:
    """The capability dense retrieval demonstrably lacks: the step 07 finding."""
    index = BM25Index(
        [
            chunk(0, "You can return a 422 Unprocessable Entity response"),
            chunk(1, "Handling errors with a plain HTTP exception"),
        ]
    )
    [top] = index.search("HTTPException 422", top_k=1)
    assert top.chunk.chunk_index == 0


def test_results_are_ranked_from_one() -> None:
    index = BM25Index([chunk(i, f"error number {i}") for i in range(3)])
    assert [scored.rank for scored in index.search("error", top_k=3)] == [1, 2, 3]


def test_scores_are_descending() -> None:
    index = BM25Index([chunk(0, "error error error"), chunk(1, "error"), chunk(2, "unrelated")])
    scores = [scored.score for scored in index.search("error", top_k=2)]
    assert scores == sorted(scores, reverse=True)


def test_a_query_with_no_matching_term_returns_nothing() -> None:
    """Not "the least bad chunk": an empty list is the honest answer, and the
    fusion in task 4 relies on it rather than on padding."""
    index = BM25Index([chunk(0, "dependency injection")])
    assert index.search("kubernetes", top_k=5) == []


def test_top_k_caps_the_result_count() -> None:
    index = BM25Index([chunk(i, "error") for i in range(10)])
    assert len(index.search("error", top_k=3)) == 3


def test_ties_are_broken_deterministically_by_chunk_id() -> None:
    """Two identical chunks must not swap places between benchmark runs."""
    index = BM25Index([chunk(1, "error"), chunk(0, "error")])
    first = [scored.chunk.chunk_id for scored in index.search("error", top_k=2)]
    second = [scored.chunk.chunk_id for scored in index.search("error", top_k=2)]
    assert first == second == sorted(first)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_is_rejected(query: str) -> None:
    index = BM25Index([chunk(0, "anything")])
    with pytest.raises(ValueError, match="empty"):
        index.search(query, top_k=5)


def test_top_k_zero_is_rejected() -> None:
    index = BM25Index([chunk(0, "anything")])
    with pytest.raises(ValueError, match="top_k"):
        index.search("anything", top_k=0)


def test_the_predicate_excludes_chunks_before_ranking() -> None:
    """Applied while accumulating, not afterwards, so a filtered query still
    returns top_k results instead of top_k minus the discarded ones."""
    index = BM25Index(
        [
            chunk(0, "error in a tutorial", doc_type="tutorial"),
            chunk(1, "error in a reference", doc_type="reference"),
        ]
    )
    results = index.search("error", top_k=5, predicate=lambda c: c.doc_type == "reference")
    assert [scored.chunk.doc_type for scored in results] == ["reference"]


# --- building from Qdrant --------------------------------------------------


class FakeRecord:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class FakeScrollClient:
    """Hands back the payloads in pages, mimicking Qdrant's offset protocol."""

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def scroll(self, **kwargs: Any) -> tuple[list[FakeRecord], int | None]:
        self.calls.append(kwargs)
        page = len(self.calls) - 1
        records = [FakeRecord(payload) for payload in self.pages[page]]
        more = page + 1 < len(self.pages)
        return records, (page + 1 if more else None)


def test_build_index_keeps_every_page_including_the_first() -> None:
    """Qdrant's own documented loop shape discards the first batch."""
    pages = [[chunk(0, "first page").to_payload()], [chunk(1, "second page").to_payload()]]
    index = build_index(FakeScrollClient(pages), "chunks_sentence")  # type: ignore[arg-type]
    assert len(index.chunks) == 2
    assert {c.chunk_index for c in index.chunks} == {0, 1}


def test_build_index_asks_for_payloads_and_not_vectors() -> None:
    """1 536 floats per point, 1 484 points, for data BM25 never reads."""
    client = FakeScrollClient([[chunk(0, "only page").to_payload()]])
    build_index(client, "chunks_sentence")  # type: ignore[arg-type]
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["with_vectors"] is False
    assert client.calls[0]["collection_name"] == "chunks_sentence"
