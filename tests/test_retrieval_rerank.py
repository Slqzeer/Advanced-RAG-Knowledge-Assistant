"""The registry and the shared ordering, against fake scorers: no model, no key."""

from collections.abc import Iterator, Sequence

import pytest

from app.core.config import Settings
from app.models.chunks import Chunk, ScoredChunk
from app.retrieval.rerank import RERANKERS, Scorer, rerank, warm_up

SETTINGS = Settings(openai_api_key=None)


def scored(index: int, score: float) -> ScoredChunk:
    chunk = Chunk(
        document_id=f"fastapi:tutorial/page-{index}",
        source="fastapi",
        title=f"Page {index}",
        doc_type="tutorial",
        chunk_index=0,
        text=f"chunk body number {index}",
        char_start=0,
        char_end=20,
    )
    return ScoredChunk(chunk=chunk, score=score, rank=index + 1)


def pool(size: int = 4) -> list[ScoredChunk]:
    """A retriever's output: descending scores, ranks from 1."""
    return [scored(index, 0.9 - 0.1 * index) for index in range(size)]


def register(name: str, scorer: Scorer) -> None:
    """Registered per test and removed by the fixture below."""
    RERANKERS[name] = scorer


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    original = dict(RERANKERS)
    yield
    RERANKERS.clear()
    RERANKERS.update(original)


def reverse_scorer(
    query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
) -> list[tuple[int, float]]:
    """Scores the last candidate highest — a no-op reranker cannot fake this."""
    return [(index, float(index)) for index in range(len(candidates))]


# --- the reordering actually happens ---------------------------------------


def test_reordering_actually_happens() -> None:
    register("reverse", reverse_scorer)
    results = rerank("q", pool(4), model="reverse", top_k=4)
    assert [r.chunk.document_id for r in results] == [
        "fastapi:tutorial/page-3",
        "fastapi:tutorial/page-2",
        "fastapi:tutorial/page-1",
        "fastapi:tutorial/page-0",
    ]


def test_rank_is_reassigned_from_one() -> None:
    register("reverse", reverse_scorer)
    assert [r.rank for r in rerank("q", pool(4), model="reverse", top_k=4)] == [1, 2, 3, 4]


def test_the_retriever_score_survives_and_the_rerank_score_is_recorded() -> None:
    """Both numbers stay answerable: 'the retriever never had it' and 'the
    reranker buried it' are different failures."""
    register("reverse", reverse_scorer)
    [best] = rerank("q", pool(4), model="reverse", top_k=1)
    assert best.chunk.document_id == "fastapi:tutorial/page-3"
    assert best.score == pytest.approx(0.6)  # what dense retrieval thought
    assert best.rerank_score == pytest.approx(3.0)


def test_top_k_trims_after_reordering_not_before() -> None:
    register("reverse", reverse_scorer)
    results = rerank("q", pool(10), model="reverse", top_k=3)
    assert [r.chunk.document_id for r in results] == [
        "fastapi:tutorial/page-9",
        "fastapi:tutorial/page-8",
        "fastapi:tutorial/page-7",
    ]


# --- errors and edges ------------------------------------------------------


def test_an_unknown_reranker_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="unknown reranker"):
        rerank("q", pool(), model="nope", top_k=3)


def test_top_k_below_one_is_rejected() -> None:
    register("reverse", reverse_scorer)
    with pytest.raises(ValueError, match="top_k"):
        rerank("q", pool(), model="reverse", top_k=0)


def test_an_empty_pool_returns_empty_without_calling_a_backend() -> None:
    """Both backends charge for being asked to rank nothing — in latency or in
    money."""
    calls: list[str] = []

    def recording(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        calls.append(query)
        return []

    register("recording", recording)
    assert rerank("q", [], model="recording", top_k=3) == []
    assert calls == []


def test_ties_break_on_chunk_id_so_two_runs_agree() -> None:
    def flat(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        return [(index, 0.5) for index in range(len(candidates))]

    register("flat", flat)
    ids = [r.chunk.chunk_id for r in rerank("q", pool(4), model="flat", top_k=4)]
    assert ids == sorted(ids)


def test_a_backend_may_return_fewer_pairs_than_it_was_given() -> None:
    """Cohere's top_n returns only the best n; the result is those n, ranked."""

    def partial(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        return [(2, 0.9), (0, 0.4)]

    register("partial", partial)
    results = rerank("q", pool(4), model="partial", top_k=4)
    assert [r.chunk.document_id for r in results] == [
        "fastapi:tutorial/page-2",
        "fastapi:tutorial/page-0",
    ]


def test_the_scorer_is_told_how_many_results_are_wanted() -> None:
    """Cohere bills per search and returns top_n; passing it through is the
    difference between ranking 30 documents and paying to return 30."""
    seen: list[int] = []

    def recording(
        query: str, candidates: Sequence[ScoredChunk], top_k: int, settings: Settings
    ) -> list[tuple[int, float]]:
        seen.append(top_k)
        return [(0, 1.0)]

    register("recording", recording)
    rerank("q", pool(4), model="recording", top_k=2)
    assert seen == [2]


def test_warm_up_is_a_no_op_for_an_unknown_backend() -> None:
    """It runs before a timed loop; a crash there would fail a benchmark that
    would otherwise have worked."""
    warm_up("nope", SETTINGS)


# --- the FlashRank backend -------------------------------------------------


def test_flashrank_is_registered() -> None:
    assert "flashrank" in RERANKERS


def test_flashrank_maps_scores_back_by_list_position() -> None:
    """The passage id is the candidate's index, not its chunk_id: a string
    round-trip through a third-party library is one more place to lose a
    mapping, and the index is already unique and already an int."""
    from app.retrieval import rerank as module

    captured: dict[str, object] = {}

    class FakeRanker:
        def rerank(self, request: object) -> list[dict[str, object]]:
            captured["passages"] = request.passages  # type: ignore[attr-defined]
            captured["query"] = request.query  # type: ignore[attr-defined]
            # Deliberately out of order and incomplete, like a real one.
            return [{"id": 2, "text": "x", "score": 0.91}, {"id": 0, "text": "y", "score": 0.12}]

    module._ranker.cache_clear()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(module, "_ranker", lambda name: FakeRanker())
    try:
        pairs = module._flashrank("what is a dependency", pool(4), 2, SETTINGS)
    finally:
        monkey.undo()

    assert pairs == [(2, pytest.approx(0.91)), (0, pytest.approx(0.12))]
    assert captured["query"] == "what is a dependency"
    assert [p["id"] for p in captured["passages"]] == [0, 1, 2, 3]  # type: ignore[index]
    assert captured["passages"][0]["text"] == "chunk body number 0"  # type: ignore[index]


requires_model = pytest.mark.requires_model


@requires_model
def test_the_real_model_prefers_the_relevant_chunk() -> None:
    """The only test that loads ONNX weights, and the only one that proves the
    cross-encoder reads the pair rather than the passage: the relevant chunk is
    given the *worse* retriever score, so a reranker that passes its input
    through cannot pass this."""
    relevant = scored(0, 0.1)
    irrelevant = scored(1, 0.9)
    candidates = [
        relevant.model_copy(
            update={
                "chunk": relevant.chunk.model_copy(
                    update={"text": "Depends() injects a dependency into a path operation."}
                )
            }
        ),
        irrelevant.model_copy(
            update={
                "chunk": irrelevant.chunk.model_copy(
                    update={"text": "Run the server with uvicorn on port 8000."}
                )
            }
        ),
    ]
    [best] = rerank("how do dependencies work", candidates, model="flashrank", top_k=1)
    assert "Depends()" in best.chunk.text


# --- the Cohere backend ----------------------------------------------------


class FakeCohereResult:
    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class FakeCohereResponse:
    def __init__(self, results: list[FakeCohereResult]) -> None:
        self.results = results


class FakeCohereClient:
    def __init__(self, results: list[FakeCohereResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def rerank(self, **kwargs: object) -> FakeCohereResponse:
        self.calls.append(kwargs)
        return FakeCohereResponse(self.results)


def test_cohere_is_registered() -> None:
    assert "cohere" in RERANKERS


def test_cohere_maps_every_score_back_to_the_chunk_it_scored() -> None:
    """The API answers with positions into the list it was sent. An off-by-one
    here pairs every score with the wrong chunk and still returns a well-formed
    ranking no aggregate metric would catch."""
    from app.retrieval.rerank import _cohere

    client = FakeCohereClient([FakeCohereResult(2, 0.93), FakeCohereResult(0, 0.11)])
    pairs = _cohere("what is a dependency", pool(4), 2, SETTINGS, client=client)

    assert pairs == [(2, pytest.approx(0.93)), (0, pytest.approx(0.11))]


def test_cohere_sends_the_chunk_texts_in_candidate_order() -> None:
    from app.retrieval.rerank import _cohere

    client = FakeCohereClient([FakeCohereResult(0, 1.0)])
    _cohere("q", pool(3), 2, SETTINGS, client=client)

    assert client.calls[0]["documents"] == [
        "chunk body number 0",
        "chunk body number 1",
        "chunk body number 2",
    ]
    assert client.calls[0]["query"] == "q"


def test_cohere_asks_for_only_as_many_as_are_wanted() -> None:
    """It bills per search; returning 30 when 5 are used is paid-for waste."""
    from app.retrieval.rerank import _cohere

    client = FakeCohereClient([FakeCohereResult(0, 1.0)])
    _cohere("q", pool(30), 5, SETTINGS, client=client)
    assert client.calls[0]["top_n"] == 5


def test_cohere_without_a_key_fails_before_the_network() -> None:
    from app.retrieval.rerank import _cohere

    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        _cohere("q", pool(3), 2, Settings(cohere_api_key=None, openai_api_key=None))
