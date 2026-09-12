import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from qdrant_client import QdrantClient

from app.core.config import Settings, get_settings
from app.models.chunks import Chunk, ScoredChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.search import matches_filters, parse_filters, rrf, search
from app.retrieval.store import chunk_from_payload, ensure_collection, get_client, upsert_chunks

SETTINGS = Settings(qdrant_collection="chunks", openai_api_key=None, retrieval_mode="dense")


def make_payload(index: int, **overrides: Any) -> dict[str, Any]:
    payload = Chunk(
        document_id="fastapi:tutorial/dependencies",
        source="fastapi",
        title="Dependencies",
        url="https://fastapi.tiangolo.com/tutorial/dependencies/",
        doc_type="tutorial",
        section="First steps",
        chunk_index=index,
        text=f"chunk body number {index}",
        char_start=index * 100,
        char_end=index * 100 + 20,
    ).to_payload()
    payload.update(overrides)
    return payload


class FakeHit:
    def __init__(self, score: float, payload: dict[str, Any]) -> None:
        self.score = score
        self.payload = payload


class FakeResponse:
    def __init__(self, points: list[FakeHit]) -> None:
        self.points = points


class FakeClient:
    """Records what search asked for, answers with as many hits as `limit`."""

    def __init__(self, payloads: list[dict[str, Any]] | None = None) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def query_points(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        payloads = self.payloads or [make_payload(i) for i in range(kwargs["limit"])]
        scores = [0.9 - 0.1 * i for i in range(len(payloads))]
        return FakeResponse([FakeHit(s, p) for s, p in zip(scores, payloads, strict=True)])


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


def run(query: str = "how do dependencies work", **kwargs: Any) -> list[ScoredChunk]:
    kwargs.setdefault("client", FakeClient())
    kwargs.setdefault("embedder", FakeEmbedder())
    kwargs.setdefault("settings", SETTINGS)
    return search(query, **kwargs)


# --- no server, no API key -------------------------------------------------


def test_results_are_ranked_from_one() -> None:
    assert [scored.rank for scored in run(top_k=3)] == [1, 2, 3]


def test_scores_are_passed_through_unchanged_and_descending() -> None:
    scores = [scored.score for scored in run(top_k=3)]
    assert scores == [0.9, 0.8, pytest.approx(0.7)]
    assert scores == sorted(scores, reverse=True)


def test_the_query_is_embedded_once_per_call() -> None:
    embedder = FakeEmbedder()
    run(embedder=embedder, top_k=5)
    assert embedder.calls == ["how do dependencies work"]


def test_a_scalar_filter_becomes_a_match_value() -> None:
    client = FakeClient()
    run(client=client, filters={"source": "fastapi"})
    [condition] = client.calls[0]["query_filter"].must
    assert condition.key == "source"
    assert condition.match.value == "fastapi"


def test_a_sequence_filter_becomes_a_match_any() -> None:
    client = FakeClient()
    run(client=client, filters={"doc_type": ["tutorial", "advanced"]})
    [condition] = client.calls[0]["query_filter"].must
    assert condition.key == "doc_type"
    assert condition.match.any == ["tutorial", "advanced"]


def test_two_filters_are_anded_in_a_stable_order() -> None:
    client = FakeClient()
    run(client=client, filters={"doc_type": "tutorial", "source": "fastapi"})
    assert [c.key for c in client.calls[0]["query_filter"].must] == ["doc_type", "source"]


def test_no_filters_means_no_filter() -> None:
    client = FakeClient()
    run(client=client)
    assert client.calls[0]["query_filter"] is None


def test_an_empty_filters_mapping_means_no_filter() -> None:
    client = FakeClient()
    run(client=client, filters={})
    assert client.calls[0]["query_filter"] is None


def test_an_unindexed_filter_key_is_rejected_by_name() -> None:
    """A typo'd key would otherwise scan the whole collection and match nothing."""
    with pytest.raises(ValueError, match="doctype"):
        run(filters={"doctype": "tutorial"})


def test_an_empty_value_list_is_rejected() -> None:
    """MatchAny([]) matches nothing, which reads as 'retrieval is broken'."""
    with pytest.raises(ValueError, match="doc_type"):
        run(filters={"doc_type": []})


def test_parse_filters_reads_key_value_pairs() -> None:
    assert parse_filters(["source=fastapi"]) == {"source": ["fastapi"]}


def test_parse_filters_splits_comma_separated_values() -> None:
    assert parse_filters(["doc_type=tutorial,advanced"]) == {"doc_type": ["tutorial", "advanced"]}


def test_parse_filters_merges_a_repeated_key() -> None:
    assert parse_filters(["doc_type=tutorial", "doc_type=advanced"]) == {
        "doc_type": ["tutorial", "advanced"]
    }


def test_parse_filters_rejects_a_pair_with_no_equals() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_filters(["doc_type"])


def test_parse_filters_rejects_an_empty_value() -> None:
    with pytest.raises(ValueError, match="doc_type"):
        parse_filters(["doc_type="])


def test_a_missing_optional_payload_key_is_none_not_an_error() -> None:
    """Qdrant omits null payload values, so this happens on a real collection."""
    payload = make_payload(0)
    del payload["url"], payload["section"]
    [scored] = run(client=FakeClient([payload]), top_k=1)
    assert scored.chunk.url is None
    assert scored.chunk.section is None


def test_a_missing_required_payload_key_names_the_key() -> None:
    payload = make_payload(0)
    del payload["text"]
    with pytest.raises(ValueError, match="text"):
        run(client=FakeClient([payload]), top_k=1)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_is_rejected_before_embedding(query: str) -> None:
    """Embedding the empty string returns a valid vector that retrieves
    plausible-looking garbage; failing fast is the only honest behaviour."""
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match="empty"):
        run(query, embedder=embedder)
    assert embedder.calls == []


def test_top_k_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        run(top_k=0)


def test_the_collection_and_payload_flags_come_from_the_settings() -> None:
    client = FakeClient()
    run(client=client)
    assert client.calls[0]["collection_name"] == "chunks"
    assert client.calls[0]["with_payload"] is True


def test_an_explicit_collection_overrides_the_settings() -> None:
    """Step 12 indexes one collection per chunking strategy and benchmarks each."""
    client = FakeClient()
    run(client=client, collection="chunks_semantic")
    assert client.calls[0]["collection_name"] == "chunks_semantic"


# --- against a real Qdrant -------------------------------------------------

requires_qdrant = pytest.mark.requires_qdrant

CORPUS = [
    "Dependency injection in FastAPI uses Depends() to declare what a path "
    "operation needs, and the framework resolves it for each request.",
    "Set a memory limit on a Docker container with the --memory flag when you "
    "run it, so one container cannot starve the host.",
    "Markdown headings start with a hash character, and nested lists are "
    "indented by two spaces under their parent item.",
]


# Distinct facets, one per CORPUS entry: a doc_type filter test whose fixtures
# all share a facet passes just as well with the filter switched off.
DOC_TYPES = ("tutorial", "deployment", "about")


def real_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        document_id=f"fastapi:doc-{index}",
        source="fastapi",
        title=f"Document {index}",
        url=None,
        doc_type=DOC_TYPES[index],
        section=None,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
    )


@pytest.fixture(scope="module")
def live() -> Iterator[tuple[QdrantClient, Settings]]:
    settings = get_settings()
    if settings.openai_api_key is None:
        pytest.skip("no OPENAI_API_KEY; search needs a real query embedding")
    client = get_client(settings)
    try:
        client.get_collections()
    except Exception as error:  # noqa: BLE001 - any transport failure means "not up"
        pytest.skip(f"Qdrant is not running ({error}); docker compose up -d qdrant")
    yield client, settings
    client.close()


@pytest.fixture
def indexed(live: tuple[QdrantClient, Settings]) -> Iterator[Settings]:
    from app.ingestion.embed import EmbeddingCache, embed_texts

    client, settings = live
    name = f"test_search_{uuid.uuid4().hex[:8]}"
    chunks = [real_chunk(i, text) for i, text in enumerate(CORPUS)]
    cache = EmbeddingCache(settings.embedding_cache_path)
    vectors = embed_texts([c.text for c in chunks], model=settings.embedding_model, cache=cache)
    ensure_collection(client, name, len(vectors[0]))
    upsert_chunks(client, name, chunks, vectors)
    yield settings.model_copy(update={"qdrant_collection": name})
    client.delete_collection(name)


@requires_qdrant
def test_the_right_document_ranks_first(indexed: Settings) -> None:
    """If this fails, something upstream is broken and no later measurement
    means anything."""
    results = search("how does dependency injection work", top_k=3, settings=indexed)
    assert results[0].chunk.document_id == "fastapi:doc-0"
    assert results[0].rank == 1
    assert results[0].score > results[-1].score


@requires_qdrant
def test_a_filter_that_matches_nothing_returns_nothing(indexed: Settings) -> None:
    assert search("dependency injection", filters={"source": "django"}, settings=indexed) == []


@requires_qdrant
def test_a_doc_type_filter_restricts_results_to_that_facet(indexed: Settings) -> None:
    results = search(
        "dependency injection", top_k=5, filters={"doc_type": "tutorial"}, settings=indexed
    )
    assert results
    assert {scored.chunk.doc_type for scored in results} == {"tutorial"}


def test_the_payload_mapper_is_the_one_from_the_store() -> None:
    """One mapper, or search results quietly lose a field the indexer writes."""
    chunk = real_chunk(0, "text")
    assert chunk_from_payload(chunk.to_payload()) == chunk


# --- retrieval modes (step 14) --------------------------------------------


def bm25_index(*texts: str) -> BM25Index:
    return BM25Index(
        [Chunk.model_validate(make_payload(i, text=text)) for i, text in enumerate(texts)]
    )


def test_the_default_mode_is_dense_and_queries_qdrant() -> None:
    client = FakeClient()
    run(client=client, top_k=3)
    assert len(client.calls) == 1


def test_lexical_mode_does_not_touch_qdrant_or_the_embedder() -> None:
    """The point of a lexical branch: no vector, no API call, no server round trip."""
    client, embedder = FakeClient(), FakeEmbedder()
    run(
        "HTTPException 422",
        mode="lexical",
        client=client,
        embedder=embedder,
        index=bm25_index("a 422 response", "unrelated prose"),
        top_k=1,
    )
    assert client.calls == []
    assert embedder.calls == []


def test_lexical_mode_retrieves_an_exact_token_dense_search_misses() -> None:
    [top] = run(
        "HTTPException 422",
        mode="lexical",
        index=bm25_index("returns a 422 response", "generic error handling prose"),
        top_k=1,
    )
    assert "422" in top.chunk.text


def test_lexical_mode_ranks_from_one() -> None:
    results = run(
        "error",
        mode="lexical",
        index=bm25_index("error one", "error two", "error three"),
        top_k=3,
    )
    assert [scored.rank for scored in results] == [1, 2, 3]


def test_an_unknown_mode_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="sparse"):
        run(mode="sparse")


def test_the_mode_falls_back_to_the_settings() -> None:
    """A benchmark row that says `lexical` must not have run dense retrieval."""
    client, embedder = FakeClient(), FakeEmbedder()
    run(
        "error",
        settings=Settings(qdrant_collection="chunks", retrieval_mode="lexical"),
        client=client,
        embedder=embedder,
        index=bm25_index("an error occurred"),
        top_k=1,
    )
    assert client.calls == []


# --- filters on the lexical branch (step 15) ------------------------------


def filtered_index() -> BM25Index:
    return BM25Index(
        [
            Chunk.model_validate(
                make_payload(0, text="error in a tutorial", doc_type="tutorial", source="fastapi")
            ),
            Chunk.model_validate(
                make_payload(1, text="error in a reference", doc_type="reference", source="fastapi")
            ),
            Chunk.model_validate(
                make_payload(2, text="error in starlette", doc_type="tutorial", source="starlette")
            ),
        ]
    )


def test_matches_filters_accepts_a_scalar_match() -> None:
    chunk = Chunk.model_validate(make_payload(0, doc_type="tutorial"))
    assert matches_filters(chunk, {"doc_type": "tutorial"})
    assert not matches_filters(chunk, {"doc_type": "reference"})


def test_matches_filters_accepts_any_of_a_sequence() -> None:
    chunk = Chunk.model_validate(make_payload(0, doc_type="tutorial"))
    assert matches_filters(chunk, {"doc_type": ["tutorial", "advanced"]})
    assert not matches_filters(chunk, {"doc_type": ["reference", "advanced"]})


def test_matches_filters_ands_several_keys() -> None:
    chunk = Chunk.model_validate(make_payload(0, doc_type="tutorial", source="fastapi"))
    assert matches_filters(chunk, {"doc_type": "tutorial", "source": "fastapi"})
    assert not matches_filters(chunk, {"doc_type": "tutorial", "source": "starlette"})


def test_matches_filters_with_no_filters_matches_everything() -> None:
    chunk = Chunk.model_validate(make_payload(0))
    assert matches_filters(chunk, None)
    assert matches_filters(chunk, {})


def test_matches_filters_rejects_an_unindexed_key_like_build_filter_does() -> None:
    """The two translations must accept and reject exactly the same mappings."""
    chunk = Chunk.model_validate(make_payload(0))
    with pytest.raises(ValueError, match="doctype"):
        matches_filters(chunk, {"doctype": "tutorial"})


def test_matches_filters_rejects_an_empty_value_list_like_build_filter_does() -> None:
    chunk = Chunk.model_validate(make_payload(0))
    with pytest.raises(ValueError, match="doc_type"):
        matches_filters(chunk, {"doc_type": []})


def test_lexical_mode_honours_a_scalar_filter() -> None:
    results = run(
        "error",
        mode="lexical",
        index=filtered_index(),
        filters={"doc_type": "reference"},
        top_k=5,
    )
    assert [scored.chunk.doc_type for scored in results] == ["reference"]


def test_lexical_mode_honours_a_sequence_filter() -> None:
    results = run(
        "error", mode="lexical", index=filtered_index(), filters={"source": ["starlette"]}, top_k=5
    )
    assert [scored.chunk.source for scored in results] == ["starlette"]


def test_lexical_mode_ands_two_filters() -> None:
    results = run(
        "error",
        mode="lexical",
        index=filtered_index(),
        filters={"doc_type": "tutorial", "source": "fastapi"},
        top_k=5,
    )
    assert [scored.chunk.chunk_index for scored in results] == [0]


def test_lexical_mode_rejects_an_unindexed_filter_key_before_searching() -> None:
    """Eagerly, not once per candidate chunk: a typo must fail the call."""
    with pytest.raises(ValueError, match="doctype"):
        run("error", mode="lexical", index=filtered_index(), filters={"doctype": "tutorial"})


# --- RRF and hybrid mode (step 16) ----------------------------------------


def scored_list(*indices: int) -> list[ScoredChunk]:
    """A ranked list of chunks, identified by chunk_index, ranked from 1."""
    return [
        ScoredChunk(
            chunk=Chunk.model_validate(make_payload(index)), score=1.0 - 0.1 * rank, rank=rank
        )
        for rank, index in enumerate(indices, start=1)
    ]


def indices_of(results: list[ScoredChunk]) -> list[int]:
    return [scored.chunk.chunk_index for scored in results]


def test_rrf_prefers_a_chunk_ranked_second_by_both_over_one_ranked_first_by_one() -> None:
    """The whole reason to fuse ranks: broad agreement beats a single strong vote.
    Chunk 9 scores 2/(60+2) = 0.0323; chunk 1 scores 1/(60+1) = 0.0164."""
    fused = rrf([scored_list(1, 9), scored_list(2, 9)], k=60, top_k=3)
    assert indices_of(fused)[0] == 9


def test_rrf_deduplicates_a_chunk_appearing_in_both_rankings() -> None:
    fused = rrf([scored_list(1, 2), scored_list(2, 1)], k=60, top_k=10)
    assert sorted(indices_of(fused)) == [1, 2]


def test_rrf_ignores_the_input_scores_entirely() -> None:
    """A cosine and a BM25 score share no scale; using them would invent a
    comparison the numbers do not support."""
    high = scored_list(1)
    low = [ScoredChunk(chunk=high[0].chunk, score=0.0001, rank=1)]
    assert rrf([low], k=60, top_k=1)[0].score == rrf([high], k=60, top_k=1)[0].score


def test_rrf_ranks_from_one_and_scores_descending() -> None:
    fused = rrf([scored_list(1, 2, 3)], k=60, top_k=3)
    assert [scored.rank for scored in fused] == [1, 2, 3]
    assert [s.score for s in fused] == sorted([s.score for s in fused], reverse=True)


def test_rrf_score_is_the_fused_score_not_a_cosine() -> None:
    """Documented consequence: abstention_rate is not comparable across modes."""
    assert rrf([scored_list(1)], k=60, top_k=1)[0].score == pytest.approx(1 / 61)


def test_rrf_caps_at_top_k() -> None:
    assert len(rrf([scored_list(1, 2, 3, 4, 5)], k=60, top_k=2)) == 2


def test_rrf_with_an_empty_ranking_still_fuses_the_other() -> None:
    """A lexical query whose terms are all unknown returns nothing; hybrid mode
    must degrade to dense rather than fail."""
    assert indices_of(rrf([scored_list(1, 2), []], k=60, top_k=2)) == [1, 2]


def test_rrf_rejects_a_k_below_one() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        rrf([scored_list(1)], k=0, top_k=1)


def test_hybrid_mode_queries_both_branches_at_the_candidate_depth() -> None:
    client, embedder = FakeClient(), FakeEmbedder()
    run(
        "error",
        mode="hybrid",
        client=client,
        embedder=embedder,
        index=bm25_index("an error occurred", "another error"),
        top_k=2,
        candidates=50,
    )
    assert client.calls[0]["limit"] == 50
    assert embedder.calls == ["error"]


def test_hybrid_mode_returns_top_k_not_candidates() -> None:
    results = run(
        "error",
        mode="hybrid",
        client=FakeClient(),
        index=bm25_index("an error occurred", "another error"),
        top_k=3,
        candidates=10,
    )
    assert len(results) == 3
    assert [scored.rank for scored in results] == [1, 2, 3]


def test_candidates_below_top_k_is_rejected() -> None:
    """Fusing two top-3 lists cannot produce 10 results; a silent short list
    would read downstream as a recall drop."""
    with pytest.raises(ValueError, match="candidates"):
        run("error", mode="hybrid", top_k=10, candidates=3, index=bm25_index("error"))


def test_hybrid_mode_honours_filters_on_both_branches() -> None:
    client = FakeClient()
    run(
        "error",
        mode="hybrid",
        client=client,
        index=filtered_index(),
        filters={"doc_type": "reference"},
        top_k=5,
        candidates=5,
    )
    [condition] = client.calls[0]["query_filter"].must
    assert condition.key == "doc_type"


def test_candidates_and_rrf_k_fall_back_to_the_settings() -> None:
    client = FakeClient()
    run(
        "error",
        mode="hybrid",
        settings=Settings(qdrant_collection="chunks", retrieval_candidates=30, rrf_k=20),
        client=client,
        index=bm25_index("an error occurred"),
        top_k=5,
    )
    assert client.calls[0]["limit"] == 30
