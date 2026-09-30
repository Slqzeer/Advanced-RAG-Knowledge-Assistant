from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import Settings, get_settings
from app.evaluation.dataset import load_dataset
from app.ingestion.embed import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    EmbeddingCache,
    build_client,
    cache_key,
    embed_query,
    embed_texts,
    embedding_client,
)

# --- the cache -------------------------------------------------------------


def test_fresh_cache_misses(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    assert cache.get(cache_key("m", "unknown")) is None


def test_put_then_get_round_trips_the_vector(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    vector = [0.5, -0.25, 0.125, 0.0]
    key = cache_key("m", "hello")
    cache.put_many([(key, "m", vector)])
    assert cache.get(key) == vector


def test_the_model_is_part_of_the_key(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    cache.put_many([(cache_key("small", "hello"), "small", [1.0])])
    assert cache.get(cache_key("large", "hello")) is None


def test_get_many_returns_only_the_hits(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    hit, miss = cache_key("m", "hit"), cache_key("m", "miss")
    cache.put_many([(hit, "m", [1.0, 2.0])])
    assert cache.get_many([hit, miss]) == {hit: [1.0, 2.0]}


def test_get_many_handles_more_keys_than_sqlite_takes_variables(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    keys = [cache_key("m", str(n)) for n in range(2500)]
    cache.put_many([(k, "m", [float(n)]) for n, k in enumerate(keys)])
    assert cache.get_many(keys) == {k: [float(n)] for n, k in enumerate(keys)}


def test_the_cache_file_and_its_parents_are_created(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "deeper" / "e.sqlite"
    EmbeddingCache(path)
    assert path.exists()


def test_vectors_survive_reopening(tmp_path: Path) -> None:
    path = tmp_path / "e.sqlite"
    key = cache_key("m", "hello")
    EmbeddingCache(path).put_many([(key, "m", [0.5, 0.5])])
    assert EmbeddingCache(path).get(key) == [0.5, 0.5]


def test_putting_the_same_key_twice_does_not_raise(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    key = cache_key("m", "hello")
    cache.put_many([(key, "m", [1.0])])
    cache.put_many([(key, "m", [2.0])])
    assert cache.get(key) == [2.0]


@pytest.mark.parametrize("text", ["hello", "héllo ünïcode", "a" * 10_000, ""])
def test_the_key_is_stable_across_calls(text: str) -> None:
    assert cache_key("m", text) == cache_key("m", text)


# --- the fake provider -----------------------------------------------------


def fake_vector(text: str, dim: int = 4) -> list[float]:
    """Deterministic, and different for different texts."""
    return ([float(sum(text.encode())), float(len(text)), 0.0, 0.5] * dim)[:dim]


class FakeClient:
    """Stands in for `openai.OpenAI`: only `.embeddings.create` is ever touched."""

    def __init__(self, dim: int = 4, truncate_first: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.dim = dim
        self.truncate_first = truncate_first
        self.embeddings = self

    def create(self, *, input: list[str], model: str, **kwargs: object) -> SimpleNamespace:
        self.batches.append(list(input))
        data = []
        for index, text in enumerate(input):
            vector = fake_vector(text, self.dim)
            if self.truncate_first and index == 0:
                vector = vector[:-1]
            data.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(data=data)


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "e.sqlite")


# --- embed_texts -----------------------------------------------------------


def test_three_texts_one_call_in_order(cache: EmbeddingCache) -> None:
    client = FakeClient()
    texts = ["alpha", "beta", "gamma"]
    assert embed_texts(texts, cache=cache, client=client) == [fake_vector(t) for t in texts]
    assert client.batches == [texts]


def test_order_survives_a_partial_cache_hit(cache: EmbeddingCache) -> None:
    texts = ["t0", "t1", "t2", "t3", "t4"]
    embed_texts(["t1", "t3"], cache=cache, client=FakeClient())

    client = FakeClient()
    assert embed_texts(texts, cache=cache, client=client) == [fake_vector(t) for t in texts]
    assert client.batches == [["t0", "t2", "t4"]]


def test_a_second_identical_call_makes_no_request(cache: EmbeddingCache) -> None:
    texts = ["alpha", "beta"]
    embed_texts(texts, cache=cache, client=FakeClient())

    client = FakeClient()
    assert embed_texts(texts, cache=cache, client=client) == [fake_vector(t) for t in texts]
    assert client.batches == []


def test_batches_respect_batch_size(cache: EmbeddingCache) -> None:
    client = FakeClient()
    texts = [f"text number {n}" for n in range(250)]
    vectors = embed_texts(texts, cache=cache, client=client, batch_size=100)
    assert len(vectors) == 250
    assert [len(b) for b in client.batches] == [100, 100, 50]


def test_duplicates_are_embedded_once(cache: EmbeddingCache) -> None:
    client = FakeClient()
    vectors = embed_texts(["same", "other", "same"], cache=cache, client=client)
    assert client.batches == [["same", "other"]]
    assert vectors[0] == vectors[2] == fake_vector("same")


def test_empty_input_makes_no_request(cache: EmbeddingCache) -> None:
    client = FakeClient()
    assert embed_texts([], cache=cache, client=client) == []
    assert client.batches == []


def test_a_short_vector_raises(cache: EmbeddingCache) -> None:
    client = FakeClient(truncate_first=True)
    with pytest.raises(ValueError, match="dimension"):
        embed_texts(["alpha", "beta"], cache=cache, client=client)


def test_a_vector_of_the_wrong_dimension_never_reaches_the_cache(cache: EmbeddingCache) -> None:
    with pytest.raises(ValueError, match="dimension"):
        embed_texts(["alpha", "beta"], cache=cache, client=FakeClient(truncate_first=True))
    assert cache.get_many([cache_key(DEFAULT_MODEL, "alpha")]) == {}


def test_provider_errors_are_not_swallowed(cache: EmbeddingCache) -> None:
    class Boom(FakeClient):
        def create(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("upstream is down")

    with pytest.raises(RuntimeError, match="upstream is down"):
        embed_texts(["alpha"], cache=cache, client=Boom())


def test_the_default_client_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries are the SDK's job — 429 and 5xx, exponential backoff. This asserts
    we asked for enough attempts that a rate limit mid-corpus does not lose work."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    assert build_client().max_retries == MAX_RETRIES


# --- embed_query -----------------------------------------------------------


def test_embed_query_returns_one_vector(cache: EmbeddingCache) -> None:
    assert embed_query("alpha", cache=cache, client=FakeClient()) == fake_vector("alpha")


def test_embed_query_shares_the_cache_with_documents(cache: EmbeddingCache) -> None:
    embed_texts(["alpha"], cache=cache, client=FakeClient())
    client = FakeClient()
    assert embed_query("alpha", cache=cache, client=client) == fake_vector("alpha")
    assert client.batches == []


def test_the_generation_gateway_overrides_do_not_touch_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only `generation.llm.complete` passes these. The embedding cache is keyed
    by model name alone, so a second provider answering to the same model name
    must never reach it."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    routed = build_client(base_url="https://openrouter.ai/api/v1", api_key="gateway-key")
    assert str(routed.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert routed.api_key == "gateway-key"
    assert "openrouter" not in str(build_client().base_url)
    gateway_for_generation = Settings(
        _env_file=None, openai_api_key="k", generation_base_url="https://openrouter.ai/api/v1"
    )
    assert "openrouter" not in str(embedding_client(gateway_for_generation).base_url)


# --- the embedding route -----------------------------------------------------


class RecordingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.models: list[str] = []

    def create(self, *, input: list[str], model: str, **kwargs: object) -> SimpleNamespace:
        self.models.append(model)
        return super().create(input=input, model=model, **kwargs)


def _route(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
    monkeypatch.setattr("app.ingestion.embed.get_settings", lambda: settings)


def test_the_embedding_route_is_its_own_setting() -> None:
    routed = Settings(
        _env_file=None, embedding_base_url="http://localhost:20128/v1", embedding_api_key="gw"
    )
    client = embedding_client(routed)
    assert str(client.base_url).rstrip("/") == "http://localhost:20128/v1"
    assert client.api_key == "gw"
    direct = embedding_client(Settings(_env_file=None, openai_api_key="k"))
    assert "api.openai.com" in str(direct.base_url)


def test_the_gateway_model_id_is_sent_and_the_cache_keeps_the_canonical_name(
    monkeypatch: pytest.MonkeyPatch, cache: EmbeddingCache
) -> None:
    _route(monkeypatch, embedding_provider_model="github/text-embedding-3-small")
    client = RecordingClient()
    embed_texts(["alpha"], model="text-embedding-3-small", cache=cache, client=client)
    assert client.models == ["github/text-embedding-3-small"]
    assert cache.get_many([cache_key("text-embedding-3-small", "alpha")])


def test_any_other_model_is_sent_as_named(
    monkeypatch: pytest.MonkeyPatch, cache: EmbeddingCache
) -> None:
    _route(monkeypatch, embedding_provider_model="github/text-embedding-3-small")
    client = RecordingClient()
    embed_texts(["alpha"], model="text-embedding-3-large", cache=cache, client=client)
    assert client.models == ["text-embedding-3-large"]


@pytest.mark.requires_api
def test_the_configured_route_serves_the_vectors_already_cached(tmp_path: Path) -> None:
    """The guard that replaces "embeddings never pass a base_url": a route may
    share the cache only if it returns the vectors the cache already holds."""
    settings = get_settings()
    real = EmbeddingCache(settings.embedding_cache_path)
    texts = [q.question for q in load_dataset(Path("data/eval/questions.jsonl"))[:3]]
    cached = real.get_many([cache_key(settings.embedding_model, t) for t in texts])
    if len(cached) < len(texts):
        pytest.skip("the eval questions are not in the embedding cache")
    fresh = embed_texts(
        texts, model=settings.embedding_model, cache=EmbeddingCache(tmp_path / "fresh.sqlite")
    )
    for text, vector in zip(texts, fresh, strict=True):
        reference = cached[cache_key(settings.embedding_model, text)]
        dot = sum(a * b for a, b in zip(vector, reference, strict=True))
        norm = (sum(a * a for a in vector) * sum(b * b for b in reference)) ** 0.5
        assert dot / norm >= 0.9999, text
