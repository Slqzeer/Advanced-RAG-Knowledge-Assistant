from pathlib import Path

import pytest

from app.ingestion.embed import EmbeddingCache, cache_key

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
