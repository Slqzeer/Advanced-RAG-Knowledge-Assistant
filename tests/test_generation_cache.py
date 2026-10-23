import os
import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from redis import Redis
from redis.exceptions import RedisError

from app.core.config import Settings
from app.generation.cache import (
    ANSWER_PREFIX,
    QVEC,
    _pairs,
    cache_key,
    cached_answer,
    clear_cache,
    connect,
    fingerprint,
    similarity_floor,
)
from app.models.answers import Answer, RetrievalStats
from app.observability.trace import QueryTrace

# The spec's denylist, spelled out rather than imported: a setting joining it is
# a decision a reviewer should see in this diff.
UNKEYED = {
    "qdrant_url",
    "qdrant_api_key",
    "redis_url",
    "cache_ttl_s",
    "cache_semantic_threshold",
    "corpus_dir",
    "embedding_cache_path",
    "openai_api_key",
    "cohere_api_key",
    "generation_api_key",
    "embedding_api_key",
    "judge_model",
    "judge_embedding_model",
    "trace_path",
    "trace_content",
    "generation_usd_per_mtok_input",
    "generation_usd_per_mtok_output",
}


def _other(name: str, value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, Path):
        return value / "other"
    if value is None:
        return (
            0.5
            if name
            in {
                "cache_semantic_threshold",
                "generation_usd_per_mtok_input",
                "generation_usd_per_mtok_output",
            }
            else "other"
        )
    return f"{value}-other"


@pytest.mark.parametrize("name", sorted(Settings.model_fields))
def test_every_setting_outside_the_denylist_changes_the_key(name: str) -> None:
    base = Settings(_env_file=None)
    other = base.model_copy(update={name: _other(name, getattr(base, name))})
    changed = cache_key("q", None, base, {}) != cache_key("q", None, other, {})
    assert changed is (name not in UNKEYED)


def test_whitespace_is_collapsed_and_case_is_kept() -> None:
    settings = Settings(_env_file=None)
    assert cache_key("  How do I   use Depends?\n", None, settings, {}) == cache_key(
        "How do I use Depends?", None, settings, {}
    )
    assert cache_key("depends", None, settings, {}) != cache_key("Depends", None, settings, {})


def test_history_is_truncated_as_contextualize_truncates_it() -> None:
    settings = Settings(_env_file=None, history_turns=2)
    turns = [{"role": "user", "content": str(n)} for n in range(5)]
    assert cache_key("q", turns, settings, {}) == cache_key("q", turns[-2:], settings, {})
    assert cache_key("q", turns, settings, {}) != cache_key("q", turns[-1:], settings, {})
    assert cache_key("q", [], settings, {}) == cache_key("q", None, settings, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 7},
        {"mode": "hybrid"},
        {"compress": ""},
        {"filters": {"doc_type": "tutorial"}},
        {"include_contexts": True},
        {"strict": True},
    ],
)
def test_each_per_call_kwarg_changes_the_key_and_the_fingerprint(kwargs: dict[str, Any]) -> None:
    settings = Settings(_env_file=None)
    assert cache_key("q", None, settings, kwargs) != cache_key("q", None, settings, {})
    assert fingerprint(settings, kwargs) != fingerprint(settings, {})


def test_the_fingerprint_ignores_the_question() -> None:
    settings = Settings(_env_file=None)
    assert fingerprint(settings, {}) == fingerprint(settings, {})
    assert cache_key("a", None, settings, {}) != cache_key("b", None, settings, {})


def _answer(text: str, refusal: str | None = None) -> Answer:
    return Answer(
        answer=text,
        retrieval=RetrievalStats(retrieved=0, used=0, dropped=0),
        latency_ms=1500.0,
        model="stub",
        refusal=refusal,  # type: ignore[arg-type]
    )


def _answerer(calls: list[str], refusal: str | None = None) -> Callable[..., Answer]:
    def answerer(question: str, **_: Any) -> Answer:
        calls.append(question)
        return _answer(f"answer to {question}", refusal)

    return answerer


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --- no Redis ---------------------------------------------------------------


def test_no_redis_url_means_no_cache_and_no_connection() -> None:
    calls: list[str] = []
    answer = cached_answer(
        "q", settings=Settings(_env_file=None), answerer=_answerer(calls), top_k=3
    )
    assert calls == ["q"]
    assert answer.cache is None and answer.warnings == []


def test_a_dead_redis_answers_with_a_warning_and_does_not_hang() -> None:
    calls: list[str] = []
    settings = Settings(_env_file=None, redis_url=f"redis://127.0.0.1:{_free_port()}/15")
    started = time.perf_counter()
    answer = cached_answer("q", settings=settings, answerer=_answerer(calls))
    assert calls == ["q"]
    assert answer.answer == "answer to q"
    assert any("cache" in warning for warning in answer.warnings)
    assert time.perf_counter() - started < 1.0


def test_the_cache_wrapper_writes_one_trace_for_a_generated_answer(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    calls: list[str] = []

    cached_answer(
        "q",
        settings=Settings(_env_file=None, trace_path=path),
        answerer=_answerer(calls),
    )

    traces = path.read_text(encoding="utf-8").splitlines()
    assert len(traces) == 1
    trace = QueryTrace.model_validate_json(traces[0])
    assert trace.question == "q" and trace.cache is None


def test_an_exact_follow_up_hit_is_counted_as_a_repeat(tmp_path) -> None:
    from unittest.mock import Mock

    path = tmp_path / "traces.jsonl"
    client = Mock(spec=Redis)
    client.get.return_value = _answer("stored").model_dump_json().encode()
    history = [{"role": "user", "content": "earlier"}]

    answer = cached_answer(
        "and Depends?",
        history=history,
        client=client,
        settings=Settings(_env_file=None, trace_path=path),
        answerer=lambda *_args, **_kwargs: pytest.fail("an exact hit must not generate"),
    )

    trace = QueryTrace.model_validate_json(path.read_text(encoding="utf-8"))
    assert answer.cache == trace.cache == "exact"
    assert trace.follow_up is True
    assert trace.follow_up_repeat is True
    assert trace.chunks == [] and trace.user_prompt is None


# --- against a real Redis 8 -------------------------------------------------

requires_redis = pytest.mark.requires_redis
DB15 = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def client() -> Iterator[Redis]:
    connection = Redis.from_url(DB15, socket_connect_timeout=0.2)
    try:
        connection.ping()
    except RedisError as error:
        pytest.skip(f"Redis is not running ({error}); docker compose up -d redis")
    connection.flushdb()
    yield connection
    connection.flushdb()
    connection.close()


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, redis_url=DB15, cache_semantic_threshold=0.9, **overrides)


def _embedder(vectors: dict[str, list[float]]) -> Callable[[str], list[float]]:
    return lambda text: vectors[text]


@requires_redis
def test_vsim_scores_are_half_of_one_plus_cosine(client: Redis) -> None:
    for name, vector in {"a": [1, 0], "b": [0, 1], "c": [1, 1]}.items():
        client.execute_command("VADD", "v", "VALUES", 2, *vector, name, "NOQUANT")
    reply = client.execute_command("VSIM", "v", "VALUES", 2, 1, 0, "WITHSCORES", "COUNT", 3)
    scores = dict(_pairs(reply))  # parsed exactly as cached_answer parses it
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(0.5)  # cosine 0
    assert scores["c"] == pytest.approx((1 + 2**-0.5) / 2, abs=1e-6)
    assert similarity_floor(0.0) == 0.5 and similarity_floor(1.0) == 1.0


@requires_redis
def test_noquant_keeps_the_vector_exact(client: Redis) -> None:
    client.execute_command("VADD", "v", "VALUES", 2, 0.123456, 0.654321, "a", "NOQUANT")
    stored = [float(x) for x in client.execute_command("VEMB", "v", "a")]
    assert stored == pytest.approx([0.123456, 0.654321], abs=1e-6)


@requires_redis
def test_a_miss_is_stored_and_the_next_call_is_an_exact_hit(client: Redis) -> None:
    calls: list[str] = []
    embed = _embedder({"q": [1, 0, 0, 0]})
    first = cached_answer("q", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    second = cached_answer("q", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    assert calls == ["q"]
    assert first.cache is None and second.cache == "exact"
    assert second.answer == first.answer
    assert second.latency_ms < first.latency_ms


@requires_redis
@pytest.mark.parametrize("refusal", ["no_context", "model_declined"])
def test_refusals_are_never_stored(client: Redis, refusal: str) -> None:
    calls: list[str] = []
    embed = _embedder({"q": [1, 0, 0, 0]})
    for _ in range(2):
        cached_answer("q", settings=_settings(), embedder=embed, answerer=_answerer(calls, refusal))
    assert calls == ["q", "q"]
    assert not list(client.scan_iter(match=ANSWER_PREFIX + "*"))


@requires_redis
def test_a_paraphrase_hits_only_under_the_same_settings(client: Redis) -> None:
    calls: list[str] = []
    embed = _embedder({"original": [1, 0, 0, 0], "paraphrase": [0.99, 0.1, 0, 0]})
    cached_answer("original", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    hit = cached_answer(
        "paraphrase", settings=_settings(), embedder=embed, answerer=_answerer(calls)
    )
    assert hit.cache == "semantic" and hit.answer == "answer to original"
    miss = cached_answer(
        "paraphrase", settings=_settings(top_k=9), embedder=embed, answerer=_answerer(calls)
    )
    assert miss.cache is None
    assert calls == ["original", "paraphrase"]


@requires_redis
def test_below_the_threshold_is_a_miss(client: Redis) -> None:
    calls: list[str] = []
    embed = _embedder({"a": [1, 0, 0, 0], "b": [0.5, 0.866, 0, 0]})  # cosine 0.5
    cached_answer("a", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    answer = cached_answer("b", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    assert answer.cache is None


@requires_redis
def test_a_follow_up_gets_no_semantic_lookup(client: Redis) -> None:
    calls: list[str] = []
    embed = _embedder({"a": [1, 0, 0, 0], "b": [1, 0, 0, 0]})
    cached_answer("a", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    history = [{"role": "user", "content": "earlier"}]
    answer = cached_answer(
        "b", history=history, settings=_settings(), embedder=embed, answerer=_answerer(calls)
    )
    assert answer.cache is None and calls == ["a", "b"]


@requires_redis
def test_an_expired_answer_is_a_miss_and_its_vector_is_removed(client: Redis) -> None:
    calls: list[str] = []
    embed = _embedder({"a": [1, 0, 0, 0], "b": [1, 0, 0, 0]})
    cached_answer("a", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    for key in client.scan_iter(match=ANSWER_PREFIX + "*"):
        client.delete(key)  # what the TTL does, without waiting for it
    answer = cached_answer("b", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    assert answer.cache is None
    # Only the vector for b is left: the one for a was VREM'd by the lookup.
    assert client.execute_command("VCARD", QVEC) == 1


@requires_redis
def test_an_undecodable_entry_is_a_miss_and_is_deleted(client: Redis) -> None:
    calls: list[str] = []
    embed = _embedder({"q": [1, 0, 0, 0]})
    cached_answer("q", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    (key,) = client.scan_iter(match=ANSWER_PREFIX + "*")
    client.set(key, b'{"schema": "from step 24"}')
    again = cached_answer("q", settings=_settings(), embedder=embed, answerer=_answerer(calls))
    assert again.cache is None and calls == ["q", "q"]


@requires_redis
def test_a_wrong_dimension_vector_warns_instead_of_failing(client: Redis) -> None:
    calls: list[str] = []
    four = _embedder({"a": [1, 0, 0, 0]})
    cached_answer("a", settings=_settings(), embedder=four, answerer=_answerer(calls))
    answer = cached_answer(
        "b", settings=_settings(), embedder=_embedder({"b": [1, 0]}), answerer=_answerer(calls)
    )
    assert answer.answer == "answer to b"
    assert any("cache" in warning for warning in answer.warnings)


@requires_redis
def test_clear_cache_removes_the_prefix_and_nothing_else(client: Redis) -> None:
    calls: list[str] = []
    embed = _embedder({"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]})
    for question in ("a", "b"):
        cached_answer(question, settings=_settings(), embedder=embed, answerer=_answerer(calls))
    client.set("someone-else", b"1")
    assert clear_cache(connect(_settings())) == 3  # two answers and the vector set
    assert client.get("someone-else") == b"1"
    assert not client.exists(QVEC)


@pytest.mark.parametrize("threshold", [None, 0.9])
def test_a_cache_embedding_failure_does_not_discard_the_answer(threshold):
    from unittest.mock import Mock

    client = Mock(spec=Redis)
    client.get.return_value = None
    calls = []

    def broken_embedder(question):
        raise RuntimeError("embedding provider unavailable")

    answer = cached_answer(
        "q",
        client=client,
        settings=Settings(_env_file=None, cache_semantic_threshold=threshold),
        embedder=broken_embedder,
        answerer=_answerer(calls),
    )
    assert answer.answer == "answer to q"
    assert calls == ["q"]
    assert len(answer.warnings) == 1
    assert "cache" in answer.warnings[0]
